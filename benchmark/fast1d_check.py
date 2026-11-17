"""fast1d (the 1+1D solver of the design helper) against the 4D reference runs and the papers.

Every benchmark of this folder is re-run with fast1d on exactly the same physics: the cell comes
from the benchmark's own make_cell (media, n2 and its M2 scaling, plates, gas sections, band
mirrors, GDD) with the parameters of its 4D reference run (reference/<name>/results.json), and the
input pulse from its own spectrum() (measured spectra, chirp, TOD). Injection as in the benchmark:
linear eigenmode, or Kerr-matched at the peak / intensity-weighted power (cells with plates). The
LG_0,l vortex benchmarks are skipped (fast1d follows a gaussian beam).

Compared, fast1d vs 4D (relative difference) and vs the paper where it gives a value:
    ftl_fs          Fourier limit of the output
    best_comp_fs    compressed with the best pure GDD (peak power)
    best_gdd_fs2    that GDD
    comp_fs         compressed with the paper's compressor GDD (benchmarks that have one)
    best_peak_power_GW, transmission, B_rad (on-axis peak), F_mirror_max_Jcm2

Usage (from benchmark/):
    python fast1d_check.py                                  # every benchmark, accuracy "default"
    python fast1d_check.py westerberg2025_250uJ omar2024_400uJ --accuracy fine
    python fast1d_check.py --dry-run                        # build the inputs only (no propagation)
Output: a table per benchmark, a summary table, and results/fast1d_check/<accuracy>.json and .md
"""
import argparse
import importlib
import inspect
import json
import sys
import time
from pathlib import Path

import numpy as np

import thunder4d as th
from thunder4d.design import fast1d

HERE = Path(__file__).resolve().parent
FS = 1e-15
QUANTITIES = (  # key, label, unit, scale of the fast1d value
    ("ftl_fs", "Fourier limit", "fs"),
    ("best_comp_fs", "compressed (best GDD)", "fs"),
    ("best_gdd_fs2", "best GDD", "fs2"),
    ("comp_fs", "compressed (paper GDD)", "fs"),
    ("best_peak_power_GW", "peak power (best GDD)", "GW"),
    ("transmission", "transmission", ""),
    ("B_rad", "B-integral (on-axis)", "rad"),
    ("F_mirror_max_Jcm2", "mirror fluence", "J/cm2"),
)
SUMMARY = ("ftl_fs", "best_comp_fs", "best_gdd_fs2", "transmission", "B_rad", "F_mirror_max_Jcm2")


# ============================================================ benchmarks
def load_benchmarks(names=None):
    """Instances of every benchmark class of the bench_*.py scripts (or of the names given)."""
    sys.path.insert(0, str(HERE))
    from framework import Benchmark
    out = []
    for path in sorted(HERE.glob("bench_*.py")):
        mod = importlib.import_module(path.stem)
        for _, cls in inspect.getmembers(mod, inspect.isclass):
            own = issubclass(cls, Benchmark) and cls.__module__ == mod.__name__ and cls.name != "unnamed"
            if own and (names is None or cls.name in names):
                out.append(cls())
    missing = set(names or ()) - {b.name for b in out}
    if missing:
        raise SystemExit(f"unknown benchmark(s) {sorted(missing)}; available: "
                         f"{sorted(b.name for b in load_benchmarks())}")
    return out


def reference_run(name):
    path = HERE / "reference" / name / "results.json"
    return json.loads(path.read_text()) if path.exists() else None


def params_of(bench, ref):
    """Parameters of the 4D reference run (JSON lists back to tuples), else the class parameters."""
    P = dict(bench.params)
    if ref is not None:
        P.update({k: tuple(v) if isinstance(v, list) else v for k, v in ref["params"].items()})
    return P


# ============================================================ benchmark -> fast1d input
def input_field(bench, P, cell):
    """Input pulse of the benchmark as a fast1d field (t [s], A [sqrt(W)]) about the pulse wavelength
    P["lambda0"] (fast1d takes its carrier for the Kerr lens, the B-integral and the beam size).

    thunder4d: A(t) = sum_Om E(Om) exp(-i Om t), spectral phase +GDD/2 Om^2;
    fast1d:    A(t) = sum_Om a(Om) exp(+i Om t), spectral phase -GDD/2 Om^2   ->  a = conj(E),
    then moved from the grid carrier w_g to w_0: A_0(t) = A_g(t) exp(i (w_g - w_0) t)."""
    grid = th.suggest_lens_grid(cell, **bench.grid_args(P), **bench.quick_preset, device="cpu")
    E, source = bench.spectrum(grid, P)
    E = np.asarray(th.to_host(E) if hasattr(th, "to_host") else E).reshape(-1)
    a = np.fft.fftshift(np.fft.ifft(np.conj(E)))
    t = (np.arange(a.size) - a.size // 2) * grid.dt
    a = a * np.exp(1j * 2 * np.pi * th.C0 * (1 / grid.lambda0 - 1 / P["lambda0"]) * t)
    power = np.abs(a) ** 2
    a *= np.sqrt(P["energy"] / (power.sum() * grid.dt))
    return t, a, P["lambda0"], source


def mirror_response(cell, lam_c):
    """One bounce of the cell's mirrors in fast1d's convention (conjugate of MPC.mirror_spectrum)."""
    w_c = 2 * np.pi * th.C0 / lam_c

    def response(omega):
        lam = 2 * np.pi * th.C0 / omega
        W = (omega - w_c) * FS
        ph = cell.gdd / 2 * W ** 2 + cell.tod / 6 * W ** 3
        if cell.mphase is not None:
            ph = ph + cell.mphase(lam)
        R = cell.refl(lam) if callable(cell.refl) else np.full_like(lam, float(cell.refl))
        return np.sqrt(np.clip(R, 0.0, 1.0)) * np.exp(-1j * ph)
    return response


def fast1d_input(bench, P, accuracy):
    """(MPCInput, description) for one benchmark, or (None, reason) if fast1d cannot run it."""
    if int(P.get("vortex") or 0):
        return None, f"LG_0,{int(P['vortex'])} vortex beam: fast1d follows a gaussian beam"
    n = int(P["n_passes"])
    cell = bench.make_cell(P, n)
    t, a, lam_c, source = input_field(bench, P, cell)
    power = np.abs(a) ** 2
    injection = P.get("injection", "linear")
    if "matching_power" in P:          # cells with plates (plate_mpc.py): Kerr-matched, peak or weighted
        Pm = power.max() if P["matching_power"] == "peak" else float(np.sum(power ** 2) / np.sum(power))
        match = Pm * P.get("matching_scale", 1.0)
        how = f"Kerr-matched at the {P['matching_power']} power ({match / 1e9:.3g} GW)"
    elif injection == "nonlinear":
        match = float(power.max())
        how = f"Kerr-matched at the peak power ({match / 1e9:.3g} GW)"
    else:
        match, how = None, "linear eigenmode"
    inp = fast1d.MPCInput(field=(t, a), wavelength=lam_c, roc=cell.R, n_passes=n, cell=cell, match_power=match,
                          mirror_response=mirror_response(cell, lam_c), accuracy=accuracy)
    return inp, f"{source}; {how}; {n} passes; cell: {'plates / gas sections' if cell.segmented else 'gas'}"


# ============================================================ fast1d metrics
def compressed_fwhm(out, gdd_fs2, up=8):
    """FWHM [fs] of the output compressed with a given GDD (fast1d's convention, zero-padded spectrum)."""
    aw = np.fft.fft(out.field_out)
    n = aw.size
    dt = out.t[1] - out.t[0]
    om = 2 * np.pi * np.fft.fftfreq(n, dt)
    aw = aw * np.exp(-0.5j * gdd_fs2 * 1e-30 * om ** 2)
    m, h = n * up, n // 2
    big = np.zeros(m, complex)
    big[:h], big[m - (n - h):] = aw[:h], aw[h:]
    p = np.abs(np.fft.ifft(big)) ** 2
    return fast1d.fwhm(np.arange(m) * dt / up, p) / FS


def metrics_of(out, P):
    m = {"ftl_fs": out.fwhm_ftl / FS, "best_comp_fs": out.fwhm_compressed / FS,
         "best_gdd_fs2": out.gdd_compression * 1e30, "best_peak_power_GW": out.peak_power_compressed / 1e9,
         "transmission": out.transmission, "B_rad": out.b_integral,
         "F_mirror_max_Jcm2": out.fluence_mirror_max * 1e-4}
    if isinstance(P.get("compressor_gdd"), (int, float)):
        m["comp_fs"] = compressed_fwhm(out, P["compressor_gdd"])
    return m


def compare(bench, ref, m):
    """{key: (fast1d, 4D, rel. diff vs 4D, paper, rel. diff vs paper)}"""
    rows = {}
    for key, _, _ in QUANTITIES:
        if key not in m:
            continue
        v = m[key]
        v4 = ref["metrics"].get(key, {}).get("value") if ref else None
        pr = bench.reference.get(key)
        paper = pr.value if pr is not None and pr.kind in ("rel", "abs", "info") else None
        rows[key] = (v, v4, (v - v4) / abs(v4) if v4 else None, paper, (v - paper) / abs(paper) if paper else None)
    return rows


# ============================================================ report
def fmt(v, unit=""):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "-"
    if unit in ("J/cm2",) or abs(v) < 0.01:
        return f"{v:.3g}"
    return f"{v:.4g}" if abs(v) < 100 else f"{v:.0f}"


def pct(x):
    return "-" if x is None else f"{100 * x:+.1f} %"


def block(name, desc, rows, runtime):
    lines = [f"\n{name}  ({desc}; fast1d {runtime:.1f} s)",
             f"  {'quantity':<26}{'fast1d':>10}{'4D':>10}{'vs 4D':>10}{'paper':>10}{'vs paper':>10}"]
    for key, label, unit in QUANTITIES:
        if key in rows:
            v, v4, d4, pp, dp = rows[key]
            lines.append(f"  {label + (f' [{unit}]' if unit else ''):<26}{fmt(v, unit):>10}{fmt(v4, unit):>10}"
                         f"{pct(d4):>10}{fmt(pp, unit):>10}{pct(dp):>10}")
    return "\n".join(lines)


def summary_markdown(results, accuracy):
    head = "| benchmark | " + " | ".join(SUMMARY) + " | fast1d time |"
    lines = [f"# fast1d vs the 4D reference runs (accuracy '{accuracy}')", "",
             "Relative difference fast1d - 4D (4D value in brackets).", "", head,
             "|" + "---|" * (len(SUMMARY) + 2)]
    for name, r in results.items():
        if "skipped" in r:
            lines.append(f"| {name} | " + " | ".join("-" for _ in SUMMARY) + f" | skipped: {r['skipped']} |")
            continue
        cells = []
        for key in SUMMARY:
            _, v4, d4, _, _ = r["rows"].get(key, (None,) * 5)
            cells.append("-" if d4 is None else f"{100 * d4:+.1f} % ({fmt(v4)})")
        lines.append(f"| {name} | " + " | ".join(cells) + f" | {r['runtime_s']:.1f} s |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("names", nargs="*", help="benchmark names (default: all)")
    ap.add_argument("--accuracy", default="default", choices=("fast", "default", "fine"))
    ap.add_argument("--dry-run", action="store_true", help="build the fast1d inputs only")
    args = ap.parse_args(argv)
    results = {}
    for bench in load_benchmarks(args.names or None):
        ref = reference_run(bench.name)
        P = params_of(bench, ref)
        inp, desc = fast1d_input(bench, P, args.accuracy)
        if inp is None:
            print(f"\n{bench.name}: skipped ({desc})")
            results[bench.name] = {"skipped": desc}
            continue
        if args.dry_run:
            g = fast1d._make_grid(inp)
            print(f"{bench.name}: {desc}; fast1d grid {g[0].size} points, carrier {inp.wavelength * 1e9:.1f} nm, "
                  f"4D reference: {'yes' if ref else 'NO'}")
            continue
        t0 = time.perf_counter()
        try:
            out = fast1d.simulate(inp)
        except fast1d.SimulationStopped as err:
            print(f"\n{bench.name}: fast1d stopped: {err}")
            results[bench.name] = {"skipped": f"fast1d stopped: {err}"}
            continue
        runtime = time.perf_counter() - t0
        rows = compare(bench, ref, metrics_of(out, P))
        print(block(bench.name, desc, rows, runtime))
        for w in out.warnings:
            print(f"  warning: {w}")
        results[bench.name] = {"description": desc, "runtime_s": runtime, "rows": rows,
                               "warnings": list(out.warnings), "n_steps": out.n_steps}
    if args.dry_run or not results:
        return results
    md = summary_markdown(results, args.accuracy)
    print("\n" + md)
    dest = HERE / "results" / "fast1d_check"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / f"{args.accuracy}.md").write_text(md)
    (dest / f"{args.accuracy}.json").write_text(json.dumps(
        {"thunder4d": th.__version__, "accuracy": args.accuracy, "date": time.strftime("%Y-%m-%d %H:%M"),
         "quantities": [q[0] for q in QUANTITIES], "row_format": ["fast1d", "4D", "diff_4D", "paper", "diff_paper"],
         "results": results}, indent=1, default=float))
    print(f"saved to {dest}")
    return results


if __name__ == "__main__":
    main()
