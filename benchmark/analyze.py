"""Analyse benchmark results written by framework.py (results/<benchmark>/<run_id>/results.json).

    python analyze.py                                  # table of every run against the papers
    python analyze.py grebing2020_1mJ                  # only one benchmark (several names allowed)
    python analyze.py --latest                         # only the most recent run of each benchmark
    python analyze.py --csv summary.csv                # every metric of every run, one row each
    python analyze.py --compare RUN_A RUN_B            # two runs side by side (convergence checks)
    python analyze.py --sweep daniault2024_4fs pressure_bar rms_width_nm ftl_fs comp_fs
    python analyze.py --plot                           # standard figure of each selected run

RUN_A / RUN_B are run directories or run ids (e.g. run_1d6084c5). Works the same for any benchmark
built on framework.Benchmark, since every run stores the same metric names and curve names.
"""
import argparse
import csv
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from framework import format_value, print_table   # noqa: E402


# ---------------------------------------------------------------- loading
def load_runs(root, names=None, include_quick=False):
    runs = []
    for fn in glob.glob(os.path.join(root, "*", "*", "results.json")):
        with open(fn) as f:
            r = json.load(f)
        r["_dir"] = os.path.dirname(fn)
        if names and r["benchmark"] not in names:
            continue
        if r["run"].get("quick") and not include_quick:
            continue
        runs.append(r)
    runs.sort(key=lambda r: (r["benchmark"], r["run"]["date"]))
    return runs


def latest(runs):
    out = {}
    for r in runs:
        out[r["benchmark"]] = r          # sorted by date: the last one wins
    return list(out.values())


def find_run(root, key):
    if os.path.isfile(os.path.join(key, "results.json")):
        path = os.path.join(key, "results.json")
    else:
        hits = glob.glob(os.path.join(root, "*", key, "results.json"))
        if len(hits) != 1:
            sys.exit(f"run {key!r}: {len(hits)} matches under {root}")
        path = hits[0]
    with open(path) as f:
        r = json.load(f)
    r["_dir"] = os.path.dirname(path)
    return r


# ---------------------------------------------------------------- reports
def score(r):
    st = [m["status"] for m in r["metrics"].values()]
    return st.count("ok"), st.count("off")


def overview(runs):
    print(f"{'benchmark':22s}{'run':24s}{'date':21s}{'grid':16s}{'done':>6s}{'ok/off':>8s}  overrides")
    print("-" * 110)
    for r in runs:
        ok, off = score(r)
        u = r["run"]
        print(f"{r['benchmark']:22s}{u['id']:24s}{u['date']:21s}{'x'.join(map(str, u['grid_shape'])):16s}"
              f"{'yes' if u['finished'] else 'NO':>6s}{f'{ok}/{off}':>8s}  {r.get('overrides') or ''}")


def compare(a, b):
    print(f"A = {a['benchmark']} {a['run']['id']} (grid {a['run']['grid_shape']}, phi_max {a['run']['phi_max']})")
    print(f"B = {b['benchmark']} {b['run']['id']} (grid {b['run']['grid_shape']}, phi_max {b['run']['phi_max']})")
    diff = {k: (a["params"].get(k), b["params"].get(k)) for k in set(a["params"]) | set(b["params"])
            if a["params"].get(k) != b["params"].get(k)}
    if diff:
        print("parameters that differ:", diff)
    print(f"\n{'metric':22s}{'A':>14s}{'B':>14s}{'B-A':>12s}{'rel':>9s}  unit")
    print("-" * 80)
    for k in a["metrics"]:
        if k not in b["metrics"]:
            continue
        va, vb = a["metrics"][k]["value"], b["metrics"][k]["value"]
        u = a["metrics"][k]["unit"]
        if va is None or vb is None:
            continue
        rel = f"{(vb - va) / abs(va) * 100:+.2f}%" if va else ""
        print(f"{k:22s}{format_value(va, u):>14s}{format_value(vb, u):>14s}{format_value(vb - va, u):>12s}{rel:>9s}  {u}")


def sweep(runs, param, keys, plot=False):
    runs = [r for r in runs if param in r["params"]]
    runs.sort(key=lambda r: r["params"][param])
    keys = keys or [k for k, m in runs[0]["metrics"].items() if m["ref"]] if runs else []
    print(f"{param:>14s}" + "".join(f"{k:>16s}" for k in keys) + "   run")
    for r in runs:
        vals = [r["metrics"].get(k, {}).get("value") for k in keys]
        print(f"{format_value(float(r['params'][param])):>14s}"
              + "".join(f"{format_value(v):>16s}" for v in vals) + f"   {r['run']['id']}")
    if plot and runs:
        import matplotlib.pyplot as plt
        x = [float(r["params"][param]) for r in runs]
        fig, ax = plt.subplots(1, len(keys), figsize=(4 * len(keys), 3.2), layout="constrained", squeeze=False)
        for a, k in zip(ax[0], keys):
            a.plot(x, [r["metrics"].get(k, {}).get("value") for r in runs], "o-")
            ref = runs[0]["metrics"].get(k, {}).get("ref")
            if ref and ref["value"] is not None:
                a.axhline(ref["value"], color="k", ls=":", label="paper")
                a.legend(fontsize=8)
            a.set_xlabel(param)
            a.set_title(f"{k} ({runs[0]['metrics'].get(k, {}).get('unit', '')})", fontsize=9)
        out = os.path.join(os.path.dirname(runs[0]["_dir"]), f"sweep_{param}.png")
        fig.savefig(out, dpi=120)
        print("figure:", out)


def write_csv(runs, path):
    cols = ["benchmark", "run_id", "label", "date", "finished", "grid", "phi_max", "overrides",
            "metric", "value", "unit", "ref", "tol", "kind", "status", "deviation", "description"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in runs:
            u = r["run"]
            for k, m in r["metrics"].items():
                ref = m["ref"] or {}
                w.writerow([r["benchmark"], u["id"], u.get("label"), u["date"], u["finished"],
                            "x".join(map(str, u["grid_shape"])), u.get("phi_max"), json.dumps(r.get("overrides")),
                            k, m["value"], m["unit"], ref.get("value"), ref.get("tol"), ref.get("kind"),
                            m["status"], m.get("deviation"), m["label"]])
    print("written:", path)


# ---------------------------------------------------------------- standard figure
def plot_run(r, show=False):
    import matplotlib.pyplot as plt
    C = np.load(os.path.join(r["_dir"], "curves.npz"))
    M = r["metrics"]
    fig, ax = plt.subplots(2, 3, figsize=(16, 8), layout="constrained")
    fig.suptitle(f"{r['title']}\n{r['run']['id']} | grid {r['run']['grid_shape']} | "
                 f"{r['run']['passes_done']} passes", fontsize=10)

    a = ax[0, 0]                                           # evolution
    n = C["pass"]
    a.plot(n, C["energy"] / C["energy"][0], "o-", ms=3, label="transmission")
    a.plot(n, C["eta00_total"], "s-", ms=3, label="HG00 fraction")
    a.plot(n, C["V_near"], "^-", ms=3, label="homogeneity (near field)")
    a.set_xlabel("pass"); a.legend(fontsize=8)
    b = a.twinx(); b.plot(n, C["B"], "k--", label="B"); b.set_ylabel("B (rad)")

    a = ax[0, 1]                                           # beam
    a.plot(n, C["w_peak_x_um"], "o-", ms=3, label="w x")
    a.plot(n, C["w_peak_y_um"], "s-", ms=3, label="w y")
    a.set_xlabel("pass"); a.set_ylabel("beam radius at centre (um)"); a.legend(fontsize=8, loc="upper left")
    b = a.twinx(); b.semilogy(n, C["I_peak"] * 1e-4, "r^-", ms=3); b.set_ylabel("peak intensity (W/cm2)", color="r")

    a = ax[0, 2]                                           # spectrum
    a.semilogy(C["lam_nm"], C["S_in"], "k:", label="input")
    a.semilogy(C["lam_nm"], C["S_out"], label="simulation")
    if "S_meas" in C:
        a.semilogy(C["S_meas_lam_nm"], C["S_meas"], label="measured (digitised)")
    if "S_ref" in C:
        a.semilogy(C["S_ref_lam_nm"], C["S_ref"], "--", label=str(C["S_ref_label"]))
    ok = C["lam_nm"][C["S_out"] > 1e-3]
    a.set_xlim(ok.min() - 0.1 * np.ptp(ok), ok.max() + 0.1 * np.ptp(ok))
    a.set_ylim(1e-3, 2); a.set_xlabel("wavelength (nm)"); a.legend(fontsize=8)
    a.set_title(f"-20 dB {format_value(M['span20_nm']['value'])} nm, RMS {format_value(M['rms_width_nm']['value'])} nm",
                fontsize=9)

    a = ax[1, 0]                                           # compression
    norm = C["P_tl"].max()
    cen = lambda t, P: t - t[np.argmax(P)]
    a.plot(cen(C["t_tl_fs"], C["P_tl"]), C["P_tl"] / norm, "k:",
           label=f"Fourier limit {format_value(M['ftl_fs']['value'])} fs")
    if "P_comp" in C:
        a.plot(cen(C["t_comp_fs"], C["P_comp"]), C["P_comp"] / norm,
               label=f"paper compressor {format_value(M['comp_fs']['value'])} fs")
    a.plot(cen(C["t_best_fs"], C["P_best"]), C["P_best"] / norm, "--",
           label=f"best GDD {format_value(M['best_comp_fs']['value'])} fs")
    w = 4 * M["ftl_fs"]["value"]
    a.set_xlim(-w, w); a.set_xlabel("time (fs)"); a.set_ylabel("power / Fourier limit"); a.legend(fontsize=8)

    a = ax[1, 1]                                           # S(theta, lambda)
    S, th = C["S_theta"], C["theta_mrad"]
    prof = S.sum(axis=1)
    inside = prof > np.exp(-2) * prof.max()
    a.pcolormesh(C["lam_theta_nm"], th, S, shading="auto", cmap="inferno")
    a.set_ylim(2 * th[inside].min(), 2 * th[inside].max()); a.set_xlim(*ax[0, 2].get_xlim())
    a.set_xlabel("wavelength (nm)"); a.set_ylabel("angle (mrad)")
    a.set_title(f"output S(theta_x, lambda), V = {M['V_weighted_x']['value']*100:.1f} %", fontsize=9)

    a = ax[1, 2]                                           # mirror
    if "mirror_F_Jcm2" in C:
        F, x = C["mirror_F_Jcm2"], C["mirror_x_mm"]
        a.plot(x, F[:, F.shape[1] // 2], label="cut y = 0")
        a.set_xlabel("x on the mirror (mm)"); a.set_ylabel("fluence (J/cm2)")
        a.set_title(f"last mirror, max over run {format_value(M['F_mirror_max_Jcm2']['value'])} J/cm2", fontsize=9)
    out = os.path.join(r["_dir"], "summary.png")
    fig.savefig(out, dpi=110)
    print("figure:", out)
    if show:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------- main
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("benchmarks", nargs="*", help="benchmark names (default: all)")
    p.add_argument("--root", default=os.path.join(HERE, "results"))
    p.add_argument("--latest", action="store_true", help="only the most recent run of each benchmark")
    p.add_argument("--quick", action="store_true", help="include --quick runs")
    p.add_argument("--all-metrics", action="store_true", help="tables: also metrics without a paper value")
    p.add_argument("--csv", default=None, help="write every metric of the selected runs to this file")
    p.add_argument("--compare", nargs=2, metavar=("RUN_A", "RUN_B"))
    p.add_argument("--sweep", nargs="+", metavar=("BENCH PARAM", "METRIC"),
                   help="benchmark, parameter, then metric names")
    p.add_argument("--plot", action="store_true", help="standard figure (summary.png in each run directory)")
    p.add_argument("--show", action="store_true", help="also open the figures")
    a = p.parse_args(argv)

    if a.compare:
        compare(find_run(a.root, a.compare[0]), find_run(a.root, a.compare[1]))
        return
    if a.sweep:
        if len(a.sweep) < 2:
            p.error("--sweep BENCH PARAM [METRIC ...]")
        sweep(load_runs(a.root, [a.sweep[0]], a.quick), a.sweep[1], a.sweep[2:], a.plot)
        return

    runs = load_runs(a.root, a.benchmarks or None, a.quick)
    if a.latest:
        runs = latest(runs)
    if not runs:
        sys.exit(f"no results under {a.root} (use --quick to include quick runs)")
    overview(runs)
    for r in runs:
        print_table(r, only_ref=not a.all_metrics)
    if a.csv:
        write_csv(runs, a.csv)
    if a.plot:
        for r in runs:
            plot_run(r, a.show)


if __name__ == "__main__":
    main()
