"""From a design to the full solver (#16).

    cfg = to_input(problem, best)                       # standard THUNDER4D input (dict)
    save_input(cfg, "my_cell.json")                     # or result.export("my_cell.json")
    report = validate_4d(problem, best)                 # coarse 4D run of all passes vs fast1d
    print(report.summary()); report.plot()

The exported file is self-contained (coating curves inline) and records in its "design" block where
the design comes from: the problem, the predictions of fast1d and the margin to every limit.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from math import log
from typing import Any

import numpy as np

from ..inputfile import default_input, run_input, save_input
from .catalog.gases import get_gas
from .catalog.mirrors import get_mirror

INCH = 0.0254
PREDICTED = ("duration_fs", "ftl_duration_fs", "throughput", "peak_power_W", "compressor_gdd_fs2", "b_total_rad",
             "b_per_pass_rad", "p_over_pcr", "peak_intensity_W_cm2", "main_peak_fraction")


def _table_or_scalar(x):
    if np.ndim(x) == 0:
        return float(x)
    lam, val = x
    return {"wavelength_nm": np.asarray(lam, float).tolist(), "value": np.asarray(val, float).tolist()}


def spectral_window(problem, cand, level=1e-4, margin=0.15):
    """Grid window [nm] holding the predicted output spectrum down to `level` (fast1d metrics),
    widened by `margin` of its span on each side; falls back to a window around the target."""
    lam0 = problem.laser.wavelength_nm
    spec = cand.metrics.get("spectrum_nm")
    if spec is not None:
        lam, s = (np.asarray(a, float) for a in spec)
        keep = s > level * s.max()
        lo, hi = lam[keep].min(), lam[keep].max()
    else:   # bandwidth of a transform-limited gaussian of the target (or 1/10 of the input) duration
        tau = problem.target.target_duration_fs or problem.laser.duration_fs / 10
        dlam = 0.441 * lam0 ** 2 / (2.998e8 * tau * 1e-15) * 1e-9
        lo, hi = lam0 - 3 * dlam, lam0 + 3 * dlam
    span = hi - lo
    lo, hi = min(lo - margin * span, 0.97 * lam0), max(hi + margin * span, 1.03 * lam0)
    return float(lo), float(hi)


def to_input(problem, cand, *, preset="coarse", injection="linear", name=None, window=None,
             precision="single", device="auto", run=None):
    """Standard THUNDER4D input (dict) for one design. window: (lambda_min_nm, lambda_max_nm),
    default from the predicted spectrum. run: overrides of the "run" block."""
    las, L = problem.laser, problem.losses
    gas, coat = get_gas(cand.gas), get_mirror(cand.coating)
    tl = las.tl_duration_fs or las.duration_fs
    gdd = las.prechirp_gdd_fs2 + cand.prechirp_gdd_fs2
    if las.tl_duration_fs and las.duration_fs > las.tl_duration_fs:   # chirped input: positive GDD
        gdd += tl ** 2 / (4 * log(2)) * np.sqrt((las.duration_fs / tl) ** 2 - 1)
    lo, hi = window or spectral_window(problem, cand)
    design = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "candidate": {"gas": cand.gas, "pressure_bar": cand.pressure_bar, "roc_m": cand.roc_m,
                      "round_trips": cand.round_trips, "k": cand.k, "branch": cand.branch,
                      "n_passes": cand.n_passes, "coating": cand.coating, "diameter_inch": cand.diameter_inch,
                      "score": cand.score if np.isfinite(cand.score) else None},
        "predicted_fast1d": {k: cand.metrics[k] for k in PREDICTED if k in cand.metrics},
        "limits": {k: {"value": v[0], "op": v[1], "limit": v[2], "mode": v[3], "margin": cand.slacks.get(k)}
                   for k, v in cand.limits.items()},
        "warnings": list(cand.warnings),
        "problem": {"laser": problem.to_dict()["laser"], "target": problem.to_dict()["target"]},
    }
    cfg = default_input(
        name=name or f"{cand.gas}_{1e3 * cand.pressure_bar:.0f}mbar_R{cand.roc_m:g}_M{cand.round_trips}k{cand.k}",
        laser={"energy_J": las.energy_J * L.mode_matching_efficiency, "duration_fs": tl,
               "wavelength_nm": las.wavelength_nm, "gdd_fs2": gdd, "tod_fs3": las.prechirp_tod_fs3},
        beam={"injection": injection},
        medium={"gas": cand.gas, "pressure_bar": cand.pressure_bar,
                "n2_1bar": gas.n2_per_bar * las.n2_polarization_factor},
        cell={"roc_m": cand.roc_m, "separation_m": cand.separation_m, "n_passes": cand.n_passes,
              "herriott": {"round_trips": cand.round_trips, "k": cand.k, "branch": cand.branch}},
        mirrors={"reflectivity": _table_or_scalar(coat.reflectivity), "gdd_fs2": _table_or_scalar(coat.gdd_fs2),
                 "diameter_m": cand.diameter_inch * INCH, "coating": cand.coating},
        grid={"preset": preset, "lambda_min_nm": lo, "lambda_max_nm": hi, "precision": precision,
              "device": device},
        run={"record_every": 1, "near_field": False, "record_mirror": True, **(run or {})},
        design=design,
    )
    if coat.r_in_percent and isinstance(cfg["mirrors"]["reflectivity"], dict):
        cfg["mirrors"]["reflectivity"]["value"] = [v / 100 for v in cfg["mirrors"]["reflectivity"]["value"]]
    if las.polarization == "circular":
        cfg["notes"] = "circular polarisation: n2_1bar already includes the factor 2/3"
    return cfg


def export_input(problem, cand, path, **kw):
    """Write one design as a standard THUNDER4D input file."""
    return save_input(to_input(problem, cand, **kw), path)


# ============================================================ 4D validation
def compress_oversampled(grid, Aw, energy, gdd_bounds_fs2, up=8, rel=1e-4, n_scan=61):
    """Best uniform GDD for the spatially integrated power, as diagnostics.compress, but on a time
    axis oversampled `up` times (zero-padded spectrum): coarse grids hold few-cycle pulses with only
    a few samples. Power normalised to `energy`."""
    from scipy import fft as sfft

    from ..backend import to_host
    FS = 1e-15
    A = to_host(Aw)
    F = np.sum(np.abs(A) ** 2, axis=2)
    E = A[F > rel * F.max()].astype(np.complex128)              # (npix, Nt), FFT order in Omega
    W = grid.Omega * FS
    nt = E.shape[1]
    n, k = nt * up, (nt + 1) // 2
    dt = grid.dt / up
    t = (np.arange(n) - n // 2) * dt / FS

    def power(Ew):
        Ep = np.zeros((Ew.shape[0], n), complex)
        Ep[:, :k], Ep[:, n - (nt - k):] = Ew[:, :k], Ew[:, k:]
        P = np.fft.fftshift(np.sum(np.abs(sfft.fft(Ep, axis=1)) ** 2, axis=0))   # same convention as ift_t
        return P * energy / (P.sum() * dt)

    def peak(g):
        return power(E * np.exp(1j * g / 2 * W ** 2)[None, :]).max()

    scan = np.linspace(*gdd_bounds_fs2, n_scan)
    pk = [peak(g) for g in scan]
    i = int(np.argmax(pk))
    a, b = scan[max(i - 1, 0)], scan[min(i + 1, n_scan - 1)]
    gr = (np.sqrt(5) - 1) / 2
    for _ in range(40):
        c, d = b - gr * (b - a), a + gr * (b - a)
        if peak(c) > peak(d):
            b = d
        else:
            a = c
    g = (a + b) / 2
    P = power(E * np.exp(1j * g / 2 * W ** 2)[None, :])
    Ptl = power(np.abs(E))
    from ..diagnostics import _fwhm
    return {"gdd_fs2": g, "t_fs": t, "P": P, "fwhm_fs": _fwhm(t, P), "tl_fwhm_fs": _fwhm(t, Ptl),
            "peak_power": float(P.max()), "peak_power_TL": float(Ptl.max())}



@dataclass
class Validation4D:
    cfg: dict
    rows: list                      # (quantity, fast1d, 4D, relative difference)
    extra: dict                     # 4D-only results (M2, HG00, ...)
    built: Any = field(repr=False, default=None)      # the built 4D simulation (thunder4d.inputfile.Built)
    fast1d: Any = field(repr=False, default=None)     # the fast1d MPCOutput
    runtime_s: float = 0.0
    comp: dict = field(repr=False, default_factory=dict)   # 4D compression analysis (pulse, GDD, FWHM)

    def summary(self):
        g = self.built.grid
        lines = [(f"4D validation of {self.cfg['name']} ({self.cfg['grid']['preset']} grid "
                  f"{g.Nx}x{g.Ny}x{g.Nt}, {self.cfg['cell']['n_passes']} passes, {self.runtime_s:.0f} s)"),
                 f"  {'quantity':<26}{'fast1d':>12}{'4D':>12}{'diff':>9}"]
        for q, a, b, d in self.rows:
            ds = f"{100 * d:+8.1f}%" if d is not None and np.isfinite(d) else f"{'':>9}"
            lines.append(f"  {q:<26}{a:>12.4g}{b:>12.4g}{ds}")
        for q, v in self.extra.items():
            lines.append(f"  {q:<26}{'':>12}{v:>12.4g}")
        return "\n".join(lines)

    def plot(self):
        from ._plot import pyplot

        plt = pyplot()

        from .. import diagnostics as dg
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        lam, s = self.fast1d.spectrum_lambda("out")
        ax[0].plot(lam * 1e9, s / s.max(), label="fast1d")
        lam4, s4 = dg.total_spectrum(self.built.grid, self.built.sim.Aw)
        ax[0].plot(lam4 * 1e9, s4 / s4.max(), "--", label="4D")
        g = self.cfg["grid"]
        ax[0].set(xlabel="wavelength (nm)", ylabel="spectrum (norm.)", yscale="log", ylim=(1e-4, 1.5),
                  xlim=(g["lambda_min_nm"], g["lambda_max_nm"]))
        ax[0].legend()
        c = self.comp
        t1 = self.fast1d.t_compressed * 1e15
        p1 = np.abs(self.fast1d.field_compressed) ** 2
        ax[1].plot(t1 - t1[np.argmax(p1)], p1 / p1.max(), label=f"fast1d {self.fast1d.fwhm_compressed * 1e15:.1f} fs")
        # fast1d and the 4D code use opposite Fourier sign conventions: the 4D time axis is reversed
        t4 = -(c["t_fs"] - c["t_fs"][np.argmax(c["P"])])
        ax[1].plot(t4, c["P"] / c["P"].max(), "--", label=f"4D {c['fwhm_fs']:.1f} fs")
        w = 4 * max(c["fwhm_fs"], self.fast1d.fwhm_compressed * 1e15)
        ax[1].set(xlabel="time (fs)", ylabel="power (norm.)", xlim=(-w, w))
        ax[1].legend()
        fig.tight_layout()
        return fig


def validate_4d(problem, cand, *, quick=None, preset="coarse", verbose=True, max_seconds=None, **to_input_kw):
    """Run the design with the 4D lens-frame solver (all passes) and compare it with fast1d
    ("default" accuracy). Coarse grid by default: minutes; preset="fine" for benchmark grade."""
    from .. import diagnostics as dg
    from .fast1d_quick import Fast1DQuick

    quick = quick or Fast1DQuick()
    ref = quick.simulate(problem, cand, accuracy="default")
    cfg = to_input(problem, cand, preset=preset, injection="nonlinear" if quick.matching == "nonlinear" else "linear",
                   **to_input_kw)
    t0 = time.time()
    b = run_input(cfg, verbose=verbose, max_seconds=max_seconds)
    sim, grid = b.sim, b.grid
    h = sim.history
    gdd1 = ref.gdd_compression * 1e30
    bounds = (min(3 * gdd1, -500.0), max(-0.2 * gdd1, 200.0)) if gdd1 < 0 else (-500.0, max(3 * gdd1, 500.0))
    comp = compress_oversampled(grid, sim.Aw, h["energy"][-1], bounds)
    fm = [f for f in h["fluence_mirror"] if f is not None]
    f4 = max(float(f.max()) for f in fm) / sim.mirror_scale ** 2 * 1e-4 if fm else np.nan
    lam1, s1 = ref.spectrum_lambda("out")
    lam4, s4 = dg.total_spectrum(grid, sim.Aw)
    s4i = np.interp(lam1, lam4, s4, left=0.0, right=0.0)
    dist = 0.5 * float(np.abs(s1 / s1.sum() - s4i / max(s4i.sum(), 1e-300)).sum())

    e_laser = problem.laser.energy_J

    def rel(a, b):
        return (b - a) / abs(a) if a else None
    pairs = [
        ("transmission", ref.energy_out / e_laser, h["energy"][-1] / e_laser),
        ("B-integral [rad]", ref.b_integral, sim.solver.B),
        ("FTL duration [fs]", ref.fwhm_ftl * 1e15, comp["tl_fwhm_fs"]),
        ("compressed duration [fs]", ref.fwhm_compressed * 1e15, comp["fwhm_fs"]),
        ("compressor GDD [fs^2]", gdd1, comp["gdd_fs2"]),
        ("compressed peak [GW]", ref.peak_power_compressed / 1e9, comp["peak_power"] / 1e9),
        ("mirror fluence [J/cm2]", ref.fluence_mirror_max * 1e-4, f4),
    ]
    rows = [(q, float(a), float(b_), rel(a, b_)) for q, a, b_ in pairs]
    rows.append(("spectrum (L1 distance)", 0.0, dist, None))
    extra = {"M2 (x, mean)": float(h["M2x_mean"][-1]), "M2 (y, mean)": float(h["M2y_mean"][-1]),
             "HG00 fraction": float(h["eta00_total"][-1]),
             "peak / FTL peak (4D)": comp["peak_power"] / comp["peak_power_TL"]}
    v = Validation4D(cfg, rows, extra, b, ref, time.time() - t0, comp)
    if verbose:
        print(v.summary())
    return v
