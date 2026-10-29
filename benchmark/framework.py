"""Common machinery of the thunder4d benchmarks (lens-frame solver).

A benchmark is a subclass of `Benchmark` that only describes the physics: the paper, the parameters,
the reference values and, if needed, how to build the cell / spectrum / extra metrics. Everything
else is shared, so every benchmark is run, stored and analysed the same way:

    python bench_grebing2020_1mJ.py                       # full run
    python bench_grebing2020_1mJ.py --quick               # 2 passes, coarse grid: checks the script
    python bench_grebing2020_1mJ.py --phi-max 0.01        # convergence check (new run directory)
    python bench_daniault2024_4fs.py --set pressure_bar=2.2 --label p2200
    python bench_kaumanns2018_18mJ.py --max-seconds 3600  # stops cleanly, run again to continue

Every run writes to  results/<benchmark>/<run_id>/ :

    results.json   metadata (version, git commit, device, grid, timing), parameters, and the metrics
                   {name: {value, unit, label, ref: {value, tol, kind, note}, status, deviation}}
    curves.npz     standard curves (history per pass, spectra, compressed pulses, S(theta, lambda),
                   mirror fluence) used by analyze.py to draw the same figures for every benchmark
    run.npz        sim.save() of the run (history and axes)
    checkpoint.pkl state of the simulation (resume)

run_id = <label>_<fingerprint>: the fingerprint hashes the parameters, the grid preset and the
numerical settings, so changing any of them starts a new run and never overwrites an old one.
analyze.py reads the results.json files: tables against the papers, comparison of two runs,
parameter sweeps, figures.
"""
from __future__ import annotations

import argparse
import ast
import datetime
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, asdict

import numpy as np

SCHEMA_VERSION = 1
FS = 1e-15
HERE = os.path.dirname(os.path.abspath(__file__))


# ====================================================================== reference values
@dataclass
class Ref:
    """Published value of a metric and how close the simulation should be.

    kind = "rel" : |v - value| <= tol * |value|
           "abs" : |v - value| <= tol
           "max" : v <= value          (paper quotes an upper bound, e.g. M2 < 1.15)
           "min" : v >= value          (paper quotes a lower bound, e.g. V > 99 %)
           "info": value shown for comparison only, never checked
    """
    value: float | None
    tol: float | None = None
    kind: str = "rel"
    note: str = ""

    def deviation(self, v):
        """Signed relative deviation (v - ref) / ref, or None."""
        if self.value in (None, 0) or v is None or not np.isfinite(v):
            return None
        return (v - self.value) / abs(self.value)

    def status(self, v):
        if v is None or not np.isfinite(v):
            return "nan"
        if self.kind == "info" or self.value is None:
            return "info"
        if self.kind == "max":
            return "ok" if v <= self.value else "off"
        if self.kind == "min":
            return "ok" if v >= self.value else "off"
        if self.tol is None:
            return "info"
        lim = self.tol * abs(self.value) if self.kind == "rel" else self.tol
        return "ok" if abs(v - self.value) <= lim else "off"

    def text(self):
        if self.value is None:
            return self.note or "-"
        s = {"max": "< ", "min": "> "}.get(self.kind, "")
        s += f"{self.value:.4g}"
        if self.kind == "rel" and self.tol:
            s += f" ±{self.tol*100:.0f}%"
        elif self.kind == "abs" and self.tol:
            s += f" ±{self.tol:.3g}"
        return s


def metric(value, unit="", label="", ref: Ref | None = None):
    v = None if value is None else float(value)
    out = {"value": v, "unit": unit, "label": label, "ref": asdict(ref) if ref else None}
    if ref is not None:
        out["status"], out["deviation"] = ref.status(v), ref.deviation(v)
    else:
        out["status"], out["deviation"] = "info", None
    return out


# ====================================================================== numerical helpers
def band_mirror(R_in, lam_lo, lam_hi, R_out=0.0, order=80):
    """Intensity reflectivity R(lambda) [m]: R_in inside the band, smooth edges with their 50 %
    points at lam_lo and lam_hi, R_out far outside."""
    lc, hw = 0.5 * (lam_lo + lam_hi), 0.5 * (lam_hi - lam_lo)

    def R(lam):
        u = np.abs((np.asarray(lam, float) - lc) / hw)
        return R_out + (R_in - R_out) * np.exp(-np.log(2) * u ** order)
    return R


def spectral_phase(grid, omega_c, gdd_fs2=0.0, tod_fs3=0.0):
    """GDD [fs^2] and TOD [fs^3] expanded about omega_c (the pulse centre, not necessarily the
    grid carrier), on grid.Omega (FFT order)."""
    W = (grid.omega - omega_c) * FS
    return (gdd_fs2 / 2 * W ** 2 + tod_fs3 / 6 * W ** 3) * grid.valid


def beam_pixels(Aw, rel=1e-4):
    """(npix, Nt) host array of the pixels that carry the beam (fluence > rel x max)."""
    import thunder4d as th
    A = th.to_host(Aw)
    F = np.sum(np.abs(A) ** 2, axis=2)
    return A[F > rel * F.max()]


def _pad(E, up):
    if up == 1:
        return E
    N = E.shape[-1]
    h = (N + 1) // 2
    out = np.zeros(E.shape[:-1] + (N * up,), complex)
    out[..., :h] = E[..., :h]
    out[..., N * up - (N - h):] = E[..., h:]
    return out


def integrated_power(grid, E_pix, phase=None, up=4):
    """Spatially integrated P(t) [W] after the spectral phase `phase` (array on grid.Omega).
    Returns sorted t [fs] and P [W] with a time step dt/up."""
    E = E_pix if phase is None else E_pix * np.exp(1j * phase)[None, :]
    A = grid.hfft.ift_t(_pad(E, up)) * up
    P = np.sum(np.abs(A) ** 2, axis=0) * grid.dA
    t = np.fft.fftfreq(grid.Nt * up, 1.0 / (grid.Nt * grid.dt))
    return np.fft.fftshift(t) / FS, np.fft.fftshift(P)


def fourier_limit(grid, E_pix, up=4):
    """Fourier limit of the spatially integrated spectrum (the 'Fourier limit' of the papers)."""
    S = np.sum(np.abs(E_pix) ** 2, axis=0)
    return integrated_power(grid, np.sqrt(S)[None, :], None, up)


def best_phase(grid, E_pix, phase_of, bounds, n_scan=61, n_golden=30, up=4):
    """Parameter p in `bounds` maximising the peak power after phase_of(p): coarse scan then
    golden section. Returns (p_best, scan values, peak powers of the scan)."""
    peak = lambda p: integrated_power(grid, E_pix, phase_of(p), up)[1].max()
    scan = np.linspace(*bounds, n_scan)
    pk = np.array([peak(p) for p in scan])
    i = int(np.argmax(pk))
    a, b = scan[max(i - 1, 0)], scan[min(i + 1, n_scan - 1)]
    gr = (np.sqrt(5) - 1) / 2
    c, d = b - gr * (b - a), a + gr * (b - a)
    fc, fd = peak(c), peak(d)
    for _ in range(n_golden):
        if fc > fd:
            b, d, fd = d, c, fc
            c = b - gr * (b - a)
            fc = peak(c)
        else:
            a, c, fc = c, d, fd
            d = a + gr * (b - a)
            fd = peak(d)
    return 0.5 * (a + b), scan, pk


def fwhm(x, y):
    """FWHM of the main peak (linear interpolation)."""
    y = np.asarray(y, float)
    i = int(np.argmax(y))
    h = y[i] / 2
    l, r = i, i
    while l > 0 and y[l] > h:
        l -= 1
    while r < len(y) - 1 and y[r] > h:
        r += 1
    xl = np.interp(h, [y[l], y[l + 1]], [x[l], x[l + 1]])
    xr = np.interp(h, [y[r], y[r - 1]], [x[r], x[r - 1]])
    return float(xr - xl)


def span(lam, S, level_db=-20):
    """(lambda_min, lambda_max) [nm] of the outermost points above level_db."""
    ok = np.nonzero(S / S.max() >= 10 ** (level_db / 10))[0]
    return lam[ok[0]] * 1e9, lam[ok[-1]] * 1e9


def rms_width(lam, S):
    """Centre of mass and RMS width [nm] of a spectrum on a wavelength axis [m]."""
    lc = np.sum(lam * S) / np.sum(S)
    return lc * 1e9, float(np.sqrt(np.sum((lam - lc) ** 2 * S) / np.sum(S)) * 1e9)


def homogeneity_1d(S, rel=np.exp(-2)):
    """Spectral homogeneity of Weitenberg et al. (Opt. Express 25, 20502, 2017), as in the papers:
    V(x) = (int sqrt(S S_ref))^2 / (int S int S_ref), S_ref = spectrum at the intensity maximum,
    S[x, lambda]. Returns V(x) and its intensity-weighted mean inside the 1/e^2 width."""
    prof = S.sum(axis=1)
    S0 = S[np.argmax(prof)]
    V = np.sum(np.sqrt(S * S0[None, :]), axis=1) ** 2 / np.maximum(S.sum(axis=1) * S0.sum(), 1e-300)
    inside = prof > rel * prof.max()
    return V, float(np.sum(V[inside] * prof[inside]) / np.sum(prof[inside]))


def spectral_similarity(la, Sa, lb, Sb):
    """(int sqrt(Sa Sb))^2 / (int Sa int Sb) on the axis la (1 = identical shapes)."""
    Sa = np.clip(np.asarray(Sa, float), 0, None)
    Sb = np.clip(np.interp(la, lb, Sb, left=0, right=0), 0, None)
    return float(np.trapezoid(np.sqrt(Sa * Sb), la) ** 2
                 / (np.trapezoid(Sa, la) * np.trapezoid(Sb, la)))


def load_digitized(path):
    """Optional curve digitised from a paper figure: CSV with 2 columns (x, y), y normalised."""
    if not path:
        return None
    path = path if os.path.isabs(path) else os.path.join(HERE, path)
    if not os.path.exists(path):
        return None
    d = np.loadtxt(path, delimiter=",")
    d = d[np.argsort(d[:, 0])]
    return d[:, 0], d[:, 1] / d[:, 1].max()


def _git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=HERE,
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


def _jsonable(v):
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    if isinstance(v, (tuple, list)):
        return [_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {k: _jsonable(x) for k, x in v.items()}
    return v


# ====================================================================== simulation
def _simulation_class():
    import thunder4d as th

    class BenchSimulation(th.LensFrameSimulation):
        """LensFrameSimulation that also records the exact peak intensity at the cell centre."""

        def record(self, p, mirror_fluence=None, mirror_profiles=None):
            super().record(p, mirror_fluence, mirror_profiles)
            self.history.setdefault("I_peak", []).append(self.solver.peak_intensity(self.Aw))

    return BenchSimulation


# ====================================================================== the benchmark
class Benchmark:
    """Base class. Subclasses set the class attributes and override the hooks they need."""

    name = "unnamed"                  # results/<name>/ ; also the key used by analyze.py
    title = ""
    paper = {}                        # {"citation": ..., "doi": ..., "url": ...}
    params = {}                       # physical parameters; any of them can be changed with --set
    reference = {}                    # {metric name: Ref(...)}
    measured = {}                     # optional digitised curves: {"spectrum": "data/....csv"}

    # grid: presets for suggest_lens_grid, finest first; the finest that fits in memory is used
    presets = [dict(box_factor=8, points_per_waist=8, time_window_factor=8),
               dict(box_factor=7, points_per_waist=7, time_window_factor=7),
               dict(box_factor=6, points_per_waist=6, time_window_factor=6)]
    quick_preset = dict(box_factor=5, points_per_waist=3, time_window_factor=4)
    quick_passes = 2
    n_fields = 10                     # full-size arrays alive during a step (memory estimate)
    record_every = 1                  # full diagnostics every n passes (last pass always)
    compression_up = 4                # time-axis oversampling of the compressed pulses
    gdd_bounds = None                 # search range of the best GDD [fs^2]; default from params

    # ---------------------------------------------------------------- hooks
    def make_cell(self, P, n_passes):
        raise NotImplementedError

    def grid_carrier(self, P):
        """Carrier of the grid [m]; default the pulse centre lambda0."""
        return P["lambda0"]

    def grid_args(self, P):
        return dict(lambda0=self.grid_carrier(P), lambda_min=P["lambda_min"],
                    lambda_max=P["lambda_max"], pulse_fwhm=P.get("window_fwhm", P["fwhm"]))

    def spectrum(self, grid, P):
        """Complex spectral amplitude E(Om). Default: gaussian at lambda0 with optional GDD/TOD
        (expanded about lambda0)."""
        import thunder4d as th
        E = th.gaussian_spectrum(grid, fwhm_fs=P["fwhm"] / FS, center_nm=P["lambda0"] * 1e9)
        wc = 2 * np.pi * th.C0 / P["lambda0"]
        ph = spectral_phase(grid, wc, P.get("gdd_fs2", 0.0), P.get("tod_fs3", 0.0))
        return E * np.exp(1j * ph), "gaussian"

    def extra_metrics(self, ctx):
        """Benchmark-specific metrics and curves: return (metrics dict, curves dict)."""
        return {}, {}

    # ---------------------------------------------------------------- shared steps
    def pulse(self, ctx):
        """Gaussian beam at the cell centre, waist of the linear or Kerr-matched eigenmode.
        Beam sizes are given at lambda0 and scaled to the grid carrier (w ~ sqrt(lambda))."""
        import thunder4d as th
        P, grid, cell = ctx.P, ctx.grid, ctx.cell
        at_grid = lambda w: w * np.sqrt(grid.lambda0 / P["lambda0"])
        E, src = self.spectrum(grid, P)
        lin = cell.linear_mode(P["lambda0"])
        probe = th.Pulse.at_focus(grid, E, th.Beam(th.gaussian(at_grid(lin["w0"]))), P["energy"])
        try:
            mm = cell.nonlinear_mode(probe.peak_power(), P["lambda0"])
        except ValueError as err:                 # P >= Pc: no matched mode
            print("  ", err)
            mm = None
        w0 = lin["w0"] if (P.get("injection", "linear") == "linear" or mm is None) else mm["w0"]
        beam = th.Beam(th.gaussian(at_grid(w0)), chromatic_size="eigenmode")
        pulse = th.Pulse.at_focus(grid, E, beam, P["energy"])
        ctx.lin, ctx.mm, ctx.w0_ref, ctx.spectrum_source = lin, mm, w0, src
        ctx.mode_reference = at_grid(w0)
        return pulse

    def pick_grid(self, ctx, args):
        import thunder4d as th
        cell, ga = ctx.cell, self.grid_args(ctx.P)
        if args.quick:
            presets = [self.quick_preset]
        elif args.preset is not None:
            presets = [self.presets[args.preset]]
        else:
            presets = self.presets
        budget = _budget_gb(args)
        for p in presets:
            g = th.suggest_lens_grid(cell, **ga, **p, device="cpu")       # axes only: cheap
            need = self.n_fields * np.prod(g.shape) * 8 / 1e9
            ok = need <= budget or args.preset is not None or args.quick
            print(f"  preset {p}: {g.Nx}x{g.Ny}x{g.Nt}, ~{need:.2f} GB" + ("  <- used" if ok else ""))
            if ok:
                return th.suggest_lens_grid(cell, **ga, **p, device=args.device), p
        raise MemoryError("no preset fits the memory budget: add a coarser preset, raise "
                          "--budget-gb or force one with --preset")

    # ---------------------------------------------------------------- standard metrics
    def standard_metrics(self, ctx):
        import thunder4d as th
        from thunder4d import diagnostics as dg
        sim, grid, P, ref = ctx.sim, ctx.grid, ctx.P, self.reference
        h = sim.history
        R = lambda k: ref.get(k)
        M, C = {}, {}
        wc = 2 * np.pi * th.C0 / P["lambda0"]
        up = self.compression_up

        # --- geometry
        M["w0_um"] = metric(ctx.lin["w0"] * 1e6, "um", "linear eigenmode waist (1/e2 radius)", R("w0_um"))
        M["w_mirror_um"] = metric(ctx.lin["w_mirror"] * 1e6, "um", "linear mode on mirror (1/e2 radius)",
                                  R("w_mirror_um"))
        M["w0_injected_um"] = metric(ctx.w0_ref * 1e6, "um", f"injected waist ({P.get('injection', 'linear')})")
        if ctx.mm is not None:
            M["P_over_Pc"] = metric(ctx.mm["P_over_Pc"], "", "input peak power / critical power", R("P_over_Pc"))
            M["B_estimate_rad"] = metric(ctx.mm["B_total"], "rad", "B-integral, matched-mode estimate")

        # --- energy, nonlinear phase, intensity
        E = np.array(h["energy"])
        M["transmission"] = metric(E[-1] / E[0], "", "cell transmission", R("transmission"))
        M["B_rad"] = metric(sim.solver.B, "rad", "B-integral, on-axis peak", R("B_rad"))
        I = np.array(h["I_peak"])
        M["I_peak_max_Wcm2"] = metric(I.max() * 1e-4, "W/cm2", "max peak intensity at the cell centre",
                                      R("I_peak_max_Wcm2"))
        Fm = [f for f in h["fluence_mirror"] if f is not None]
        if Fm:
            s = sim.mirror_scale
            Fmax = max(float(f.max()) for f in Fm) / s ** 2          # frame -> physical
            M["F_mirror_max_Jcm2"] = metric(Fmax * 1e-4, "J/cm2", "max fluence on the mirrors",
                                            R("F_mirror_max_Jcm2"))
            x = grid.xs * s
            F = Fm[-1] / s ** 2
            C["mirror_x_mm"], C["mirror_F_Jcm2"] = x * 1e3, F * 1e-4
        # energy near the edge of the box (absorber): must stay small
        rr = np.sqrt(grid.xs[:, None] ** 2 + grid.ys[None, :] ** 2)
        edge = rr > 0.7 * 0.5 * grid.Nx * grid.dx
        ef = max(float(f[edge].sum() / f.sum()) for f in h["fluence"])
        M["edge_fraction"] = metric(ef, "", "max fluence fraction outside 70 % of the box (should be ~0)",
                                    Ref(1e-3, kind="max", note="numerical check"))

        # --- beam
        M["eta00_final"] = metric(h["eta00_total"][-1], "", "HG00 fraction, output", R("eta00_final"))
        M["M2x"] = metric(h["M2x_mean"][-1], "", "M2 x (spectrally weighted)", R("M2x"))
        M["M2y"] = metric(h["M2y_mean"][-1], "", "M2 y (spectrally weighted)", R("M2y"))
        w = np.array(h["w_peak_x"])
        M["w_breathing"] = metric((w.max() - w.min()) / w[0], "", "beam-size oscillation (max-min)/w_in")

        # --- spectrum
        lam, S = dg.total_spectrum(grid, sim.Aw)
        lo, hi = span(lam, S, -20)
        lc, rms = rms_width(lam, S)
        M["span20_nm"] = metric(hi - lo, "nm", "-20 dB spectral width", R("span20_nm"))
        M["rms_width_nm"] = metric(rms, "nm", "RMS spectral width", R("rms_width_nm"))
        M["lambda_centroid_nm"] = metric(lc, "nm", "spectral centre of mass")
        C["lam_nm"], C["S_in"], C["S_out"] = lam * 1e9, h["spectrum"][0] / np.max(h["spectrum"][0]), S / S.max()
        meas = load_digitized(self.measured.get("spectrum"))
        if meas is not None:
            C["S_meas_lam_nm"], C["S_meas"] = meas
            M["similarity_measured"] = metric(spectral_similarity(meas[0], meas[1], lam * 1e9, S / S.max()),
                                              "", "spectral similarity with the measurement",
                                              R("similarity_measured"))

        # --- compression
        Epix = beam_pixels(sim.Aw)
        t_tl, P_tl = fourier_limit(grid, Epix, up)
        M["ftl_fs"] = metric(fwhm(t_tl, P_tl), "fs", "Fourier limit (integrated spectrum)", R("ftl_fs"))
        C["t_tl_fs"], C["P_tl"] = t_tl, P_tl
        cg = P.get("compressor_gdd")
        if cg is not None:
            t_c, P_c = integrated_power(grid, Epix, spectral_phase(grid, wc, cg), up)
            M["comp_fs"] = metric(fwhm(t_c, P_c), "fs", f"compressed with the paper's {cg:.0f} fs2",
                                  R("comp_fs"))
            M["comp_peak_ratio"] = metric(P_c.max() / P_tl.max(), "", "peak power / Fourier limit (paper GDD)",
                                          R("comp_peak_ratio"))
            C["t_comp_fs"], C["P_comp"] = t_c, P_c
        bounds = self.gdd_bounds or ((3 * cg, -0.2 * cg) if cg and cg < 0 else (-20000, 20000))
        g, _, _ = best_phase(grid, Epix, lambda d: spectral_phase(grid, wc, d), bounds, up=up)
        t_b, P_b = integrated_power(grid, Epix, spectral_phase(grid, wc, g), up)
        M["best_gdd_fs2"] = metric(g, "fs2", "GDD maximising the peak power", R("best_gdd_fs2"))
        M["best_comp_fs"] = metric(fwhm(t_b, P_b), "fs", "compressed with the best GDD", R("best_comp_fs"))
        M["best_peak_ratio"] = metric(P_b.max() / P_tl.max(), "", "peak power / Fourier limit (best GDD)")
        M["best_peak_power_GW"] = metric(P_b.max() / 1e9, "GW", "compressed peak power (best GDD)")
        C["t_best_fs"], C["P_best"] = t_b, P_b

        # --- spatio-spectral homogeneity of the collimated output (near field), along x
        theta, lam_t, S_th = dg.spectrogram_theta_lambda(grid, sim.Aw, per_lambda=False)
        V_x, V_mean = homogeneity_1d(S_th)
        prof = S_th.sum(axis=1)
        M["V_weighted_x"] = metric(V_mean, "", "homogeneity, intensity weighted in 1/e2 (x)", R("V_weighted_x"))
        M["V_min_fwhm"] = metric(np.nanmin(V_x[prof >= 0.5 * prof.max()]), "", "min homogeneity within FWHM",
                                 R("V_min_fwhm"))
        M["V_min_e2"] = metric(np.nanmin(V_x[prof >= np.exp(-2) * prof.max()]), "", "min homogeneity within 1/e2",
                               R("V_min_e2"))
        C["theta_mrad"], C["lam_theta_nm"] = theta * 1e3, lam_t * 1e9
        C["S_theta"], C["V_theta"] = (S_th / S_th.max()).astype(np.float32), V_x

        # --- history
        C["pass"] = np.array(h["pass"])
        for k in ("energy", "B", "eta00_total", "V_near", "V_far", "M2x_mean", "M2y_mean",
                  "peak_power", "I_peak"):
            C[k] = np.array(h[k], float)
        C["w_peak_x_um"] = np.array(h["w_peak_x"]) * 1e6
        C["w_peak_y_um"] = np.array(h["w_peak_y"]) * 1e6
        return M, C

    # ---------------------------------------------------------------- main
    def main(self, argv=None):
        args = parse_args(self, argv)
        if args.gpu_id is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
        import thunder4d as th

        P = dict(self.params)
        for k, v in args.overrides.items():
            if k not in P:
                raise KeyError(f"unknown parameter {k!r}; parameters: {', '.join(P)}")
            P[k] = v
        if args.quick:
            P["n_passes"] = min(self.quick_passes, P["n_passes"])
        ctx = Context(P=P, args=args)
        n_passes = P["n_passes"]
        print(f"=== {self.name}: {self.title}")
        print(f"thunder4d {th.__version__} | GPU available: {th.gpu_available()} | device {args.device}")
        ctx.cell = self.make_cell(P, n_passes)
        print(ctx.cell.report())
        ctx.grid, ctx.preset = self.pick_grid(ctx, args)
        print(ctx.grid.describe())
        pulse = self.pulse(ctx)
        print(f"input: {ctx.spectrum_source}, {P['energy']*1e3:.4g} mJ, peak {pulse.peak_power()/1e9:.3g} GW, "
              f"injected waist {ctx.w0_ref*1e6:.0f} um ({P.get('injection', 'linear')})")

        settings = dict(phi_max=args.phi_max, gouy_step=args.gouy_step, quick=args.quick,
                        n_passes=n_passes, preset=ctx.preset, record_every=args.record_every)
        fp = json.dumps({"params": {k: str(v) for k, v in sorted(P.items())},
                         "settings": {k: str(v) for k, v in sorted(settings.items())}})
        run_id = f"{args.label or ('quick' if args.quick else 'run')}_{hashlib.md5(fp.encode()).hexdigest()[:8]}"
        run_dir = os.path.join(args.out, self.name, run_id)
        os.makedirs(run_dir, exist_ok=True)
        print("run directory:", run_dir)

        Sim = _simulation_class()
        sim = Sim(pulse, ctx.cell, phi_max=args.phi_max, gouy_step=args.gouy_step,
                  mode_reference=ctx.mode_reference, near_field=True, record_mirror=True,
                  record_every=args.record_every)
        ckpt = os.path.join(run_dir, "checkpoint.pkl")
        prev_wall = 0.0
        if not args.no_resume and os.path.exists(ckpt):
            sim.resume(ckpt)
            if os.path.exists(os.path.join(run_dir, "results.json")):
                with open(os.path.join(run_dir, "results.json")) as f:
                    prev_wall = json.load(f)["run"].get("wall_s", 0.0)
            print(f"resumed at pass {sim.passes_done}")
        t0 = time.time()
        if not sim.done:
            sim.run(max_seconds=args.max_seconds)
            sim.checkpoint(ckpt)
        wall = prev_wall + time.time() - t0
        ctx.sim, ctx.run_dir = sim, run_dir

        print("analysing ...")
        M, C = self.standard_metrics(ctx)
        M2, C2 = self.extra_metrics(ctx)
        M.update(M2)
        C.update(C2)

        g = ctx.grid
        result = {
            "schema": SCHEMA_VERSION,
            "benchmark": self.name, "title": self.title, "paper": self.paper,
            "run": {"id": run_id, "label": args.label, "date": datetime.datetime.now().isoformat(timespec="seconds"),
                    "thunder4d": th.__version__, "git": _git_commit(), "python": sys.version.split()[0],
                    "device": g.device, "fft": g.fft.backend, "solver": "LensFrameSimulation",
                    "grid_shape": list(g.shape), "grid": g.describe(),
                    "dx_um": g.dx * 1e6, "dt_fs": g.dt / FS, "mirror_scale": sim.mirror_scale,
                    "finished": bool(sim.done), "passes_done": sim.passes_done, "wall_s": wall,
                    "nsteps": sim.solver.nsteps, "command": " ".join(sys.argv), **_jsonable(settings)},
            "params": _jsonable(P),
            "overrides": _jsonable(args.overrides),
            "metrics": M,
        }
        with open(os.path.join(run_dir, "results.json"), "w") as f:
            json.dump(result, f, indent=1)
        np.savez_compressed(os.path.join(run_dir, "curves.npz"), **C)
        sim.save(os.path.join(run_dir, "run.npz"), include_field=args.save_field)
        print_table(result)
        print(f"\n{'finished' if sim.done else 'NOT finished (time budget): run the same command again'}"
              f" | {wall:.0f} s | results in {run_dir}")
        return result


class Context:
    """Everything a hook may need: P, args, cell, grid, preset, lin, mm, w0_ref, sim, run_dir."""

    def __init__(self, **kw):
        self.__dict__.update(kw)


def _budget_gb(args):
    if args.device != "cpu":
        import thunder4d as th
        if th.gpu_available():
            import cupy as cp
            free, total = cp.cuda.runtime.memGetInfo()
            print(f"GPU: {free/1e9:.1f} GB free of {total/1e9:.1f} GB")
            return 0.8 * free / 1e9
    print(f"CPU run, budget {args.budget_gb} GB")
    return args.budget_gb


def _parse_value(s):
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return s


def parse_args(bench, argv=None):
    p = argparse.ArgumentParser(description=f"{bench.name}: {bench.title}",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--quick", action="store_true", help="few passes on a coarse grid: checks the script only")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "gpu"])
    p.add_argument("--gpu-id", default=None, help="CUDA device index (sets CUDA_VISIBLE_DEVICES)")
    p.add_argument("--budget-gb", type=float, default=16.0, help="memory budget on the CPU")
    p.add_argument("--preset", type=int, default=None, help="force grid preset i (0 = finest)")
    p.add_argument("--phi-max", type=float, default=0.02, help="peak nonlinear phase per step")
    p.add_argument("--gouy-step", type=float, default=0.05, help="max Gouy phase per step [rad]")
    p.add_argument("--record-every", type=int, default=bench.record_every)
    p.add_argument("--max-seconds", type=float, default=None, help="stop cleanly after this wall time")
    p.add_argument("--no-resume", action="store_true", help="ignore an existing checkpoint")
    p.add_argument("--save-field", action="store_true", help="also store the final field in run.npz")
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                   help="change a parameter, e.g. --set pressure_bar=2.2 (repeatable)")
    p.add_argument("--label", default=None, help="prefix of the run directory name")
    p.add_argument("--out", default=os.path.join(HERE, "results"), help="results root")
    a = p.parse_args(argv)
    ov = {}
    for s in a.overrides:
        if "=" not in s:
            p.error(f"--set expects KEY=VALUE, got {s!r}")
        k, v = s.split("=", 1)
        ov[k.strip()] = _parse_value(v.strip())
    a.overrides = ov
    return a


# ====================================================================== printing
def format_value(v, unit=""):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "nan"
    a = abs(v)
    if a == 0:
        return "0"
    if a < 1e-3 or a >= 1e6:
        return f"{v:.3e}"
    if a < 1:
        return f"{v:.4f}"
    return f"{v:.4g}"


def print_table(result, only_ref=False):
    print(f"\n{result['benchmark']} | {result['run']['id']} | grid {result['run']['grid_shape']}")
    print(f"{'metric':22s}{'simulation':>14s} {'unit':7s}{'paper':>18s}{'dev':>8s}  status  description")
    print("-" * 110)
    for k, m in result["metrics"].items():
        if only_ref and m["ref"] is None:
            continue
        ref = Ref(**m["ref"]).text() if m["ref"] else "-"
        dev = f"{m['deviation']*100:+.0f}%" if m.get("deviation") is not None else ""
        st = {"ok": "ok", "off": "OFF", "info": "", "nan": "nan"}[m["status"]]
        print(f"{k:22s}{format_value(m['value'], m['unit']):>14s} {m['unit']:7s}{ref:>18s}{dev:>8s}  "
              f"{st:6s}  {m['label']}")
