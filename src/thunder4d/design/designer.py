"""Design helper, part 2: the pipeline.

    1) screen()   tier 0, analytic: enumerate (gas, RoC, N, k, branch, coating, ...), reject what
                  violates a hard limit, rank the rest                     [estimators: to come]
    2) refine()   tier 1, your fast 1+1D code: the n best (diverse) candidates are re-simulated,
                  their continuous parameters tuned (pressure, ...) and their discrete ones
                  scanned in a small neighbourhood (N +- dn, ...)
    3) result     DesignResult: ranked, refined configurations, saved as JSON, and each one
                  convertible to a thunder4d MPC / Material for 4D verification (tier 2)

The 1+1D code is plugged in through the QuickSolver protocol: any callable
``quick(problem, candidate) -> QuickResult``. Constraints and the objective are evaluated by the
same function (``evaluate``) at every tier, so a candidate is judged identically whatever produced
its metrics.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from math import gcd
from typing import Protocol

import numpy as np
from scipy.optimize import minimize

from ..cell import MPC
from .catalog.gases import get_gas
from .catalog.mirrors import cavity_coatings, get_mirror
from .specs import DesignProblem, Tolerance

INCH = 0.0254
_UNITS = {"spot_separation": " [w]", "edge_clearance": " [w]", "fluence": " [J/cm2]",
          "ionization": " [W/cm2]", "separation": " [m]",
          "chamber_length": " [m]", "b_per_pass": " [rad]", "b_total": " [rad]", "pressure": " [bar]",
          "avg_power_mirror": " [W]"}
_ZERO_SCALE = {"prechirp_gdd_fs2": 1000.0, "compressor_gdd_fs2": 1000.0, "pressure_bar": 0.1}


def _jsonable(v):
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, dict):
        return {k: _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, float) and not np.isfinite(v):
        return None
    return v


# ============================================================ one configuration
def linear_mode(roc_m, d_m, lambda0_m=1030e-9):
    """(w0, w_mirror, zR) [m] of the empty two-mirror cell; NaN if unstable."""
    if not 0 < d_m < 2 * roc_m:
        return np.nan, np.nan, np.nan
    zr = 0.5 * np.sqrt(d_m * (2 * roc_m - d_m))
    w0 = np.sqrt(lambda0_m * zr / np.pi)
    return w0, w0 * np.sqrt(1 + (0.5 * d_m / zr) ** 2), zr


def spot_pattern(problem, cand, w_mirror):
    """Circular pattern of M spots per mirror: radius, spot separation and edge clearance (in w)."""
    S = problem.constraints.spots
    a = 0.5 * cand.diameter_inch * INCH                          # mirror radius
    clear = S.edge_clearance_w.allowed() if S.edge_clearance_w.mode != "off" else 0.0
    out = {}
    if S.pattern_radius_m is None:
        r = a - (clear or 0.0) * w_mirror                        # largest radius that keeps the clearance
        if r <= 0:
            out["edge_clearance_w"] = a / w_mirror               # spot too big for the mirror
            r = 0.0
    else:
        r = S.pattern_radius_m
        out["edge_clearance_w"] = (a - r) / w_mirror
    m = cand.round_trips
    out["pattern_radius_m"] = r
    out["spot_separation_w"] = (2 * r * np.sin(np.pi / m) if m > 1 else 2 * r) / w_mirror
    return out


@dataclass
class Candidate:
    """One configuration. Geometry: RoC, round trips M of one re-entrant cycle and k
    (d = R (1 +- cos(pi k / M))); n_passes = passes actually used (default: the full cycle 2 M)."""
    gas: str
    pressure_bar: float
    roc_m: float
    round_trips: int
    k: int
    branch: str = "concentric"
    n_passes: int | None = None
    diameter_inch: float = 3.0
    coating: str = "dielectric_1030nm"
    prechirp_gdd_fs2: float = 0.0
    tier: str = "screen"                 # "screen" | "refined"
    metrics: dict = field(default_factory=dict)
    slacks: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)
    score: float = np.inf
    violation: float = 0.0               # sum of squared negative slacks (0 = all limits met)
    objective: float = np.nan            # score = objective + penalty (lower is better)
    penalty: float = 0.0                 # soft-limit penalties (inf if a hard limit is violated)
    limits: dict = field(default_factory=dict)   # name -> (value, "<=" or ">=", limit, mode)

    def __post_init__(self):
        if self.n_passes is None:
            self.n_passes = 2 * self.round_trips

    @property
    def separation_m(self):
        c = np.cos(np.pi * self.k / self.round_trips)
        return self.roc_m * (1 + c) if self.branch == "concentric" else self.roc_m * (1 - c)

    def linear_mode(self, lambda0_m=1030e-9):
        """(w0, w_mirror, zR) of the empty cell [m]."""
        return linear_mode(self.roc_m, self.separation_m, lambda0_m)

    def material(self, polarization_factor=1.0):
        return get_gas(self.gas).material(self.pressure_bar, polarization_factor=polarization_factor)

    def to_cell(self, polarization_factor=1.0, lambda0_m=1030e-9, **kw):
        """thunder4d MPC for 4D verification (mirror R(lambda) and GDD curve from the catalogue)."""
        c = get_mirror(self.coating)
        tabulated = np.ndim(c.gdd_fs2) > 0
        kw.setdefault("mirror_reflectivity", c.reflectivity_si)
        kw.setdefault("mirror_diameter", self.diameter_inch * INCH)
        if tabulated:
            kw.setdefault("mirror_phase", lambda lam: c.phase(lam, lambda0_m))
        else:
            kw.setdefault("mirror_gdd_fs2", float(c.gdd_fs2))
        return MPC.herriott(self.roc_m, 2 * self.round_trips, self.k, self.material(polarization_factor),
                            n_passes=self.n_passes,
                            branch=self.branch, **kw)

    def report(self, problem=None):
        """Readable summary: result, score breakdown, and every checked limit (value vs limit)."""
        m = self.metrics
        lines = [(f"{self.gas} {self.pressure_bar:.3f} bar | R = {self.roc_m} m, M = {self.round_trips} "
                  f"round trips, k = {self.k}, {self.branch}, d = {self.separation_m:.3f} m, "
                  f"{self.n_passes} passes | {self.coating}, {self.diameter_inch:g}\"")]
        if "duration_fs" in m:
            out = f"  output : {m['duration_fs']:.1f} fs"
            if problem is not None:
                tau0 = problem.laser.duration_fs
                out += f" (from {tau0:.0f} fs, x{tau0 / m['duration_fs']:.1f})"
            if "throughput" in m:
                out += f", throughput {100 * m['throughput']:.1f} %"
            if "peak_power_W" in m:
                out += f", peak power {m['peak_power_W'] / 1e9:.3g} GW"
            lines.append(out)
        status = "REJECTED (hard limit violated)" if not np.isfinite(self.score) else f"{self.score:.3g}"
        lines.append(f"  score  : {status}  = objective {self.objective:.3g} + penalties {self.penalty:.3g}"
                     f"   (lower is better)")
        lines.append(f"  {'limit':<18}{'value':>12}{'':>4}{'limit':>12}{'margin':>9}  mode")
        for name, (val, op, lim, mode) in sorted(self.limits.items(), key=lambda kv: self.slacks.get(kv[0], 0)):
            unit = _UNITS.get(name, "")
            if isinstance(lim, tuple):
                lim_s = ", ".join(str(x) for x in lim)
            elif isinstance(lim, (int, float)):
                lim_s = f"{lim:.4g}"
            else:
                lim_s = str(lim)
            val_s = f"{val:.4g}" if isinstance(val, (int, float)) else str(val)
            flag = "" if self.slacks.get(name, 0) >= 0 else "  <-- violated"
            lines.append(f"  {name + unit:<18}{val_s:>12}{op:>4}{lim_s:>12}{100 * self.slacks.get(name, 0):>8.0f}%"
                         f"  {mode}{flag}")
        lines += self._info_lines(problem)
        lines += [f"  warning: {w}" for w in self.warnings]
        return "\n".join(lines)

    def tolerances(self, problem=None, lambda0_m=None):
        """Relative change of w0 and w_mirror (linear mode) for the alignment errors of
        problem.tolerance: {"separation": (dw0, dwm), "roc": (dw0, dwm)}, worst sign, NaN = unstable."""
        tol = problem.tolerance if problem is not None else Tolerance()
        lam = lambda0_m or (problem.laser.lambda0 if problem is not None else 1030e-9)
        d, r = self.separation_m, self.roc_m
        w0, wm, _ = linear_mode(r, d, lam)

        def worst(cases):
            dw0 = max(abs(linear_mode(rr, dd, lam)[0] / w0 - 1) for rr, dd in cases)
            dwm = max(abs(linear_mode(rr, dd, lam)[1] / wm - 1) for rr, dd in cases)
            return dw0, dwm
        e, t = tol.separation_error_mm * 1e-3, tol.roc_tolerance_rel
        return {"separation": worst([(r, d - e), (r, d + e)]),
                "roc": worst([(r * (1 - t), d), (r * (1 + t), d)])}

    def _info_lines(self, problem):
        m, lines = self.metrics, []
        if "pattern_radius_m" in m:
            lines.append(f"  spots  : {self.round_trips} per mirror on r = {1e3 * m['pattern_radius_m']:.1f} mm, "
                         f"{m.get('spot_separation_w', np.nan):.1f} w apart")
        tol = self.tolerances(problem)
        t = problem.tolerance if problem is not None else Tolerance()

        def fmt(x):
            return "unstable" if not np.isfinite(x) else f"{100 * x:.1f} %"
        lines.append(f"  align  : d +- {t.separation_error_mm:g} mm -> w0 {fmt(tol['separation'][0])}, "
                     f"w_mirror {fmt(tol['separation'][1])} | RoC +- {100 * t.roc_tolerance_rel:g} % -> "
                     f"w0 {fmt(tol['roc'][0])}, w_mirror {fmt(tol['roc'][1])}")
        return lines

    def to_dict(self):
        d = asdict(self)
        d["separation_m"] = self.separation_m
        return _jsonable(d)


# ============================================================ the 1+1D hook
@dataclass
class QuickResult:
    """What the fast 1+1D code must return (missing fields = constraint not evaluated).

    Per-bounce arrays (length n_passes) let the fluence limit use the LIDT at the local duration.
    """
    duration_fs: float                              # compressed output, FWHM
    peak_power_W: float                             # compressed output
    throughput: float                               # energy out / energy in (after compression)
    main_peak_fraction: float | None = None         # energy in the main peak / total
    compressor_gdd_fs2: float | None = None         # optimal compression found by the code
    b_total_rad: float | None = None
    b_per_pass_rad: float | None = None             # max over passes
    p_over_pcr: float | None = None                 # max over passes
    peak_intensity_W_cm2: float | None = None       # max at the waist
    mirror_fluence_J_cm2: np.ndarray | None = None  # peak fluence per bounce
    duration_per_bounce_fs: np.ndarray | None = None
    out_of_band_fraction: float | None = None       # spectrum outside the coating band
    spectrum_nm: tuple[np.ndarray, np.ndarray] | None = None
    extras: dict = field(default_factory=dict)

    def as_metrics(self):
        return {k: v for k, v in asdict(self).items() if v is not None and k != "extras"} | self.extras


class QuickSolver(Protocol):
    def __call__(self, problem: DesignProblem, candidate: Candidate) -> QuickResult: ...


def _names(x):
    """Accept a single name or any sequence of names."""
    return (x,) if isinstance(x, str) else tuple(x)


def allowed_coatings(problem):
    """Cavity coatings the problem allows: SearchSpace.coatings, else MirrorConstraints.allowed_coatings,
    else every cavity coating of the catalogue."""
    names = problem.search.coatings or problem.constraints.mirrors.allowed_coatings
    if names is None:
        return tuple(c.name for c in cavity_coatings())
    names = _names(names)
    for n in names:
        get_mirror(n)            # KeyError with the list of available coatings if misspelled
    return names


def allowed_diameters(problem):
    return _names(problem.search.diameters_inch or problem.constraints.mirrors.allowed_diameters_inch)


# ============================================================ constraints + objective
def evaluate(problem: DesignProblem, cand: Candidate, metrics: dict):
    """Fill cand.metrics / slacks / warnings / score from a metrics dict (any tier)."""
    C, T, L = problem.constraints, problem.target, problem.laser
    gas, coat = get_gas(cand.gas), get_mirror(cand.coating)
    m = dict(metrics)
    m.setdefault("separation_m", cand.separation_m)
    m.setdefault("chamber_length_m", cand.separation_m + C.space.chamber_margin_m)
    m.setdefault("pressure_bar", cand.pressure_bar)
    w_m = m.get("w_mirror_max_m") or cand.linear_mode(L.lambda0)[1]
    if np.isfinite(w_m):
        for key, val in spot_pattern(problem, cand, w_m).items():
            m.setdefault(key, val)

    checks = {   # name: (Limit, metric key, catalogue default for value=None)
        "separation": (C.space.max_separation_m, "separation_m", None),
        "chamber_length": (C.space.max_chamber_length_m, "chamber_length_m",
                           C.space.fixed_chamber_length_m),
        "p_over_pcr": (C.kerr.p_over_pcr, "p_over_pcr", None),
        "b_per_pass": (C.kerr.b_per_pass_rad, "b_per_pass_rad", None),
        "b_total": (C.kerr.b_total_rad, "b_total_rad", None),
        "ionization": (C.ionization.peak_intensity_W_cm2, "peak_intensity_W_cm2",
                       gas.ionization_threshold),
        "bandwidth": (C.mirrors.bandwidth, "out_of_band_fraction", None),
        "avg_power_mirror": (C.mirrors.max_average_power_W, "average_power_mirror_W", None),
        "spot_separation": (C.spots.min_spot_separation_w, "spot_separation_w", None),
        "edge_clearance": (C.spots.edge_clearance_w, "edge_clearance_w", None),
        "throughput": (T.min_throughput, "throughput", None),
        "main_peak": (T.min_main_peak_fraction, "main_peak_fraction", None),
    }
    slacks, penalty, warnings, limits = {}, 0.0, [], {}
    for name, (lim, key, default) in checks.items():
        if key not in m or lim.mode == "off":
            continue
        s = lim.slack(m[key], default)
        if s is None:
            continue
        slacks[name] = float(s)
        penalty += lim.penalty(m[key], default)
        limits[name] = (float(m[key]), "<=" if lim.bound == "max" else ">=", lim.allowed(default), lim.mode)

    # fluence: LIDT at the local duration, worst bounce
    fl, tau = m.get("mirror_fluence_J_cm2"), m.get("duration_per_bounce_fs")
    if fl is not None and C.mirrors.fluence.mode != "off":
        fl = np.atleast_1d(fl)
        tau = np.atleast_1d(tau) if tau is not None else np.full_like(fl, L.duration_fs)
        worst, lim = np.inf, C.mirrors.fluence
        for f, t in zip(fl, tau):
            s = lim.slack(f, coat.lidt(t))
            if s is not None and s < worst:
                worst, pen, at = s, lim.penalty(f, coat.lidt(t)), (float(f), lim.allowed(coat.lidt(t)))
        if np.isfinite(worst):
            slacks["fluence"] = float(worst)
            penalty += pen
            limits["fluence"] = (at[0], "<=", at[1], lim.mode)
        elif C.mirrors.fluence.value is None:
            warnings.append(f"no LIDT for coating {coat.name!r}: fluence not checked")

    # hardware actually allowed (always hard): coating and mirror diameter
    for name, val, allowed in (("coating", cand.coating, allowed_coatings(problem)),
                               ("diameter", cand.diameter_inch, allowed_diameters(problem))):
        if val not in allowed:
            penalty = np.inf
            slacks[name] = -1.0
            limits[name] = (val, "in", allowed, "hard")

    # pressure window (always hard)
    pmin, pmax = C.pressure.pressure_bar
    # the quick solver stopped (shock, runaway): a hard violation of its own
    if m.get("solver_failed"):
        penalty = np.inf
        slacks["simulation"] = -1.0
        limits["simulation"] = ("stopped", "", "model", "hard")

    # always reported, so that a design stuck at the pressure ceiling shows it
    if not pmin <= cand.pressure_bar <= pmax:
        penalty = np.inf
        slacks["pressure"] = -1.0
    else:
        slacks["pressure"] = (pmax - cand.pressure_bar) / pmax if pmax else 0.0
    limits["pressure"] = (cand.pressure_bar, "<=", pmax, "hard")
    if gas.raman_active:
        warnings.append(f"{gas.name}: rotational Raman not modelled (planned)")
    warnings += list(m.get("solver_warnings", []))

    # objective (lower is better)
    obj = _objective(problem, m)
    cand.metrics, cand.slacks, cand.warnings, cand.limits = m, slacks, warnings, limits
    cand.violation = float(sum(s ** 2 for s in slacks.values() if s < 0))
    cand.objective, cand.penalty = float(obj), float(penalty)
    cand.score = obj + penalty if np.isfinite(penalty) and np.isfinite(obj) else np.inf
    return cand


def _objective(problem, m):
    T, L = problem.target, problem.laser
    dur, thr = m.get("duration_fs"), m.get("throughput", 1.0)
    if dur is None:
        return 0.0
    if not np.isfinite(dur):          # the quick solver failed (e.g. beam collapse): never ranked
        return np.inf
    if T.objective == "shortest":
        return dur / L.duration_fs
    if T.objective == "max_peak_power":
        return -m.get("peak_power_W", thr * L.peak_power_W * L.duration_fs / dur) / L.peak_power_W
    # target objectives: candidates that all hit the target are separated by their losses
    if T.objective == "target_duration":
        return ((dur - T.target_duration_fs) / T.target_duration_fs) ** 2 + 1e-3 * (1 - thr)
    if T.objective == "target_compression":
        return ((L.duration_fs / dur - T.target_compression) / T.target_compression) ** 2 + 1e-3 * (1 - thr)
    # weighted
    w = T.weights
    smax = problem.constraints.space.max_separation_m.allowed() or 1.0
    pmax = problem.constraints.pressure.pressure_bar[1] or 1.0
    return (w.get("duration", 0) * dur / L.duration_fs
            - w.get("throughput", 0) * thr
            + w.get("length", 0) * m.get("separation_m", 0) / smax
            + w.get("pressure", 0) * m.get("pressure_bar", 0) / pmax
            + w.get("robustness", 0) * m.get("jitter_sensitivity", 0))


# ============================================================ starting points (until screen(), #15)
def herriott_k(round_trips, k_range=None):
    """Re-entrant indices k for M round trips: k coprime with M and k < M/2, so that the concentric
    branch has d > R and the planar one d < R (each geometry appears once)."""
    m = round_trips
    lo, hi = k_range or (1, m)
    return [k for k in range(max(1, lo), min(hi, m) + 1) if 2 * k < m and gcd(k, m) == 1]


def prefilter(problem, cand):
    """Hard limits that can be checked without any simulation (linear cell mode, input pulse).
    Returns the list of violated limits; conservative: never rejects a candidate that could pass."""
    las, C = problem.laser, problem.constraints
    out = []
    d = cand.separation_m
    smax = C.space.max_separation_m.allowed()
    if C.space.max_separation_m.mode == "hard" and smax is not None and d > smax:
        out.append("separation")
    w0, wm, _ = cand.linear_mode(las.lambda0)
    if not np.isfinite(w0):
        return [*out, "stability"]
    # The Kerr effect changes the spot sizes by up to (1 - P/Pcr_max)^(1/4) (smaller when matched,
    # oscillating when not): every check below uses the most favourable size, so that no candidate
    # that could pass is rejected.
    pmax = min(C.kerr.p_over_pcr.allowed() or 0.0, 0.99) if C.kerr.p_over_pcr.mode != "off" else 0.0
    shrink = (1 - pmax) ** 0.25
    fluence = 2 * las.energy_J / (np.pi * (wm / shrink) ** 2) * 1e-4
    lim = C.mirrors.fluence
    s = lim.slack(fluence, get_mirror(cand.coating).lidt(las.duration_fs))
    if lim.mode == "hard" and s is not None and s < 0:
        out.append("fluence")
    lim = C.ionization.peak_intensity_W_cm2
    intensity = 2 * las.peak_power_W / (np.pi * w0 ** 2) * 1e-4      # Kerr focusing only increases it
    s = lim.slack(intensity, get_gas(cand.gas).ionization_threshold)
    if lim.mode == "hard" and s is not None and s < 0:
        out.append("ionization")
    for name, key in (("spot_separation", "spot_separation_w"), ("edge_clearance", "edge_clearance_w")):
        lim = getattr(C.spots, "min_spot_separation_w" if name == "spot_separation" else "edge_clearance_w")
        val = spot_pattern(problem, cand, wm * shrink).get(key)
        s = lim.slack(val) if val is not None else None
        if lim.mode == "hard" and s is not None and s < 0:
            out.append(name)
    return out


def start_pressure(problem, cand, p_over_pcr=0.3):
    """Analytic first guess of the pressure for one geometry: the lowest of
        - the pressure giving P/Pcr = p_over_pcr,
        - the pressure giving 90 % of the B-per-pass limit,
        - for target_duration / target_compression: the pressure giving the B-integral that the
          gaussian SPM estimate needs for that compression (fast1d's SPM uses the mean phase, B/2).
    Linear cell mode; on-axis B per pass = 4 k0 n2 P atan(d / 2 zR) / lambda0."""
    las, C, T = problem.laser, problem.constraints, problem.target
    n2_bar = get_gas(cand.gas).n2_per_bar * las.n2_polarization_factor
    P, lam = las.peak_power_W, las.lambda0
    d, R = cand.separation_m, cand.roc_m
    zr = 0.5 * np.sqrt(d * (2 * R - d))
    b_pass_per_bar = 4 * (2 * np.pi / lam) * n2_bar * P * np.arctan(d / (2 * zr)) / lam
    p = [p_over_pcr * lam ** 2 / (2 * np.pi * n2_bar * P)]
    b_lim = C.kerr.b_per_pass_rad
    if b_lim.mode == "hard" and b_lim.allowed() is not None:
        p.append(0.9 * b_lim.allowed() / b_pass_per_bar)
    factor = None
    if T.objective == "target_duration" and T.target_duration_fs:
        factor = las.duration_fs / T.target_duration_fs
    elif T.objective == "target_compression" and T.target_compression:
        factor = T.target_compression
    if factor and factor > 1:
        b_needed = 2 * np.sqrt((factor ** 2 - 1) / (4 / (3 * np.sqrt(3))))
        p.append(b_needed / (cand.n_passes * b_pass_per_bar))
    res, (pmin, pmax) = C.pressure.resolution_bar, C.pressure.pressure_bar
    return float(np.clip(round(min(p) / res) * res, max(pmin, res), pmax))


def starting_points(problem, gases=None, round_trips=None, k="all", p_over_pcr=0.3, verbose=True):
    """Provisional screen() (until #15): a grid of geometries, each starting at an analytic guess of
    the pressure (start_pressure), without the ones that a hard limit already excludes analytically.

    gases       : None -> SearchSpace.gases or GasConstraints.allowed
    round_trips : None -> every M in SearchSpace.round_trips (inclusive range), or any list of M
    k           : "all" (every valid k) | "smallest" (one geometry per M: fewer, faster)
    Passes per geometry: 2 M + SearchSpace.pass_offsets, at least SearchSpace.min_passes.
    RoC, branches, k range, coatings and diameters come from the search space / constraints."""
    C, S = problem.constraints, problem.search
    gases = gases or S.gases or C.gas.allowed
    rocs = S.roc_m or C.mirrors.allowed_roc_m
    ms = range(S.round_trips[0], S.round_trips[1] + 1) if round_trips is None else round_trips
    starts, rejected = [], {}
    for g in gases:
        for roc in rocs:
            for branch in S.branches:
                for m in ms:
                    ks = herriott_k(m, S.k_range)
                    passes = sorted({2 * m + o for o in S.pass_offsets if S.min_passes <= 2 * m + o <= 2 * m})
                    for kk in (ks if k == "all" else ks[:1]):
                        for n in passes:
                            for coat in allowed_coatings(problem):
                                for dia in allowed_diameters(problem):
                                    c = Candidate(g, 1.0, roc, m, kk, branch, n_passes=n,
                                                  diameter_inch=dia, coating=coat)
                                    why = prefilter(problem, c)
                                    for w in why:
                                        rejected[w] = rejected.get(w, 0) + 1
                                    if not why:
                                        c.pressure_bar = start_pressure(problem, c, p_over_pcr)
                                        starts.append(c)
    if verbose:
        detail = ", ".join(f"{w}: {n}" for w, n in sorted(rejected.items(), key=lambda x: -x[1]))
        print(f"{len(starts)} starting points kept" + (f" (rejected analytically: {detail})" if detail else ""))
    return starts


# ============================================================ helpers
# families used for diversity, from the coarsest to the finest: first the best candidate of
# each gas, then of each (gas, RoC, branch) in ranking order, then the next best
DIVERSITY_LEVELS = (("gas",), ("gas", "roc_m", "branch"))


def select_diverse(cands, n, diversity=True):
    """n best candidates; with diversity, spread over gases, branches and RoCs before repeating."""
    ranked = sorted(cands, key=lambda c: (not np.isfinite(c.score), c.score, c.violation))
    if not diversity:
        return ranked[:n]
    out = []
    for level in DIVERSITY_LEVELS:
        seen = {tuple(getattr(c, a) for a in level) for c in out}
        for c in ranked:
            if len(out) == n:
                return out
            fam = tuple(getattr(c, a) for a in level)
            if fam not in seen and not any(c is o for o in out):
                out.append(c)
                seen.add(fam)
    for c in ranked:                     # then fill with the next best
        if len(out) == n:
            break
        if not any(c is o for o in out):
            out.append(c)
    return out


# ============================================================ results
@dataclass
class DesignResult:
    problem: DesignProblem
    screened: list = field(default_factory=list)
    refined: list = field(default_factory=list)
    traces: dict = field(default_factory=dict)

    @property
    def best(self):
        pool = self.refined or self.screened
        return min(pool, key=lambda c: c.score) if pool else None

    def table(self, which="refined"):
        rows = self.refined if which == "refined" else self.screened
        head = (f"{'#':>2} {'gas':>4} {'p[bar]':>7} {'R[m]':>5} {'M':>3} {'k':>3} {'N':>4} {'branch':>10} "
                f"{'d[m]':>6} {'tau[fs]':>8} {'T':>5} {'score':>8}")
        lines = [head]
        for i, c in enumerate(sorted(rows, key=lambda c: c.score)):
            m = c.metrics
            lines.append(f"{i:>2} {c.gas:>4} {c.pressure_bar:>7.3f} {c.roc_m:>5.2f} "
                         f"{c.round_trips:>3} {c.k:>3} {c.n_passes:>4} {c.branch:>10} {c.separation_m:>6.3f} "
                         f"{m.get('duration_fs', np.nan):>8.1f} {m.get('throughput', np.nan):>5.3f} "
                         f"{c.score:>8.3g}")
        return "\n".join(lines)

    def export(self, path, which=0, **kw):
        """Write a refined design (rank `which`, 0 = best) as a standard THUNDER4D input file."""
        from .export import export_input
        cand = sorted(self.refined or self.screened, key=lambda c: c.score)[which]
        return export_input(self.problem, cand, path, **kw)

    def save_json(self, path):
        data = {"problem": self.problem.to_dict(),
                "refined": [c.to_dict() for c in sorted(self.refined, key=lambda c: c.score)],
                "screened_top": [c.to_dict() for c in sorted(self.screened, key=lambda c: c.score)]}
        with open(path, "w") as f:
            json.dump(_jsonable(data), f, indent=2, default=str)


# ============================================================ the designer
def _install_catalogues(gases, mirrors):
    """Worker initializer: the catalogues of the parent process (custom gases / coatings included)."""
    from .catalog import gases as gas_module, mirrors as mirror_module
    gas_module.GASES.clear()
    gas_module.GASES.update(gases)
    mirror_module.MIRRORS.clear()
    mirror_module.MIRRORS.update(mirrors)


def _visit_key(c):
    """Identity of a configuration for the refinement: geometry, coating, mirror size and passes."""
    return (c.gas, c.roc_m, c.round_trips, c.k, c.branch, c.coating, c.diameter_inch, c.n_passes)


def _evaluate_task(args):
    problem, quick, cand = args
    return evaluate(problem, cand, quick(problem, cand).as_metrics())


def _refine_task(args):
    problem, quick, cand = args
    return MPCDesigner(problem, quick, verbose=False)._refine_one(cand)


class MPCDesigner:
    """designer = MPCDesigner(problem, quick=my_1p1d_adapter, workers=4)
    result = designer.run()          # screen -> top n -> refine -> DesignResult
    result = designer.refine(cands)  # or refine hand-picked candidates directly

    workers > 1 runs the quick solver in parallel processes. The quick solver must then be
    picklable (a module-level function or class instance, e.g. Fast1DQuick, not a lambda) and a
    script must put its work under ``if __name__ == "__main__":`` (Windows). Jupyter needs neither.
    The gas and mirror catalogues, custom entries included, are copied to the workers.
    """

    def __init__(self, problem: DesignProblem, quick: QuickSolver | None = None, verbose=True, workers=None):
        self.problem, self.quick, self.verbose, self.workers = problem, quick, verbose, workers or 1

    # -------------------------------------------------------- step 1
    def screen(self, keep=None, **starting_kw):
        """Tier 0: starting_points() evaluated with the analytic estimates (AnalyticQuick), the
        `keep` best (diverse) kept for refine() (default 10 x n_candidates); their analytic
        metrics are dropped so that refine() re-evaluates them with the quick solver."""
        from .scan import AnalyticQuick
        starts = starting_points(self.problem, verbose=self.verbose, **starting_kw)
        tier0 = AnalyticQuick()
        for c in starts:
            evaluate(self.problem, c, tier0(self.problem, c).as_metrics())
        keep = keep or 10 * self.problem.refinement.n_candidates
        kept = select_diverse(starts, keep, self.problem.refinement.diversity)
        if self.verbose:
            n_ok = sum(np.isfinite(c.score) for c in starts)
            print(f"screen: {len(starts)} starting points, {n_ok} feasible by the analytic estimates, "
                  f"{len(kept)} kept for the quick solver")
        return [replace(c, metrics={}, slacks={}, limits={}, warnings=[], score=np.inf) for c in kept]

    # -------------------------------------------------------- step 2
    def refine(self, candidates, max_attempts=None):
        """Refine candidates in diversity order until n_candidates feasible configurations are
        found, or max_attempts candidates were tried (default 3 * n_candidates)."""
        if self.quick is None:
            raise ValueError("refine() needs a QuickSolver (your 1+1D code adapter)")
        if self.workers > 1:
            from concurrent.futures import ProcessPoolExecutor

            from .catalog.gases import GASES
            from .catalog.mirrors import MIRRORS
            with ProcessPoolExecutor(self.workers, initializer=_install_catalogues,
                                     initargs=(dict(GASES), dict(MIRRORS))) as pool:
                return self._refine(candidates, max_attempts, pool.map)
        return self._refine(candidates, max_attempts, map)

    def _refine(self, candidates, max_attempts, pmap):
        R = self.problem.refinement
        max_attempts = max_attempts or 3 * R.n_candidates
        todo = [c for c in candidates if not c.metrics]
        if self.verbose and todo:
            print(f"evaluating {len(todo)} starting points ({self.workers} worker(s))...")
        done = iter(pmap(_evaluate_task, [(self.problem, self.quick, c) for c in todo]))
        candidates = [c if c.metrics else next(done) for c in candidates]
        order = select_diverse(candidates, len(candidates), R.diversity)
        result = DesignResult(self.problem, screened=list(candidates))
        tried, pos = 0, 0
        explored, found = set(), set()
        while len(result.refined) < R.n_candidates and tried < max_attempts and pos < len(order):
            need = R.n_candidates - len(result.refined)
            batch = []
            while pos < len(order) and len(batch) < max(need, self.workers) and tried + len(batch) < max_attempts:
                c = order[pos]
                pos += 1
                if _visit_key(c) in explored:    # inside the neighbourhood of a candidate already refined
                    continue
                batch.append(c)
                explored.update(_visit_key(nb) for nb in self._neighbours(c))
            if not batch:
                break
            outcomes = pmap(_refine_task, [(self.problem, self.quick, c) for c in batch])
            for c, (best, trace) in zip(batch, outcomes):
                tried += 1
                if len(result.refined) == R.n_candidates:
                    continue
                tag = f"[{len(result.refined) + 1}/{R.n_candidates}, try {tried}]"
                if best is None or not np.isfinite(best.score):
                    if self.verbose and best is not None:
                        print(f"{tag} {c.gas}, R = {c.roc_m} m, N = {c.n_passes}, k = {c.k}, {c.branch}, "
                              f"d = {c.separation_m:.3f} m: no feasible point, "
                              f"violated: {[k for k, s in best.slacks.items() if s < 0]}")
                    continue
                step = self.problem.constraints.pressure.resolution_bar
                design = (*_visit_key(best), round(best.pressure_bar / step))
                if design in found:              # two starting points converged to the same design
                    continue
                found.add(design)
                best.tier = "refined"
                result.refined.append(best)
                if R.keep_traces:
                    result.traces[len(result.refined) - 1] = trace
                if self.verbose:
                    print(f"{tag} {best.gas} {best.pressure_bar:.3f} bar, R = {best.roc_m} m, N = {best.n_passes}, "
                          f"k = {best.k}, {best.branch}, d = {best.separation_m:.3f} m, "
                          f"score start {c.score:.3g} -> refined {best.score:.3g}")
        if self.verbose and len(result.refined) < R.n_candidates:
            print(f"only {len(result.refined)} feasible configuration(s) after {tried} "
                  f"tries: relax the limits printed above, widen the starting points, or raise max_attempts")
        return result

    def _refine_one(self, c):
        """Neighbourhood scan + continuous tuning of one candidate -> (best, trace).
        The untuned neighbours compete too: tuning never returns something worse than its start."""
        best, trace = None, []
        for nb in self._neighbours(c):
            for cand in (self._evaluate(nb), self._tune(nb, trace)):
                if best is None or cand.score < best.score or (
                        not np.isfinite(best.score) and cand.violation < best.violation):
                    best = cand
        for t in trace:   # keep the traces light: no spectra
            t.metrics.pop("spectrum_nm", None)
        return best, trace

    def run(self, keep=None, **starting_kw):
        """screen() then refine(): starting points -> analytic ranking -> quick solver."""
        return self.refine(self.screen(keep, **starting_kw))

    # -------------------------------------------------------- internals
    def _evaluate(self, cand):
        return evaluate(self.problem, cand, self.quick(self.problem, cand).as_metrics())

    def _neighbours(self, c):
        """Same geometry with N +- neighbour_n_passes (within [min_passes, 2 M]), and the same (M, k)
        on the neighbouring RoCs of the allowed list."""
        R, S = self.problem.refinement, self.problem.search
        allowed_roc = sorted(self.problem.constraints.mirrors.allowed_roc_m)
        rocs = [c.roc_m]
        if R.neighbour_roc_steps and c.roc_m in allowed_roc:
            i = allowed_roc.index(c.roc_m)
            lo, hi = max(0, i - R.neighbour_roc_steps), i + R.neighbour_roc_steps + 1
            rocs = allowed_roc[lo:hi]
        for roc in rocs:
            for dn in range(-R.neighbour_n_passes, R.neighbour_n_passes + 1):
                n = c.n_passes + dn
                if S.min_passes <= n <= 2 * c.round_trips:
                    yield replace(c, roc_m=roc, n_passes=n, metrics={}, slacks={},
                                  limits={}, warnings=[], score=np.inf)

    def _tune(self, cand, trace):
        R, P = self.problem.refinement, self.problem.constraints.pressure
        names = list(R.tune)
        x0 = np.array([getattr(cand, n) or 0.0 for n in names], dtype=float)
        # optimise in units of the starting value (or of a typical step when it starts at 0)
        scale = np.array([abs(v) if v else _ZERO_SCALE.get(n, 1.0) for n, v in zip(names, x0)])

        def apply(x):
            kw = dict(zip(names, x * scale))
            if "pressure_bar" in kw:
                lo = max(P.pressure_bar[0], P.resolution_bar)   # never an empty cell (n2 = 0)
                kw["pressure_bar"] = float(np.clip(kw["pressure_bar"], lo, P.pressure_bar[1]))
            return replace(cand, **kw, metrics={}, slacks={}, warnings=[], score=np.inf)

        def f(x):
            c = self._evaluate(apply(x))
            trace.append(c)
            return c.score if np.isfinite(c.score) else 1e3 * (1.0 + c.violation)

        if not names:
            return self._evaluate(cand)
        if R.method == "grid":
            raise NotImplementedError("grid refinement: to come")
        if R.method == "nelder-mead":
            method, opts = "Nelder-Mead", {"maxfev": R.max_evaluations, "xatol": R.xtol_rel,
                                           "fatol": 1e-6}
        else:
            method, opts = "Powell", {"maxfev": R.max_evaluations, "xtol": R.xtol_rel}
        res = minimize(f, x0 / scale, method=method, options=opts)
        best = self._evaluate(apply(res.x))
        if "pressure_bar" in names:   # snap to the regulator resolution: best of the two neighbours
            step, p = P.resolution_bar, best.pressure_bar
            snapped = [self._evaluate(replace(best, pressure_bar=round(v, 10), metrics={}, slacks={},
                                              limits={}, warnings=[], score=np.inf))
                       for v in {np.floor(p / step) * step, np.ceil(p / step) * step} if v > 0]
            best = min(snapped, key=lambda x: (x.score, x.violation)) if snapped else best
        return best
