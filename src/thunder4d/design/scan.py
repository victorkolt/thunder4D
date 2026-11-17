"""Design-space scan (#15): which constraint limits each region.

    m = design_map(problem, gas="Ar", roc_m=2.0, k=2)                    # pressure x round trips
    m = design_map(problem, gas="Ar", roc_m=2.0, k=2, quick=Fast1DQuick(), workers=8)
    m.plot();  m.limiting[i, j];  m.values["duration_fs"]

Every point is a Candidate evaluated like in the design loop (same constraints, same scores), with
either the analytic estimates of AnalyticQuick (instant, approximate duration) or fast1d.
For each point, `limiting` is the most violated hard limit, or "feasible" (then `tightest` is the
limit with the smallest margin).
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field, replace
from typing import Any, cast

import numpy as np

from .catalog.gases import get_gas
from .catalog.mirrors import get_mirror
from .designer import (
    Candidate,
    QuickResult,
    _evaluate_task,
    _install_catalogues,
    allowed_plate_thicknesses,
    catalogue_snapshot,
    herriott_k,
    linear_estimates,
)
from .specs import Layout

LAMBDA_EFF = 4 / (3 * np.sqrt(3))       # gaussian SPM: broadening F = sqrt(1 + LAMBDA_EFF phi^2)

# one colour per constraint, the same on every map
CONSTRAINT_COLORS = {
    "fluence": "#1f77b4", "b_per_pass": "#ff7f0e", "p_over_pcr": "#d62728", "ionization": "#9467bd",
    "bandwidth": "#e7ba52", "spot_separation": "#8c564b", "edge_clearance": "#c49c94", "separation": "#7f7f7f",
    "chamber_length": "#bcbd22", "b_total": "#17becf", "avg_power_mirror": "#e377c2", "pressure": "#393b79",
    "simulation": "#000000", "coating": "#636363", "diameter": "#636363",
    "plate_intensity": "#2ca02c", "self_focusing": "#98df8a", "plate_fluence": "#aec7e8",
    "b_per_plate": "#ffbb78", "plate_clearance": "#c7c7c7", "plate_layout": "#c7c7c7",
}


# ============================================================ tier 0: analytic quick solver
def marburger_distance(P_over_Pc, w, lam, n0):
    """Self-focusing distance [m] of a collimated gaussian of radius w (Marburger):
    0.367 k w^2 / sqrt((sqrt(P/Pc) - 0.852)^2 - 0.0219); inf below the threshold."""
    x = (np.sqrt(P_over_Pc) - 0.852) ** 2 - 0.0219
    if P_over_Pc <= 1 or x <= 0:
        return np.inf
    return 0.367 * (2 * np.pi * n0 / lam) * w ** 2 / np.sqrt(x)


def matched_walk(cell, P, lam):
    """Kerr-matched mode of a cell with plates / gas sections (moment method, as MPC.nonlinear_mode),
    walked through both pass directions: w on the mirrors, B per pass, and per plate and per gas
    the worst values over the two directions.

    Returns {"w_mirror", "w_centre", "B_pass": [odd, even], "plates": {Plate: {w_entry, w_exit, w_min,
    B, P_over_Pc}}, "gas": {name: {w_min, P_over_Pc}}}; raises ValueError without a matched mode."""
    S, Sp = cell._matched_moments(P, lam)
    ng = float(cell.ref_medium.n_lam(lam))
    kv = 2 * np.pi / lam
    xg, wg = np.polynomial.legendre.leggauss(24)
    out = {"w_mirror": float(np.sqrt(S)), "B_pass": [], "plates": {}, "gas": {}}
    for p in (1, 2):
        rec = []
        S1, Sp1 = cell._walk_moments(S, Sp, P, lam, p, record=rec)
        B = 0.0
        for seg, S0, Sp0, r in rec:
            du = seg["u1"] - seg["u0"]
            med = seg["medium"]
            Se, _, smin = cell._segment_moments(S0, Sp0, du, r, lam)
            K = (lam / (np.pi * ng)) ** 2 * (1 - r)
            Spp = 2 * (K + Sp0 ** 2 / 4) / S0
            uu = (xg + 1) / 2 * du
            Su = S0 + Sp0 * uu + Spp * uu ** 2 / 2
            scale = float(med.n_lam(lam)) / ng                   # physical length per reduced length
            Bseg = kv * med.n2 * np.sum(wg * 2 * P / (np.pi * Su)) * du / 2 * scale if med.n2 else 0.0
            B += Bseg
            if seg["plate"] is not None:
                q = seg["plate"]
                d = out["plates"].setdefault(q, {"w_entry": np.inf, "w_min": np.inf, "B": 0.0, "P_over_Pc": r,
                                                 "_B": {}})
                d["_B"][p] = d["_B"].get(p, 0.0) + Bseg           # a plate split at the centre: summed
                if seg["entry_face"] is not None:
                    d["w_entry"] = min(d["w_entry"], float(np.sqrt(S0)))
                if seg["exit_face"] is not None:
                    d["w_entry"] = min(d["w_entry"], float(np.sqrt(Se)))   # the face hit first on the other pass
                d["w_min"] = min(d["w_min"], float(np.sqrt(smin)))
            else:
                g = out["gas"].setdefault(med.name, {"w_min": np.inf, "P_over_Pc": r, "medium": med})
                g["w_min"] = min(g["w_min"], float(np.sqrt(smin)))
            if p == 1 and seg["u0"] <= cell.equivalent_separation(lam) / 2 + 1e-15 <= seg["u1"] + 2e-15 \
                    and "w_centre" not in out:
                uc = cell.equivalent_separation(lam) / 2 - seg["u0"]
                out["w_centre"] = float(np.sqrt(S0 + Sp0 * uc + Spp * uc ** 2 / 2))
        out["B_pass"].append(float(B))
        S, Sp = S1, Sp1 - 4 * S1 / cell.R
    for d in out["plates"].values():
        d["B"] = max(d.pop("_B").values())
    return out


@dataclass
class AnalyticQuick:
    """Closed-form estimates (no propagation): Kerr-matched mode, P/Pcr, B per pass and total,
    peak intensity, mirror fluence, throughput R^N, and an SPM duration estimate
    tau_in / sqrt(1 + 0.77 (B/2)^2) (fast1d's mean phase is B/2 on axis). Same interface as
    Fast1DQuick; durations are indicative only (no dispersion, no compressibility).

    Cells with plates / gas sections, plate_mode:
      "matched" : Kerr-matched mode of the loaded cell (moment method, ~50 ms); bulk / hybrid cells are
                  injected in their matched mode. B per pass: the larger of the two pass directions.
      "linear"  : thin plates and gas sections in the linear mode of the equivalent cell, no Kerr lens
                  (~0.1 ms): what MPCDesigner.screen() uses on its thousands of starting points.
    Per plate (its own material): peak intensity, B per crossing, entrance fluence and thickness /
    Marburger self-focusing distance; per gas: peak intensity and P/Pcr; throughput includes the plate
    faces."""
    matching: str = "linear"
    plate_mode: str = "matched"

    def __call__(self, problem, cand):
        las = problem.laser
        eta = getattr(problem.losses, "mode_matching_efficiency", 1.0)
        P = las.peak_power_W * eta
        if cand.segmented:
            return self._segmented(problem, cand, P, eta)
        cell = cand.to_cell(las.n2_polarization_factor, las.lambda0)
        pc = cell.medium.critical_power(las.lambda0) if cell.medium.n2 else np.inf
        if P >= pc:
            return QuickResult(duration_fs=np.nan, peak_power_W=0.0, throughput=0.0, p_over_pcr=P / pc,
                               extras={"solver_warnings": ["P >= Pcr: self-focusing"]})
        mm = cell.nonlinear_mode(P, las.lambda0)
        lin = cell.linear_mode(las.lambda0)
        w0 = mm["w0"] if self.matching == "nonlinear" else lin["w0"]
        wm = mm["w_mirror"]
        b_pass = mm["B_per_pass"]
        b_tot = b_pass * cand.n_passes
        broad = np.sqrt(1 + LAMBDA_EFF * (b_tot / 2) ** 2)
        tau = las.duration_fs / broad
        r = float(get_mirror(cand.coating).R(las.wavelength_nm))
        thr = eta * r ** cand.n_passes
        fluence = 2 * las.energy_J * eta / (np.pi * wm ** 2) * 1e-4
        return QuickResult(
            duration_fs=tau, peak_power_W=0.94 * las.energy_J * thr / (tau * 1e-15), throughput=thr,
            b_total_rad=b_tot, b_per_pass_rad=b_pass, p_over_pcr=P / pc if np.isfinite(pc) else None,
            peak_intensity_W_cm2=2 * P / (np.pi * w0 ** 2) * 1e-4,
            mirror_fluence_J_cm2=np.full(cand.n_passes, fluence),
            duration_per_bounce_fs=np.full(cand.n_passes, las.duration_fs),
            extras={"w_mirror_max_m": float(wm), "broadening": float(broad)})

    # -------------------------------------------------------- plates / gas sections
    def _segmented(self, problem, cand, P, eta):
        las = problem.laser
        lam = las.lambda0
        energy = las.energy_J * eta
        gas_ratio = max((P * 2 * np.pi * get_gas(g).n2_per_bar * p * las.n2_polarization_factor / lam ** 2
                         for g, p in cand.gases), default=0.0)
        if gas_ratio >= 1:
            return QuickResult(duration_fs=np.nan, peak_power_W=0.0, throughput=0.0, p_over_pcr=gas_ratio,
                               extras={"solver_warnings": ["P >= Pcr in the gas: self-focusing"]})
        if self.plate_mode == "linear":
            est = linear_estimates(problem, cand)
            w0, wm, _ = cand.linear_mode(lam)
            b_pass = sum(e["B"] for e in est["plates"]) + sum(g["B"] for g in est["gas"])
            b_tot = b_pass * cand.n_passes
            plates = [{"material": e["material"], "w_face": e["w"], "w_max": e["w"], "B": e["B"],
                       "P_over_Pc": e["P_over_Pc"], "n0": e["n0"], "t": e["thickness_m"]} for e in est["plates"]]
            gas = {}
            for g in est["gas"]:
                old = gas.get(g["gas"])
                if old is None or g["w_min"] < old["w_min"]:
                    gas[g["gas"]] = {"w_min": g["w_min"], "P_over_Pc": g["P_over_Pc"]}
            amb = cand.material()
            t_plates = float(np.prod([p.transmission(amb, lam) for p in cand.plates()]))
        else:
            cell = cand.to_cell(las.n2_polarization_factor, lam)
            try:
                mm = matched_walk(cell, P, lam)
            except ValueError as err:          # no matched mode (self-focusing in the plates)
                return QuickResult(duration_fs=np.nan, peak_power_W=0.0, throughput=0.0, p_over_pcr=gas_ratio or None,
                                   extras={"solver_failed": f"no Kerr-matched mode: {err}",
                                           "solver_warnings": [f"no Kerr-matched mode: {err}"]})
            wm, w0 = mm["w_mirror"], mm.get("w_centre", np.nan)
            b_pass = max(mm["B_pass"])
            b_tot = mm["B_pass"][0] * ((cand.n_passes + 1) // 2) + mm["B_pass"][1] * (cand.n_passes // 2)
            plates = [{"material": q.material.name, "w_face": d["w_entry"], "w_max": d["w_entry"], "B": d["B"],
                       "P_over_Pc": d["P_over_Pc"], "n0": float(q.material.n_lam(lam)), "t": q.t, "w_min": d["w_min"]}
                      for q, d in mm["plates"].items()]
            gas = {name: {"w_min": g["w_min"], "P_over_Pc": g["P_over_Pc"]} for name, g in mm["gas"].items()}
            t_plates = float(cell.plate_transmission(lam))
        broad = np.sqrt(1 + LAMBDA_EFF * (b_tot / 2) ** 2)
        tau = las.duration_fs / broad
        r = float(get_mirror(cand.coating).R(las.wavelength_nm))
        thr = eta * (r * t_plates) ** cand.n_passes
        per_plate = [{"material": d["material"],
                      "intensity_W_cm2": 2 * P / (np.pi * d.get("w_min", d["w_face"]) ** 2) * 1e-4,
                      "b_rad": d["B"],
                      "self_focusing": d["t"] / marburger_distance(d["P_over_Pc"], d["w_max"], lam, d["n0"]),
                      "fluence_J_cm2": 2 * energy / (np.pi * d["w_face"] ** 2) * 1e-4,
                      "p_over_pcr": d["P_over_Pc"]} for d in plates]
        sections = [{"gas": name, "intensity_W_cm2": 2 * P / (np.pi * g["w_min"] ** 2) * 1e-4,
                     "p_over_pcr": g["P_over_Pc"]} for name, g in gas.items() if not get_gas(name).is_vacuum]
        b_plates = sum(d["B"] for d in plates)
        extras = {"w_mirror_max_m": float(wm), "broadening": float(broad), "w_centre_m": float(w0),
                  "plates": per_plate, "gas_sections": sections,
                  "plate_b_fraction": float(min(b_plates / b_pass, 1.0)) if b_pass else 1.0}
        if per_plate:
            extras.update({k: max(p[key] for p in per_plate) for k, key in (
                ("plate_peak_intensity_W_cm2", "intensity_W_cm2"), ("b_per_plate_rad", "b_rad"),
                ("plate_self_focusing", "self_focusing"), ("plate_fluence_J_cm2", "fluence_J_cm2"),
                ("plate_p_over_pcr", "p_over_pcr"))})
        return QuickResult(
            duration_fs=tau, peak_power_W=0.94 * las.energy_J * thr / (tau * 1e-15), throughput=thr,
            b_total_rad=b_tot, b_per_pass_rad=b_pass, p_over_pcr=gas_ratio if sections else None,
            peak_intensity_W_cm2=max((x["intensity_W_cm2"] for x in sections), default=None),
            # fluence: the wings of the pulse stay near the LINEAR mode (smaller on the mirrors when the
            # plates defocus the peak): the smaller of the two spots, as the 4D runs show
            mirror_fluence_J_cm2=np.full(cand.n_passes, 2 * energy / (np.pi * min(wm, cand.linear_mode(lam)[1]) ** 2)
                                         * 1e-4),
            duration_per_bounce_fs=np.full(cand.n_passes, las.duration_fs),
            extras=extras)


# ============================================================ the map
CANDIDATE_AXES = ("pressure_bar", "round_trips", "n_passes", "roc_m", "k", "plate_thickness_mm",
                  "plate_position_rel")
LASER_AXES = ("energy_J", "duration_fs")


@dataclass
class DesignMap:
    x_name: str
    x: np.ndarray
    y_name: str
    y: np.ndarray
    limiting: np.ndarray                  # (ny, nx) str: most violated hard limit, or "feasible"
    tightest: np.ndarray                  # (ny, nx) str: smallest-margin limit (all points)
    margin: np.ndarray                    # (ny, nx) margin of `tightest` (negative = violated)
    values: dict                          # metric -> (ny, nx) array (duration_fs, throughput, ...)
    candidates: Any = field(repr=False, default=None)   # (ny, nx) array of Candidate or None
    problem: Any = field(repr=False, default=None)
    solver: str = "analytic"

    @property
    def feasible(self):
        return self.limiting == "feasible"

    def best(self):
        """Feasible point with the lowest score (None if nothing is feasible)."""
        sc = self.values["score"]
        if not np.any(np.isfinite(sc)):
            return None
        i, j = np.unravel_index(np.nanargmin(np.where(np.isfinite(sc), sc, np.nan)), sc.shape)
        return self.candidates[i, j]

    def summary(self):
        valid = self.limiting[self.limiting != ""]
        names, counts = np.unique(valid, return_counts=True)
        share = ", ".join(f"{n}: {c / valid.size:.0%}" for n, c in
                          sorted(zip(names, counts, strict=True), key=lambda x: -x[1]))
        b = self.best()
        out = [(f"{self.y_name} x {self.x_name} map ({self.solver}), {valid.size} valid points "
               f"({self.limiting.size - valid.size} not a re-entrant geometry): {share}")]
        if b is not None:
            out.append(f"best: {b.gas} {b.pressure_bar:.3f} bar, R = {b.roc_m} m, M = {b.round_trips}, k = {b.k}, "
                       f"{b.n_passes} passes -> {b.metrics.get('duration_fs', np.nan):.1f} fs, score {b.score:.3g}")
        return "\n".join(out)

    def plot(self, ax=None, contour="duration_fs", levels=None, mark=(), show_best=True):
        """Regions coloured by limiting constraint (feasible region in white), contours of a metric,
        best feasible point and `mark` (list of (x, y, label)) marked."""
        from ._plot import pyplot

        plt = pyplot()
        from matplotlib.colors import ListedColormap
        from matplotlib.patches import Patch

        if ax is None:
            _, ax = plt.subplots(figsize=(8, 5.5))
        names = sorted(n for n in set(self.limiting.ravel()) if n and n != "feasible")
        extra = iter(list(cast(Any, plt.get_cmap("tab20b")).colors))
        colors = [CONSTRAINT_COLORS.get(n) or next(extra) for n in names]
        cmap = ListedColormap(["white", "#dddddd", *(colors or ["white"])])
        code = np.zeros(self.limiting.shape)
        code[self.limiting == ""] = 1                       # not a valid geometry
        for i, n in enumerate(names):
            code[self.limiting == n] = i + 2
        xs, ys = self._scale(self.x_name, self.x), self._scale(self.y_name, self.y)
        ax.pcolormesh(xs, ys, code, cmap=cmap, vmin=-0.5, vmax=len(names) + 1.5, shading="nearest")
        feas = self.feasible.astype(float)
        if 0 < feas.sum() < feas.size:
            ax.contour(xs, ys, feas, levels=[0.5], colors="k", linewidths=1.5)
        v = self.values.get(contour)
        if v is not None and np.any(np.isfinite(v)):
            if levels is None and contour == "duration_fs" and getattr(self.problem.target, "target_duration_fs", None):
                t = self.problem.target.target_duration_fs
                levels = sorted({t * f for f in (0.5, 0.75, 1.0, 1.5, 2.0, 3.0)})
            cs = ax.contour(xs, ys, np.where(np.isfinite(v), v, np.nan), levels=levels, colors="0.25",
                            linewidths=0.8)
            ax.clabel(cs, fmt="%g", fontsize=8)
        if show_best and (b := self.best()) is not None:
            ax.plot(*self._point(b), "*", ms=14, mfc="gold", mec="k", label="best feasible")
        for x, y, lab in mark:
            ax.plot(self._scale(self.x_name, x), self._scale(self.y_name, y), "o", mfc="none", mec="k", ms=10)
            ax.annotate(lab, (self._scale(self.x_name, x), self._scale(self.y_name, y)), xytext=(6, 6),
                        textcoords="offset points", fontsize=9)
        handles = [Patch(facecolor="white", edgecolor="k", label="feasible")]
        handles += [Patch(facecolor=c, label=n) for n, c in zip(names, colors, strict=True)]
        ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=9, title="limited by")
        ax.set(xlabel=self._label(self.x_name), ylabel=self._label(self.y_name),
               title=f"{self.problem.laser.energy_J * 1e3:g} mJ, {self.problem.laser.duration_fs:g} fs "
                     f"({self.solver}; contours: {contour})")
        return ax

    # -------------------------------------------------------- helpers
    @staticmethod
    def _scale(name, v):
        v = np.asarray(v, float)
        return v * 1e3 if name in ("pressure_bar", "energy_J") else v

    @staticmethod
    def _label(name):
        return {"pressure_bar": "pressure (mbar)", "round_trips": "round trips M", "n_passes": "passes N",
                "roc_m": "RoC (m)", "k": "k", "energy_J": "energy (mJ)", "duration_fs": "duration (fs)",
                "plate_thickness_mm": "plate thickness (mm)",
                "plate_position_rel": "plate position (fraction of d/2)"}[name]

    def _point(self, c):
        def val(name):
            return getattr(self.problem.laser if name in LASER_AXES else c, name)
        return self._scale(self.x_name, val(self.x_name)), self._scale(self.y_name, val(self.y_name))


def design_map(problem, *, gas, roc_m, k=1, round_trips=None, branch="concentric", x=None, y=None, quick=None,
               coating=None, diameter_inch=None, pass_offset=0, workers=None, layout=None):
    """Scan two parameters, everything else fixed. Default axes: x = pressure (from the pressure
    constraint, 41 points), y = round trips (SearchSpace.round_trips).

    x, y   : (name, values) with name in pressure_bar, round_trips, n_passes, roc_m, k, energy_J,
             duration_fs. n_passes as an axis keeps the geometry (M, k) and must stay <= 2 M.
    k      : fixed index, or "smallest" (smallest valid k for each M). Points where (M, k) is not a
             valid Herriott geometry are left blank.
    round_trips : fixed number of round trips M; required unless round_trips is a scanned axis.
    quick  : None -> AnalyticQuick (instant); Fast1DQuick() for 1+1D accuracy (use workers).
    layout : bulk / hybrid cell (specs.Layout: plates, gas sections), e.g. Layout.identical(2, "FS", 1.0, 0.7).
             Axes plate_thickness_mm (every plate) and plate_position_rel (outermost plates at +-value,
             the others scaled) can then be scanned. A vacuum-filled cell needs a layout; its default
             x axis is the plate thickness over the stock list."""
    C, S = problem.constraints, problem.search
    res, (pmin, pmax) = C.pressure.resolution_bar, C.pressure.pressure_bar
    vac = get_gas(gas).is_vacuum
    if vac and not (layout and layout.plates):
        raise ValueError("design_map: a vacuum-filled cell needs layout=Layout(plates=...)")
    if x is None:
        x = (("plate_thickness_mm", np.asarray(allowed_plate_thicknesses(problem))) if vac
             else ("pressure_bar", np.linspace(max(pmin, res), pmax, 41)))
    y = y or ("round_trips", np.arange(S.round_trips[0], S.round_trips[1] + 1))
    for name, _ in (x, y):
        if name not in CANDIDATE_AXES + LASER_AXES:
            raise ValueError(f"unknown axis {name!r}: use one of {CANDIDATE_AXES + LASER_AXES}")
    quick = quick or AnalyticQuick()
    coat = coating or (C.mirrors.allowed_coatings or [Candidate.coating])[0]
    dia = diameter_inch or max(C.mirrors.allowed_diameters_inch)
    if "round_trips" not in (x[0], y[0]) and round_trips is None:
        raise ValueError("design_map: give round_trips=M (the number of round trips of the cell) when "
                         "round_trips is not one of the two scanned axes")
    base: dict[str, Any] = {"gas": gas, "pressure_bar": 0.0 if vac else 0.1, "roc_m": roc_m,
            "round_trips": round_trips or 10, "k": 1 if k == "smallest" else k, "branch": branch, "coating": coat,
            "diameter_inch": dia, "lambda_nm": problem.laser.wavelength_nm}
    if layout:
        base["layout"] = layout

    tasks, where = [], []
    for i, yv in enumerate(y[1]):
        for j, xv in enumerate(x[1]):
            kw, laser = dict(base), {}
            for name, v in ((x[0], xv), (y[0], yv)):
                (laser if name in LASER_AXES else kw)[name] = v
            kw["round_trips"] = int(kw["round_trips"])
            n_passes = kw.pop("n_passes", None)
            if k == "smallest":
                ks = herriott_k(kw["round_trips"], S.k_range)
                if not ks:
                    continue
                kw["k"] = ks[0]
            kw["k"] = int(kw["k"])
            if kw["k"] not in herriott_k(kw["round_trips"]):
                continue
            lay = kw.get("layout", Layout())
            if "plate_thickness_mm" in kw:
                lay = replace(lay, plates=tuple(replace(p, thickness_mm=float(kw["plate_thickness_mm"]))
                                                for p in lay.plates))
            if "plate_position_rel" in kw:
                outer = max((abs(p.position_rel) for p in lay.plates), default=0.0) or 1.0
                lay = replace(lay, plates=tuple(replace(p, position_rel=float(kw["plate_position_rel"]) * p.position_rel
                                                        / outer if p.position_rel else 0.0) for p in lay.plates))
            kw.pop("plate_thickness_mm", None)
            kw.pop("plate_position_rel", None)
            if lay:
                kw["layout"] = lay
            n = int(n_passes) if n_passes is not None else 2 * kw["round_trips"] + pass_offset
            if not 1 <= n <= 2 * kw["round_trips"]:
                continue
            cand = Candidate(**kw, n_passes=n)
            if not cand.plate_layout_ok():
                continue
            prob = problem
            if laser:
                prob = copy.deepcopy(problem)
                prob.laser = replace(prob.laser, **laser)
            tasks.append((prob, quick, cand))
            where.append((i, j))

    if not tasks:
        raise ValueError(f"design_map: no valid Herriott geometry (k = {k}, round trips "
                         f"{round_trips or list(y[1] if y[0] == 'round_trips' else x[1])}): k must be coprime "
                         f"with the number of round trips and smaller than half of it")
    if workers and workers > 1:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(workers, initializer=_install_catalogues,
                                 initargs=catalogue_snapshot()) as pool:
            done = list(pool.map(_evaluate_task, tasks, chunksize=max(1, len(tasks) // (8 * workers))))
    else:
        done = [_evaluate_task(t) for t in tasks]

    shape = (len(y[1]), len(x[1]))
    limiting = np.full(shape, "", dtype=object)
    tightest = np.full(shape, "", dtype=object)
    margin = np.full(shape, np.nan)
    cands = np.full(shape, None, dtype=object)
    keys = ("duration_fs", "throughput", "b_total_rad", "b_per_pass_rad", "p_over_pcr", "peak_intensity_W_cm2",
            "separation_m", "score", "plate_peak_intensity_W_cm2", "plate_self_focusing", "plate_b_fraction")
    values = {key: np.full(shape, np.nan) for key in keys}
    for (i, j), c in zip(where, done, strict=True):
        cands[i, j] = c
        hard = {n: s for n, s in c.slacks.items() if c.limits.get(n, (0, 0, 0, "hard"))[3] == "hard"}
        violated = {n: s for n, s in hard.items() if s < 0}
        physical = {n: v for n, v in violated.items() if n != "simulation"}
        # a physical limit when one is crossed; "simulation" only when the model broke down alone
        worst = physical or violated
        limiting[i, j] = min(worst, key=lambda n: worst[n]) if violated else "feasible"
        if c.slacks:
            name = min(c.slacks, key=lambda n: c.slacks[n])
            tightest[i, j], margin[i, j] = name, c.slacks[name]
        for key in keys:
            v = c.score if key == "score" else c.metrics.get(key, np.nan)
            values[key][i, j] = v if v is not None and np.isfinite(v) else np.nan
    solver = "analytic estimates" if isinstance(quick, AnalyticQuick) else f"fast1d ({quick.accuracy})"
    # drop the rows / columns without any valid geometry (e.g. even M for an even k)
    rows = np.any(limiting != "", axis=1)
    cols = np.any(limiting != "", axis=0)
    sel = np.ix_(rows, cols)
    return DesignMap(x[0], np.asarray(x[1])[cols], y[0], np.asarray(y[1])[rows], limiting[sel].astype(str),
                     tightest[sel].astype(str), margin[sel], {k_: v[sel] for k_, v in values.items()},
                     cands[sel], problem, solver)
