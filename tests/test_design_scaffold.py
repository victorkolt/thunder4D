import json

import numpy as np
import pytest

from thunder4d import MPC
from thunder4d.design import (
    Candidate,
    DesignProblem,
    LaserSpec,
    MPCDesigner,
    MPCDesigner as _D,
    QuickResult,
    Target,
    evaluate,
)
from thunder4d.design.catalog import Coating, get_gas, get_mirror
from thunder4d.design.designer import herriott_k, prefilter, spot_pattern, starting_points
from thunder4d.design.specs import Max, Min


# ------------------------------------------------------------ Limit
def test_limit_max_and_min():
    lim = Max(1.0, safety=0.5)
    assert lim.allowed() == 0.5
    assert lim.slack(0.25) == pytest.approx(0.5)
    assert lim.penalty(0.6) == np.inf
    soft = Min(0.8, mode="soft", weight=2.0)
    assert soft.penalty(0.9) == 0.0
    assert soft.penalty(0.4) == pytest.approx(2.0 * 0.5 ** 2)


def test_limit_takes_catalogue_default_when_value_is_none():
    lim = Max(None, safety=0.5)
    assert lim.slack(1.0) is None
    assert lim.slack(1.0, default=4.0) == pytest.approx(0.5)


# ------------------------------------------------------------ laser
def test_laser_preset_and_peak_power():
    las = LaserSpec.preset("mpc1_18mJ_350fs")
    assert las.energy_J == 18e-3
    assert las.peak_power_W == pytest.approx(0.9394 * 18e-3 / 350e-15)
    assert LaserSpec(polarization="circular").n2_polarization_factor == pytest.approx(2 / 3)


# ------------------------------------------------------------ catalogues
def test_gas_material_and_threshold():
    ar = get_gas("Ar")
    mat = ar.material(0.13)
    assert mat.n2 == pytest.approx(ar.n2_per_bar * 0.13)
    assert ar.ionization_threshold == pytest.approx(0.2 * 4e9 * 15.760 ** 4)
    assert get_gas("He").ionization_threshold > ar.ionization_threshold


def test_coating_lidt_scaling_and_tables():
    c = Coating("t", lidt_J_cm2=1.0, lidt_ref_fs=1000.0, lidt_exponent=0.5,
                reflectivity=(np.array([1000.0, 1100.0]), np.array([99.0, 99.8])),
                r_in_percent=True)
    assert c.lidt(250.0) == pytest.approx(0.5)
    assert c.R(1050.0) == pytest.approx(0.994)


def test_coating_phase_has_the_tabulated_gdd():
    lam = np.linspace(900, 1200, 301)
    c = Coating("t", gdd_fs2=(lam, np.full_like(lam, -100.0)))
    lam0 = 1030e-9
    w0 = 2 * np.pi * 299792458 / lam0 * 1e-15
    dw = 0.1
    lams = 2 * np.pi * 299792458 / (np.array([w0 - dw, w0, w0 + dw]) * 1e15)
    ph = c.phase(lams, lam0)
    assert (ph[0] - 2 * ph[1] + ph[2]) / dw ** 2 == pytest.approx(-100.0, rel=1e-3)


# ------------------------------------------------------------ candidates
def test_candidate_geometry_matches_mpc_herriott():
    cand = Candidate("Ar", 0.13, roc_m=2.0, round_trips=15, k=4, coating="dielectric_1030nm")
    cell = cand.to_cell()
    assert isinstance(cell, MPC)
    assert cell.d == pytest.approx(cand.separation_m)
    assert cell.diameter == pytest.approx(3 * 0.0254)


def test_neighbours_vary_passes_in_the_same_geometry():
    prob = DesignProblem()
    prob.refinement.neighbour_n_passes = 2
    c = Candidate("Ar", 0.1, 2.0, 15, 2, n_passes=29)
    nbs = list(_D(prob)._neighbours(c))
    assert sorted(n.n_passes for n in nbs) == [27, 28, 29, 30]        # never beyond 2 M = 30
    assert {n.separation_m for n in nbs} == {c.separation_m}


def test_evaluate_rejects_hard_violation_and_checks_fluence():
    prob = DesignProblem()
    c = Candidate("Ar", 0.5, 2.0, 15, 4)
    evaluate(prob, c, {"duration_fs": 50.0, "throughput": 0.9, "p_over_pcr": 0.9})
    assert not np.isfinite(c.score)
    c = Candidate("Ar", 0.5, 2.0, 15, 4)
    evaluate(prob, c, {"duration_fs": 50.0, "throughput": 0.9,
                       "mirror_fluence_J_cm2": np.array([0.1, 0.2])})
    assert c.slacks["fluence"] == pytest.approx((0.4 - 0.2) / 0.4)
    assert np.isfinite(c.score)


# ------------------------------------------------------------ refine with a toy 1+1D code
def toy_quick(problem, cand):
    """Duration minimal at 0.4 bar, throughput from the coating R^N."""
    dur = 40.0 + 200.0 * (cand.pressure_bar - 0.4) ** 2
    thr = float(get_mirror(cand.coating).R(1030.0)) ** cand.n_passes
    return QuickResult(duration_fs=dur, peak_power_W=thr / dur, throughput=thr,
                       p_over_pcr=0.2, mirror_fluence_J_cm2=np.full(cand.n_passes, 0.1))


def test_refine_finds_the_toy_optimum_and_saves(tmp_path):
    prob = DesignProblem(target=Target(objective="shortest"))
    prob.refinement.n_candidates = 2
    prob.refinement.neighbour_n_passes = 0
    cands = [Candidate("Ar", 0.2, 2.0, 15, 4), Candidate("He", 0.2, 2.0, 10, 3),
             Candidate("Ar", 0.25, 2.0, 15, 4)]
    for c in cands:
        evaluate(prob, c, toy_quick(prob, c).as_metrics())
    res = MPCDesigner(prob, quick=toy_quick, verbose=False).refine(cands)
    assert len(res.refined) == 2
    assert {c.gas for c in res.refined} == {"Ar", "He"}            # diversity
    assert res.best.pressure_bar == pytest.approx(0.4, abs=0.011)  # snapped to 0.01 bar
    res.save_json(tmp_path / "design.json")
    data = json.loads((tmp_path / "design.json").read_text())
    assert data["refined"][0]["tier"] == "refined"
    assert "laser" in data["problem"]


def test_herriott_k_respects_the_branch():
    for m in (3, 7, 10, 15):
        for k in herriott_k(m):
            for branch in ("concentric", "planar"):
                c = Candidate("Ar", 0.1, 2.0, m, k, branch)
                assert (c.separation_m > 2.0) == (branch == "concentric")
    assert herriott_k(15) == [1, 2, 4, 7]                 # the classic 15 round trips, k = 2
    assert herriott_k(10) == [1, 3]


def test_candidate_geometry_and_passes():
    c = Candidate("Ar", 0.17, 2.0, 15, 2)
    assert c.n_passes == 30
    assert c.separation_m == pytest.approx(2.0 * (1 + np.cos(np.pi * 2 / 15)))
    cell = Candidate("Ar", 0.17, 2.0, 15, 2, n_passes=29).to_cell()
    assert cell.d == pytest.approx(c.separation_m)


def test_starting_points_fit_the_space_and_pass_the_prefilter():
    prob = DesignProblem()
    prob.constraints.space.max_separation_m.value = 2.0
    prob.search.round_trips = (3, 10)
    starts = starting_points(prob, gases=("Ar",), verbose=False)
    assert starts and all(c.separation_m <= 2.0 for c in starts)
    assert {c.n_passes for c in starts} >= {5, 6, 19, 20}          # 2 M and 2 M - 1, at least 5
    assert all(not prefilter(prob, c) for c in starts)
    assert all(prob.constraints.pressure.pressure_bar[0] < c.pressure_bar <= 3.0 for c in starts)


def test_spot_pattern_follows_the_mirror_size():
    prob = DesignProblem()
    c = Candidate("Ar", 0.1, 2.0, 15, 2, diameter_inch=3.0)
    w = 1e-3
    auto = spot_pattern(prob, c, w)
    assert auto["pattern_radius_m"] == pytest.approx(0.5 * 3 * 0.0254 - 2.5 * w)
    assert auto["spot_separation_w"] == pytest.approx(2 * auto["pattern_radius_m"] * np.sin(np.pi / 15) / w)
    prob.constraints.spots.pattern_radius_m = 0.03
    fixed = spot_pattern(prob, c, w)
    assert fixed["edge_clearance_w"] == pytest.approx((0.0381 - 0.03) / w)
    small = spot_pattern(DesignProblem(), Candidate("Ar", 0.1, 2.0, 15, 2, diameter_inch=0.5), 5e-3)
    assert small["edge_clearance_w"] < 2.5                           # spot too large: flagged


def test_tolerances_are_reported():
    c = Candidate("Ar", 0.17, 2.0, 15, 1)                             # near-concentric: sensitive
    t = c.tolerances(DesignProblem())
    assert t["separation"][0] > Candidate("Ar", 0.17, 2.0, 15, 4).tolerances()["separation"][0]
