import numpy as np
import pytest

import thunder4d as th
from thunder4d.design import (
    AnalyticQuick,
    Candidate,
    DesignProblem,
    Fast1DQuick,
    LaserSpec,
    MPCDesigner,
    Target,
    design_map,
    evaluate,
    fast1d,
    to_input,
    validate_4d,
)


def problem(energy=2e-3, tau=190.0, target=40.0):
    return DesignProblem(laser=LaserSpec(energy_J=energy, duration_fs=tau, rep_rate_Hz=1e3),
                         target=Target(objective="target_duration", target_duration_fs=target))


def evaluated(prob, cand, quick=None):
    return evaluate(prob, cand, (quick or Fast1DQuick())(prob, cand).as_metrics())


# ------------------------------------------------------------ export (#16)
def test_to_input_is_a_valid_self_contained_file(tmp_path):
    prob = problem()
    c = evaluated(prob, Candidate("Ar", 0.25, 1.0, 10, 3))
    cfg = to_input(prob, c)
    loaded = th.load_input(cfg)
    assert loaded["cell"]["n_passes"] == 20
    assert loaded["cell"]["separation_m"] == pytest.approx(c.separation_m)
    assert loaded["cell"]["herriott"] == {"round_trips": 10, "k": 3, "branch": "concentric"}
    assert loaded["grid"]["lambda_min_nm"] < 1030 < loaded["grid"]["lambda_max_nm"]
    d = loaded["design"]
    assert d["predicted_fast1d"]["duration_fs"] == pytest.approx(c.metrics["duration_fs"])
    assert set(d["limits"]) == set(c.limits)
    th.save_input(cfg, tmp_path / "c.json")


def test_result_export(tmp_path):
    prob = problem()
    prob.refinement.n_candidates = 1
    prob.refinement.neighbour_n_passes = 0
    prob.refinement.max_evaluations = 6
    res = MPCDesigner(prob, quick=Fast1DQuick(), verbose=False).refine([Candidate("Ar", 0.25, 1.0, 10, 3)])
    path = res.export(tmp_path / "best.json")
    assert th.load_input(path)["medium"]["gas"] == "Ar"


def test_validate_4d_agrees_with_fast1d():
    prob = problem()
    c = evaluated(prob, Candidate("Ar", 0.25, 1.0, 10, 3))
    v = validate_4d(prob, c, verbose=False)
    rows = {q: d for q, _, _, d in v.rows}
    assert abs(rows["transmission"]) < 0.02
    assert abs(rows["B-integral [rad]"]) < 0.06
    assert abs(rows["compressed duration [fs]"]) < 0.06
    assert "4D validation" in v.summary()


# ------------------------------------------------------------ analytic tier and maps (#15)
def test_analytic_quick_trends():
    prob = problem()
    taus = [evaluated(prob, Candidate("Ar", p, 1.0, 10, 3), AnalyticQuick()).metrics["duration_fs"]
            for p in (0.1, 0.2, 0.4)]
    assert taus[0] > taus[1] > taus[2]


def test_map_drops_invalid_rows_and_names_the_limits():
    prob = problem()
    prob.search.round_trips = (5, 12)
    m = design_map(prob, gas="Ar", roc_m=1.0, k=2, x=("pressure_bar", np.linspace(0.05, 2.0, 12)))
    assert set(m.y) == {5, 7, 9, 11}                        # k = 2: only odd M are re-entrant
    assert m.limiting.shape == (4, 12)
    assert "feasible" in m.limiting and set(m.limiting.ravel()) - {"feasible"}
    assert np.all(m.margin[m.feasible] >= 0)
    assert m.best().score == pytest.approx(np.nanmin(m.values["score"]))
    assert "map" in m.summary()


def test_map_needs_round_trips_unless_scanned():
    prob = problem()
    kw = {"gas": "Ar", "roc_m": 1.0, "k": 2, "x": ("energy_J", [1e-3, 2e-3]), "y": ("pressure_bar", [0.2, 0.4])}
    with pytest.raises(ValueError, match="round_trips"):
        design_map(prob, **kw)
    m = design_map(prob, round_trips=15, **kw)
    assert m.limiting.shape == (2, 2) and m.candidates[0, 0].round_trips == 15
    with pytest.raises(ValueError, match="valid Herriott geometry"):
        design_map(prob, round_trips=10, **kw)                  # k = 2 is not coprime with 10


def test_map_with_fast1d_and_laser_axis():
    prob = problem()
    m = design_map(prob, gas="Ar", roc_m=1.0, k=3, round_trips=10, quick=Fast1DQuick(),
                   x=("energy_J", [1e-3, 2e-3]), y=("pressure_bar", [0.2, 0.4]))
    assert m.values["duration_fs"].shape == (2, 2)
    assert m.values["duration_fs"][0, 1] < m.values["duration_fs"][0, 0]   # more energy, shorter


def test_screen_feeds_refine():
    prob = problem()
    prob.search.round_trips = (5, 12)
    prob.search.roc_m = (1.0,)
    prob.refinement.n_candidates = 1
    prob.refinement.neighbour_n_passes = 0
    prob.refinement.max_evaluations = 6
    d = MPCDesigner(prob, quick=Fast1DQuick(), verbose=False)
    kept = d.screen(keep=4, gases=("Ar",))
    assert len(kept) <= 4 and all(not c.metrics for c in kept)
    assert np.isfinite(d.refine(kept).best.score)


# ------------------------------------------------------------ fast1d guards
def test_runaway_stops_quickly_and_is_rejected():
    prob = DesignProblem(laser=LaserSpec(energy_J=18e-3, duration_fs=500.0),
                         target=Target(objective="target_duration", target_duration_fs=25.0))
    c = Candidate("Ar", 0.17, 2.0, 29, 2)                    # B ~ 200 rad: optical shock
    with pytest.raises(fast1d.SimulationStopped):
        Fast1DQuick().simulate(prob, c)
    evaluated(prob, c)
    assert not np.isfinite(c.score) and "simulation" in c.slacks


def test_plot_without_matplotlib_says_how_to_install_it(monkeypatch):
    import sys

    from thunder4d.design._plot import pyplot

    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", None)       # as if it was not installed
    with pytest.raises(ImportError, match=r"thunder4d\[examples\]"):
        pyplot()
