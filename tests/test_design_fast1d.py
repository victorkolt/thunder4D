import numpy as np
import pytest

from thunder4d.design import (
    Candidate,
    DesignProblem,
    Fast1DQuick,
    LaserSpec,
    Max,
    MPCDesigner,
    Off,
    Target,
    evaluate,
    fast1d,
    starting_points,
)
from thunder4d.design.catalog import Coating, register_mirror

# Small, fast case: 2 mJ, 190 fs in Ar, ~5x compression
CELL = {"roc": 1.0, "N": 10, "k": 3, "gas": "Ar", "pressure": 0.6, "energy": 2e-3, "fwhm": 190e-15}


def rel(a, b):
    return abs(a / b - 1)


def test_presets_fill_only_missing_numerics():
    inp = fast1d.MPCInput(accuracy="fast", dphi_max=0.01)
    assert inp.dphi_max == 0.01 and inp.precision == "single" and inp.q_substeps == 8
    assert fast1d.MPCInput().precision == "double"          # no accuracy -> "default"
    assert inp.with_accuracy("default").dphi_max == 0.05


def test_fast_agrees_with_fine():
    fine = fast1d.simulate(fast1d.MPCInput.herriott(accuracy="fine", **CELL))
    fast = fast1d.simulate(fast1d.MPCInput.herriott(accuracy="fast", **CELL))
    for attr in ("fwhm_compressed", "peak_power_compressed", "b_integral", "fluence_mirror_max",
                 "intensity_max"):
        assert rel(getattr(fast, attr), getattr(fine, attr)) < 0.02, attr
    assert fast.runtime < fine.runtime


def test_lossless_cell_conserves_energy_and_photons():
    inp = fast1d.MPCInput.herriott(accuracy="default", mirror_reflectivity=1.0, self_steepening=False, **CELL)
    out = fast1d.simulate(inp)
    assert out.transmission == pytest.approx(1.0, abs=1e-6)       # RK4IP: not exactly conservative
    assert out.conservation_drift < 1e-6                           # energy, without self-steepening


def test_mirror_response_matches_scalar_mirror():
    kw = {**CELL, "accuracy": "default"}
    a = fast1d.simulate(fast1d.MPCInput.herriott(mirror_reflectivity=0.99, mirror_gdd=-20e-30, **kw))
    w0 = 2 * np.pi * 299792458.0 / 1030e-9

    def response(omega):
        return np.sqrt(0.99) * np.exp(-0.5j * -20e-30 * (omega - w0) ** 2)

    b = fast1d.simulate(fast1d.MPCInput.herriott(mirror_response=response, **kw))
    assert rel(b.fwhm_compressed, a.fwhm_compressed) < 1e-6
    assert rel(b.energy_out, a.energy_out) < 1e-9


def test_collapse_is_reported_not_crashed():
    with pytest.raises(fast1d.BeamCollapse):
        fast1d.simulate(fast1d.MPCInput.herriott(accuracy="fast", mirror_gdd=-150e-30, **CELL))
    prob = problem()
    cand = Candidate("Ar", 2.5, 1.0, 10, 3)                # P/Pcr > 1 at injection
    evaluate(prob, cand, Fast1DQuick()(prob, cand).as_metrics())
    assert not np.isfinite(cand.score) and cand.slacks["p_over_pcr"] < 0


def test_validate_reports_each_check():
    rep = fast1d.validate(fast1d.MPCInput.herriott(accuracy="fast", **CELL))
    assert set(rep.variants) == {"steps", "grid", "precision"}
    assert rep.ok, rep.summary()
    assert "PASSED" in rep.summary()


# ------------------------------------------------------------ adapter
def problem():
    return DesignProblem(laser=LaserSpec(energy_J=2e-3, duration_fs=190.0, rep_rate_Hz=1e3))


def test_adapter_coating_sign_convention():
    """A coating with a flat GDD must act like fast1d's scalar mirror_gdd (same sign)."""
    lam = np.linspace(900.0, 1200.0, 601)
    register_mirror(Coating("flat_m20", reflectivity=0.99, gdd_fs2=(lam, np.full_like(lam, -20.0)),
                            lidt_J_cm2=1.0), overwrite=True)
    cand = Candidate("Ar", 0.6, 1.0, 10, 3, coating="flat_m20")
    q = Fast1DQuick(accuracy="default")
    out = q.simulate(problem(), cand)
    ref = fast1d.simulate(fast1d.MPCInput(energy=2e-3, fwhm=190e-15, gas="Ar", pressure=0.6, roc=1.0,
                                          length=cand.separation_m, n_passes=20, mirror_reflectivity=0.99,
                                          mirror_gdd=-20e-30, n2=out.inp.n2))
    assert rel(out.fwhm_compressed, ref.fwhm_compressed) < 1e-3
    assert rel(out.gdd_compression, ref.gdd_compression) < 3e-3   # tabulated vs analytic phase


def test_adapter_quick_result_feeds_evaluate():
    prob = problem()
    cand = Candidate("Ar", 0.6, 1.0, 10, 3)
    res = Fast1DQuick()(prob, cand)
    assert len(res.mirror_fluence_J_cm2) == 19            # mirror-to-mirror transits: N - 1 bounces
    assert len(res.duration_per_bounce_fs) == 19
    assert 0 < res.main_peak_fraction <= 1
    evaluate(prob, cand, res.as_metrics())
    assert "fluence" in cand.limits and np.isfinite(cand.objective)


def test_adapter_prechirp_lengthens_the_input():
    prob = problem()
    cand = Candidate("Ar", 0.6, 1.0, 10, 3, prechirp_gdd_fs2=50000.0)
    inp = Fast1DQuick().build_input(prob, cand)
    t, a = inp.field
    assert fast1d.fwhm(t, np.abs(a) ** 2) > 1.5 * 190e-15


def test_parallel_refine_matches_serial():
    prob = problem()
    prob.refinement.n_candidates = 1
    prob.refinement.neighbour_n_passes = 0
    prob.refinement.max_evaluations = 8
    starts = [Candidate("Ar", 0.6, 1.0, 10, 3), Candidate("Kr", 0.3, 1.0, 10, 3)]
    serial = MPCDesigner(prob, quick=Fast1DQuick(), verbose=False).refine(starts)
    parallel = MPCDesigner(prob, quick=Fast1DQuick(), verbose=False, workers=2).refine(starts)
    assert serial.best.pressure_bar == pytest.approx(parallel.best.pressure_bar)
    assert serial.best.score == pytest.approx(parallel.best.score)


def test_workers_see_custom_coatings():
    register_mirror(Coating("custom_hr", reflectivity=0.998, lidt_J_cm2=1.0, band_nm=(900.0, 1200.0)),
                    overwrite=True)
    prob = problem()
    prob.refinement.n_candidates = 1
    prob.refinement.neighbour_n_passes = 0
    prob.refinement.max_evaluations = 4
    res = MPCDesigner(prob, quick=Fast1DQuick(), verbose=False, workers=2).refine(
        [Candidate("Ar", 0.25, 1.0, 10, 3, coating="custom_hr")])
    assert res.best.coating == "custom_hr" and np.isfinite(res.best.score)


def test_finds_the_reference_cell():
    """18 mJ / 500 fs -> 25 fs: with limits calibrated on the working cell (Ar, R = 2 m,
    15 round trips, k = 2, ~170 mbar), the helper must find it."""
    prob = DesignProblem(laser=LaserSpec(energy_J=18e-3, duration_fs=500.0, rep_rate_Hz=1e3),
                         target=Target(objective="target_duration", target_duration_fs=25.0))
    C = prob.constraints
    C.space.max_separation_m.value = 5.0
    C.mirrors.allowed_coatings = ["generic_dielectric"]
    C.kerr.b_per_pass_rad.value = 3.0
    C.mirrors.fluence.value = 0.7
    C.ionization.peak_intensity_W_cm2 = Max(3.5e13)
    prob.target.min_main_peak_fraction = Off()
    prob.search.roc_m = (2.0,)
    prob.search.round_trips = (13, 17)
    prob.search.branches = ("concentric",)
    prob.search.pass_offsets = (0,)
    prob.refinement.n_candidates = 1
    prob.refinement.neighbour_n_passes = 0
    prob.refinement.max_evaluations = 20
    starts = starting_points(prob, gases=("Ar",), verbose=False)
    res = MPCDesigner(prob, quick=Fast1DQuick(), verbose=False).refine(starts)
    best = res.best
    assert (best.roc_m, best.round_trips, best.k) == (2.0, 15, 2)
    assert 0.14 <= best.pressure_bar <= 0.20
    assert best.metrics["duration_fs"] == pytest.approx(25.0, rel=0.05)
