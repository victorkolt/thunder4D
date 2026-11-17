"""Design helper for bulk and hybrid MPCs: layouts (plates with individual materials, gas sections),
geometry, analytic tier, fast1d with plates, constraints, search, export to the input file.
Reference cell: Westerberg et al., APL Photonics 10, 096103 (2025): R = 200 mm, 7 round trips, two
1 mm AR-coated FS plates at +-140 mm (windows of an evacuated section in air), 250 uJ, 180 fs."""
import numpy as np
import pytest

import thunder4d as th
from thunder4d.design import (
    AnalyticQuick,
    Candidate,
    DesignProblem,
    Fast1DQuick,
    LaserSpec,
    Layout,
    MPCDesigner,
    PlateSpec,
    RegionSpec,
    SearchSpace,
    Target,
    design_map,
    evaluate,
    fast1d,
    to_input,
)
from thunder4d.design.catalog import PLATES, get_gas, get_plate
from thunder4d.design.designer import plate_options, prefilter, starting_points
from thunder4d.design.specs import Max
from thunder4d.inputfile import build_simulation, input_problems, load_input, make_cell

LAM0 = 1030e-9
FS2 = Layout.identical(2, "FS", 1.0, 0.733)
WINDOWS = Layout((PlateSpec("FS", 1.0, -0.733), PlateSpec("FS", 1.0, 0.733)), (RegionSpec(-0.733, 0.733, "vacuum"),))


def westerberg_problem(target=40.0):
    prob = DesignProblem(laser=LaserSpec(energy_J=250e-6, duration_fs=180.0),
                         target=Target(objective="target_duration", target_duration_fs=target))
    prob.constraints.mirrors.allowed_roc_m = (0.2, 0.3)
    return prob


def westerberg(layout=FS2, gas="vacuum", pressure=0.0, **kw):
    return Candidate(gas, pressure, 0.2, 7, 1, "concentric", n_passes=14, layout=layout, **kw)


def evaluated(prob, cand, quick=None):
    return evaluate(prob, cand, (quick or AnalyticQuick())(prob, cand).as_metrics())


# ------------------------------------------------------------ catalogue, layout and geometry
def test_vacuum_and_plate_catalogue():
    vac = get_gas("vacuum")
    assert vac.is_vacuum and vac.ionization_threshold is None and vac.material(1.0).n2 == 0
    assert get_plate("FS").solid().n2 == pytest.approx(2.2e-20)
    assert get_plate("FS").lidt(100.0) is None
    assert set(PLATES) <= set(th.materials.SOLIDS)
    with pytest.raises(KeyError, match="available"):
        get_plate("unobtainium")


def test_layout_helpers():
    assert [p.position_rel for p in Layout.identical(3, "FS", 1.0, 0.6).plates] == pytest.approx([-0.6, 0, 0.6])
    assert Layout.identical(1, "FS", 1.0, 0.5).plates[0].position_rel == 0.5
    assert not Layout() and FS2 and Layout(regions=(RegionSpec(-0.5, 0.5, "vacuum"),))
    assert FS2.label == "2xFS 1mm" and WINDOWS.label == "2xFS 1mm +vacuum"
    mixed = Layout((PlateSpec("FS", 1.0, -0.5), PlateSpec("CaF2", 2.0, 0.5)))
    up = mixed.shifted_thickness((0.5, 1.0, 2.0, 3.0), 1)
    assert [p.thickness_mm for p in up.plates] == [2.0, 3.0]
    assert mixed.shifted_thickness((0.5, 1.0, 2.0), 1) is None          # 2 mm has no thicker neighbour


def test_plates_make_the_mirrors_further_apart_than_the_equivalent_cell():
    c = westerberg()
    n = th.Solid("FS").n_lam(LAM0)
    assert c.equivalent_separation_m == pytest.approx(0.2 * (1 + np.cos(np.pi / 7)))
    assert c.separation_m - c.equivalent_separation_m == pytest.approx(2e-3 * (1 - 1 / n), rel=1e-6)
    assert c.plate_positions_m == pytest.approx([-0.733 * c.separation_m / 2, 0.733 * c.separation_m / 2])
    cell = c.to_cell(lambda0_m=LAM0)
    assert cell.d == pytest.approx(c.separation_m)
    assert cell.equivalent_separation(LAM0) == pytest.approx(c.equivalent_separation_m, rel=1e-9)
    assert len(cell.plates) == 2 and cell.medium.n2 == 0


def test_gas_sections_and_individual_plates_are_built_as_given():
    lay = Layout((PlateSpec("FS", 1.0, -0.7), PlateSpec("CaF2", 2.0, 0.7, coating="bare")),
                 (RegionSpec(-0.7, 0.7, "vacuum"),))
    c = westerberg(lay, gas="air", pressure=1.0)
    cell = c.to_cell(lambda0_m=LAM0)
    assert [p.material.name for p in cell.plates] == ["FS", "CaF2"]
    assert cell.plates[1].coating_A.is_bare and not cell.plates[0].coating_A.is_bare
    (z0, z1, med), = cell.regions
    assert (z0, z1) == pytest.approx((-0.35 * c.separation_m, 0.35 * c.separation_m)) and med.n2 == 0
    assert cell.medium.name == "air"
    assert cell.equivalent_separation(LAM0) == pytest.approx(c.equivalent_separation_m, rel=1e-9)
    assert c.gases[0] == ("air", 1.0) and c.tunable_pressure
    # a section with its own fixed pressure, inside a vacuum cell: nothing to tune
    fixed = westerberg(Layout(FS2.plates, (RegionSpec(-0.5, 0.5, "Ar", 0.2),)))
    assert not fixed.tunable_pressure and ("Ar", 0.2) in fixed.gases
    assert westerberg(Layout(FS2.plates, (RegionSpec(-0.5, 0.5, "Ar"),))).tunable_pressure


def test_layout_checks():
    assert westerberg(Layout.identical(1, "FS", 1.0, 0.5)).plate_positions_m == pytest.approx(
        [0.25 * westerberg(Layout.identical(1, "FS", 1.0, 0.5)).separation_m])
    assert westerberg(Layout.identical(2, "FS", 1.0, 0.0005)).plate_layout_ok() is False    # overlap
    assert westerberg(Layout.identical(2, "FS", 1.0, 0.999)).plate_layout_ok() is False     # out of the cell
    one_mirror = Layout(regions=(RegionSpec(0.5, 1.0, "vacuum"),))     # the two mirrors in different gases
    assert westerberg(one_mirror, gas="Ar", pressure=0.1).plate_layout_ok() is False
    prob = westerberg_problem()
    assert "plate_layout" in prefilter(prob, westerberg(Layout.identical(2, "FS", 1.0, 0.999)))
    assert "plate_clearance" in prefilter(prob, westerberg(Layout.identical(2, "FS", 1.0, 0.95)))


# ------------------------------------------------------------ analytic tier
def test_matched_estimates_are_those_of_the_cell():
    prob, c = westerberg_problem(), westerberg()
    q = AnalyticQuick()(prob, c)
    m = c.to_cell(lambda0_m=LAM0).nonlinear_mode(prob.laser.peak_power_W, LAM0)
    assert q.b_per_pass_rad == pytest.approx(max(m["B_pass"]), rel=1e-6)
    assert q.b_total_rad == pytest.approx(m["B_total"], rel=1e-6)
    assert q.extras["plate_peak_intensity_W_cm2"] == pytest.approx(max(d["I_peak_W_cm2"] for d in m["plates"]),
                                                                    rel=1e-3)
    assert q.extras["plate_b_fraction"] == pytest.approx(1.0)
    assert q.p_over_pcr is None and q.peak_intensity_W_cm2 is None         # no gas
    assert 200 < q.extras["plate_p_over_pcr"] < 300                        # ~250 Pc, as the paper
    assert [p["material"] for p in q.extras["plates"]] == ["FS", "FS"]


def test_linear_screen_estimate_is_fast_and_overestimates_the_plates():
    prob, c = westerberg_problem(), westerberg()
    lin = AnalyticQuick(plate_mode="linear")(prob, c)
    mat = AnalyticQuick()(prob, c)
    # the Kerr lens of the plates widens the beam on them: the linear mode is pessimistic there
    assert lin.extras["plate_peak_intensity_W_cm2"] > mat.extras["plate_peak_intensity_W_cm2"]
    assert lin.b_per_pass_rad > mat.b_per_pass_rad
    assert lin.extras["plate_p_over_pcr"] == pytest.approx(mat.extras["plate_p_over_pcr"], rel=1e-3)


def test_published_cells_pass_the_default_limits():
    prob = westerberg_problem()
    for c in (westerberg(), westerberg(WINDOWS, gas="air", pressure=1.0)):
        c = evaluated(prob, c)
        assert np.isfinite(c.score), c.report(prob)
        for name in ("plate_intensity", "self_focusing", "plate_clearance", "b_per_pass"):
            assert c.slacks[name] > 0
        assert any("no LIDT for FS plates" in w for w in c.warnings)
    bulk = evaluated(prob, westerberg())
    assert "pressure" not in bulk.limits and "ionization" not in bulk.limits and "p_over_pcr" not in bulk.limits
    windowed = evaluated(prob, westerberg(WINDOWS, gas="air", pressure=1.0))
    assert "ionization" in windowed.limits and windowed.metrics["gas_sections"][0]["gas"] == "air"
    rep = windowed.report(prob)
    assert "plate  : FS 1 mm" in rep and "section: vacuum" in rep and "air 1.000 bar |" in rep


def test_each_plate_is_checked_with_its_own_material():
    prob = westerberg_problem()
    prob.constraints.plates.allowed_materials = ("FS", "CaF2")
    from thunder4d.design.catalog import solids
    old = solids.PLATES["CaF2"]
    solids.PLATES["CaF2"] = old.with_(intensity_threshold_W_cm2=1e11)
    try:
        lay = Layout((PlateSpec("FS", 1.0, -0.733), PlateSpec("CaF2", 1.0, 0.733)))
        c = evaluated(prob, westerberg(lay))
        assert c.slacks["plate_intensity"] < 0 and "(CaF2)" in c.limits["plate_intensity"][3]
    finally:
        solids.PLATES["CaF2"] = old


def test_plate_limits_and_allowed_hardware():
    prob = westerberg_problem()
    prob.constraints.plates.peak_intensity_W_cm2 = Max(1e11)
    c = evaluated(prob, westerberg())
    assert not np.isfinite(c.score) and c.slacks["plate_intensity"] < 0
    prob = westerberg_problem()
    prob.constraints.plates.fluence = Max(1e-3)            # explicit value: checked without a catalogue LIDT
    assert evaluated(prob, westerberg()).slacks["plate_fluence"] < 0
    prob = westerberg_problem()
    for spec, name in ((PlateSpec("CaF2", 1.0, 0.7), "plate_material"), (PlateSpec("FS", 1.5, 0.7), "plate_thickness"),
                       (PlateSpec("FS", 1.0, 0.7, coating="bare"), "plate_coating")):
        c = evaluated(prob, westerberg(Layout((PlateSpec("FS", 1.0, -0.7), spec))))
        assert c.slacks[name] == -1.0 and not np.isfinite(c.score)


def test_bare_plates_lose_fresnel_reflections():
    prob = westerberg_problem()
    bare = AnalyticQuick()(prob, westerberg(Layout.identical(2, "FS", 1.0, 0.733, coating="bare")))
    ar = AnalyticQuick()(prob, westerberg())
    r = ((1.45 - 1) / (1.45 + 1)) ** 2
    assert bare.throughput / ar.throughput == pytest.approx((1 - r) ** (4 * 14), rel=0.02)


def test_hybrid_cell_splits_the_nonlinearity():
    prob = DesignProblem(laser=LaserSpec(energy_J=2e-3, duration_fs=250.0))
    c = Candidate("Ar", 0.5, 1.0, 11, 1, "concentric", layout=Layout.identical(2, "FS", 1.0, 0.6))
    gas_only = AnalyticQuick()(prob, Candidate("Ar", 0.5, 1.0, 11, 1, "concentric"))
    q = AnalyticQuick()(prob, c)
    assert 0 < q.extras["plate_b_fraction"] < 1
    assert q.b_per_pass_rad > gas_only.b_per_pass_rad
    assert q.p_over_pcr == pytest.approx(gas_only.p_over_pcr, rel=1e-3)
    assert evaluated(prob, c).limits["pressure"][0] == 0.5


# ------------------------------------------------------------ fast1d with plates and gas sections
def test_fast1d_follows_the_matched_mode_of_the_loaded_cell():
    prob = westerberg_problem()
    cell = westerberg().to_cell(lambda0_m=LAM0)
    P = prob.laser.peak_power_W
    out = fast1d.simulate(fast1d.MPCInput(energy=250e-6, fwhm=180e-15, roc=0.2, n_passes=14, mirror_reflectivity=1.0,
                                          cell=cell, match_power=P, accuracy="default"))
    m = cell.nonlinear_mode(P, LAM0)
    # the pulse peak sees the matched mode: same mirror spot on the first passes (it then drifts slowly
    # as the peak power changes), same B per plate
    assert out.pass_w_mirror[:3] == pytest.approx(np.full(3, m["w_mirror"]), rel=0.002)
    assert out.pass_w_mirror == pytest.approx(np.full(14, m["w_mirror"]), rel=0.05)
    assert out.plate_crossings[0]["b"] == pytest.approx(m["plates"][0]["B"], rel=0.02)
    assert len(out.plate_crossings) == 28 and {r["plate"] for r in out.plate_crossings} == {0, 1}
    assert out.segmented and out.peak_power_over_pcr == 0.0 and out.gas_intensity_max == {}
    assert 38 < out.fwhm_compressed * 1e15 < 50      # paper 42 fs, 4D (coarse) 43.5 fs
    assert out.transmission == pytest.approx(1.0)    # ideal AR, R = 1 mirrors


def test_fast1d_plate_faces_and_gas_sections():
    lay = Layout((PlateSpec("FS", 1.0, -0.733, coating="bare"), PlateSpec("FS", 1.0, 0.733, coating="bare")),
                 (RegionSpec(-0.733, 0.733, "vacuum"),))
    cell = westerberg(lay, gas="air", pressure=1.0).to_cell(lambda0_m=LAM0)
    out = fast1d.simulate(fast1d.MPCInput(energy=1e-9, fwhm=180e-15, roc=0.2, n_passes=4, mirror_reflectivity=1.0,
                                          cell=cell, accuracy="fast"))
    assert out.transmission == pytest.approx(float(cell.plate_transmission(LAM0)) ** 4, rel=1e-3)
    assert set(out.gas_intensity_max) == {"air"}


def test_fast1d_quick_on_bulk_and_windowed_cells():
    prob = westerberg_problem()
    for c in (westerberg(), westerberg(WINDOWS, gas="air", pressure=1.0)):
        q = Fast1DQuick()(prob, c)
        ana = AnalyticQuick()(prob, c)
        assert q.b_per_pass_rad == pytest.approx(ana.b_per_pass_rad, rel=0.05)
        assert q.extras["plate_peak_intensity_W_cm2"] == pytest.approx(ana.extras["plate_peak_intensity_W_cm2"],
                                                                        rel=0.05)
        assert len(q.extras["plates"]) == 2 and abs(q.duration_fs - 43.0) < 5.0
        assert np.isfinite(evaluate(prob, c, q.as_metrics()).score)


def test_a_gas_cell_given_as_a_thunder4d_cell_gives_the_same_result():
    inp = fast1d.MPCInput.herriott(roc=2.0, N=15, k=2, gas="Ar", pressure=0.13, energy=18e-3, fwhm=350e-15,
                                   accuracy="default")
    a = fast1d.simulate(inp)
    cell = th.MPC(2.0, inp.length, th.Material("Ar", 0.13), inp.n_passes)
    b = fast1d.simulate(fast1d.MPCInput(**{**inp.__dict__, "cell": cell}))
    assert b.segmented and not a.segmented
    for key in ("fwhm_compressed", "b_integral", "transmission", "fluence_mirror_max", "peak_power_over_pcr"):
        assert getattr(b, key) == pytest.approx(getattr(a, key), rel=0.01), key


def test_bulk_mirror_fluence_uses_the_smaller_spot():
    prob, c = westerberg_problem(), westerberg()
    for quick in (AnalyticQuick(), Fast1DQuick()):
        q = quick(prob, c)
        w_lin = c.linear_mode(LAM0)[1]
        assert q.extras["w_mirror_max_m"] > w_lin            # the plates defocus the peak on the mirrors
        assert q.mirror_fluence_J_cm2.max() == pytest.approx(2 * 250e-6 / (np.pi * w_lin ** 2) * 1e-4, rel=0.03)


# ------------------------------------------------------------ search
def test_plate_options_and_starting_points():
    prob = westerberg_problem()
    mixed = Layout((PlateSpec("FS", 1.0, -0.7), PlateSpec("FS", 2.0, 0.7)))
    prob.search = SearchSpace(gases=("vacuum", "Ar"), round_trips=(6, 8), plate_counts=(2,),
                              plate_thicknesses_mm=(1.0, 2.0), plate_positions_rel=(0.7,), include_no_plates=True,
                              layouts=(mixed,))
    assert all(o.plates for o in plate_options(prob, "vacuum"))     # vacuum: never empty
    assert Layout() in plate_options(prob, "Ar") and mixed in plate_options(prob, "Ar")
    starts = starting_points(prob, verbose=False)
    vac = [c for c in starts if c.is_vacuum]
    assert vac and all(c.has_plates and c.pressure_bar == 0.0 for c in vac)
    assert {c.layout.label for c in vac} == {"2xFS 1mm", "2xFS 2mm", "1xFS 1mm 1xFS 2mm"}
    assert any(not c.segmented for c in starts if c.gas == "Ar")
    prob.search = SearchSpace(gases=("Ar",), round_trips=(6, 8))      # no plate search: gas helper unchanged
    assert not any(c.segmented for c in starting_points(prob, verbose=False))


def test_neighbours_step_every_plate_through_the_stock_thicknesses():
    prob = westerberg_problem()
    prob.constraints.plates.allowed_thicknesses_mm = (0.5, 1.0, 2.0, 3.0)
    prob.refinement.neighbour_n_passes = 0
    nbs = list(MPCDesigner(prob, AnalyticQuick())._neighbours(westerberg()))
    assert sorted(c.plate_thickness_mm for c in nbs) == [0.5, 1.0, 2.0]
    assert all(len({p.thickness_mm for p in c.layout.plates}) == 1 for c in nbs)


def test_bulk_and_windowed_designs_end_to_end():
    prob = westerberg_problem()
    prob.search = SearchSpace(gases=("vacuum",), roc_m=(0.2,), round_trips=(5, 9), plate_counts=(2,),
                              plate_thicknesses_mm=(0.5, 1.0, 2.0), plate_positions_rel=(0.5, 0.75))
    prob.refinement.n_candidates = 2
    res = MPCDesigner(prob, quick=Fast1DQuick(), verbose=False).run()
    assert len(res.refined) == 2
    best = res.best
    assert best.has_plates and best.is_vacuum and np.isfinite(best.score)
    assert abs(best.metrics["duration_fs"] - 40.0) < 4.0
    assert "layout" in res.table()
    # the published layout as a template, air fixed at 1 bar
    prob.search = SearchSpace(gases=("air",), roc_m=(0.2,), round_trips=(6, 8), layouts=(WINDOWS,))
    prob.constraints.pressure.pressure_bar = (1.0, 1.0)
    prob.refinement.tune = ()
    res = MPCDesigner(prob, quick=Fast1DQuick(), verbose=False).run()
    assert res.refined and all(c.layout.regions == WINDOWS.regions for c in res.refined)
    assert all(c.pressure_bar == 1.0 for c in res.refined)


def test_design_map_over_plate_thickness_and_position():
    prob = westerberg_problem()
    prob.constraints.plates.allowed_thicknesses_mm = (0.5, 1.0, 2.0, 3.0)
    m = design_map(prob, gas="vacuum", roc_m=0.2, k=1, y=("round_trips", [6, 7, 8]),
                   layout=Layout.identical(2, "FS", 1.0, 0.7))
    assert m.x_name == "plate_thickness_mm" and list(m.x) == [0.5, 1.0, 2.0, 3.0]
    assert np.all(np.diff(m.values["b_per_pass_rad"], axis=1) > 0)      # thicker plates, more B
    m = design_map(prob, gas="vacuum", roc_m=0.2, k=1, round_trips=7, x=("plate_position_rel", [0.3, 0.5, 0.7]),
                   y=("n_passes", [12, 14]), layout=Layout.identical(2, "FS", 1.0, 0.7))
    assert np.all(np.diff(m.values["b_per_pass_rad"], axis=1) < 0)      # towards the mirrors, larger beam
    with pytest.raises(ValueError, match="layout"):
        design_map(prob, gas="vacuum", roc_m=0.2, k=1, round_trips=7)


# ------------------------------------------------------------ export to the input file
def test_export_round_trip_rebuilds_the_cell(tmp_path):
    prob = westerberg_problem()
    lay = Layout((PlateSpec("FS", 1.0, -0.733, ar_reflectivity=1e-3), PlateSpec("CaF2", 1.0, 0.733)),
                 (RegionSpec(-0.733, 0.733, "vacuum"),))
    c = evaluated(prob, westerberg(lay, gas="air", pressure=1.0))
    cfg = to_input(prob, c)
    assert cfg["beam"]["injection"] == "nonlinear" and cfg["medium"]["gas"] == "air"
    assert [p["material"] for p in cfg["plates"]] == ["FS", "CaF2"]
    assert [p["position_m"] for p in cfg["plates"]] == pytest.approx(c.plate_positions_m)
    assert cfg["plates"][0]["coating"] == {"reflectivity": 1e-3} and cfg["plates"][1]["coating"] == "ar"
    assert cfg["regions"][0]["gas"] == "vacuum"
    assert cfg["regions"][0]["z1_m"] == pytest.approx(0.733 * c.separation_m / 2)
    assert cfg["design"]["candidate"]["equivalent_separation_m"] == pytest.approx(c.equivalent_separation_m)
    th.save_input(cfg, tmp_path / "bulk.json")
    cell = make_cell(load_input(tmp_path / "bulk.json"))
    ref = c.to_cell(lambda0_m=LAM0)
    assert cell.d == pytest.approx(ref.d)
    assert cell.equivalent_separation() == pytest.approx(ref.equivalent_separation(), rel=1e-9)
    assert float(cell.plate_transmission(LAM0)) == pytest.approx(float(ref.plate_transmission(LAM0)))
    # the design block round-trips into a Candidate
    again = Candidate(**{k: v for k, v in cfg["design"]["candidate"].items()
                         if k not in ("score", "equivalent_separation_m")})
    assert again.layout == lay


def test_plate_cell_input_builds_a_matched_pulse():
    prob = westerberg_problem()
    b = build_simulation(to_input(prob, evaluated(prob, westerberg())))
    assert b.cell.segmented and np.isfinite(b.pulse.input_R)       # curved wavefront at the centre
    lin = build_simulation(to_input(prob, evaluated(prob, westerberg()), injection="linear"))
    assert not np.isfinite(lin.pulse.input_R)


def test_input_file_plate_and_region_validation():
    prob = westerberg_problem()
    cfg = th.load_input(to_input(prob, evaluated(prob, westerberg(WINDOWS, gas="air", pressure=1.0))))
    assert input_problems(cfg) == []
    bad = dict(cfg, plates=[{"material": "glass", "thickness_m": 0, "coating": "matte", "colour": 1}],
               regions=[{"z0_m": 0.1, "z1_m": -0.1, "gas": "Ar"}])
    msgs = " ".join(m for _, m in input_problems(bad))
    for word in ("unknown key", "material", "thickness_m", "position_m", "coating", "z0_m < z1_m",
                 "regions[0].pressure_bar"):
        assert word in msgs
