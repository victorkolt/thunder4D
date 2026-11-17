import json

import numpy as np
import pytest

import thunder4d as th
from thunder4d.inputfile import make_cell, phase_from_gdd


def small_input(**over):
    cfg = th.default_input(
        name="small",
        laser={"energy_J": 1e-3, "duration_fs": 200.0},
        medium={"gas": "Ar", "pressure_bar": 0.5},
        cell={"roc_m": 1.0, "separation_m": 1.9, "n_passes": 2},
        grid={"lambda_min_nm": 990.0, "lambda_max_nm": 1075.0, "device": "cpu"},
        run={"near_field": False},
    )
    for block, vals in over.items():
        cfg[block].update(vals)
    return cfg


def test_missing_fields_are_all_listed():
    with pytest.raises(ValueError) as err:
        th.load_input({"schema": "thunder4d-input"})
    msg = str(err.value)
    for key in ("laser.energy_J", "medium.gas", "cell.n_passes", "grid.lambda_min_nm"):
        assert key in msg


def test_typos_are_rejected_but_design_block_is_free():
    with pytest.raises(KeyError, match="laser.energy"):
        th.load_input({"schema": "thunder4d-input", "laser": {"energy": 1e-3}})
    cfg = small_input()
    cfg["design"] = {"anything": {"nested": [1, 2]}}
    assert th.load_input(cfg)["design"]["anything"]["nested"] == [1, 2]


def test_geometry_checks():
    with pytest.raises(ValueError, match="unstable"):
        th.load_input(small_input(cell={"separation_m": 2.5}))
    with pytest.raises(ValueError, match="lambda_min_nm"):
        th.load_input(small_input(grid={"lambda_min_nm": 1040.0}))


def test_save_load_roundtrip(tmp_path):
    cfg = small_input(mirrors={"reflectivity": {"wavelength_nm": [900, 1200], "value": [0.99, 0.999]}})
    path = th.save_input(cfg, tmp_path / "cell.json")
    raw = json.loads(path.read_text())
    assert raw["schema"] == "thunder4d-input" and raw["version"] == 1
    assert th.load_input(path) == th.load_input(cfg)


def test_mirror_tables_match_scalars():
    lam = np.linspace(900.0, 1200.0, 301)
    tab = make_cell(small_input(mirrors={"gdd_fs2": {"wavelength_nm": lam.tolist(), "value": [-50.0] * 301},
                                         "reflectivity": {"wavelength_nm": [900, 1200], "value": [0.99, 0.99]}}))
    sca = make_cell(small_input(mirrors={"gdd_fs2": -50.0, "reflectivity": 0.99}))
    g = th.suggest_lens_grid(sca, 1030e-9, 990e-9, 1075e-9, 200e-15, box_factor=4, points_per_waist=2,
                             time_window_factor=3, device="cpu")
    a, b = tab.mirror_spectrum(g), sca.mirror_spectrum(g)
    band = g.valid & (np.abs(g.omega - g.omega0) < 0.05 * g.omega0)
    assert np.allclose(a[band], b[band], atol=2e-3)


def test_phase_from_gdd_has_the_gdd():
    lam0 = 1030e-9
    w0 = 2 * np.pi * 299792458.0 / lam0 * 1e-15
    dw = 0.1
    lams = 2 * np.pi * 299792458.0 / (np.array([w0 - dw, w0, w0 + dw]) * 1e15)
    ph = phase_from_gdd(lams, np.linspace(900, 1200, 301), np.full(301, 80.0), lam0)
    assert (ph[0] - 2 * ph[1] + ph[2]) / dw ** 2 == pytest.approx(80.0, rel=2e-3)


def test_run_input_runs_and_starts_after_a_mirror():
    b = th.run_input(small_input(), verbose=False)
    assert b.sim.passes_done == 2 and len(b.sim.history["energy"]) == 3
    assert b.sim.solver.B > 0
    # the input recorded at pass 0 is the pulse given at the cell centre
    assert b.sim.history["energy"][0] == pytest.approx(1e-3, rel=1e-6)


# ============================================================ field descriptions (GUI forms)
def test_every_key_of_the_input_is_described():
    from thunder4d.inputfile import BLOCKS, FIELDS, field_paths

    assert set(field_paths()) == set(FIELDS)
    assert {p.split(".")[0] if "." in p else "general" for p in FIELDS} == set(BLOCKS)
    for path, field in FIELDS.items():
        if field.show_if is not None:
            assert field.show_if[0] in FIELDS, path


def test_problems_point_at_described_keys():
    from thunder4d.inputfile import FIELDS, input_problems

    cfg = small_input(laser={"energy_J": None, "shape": "measured"}, cell={"separation_m": 3.0},
                      beam={"injection": "waist"}, grid={"lambda_max_nm": 1000.0})
    problems = input_problems(cfg)
    assert {p for p, _ in problems} == {"laser.energy_J", "laser.spectrum", "cell.separation_m", "beam.w0_m",
                                        "grid.lambda_min_nm"}
    assert all(p in FIELDS for p, _ in problems)
    assert input_problems(small_input()) == []


def test_describe_input_without_building_the_pulse():
    from thunder4d.inputfile import describe_input, make_cell, make_grid

    cfg = th.load_input(small_input())
    s = describe_input(cfg)
    cell = make_cell(cfg)
    assert s["w0_m"] == pytest.approx(cell.linear_mode(1030e-9)["w0"])
    grid = make_grid(cfg, cell)
    assert s["grid_shape"] == (grid.Nx, grid.Ny, grid.Nt)
    assert s["peak_power_W"] == pytest.approx(0.9394 * 1e-3 / 200e-15, rel=1e-3)
    assert s["p_over_pcr"] == pytest.approx(s["peak_power_W"] / cell.medium.critical_power(1030e-9))
    assert s["path_length_m"] == pytest.approx(2 * 1.9)


# ============================================================ plates and gas regions (bulk and hybrid cells)
WESTERBERG = {"roc_m": 0.2, "separation_m": 0.382, "n_passes": 14,
              "plates": [{"material": "FS", "thickness_m": 1e-3, "position_m": 0.14},
                         {"material": "FS", "thickness_m": 1e-3, "position_m": -0.14}],
              "regions": [{"z0_m": -0.14, "z1_m": 0.14, "gas": "vacuum"}]}


def hybrid_input(**cell):
    cfg = small_input(medium={"gas": "air", "pressure_bar": 1.0}, cell={**WESTERBERG, **cell},
                      grid={"lambda_min_nm": 880.0, "lambda_max_nm": 1220.0})
    cfg["laser"].update(energy_J=100e-6, duration_fs=180.0)
    return cfg


def test_input_cell_is_the_cell_of_the_script():
    from thunder4d.inputfile import cell_kind

    cfg = th.load_input(hybrid_input())
    cell = make_cell(cfg)
    by_hand = th.MPC(0.2, 0.382, th.Material("air", 1.0, 293.15), 14, wavelength=1030e-9,
                     plates=[th.Plate(th.Solid("FS"), 1e-3, 0.14, coating="ar"),
                             th.Plate(th.Solid("FS"), 1e-3, -0.14, coating="ar")],
                     regions=[(-0.14, 0.14, th.vacuum())])
    assert cell.equivalent_separation() == pytest.approx(by_hand.equivalent_separation())
    assert [p.position for p in cell.plates] == [-0.14, 0.14]
    assert cell_kind(cfg) == "hybrid"
    bulk = th.load_input(hybrid_input()) | {"medium": {**cfg["medium"], "gas": "vacuum", "pressure_bar": None}}
    assert cell_kind(bulk) == "bulk" and make_cell(bulk).medium.n2 == 0
    assert cell_kind(th.load_input(small_input())) == "gas"


def test_plate_and_region_problems_are_listed():
    from thunder4d.inputfile import input_problems

    cfg = th.load_input(hybrid_input())
    cfg["cell"]["plates"] = [{"material": "glass", "thickness_m": -1.0, "position_m": 0.0, "coating_a": "x"}]
    cfg["cell"]["regions"] = [{"z0_m": 0.1, "z1_m": 0.0}, {"z0_m": 0.0, "z1_m": 0.1, "gas": "Ar"}]
    text = "\n".join(msg for path, msg in input_problems(cfg))
    for part in ("unknown material", "thickness_m must be > 0", "coating_a", "need z0_m < z1_m",
                 "pressure_bar is required"):
        assert part in text


def test_vacuum_needs_no_pressure_and_roundtrips(tmp_path):
    cfg = hybrid_input()
    cfg["medium"].update(gas="vacuum", pressure_bar=None)
    cfg["cell"]["regions"] = []
    th.save_input(cfg, tmp_path / "bulk.json")
    assert th.load_input(tmp_path / "bulk.json")["cell"]["plates"][0]["material"] == "FS"


def test_hybrid_input_runs_and_describes_the_plates():
    from thunder4d.inputfile import describe_input

    cfg = hybrid_input()
    cfg["grid"]["preset"] = "coarse"
    d = describe_input(cfg)
    assert d["kind"] == "hybrid" and len(d["plates_p_over_pcr"]) == 2 and d["plates_p_over_pcr"][0] > 50
    b = th.build_simulation(cfg)
    b.sim.run(verbose=False)
    assert 0.9 < b.sim.history["energy"][-1] / b.sim.history["energy"][0] < 1.0
