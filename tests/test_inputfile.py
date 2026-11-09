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
