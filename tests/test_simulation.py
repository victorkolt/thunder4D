import numpy as np
import pytest

import thunder4d as m

from helpers import LAM0, moments, overlap, vacuum


def herriott(medium):
    """4 passes of 90 deg Gouy phase: re-entrant after one cycle of 4 passes."""
    return m.MPC.herriott(R=0.5, N=4, k=1, branch="planar", medium=medium)


def make_grid(cell, box_factor=8):
    return m.suggest_grid(cell, LAM0, 1000e-9, 1060e-9, 200e-15, box_factor=box_factor,
                          dtype=np.complex128, device="cpu")


def eigenmode_pulse(cell, g, energy=1e-3, w0=None, **beam_kw):
    w0 = w0 or cell.linear_mode(LAM0)["w0"]
    beam = m.Beam(m.gaussian(w0), chromatic_size="eigenmode", **beam_kw)
    return m.Pulse.at_focus(g, m.gaussian_spectrum(g, fwhm_fs=200), beam, energy)


def run(pulse, cell, passes=None, **kw):
    sim = m.MPCSimulation(pulse, cell, near_field=False, record_mirror=False, **kw)
    return sim.run(n_passes=passes, verbose=False)


# ------------------------------------------------------------ linear propagation through the cell
@pytest.fixture
def linear_cell():
    return herriott(vacuum())


def test_eigenmode_is_unchanged_by_one_pass(linear_cell):
    g = make_grid(linear_cell)
    p = eigenmode_pulse(linear_cell, g)
    assert overlap(p.Aw, run(p, linear_cell, 1).Aw) > 0.9999


def test_off_axis_beam_returns_after_a_full_cycle(linear_cell):
    g = make_grid(linear_cell)
    p = eigenmode_pulse(linear_cell, g, offset_m=(150e-6, 0.0))
    assert overlap(p.Aw, run(p, linear_cell, 4).Aw) > 0.9999


def test_off_axis_beam_is_mirrored_after_half_a_cycle(linear_cell):
    g = make_grid(linear_cell)
    sim = run(eigenmode_pulse(linear_cell, g, offset_m=(150e-6, 0.0)), linear_cell, 2)
    F = np.sum(np.abs(sim.Aw) ** 2, axis=2)
    assert moments(g.X, F)[0] == pytest.approx(-150e-6, rel=1e-3)


def test_top_hat_returns_after_a_full_cycle(linear_cell):
    g = make_grid(linear_cell, box_factor=14)
    w0 = linear_cell.linear_mode(LAM0)["w0"]
    p = m.Pulse.at_focus(g, m.gaussian_spectrum(g, fwhm_fs=200), m.Beam(m.top_hat(2 * w0)), 1e-3)
    assert overlap(p.Aw, run(p, linear_cell, 4).Aw) > 0.999


def test_record_every_sets_the_recorded_passes(linear_cell):
    g = make_grid(linear_cell)
    sim = run(eigenmode_pulse(linear_cell, g), linear_cell, 4, record_every=2)
    assert sim.history["pass"] == [0, 2, 4]


def test_save_writes_the_history_and_the_final_field(linear_cell, tmp_path):
    g = make_grid(linear_cell)
    run(eigenmode_pulse(linear_cell, g), linear_cell, 2).save(tmp_path / "run.npz")
    with np.load(tmp_path / "run.npz") as f:
        assert len(f["energy"]) == 3
        assert f["Aw_final"].shape == g.shape


# ------------------------------------------------------------ nonlinear: matched gaussian in argon
@pytest.fixture(scope="module")
def kerr():
    """Kerr-matched gaussian, 1.2 mJ, 200 fs, through 4 passes in argon at 1 bar (B ~ 5 rad)."""
    cell = herriott(m.Material("Ar", 1.0))
    g = make_grid(cell)
    mm = cell.match(eigenmode_pulse(cell, g, energy=1.2e-3), verbose=False)
    p = eigenmode_pulse(cell, g, energy=1.2e-3, w0=mm["w0"])
    return cell, p, mm


@pytest.fixture(scope="module")
def kerr_run(kerr):
    cell, p, _ = kerr
    return run(p, cell)


def test_matched_gaussian_loses_less_than_1_percent_of_its_energy(kerr_run):
    E = kerr_run.history["energy"]
    assert E[-1] / E[0] > 0.99


def test_simulated_B_matches_the_kerr_lens_theory_at_low_power():
    """match() assumes an aberration-free (parabolic) Kerr lens: exact only for P << Pc (here 0.03)."""
    cell = herriott(m.Material("Ar", 1.0))
    g = make_grid(cell)
    mm = cell.match(eigenmode_pulse(cell, g, energy=0.1e-3), verbose=False)
    sim = run(eigenmode_pulse(cell, g, energy=0.1e-3, w0=mm["w0"]), cell)
    assert sim.solver.B == pytest.approx(mm["B_total"], rel=0.01)


def test_halving_phi_max_does_not_change_the_output(kerr, kerr_run):
    cell, p, _ = kerr
    assert overlap(kerr_run.Aw, run(p, cell, phi_max=0.01).Aw) > 0.9999


def test_resume_continues_exactly_like_an_uninterrupted_run(kerr, kerr_run, tmp_path):
    cell, p, _ = kerr
    run(p, cell, 2).checkpoint(tmp_path / "ckpt.pkl")
    resumed = m.MPCSimulation(p, cell, near_field=False, record_mirror=False).resume(tmp_path / "ckpt.pkl")
    resumed.run(verbose=False)
    np.testing.assert_array_equal(resumed.Aw, kerr_run.Aw)
    assert resumed.solver.B == kerr_run.solver.B
