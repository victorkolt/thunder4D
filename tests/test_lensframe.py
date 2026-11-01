import numpy as np
import pytest

import thunder4d as m
from thunder4d import diagnostics as dg

from helpers import LAM0, overlap, vacuum


def near_concentric(medium, n_passes=2):
    """The cell of benchmark/lens_frame_check.py: d close to 2R, where the lens frame pays off."""
    return m.MPC(R=0.5, d=0.94, medium=medium, n_passes=n_passes)


def eigenmode_pulse(cell, g, energy=1e-3, w0=None):
    w0 = w0 or cell.linear_mode(LAM0)["w0"]
    beam = m.Beam(m.gaussian(w0), chromatic_size="eigenmode")
    return m.Pulse.at_focus(g, m.gaussian_spectrum(g, fwhm_fs=200), beam, energy)


# ------------------------------------------------------------ suggest_lens_grid
BAND = {"lambda0": LAM0, "lambda_min": 900e-9, "lambda_max": 1200e-9, "pulse_fwhm": 200e-15}


def test_suggest_lens_grid_spectral_window_covers_the_band():
    g = m.suggest_lens_grid(near_concentric(m.Material("Ar", 1.0)), **BAND, device="cpu")
    lam = g.lam[g.valid]
    assert lam.min() <= 900e-9
    assert lam.max() >= 1200e-9


def test_suggest_lens_grid_time_window_holds_the_pulse():
    g = m.suggest_lens_grid(near_concentric(m.Material("Ar", 1.0)), **BAND, time_window_factor=8,
                            device="cpu")
    assert g.Nt * g.dt >= 8 * 200e-15


# ------------------------------------------------------------ lens frame vs standard solver
def lens_grid(cell):
    return m.suggest_lens_grid(cell, LAM0, 1000e-9, 1060e-9, 200e-15, box_factor=8, points_per_waist=4,
                               dtype=np.complex128, device="cpu")


def test_eigenmode_is_unchanged_by_one_pass():
    cell = near_concentric(vacuum(), n_passes=1)
    g = lens_grid(cell)
    p = eigenmode_pulse(cell, g)
    sim = m.LensFrameSimulation(p, cell, near_field=False, record_mirror=False).run(verbose=False)
    assert overlap(p.Aw, sim.Aw) > 0.9999


@pytest.fixture(scope="module")
def kerr_pair():
    """Kerr-matched gaussian, 1 mJ, 200 fs, 2 passes in argon at 1 bar (B ~ 3.7 rad), run with both
    solvers. The standard grid has to hold the beam on the mirrors, the lens grid only at the waist."""
    cell = near_concentric(m.Material("Ar", 1.0))
    grids = {
        m.MPCSimulation: m.suggest_grid(cell, LAM0, 1000e-9, 1060e-9, 200e-15, box_factor=6,
                                        points_per_waist=4, dtype=np.complex128, device="cpu"),
        m.LensFrameSimulation: lens_grid(cell),
    }
    sims = []
    for Sim, g in grids.items():
        w0 = cell.match(eigenmode_pulse(cell, g), verbose=False)["w0"]
        sim = Sim(eigenmode_pulse(cell, g, w0=w0), cell, near_field=False, record_mirror=False)
        sims.append(sim.run(verbose=False))
    return sims


def test_lens_frame_B_matches_the_standard_solver(kerr_pair):
    std, lens = kerr_pair
    assert lens.solver.B == pytest.approx(std.solver.B, rel=0.005)


@pytest.mark.parametrize("key", ["energy", "eta00_total", "w_peak_x"])
def test_lens_frame_history_matches_the_standard_solver(kerr_pair, key):
    std, lens = kerr_pair
    assert lens.history[key][-1] == pytest.approx(std.history[key][-1], rel=0.005)


def test_lens_frame_spectrum_matches_the_standard_solver(kerr_pair):
    (la, Sa), (lb, Sb) = (dg.total_spectrum(s.grid, s.Aw) for s in kerr_pair)
    Sa, Sb = Sa / Sa.max(), np.interp(la, lb, Sb / Sb.max())
    similarity = np.trapezoid(np.sqrt(Sa * Sb), la) ** 2 / (np.trapezoid(Sa, la) * np.trapezoid(Sb, la))
    assert similarity > 0.9999
