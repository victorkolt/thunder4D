import numpy as np
import pytest
import thunder4d as m
from helpers import LAM0

AR = m.Material("Ar", 1.0)


def make_grid():
    return m.Grid(16, 20e-6, 32, LAM0, dt=4e-15, dtype=np.complex128, device="cpu")


# ------------------------------------------------------------ geometry
@pytest.mark.parametrize("branch, N, k", [("planar", 4, 1), ("planar", 20, 3), ("concentric", 20, 9),
                                          ("concentric", 26, 1)])
def test_herriott_cell_is_reentrant(branch, N, k):
    cell = m.MPC.herriott(R=0.5, N=N, k=k, medium=AR, branch=branch)
    turns = N * cell.theta / (2 * np.pi)
    assert turns == pytest.approx(round(turns), abs=1e-9)


def test_concentric_branch_rejects_odd_N():
    with pytest.raises(ValueError):
        m.MPC.herriott(R=0.5, N=21, k=2, medium=AR, branch="concentric")


def test_unknown_branch_is_rejected():
    with pytest.raises(ValueError):
        m.MPC.herriott(R=0.5, N=20, k=3, medium=AR, branch="confocal")


@pytest.mark.parametrize("d", [-0.1, 0.0, 1.0, 1.2])
def test_unstable_cell_is_rejected(d):
    with pytest.raises(ValueError):
        m.MPC(R=0.5, d=d, medium=AR, n_passes=4)


# ------------------------------------------------------------ linear eigenmode
@pytest.fixture(params=[0.3, 0.5, 0.9])
def cell(request):
    return m.MPC(R=0.5, d=request.param * 0.5, medium=AR, n_passes=10)


def test_mirror_spot_follows_gaussian_beam_expansion(cell):
    mode = cell.linear_mode(LAM0)
    z = cell.d / 2
    assert mode["w_mirror"] == pytest.approx(mode["w0"] * np.sqrt(1 + (z / mode["zR"]) ** 2), rel=1e-12)


def test_eigenmode_wavefront_matches_the_mirror_curvature(cell):
    mode = cell.linear_mode(LAM0)
    z = cell.d / 2
    assert z + mode["zR"] ** 2 / z == pytest.approx(cell.R, rel=1e-12)


def test_gouy_phase_per_pass_matches_the_eigenmode(cell):
    mode = cell.linear_mode(LAM0)
    assert cell.theta == pytest.approx(2 * np.arctan(cell.d / 2 / mode["zR"]), rel=1e-12)


# ------------------------------------------------------------ Kerr-lens matching
@pytest.fixture
def herriott():
    return m.MPC.herriott(R=0.5, N=20, k=3, branch="planar", medium=AR)


def test_kerr_B_per_pass_tends_to_2_theta_P_over_Pc(herriott):
    Pc = AR.critical_power(LAM0)
    B = herriott.nonlinear_mode(1e-4 * Pc, lam=LAM0)["B_per_pass"]
    assert B == pytest.approx(2 * herriott.theta * 1e-4, rel=1e-3)


def test_no_matched_mode_above_the_critical_power(herriott):
    with pytest.raises(ValueError):
        herriott.nonlinear_mode(1.01 * AR.critical_power(LAM0), lam=LAM0)


# ------------------------------------------------------------ mirror bounce
def test_scalar_reflectivity_scales_the_energy():
    g = make_grid()
    A = np.random.default_rng(0).normal(size=g.shape) + 0j
    E0 = g.energy(A)
    m.MPC(R=0.5, d=0.5, medium=AR, n_passes=4, mirror_reflectivity=0.9).bounce(A, g)
    assert g.energy(A) == pytest.approx(0.9 * E0, rel=1e-12)


def test_reflectivity_curve_filters_each_wavelength():
    g = make_grid()
    R = lambda lam: np.where(lam < LAM0, 0.5, 1.0)
    A = np.ones(g.shape, complex)
    m.MPC(R=0.5, d=0.5, medium=AR, n_passes=4, mirror_reflectivity=R).bounce(A, g)
    np.testing.assert_allclose(np.abs(A[0, 0, :]) ** 2, R(g.lam), rtol=1e-12)


def test_mirror_dispersion_uses_the_add_spectral_phase_convention():
    g = make_grid()
    cell = m.MPC(R=0.5, d=0.5, medium=AR, n_passes=4, mirror_gdd_fs2=500.0, mirror_tod_fs3=2000.0)
    expected = m.add_spectral_phase(g, np.ones(g.Nt), gdd_fs2=500.0, tod_fs3=2000.0)
    np.testing.assert_allclose(cell.bounce_factor(g)[0, 0, :], expected, rtol=1e-12)


def test_mirror_aperture_blocks_outside_the_diameter():
    g = make_grid()
    F = np.abs(m.MPC(R=0.5, d=0.5, medium=AR, n_passes=4, mirror_diameter=200e-6).bounce_factor(g)[:, :, 0])
    i = int(np.argmin(np.abs(g.x - 140e-6)))
    assert F[0, 0] == pytest.approx(1.0)
    assert F[i, 0] < 1e-3
