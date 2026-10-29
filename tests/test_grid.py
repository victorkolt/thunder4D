import numpy as np
import pytest
import thunder4d as m
from thunder4d.grid import _next_fast
from helpers import LAM0


def make_grid(**kw):
    return m.Grid(8, 10e-6, 64, LAM0, dt=2e-15, dtype=np.complex128, device="cpu", **kw)


def random_field(g):
    rng = np.random.default_rng(0)
    return rng.normal(size=g.shape) + 1j * rng.normal(size=g.shape)


def test_next_fast_is_the_smallest_2_3_5_number():
    smooth = sorted(2 ** a * 3 ** b * 5 ** c for a in range(12) for b in range(8) for c in range(6))
    for n in range(1, 1000):
        assert _next_fast(n) == next(v for v in smooth if v >= n)


def test_grid_needs_dt_or_lambda_min():
    with pytest.raises(ValueError):
        m.Grid(8, 10e-6, 64, LAM0, device="cpu")


@pytest.mark.parametrize("Nt", [64, 65])
def test_grid_spectral_window_covers_the_band(Nt):
    g = m.Grid(8, 10e-6, Nt, LAM0, lambda_min=900e-9, lambda_max=1200e-9, device="cpu")
    lam = g.lam[g.valid]
    assert lam.min() <= 900e-9
    assert lam.max() >= 1200e-9


def test_energy_is_the_same_in_time_and_frequency():
    g = make_grid()
    Aw = random_field(g)
    assert g.energy_t(g.fft.ift_t(Aw)) == pytest.approx(g.energy(Aw), rel=1e-12)


def test_ift_t_inverts_ft_t():
    g = make_grid()
    At = random_field(g)
    np.testing.assert_allclose(g.fft.ift_t(g.fft.ft_t(At)), At, atol=1e-12)


def test_phase_exp_plus_i_omega_t0_delays_by_t0():
    g = make_grid()
    t0 = 20e-15
    E = np.exp(-(g.Omega * 5e-15) ** 2) * np.exp(1j * g.Omega * t0)
    assert g.t[np.argmax(np.abs(g.fft.ift_t(E)))] == pytest.approx(t0)


@pytest.mark.parametrize("method", ["fft2", "ifft2", "ft_t", "ift_t"])
def test_numpy_and_scipy_backends_agree(method):
    a, b = make_grid(fft_backend="numpy"), make_grid(fft_backend="scipy")
    A = random_field(a)
    np.testing.assert_allclose(getattr(a.fft, method)(A), getattr(b.fft, method)(A), atol=1e-12)


# ------------------------------------------------------------ suggest_grid
BAND = dict(lambda0=LAM0, lambda_min=900e-9, lambda_max=1200e-9, pulse_fwhm=200e-15)


@pytest.fixture
def cell():
    return m.MPC.herriott(R=0.5, N=20, k=3, branch="planar", medium=m.Material("Ar", 1.0))


def test_suggest_grid_box_holds_the_mirror_spot(cell):
    g = m.suggest_grid(cell, **BAND, box_factor=10, device="cpu")
    assert g.Nx * g.dx == pytest.approx(10 * cell.linear_mode(1200e-9)["w_mirror"])


def test_suggest_grid_resolves_the_smallest_waist(cell):
    g = m.suggest_grid(cell, **BAND, points_per_waist=8, device="cpu")
    assert g.dx <= cell.linear_mode(900e-9)["w0"] / 8


def test_suggest_grid_spectral_window_covers_the_band(cell):
    g = m.suggest_grid(cell, **BAND, device="cpu")
    lam = g.lam[g.valid]
    assert lam.min() <= 900e-9
    assert lam.max() >= 1200e-9


def test_suggest_grid_time_window_holds_the_pulse(cell):
    g = m.suggest_grid(cell, **BAND, time_window_factor=8, device="cpu")
    assert g.Nt * g.dt >= 8 * 200e-15
