import numpy as np
import pytest
import thunder4d as m
from thunder4d import diagnostics as dg
from helpers import LAM0, moments

W = 100e-6


def make_grid():
    return m.Grid(64, 12.5e-6, 64, LAM0, dt=4e-15, dtype=np.complex128, device="cpu")


def pulse(g, profile=None, **beam_kw):
    """1 mJ, 50 fs, same spatial profile at every colour unless beam_kw says otherwise."""
    beam = m.Beam(profile or m.gaussian(W), **beam_kw)
    return m.Pulse.at_focus(g, m.gaussian_spectrum(g, fwhm_fs=50), beam, 1e-3)


@pytest.fixture
def gauss():
    g = make_grid()
    return g, pulse(g).Aw


# ------------------------------------------------------------ energy bookkeeping
def test_total_spectrum_sums_to_the_energy(gauss):
    g, A = gauss
    assert np.sum(dg.total_spectrum(g, A, per_lambda=False)[1]) == pytest.approx(g.energy(A), rel=1e-12)


def test_fluence_integrates_to_the_energy(gauss):
    g, A = gauss
    assert np.sum(dg.fluence(g, A)) * g.dA == pytest.approx(g.energy(A), rel=1e-12)


def test_temporal_power_integrates_to_the_energy(gauss):
    g, A = gauss
    assert np.sum(dg.temporal_power(g, A)[1]) * g.dt == pytest.approx(g.energy(A), rel=1e-12)


# ------------------------------------------------------------ beam size and quality
def test_beam_moments_give_the_gaussian_radius(gauss):
    g, A = gauss
    np.testing.assert_allclose(dg.beam_moments(g, A)["wx"], W, rtol=1e-3)


def test_beam_moments_give_M2_of_1_for_a_gaussian(gauss):
    g, A = gauss
    np.testing.assert_allclose(dg.beam_moments(g, A)["M2x"], 1.0, rtol=1e-3)


def test_beam_moments_give_M2_of_3_for_hg10():
    g = make_grid()
    bm = dg.beam_moments(g, pulse(g, m.hermite_gauss(W, 1, 0)).Aw)
    np.testing.assert_allclose(bm["M2x"], 3.0, rtol=1e-3)
    np.testing.assert_allclose(bm["M2y"], 1.0, rtol=1e-3)


# ------------------------------------------------------------ modal content
def test_mode_content_of_a_gaussian_is_the_fundamental(gauss):
    g, A = gauss
    assert dg.mode_content(g, A, lambda lam: W)["eta00_total"] == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize("mn", [(1, 0), (0, 2), (2, 1)])
def test_mode_content_puts_hg_mn_in_order_m_plus_n(mn):
    g = make_grid()
    frac = dg.mode_content(g, pulse(g, m.hermite_gauss(W, *mn)).Aw, lambda lam: W)["order_frac"]
    assert frac[sum(mn)] == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize("p, l", [(0, 1), (1, -1), (0, -2)])
def test_lg_mode_content_finds_lg_pl(p, l):
    g = make_grid()
    out = dg.lg_mode_content(g, pulse(g, m.laguerre_gauss(W, p, l)).Aw, lambda lam: W)
    assert out["frac"][p, l + 2] == pytest.approx(1.0, abs=1e-6)


# ------------------------------------------------------------ near field, homogeneity, compression
def test_near_field_divergence_of_a_gaussian_is_lambda_over_pi_w(gauss):
    g, A = gauss
    th, Ef = dg.near_field(g, A, idx=[0], theta_y=[0.0])
    assert 2 * moments(th, np.abs(Ef[:, 0, 0]) ** 2)[1] == pytest.approx(LAM0 / (np.pi * W), rel=1e-3)


def test_homogeneity_of_a_separable_field_is_1(gauss):
    g, A = gauss
    assert dg.homogeneity(g, A)[1] == pytest.approx(1.0, abs=1e-9)


def test_homogeneity_drops_with_spatial_chirp():
    g = make_grid()
    assert dg.homogeneity(g, pulse(g, spatial_chirp_mm_per_nm=(0.01, 0.0)).Aw)[1] < 0.99


def test_compress_finds_no_gdd_for_a_transform_limited_pulse(gauss):
    g, A = gauss
    assert abs(dg.compress(g, A)["gdd_fs2"]) < 0.5


def test_compress_undoes_an_applied_gdd():
    g = make_grid()
    assert dg.compress(g, pulse(g).add_spectral_phase(gdd_fs2=1000.0).Aw)["gdd_fs2"] == pytest.approx(-1000.0, abs=1.0)


def test_compressed_pulse_is_transform_limited():
    g = make_grid()
    c = dg.compress(g, pulse(g).add_spectral_phase(gdd_fs2=1000.0).Aw)
    assert c["fwhm_fs"] == pytest.approx(c["tl_fwhm_fs"], rel=1e-3)
