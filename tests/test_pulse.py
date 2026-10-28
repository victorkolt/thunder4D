import numpy as np
import pytest
import thunder4d as m
from helpers import LAM0, moments, overlap, vacuum

FWHM_PER_SIGMA = 2 * np.sqrt(2 * np.log(2))   # gaussian intensity
W = 60e-6                                     # beam radius for the profile tests


def make_grid(Nx=32, dx=20e-6, Nt=128, dt=4e-15):
    return m.Grid(Nx, dx, Nt, LAM0, dt=dt, dtype=np.complex128, device="cpu")


def temporal_intensity(g, E):
    return np.abs(g.fft.ift_t(E)) ** 2


def fine_xy(w, n=257, half=6):
    x = np.linspace(-half * w, half * w, n)
    return x[:, None], x[None, :], (x[1] - x[0]) ** 2


def gaussian_pulse(g, w=100e-6, fwhm_fs=50, energy=1e-3, **beam_kw):
    E = m.gaussian_spectrum(g, fwhm_fs=fwhm_fs)
    return m.Pulse.at_focus(g, E, m.Beam(m.gaussian(w), **beam_kw), energy)


# ------------------------------------------------------------ spectral amplitude and phase
def test_gaussian_spectrum_sets_the_intensity_fwhm():
    g = make_grid()
    _, sigma = moments(g.t, temporal_intensity(g, m.gaussian_spectrum(g, fwhm_fs=50)))
    assert FWHM_PER_SIGMA * sigma == pytest.approx(50e-15, rel=1e-3)


def test_gaussian_spectrum_sets_the_bandwidth_in_nm():
    g = make_grid(Nx=4, Nt=1024)
    _, sigma = moments(g.Omega, np.abs(m.gaussian_spectrum(g, fwhm_nm=10)) ** 2)
    assert FWHM_PER_SIGMA * sigma == pytest.approx(2 * np.pi * m.C0 * 10e-9 / LAM0 ** 2, rel=1e-3)


def test_positive_delay_moves_the_pulse_later():
    g = make_grid()
    E = m.add_spectral_phase(g, m.gaussian_spectrum(g, fwhm_fs=30), delay_fs=40)
    mean, _ = moments(g.t, temporal_intensity(g, E))
    assert mean == pytest.approx(40e-15, rel=1e-3)


def test_positive_gdd_is_an_up_chirp():
    g = make_grid(Nt=512)
    E = m.add_spectral_phase(g, m.gaussian_spectrum(g, fwhm_fs=30), gdd_fs2=2000)
    t, A = g.ts, g.shift_w(g.fft.ift_t(E))
    keep = np.abs(A) ** 2 > 0.1 * np.max(np.abs(A) ** 2)
    inst = -np.gradient(np.unwrap(np.angle(A[keep])), t[keep])   # carrier exp(-i omega0 t)
    assert np.polyfit(t[keep], inst, 1)[0] > 0


def test_gdd_stretches_a_gaussian_as_expected():
    g = make_grid(Nt=512)
    E = m.add_spectral_phase(g, m.gaussian_spectrum(g, fwhm_fs=30), gdd_fs2=2000)
    _, sigma = moments(g.t, temporal_intensity(g, E))
    expected = 30 * np.sqrt(1 + (4 * np.log(2) * 2000 / 30 ** 2) ** 2)
    assert FWHM_PER_SIGMA * sigma == pytest.approx(expected * 1e-15, rel=1e-3)


def test_spectrum_from_temporal_field_round_trip():
    g = make_grid()
    t = g.ts
    field = np.exp(-(t / 30e-15) ** 2) * np.exp(1j * (t / 40e-15) ** 2)
    E = m.spectrum_from_temporal_field(g, t / 1e-15, field)
    np.testing.assert_allclose(g.shift_w(g.fft.ift_t(E)), field, atol=1e-12)


def test_measured_spectrum_round_trip():
    g = make_grid(Nx=4, Nt=1024)
    E0 = m.gaussian_spectrum(g, fwhm_nm=10)
    lam = g.lam[g.valid]
    S_lam = np.abs(E0[g.valid]) ** 2 * 2 * np.pi * m.C0 / lam ** 2     # what a spectrometer reads
    E = m.measured_spectrum(g, lam * 1e9, S_lam)
    np.testing.assert_allclose(np.abs(E), np.abs(E0), atol=1e-12)


# ------------------------------------------------------------ Pulse.at_focus
def test_at_focus_sets_the_energy():
    assert gaussian_pulse(make_grid()).energy == pytest.approx(1e-3, rel=1e-12)


def test_gaussian_peak_power():
    p = gaussian_pulse(make_grid(), fwhm_fs=50, energy=1e-3)
    assert p.peak_power() == pytest.approx(2 * np.sqrt(np.log(2) / np.pi) * 1e-3 / 50e-15, rel=1e-3)


def test_weighted_power_of_a_gaussian_is_peak_over_sqrt2():
    p = gaussian_pulse(make_grid())
    assert p.weighted_power() == pytest.approx(p.peak_power() / np.sqrt(2), rel=1e-3)


def test_gaussian_w_is_the_1_over_e2_intensity_radius():
    g = make_grid()
    F = np.sum(np.abs(gaussian_pulse(g, w=100e-6).Aw) ** 2, axis=2)
    assert 2 * moments(g.X, F)[1] == pytest.approx(100e-6, rel=1e-3)


# ------------------------------------------------------------ spatial profiles
@pytest.mark.parametrize("mn", [(0, 0), (1, 0), (2, 1), (3, 3)])
def test_hermite_gauss_has_unit_power(mn):
    X, Y, dA = fine_xy(W)
    assert np.sum(np.abs(m.hermite_gauss(W, *mn)(X, Y)) ** 2) * dA == pytest.approx(1, rel=1e-6)


@pytest.mark.parametrize("a, b", [((0, 0), (1, 0)), ((1, 0), (0, 1)), ((2, 0), (0, 0))])
def test_hermite_gauss_modes_are_orthogonal(a, b):
    X, Y, dA = fine_xy(W)
    u, v = m.hermite_gauss(W, *a)(X, Y), m.hermite_gauss(W, *b)(X, Y)
    assert abs(np.sum(np.conj(u) * v) * dA) < 1e-9


@pytest.mark.parametrize("p, l, kind", [(0, 0, "exp"), (0, 2, "exp"), (1, -1, "exp"), (2, 1, "cos"), (1, 3, "sin")])
def test_laguerre_gauss_has_unit_power(p, l, kind):
    X, Y, dA = fine_xy(W)
    u = m.laguerre_gauss(W, p, l, kind)(X, Y)
    assert np.sum(np.abs(u) ** 2) * dA == pytest.approx(1, rel=1e-6)


@pytest.mark.parametrize("a, b", [((0, 1), (0, -1)), ((1, 0), (0, 0)), ((1, 1), (0, 1))])
def test_laguerre_gauss_modes_are_orthogonal(a, b):
    X, Y, dA = fine_xy(W)
    u, v = m.laguerre_gauss(W, *a)(X, Y), m.laguerre_gauss(W, *b)(X, Y)
    assert abs(np.sum(np.conj(u) * v) * dA) < 1e-9


@pytest.mark.parametrize("mode", [m.hermite_gauss(W, 0, 0), m.laguerre_gauss(W)], ids=["HG00", "LG00"])
def test_lowest_order_mode_is_the_gaussian(mode):
    X, Y, dA = fine_xy(W)
    g = m.gaussian(W)(X, Y)
    np.testing.assert_allclose(mode(X, Y), g / np.sqrt(np.sum(g ** 2) * dA), atol=1e-9)


def test_super_gaussian_of_order_2_is_the_gaussian():
    X, Y, _ = fine_xy(W)
    np.testing.assert_allclose(m.super_gaussian(W, 2)(X, Y), m.gaussian(W)(X, Y))


@pytest.mark.parametrize("side, kept", [("+", lambda x: x < 0.3 * W), ("-", lambda x: x > 0.3 * W),
                                        ("both", lambda x: np.abs(x) < 0.3 * W)])
def test_knife_edge_blocks_the_right_side(side, kept):
    X, Y, _ = fine_xy(W)
    clipped = m.knife_edge(m.gaussian(W), 0.3 * W, side=side)(X, Y)
    np.testing.assert_allclose(clipped, m.gaussian(W)(X, Y) * kept(X))


@pytest.fixture
def camera():
    """Image of a gaussian spot at x = +50 um (columns), y = 0 (rows)."""
    px = 5e-6
    xs = (np.arange(101) - 50) * px
    ys = (np.arange(81) - 40) * px
    return px, np.exp(-2 * ((xs[None, :] - 50e-6) ** 2 + ys[:, None] ** 2) / 40e-6 ** 2)


def test_measured_profile_reads_columns_as_x(camera):
    px, img = camera
    prof = m.measured_profile(img, px, center=(0.0, 0.0))
    assert prof(50e-6, 0.0) == pytest.approx(1.0)
    assert prof(0.0, 50e-6) < 0.1


def test_measured_profile_centres_on_the_centroid(camera):
    px, img = camera
    assert m.measured_profile(img, px)(0.0, 0.0) == pytest.approx(1.0, rel=1e-3)


@pytest.mark.parametrize("name", ["tilt_x", "tilt_y", "defocus", "astig_0", "astig_45", "coma_x", "coma_y",
                                  "trefoil_0", "trefoil_30", "spherical"])
def test_zernike_coefficient_is_the_rms_in_waves(name):
    x = np.linspace(-1, 1, 801)
    X, Y = np.meshgrid(x, x, indexing="ij")
    Wf = m.zernike_wavefront(X, Y, {name: 0.1}, pupil_radius=1.0, lambda0=1e-6)
    assert np.sqrt(np.mean(Wf[X ** 2 + Y ** 2 <= 1] ** 2)) == pytest.approx(0.1e-6, rel=1e-2)


# ------------------------------------------------------------ Beam conventions
def beam_intensity(beam, g, lam=LAM0):
    return np.abs(beam.field(g.X, g.Y, 2 * np.pi * m.C0 / lam, g.omega0)) ** 2


def test_offset_moves_the_beam():
    g = make_grid(Nx=64, dx=10e-6)
    I = beam_intensity(m.Beam(m.gaussian(W), offset_m=(50e-6, -30e-6)), g)
    assert moments(g.X, I)[0] == pytest.approx(50e-6, rel=1e-6)
    assert moments(g.Y, I)[0] == pytest.approx(-30e-6, rel=1e-6)


def test_tilt_sets_the_propagation_angle():
    g = make_grid(Nx=64, dx=10e-6)
    U = m.Beam(m.gaussian(W), tilt_urad=(2000.0, 0.0)).field(g.X, g.Y, g.omega0, g.omega0)
    kx_mean, _ = moments(g.kx[:, None], np.abs(np.fft.fft2(U)) ** 2)
    assert kx_mean / (g.omega0 / m.C0) == pytest.approx(2e-3, rel=1e-3)


def test_pulse_front_tilt_delays_positive_x():
    g = make_grid(Nx=64, dx=10e-6)
    At = g.fft.ift_t(gaussian_pulse(g, fwhm_fs=30, pft_fs_per_mm=(200.0, 0.0)).Aw)
    i = int(np.argmin(np.abs(g.x - 100e-6)))
    t_mean, _ = moments(g.t, np.abs(At[i, 0, :]) ** 2)
    assert t_mean == pytest.approx(200 * 0.1 * 1e-15, rel=1e-3)


def test_spatial_chirp_moves_each_colour_along_x():
    g = make_grid(Nx=64, dx=10e-6)
    I = beam_intensity(m.Beam(m.gaussian(W), spatial_chirp_mm_per_nm=(0.005, 0.0)), g, lam=1050e-9)
    assert moments(g.X, I)[0] == pytest.approx(0.005e-3 * 20, rel=1e-6)


@pytest.mark.parametrize("kind, exponent", [(None, 0.0), ("eigenmode", 0.5), ("focus", 1.0)])
def test_chromatic_size_scales_with_wavelength(kind, exponent):
    g = make_grid(Nx=64, dx=10e-6)
    I = beam_intensity(m.Beam(m.gaussian(W), chromatic_size=kind), g, lam=1100e-9)
    assert 2 * moments(g.X, I)[1] == pytest.approx(W * (1100e-9 / LAM0) ** exponent, rel=1e-4)


def test_unknown_chromatic_size_is_rejected():
    with pytest.raises(ValueError):
        m.Beam(m.gaussian(W), chromatic_size="mode")


# ------------------------------------------------------------ near-field injection
F, W_NF = 0.5, 1.64e-3        # lens and near-field radius: focal waist ~ 100 um


def focus_grid():
    return make_grid(Nx=64, dx=12.5e-6, Nt=16)


def test_gaussian_near_field_focuses_to_lambda_f_over_pi_w():
    g = focus_grid()
    p = m.Pulse.from_near_field(g, m.gaussian_spectrum(g, fwhm_fs=100), m.Beam(m.gaussian(W_NF)), 1e-3,
                                nf_size=8 * W_NF, nf_points=128, focal_length=F, verbose=False)
    I = np.abs(p.Aw[:, :, 0]) ** 2
    assert 2 * moments(g.X, I)[1] == pytest.approx(LAM0 * F / (np.pi * W_NF), rel=1e-2)


def test_best_focal_length_matches_the_target_waist():
    g = focus_grid()
    p = m.Pulse.from_near_field(g, m.gaussian_spectrum(g, fwhm_fs=100), m.Beam(m.gaussian(W_NF)), 1e-3,
                                nf_size=8 * W_NF, nf_points=128, target_waist=100e-6, verbose=False)
    assert p.focal_length == pytest.approx(np.pi * W_NF * 100e-6 / LAM0, rel=1e-2)


def test_from_near_field_needs_a_focal_length_or_a_target_waist():
    g = focus_grid()
    with pytest.raises(ValueError):
        m.Pulse.from_near_field(g, m.gaussian_spectrum(g, fwhm_fs=100), m.Beam(m.gaussian(W_NF)), 1e-3,
                                nf_size=8 * W_NF)


def test_imaged_near_field_is_the_profile_at_the_mirror_plane():
    cell = m.MPC.herriott(R=0.5, N=4, k=1, branch="planar", medium=vacuum())
    g = make_grid(Nx=64, dx=25e-6, Nt=16)
    beam = m.Beam(m.top_hat(1.12 * cell.linear_mode(LAM0)["w_mirror"]))
    p = m.Pulse.imaged_near_field(g, m.gaussian_spectrum(g, fwhm_fs=100), beam, 1e-3, cell, verbose=False)
    A = m.UPPESolver(g, cell.medium).linear(p.Aw, cell.d / 2)[:, :, 0]
    assert overlap(np.abs(A), beam.profile(g.X, g.Y)) > 0.9999
