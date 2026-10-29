import numpy as np
import pytest
import thunder4d as m
from helpers import LAM0, moments, vacuum

K0 = 2 * np.pi / LAM0


def uniform_pulse(g, fwhm_fs, peak_intensity):
    """Same pulse at every (x, y): only kx = ky = 0, so no diffraction and no self-focusing."""
    E = m.gaussian_spectrum(g, fwhm_fs=fwhm_fs)
    E *= np.sqrt(peak_intensity) / np.abs(g.fft.ift_t(E)).max()
    return np.broadcast_to(E, g.shape).astype(g.dtype)


def phase_fit(g, A_in, A_out):
    """Cubic fit of the spectral phase added between two fields at pixel (0, 0).
    Returns the group delay [fs] and the GDD [fs^2] at Omega = 0."""
    keep = np.abs(A_in[0, 0]) > 1e-3 * np.abs(A_in[0, 0]).max()
    o = np.argsort(g.Omega[keep])
    W = g.Omega[keep][o] * 1e-15
    phi = np.unwrap(np.angle(A_out[0, 0, keep] / A_in[0, 0, keep])[o])
    c = np.polyfit(W, phi, 3, w=np.abs(A_in[0, 0, keep])[o])
    return c[2], 2 * c[1]


# ------------------------------------------------------------ linear step
def test_diffraction_follows_the_gaussian_beam_width():
    w0 = 100e-6
    g = m.Grid(128, w0 / 8, 4, LAM0, dt=10e-15, dtype=np.complex128, device="cpu")
    A = np.repeat(m.gaussian(w0)(g.X, g.Y)[:, :, None], g.Nt, axis=2).astype(complex)
    zR = np.pi * w0 ** 2 / LAM0
    out = m.UPPESolver(g, vacuum()).linear(A, zR)
    assert 2 * moments(g.X, np.abs(out[:, :, 0]) ** 2)[1] == pytest.approx(np.sqrt(2) * w0, rel=1e-3)


def test_linear_propagation_is_reversible():
    g = m.Grid(32, 12.5e-6, 16, LAM0, dt=4e-15, dtype=np.complex128, device="cpu")
    rng = np.random.default_rng(0)
    A = rng.normal(size=g.shape) + 1j * rng.normal(size=g.shape)
    s = m.UPPESolver(g, vacuum())
    np.testing.assert_allclose(s.linear(s.linear(A, 0.1), -0.1), A, atol=1e-10)


@pytest.fixture
def through_argon():
    """20 fs pulse through 20 m of argon at 1 bar, linear propagation."""
    g = m.Grid(4, 1e-3, 512, LAM0, dt=2e-15, dtype=np.complex128, device="cpu")
    gas = m.Material("Ar", 1.0, n2=0.0)
    A = uniform_pulse(g, 20, 1.0)
    return g, gas, A, m.UPPESolver(g, gas).propagate(A.copy(), 20.0)


def test_gas_dispersion_adds_the_material_gdd(through_argon):
    g, gas, A_in, A_out = through_argon
    assert phase_fit(g, A_in, A_out)[1] == pytest.approx(gas.gvd(LAM0) * 20.0, rel=1e-3)


def test_moving_frame_removes_the_group_delay(through_argon):
    g, gas, A_in, A_out = through_argon
    assert abs(phase_fit(g, A_in, A_out)[0]) < 0.05


# ------------------------------------------------------------ Kerr step
@pytest.fixture
def spm():
    """Pure SPM: 1 rad of peak Kerr phase, no dispersion, no diffraction, no self-steepening."""
    g = m.Grid(4, 1e-3, 256, LAM0, dt=4e-15, dtype=np.complex128, device="cpu")
    n2, I0 = 1e-23, 1e16
    L = 1.0 / (K0 * n2 * I0)
    A = uniform_pulse(g, 200, I0)
    s = m.UPPESolver(g, vacuum(n2), self_steepening=False)
    return g, A, s.propagate(A.copy(), L), s


def test_kerr_phase_at_the_peak_is_k0_n2_I0_L(spm):
    g, A_in, A_out, _ = spm
    At_in, At_out = g.fft.ift_t(A_in)[0, 0], g.fft.ift_t(A_out)[0, 0]
    assert np.angle(At_out[0] / At_in[0]) == pytest.approx(1.0, rel=1e-4)


def test_kerr_leaves_the_temporal_intensity_unchanged(spm):
    g, A_in, A_out, _ = spm
    I_in, I_out = np.abs(g.fft.ift_t(A_in)[0, 0]) ** 2, np.abs(g.fft.ift_t(A_out)[0, 0]) ** 2
    np.testing.assert_allclose(I_out, I_in, atol=1e-6 * I_in.max())


def test_B_integral_counts_the_peak_kerr_phase(spm):
    assert spm[3].B == pytest.approx(1.0, rel=1e-9)


@pytest.mark.parametrize("self_steepening", [False, True])
def test_kerr_conserves_energy(self_steepening):
    g = m.Grid(4, 1e-3, 512, LAM0, dt=2e-15, dtype=np.complex128, device="cpu")
    n2, I0 = 1e-23, 1e16
    A = uniform_pulse(g, 30, I0)
    out = m.UPPESolver(g, vacuum(n2), self_steepening=self_steepening).propagate(A.copy(), 3 / (K0 * n2 * I0))
    assert g.energy(out) == pytest.approx(g.energy(A), rel=1e-6)


def test_step_size_keeps_the_kerr_phase_per_step_below_phi_max():
    g = m.Grid(4, 1e-3, 256, LAM0, dt=4e-15, dtype=np.complex128, device="cpu")
    n2, I0, phi_max = 1e-23, 1e16, 0.05
    s = m.UPPESolver(g, vacuum(n2), phi_max=phi_max)
    s.propagate(uniform_pulse(g, 200, I0), 1.0 / (K0 * n2 * I0))
    assert s.B / s.nsteps <= phi_max
    assert s.nsteps <= s.B / (phi_max * 2 ** -0.25) + 1     # steps are quantised to 2^(-1/4)


# ------------------------------------------------------------ absorbers
@pytest.fixture
def box():
    g = m.Grid(64, 20e-6, 8, LAM0, dt=5e-15, dtype=np.complex128, device="cpu")
    return g, m.UPPESolver(g, vacuum())


def test_absorber_leaves_a_centred_beam_untouched(box):
    g, s = box
    A = m.gaussian(100e-6)(g.X, g.Y)[:, :, None] * m.gaussian_spectrum(g, fwhm_fs=100)[None, None, :]
    E0 = g.energy(A)
    assert g.energy(s.absorb(A.astype(complex))) == pytest.approx(E0, rel=1e-9)


def test_absorber_removes_the_field_at_the_box_edge(box):
    g, s = box
    A = s.absorb(np.ones(g.shape, complex))
    assert abs(A[g.Nx // 2, 0, 0]) < 0.01
