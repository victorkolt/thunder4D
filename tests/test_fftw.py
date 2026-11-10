"""fft_backend='fftw' (pyFFTW): same transforms as scipy.fft, same propagation."""
import numpy as np
import pytest

import thunder4d as m
from thunder4d.grid import FFT, fftw_available

from helpers import LAM0, overlap

needs_fftw = pytest.mark.skipif(not fftw_available(), reason="needs pyFFTW")
TRANSFORMS = ("fft2", "ifft2", "ft_t", "ift_t")


@pytest.mark.skipif(fftw_available(), reason="pyFFTW is installed")
def test_fftw_request_without_pyfftw_raises():
    with pytest.raises(RuntimeError):
        FFT("fftw")


def test_unknown_backend_is_rejected():
    with pytest.raises(ValueError):
        FFT("mkl")


@needs_fftw
@pytest.mark.parametrize("dtype", [np.complex128, np.complex64])
@pytest.mark.parametrize("name", TRANSFORMS)
def test_fftw_transforms_match_scipy(dtype, name):
    rng = np.random.default_rng(0)
    a = (rng.normal(size=(12, 10, 16)) + 1j * rng.normal(size=(12, 10, 16))).astype(dtype)
    keep = a.copy()
    ref, out = getattr(FFT("scipy"), name)(a), getattr(FFT("fftw"), name)(a)
    tol = 1e-12 if dtype == np.complex128 else 1e-5
    assert out.dtype == ref.dtype
    np.testing.assert_allclose(out, ref, rtol=0, atol=tol * np.abs(ref).max())
    np.testing.assert_array_equal(a, keep)                       # input preserved, like scipy


@needs_fftw
def test_fftw_returns_a_new_array_at_every_call():
    f = FFT("fftw")
    a = np.ones((4, 4, 8), np.complex64)
    first = f.ft_t(a)
    second = f.ft_t(2 * a)
    assert first is not second
    np.testing.assert_allclose(second, 2 * first)


@needs_fftw
def test_fftw_handles_non_contiguous_input():
    rng = np.random.default_rng(1)
    a = rng.normal(size=(8, 6, 20)) + 1j * rng.normal(size=(8, 6, 20))
    view = a[:, :, ::2]
    np.testing.assert_allclose(FFT("fftw").ift_t(view), FFT("scipy").ift_t(view), atol=1e-12)


@needs_fftw
def test_other_arrays_fall_back_to_scipy():
    f = FFT("fftw")
    E = np.exp(-np.linspace(-3, 3, 32) ** 2)                      # 1D real spectrum
    np.testing.assert_allclose(f.ift_t(E), FFT("scipy").ift_t(E))
    assert f._fftw is not None and not f._fftw.plans


@needs_fftw
def test_lens_frame_simulation_fftw_matches_scipy():
    cell = m.MPC(R=0.5, d=0.94, medium=m.Material("Ar", 1.0), n_passes=2)
    beam = m.Beam(m.gaussian(cell.linear_mode(LAM0)["w0"]), chromatic_size="eigenmode")
    out = {}
    for backend in ("scipy", "fftw"):
        g = m.suggest_lens_grid(cell, LAM0, 1000e-9, 1060e-9, 200e-15, box_factor=8, points_per_waist=4,
                                dtype=np.complex128, device="cpu", fft_backend=backend)
        assert g.fft.backend == backend
        p = m.Pulse.at_focus(g, m.gaussian_spectrum(g, fwhm_fs=200), beam, 3e-3)
        out[backend] = m.LensFrameSimulation(p, cell, near_field=False, record_mirror=False).run(verbose=False)
    assert overlap(out["fftw"].Aw, out["scipy"].Aw) > 1 - 1e-12
    assert out["fftw"].solver.B == pytest.approx(out["scipy"].solver.B, rel=1e-10)


def test_input_file_rejects_unknown_fft_backend():
    cfg = m.default_input(laser={"energy_J": 1e-3, "duration_fs": 200}, medium={"gas": "Ar", "pressure_bar": 1.0},
                          cell={"roc_m": 0.5, "separation_m": 0.94, "n_passes": 2},
                          grid={"lambda_min_nm": 1000, "lambda_max_nm": 1060, "fft_backend": "mkl"})
    with pytest.raises(ValueError, match="fft_backend"):
        m.load_input(cfg)
