"""Fused Numba kernels (thunder4d.kernels): same propagation as the NumPy array operations."""
import numpy as np
import pytest

import thunder4d as m
from thunder4d.kernels import numba_available, resolve_kernels

from helpers import LAM0, overlap

needs_numba = pytest.mark.skipif(not numba_available(), reason="needs Numba")


# ------------------------------------------------------------ choice of kernels
def test_numpy_kernels_are_always_available():
    assert resolve_kernels("numpy", "cpu") == "numpy"


def test_auto_picks_numba_on_the_cpu_when_installed():
    assert resolve_kernels("auto", "cpu") == ("numba" if numba_available() else "numpy")


def test_gpu_grids_use_the_array_module():
    assert resolve_kernels("auto", "gpu") == "numpy"
    with pytest.raises(ValueError):
        resolve_kernels("numba", "gpu")


def test_unknown_kernels_are_rejected():
    with pytest.raises(ValueError):
        resolve_kernels("fortran", "cpu")


@pytest.mark.skipif(numba_available(), reason="Numba is installed")
def test_numba_request_without_numba_raises():
    with pytest.raises(RuntimeError):
        resolve_kernels("numba", "cpu")


# ------------------------------------------------------------ numba vs numpy
def cell():
    return m.MPC(R=0.5, d=0.94, medium=m.Material("Ar", 1.0), n_passes=2)


def kerr_pulse(g, c, energy=3e-3):
    beam = m.Beam(m.gaussian(c.linear_mode(LAM0)["w0"]), chromatic_size="eigenmode")
    return m.Pulse.at_focus(g, m.gaussian_spectrum(g, fwhm_fs=200), beam, energy)


def standard_grid(c, dtype):
    return m.suggest_grid(c, LAM0, 1000e-9, 1060e-9, 200e-15, box_factor=6, points_per_waist=3,
                          dtype=dtype, device="cpu")


def lens_grid(c, dtype):
    return m.suggest_lens_grid(c, LAM0, 1000e-9, 1060e-9, 200e-15, box_factor=8, points_per_waist=4,
                               dtype=dtype, device="cpu")


@needs_numba
def test_uppe_solver_numba_matches_numpy():
    c = cell()
    g = standard_grid(c, np.complex128)
    p = kerr_pulse(g, c)
    out, B = {}, {}
    for k in ("numpy", "numba"):
        s = m.UPPESolver(g, c.medium, kernels=k)
        assert s.kernels == k
        out[k], B[k] = s.propagate(p.Aw.copy(), 0.05), s.B
    np.testing.assert_allclose(out["numba"], out["numpy"], rtol=0, atol=1e-10 * np.abs(out["numpy"]).max())
    assert B["numba"] == pytest.approx(B["numpy"], rel=1e-12)


@needs_numba
@pytest.mark.parametrize("dtype", [np.complex128, np.complex64])
def test_lens_frame_solver_numba_matches_numpy(dtype):
    c = cell()
    g = lens_grid(c, dtype)
    p = kerr_pulse(g, c)
    out, B = {}, {}
    for k in ("numpy", "numba"):
        s = m.LensFrameSolver(g, c, kernels=k)
        A = s.propagate_zeta(p.Aw.copy(), -s.zeta_m, s.zeta_m)
        out[k], B[k] = s.propagate_linear_zeta(A, s.zeta_m, 0.0), s.B
        assert out[k].dtype == g.dtype
    tol = 1e-10 if dtype == np.complex128 else 1e-4
    np.testing.assert_allclose(out["numba"], out["numpy"], rtol=0, atol=tol * np.abs(out["numpy"]).max())
    assert B["numba"] == pytest.approx(B["numpy"], rel=tol)


@needs_numba
def test_lens_frame_simulation_numba_matches_numpy():
    c = cell()
    g = lens_grid(c, np.complex128)
    p = kerr_pulse(g, c)
    sims = {k: m.LensFrameSimulation(p, c, kernels=k, near_field=False, record_mirror=False).run(verbose=False)
            for k in ("numpy", "numba")}
    assert overlap(sims["numba"].Aw, sims["numpy"].Aw) > 1 - 1e-12
    assert sims["numba"].solver.B == pytest.approx(sims["numpy"].solver.B, rel=1e-10)


@needs_numba
def test_peak_intensity_and_nan_propagate_like_numpy():
    c = cell()
    g = lens_grid(c, np.complex128)
    s_nb, s_np = (m.LensFrameSolver(g, c, kernels=k) for k in ("numba", "numpy"))
    A = kerr_pulse(g, c).Aw
    assert s_nb.peak_intensity(A) == pytest.approx(s_np.peak_intensity(A), rel=1e-12)
    A = A.copy()
    A[1, 2, 3] = np.nan
    assert np.isnan(s_nb.peak_intensity(A))


@needs_numba
def test_propagator_cache_stays_bounded():
    c = cell()
    s = m.UPPESolver(standard_grid(c, np.complex64), c.medium, kernels="numba", cache_size=2)
    for h in (1e-3, 2e-3, 3e-3, 4e-3):
        s._propagator(h)
    assert len(s._cache) == 2


@needs_numba
def test_a_custom_nonlinear_term_disables_the_fused_rk4():
    class Doubled(m.UPPESolver):
        def nonlinear(self, Aw, track=False):
            return 2 * super().nonlinear(Aw, track)

    c = cell()
    g = standard_grid(c, np.complex128)
    assert m.UPPESolver(g, c.medium, kernels="numba")._fused_rk4
    assert not Doubled(g, c.medium, kernels="numba")._fused_rk4


def test_input_file_rejects_unknown_kernels():
    cfg = m.default_input(laser={"energy_J": 1e-3, "duration_fs": 200}, medium={"gas": "Ar", "pressure_bar": 1.0},
                          cell={"roc_m": 0.5, "separation_m": 0.94, "n_passes": 2},
                          grid={"lambda_min_nm": 1000, "lambda_max_nm": 1060}, solver={"kernels": "fortran"})
    with pytest.raises(ValueError, match="kernels"):
        m.load_input(cfg)
