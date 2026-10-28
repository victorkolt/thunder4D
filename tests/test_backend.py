import numpy as np
import pytest
import thunder4d as m
from thunder4d.backend import resolve_device, to_host
from helpers import LAM0

needs_gpu = pytest.mark.skipif(not m.gpu_available(), reason="needs CuPy and a CUDA GPU")


def test_cpu_is_always_available():
    assert resolve_device("cpu") == "cpu"


def test_unknown_device_is_rejected():
    with pytest.raises(ValueError):
        resolve_device("tpu")


@pytest.mark.skipif(m.gpu_available(), reason="a GPU is available")
def test_gpu_request_without_a_gpu_raises():
    with pytest.raises(RuntimeError):
        resolve_device("gpu")


def test_to_host_returns_numpy_arrays_as_they_are():
    a = np.zeros(3)
    assert to_host(a) is a


@needs_gpu
def test_gpu_propagation_matches_the_cpu():
    out = {}
    for device in ("cpu", "gpu"):
        g = m.Grid(32, 25e-6, 32, LAM0, dt=4e-15, dtype=np.complex128, device=device)
        p = m.Pulse.at_focus(g, m.gaussian_spectrum(g, fwhm_fs=50), m.Beam(m.gaussian(150e-6)), 0.2e-3)
        out[device] = to_host(m.UPPESolver(g, m.Material("Ar", 1.0)).propagate(p.Aw, 0.2))
    np.testing.assert_allclose(out["gpu"], out["cpu"], rtol=1e-9, atol=1e-9 * np.abs(out["cpu"]).max())
