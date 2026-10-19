"""Array backend: NumPy on the CPU, CuPy on an NVIDIA GPU.

Only the heavy arrays live on the device: the field A(x, y, Om), the propagators, the absorbers and
the mirror transfer function. Axes (grid.x, grid.lam, ...) and everything returned by
`thunder4d.diagnostics` are always NumPy arrays; `to_host` converts a device array.
"""
import numpy as np

_GPU_OK = None


def gpu_available():
    """True if CuPy is installed and a working CUDA device can run a kernel and an FFT."""
    global _GPU_OK
    if _GPU_OK is None:
        try:
            import cupy as cp
            import cupyx.scipy.fft as cfft
            if cp.cuda.runtime.getDeviceCount() < 1:
                raise RuntimeError("no CUDA device")
            a = cp.ones((8, 8), cp.complex64)
            cfft.fft2(a, axes=(0, 1))
            float(cp.sum(cp.abs(a) ** 2))       # compiles a kernel: fails on unsupported GPUs
            _GPU_OK = True
        except Exception:
            _GPU_OK = False
    return _GPU_OK


def resolve_device(device="auto"):
    """'auto' -> 'gpu' if CuPy works else 'cpu'; 'cpu' / 'gpu' are forced."""
    device = "cpu" if device is None else str(device).lower()
    if device == "auto":
        return "gpu" if gpu_available() else "cpu"
    if device == "cpu":
        return "cpu"
    if device in ("gpu", "cuda"):
        if not gpu_available():
            raise RuntimeError("device='gpu' requested but CuPy with a working CUDA device was not "
                               "found (pip install cupy-cuda12x); use device='auto' or 'cpu'")
        return "gpu"
    raise ValueError("device must be 'auto', 'cpu' or 'gpu'")


def get_xp(device):
    """Array module for a resolved device name."""
    if device == "gpu":
        import cupy
        return cupy
    return np


def is_device_array(a):
    return type(a).__module__.split(".")[0] == "cupy"


def xp_of(a):
    """Array module that owns `a` (numpy for anything that is not a CuPy array)."""
    return get_xp("gpu") if is_device_array(a) else np


def to_host(a):
    """NumPy view/copy of an array that may live on the GPU (no copy for NumPy input)."""
    return a.get() if is_device_array(a) else np.asarray(a)
