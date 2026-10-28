import ctypes
import glob
import os
import site
import sys

import numpy as np

_GPU_OK = None
_GPU_ERR = None
_DLL_HANDLES = []          # keeps the added DLL directories alive


def _windows_nvidia_dlls():
    """On Windows, make the CUDA DLLs installed by pip or uv (nvidia-*-cu12 packages) findable by CuPy."""
    if sys.platform != "win32":
        return
    roots = set(getattr(site, "getsitepackages", lambda: [])())
    roots.add(site.getusersitepackages())
    roots.add(os.path.join(sys.prefix, "Lib", "site-packages"))
    for root in roots:
        nv = os.path.join(root, "nvidia")
        for d in glob.glob(os.path.join(nv, "*", "bin")):
            try:
                _DLL_HANDLES.append(os.add_dll_directory(d))
            except OSError:
                pass
        for pattern in ("cublasLt64_*.dll", "cublas64_*.dll"):   # cublasLt first: cublas depends on it
            for f in glob.glob(os.path.join(nv, "cublas", "bin", pattern)):
                try:
                    ctypes.WinDLL(f)
                except OSError:
                    pass


def gpu_available():
    """True if CuPy is installed and a working CUDA device can run an FFT, cuBLAS and a kernel."""
    global _GPU_OK, _GPU_ERR
    if _GPU_OK is None:
        try:
            _windows_nvidia_dlls()
            import cupy as cp
            import cupyx.scipy.fft as cfft
            if cp.cuda.runtime.getDeviceCount() < 1:
                raise RuntimeError("no CUDA device")
            a = cp.ones((8, 8), cp.complex64)
            cfft.fft2(a, axes=(0, 1))
            cp.vdot(a, a)                       # needs cuBLAS
            float(cp.sum(cp.abs(a) ** 2))       # compiles a kernel: fails on unsupported GPUs
            _GPU_OK = True
        except Exception as e:
            _GPU_OK = False
            _GPU_ERR = f"{type(e).__name__}: {e}"
    return _GPU_OK


def gpu_error():
    """Why the GPU is unavailable (None if it works)."""
    gpu_available()
    return _GPU_ERR


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
                               "found (uv pip install cupy-cuda12x); use device='auto' or 'cpu'")
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
