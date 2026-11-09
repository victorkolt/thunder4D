"""Fused CPU kernels (Numba) for the 4D solvers (UPPESolver, LensFrameSolver).

On the CPU, the FFTs run on all cores (scipy.fft, workers=-1) but the NumPy array operations
between them run on one core and allocate temporaries: in the RK4 Kerr sub-step they cost as much
as the FFTs. These kernels do the same operations in one multithreaded pass over memory each.
They never replace an FFT. They are used when the grid lives on the CPU and Numba is installed
(`pip install "thunder4d[numba]"`), unless the solver is built with kernels="numpy".

Kernels compile on first use (a few seconds, then cached on disk). The thread count is Numba's
(NUMBA_NUM_THREADS, default: all cores). No fastmath: NaN/inf must propagate as in NumPy.
"""
import importlib.util

_HAS_NUMBA = importlib.util.find_spec("numba") is not None

KERNELS = ("auto", "numba", "numpy")


def numba_available():
    """True if Numba is installed (fused CPU kernels usable)."""
    return _HAS_NUMBA


def resolve_kernels(kernels, device):
    """'auto' -> 'numba' on the CPU if Numba is installed, else 'numpy' (also on the GPU, where
    the array module is CuPy). 'numba' is forced (error if unavailable or on the GPU)."""
    k = "auto" if kernels is None else str(kernels).lower()
    if k not in KERNELS:
        raise ValueError(f"kernels must be one of {KERNELS}")
    if k == "numpy":
        return "numpy"
    if device != "cpu":
        if k == "numba":
            raise ValueError("kernels='numba' runs on the CPU only (the grid is on the GPU)")
        return "numpy"
    if not _HAS_NUMBA:
        if k == "numba":
            raise RuntimeError("kernels='numba' requested but Numba is not installed "
                               "(uv pip install numba, or thunder4d[numba])")
        return "numpy"
    return "numba"


def load():
    """The module of jitted kernels (imports Numba on first call)."""
    from . import _numba_kernels
    return _numba_kernels
