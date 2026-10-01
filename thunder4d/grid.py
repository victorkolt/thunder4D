"""Simulation grid and FFT conventions.

Conventions (read this once, everything else follows)
------------------------------------------------------
* Envelope A(x, y, t) with carrier exp[i(k0 z - w0 t)], |A|^2 = intensity in W/m^2.
* Temporal transform  A~(Om) = sum_t A(t) exp(+i Om t)   -> grid.fft.ft_t
                      A(t)   = (1/Nt) sum_Om A~ exp(-i Om t) -> grid.fft.ift_t
  => a spectral phase +GDD/2 Om^2 is an up-chirp, exp(+i Om t0) delays by t0.
* Spatial transform: standard fft2 (plane wave exp(+i kx x)), propagation exp(+i kz z).
* ALL arrays are kept in FFT order (index 0 is x=0, t=0, Om=0). Operations used by
  the solver are local, so the order never matters there; use grid.xs, grid.lams,
  grid.shift_xy(), grid.shift_w() for plotting.
* The solver's home domain is (x, y, Om): complex arrays of shape (Nx, Ny, Nt).
"""
import numpy as np

C0 = 299_792_458.0


class FFT:
    """Thin wrapper so the backend can be switched (numpy default, scipy multi-thread optional)."""

    def __init__(self, backend="auto", workers=None):
        if backend == "auto":
            try:
                import scipy.fft  # noqa: F401  (pocketfft C++: ~8x faster than numpy.fft for 2D)
                backend = "scipy"
            except ImportError:
                backend = "numpy"
        if backend == "scipy":
            import scipy.fft as mod
            self._kw = {"workers": workers if workers is not None else -1}
        elif backend == "numpy":
            mod = np.fft
            self._kw = {}
        else:
            raise ValueError("backend must be 'auto', 'numpy' or 'scipy'")
        self.mod = mod
        self.backend = backend

    def fft2(self, a):
        return self.mod.fft2(a, axes=(0, 1), **self._kw)

    def ifft2(self, a):
        return self.mod.ifft2(a, axes=(0, 1), **self._kw)

    def ft_t(self, a):   # time -> Omega
        return self.mod.ifft(a, axis=-1, norm="forward", **self._kw)

    def ift_t(self, a):  # Omega -> time
        return self.mod.fft(a, axis=-1, norm="forward", **self._kw)


def _next_fast(n):
    """Smallest 2^a 3^b 5^c >= n (fast sizes for pocketfft)."""
    n = int(np.ceil(n))
    best = 2 ** int(np.ceil(np.log2(max(n, 1))))
    for a in range(0, 20):
        for b in range(0, 13):
            for c in range(0, 9):
                v = 2 ** a * 3 ** b * 5 ** c
                if n <= v < best:
                    best = v
    return best


class Grid:
    """(x, y, t) grid centred on lambda0.

    Parameters
    ----------
    Nx, dx : transverse points and step [m] (Ny, dy default to the same)
    Nt     : number of temporal points
    lambda0: carrier wavelength [m] (defines the moving frame and Omega = 0)
    dt     : time step [s]; or give lambda_min (and optionally lambda_max) [m]
             and dt is chosen so that the spectral window covers them.
    dtype  : np.complex64 (default, half the memory) or np.complex128.
    """

    def __init__(self, Nx, dx, Nt, lambda0, *, dt=None, lambda_min=None, lambda_max=None,
                 Ny=None, dy=None, dtype=np.complex64, fft_backend="auto", workers=None):
        self.Nx, self.Ny = int(Nx), int(Ny or Nx)
        self.dx, self.dy = float(dx), float(dy or dx)
        self.Nt = int(Nt)
        self.lambda0 = float(lambda0)
        self.omega0 = 2 * np.pi * C0 / lambda0
        if dt is None:
            if lambda_min is None:
                raise ValueError("give dt or lambda_min")
            w_hi = 2 * np.pi * C0 / lambda_min - self.omega0
            w_lo = self.omega0 - 2 * np.pi * C0 / lambda_max if lambda_max else w_hi
            dt = np.pi / max(w_hi, w_lo)
        self.dt = float(dt)

        self.x = np.fft.fftfreq(self.Nx, 1.0 / (self.Nx * self.dx))
        self.y = np.fft.fftfreq(self.Ny, 1.0 / (self.Ny * self.dy))
        self.kx = 2 * np.pi * np.fft.fftfreq(self.Nx, self.dx)
        self.ky = 2 * np.pi * np.fft.fftfreq(self.Ny, self.dy)
        self.t = np.fft.fftfreq(self.Nt, 1.0 / (self.Nt * self.dt))
        self.Omega = 2 * np.pi * np.fft.fftfreq(self.Nt, self.dt)
        self.omega = self.omega0 + self.Omega
        self.valid = self.omega > 0
        with np.errstate(divide="ignore"):
            self.lam = np.where(self.valid, 2 * np.pi * C0 / np.where(self.valid, self.omega, 1.0), np.inf)

        self.dtype = np.dtype(dtype)
        self.rdtype = np.float32 if self.dtype == np.complex64 else np.float64
        self.fft = FFT(fft_backend, workers)

    # ------------------------------------------------------------------ helpers
    @property
    def shape(self):
        return (self.Nx, self.Ny, self.Nt)

    @property
    def X(self):
        return self.x[:, None]

    @property
    def Y(self):
        return self.y[None, :]

    @property
    def r2(self):
        return self.x[:, None] ** 2 + self.y[None, :] ** 2

    @property
    def dA(self):
        return self.dx * self.dy

    def energy(self, Aw):
        """Energy [J] of a field in the (x, y, Omega) domain."""
        return float(np.vdot(Aw, Aw).real) * self.dx * self.dy * self.dt / self.Nt

    def energy_t(self, At):
        return float(np.vdot(At, At).real) * self.dx * self.dy * self.dt

    # sorted axes for display
    xs = property(lambda s: np.fft.fftshift(s.x))
    ys = property(lambda s: np.fft.fftshift(s.y))
    ts = property(lambda s: np.fft.fftshift(s.t))
    kxs = property(lambda s: np.fft.fftshift(s.kx))
    lams = property(lambda s: np.fft.fftshift(s.lam))
    Omegas = property(lambda s: np.fft.fftshift(s.Omega))

    @staticmethod
    def shift_xy(a):
        return np.fft.fftshift(a, axes=(0, 1))

    @staticmethod
    def shift_w(a, axis=-1):
        return np.fft.fftshift(a, axes=axis)

    def describe(self):
        mem = np.prod(self.shape) * self.dtype.itemsize / 1e6
        lv = self.lam[self.valid]
        return (f"Grid {self.Nx}x{self.Ny}x{self.Nt} ({mem:.0f} MB per field) | "
                f"box {self.Nx*self.dx*1e3:.2f} x {self.Ny*self.dy*1e3:.2f} mm, dx = {self.dx*1e6:.1f} um | "
                f"T = {self.Nt*self.dt*1e15:.0f} fs, dt = {self.dt*1e15:.2f} fs | "
                f"lambda window {lv.min()*1e9:.0f}-{lv.max()*1e9:.0f} nm")


def suggest_grid(cell, lambda0, lambda_min, lambda_max, pulse_fwhm, *, box_factor=8.0,
                 points_per_waist=8.0, time_window_factor=8.0, **grid_kw):
    """Pick (Nx, dx, Nt, dt) for a cell.

    * box = box_factor x largest mirror spot radius (increase to ~12 for top-hat / aberrated inputs,
      their higher-order modes are larger than the fundamental),
    * dx  = min(smallest waist / points_per_waist, Nyquist of the mirror curvature at the box corner),
    * Nt  from the spectral window and time_window_factor x input FWHM.
    """
    wm = cell.linear_mode(lambda_max)["w_mirror"]
    w0 = cell.linear_mode(lambda_min)["w0"]
    L = box_factor * wm
    dx_waist = w0 / points_per_waist
    dx_curv = 0.8 * lambda_min * cell.R / (np.sqrt(2) * L)
    dx = min(dx_waist, dx_curv)
    Nx = _next_fast(L / dx)
    w_hi = 2 * np.pi * C0 / lambda_min - 2 * np.pi * C0 / lambda0
    w_lo = 2 * np.pi * C0 / lambda0 - 2 * np.pi * C0 / lambda_max
    dt = np.pi / max(w_hi, w_lo)
    Nt = _next_fast(time_window_factor * pulse_fwhm / dt)
    return Grid(Nx, L / Nx, Nt, lambda0, dt=dt, **grid_kw)
