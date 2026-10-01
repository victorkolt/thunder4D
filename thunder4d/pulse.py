"""Module 1: laser pulse initialisation.

A pulse is built as  A(x, y, Om) = E(Om) * U(x, y; Om)  where
  * E(Om) is a 1D complex spectral amplitude (gaussian / measured / from a retrieved pulse)
    to which any spectral phase can be added,
  * U is a normalised spatial field at each frequency, produced by a `Beam`
    (profile + Zernike aberrations + space-time couplings), either defined directly at the
    cell centre, or defined in the collimated near field and focused into the cell with an
    achromatic lens (per-frequency Fraunhofer transform: the physically correct way to get
    a top-hat near field into the cell).
"""
import numpy as np
from .grid import C0

FS = 1e-15


# ============================================================ spectral amplitude
def gaussian_spectrum(grid, fwhm_fs=None, fwhm_nm=None, center_nm=None):
    """Transform-limited gaussian. Give either the intensity FWHM duration or bandwidth."""
    lc = center_nm * 1e-9 if center_nm else grid.lambda0
    wc = 2 * np.pi * C0 / lc
    if fwhm_fs is not None:
        dw = 4 * np.log(2) / (fwhm_fs * FS)
    elif fwhm_nm is not None:
        dw = 2 * np.pi * C0 * fwhm_nm * 1e-9 / lc ** 2
    else:
        raise ValueError("give fwhm_fs or fwhm_nm")
    S = np.exp(-4 * np.log(2) * (grid.omega - wc) ** 2 / dw ** 2)
    return np.sqrt(S) * grid.valid


def measured_spectrum(grid, wavelength_nm, intensity, phase_rad=None, background=0.0):
    """Spectrum measured vs wavelength (spectrometer), with the lambda -> omega Jacobian.
    Optional spectral phase (e.g. from a d-scan retrieval) given on the same wavelength axis."""
    lam = np.asarray(wavelength_nm, float) * 1e-9
    I = np.clip(np.asarray(intensity, float) - background, 0, None)
    w = 2 * np.pi * C0 / lam
    Sw = I * lam ** 2 / (2 * np.pi * C0)
    o = np.argsort(w)
    amp = np.sqrt(np.clip(np.interp(grid.omega, w[o], Sw[o], left=0, right=0), 0, None))
    E = amp.astype(complex)
    if phase_rad is not None:
        ph = np.interp(grid.omega, w[o], np.unwrap(np.asarray(phase_rad, float)[o]), left=0, right=0)
        E = E * np.exp(1j * ph)
    return E * grid.valid


def spectrum_from_temporal_field(grid, t_fs, field):
    """Complex temporal envelope (e.g. *_retrieved_pulse.txt: t, |E|, phase) -> E(Om).
    The envelope must be relative to grid.lambda0."""
    t = np.asarray(t_fs, float) * FS
    f = np.asarray(field, complex)
    o = np.argsort(t)
    At = (np.interp(grid.t, t[o], f[o].real, left=0, right=0)
          + 1j * np.interp(grid.t, t[o], f[o].imag, left=0, right=0))
    return grid.fft.ft_t(At) * grid.valid


def add_spectral_phase(grid, E, gdd_fs2=0.0, tod_fs3=0.0, fod_fs4=0.0, delay_fs=0.0, phase=None):
    """E(Om) * exp(i phi), phi = GDD/2 Om^2 + TOD/6 Om^3 + FOD/24 Om^4 + delay*Om (+ phase(Om)).
    `phase` may be a callable of the angular frequency offset Om [rad/s] or an array on grid.Omega."""
    W = grid.Omega * FS  # rad/fs
    phi = gdd_fs2 / 2 * W ** 2 + tod_fs3 / 6 * W ** 3 + fod_fs4 / 24 * W ** 4 + delay_fs * W
    if phase is not None:
        phi = phi + (phase(grid.Omega) if callable(phase) else np.asarray(phase))
    return E * np.exp(1j * phi)


# ============================================================ spatial profiles
class Profile:
    """Callable spatial amplitude f(X, Y) with a characteristic radius (used as Zernike pupil)."""

    def __init__(self, func, radius, name):
        self.func, self.radius, self.name = func, radius, name

    def __call__(self, X, Y):
        return self.func(X, Y)


def gaussian(w):
    """exp(-r^2/w^2) field (1/e^2 intensity radius w)."""
    return Profile(lambda X, Y: np.exp(-(X ** 2 + Y ** 2) / w ** 2), w, f"gaussian w={w*1e3:.3g} mm")


def super_gaussian(w, order=10):
    """Field exp(-(r/w)^order): order 2 -> gaussian, >= 10 -> top-hat with soft edges."""
    return Profile(lambda X, Y: np.exp(-(np.sqrt(X ** 2 + Y ** 2) / w) ** order), w,
                   f"super-gaussian w={w*1e3:.3g} mm, order {order}")


def knife_edge(profile, edge, axis="x", side="+", width=None):
    """Clip a profile with a hard edge (knife edge / slit jaw) at x = edge (axis 'x' or 'y').
    side '+' blocks x > edge, '-' blocks x < edge, 'both' keeps |x| < |edge| (slit).
    width: 10-90 % edge width [m]; None -> 0 (one-pixel step on the grid where it is evaluated)."""
    def f(X, Y):
        X, Y = np.broadcast_arrays(X, Y)
        c = X if axis == "x" else Y
        if width:
            s = width / 2.2
            step = lambda u: 0.5 * (1 + np.tanh(u / s))
        else:
            step = lambda u: (u > 0).astype(float)
        if side == "+":
            T = step(edge - c)
        elif side == "-":
            T = step(c - edge)
        else:
            T = step(abs(edge) - np.abs(c))
        return profile(X, Y) * T
    return Profile(f, profile.radius, f"{profile.name}, knife edge {axis}{side} at {edge*1e6:.0f} um")


def top_hat(radius, order=24):
    return super_gaussian(radius, order)


def measured_profile(image, pixel_size, center="centroid", background=0.0):
    """Camera near-field image (rows = y, cols = x) -> amplitude profile (sqrt of intensity).
    Bilinear interpolation, zero outside the image."""
    img = np.clip(np.asarray(image, float) - background, 0, None)
    ny, nx = img.shape
    xs = (np.arange(nx) - (nx - 1) / 2) * pixel_size
    ys = (np.arange(ny) - (ny - 1) / 2) * pixel_size
    if center == "centroid":
        s = img.sum()
        cx, cy = (img.sum(0) @ xs) / s, (img.sum(1) @ ys) / s
    else:
        cx, cy = center
    amp = np.sqrt(img)
    rad = 2 * np.sqrt(((img.sum(0) * (xs - cx) ** 2).sum()) / img.sum())

    def f(X, Y):
        X, Y = np.broadcast_arrays(X, Y)
        fi = (X + cx - xs[0]) / pixel_size
        fj = (Y + cy - ys[0]) / pixel_size
        i0 = np.floor(fi).astype(int)
        j0 = np.floor(fj).astype(int)
        a, b = fi - i0, fj - j0
        ok = (i0 >= 0) & (i0 < nx - 1) & (j0 >= 0) & (j0 < ny - 1)
        i0c, j0c = np.clip(i0, 0, nx - 2), np.clip(j0, 0, ny - 2)
        v = ((1 - a) * (1 - b) * amp[j0c, i0c] + a * (1 - b) * amp[j0c, i0c + 1]
             + (1 - a) * b * amp[j0c + 1, i0c] + a * b * amp[j0c + 1, i0c + 1])
        return np.where(ok, v, 0.0)

    return Profile(f, rad, "measured")


# ============================================================ aberrations
def _zernike(name, rho, th):
    Z = {
        "piston": lambda: np.ones_like(rho),
        "tilt_x": lambda: 2 * rho * np.cos(th),
        "tilt_y": lambda: 2 * rho * np.sin(th),
        "defocus": lambda: np.sqrt(3) * (2 * rho ** 2 - 1),
        "astig_0": lambda: np.sqrt(6) * rho ** 2 * np.cos(2 * th),
        "astig_45": lambda: np.sqrt(6) * rho ** 2 * np.sin(2 * th),
        "coma_x": lambda: np.sqrt(8) * (3 * rho ** 3 - 2 * rho) * np.cos(th),
        "coma_y": lambda: np.sqrt(8) * (3 * rho ** 3 - 2 * rho) * np.sin(th),
        "trefoil_0": lambda: np.sqrt(8) * rho ** 3 * np.cos(3 * th),
        "trefoil_30": lambda: np.sqrt(8) * rho ** 3 * np.sin(3 * th),
        "spherical": lambda: np.sqrt(5) * (6 * rho ** 4 - 6 * rho ** 2 + 1),
    }
    return Z[name]()


def zernike_wavefront(X, Y, coeffs_waves, pupil_radius, lambda0):
    """Wavefront error [m] from RMS-normalised Zernike coefficients given in waves at lambda0.
    The phase is then (omega/c) * W, i.e. achromatic path error (correct for all frequencies)."""
    X, Y = np.broadcast_arrays(X, Y)
    rho = np.sqrt(X ** 2 + Y ** 2) / pupil_radius
    th = np.arctan2(Y, X)
    W = np.zeros(X.shape)
    for name, c in coeffs_waves.items():
        W += c * _zernike(name, rho, th)
    return W * lambda0


# ============================================================ beam = profile + aberrations + STC
class Beam:
    """Spatial part of the pulse, possibly frequency dependent.

    profile                     : Profile (gaussian, top_hat, measured_profile, ...)
    zernike                     : dict {"astig_0": 0.1, "coma_x": 0.05, ...} RMS waves @ lambda0
    pupil_radius                : Zernike normalisation radius [m] (default profile.radius)
    spatial_chirp_mm_per_nm     : (sx, sy) transverse shift of each colour  x0(l) = s (l - l0)
    angular_dispersion_urad_per_nm : (ax, ay) propagation angle vs wavelength
    pft_fs_per_mm               : (px, py) pulse-front tilt, delay = p.x
    pfc_fs_per_mm2              : pulse-front curvature, delay = a.r^2 (lens chromatism)
    radial_gdd_fs2_per_mm2      : GDD varying as r^2 (radial chirp)
    tilt_urad, offset_m         : pointing / position
    custom_phase                : callable (X, Y, Om) -> phase [rad], anything else
    """

    def __init__(self, profile, zernike=None, pupil_radius=None, spatial_chirp_mm_per_nm=(0.0, 0.0),
                 angular_dispersion_urad_per_nm=(0.0, 0.0), pft_fs_per_mm=(0.0, 0.0), pfc_fs_per_mm2=0.0,
                 radial_gdd_fs2_per_mm2=0.0, tilt_urad=(0.0, 0.0), offset_m=(0.0, 0.0), custom_phase=None):
        self.profile = profile
        self.zernike = zernike or {}
        self.pupil = pupil_radius or profile.radius
        self.sc = spatial_chirp_mm_per_nm
        self.ad = angular_dispersion_urad_per_nm
        self.pft = pft_fs_per_mm
        self.pfc = pfc_fs_per_mm2
        self.rgdd = radial_gdd_fs2_per_mm2
        self.tilt = tilt_urad
        self.off = offset_m
        self.custom = custom_phase
        self._W = None

    def field(self, X, Y, omega, omega0):
        """Complex field at one angular frequency (not normalised)."""
        lam, lam0 = 2 * np.pi * C0 / omega, 2 * np.pi * C0 / omega0
        dl_nm = (lam - lam0) * 1e9
        Om = (omega - omega0) * FS          # rad/fs
        k = omega / C0
        sx, sy = self.sc[0] * 1e-3 * dl_nm, self.sc[1] * 1e-3 * dl_nm
        U = self.profile(X - self.off[0] - sx, Y - self.off[1] - sy).astype(complex)
        Xmm, Ymm = X * 1e3, Y * 1e3
        r2mm = Xmm ** 2 + Ymm ** 2
        ph = (Om * (self.pft[0] * Xmm + self.pft[1] * Ymm)
              + k * 1e-6 * ((self.ad[0] * dl_nm + self.tilt[0]) * X + (self.ad[1] * dl_nm + self.tilt[1]) * Y)
              + Om * self.pfc * r2mm + 0.5 * self.rgdd * r2mm * Om ** 2)
        if self.zernike:
            if self._W is None or self._W[0] is not X:
                self._W = (X, zernike_wavefront(X, Y, self.zernike, self.pupil, lam0))
            ph = ph + k * self._W[1]
        if self.custom is not None:
            ph = ph + self.custom(X, Y, omega - omega0)
        return U * np.exp(1j * ph)


# ============================================================ the pulse object
class Pulse:
    """Field A(x, y, Om) on a Grid (home domain of the solver)."""

    def __init__(self, grid, Aw):
        self.grid = grid
        self.Aw = np.asarray(Aw, dtype=grid.dtype)

    # ---------------------------------------------------------------- builders
    @staticmethod
    def _support(E, rel=1e-8):
        S = np.abs(E) ** 2
        return np.nonzero(S > rel * S.max())[0]

    @classmethod
    def at_focus(cls, grid, E, beam, energy):
        """Beam defined directly in the cell-centre plane (e.g. an ideal gaussian of waist w0)."""
        Aw = np.zeros(grid.shape, grid.dtype)
        X, Y = grid.X, grid.Y
        for j in cls._support(E):
            U = beam.field(X, Y, grid.omega[j], grid.omega0)
            p = np.sum(np.abs(U) ** 2) * grid.dA
            Aw[:, :, j] = E[j] * U / np.sqrt(p)
        return cls(grid, Aw).set_energy(energy)

    @classmethod
    def from_near_field(cls, grid, E, beam, energy, nf_size, nf_points=256, focal_length=None,
                        target_waist=None, verbose=True):
        """(Fourier-plane injection: see imaged_near_field for a near field imaged on the mirror.)
        Beam defined in the collimated near field and focused into the cell centre by an
        achromatic lens of focal length f (near field in the front focal plane -> exact Fourier
        transform, flat phase at focus). Every colour gets its own far field (spot size ~ lambda).

        If focal_length is None, f is chosen to maximise the overlap at lambda0 of the focal
        field with a gaussian of waist `target_waist` (e.g. the nonlinearly matched waist)."""
        Xn = ((np.arange(nf_points) - nf_points / 2) * nf_size / nf_points)
        dX = Xn[1] - Xn[0]
        XN, YN = Xn[:, None], Xn[None, :]
        if focal_length is None:
            if target_waist is None:
                raise ValueError("give focal_length or target_waist")
            focal_length = cls._best_focal(grid, beam, XN, YN, Xn, dX, target_waist)
            if verbose:
                print(f"[Pulse] matching lens f = {focal_length:.3f} m for waist {target_waist*1e6:.0f} um")
        Aw = np.zeros(grid.shape, grid.dtype)
        worst = 1.0
        for j in cls._support(E):
            w = grid.omega[j]
            Unf = beam.field(XN, YN, w, grid.omega0)
            Unf /= np.sqrt(np.sum(np.abs(Unf) ** 2) * dX ** 2)
            U = cls._mft(Unf, Xn, dX, grid.x, grid.y, w, focal_length)
            p = np.sum(np.abs(U) ** 2) * grid.dA
            worst = min(worst, p)
            Aw[:, :, j] = E[j] * U / np.sqrt(p)
        if verbose and worst < 0.99:
            print(f"[Pulse] WARNING: only {worst*100:.1f}% of the focused power fits in the box")
        p = cls(grid, Aw).set_energy(energy)
        p.focal_length = focal_length
        return p

    @classmethod
    def imaged_near_field(cls, grid, E, beam, energy, cell, image_z=None, curvature_radius=None,
                          verbose=True):
        """Near field IMAGED (achromatic relay, same size for every colour) onto the plane at
        distance image_z [m] from the cell centre, on the input side (default: the mirror plane,
        d/2), with a spherical wavefront matching the cell mode there (default: diverging with
        R(z) = z + zR^2/z of the linear eigenmode, = mirror ROC at the mirror plane).
        The field is then propagated back (linear, exact) to the cell centre, where the
        simulation starts. The first mirror of pass 1 then sees the top-hat image.

        Size the profile to the mode: for a top-hat, radius ~ 1.12 x mode radius at that plane
        maximises the overlap with the gaussian (eta = 0.815)."""
        from .solver import UPPESolver
        z = cell.d / 2 if image_z is None else float(image_z)
        if curvature_radius is None:
            zR = cell.linear_mode(grid.lambda0)["zR"]
            curvature_radius = z + zR ** 2 / z
        X, Y = grid.X, grid.Y
        r2 = grid.r2
        Aw = np.zeros(grid.shape, grid.dtype)
        for j in cls._support(E):
            w = grid.omega[j]
            U = beam.field(X, Y, w, grid.omega0) * np.exp(1j * (w / C0) * r2 / (2 * curvature_radius))
            U /= np.sqrt(np.sum(np.abs(U) ** 2) * grid.dA)
            Aw[:, :, j] = E[j] * U
        lin = UPPESolver(grid, cell.medium)
        lin._cache.clear()
        Aw = lin.linear(Aw, -z)
        del lin
        p = cls(grid, Aw)
        frac = grid.energy(p.Aw)
        p.set_energy(energy)
        if verbose:
            print(f"[Pulse] near field imaged at z = {z*100:.1f} cm from the centre, "
                  f"wavefront R = {curvature_radius:.3f} m")
        return p

    @staticmethod
    def _mft(Unf, Xn, dX, x, y, omega, f):
        """Fraunhofer field at the focus of lens f (matrix Fourier transform, exact scaling)."""
        k = omega / C0
        Fx = np.exp(-1j * k * np.outer(x, Xn) / f)
        Fy = np.exp(-1j * k * np.outer(y, Xn) / f)
        lam = 2 * np.pi / k
        return (Fx @ Unf @ Fy.T) * dX ** 2 / (1j * lam * f)

    @classmethod
    def _best_focal(cls, grid, beam, XN, YN, Xn, dX, wt):
        Unf = beam.field(XN, YN, grid.omega0, grid.omega0)
        I = np.abs(Unf) ** 2
        W = 2 * np.sqrt(np.sum(I * (XN - np.sum(I * XN) / I.sum()) ** 2) / I.sum())
        f0 = np.pi * W * wt / grid.lambda0
        G = np.exp(-grid.r2 / wt ** 2)
        G /= np.sqrt(np.sum(G ** 2) * grid.dA)

        def eta(lf):
            U = cls._mft(Unf, Xn, dX, grid.x, grid.y, grid.omega0, np.exp(lf))
            return abs(np.sum(G * U)) ** 2 / (np.sum(np.abs(U) ** 2) * np.sum(G ** 2))

        a, b = np.log(f0) - 1.2, np.log(f0) + 1.2
        gr = (np.sqrt(5) - 1) / 2
        c, d = b - gr * (b - a), a + gr * (b - a)
        for _ in range(40):
            if eta(c) > eta(d):
                b = d
            else:
                a = c
            c, d = b - gr * (b - a), a + gr * (b - a)
        return float(np.exp((a + b) / 2))

    # ---------------------------------------------------------------- utilities
    def set_energy(self, energy):
        self.Aw *= np.sqrt(energy / self.grid.energy(self.Aw))
        return self

    @property
    def energy(self):
        return self.grid.energy(self.Aw)

    def At(self):
        return self.grid.fft.ift_t(self.Aw)

    def power(self):
        """Spatially integrated power P(t) [W], FFT order."""
        At = self.At()
        return np.sum(np.abs(At) ** 2, axis=(0, 1)) * self.grid.dA

    def peak_power(self):
        return float(self.power().max())

    def weighted_power(self):
        """Intensity-weighted mean power  int P^2 dt / int P dt  (= Ppeak/sqrt2 for a gaussian)."""
        P = self.power()
        return float(np.sum(P ** 2) / np.sum(P))

    def add_spectral_phase(self, **kw):
        E1 = add_spectral_phase(self.grid, np.ones(self.grid.Nt, complex), **kw)
        self.Aw *= E1.astype(self.grid.dtype)[None, None, :]
        return self

    def copy(self):
        return Pulse(self.grid, self.Aw.copy())
