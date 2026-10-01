"""Diagnostics on a field A(x, y, Om) (FFT order). Outputs are returned SORTED (ascending axes).

near field  = the plane where the field is (cell centre when recorded by MPCSimulation)
far field   = angular spectrum on a common angle grid theta (what you see after collimating the
              output with a lens of focal f: x_out = f * theta). Computed with an exact per-colour
              matrix Fourier transform, so a given theta means the same angle for all colours.
"""
import numpy as np
from .grid import C0

FS = 1e-15


# ------------------------------------------------------------------ basic quantities
def spectral_support(grid, Aw, rel=1e-4):
    S = np.sum(np.abs(Aw) ** 2, axis=(0, 1))
    return np.nonzero((S > rel * S.max()) & grid.valid)[0]


def total_spectrum(grid, Aw, per_lambda=True):
    """Spatially integrated spectrum, sorted by wavelength. Returns (lam [m], S)."""
    S = np.sum(np.abs(Aw) ** 2, axis=(0, 1)) * grid.dA * grid.dt / grid.Nt
    return _to_lambda(grid, S, per_lambda)


def _to_lambda(grid, S_w, per_lambda=True, axis=-1):
    """Om-ordered array -> ascending-wavelength array (Jacobian 2 pi c / lam^2 if per_lambda)."""
    idx = np.nonzero(grid.valid)[0]
    idx = idx[np.argsort(grid.lam[idx])]
    lam = grid.lam[idx]
    S = np.take(S_w, idx, axis=axis)
    if per_lambda:
        shape = [1] * S.ndim
        shape[axis] = -1
        S = S * (2 * np.pi * C0 / lam ** 2).reshape(shape)
    return lam, S


def fluence(grid, Aw):
    """F(x, y) [J/m^2], sorted."""
    F = np.sum(np.abs(Aw) ** 2, axis=2) * grid.dt / grid.Nt
    return grid.shift_xy(F)


def temporal_power(grid, Aw):
    """P(t) [W], sorted time axis."""
    A = grid.fft.ift_t(Aw)
    return grid.ts, grid.shift_w(np.sum(np.abs(A) ** 2, axis=(0, 1)) * grid.dA)


# ------------------------------------------------------------------ far field
def far_field(grid, Aw, theta_max=None, n_theta=None, idx=None, theta_y=None):
    """Angular spectrum E(theta_x, theta_y, Om) on a common angle grid (sorted, odd size so that
    theta = 0 is sampled). theta_max default: half the largest kx at lambda0.
    theta_y: optional explicit theta_y values (e.g. [0.] for a single line).
    Returns (theta_x, Ef) with Ef of shape (n_theta, len(theta_y), Nt), FFT order along Om."""
    n_theta = n_theta or (grid.Nx // 2) * 2 + 1
    if theta_max is None:
        theta_max = 0.5 * np.abs(grid.kx).max() / (grid.omega0 / C0)
    th = np.linspace(-theta_max, theta_max, n_theta)
    thy = th if theta_y is None else np.atleast_1d(np.asarray(theta_y, float))
    idx = spectral_support(grid, Aw) if idx is None else idx
    Ef = np.zeros((len(th), len(thy), grid.Nt), complex)
    for j in idx:
        k = grid.omega[j] / C0
        Fx = np.exp(-1j * k * np.outer(th, grid.x)) * grid.dx * k / (2 * np.pi)
        Fy = np.exp(-1j * k * np.outer(thy, grid.y)) * grid.dy * k / (2 * np.pi)
        Ef[:, :, j] = Fx @ Aw[:, :, j] @ Fy.T
    return th, Ef


# ------------------------------------------------------------------ spectrograms
def spectrogram_x_lambda(grid, Aw, y0=0.0, slit=None, axis="x", per_lambda=True):
    """Spatio-spectral trace S(x, lambda) along a line (imaging spectrometer).
    slit : width [m] integrated around y0 (None = single row). axis 'x' or 'y'.
    Returns (x_sorted [m], lam [m], S[x, lam])."""
    if axis == "y":
        Aw = np.swapaxes(Aw, 0, 1)
        coord, other, d = grid.y, grid.x, grid.dx
    else:
        coord, other, d = grid.x, grid.y, grid.dy
    if slit is None:
        sel = [int(np.argmin(np.abs(other - y0)))]
    else:
        sel = np.nonzero(np.abs(other - y0) <= slit / 2)[0]
    S = np.sum(np.abs(Aw[:, sel, :]).astype(np.float64) ** 2, axis=1)
    lam, S = _to_lambda(grid, S, per_lambda, axis=1)
    return np.fft.fftshift(coord), lam, np.fft.fftshift(S, axes=0)


def spectrogram_theta_lambda(grid, Aw, theta_max=None, n_theta=None, per_lambda=True):
    """Angle-resolved spectrum S(theta_x, lambda) at theta_y = 0 (output after collimation:
    x_out = f * theta). Returns (theta [rad], lam [m], S[theta, lam])."""
    th, Ef = far_field(grid, Aw, theta_max, n_theta, theta_y=[0.0])
    S = np.abs(Ef[:, 0, :]).astype(np.float64) ** 2
    lam, S = _to_lambda(grid, S, per_lambda, axis=1)
    return th, lam, S


# ------------------------------------------------------------------ homogeneity
def homogeneity(grid, Aw=None, S_xyw=None, rel=1e-3):
    """Spectral homogeneity V(x, y) = (sum sqrt(S S0))^2 / (sum S sum S0), S0 = total spectrum,
    and its fluence-weighted mean. Pass either a field or an intensity cube S_xyw."""
    S = (np.abs(Aw) ** 2 if S_xyw is None else S_xyw).astype(np.float64)
    S = S / S.max()
    S0 = S.sum(axis=(0, 1))
    num = np.sum(np.sqrt(S * S0[None, None, :]), axis=2) ** 2
    den = S.sum(axis=2) * S0.sum()
    F = S.sum(axis=2)
    V = np.where(F > rel * F.max(), num / np.maximum(den, 1e-300), np.nan)
    Vm = np.nansum(V * F) / np.sum(F[~np.isnan(V)])
    return np.fft.fftshift(V, axes=(0, 1)) if S_xyw is None else V, float(Vm)


# ------------------------------------------------------------------ modal content
def _hermite_functions(x, w, M):
    """Normalised 1D HG functions u_m(x) for m = 0..M, shape (M+1, len(x))."""
    xi = np.sqrt(2) * x / w
    psi = np.zeros((M + 1,) + xi.shape)
    psi[0] = np.pi ** -0.25 * np.exp(-xi ** 2 / 2)
    if M >= 1:
        psi[1] = np.sqrt(2) * xi * psi[0]
    for m in range(1, M):
        psi[m + 1] = np.sqrt(2 / (m + 1)) * xi * psi[m] - np.sqrt(m / (m + 1)) * psi[m - 1]
    return psi * np.sqrt(np.sqrt(2) / w)


def mode_content(grid, Aw, w0_of_lambda, max_order=6, idx=None):
    """Decomposition on Hermite-Gauss modes of waist w0(lambda) with flat phase (cell centre).
    Returns dict: lam, eta00(lam) (fundamental fraction per colour), order_frac[N] (energy fraction
    in modes of order m+n = N, spectrally integrated, last bin = everything above max_order)."""
    idx = spectral_support(grid, Aw, 1e-3) if idx is None else idx
    lam = grid.lam[idx]
    M = max_order
    P = np.sum(np.abs(Aw[:, :, idx]) ** 2, axis=(0, 1)) * grid.dA
    order_E = np.zeros(M + 2)
    eta = np.zeros(len(idx))
    for n, j in enumerate(idx):
        w = w0_of_lambda(lam[n])
        ux = _hermite_functions(grid.x, w, M)
        uy = _hermite_functions(grid.y, w, M)
        c = ux @ Aw[:, :, j] @ uy.T * grid.dA          # (M+1, M+1)
        p = np.abs(c) ** 2
        eta[n] = p[0, 0] / P[n]
        mm, nn = np.meshgrid(np.arange(M + 1), np.arange(M + 1), indexing="ij")
        for N in range(M + 1):
            order_E[N] += p[mm + nn == N].sum()
        order_E[M + 1] += max(P[n] - p[mm + nn <= M].sum(), 0.0)
    o = np.argsort(lam)
    return {"lam": lam[o], "eta00": eta[o], "order_frac": order_E / P.sum(),
            "eta00_total": float(np.sum(eta * P) / P.sum())}


# ------------------------------------------------------------------ beam size / M2
def beam_moments(grid, Aw, idx=None):
    """Per colour: second-moment diameters/2 (wx, wy) [m] and M2x, M2y. Sorted by wavelength."""
    idx = spectral_support(grid, Aw, 1e-3) if idx is None else idx
    E = Aw[:, :, idx].astype(np.complex128)
    E /= np.abs(E).max()
    I = np.abs(E) ** 2
    P = I.sum(axis=(0, 1))
    out = {}
    for ax, coord, kc in ((0, grid.x, grid.kx), (1, grid.y, grid.ky)):
        sh = [1, 1, 1]
        sh[ax] = -1
        c = coord.reshape(sh)
        kk = kc.reshape(sh)
        xm = (I * c).sum(axis=(0, 1)) / P
        x2 = (I * c ** 2).sum(axis=(0, 1)) / P - xm ** 2
        Ek = np.fft.fft(E, axis=ax)
        Ik = np.abs(Ek) ** 2
        Pk = Ik.sum(axis=(0, 1))
        km = (Ik * kk).sum(axis=(0, 1)) / Pk
        k2 = (Ik * kk ** 2).sum(axis=(0, 1)) / Pk - km ** 2
        dE = np.fft.ifft(1j * kk * Ek, axis=ax)
        xk = (c * np.imag(np.conj(E) * dE)).sum(axis=(0, 1)) / P - xm * km
        name = "x" if ax == 0 else "y"
        out["w" + name] = 2 * np.sqrt(np.maximum(x2, 0))
        out["M2" + name] = 2 * np.sqrt(np.maximum(x2 * k2 - xk ** 2, 0))
    lam = grid.lam[idx]
    o = np.argsort(lam)
    out = {k: v[o] for k, v in out.items()}
    out["lam"] = lam[o]
    wts = P[o] / P.sum()
    out["M2x_mean"] = float(np.sum(out["M2x"] * wts))
    out["M2y_mean"] = float(np.sum(out["M2y"] * wts))
    return out


# ------------------------------------------------------------------ compression
def _fwhm(t, y):
    y = y / y.max()
    above = np.nonzero(y >= 0.5)[0]
    if len(above) < 2:
        return np.nan
    i0, i1 = above[0], above[-1]
    tl = np.interp(0.5, [y[i0 - 1], y[i0]], [t[i0 - 1], t[i0]]) if i0 > 0 else t[i0]
    tr = np.interp(0.5, [y[i1 + 1], y[i1]], [t[i1 + 1], t[i1]]) if i1 < len(t) - 1 else t[i1]
    return tr - tl


def compress(grid, Aw, gdd_bounds_fs2=(-30000, 30000), tod_fs3=0.0, n_scan=61, rel=1e-4):
    """Best uniform GDD (e.g. chirped mirrors) maximising the peak of the spatially integrated
    power: coarse scan over gdd_bounds_fs2 then golden-section refinement.
    Returns dict with gdd_fs2, t_fs, P(t) [W] (compressed), FWHM, transform-limited FWHM,
    peak power and peak power of the (spatially resolved) transform limit."""
    F = np.sum(np.abs(Aw) ** 2, axis=2)
    mask = F > rel * F.max()
    E = Aw[mask]                                   # (npix, Nt)
    W = grid.Omega * FS

    def P_of(gdd):
        ph = np.exp(1j * (gdd / 2 * W ** 2 + tod_fs3 / 6 * W ** 3)).astype(E.dtype)
        A = grid.fft.ift_t(E * ph[None, :])
        return np.sum(np.abs(A).astype(np.float64) ** 2, axis=0) * grid.dA

    scan = np.linspace(*gdd_bounds_fs2, n_scan)
    pk = [P_of(g).max() for g in scan]
    i = int(np.argmax(pk))
    a, b = scan[max(i - 1, 0)], scan[min(i + 1, n_scan - 1)]
    gr = (np.sqrt(5) - 1) / 2
    c, d = b - gr * (b - a), a + gr * (b - a)
    fc, fd = P_of(c).max(), P_of(d).max()
    for _ in range(30):
        if fc > fd:
            b, d, fd = d, c, fc
            c = b - gr * (b - a)
            fc = P_of(c).max()
        else:
            a, c, fc = c, d, fd
            d = a + gr * (b - a)
            fd = P_of(d).max()
    g = (a + b) / 2
    P = np.fft.fftshift(P_of(g))
    t = grid.ts / FS
    Etl = np.abs(E).astype(np.float64)             # flat phase in every pixel
    Ptl = np.fft.fftshift(np.sum(np.abs(grid.fft.ift_t(Etl)) ** 2, axis=0) * grid.dA)
    return {"gdd_fs2": g, "t_fs": t, "P": P, "fwhm_fs": _fwhm(t, P), "tl_fwhm_fs": _fwhm(t, Ptl),
            "peak_power": float(P.max()), "peak_power_TL": float(Ptl.max()),
            "gdd_scan_fs2": scan, "peak_scan": np.array(pk)}


def local_gdd_map(grid, Aw, rel_fluence=1e-2, rel_spectrum=1e-2):
    """Pixel-wise GDD [fs^2] from a weighted quadratic fit of the spectral phase
    (radial chirp map). Returns sorted GDD(x, y) with NaN outside the beam."""
    F = np.sum(np.abs(Aw) ** 2, axis=2)
    S0 = np.sum(np.abs(Aw) ** 2, axis=(0, 1))
    idx = np.nonzero((S0 > rel_spectrum * S0.max()) & grid.valid)[0]
    idx = idx[np.argsort(grid.Omega[idx])]
    W = grid.Omega[idx] * FS
    mask = F > rel_fluence * F.max()
    E = Aw[mask][:, idx].astype(np.complex128)
    E /= np.abs(E).max()
    ph = np.unwrap(np.angle(E), axis=1)
    wt = np.abs(E) ** 2
    V = np.stack([np.ones_like(W), W, W ** 2], axis=1)            # (n, 3)
    G = np.einsum("pn,ni,nj->pij", wt, V, V)
    r = np.einsum("pn,ni,pn->pi", wt, V, ph)
    coef = np.linalg.solve(G, r[..., None])[..., 0]
    out = np.full(F.shape, np.nan)
    out[mask] = 2 * coef[:, 2]
    return grid.shift_xy(out)


def radial_profile(grid, M2d, nbins=None):
    """Azimuthal average of a sorted 2D map, returns (r, value)."""
    X, Y = np.meshgrid(grid.xs, grid.ys, indexing="ij")
    r = np.sqrt(X ** 2 + Y ** 2)
    nb = nbins or grid.Nx // 2
    edges = np.linspace(0, r.max() / np.sqrt(2), nb + 1)
    ok = ~np.isnan(M2d)
    s, _ = np.histogram(r[ok], edges, weights=M2d[ok])
    n, _ = np.histogram(r[ok], edges)
    with np.errstate(invalid="ignore"):
        return 0.5 * (edges[1:] + edges[:-1]), s / n


# ------------------------------------------------------------------ wavelength-resolved profiles
def band_indices(grid, lam_c, bandwidth=0.0):
    """Om indices inside [lam_c - bw/2, lam_c + bw/2]; bandwidth 0 -> nearest bin only
    (the grid spectral resolution is lambda^2 / (c T))."""
    if bandwidth > 0:
        idx = np.nonzero(grid.valid & (np.abs(grid.lam - lam_c) <= bandwidth / 2))[0]
        if len(idx):
            return idx
    return np.array([int(np.argmin(np.where(grid.valid, np.abs(grid.lam - lam_c), np.inf)))])


def profiles_1d(grid, Aw, wavelengths, bandwidth=0.0, axis="x", mode="cut"):
    """1D beam profiles for selected wavelengths (bandpass of width `bandwidth`) + the
    spectrally integrated ("global") profile, in the plane where the field is.

    mode 'cut'        : line through y = 0 (x = 0 if axis='y')
    mode 'integrated' : projection on the axis (like a 1D camera / slit-less lineout)
    Returns (coord_sorted [m], profiles[n_lambda + 1, N] (last row = global, not normalised),
             band_energy[n_lambda + 1] (J, fraction of the pulse in each band; last = total)).
    """
    I = None
    A = Aw if axis == "x" else np.swapaxes(Aw, 0, 1)
    coord = grid.x if axis == "x" else grid.y
    d_other = grid.dy if axis == "x" else grid.dx

    def lineout(idx):
        sub = np.abs(A[:, :, idx]).astype(np.float64) ** 2
        s = sub.sum(axis=2)
        return s[:, 0] if mode == "cut" else s.sum(axis=1) * d_other

    rows, en = [], []
    norm = grid.dA * grid.dt / grid.Nt
    for lc in wavelengths:
        idx = band_indices(grid, lc, bandwidth)
        rows.append(lineout(idx))
        en.append(float(np.sum(np.abs(A[:, :, idx]).astype(np.float64) ** 2)) * norm)
    rows.append(lineout(np.nonzero(grid.valid)[0]))
    en.append(grid.energy(Aw))
    return np.fft.fftshift(coord), np.fft.fftshift(np.array(rows), axes=1), np.array(en)


def far_profiles_1d(grid, Aw, wavelengths, bandwidth=0.0, theta_max=None, n_theta=None):
    """Same as profiles_1d (cut through theta_y = 0) in the far field (angle, common for all
    colours = beam after collimation). Returns (theta [rad], profiles[n_lambda + 1, n_theta])."""
    idx_all = spectral_support(grid, Aw)
    th, Ef = far_field(grid, Aw, theta_max, n_theta, idx=idx_all, theta_y=[0.0])
    S = np.abs(Ef[:, 0, :]) ** 2
    rows = [S[:, band_indices(grid, lc, bandwidth)].sum(axis=1) for lc in wavelengths]
    rows.append(S.sum(axis=1))
    return th, np.array(rows)
