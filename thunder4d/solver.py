"""4D UPPE in the frame moving at the group velocity at lambda0.

    dA(kx,ky,Om)/dz = i [kz - k0 - k1 Om] A  +  i gamma(Om) FT_t[ |A|^2 A ]
    kz = sqrt(k(w)^2 - kx^2 - ky^2)      (non-paraxial, full dispersion of the gas)
    gamma(Om) = n2 w / c * n(w0)/n(w)     (self-steepening included; set self_steepening=False
                                          for gamma = n2 w0/c)
The Kerr term does not depend on (kx, ky) so it is evaluated in (x, y, Om) with 1D temporal FFTs;
the linear step needs only 2D spatial FFTs. Strang splitting, RK4 for the nonlinear sub-step,
step size set by the peak nonlinear phase per step (phi_max), like the B-driven dz of the MATLAB code.
Extension point: override `nonlinear(Aw)` (ionisation, Raman, ...).
"""
import time
import numpy as np
from .grid import C0


class UPPESolver:
    def __init__(self, grid, medium, *, phi_max=0.02, dz_max=None, paraxial=False,
                 self_steepening=True, absorber_frac=0.9, absorber_order=16, cache_size=4):
        self.grid, self.medium = grid, medium
        g = grid
        self.phi_max, self.dz_max = phi_max, dz_max
        w, w0 = g.omega, g.omega0
        k = medium.k(np.where(g.valid, w, w0))
        self.k0 = float(medium.k(w0))
        h = w0 * 1e-4
        self.k1 = float((medium.k(w0 + h) - medium.k(w0 - h)) / (2 * h))
        self.n2 = medium.n2

        xp = self.xp = g.xp                          # numpy on the CPU, cupy on the GPU
        KX2 = xp.asarray(g.kx[:, None] ** 2 + g.ky[None, :] ** 2)
        self.D = xp.empty(g.shape, np.complex64 if g.dtype == np.complex64 else np.complex128)
        for j in range(g.Nt):
            if not g.valid[j]:
                self.D[:, :, j] = 0
                continue
            kj = float(k[j])
            if paraxial:
                kz = kj - KX2 / (2 * kj)
            else:
                kz = xp.sqrt(kj ** 2 - KX2 + 0j)     # evanescent -> +i|.|, decays
            self.D[:, :, j] = kz - self.k0 - self.k1 * float(g.Omega[j])
        self._cache, self.cache_size = {}, cache_size

        n = medium.n(np.where(g.valid, w, w0))
        n0 = medium.n(w0)
        gam = self.n2 * (w / C0 * n0 / n if self_steepening else np.full_like(w, w0 / C0))
        # taper the nonlinear source to zero at the edges of the spectral window
        u = np.abs(g.Omega) / np.abs(g.Omega).max()
        gam = gam * np.exp(-(u / 0.95) ** 40) * g.valid
        self.igamma = xp.asarray((1j * gam).astype(g.dtype)[None, None, :])

        # soft absorbers applied once per half-pass (not per step)
        ax = lambda v: np.exp(-(np.abs(v) / (absorber_frac * np.abs(v).max())) ** absorber_order)
        self.absorber_xy = xp.asarray((ax(g.x)[:, None] * ax(g.y)[None, :]).astype(g.rdtype)[:, :, None])
        self.absorber_w = xp.asarray((ax(g.Omega) * g.valid).astype(g.rdtype)[None, None, :])
        self.absorber_t = xp.asarray(ax(g.t).astype(g.rdtype)[None, None, :])

        self.B = 0.0           # accumulated peak (on-axis) B-integral
        self.nsteps = 0
        self.z = 0.0
        self._Ipeak = None

    # ------------------------------------------------------------ operators
    def linear(self, Aw, h):
        if h == 0:
            return Aw
        F = self.grid.fft
        Ak = F.fft2(Aw)
        Ak *= self._propagator(h)
        return F.ifft2(Ak).astype(self.grid.dtype, copy=False)

    def _propagator(self, h):
        """exp(i D h), with a small LRU cache (steps are quantised, see propagate)."""
        key = float(f"{h:.9e}")
        P = self._cache.pop(key, None)
        if P is None:
            P = self.xp.multiply(self.D, self.D.dtype.type(1j * h))
            self.xp.exp(P, out=P)
            if self.cache_size > 0 and len(self._cache) >= self.cache_size:
                self._cache.pop(next(iter(self._cache)))
        if self.cache_size > 0:
            self._cache[key] = P
        return P

    def nonlinear(self, Aw, track=False):
        F = self.grid.fft
        A = F.ift_t(Aw)
        I = A.real ** 2 + A.imag ** 2
        if track:
            self._Ipeak = float(I.max())
        A *= I
        out = F.ft_t(A)
        out *= self.igamma
        return out.astype(self.grid.dtype, copy=False)

    def _rk4(self, Aw, h):
        k = self.nonlinear(Aw, track=True)
        acc = k.copy()
        tmp = self.xp.empty_like(Aw)
        for c, wgt in ((0.5, 2.0), (0.5, 2.0), (1.0, 1.0)):
            self.xp.multiply(k, c * h, out=tmp)
            tmp += Aw
            k = self.nonlinear(tmp)
            if wgt == 2.0:
                acc += k
                acc += k
            else:
                acc += k
        acc *= h / 6
        Aw += acc
        return Aw

    def peak_intensity(self, Aw):
        A = self.grid.fft.ift_t(Aw)
        return float((A.real ** 2 + A.imag ** 2).max())

    # ------------------------------------------------------------ propagation
    def propagate(self, Aw, L):
        """Propagate over L [m] (Strang: L/2 - N - L/2, consecutive linear halves merged)."""
        Aw = self.grid.asarray(Aw)
        if self.n2 == 0:
            Aw = self.linear(Aw, L)
            self.z += L
            return Aw
        if self._Ipeak is None:
            self._Ipeak = self.peak_intensity(Aw)
        dz_max = self.dz_max or L / 4
        z, pending = 0.0, 0.0
        while z < L * (1 - 1e-12):
            h_req = self.phi_max / (self.k0 * self.n2 * max(self._Ipeak, 1e-30))
            # quantise to dz_max * 2^(-n/4) so that exp(i D h) can be reused from the cache
            h = dz_max * 2.0 ** (-np.ceil(max(0.0, 4 * np.log2(dz_max / h_req))) / 4)
            h = min(h, L - z)
            Aw = self.linear(Aw, pending + h / 2)
            Aw = self._rk4(Aw, h)
            self.B += self.k0 * self.n2 * self._Ipeak * h
            pending = h / 2
            z += h
            self.nsteps += 1
        Aw = self.linear(Aw, pending)
        self.z += L
        return Aw

    def absorb(self, Aw, temporal=False):
        Aw *= self.absorber_xy
        Aw *= self.absorber_w
        if temporal:
            A = self.grid.fft.ift_t(Aw)
            A *= self.absorber_t
            Aw = self.grid.fft.ft_t(A).astype(self.grid.dtype, copy=False)
        return Aw
