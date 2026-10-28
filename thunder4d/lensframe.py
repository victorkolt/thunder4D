"""Beam-following ("lens-frame") propagation: a grid that zooms with the beam inside the cell.

Field written as  A = (1/s) B(x/s, y/s, zeta) exp(i k(w) r^2 / 2R(z)),  s = sqrt(1 + z^2/zR^2),
zR = sqrt(d(2R-d))/2, zeta = zR atan(z/zR). B obeys (paraxial, exact):
    dB/dzeta = i/(2k) Lap B - i k r^2/(2 zR^2) B + i beta(Om) s^2 B + i gamma |B|^2 B
The Kerr term is unchanged in zeta; the curved mirror is the identity in this frame; at the cell
centre B is the physical field. The grid only has to hold the beam at the waist scale.
"""
import time
import numpy as np

from .grid import C0, Grid, _next_fast
from .solver import UPPESolver
from .simulation import MPCSimulation
from . import diagnostics as dg

FS = 1e-15


def lens_frame_scale(cell):
    """Geometric reference of the cell: zR, zeta at the mirror, mirror scale factor s_m."""
    zR = np.sqrt(cell.d * (2 * cell.R - cell.d)) / 2
    zeta_m = zR * np.arctan(cell.d / (2 * zR))
    s_m = np.sqrt(1 + (cell.d / (2 * zR)) ** 2)
    return zR, zeta_m, s_m


def suggest_lens_grid(cell, lambda0, lambda_min, lambda_max, pulse_fwhm, *, box_factor=6.0,
                      points_per_waist=8.0, time_window_factor=8.0, **grid_kw):
    """box = box_factor x waist at lambda_max, dx = waist at lambda_min / points_per_waist.
    box_factor ~ 6 for matched gaussians, 10-14 for top-hats, hard edges, strong mismatch."""
    w0_hi = cell.linear_mode(lambda_max)["w0"]
    w0_lo = cell.linear_mode(lambda_min)["w0"]
    L = box_factor * w0_hi
    dx = w0_lo / points_per_waist
    Nx = _next_fast(L / dx)
    w_hi = 2 * np.pi * C0 / lambda_min - 2 * np.pi * C0 / lambda0
    w_lo = 2 * np.pi * C0 / lambda0 - 2 * np.pi * C0 / lambda_max
    dt = np.pi / max(w_hi, w_lo)
    Nt = _next_fast(time_window_factor * pulse_fwhm / dt)
    return Grid(Nx, L / Nx, Nt, lambda0, dt=dt, **grid_kw)


class LensFrameSolver(UPPESolver):
    """Split step in zeta: [kinetic+dispersion](h/2) [potential(h/2) Kerr-RK4(h) potential(h/2)]
    [kinetic+dispersion](h/2), kinetic halves merged (one 2D FFT pair per step).
    phi_max: peak nonlinear phase per step; gouy_step: max step in accumulated Gouy phase [rad]."""

    def __init__(self, grid, cell, *, phi_max=0.02, gouy_step=0.05, self_steepening=True,
                 absorber_frac=0.9, absorber_order=16):
        # UPPESolver.__init__ is not called: it would build the 3D operator we want to avoid
        g, medium = grid, cell.medium
        self.grid, self.medium, self.cell = grid, medium, cell
        xp = self.xp = g.xp
        self.phi_max = phi_max
        self.zR, self.zeta_m, self.s_m = lens_frame_scale(cell)
        self.h_max = gouy_step * self.zR

        w, w0 = g.omega, g.omega0
        k = np.where(g.valid, medium.k(np.where(g.valid, w, w0)), medium.k(w0))
        self.k0 = float(medium.k(w0))
        hw = w0 * 1e-4
        self.k1 = float((medium.k(w0 + hw) - medium.k(w0 - hw)) / (2 * hw))
        self.n2 = medium.n2
        self._beta = np.where(g.valid, k - self.k0 - self.k1 * g.Omega, 0.0)
        self._cdt = np.complex64 if g.dtype == np.complex64 else np.complex128
        self._kx2 = (g.kx ** 2)[:, None, None]
        self._ky2 = (g.ky ** 2)[None, :, None]
        self._x2 = (g.x ** 2)[:, None, None]
        self._y2 = (g.y ** 2)[None, :, None]
        self._kk = k[None, None, :]

        n = medium.n(np.where(g.valid, w, w0))
        n0 = medium.n(w0)
        gam = self.n2 * (w / C0 * n0 / n if self_steepening else np.full_like(w, w0 / C0))
        u = np.abs(g.Omega) / np.abs(g.Omega).max()
        gam = gam * np.exp(-(u / 0.95) ** 40) * g.valid
        self.igamma = xp.asarray((1j * gam).astype(g.dtype)[None, None, :])

        ax = lambda v: np.exp(-(np.abs(v) / (absorber_frac * np.abs(v).max())) ** absorber_order)
        self.absorber_xy = xp.asarray((ax(g.x)[:, None] * ax(g.y)[None, :]).astype(g.rdtype)[:, :, None])
        self.absorber_w = xp.asarray((ax(g.Omega) * g.valid).astype(g.rdtype)[None, None, :])
        self.absorber_t = xp.asarray(ax(g.t).astype(g.rdtype)[None, None, :])

        self.B, self.nsteps, self.z, self._Ipeak = 0.0, 0, 0.0, None
        self._cache, self.cache_size = {}, 0

    def z_of_zeta(self, zeta):
        return self.zR * np.tan(zeta / self.zR)

    def _kinetic(self, Bw, h, dz):
        """exp(-i k_perp^2 h / 2k) exp(i beta dz), in (kx, ky, Om)."""
        if h == 0 and dz == 0:
            return Bw
        F, xp = self.grid.fft, self.xp
        Bk = F.fft2(Bw)
        Bk *= xp.asarray(np.exp(-1j * self._kx2 * h / (2 * self._kk)).astype(self._cdt))
        Bk *= xp.asarray(np.exp(-1j * self._ky2 * h / (2 * self._kk)).astype(self._cdt))
        Bk *= xp.asarray(np.exp(1j * self._beta * dz).astype(self._cdt)[None, None, :])
        return F.ifft2(Bk).astype(self.grid.dtype, copy=False)

    def _potential(self, Bw, h):
        """exp(-i k (xi^2 + eta^2) h / (2 zR^2)), in (xi, eta, Om)."""
        if h == 0:
            return Bw
        xp, c = self.xp, h / (2 * self.zR ** 2)
        Bw *= xp.asarray(np.exp(-1j * self._kk * self._x2 * c).astype(self._cdt))
        Bw *= xp.asarray(np.exp(-1j * self._kk * self._y2 * c).astype(self._cdt))
        return Bw

    def propagate_zeta(self, Bw, zeta_a, zeta_b):
        """Propagate B from zeta_a to zeta_b (within one pass, |zeta| <= zeta_m)."""
        Bw = self.grid.asarray(Bw)
        nonlinear = self.n2 != 0
        if nonlinear and self._Ipeak is None:
            self._Ipeak = self.peak_intensity(Bw)
        zeta, pend_h, pend_dz = zeta_a, 0.0, 0.0
        while zeta < zeta_b - 1e-15 * self.zR:
            h = self.h_max
            if nonlinear:
                h = min(h, self.phi_max / (self.k0 * self.n2 * max(self._Ipeak, 1e-30)))
            h = min(h, zeta_b - zeta)
            zm = zeta + h / 2
            dz1 = self.z_of_zeta(zm) - self.z_of_zeta(zeta)
            dz2 = self.z_of_zeta(zeta + h) - self.z_of_zeta(zm)
            Bw = self._kinetic(Bw, pend_h + h / 2, pend_dz + dz1)
            Bw = self._potential(Bw, h / 2)
            if nonlinear:
                Bw = self._rk4(Bw, h)
                self.B += self.k0 * self.n2 * self._Ipeak * h       # dB = k0 n2 I_B dzeta
            Bw = self._potential(Bw, h / 2)
            pend_h, pend_dz = h / 2, dz2
            zeta += h
            self.nsteps += 1
        Bw = self._kinetic(Bw, pend_h, pend_dz)
        self.z += self.z_of_zeta(zeta_b) - self.z_of_zeta(zeta_a)
        return Bw
    
    def propagate_linear_zeta(self, Bw, zeta_a, zeta_b):
        """Linear propagation (no Kerr) from zeta_a to zeta_b, forwards or backwards."""
        Bw = self.grid.asarray(Bw)
        n = max(1, int(np.ceil(abs(zeta_b - zeta_a) / self.h_max)))
        h = (zeta_b - zeta_a) / n
        zeta = zeta_a
        for _ in range(n):
            zm = zeta + h / 2
            Bw = self._kinetic(Bw, h / 2, self.z_of_zeta(zm) - self.z_of_zeta(zeta))
            Bw = self._potential(Bw, h)
            Bw = self._kinetic(Bw, h / 2, self.z_of_zeta(zeta + h) - self.z_of_zeta(zm))
            zeta += h
        return Bw
    
    def propagate(self, Aw, L):
        raise NotImplementedError("use propagate_zeta (lens frame) or the standard UPPESolver")

    def mirror(self, Bw):
        """Bounce in the lens frame: spectral response and aperture only (curvature cancels)."""
        g, cell = self.grid, self.cell
        W = g.Omega * FS
        spec = np.exp(1j * (cell.gdd / 2 * W ** 2 + cell.tod / 6 * W ** 3)).astype(complex)
        lam = np.where(g.valid, g.lam, 1.0)
        spec *= np.sqrt(np.clip(cell.refl(lam), 0, 1)) if callable(cell.refl) else np.sqrt(cell.refl)
        if cell.mphase is not None:
            spec *= np.exp(1j * cell.mphase(lam))
        spec *= g.valid
        Bw *= self.xp.asarray(spec.astype(g.dtype)[None, None, :])
        if cell.diameter is not None:                      # physical radius r = s_m * xi
            ap = np.exp(-(self.s_m * np.sqrt(g.r2) / (cell.diameter / 2)) ** 30)
            Bw *= self.xp.asarray(ap.astype(g.rdtype)[:, :, None])
        return Bw

def imaged_near_field_lens(grid, E, beam, energy, cell, image_z=None, verbose=True):
    """Lens-frame version of Pulse.imaged_near_field: near field imaged (same size for every colour)
    onto the plane at distance image_z from the centre (default: the mirror, d/2), with the
    wavefront of the cell eigenmode, then propagated back to the centre in the lens frame.

    In the frame the eigenmode wavefront cancels exactly, so the field at that plane is just the
    profile with coordinates scaled by s(z): B(xi) = s U(s xi). `beam` is given in PHYSICAL
    coordinates at the image plane (e.g. top_hat(1.12 * w_mirror))."""
    from .pulse import Pulse
    S = LensFrameSolver(grid, cell)
    z = cell.d / 2 if image_z is None else float(image_z)
    s = np.sqrt(1 + (z / S.zR) ** 2)
    zeta = S.zR * np.arctan(z / S.zR)
    Xs, Ys = s * grid.X, s * grid.Y
    Aw = np.zeros(grid.shape, grid.dtype)
    for j in Pulse._support(E):
        U = beam.field(Xs, Ys, grid.omega[j], grid.omega0)
        U /= np.sqrt(np.sum(np.abs(U) ** 2) * grid.dA)
        Aw[:, :, j] = E[j] * U
    p = Pulse(grid, Aw)
    p.Aw = S.propagate_linear_zeta(p.Aw, zeta, 0.0)
    p.set_energy(energy)
    if verbose:
        print(f"[Pulse] near field imaged at z = {z*100:.1f} cm (lens frame, s = {s:.2f})")
    return p

class LensFrameSimulation(MPCSimulation):
    """MPCSimulation on a beam-following grid. Cell-centre records are physical; mirror records
    are in frame coordinates (physical x = mirror_scale * x, fluence / mirror_scale**2)."""

    def __init__(self, pulse, cell, *, phi_max=0.02, gouy_step=0.05, **kw):
        solver = LensFrameSolver(pulse.grid, cell, phi_max=phi_max, gouy_step=gouy_step)
        self.mirror_scale = solver.s_m
        super().__init__(pulse, cell, solver=solver, **kw)

    def run(self, n_passes=None, verbose=True, max_seconds=None):
        n = n_passes or (self.cell.n_passes - self.passes_done)
        S, t_start = self.solver, time.time()
        for _ in range(n):
            if max_seconds is not None and time.time() - t_start > max_seconds:
                break
            p = self.passes_done + 1
            self.Aw = S.propagate_zeta(self.Aw, 0.0, S.zeta_m)            # centre -> mirror
            Fm = dg.fluence(self.grid, self.Aw).astype(np.float32) if self.record_mirror else None
            Pm = self._mirror_profiles() if (self.record_mirror and self.prof_lams) else None
            self.Aw = S.mirror(self.Aw)
            self.Aw = S.absorb(self.Aw)
            self.Aw = S.propagate_zeta(self.Aw, -S.zeta_m, 0.0)           # mirror -> centre
            self.Aw = S.absorb(self.Aw, temporal=True)
            self.passes_done = p
            if p % self.record_every == 0 or p == self.cell.n_passes:
                self.record(p, Fm, Pm)
                if verbose:
                    h = self.history
                    print(f"pass {p:3d}/{self.cell.n_passes} | B = {S.B:6.2f} rad | "
                          f"E = {h['energy'][-1]*1e3:.4g} mJ | eta00 = {h['eta00_total'][-1]:.4f} | "
                          f"V_far = {h['V_far'][-1]:.4f} | steps {S.nsteps} | {h['wall_s'][-1]:.0f} s",
                          flush=True)
        return self