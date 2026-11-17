"""Beam-following ("lens-frame") propagation: a grid that zooms with the beam inside the cell.

Field written as  A = (1/s) B(x/s, y/s, zeta) exp(i k(w) r^2 / 2R(z)),  s = sqrt(1 + z^2/zR^2),
zR = sqrt(d(2R-d))/2, zeta = zR atan(z/zR). B obeys (paraxial, exact):
    dB/dzeta = i/(2k) Lap B - i k r^2/(2 zR^2) B + i beta(Om) s^2 B + i gamma |B|^2 B
The Kerr term is unchanged in zeta; the curved mirror is the identity in this frame; at the cell
centre B is the physical field. The grid only has to hold the beam at the waist scale.

Plates (bulk / hybrid MPC): z is the REDUCED coordinate of the equivalent cell (see thunder4d.cell),
in which a plate of thickness t is a stretch t n_g/n_p of the filling medium, so the frame (zR, s,
curvature) is the one of the equivalent cell and is exact across the plates. A solver built with
medium=<plate material> keeps the diffraction of the filling medium and applies the plate's own
dispersion (in its own group-velocity frame) and Kerr effect per PHYSICAL length, i.e. times
dz_scale = n_p/n_g per reduced length. Neglected: the colour dependence of n_g/n_p in the reduced
length (relative error ~ dn/n x t/zR on the diffraction inside the plate).
"""
from typing import Any

import numpy as np

from .grid import _sized_grid
from .pulse import Pulse
from .simulation import MPCSimulation
from .solver import UPPESolver


def lens_frame_scale(cell, lam=None):
    """Geometric reference of the (equivalent) cell: zR, zeta at the mirror, mirror scale s_m."""
    d = cell.equivalent_separation(lam)
    zR = np.sqrt(d * (2 * cell.R - d)) / 2
    zeta_m = zR * np.arctan(d / (2 * zR))
    s_m = np.sqrt(1 + (d / (2 * zR)) ** 2)
    return zR, zeta_m, s_m


def suggest_lens_grid(cell, lambda0, lambda_min, lambda_max, pulse_fwhm, *, box_factor=6.0,
                      points_per_waist=8.0, time_window_factor=8.0, **grid_kw):
    """box = box_factor x waist at lambda_max, dx = waist at lambda_min / points_per_waist.
    box_factor ~ 6 for matched gaussians, 10-14 for top-hats, hard edges, strong mismatch."""
    w0_hi = cell.linear_mode(lambda_max)["w0"]
    w0_lo = cell.linear_mode(lambda_min)["w0"]
    L = box_factor * w0_hi
    dx = w0_lo / points_per_waist
    return _sized_grid(L, dx, lambda0, lambda_min, lambda_max, pulse_fwhm, time_window_factor, **grid_kw)


class LensFrameSolver(UPPESolver):
    """Split step in zeta: [kinetic+dispersion](h/2) [potential(h/2) Kerr-RK4(h) potential(h/2)]
    [kinetic+dispersion](h/2), kinetic halves merged (one 2D FFT pair per step).
    phi_max: peak nonlinear phase per step; gouy_step: max step in accumulated Gouy phase [rad].
    kernels: 'auto' | 'numba' | 'numpy', see UPPESolver.
    medium: None (filling medium of the cell) or a plate material: propagation inside that plate,
    in the frame of the cell (see the module docstring)."""

    def __init__(self, grid, cell, *, medium=None, phi_max=0.02, gouy_step=0.05, self_steepening=True,
                 absorber_frac=0.9, absorber_order=16, kernels="auto"):
        # UPPESolver.__init__ is not called: it would build the 3D operator we want to avoid
        med = cell.medium if medium is None else medium
        self._setup(grid, med, phi_max, self_steepening, absorber_frac, absorber_order, kernels)
        self.cell = cell
        self.zR, self.zeta_m, self.s_m = lens_frame_scale(cell, grid.lambda0)
        self.h_max = gouy_step * self.zR
        g = grid
        # physical length per reduced length (1 in the filling medium, n_p/n_g in a plate)
        ref = cell.ref_medium      # diffraction of the equivalent cell (medium at the mirrors)
        self.dz_scale = float(med.n(g.omega0) / ref.n(g.omega0))
        k = ref.k(np.where(g.valid, g.omega, g.omega0))
        self._beta = np.where(g.valid, self.k - self.k0 - self.k1 * g.Omega, 0.0) * self.dz_scale
        if self.dz_scale != 1.0:          # Kerr per reduced length = Kerr per physical length x dz_scale
            self.n2 = self.n2 * self.dz_scale
            self.igamma = self.igamma * self.dz_scale
            self._ig1 = self._ig1 * self.dz_scale
        self._cdt = np.complex64 if g.dtype == np.complex64 else np.complex128
        self._kx2 = (g.kx ** 2)[:, None, None]
        self._ky2 = (g.ky ** 2)[None, :, None]
        self._x2 = (g.x ** 2)[:, None, None]
        self._y2 = (g.y ** 2)[None, :, None]
        self._kk = k[None, None, :]
        if self._nb:   # 1D factors of the separable phases, combined in one pass by mul_separable
            self._k1, self._kx2_1, self._ky2_1 = k, g.kx ** 2, g.ky ** 2
            self._x2_1, self._y2_1 = g.x ** 2, g.y ** 2
            self._ones_w = np.ones(g.Nt, self._cdt)

    def z_of_zeta(self, zeta):
        return self.zR * np.tan(zeta / self.zR)

    def _kinetic(self, Bw, h, dz):
        """exp(-i k_perp^2 h / 2k) exp(i beta dz), in (kx, ky, Om)."""
        if h == 0 and dz == 0:
            return Bw
        F, xp = self.grid.fft, self.xp
        Bk = F.fft2(Bw)
        if self._nb:
            c = h / (2 * self._k1)
            self._k.mul_separable(Bk, np.exp(-1j * np.outer(self._kx2_1, c)).astype(self._cdt),
                             np.exp(-1j * np.outer(self._ky2_1, c)).astype(self._cdt),
                             np.exp(1j * self._beta * dz).astype(self._cdt))
            return F.ifft2(Bk).astype(self.grid.dtype, copy=False)
        Bk *= xp.asarray(np.exp(-1j * self._kx2 * h / (2 * self._kk)).astype(self._cdt))
        Bk *= xp.asarray(np.exp(-1j * self._ky2 * h / (2 * self._kk)).astype(self._cdt))
        Bk *= xp.asarray(np.exp(1j * self._beta * dz).astype(self._cdt)[None, None, :])
        return F.ifft2(Bk).astype(self.grid.dtype, copy=False)

    def _potential(self, Bw, h):
        """exp(-i k (xi^2 + eta^2) h / (2 zR^2)), in (xi, eta, Om)."""
        if h == 0:
            return Bw
        xp, c = self.xp, h / (2 * self.zR ** 2)
        if self._nb:
            kc = self._k1 * c
            self._k.mul_separable(Bw, np.exp(-1j * np.outer(self._x2_1, kc)).astype(self._cdt),
                             np.exp(-1j * np.outer(self._y2_1, kc)).astype(self._cdt), self._ones_w)
            return Bw
        Bw *= xp.asarray(np.exp(-1j * self._kk * self._x2 * c).astype(self._cdt))
        Bw *= xp.asarray(np.exp(-1j * self._kk * self._y2 * c).astype(self._cdt))
        return Bw

    def propagate_zeta(self, Bw, zeta_a, zeta_b):
        """Propagate B from zeta_a to zeta_b (within one pass, |zeta| <= zeta_m)."""
        Bw = self.grid.asarray(Bw)
        nonlinear = self.n2 != 0
        if self._Ipeak is None:
            self._Ipeak = self.peak_intensity(Bw)
        zeta, pend_h, pend_dz = zeta_a, 0.0, 0.0
        while zeta < zeta_b - 1e-15 * self.zR:
            h = self.h_max
            if nonlinear:
                h = min(h, self.phi_max / (self.kv0 * self.n2 * max(self._Ipeak, 1e-30)))
            h = min(h, zeta_b - zeta)
            zm = zeta + h / 2
            dz1 = self.z_of_zeta(zm) - self.z_of_zeta(zeta)
            dz2 = self.z_of_zeta(zeta + h) - self.z_of_zeta(zm)
            Bw = self._kinetic(Bw, pend_h + h / 2, pend_dz + dz1)
            Bw = self._potential(Bw, h / 2)
            if nonlinear:
                Bw = self._rk4(Bw, h)
                self.B += self.kv0 * self.n2 * self._Ipeak * h      # dB = kv0 n2 I_B dzeta
            Bw = self._potential(Bw, h / 2)
            pend_h, pend_dz = h / 2, dz2
            zeta += h
            self.nsteps += 1
        Bw = self._kinetic(Bw, pend_h, pend_dz)
        self.z += (self.z_of_zeta(zeta_b) - self.z_of_zeta(zeta_a)) * self.dz_scale
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
        Bw *= self.xp.asarray(cell.mirror_spectrum(g).astype(g.dtype)[None, None, :])
        ap = cell.mirror_aperture(self.s_m * np.sqrt(g.r2))     # physical radius r = s_m * xi
        Bw *= self.xp.asarray(ap.astype(g.rdtype)[:, :, None])
        return Bw


def imaged_near_field_lens(grid, E, beam, energy, cell, image_z=None, verbose=True):
    """Lens-frame version of Pulse.imaged_near_field: near field imaged (same size for every colour)
    onto the plane at distance image_z from the centre (default: the mirror, d/2), with the
    wavefront of the cell eigenmode, then propagated back to the centre in the lens frame.

    In the frame the eigenmode wavefront cancels exactly, so the field at that plane is just the
    profile with coordinates scaled by s(z): B(xi) = s U(s xi). `beam` is given in PHYSICAL
    coordinates at the image plane (e.g. top_hat(1.12 * w_mirror)). With plates / gas regions,
    image_z is the physical position on pass 1 (towards mirror B) and the frame is the one of the
    equivalent cell."""
    if cell.segmented:          # plates / gas regions: equivalent cell, walk back through the segments
        from .simulation import Walker, make_solvers
        solvers = make_solvers(grid, cell, LensFrameSolver, {})
        walker = Walker(grid, cell, solvers, lens=True)
        z = cell.d / 2 if image_z is None else float(image_z)
        zr = cell.u_of_z(z, 1, grid.lambda0) - walker.uc      # reduced distance to the centre
        zR = solvers.zR
    else:
        S = LensFrameSolver(grid, cell)
        z = cell.d / 2 if image_z is None else float(image_z)
        zr, zR = z, S.zR
    s = np.sqrt(1 + (zr / zR) ** 2)
    zeta = zR * np.arctan(zr / zR)
    Xs, Ys = s * grid.X, s * grid.Y
    Aw = np.zeros(grid.shape, grid.dtype)
    for j in Pulse._support(E):
        U = beam.field(Xs, Ys, grid.omega[j], grid.omega0)
        U /= np.sqrt(np.sum(np.abs(U) ** 2) * grid.dA)
        Aw[:, :, j] = E[j] * U
    p = Pulse(grid, Aw)
    if cell.segmented:
        p.Aw = walker.walk(p.Aw, 1, walker.uc + zr, walker.uc, False)
    else:
        p.Aw = S.propagate_linear_zeta(p.Aw, zeta, 0.0)
    p.set_energy(energy)
    if verbose:
        print(f"[Pulse] near field imaged at z = {z*100:.1f} cm (lens frame, s = {s:.2f})")
    return p


class LensFrameSimulation(MPCSimulation):
    _lens = True

    """MPCSimulation on a beam-following grid. Cell-centre records are physical; mirror records
    and sim.state (right after a mirror) are in frame coordinates (physical x = mirror_scale * x,
    fluence / mirror_scale**2). Plates: segment walk of MPCSimulation, each segment in zeta of the
    equivalent cell."""

    solver: Any     # LensFrameSolver, or a _SolverSet of them (one per medium) with plates

    def __init__(self, pulse, cell, *, phi_max=0.02, gouy_step=0.05, self_steepening=True,
                 kernels="auto", **kw):
        self.grid, self.cell = pulse.grid, cell
        kw_s = {"phi_max": phi_max, "gouy_step": gouy_step, "self_steepening": self_steepening, "kernels": kernels}
        solver = self._make_solvers(LensFrameSolver, kw_s)
        self.mirror_scale = solver.s_m
        super().__init__(pulse, cell, solver=solver, **kw)

    def _to_centre(self, Aw):
        return self.solver.propagate_zeta(Aw, -self.solver.zeta_m, 0.0)

    def _to_mirror(self, Aw):
        return self.solver.propagate_zeta(Aw, 0.0, self.solver.zeta_m)

    def _bounce(self, Aw):
        return self.solver.mirror(Aw)

    def _from_centre(self, Aw):
        return self.solver.propagate_linear_zeta(Aw, 0.0, -self.solver.zeta_m)

    def _centre_view(self, Aw):
        return self.solver.propagate_linear_zeta(Aw.copy(), -self.solver.zeta_m, 0.0)
