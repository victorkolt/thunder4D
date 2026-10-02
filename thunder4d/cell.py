"""Module 2: the multipass cell.

Unfolded picture used by the solver: one pass = cell centre -> d/2 -> curved mirror -> d/2 -> cell
centre. The field is recorded at the cell centre (waist plane of the eigenmode, flat phase) after
every pass, which is where the modal decomposition on the cell eigenmodes is exact.

Nonlinear mode matching (aberration-free Kerr lens, uniform medium, power P constant along a pass):
the envelope equation of a gaussian becomes the linear one with lambda -> lambda sqrt(1 - P/Pc),
Pc = lambda^2/(2 pi n0 n2). Hence the matched waist and mirror spot are the linear ones scaled by
(1 - P/Pc)^(1/4), and the B-integral per pass (on axis, peak) is
    B = 8 pi n2 P / (lambda lambda_eff) * atan(d / 2 zR')      (-> 2 theta P/Pc for P << Pc).
A pulse can only be matched at one power: the wings of the pulse see the linear mode. This
time-dependent mismatch is precisely one of the couplings the 4D code resolves.
"""
import numpy as np
from .grid import C0

FS = 1e-15


class MPC:
    """Symmetric two-mirror cell.

    R            : mirror radius of curvature [m]
    d            : mirror separation [m]
    medium       : Material
    n_passes     : number of passes (centre -> mirror -> centre = 1 pass = 1 bounce)
    mirror_gdd_fs2, mirror_tod_fs3 : dispersion per bounce
    mirror_reflectivity : scalar, or callable R(lambda[m]) (intensity)
    mirror_phase : optional callable phi(lambda[m]) [rad] per bounce (measured GDD curves, ...)
    mirror_diameter : clear aperture [m] (soft super-gaussian edge), None = infinite
    """

    def __init__(self, R, d, medium, n_passes, *, mirror_gdd_fs2=0.0, mirror_tod_fs3=0.0,
                 mirror_reflectivity=1.0, mirror_phase=None, mirror_diameter=None):
        if not 0 < d < 2 * R:
            raise ValueError("unstable cell: need 0 < d < 2R")
        self.R, self.d, self.medium, self.n_passes = float(R), float(d), medium, int(n_passes)
        self.gdd, self.tod = mirror_gdd_fs2, mirror_tod_fs3
        self.refl, self.mphase = mirror_reflectivity, mirror_phase
        self.diameter = mirror_diameter
        self.g = 1 - d / R
        self.theta = np.arccos(self.g)  # Gouy phase per pass
        self._bounce_cache = None

    @classmethod
    def herriott(cls, R, N, k, medium, n_passes=None, **kw):
        """Re-entrant Herriott cell: N round trips (= spots per mirror), d = R (1 + cos(pi k/N)).
        Default n_passes = 2N (one full re-entrant cycle)."""
        d = R * (1 + np.cos(np.pi * k / N))
        return cls(R, d, medium, n_passes or 2 * N, **kw)

    # ------------------------------------------------------------ modes
    def linear_mode(self, lam):
        n0 = self.medium.n(2 * np.pi * C0 / lam)
        lm = lam / n0
        w0 = np.sqrt(lm / (2 * np.pi) * np.sqrt(self.d * (2 * self.R - self.d)))
        wm = np.sqrt(lm * self.R / np.pi * np.sqrt(self.d / (2 * self.R - self.d)))
        return {"w0": w0, "w_mirror": wm, "zR": np.pi * w0 ** 2 / lm, "gouy_per_pass": self.theta}

    def nonlinear_mode(self, P, lam=None, verbose=False):
        """Kerr-lens matched eigenmode for power P [W]."""
        lam = lam or self._lam0
        Pc = self.medium.critical_power(lam)
        r = P / Pc
        if r >= 1:
            raise ValueError(f"P/Pc = {r:.2f} >= 1: no matched mode (self-focusing)")
        lin = self.linear_mode(lam)
        s = (1 - r) ** 0.25
        w0, wm = lin["w0"] * s, lin["w_mirror"] * s
        lam_eff = lam * np.sqrt(1 - r)
        zRp = np.pi * w0 ** 2 / lam_eff
        B = 8 * np.pi * self.medium.n2 * P / (lam * lam_eff) * np.arctan(self.d / (2 * zRp))
        out = {"P": P, "Pc": Pc, "P_over_Pc": r, "w0": w0, "w_mirror": wm,
               "w0_linear": lin["w0"], "w_mirror_linear": lin["w_mirror"], "B_per_pass": B,
               "B_total": B * self.n_passes,
               "I_peak_center_W_cm2": 2 * P / (np.pi * w0 ** 2) * 1e-4}
        if verbose:
            print(self.report(out))
        return out

    def match(self, pulse, power="peak", verbose=True):
        """Nonlinear mode matching from a Pulse (or a power in W). power: 'peak' | 'weighted'."""
        self._lam0 = pulse.grid.lambda0 if hasattr(pulse, "grid") else self._lam0
        if hasattr(pulse, "grid"):
            P = pulse.peak_power() if power == "peak" else pulse.weighted_power()
        else:
            P = float(pulse)
        return self.nonlinear_mode(P, self._lam0, verbose=verbose)

    _lam0 = 1030e-9

    def report(self, m=None):
        s = (f"MPC: R = {self.R:.3f} m, d = {self.d:.4f} m, {self.n_passes} passes, "
             f"Gouy/pass = {np.degrees(self.theta):.2f} deg, {self.medium}")
        if m:
            s += (f"\n  P = {m['P']/1e9:.3g} GW, Pc = {m['Pc']/1e9:.3g} GW, P/Pc = {m['P_over_Pc']:.3f}"
                  f"\n  waist  {m['w0_linear']*1e6:.0f} um (linear) -> {m['w0']*1e6:.0f} um (matched)"
                  f"\n  mirror {m['w_mirror_linear']*1e6:.0f} um (linear) -> {m['w_mirror']*1e6:.0f} um (matched)"
                  f"\n  B/pass ~ {m['B_per_pass']:.3f} rad, total ~ {m['B_total']:.2f} rad, "
                  f"I_peak(centre) ~ {m['I_peak_center_W_cm2']:.3g} W/cm^2")
        return s

    # ------------------------------------------------------------ mirror
    def bounce_factor(self, grid):
        """Complex transfer of one mirror in (x, y, Om) (cached)."""
        if self._bounce_cache is not None and self._bounce_cache[0] is grid:
            return self._bounce_cache[1]
        W = grid.Omega * FS
        spec = np.exp(1j * (self.gdd / 2 * W ** 2 + self.tod / 6 * W ** 3)).astype(complex)
        lam = grid.lam
        if callable(self.refl):
            spec *= np.sqrt(np.clip(self.refl(np.where(grid.valid, lam, 1.0)), 0, 1))
        else:
            spec *= np.sqrt(self.refl)
        if self.mphase is not None:
            spec *= np.exp(1j * self.mphase(np.where(grid.valid, lam, 1.0)))
        spec *= grid.valid
        F = np.exp(-1j * (grid.r2[:, :, None] * (grid.omega / C0 / self.R)[None, None, :]))
        F *= spec[None, None, :]
        if self.diameter is not None:
            ap = np.exp(-(np.sqrt(grid.r2) / (self.diameter / 2)) ** 30)
            F *= ap[:, :, None]
        F = F.astype(grid.dtype)
        self._bounce_cache = (grid, F)
        return F

    def bounce(self, Aw, grid):
        Aw *= self.bounce_factor(grid)
        return Aw
