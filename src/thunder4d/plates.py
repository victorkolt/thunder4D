"""Solid plates inside a multipass cell (bulk and hybrid MPCs): coatings and plates.

Plates are at normal incidence and are crossed once per pass. For diffraction (paraxial), a
plate of thickness t and index n_p in a cell filled with n_g is equivalent to a stretch of the
filling medium of REDUCED length t n_g / n_p ("gas-referenced" length): the loaded cell is an
equivalent empty cell of separation d_eq = d - sum t (1 - n_g/n_p) (see MPC.equivalent_separation).
Dispersion and Kerr effect act over the physical thickness t.

Surfaces: the envelope is intensity-normalised, so one face multiplies it by sqrt(1 - R(lambda))
times the coating phase. 'bare' faces use the normal-incidence Fresnel reflectivity
((n_p - n_g)/(n_p + n_g))^2 with n(lambda); a Coating gives R(lambda) and an optional spectral phase.
"""
import numpy as np
from scipy.constants import c as C0, femto as FS


class Coating:
    """Coating of one plate face, as seen in TRANSMISSION.

    Coating.bare()                        # uncoated: Fresnel, from n(lambda)
    Coating.ideal_ar()                    # R = 0 at all wavelengths
    Coating(reflectivity=0.002)           # flat residual reflectivity
    Coating(reflectivity=lambda lam: ..., gdd_fs2=5.0)    # R(lambda [m]) and dispersion
    Coating.from_table(lam_m, R, phase_rad=None)          # measured / vendor curve (R in 0..1)
    gdd_fs2, tod_fs3 [fs^2, fs^3] and phase(lambda [m]) [rad] are added in transmission.
    """

    def __init__(self, reflectivity=0.0, *, gdd_fs2=0.0, tod_fs3=0.0, phase=None, name="coating",
                 _bare=False):
        self.reflectivity, self.gdd, self.tod, self.phase = reflectivity, float(gdd_fs2), float(tod_fs3), phase
        self.name, self.is_bare = name, _bare

    @classmethod
    def bare(cls):
        return cls(name="bare", _bare=True)

    @classmethod
    def ideal_ar(cls):
        return cls(0.0, name="ideal AR")

    @classmethod
    def from_table(cls, lam, R, phase_rad=None, name="table", **kw):
        """Linear interpolation of a tabulated curve (lam [m], R in 0..1, optional phase [rad]);
        constant extrapolation at the edges."""
        lam, R = np.asarray(lam, float), np.asarray(R, float)
        order = np.argsort(lam)
        lam, R = lam[order], R[order]

        def refl(x):
            return np.interp(x, lam, R)
        phase = None
        if phase_rad is not None:
            ph = np.asarray(phase_rad, float)[order]

            def phase(x):
                return np.interp(x, lam, ph)
        return cls(refl, phase=phase, name=name, **kw)

    @classmethod
    def parse(cls, spec):
        """'bare' | 'ar' | Coating -> Coating."""
        if isinstance(spec, Coating):
            return spec
        if spec is None or str(spec).lower() == "bare":
            return cls.bare()
        if str(spec).lower() in ("ar", "ideal_ar", "ideal ar"):
            return cls.ideal_ar()
        raise ValueError(f"coating {spec!r}: use 'bare', 'ar' or a Coating")

    def R(self, lam):
        lam = np.asarray(lam, float)
        r = self.reflectivity(lam) if callable(self.reflectivity) else np.full_like(lam, self.reflectivity)
        return np.clip(r, 0.0, 1.0)

    def spectral_phase(self, lam, lam0):
        """Phase added in transmission [rad] (GDD/TOD around lam0, plus the tabulated phase)."""
        lam = np.asarray(lam, float)
        W = 2 * np.pi * C0 * (1 / lam - 1 / lam0) * FS        # Omega in rad/fs
        ph = self.gdd / 2 * W ** 2 + self.tod / 6 * W ** 3
        if self.phase is not None:
            ph = ph + self.phase(lam)
        return ph

    def __repr__(self):
        return f"Coating({self.name})"


class Plate:
    """A plate at normal incidence, crossed once per pass.

    material   : Solid (or any medium with n(omega), n2)
    thickness  : [m]
    position   : signed distance [m] from the cell centre to the plate centre, along the cell axis,
                 POSITIVE TOWARDS MIRROR B (the mirror hit at the end of pass 1)
    coating    : 'bare' | 'ar' | Coating, or a pair (face A, face B); face A looks at mirror A
    """

    def __init__(self, material, thickness, position=0.0, *, coating="bare", name=None):
        self.material, self.t, self.position = material, float(thickness), float(position)
        if self.t <= 0:
            raise ValueError("plate thickness must be > 0")
        if isinstance(coating, (tuple, list)):
            self.coating_A, self.coating_B = Coating.parse(coating[0]), Coating.parse(coating[1])
        else:
            self.coating_A = self.coating_B = Coating.parse(coating)
        self.name = name or f"{getattr(material, 'name', 'plate')} {self.t*1e3:g} mm"

    def reduced_length(self, ambient, lam):
        """Length of filling medium with the same diffraction: t n_g / n_p."""
        return self.t * float(ambient.n_lam(lam) / self.material.n_lam(lam))

    def face_R(self, face, ambient, lam):
        """Power reflectivity of face 'A' or 'B' at wavelengths lam."""
        coat = self.coating_A if face == "A" else self.coating_B
        lam = np.asarray(lam, float)
        if not coat.is_bare:
            return coat.R(lam)
        n1, n2 = ambient.n_lam(lam), self.material.n_lam(lam)
        return ((n2 - n1) / (n2 + n1)) ** 2

    def face_transfer(self, face, ambient, lam, lam_ref):
        """Complex amplitude transmission of one face for the intensity-normalised envelope:
        sqrt(1 - R) exp(i phase); the same in both directions."""
        coat = self.coating_A if face == "A" else self.coating_B
        T = 1 - self.face_R(face, ambient, lam)
        return np.sqrt(T) * np.exp(1j * coat.spectral_phase(lam, lam_ref))

    def transmission(self, ambient, lam):
        """Power transmission of the plate (both faces, no absorption)."""
        return (1 - self.face_R("A", ambient, lam)) * (1 - self.face_R("B", ambient, lam))

    def describe(self, ambient, lam):
        T = float(self.transmission(ambient, lam))
        return (f"{self.name}: {self.material}, z = {self.position*1e3:+.1f} mm, faces "
                f"{self.coating_A.name}/{self.coating_B.name}, T(lambda0) = {T*100:.2f} %, "
                f"reduced length {self.reduced_length(ambient, lam)*1e3:.3f} mm")

    def __repr__(self):
        return f"Plate({self.name}, z = {self.position*1e3:+.1f} mm)"
