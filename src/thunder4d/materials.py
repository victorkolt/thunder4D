"""Nonlinear media: gases (Material) and solids (Solid).

GASES

Dispersion: Borzsonyi et al., Appl. Opt. 47, 4856 (2008)
    n^2 - 1 = (p/p0)(T0/T) [B1 l^2/(l^2 - C1) + B2 l^2/(l^2 - C2)],  l in um, p0 = 1 bar, T0 = 273.15 K
n2 [m^2/W at 1 bar, room temperature, ~800 nm]: APPROXIMATE literature values (electronic
Kerr only, no Raman for N2/air). Replace them with the values of your get_material.m
through Material(..., n2_1bar=...).

SOLIDS
Dispersion: 3-term Sellmeier  n^2 - 1 = sum B_i l^2/(l^2 - C_i^2),  l and C_i in um (references in
SOLIDS). Uniaxial crystals (MgF2, sapphire) have one table per axis: the scalar solver sees ONE
index, 'o' by default (c-cut plate at normal incidence; at Brewster with the c axis along the plate
normal the beam also sees a little of n_e: choose axis='e' or a custom n_func if that matters).
n2 [m^2/W, ~1 um, room temperature]: APPROXIMATE literature values (spread between authors is
10-30 %); pass your own with Solid("FS", n2=...).
"""
import warnings
from typing import NamedTuple

import numpy as np
from scipy.constants import c as C0

SELLMEIER = {
    "He": (4977.77e-8, 28.54e-6, 1856.94e-8, 7.76e-3),
    "Ne": (9154.48e-8, 656.97e-6, 4018.63e-8, 5.728e-3),
    "Ar": (20332.29e-8, 206.12e-6, 34458.31e-8, 8.066e-3),
    "Kr": (26102.88e-8, 2.01e-6, 56946.82e-8, 10.043e-3),
    "Xe": (103701.61e-8, 12.75e-3, 31228.61e-8, 0.561e-3),
    "N2": (39209.95e-8, 1146.24e-6, 18806.48e-8, 13.476e-3),
    "air": (14926.44e-8, 19.36e-6, 41807.57e-8, 7.434e-3),
}

N2_1BAR = {"He": 3.5e-25, "Ne": 8.5e-25, "Ar": 1.0e-23, "Kr": 2.8e-23,
           "Xe": 8.1e-23, "N2": 7.4e-24, "air": 8.0e-24}


class _Medium:
    """Common interface of every medium: n(omega), k(omega), n2 [m^2/W], and helpers."""

    name: str
    n2: float

    def n(self, omega):
        raise NotImplementedError

    def k(self, omega):
        return self.n(omega) * np.asarray(omega) / C0

    def n_lam(self, lam):
        """Refractive index at wavelength lam [m]."""
        return self.n(2 * np.pi * C0 / np.asarray(lam, dtype=float))

    def group_index(self, lam):
        w = 2 * np.pi * C0 / lam
        h = w * 1e-4
        return float((self.k(w + h) - self.k(w - h)) / (2 * h) * C0)

    def gvd(self, lam):
        """GVD [fs^2/m] at wavelength lam."""
        w = 2 * np.pi * C0 / lam
        h = w * 1e-3
        k2 = (self.k(w + h) - 2 * self.k(w) + self.k(w - h)) / h ** 2
        return k2 * 1e30

    def critical_power(self, lam):
        """Self-focusing critical power of a Gaussian beam, Pc = lam^2 / (2 pi n0 n2) [W]."""
        n0 = self.n(2 * np.pi * C0 / lam)
        return lam ** 2 / (2 * np.pi * n0 * self.n2)


class Material(_Medium):
    """Homogeneous gas filling the cell.

    Material("Ar", pressure_bar=0.13)                       # table values
    Material("Ar", 0.13, n2_1bar=9.8e-24)                   # your own n2
    Material("custom", n_func=lambda omega: ..., n2=...)    # anything else (n2 absolute, m^2/W)
    """

    def __init__(self, name, pressure_bar=1.0, temperature_K=293.15, *, n2_1bar=None,
                 sellmeier=None, n_func=None, n2=None):
        self.name = name
        self.p = float(pressure_bar)
        self.T = float(temperature_K)
        self._n_func = n_func
        if n_func is None:
            self.sellmeier = sellmeier if sellmeier is not None else SELLMEIER[name]
        if n2 is not None:
            self.n2 = float(n2)
        else:
            n2b = n2_1bar if n2_1bar is not None else N2_1BAR[name]
            self.n2 = n2b * self.p * (293.15 / self.T)

    def n(self, omega):
        omega = np.asarray(omega, dtype=float)
        if self._n_func is not None:
            return np.asarray(self._n_func(omega), dtype=float)
        B1, C1, B2, C2 = self.sellmeier
        ok = omega > 0
        l2 = np.where(ok, (2 * np.pi * C0 / np.where(ok, omega, 1.0) * 1e6) ** 2, 1.0)
        n2m1 = (self.p / 1.0) * (273.15 / self.T) * (B1 * l2 / (l2 - C1) + B2 * l2 / (l2 - C2))
        return np.where(ok, np.sqrt(1 + n2m1), 1.0)

    def __repr__(self):
        return f"Material({self.name}, {self.p} bar, n2 = {self.n2:.3g} m^2/W)"


def vacuum():
    """n = 1, n2 = 0: the 'gas' of a bulk MPC in vacuum."""
    return Material("vacuum", n_func=np.ones_like, n2=0.0)


# --------------------------------------------------------------------------------- solids
class SolidData(NamedTuple):
    sellmeier: dict          # {axis: ((B1, C1), (B2, C2), ...)}, C in um
    n2: float                # m^2/W
    range_um: tuple          # validity range of the Sellmeier table
    ref: str


SOLIDS = {
    "FS": SolidData({"o": ((0.6961663, 0.0684043), (0.4079426, 0.1162414), (0.8974794, 9.896161))},
           n2=2.2e-20, range_um=(0.21, 6.7), ref="Malitson, JOSA 55, 1205 (1965)"),
    "CaF2": SolidData({"o": ((0.5675888, 0.050263605), (0.4710914, 0.1003909), (3.8484723, 34.649040))},
             n2=1.9e-20, range_um=(0.23, 9.7), ref="Malitson, Appl. Opt. 2, 1103 (1963)"),
    "BaF2": SolidData({"o": ((0.643356, 0.057789), (0.506762, 0.10968), (3.8261, 46.3864))},
             n2=2.9e-20, range_um=(0.27, 10.3), ref="Malitson, JOSA 54, 628 (1964)"),
    "MgF2": SolidData({"o": ((0.48755108, 0.04338408), (0.39875031, 0.09461442), (2.3120353, 23.793604)),
                           "e": ((0.41344023, 0.03684262), (0.50497499, 0.09076162), (2.4904862, 23.771995))},
             n2=0.9e-20, range_um=(0.2, 7.0), ref="Dodge, Appl. Opt. 23, 1980 (1984)"),
    "sapphire": SolidData({"o": ((1.4313493, 0.0726631), (0.65054713, 0.1193242), (5.3414021, 18.028251)),
                               "e": ((1.5039759, 0.0740288), (0.55069141, 0.1216529), (6.5927379, 20.072248))},
                 n2=2.9e-20, range_um=(0.2, 5.0), ref="Malitson & Dodge, JOSA 62, 1405 (1972)"),
    "YAG": SolidData({"o": ((2.282, 0.108858), (3.27644, 16.8147))},
            n2=6.2e-20, range_um=(0.4, 5.0), ref="Zelmon et al., Appl. Opt. 37, 4933 (1998)"),
}
SOLID_ALIASES = {"fused silica": "FS", "SiO2": "FS", "Al2O3": "sapphire"}


class Solid(_Medium):
    """Solid plate material.

    Solid("FS")                          # table values
    Solid("MgF2", axis="e")              # extraordinary index of a uniaxial crystal
    Solid("CaF2", n2=1.7e-20)            # your own n2 [m^2/W]
    Solid("myglass", n_func=lambda omega: ..., n2=...)
    Outside range_um the Sellmeier formula is extrapolated (n^2 clipped to >= 1 near the poles);
    the field of a well-sized grid is zero there anyway.
    """

    def __init__(self, name, *, axis="o", n2=None, n_func=None, sellmeier=None):
        key = SOLID_ALIASES.get(name, name)
        self.name, self.axis, self._n_func = key, axis, n_func
        entry = SOLIDS.get(key)
        if n_func is None:
            if sellmeier is None:
                if entry is None:
                    raise ValueError(f"unknown solid {name!r}: available {list(SOLIDS)}, or give "
                                     f"sellmeier=((B1, C1), ...) or n_func")
                if axis not in entry.sellmeier:
                    raise ValueError(f"{key} has axes {list(entry.sellmeier)}, not {axis!r}")
                sellmeier = entry.sellmeier[axis]
            self.sellmeier = tuple((float(b), float(c)) for b, c in sellmeier)
        if n2 is None:
            if entry is None:
                raise ValueError(f"give n2 for the custom solid {name!r}")
            n2 = entry.n2
        self.n2 = float(n2)
        self.range_um = entry.range_um if entry and sellmeier is None else None
        self.p = None          # no pressure (keeps the attribute set of Material)

    def n(self, omega):
        omega = np.asarray(omega, dtype=float)
        if self._n_func is not None:
            return np.asarray(self._n_func(omega), dtype=float)
        ok = omega > 0
        l2 = np.where(ok, (2 * np.pi * C0 / np.where(ok, omega, 1.0) * 1e6) ** 2, 1.0)
        n2m1 = sum(b * l2 / (l2 - c * c) for b, c in self.sellmeier)
        return np.where(ok, np.sqrt(np.maximum(1 + n2m1, 1.0)), 1.0)

    def check_range(self, lam_min, lam_max):
        """Warn if [lam_min, lam_max] (m) leaves the validity range of the Sellmeier table."""
        if self.range_um and (lam_min * 1e6 < self.range_um[0] or lam_max * 1e6 > self.range_um[1]):
            warnings.warn(f"{self.name}: {lam_min*1e9:.0f}-{lam_max*1e9:.0f} nm leaves the Sellmeier "
                          f"range {self.range_um} um", stacklevel=2)

    def __repr__(self):
        ax = f", axis {self.axis}" if self.axis != "o" else ""
        return f"Solid({self.name}{ax}, n2 = {self.n2:.3g} m^2/W)"
