"""Nonlinear media (gases).

Dispersion: Borzsonyi et al., Appl. Opt. 47, 4856 (2008)
    n^2 - 1 = (p/p0)(T0/T) [B1 l^2/(l^2 - C1) + B2 l^2/(l^2 - C2)],  l in um, p0 = 1 bar, T0 = 273.15 K
n2 [m^2/W at 1 bar, room temperature, ~800 nm]: APPROXIMATE literature values (electronic
Kerr only, no Raman for N2/air). Replace them with the values of your get_material.m
through Material(..., n2_1bar=...).
"""
import numpy as np
from .grid import C0

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


class Material:
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

    def k(self, omega):
        return self.n(omega) * np.asarray(omega) / C0

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

    def __repr__(self):
        return f"Material({self.name}, {self.p} bar, n2 = {self.n2:.3g} m^2/W)"
