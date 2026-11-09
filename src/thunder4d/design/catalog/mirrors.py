"""Mirror / coating catalogue for the design helper.

A Coating carries what both the design helper and the solver need:
    reflectivity R(lambda), GDD(lambda) per bounce, usable band, LIDT and its duration scaling.
``coating.reflectivity`` and ``coating.phase`` plug directly into
``MPC(..., mirror_reflectivity=..., mirror_phase=...)``.

Data can be scalars (generic coatings) or tables loaded from the vendor curves (CSV with two
columns: wavelength in nm, value). Tables are interpolated, and set to the edge value outside
the tabulated range unless ``outside="zero"`` (reflectivity) is requested.

LIDT scaling with pulse duration:  LIDT(tau) = lidt_J_cm2 * (tau / lidt_ref_fs) ** lidt_exponent
(exponent ~0.3-0.5 is often quoted; set your own). The design helper evaluates the fluence limit
at the local pulse duration, so the last passes (shortest pulses) are the critical ones.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal, cast

import numpy as np
from scipy.constants import c as C0
from scipy.integrate import cumulative_trapezoid

Kind = Literal["dielectric", "metallic", "chirped", "other"]
Role = Literal["cavity", "compressor", "both"]


def load_curve(path, delimiter=None, skip_header=0, wavelength_unit="nm"):
    """Two-column table (wavelength, value) -> (wavelength_nm, value), sorted."""
    data = np.genfromtxt(path, delimiter=delimiter, skip_header=skip_header)
    lam, val = data[:, 0], data[:, 1]
    if wavelength_unit == "um":
        lam = lam * 1e3
    elif wavelength_unit == "m":
        lam = lam * 1e9
    i = np.argsort(lam)
    return lam[i], val[i]


@dataclass
class Coating:
    """One mirror coating (cavity or compressor).

    reflectivity     : scalar, or (wavelength_nm, R) table, R in [0, 1] (or in % with r_in_percent)
    gdd_fs2          : scalar, or (wavelength_nm, GDD in fs^2) table, per bounce
    band_nm          : usable band (min, max); the broadened spectrum must fit inside it
    lidt_J_cm2       : damage threshold (fluence) at lidt_ref_fs
    """
    name: str
    kind: Kind = "dielectric"
    role: Role = "cavity"
    band_nm: tuple[float, float] = (0.0, np.inf)
    reflectivity: float | tuple[np.ndarray, np.ndarray] = 1.0
    gdd_fs2: float | tuple[np.ndarray, np.ndarray] = 0.0
    lidt_J_cm2: float | None = None
    lidt_ref_fs: float = 1000.0
    lidt_exponent: float = 0.0
    pair: str | None = None              # complementary chirped mirror (e.g. L/S pair)
    diameters_inch: tuple[float, ...] = ()   # available sizes (empty = any)
    r_in_percent: bool = False
    vendor: str = ""
    source: str = ""                     # "vendor theoretical", "measured 2026-xx", ...
    notes: str = ""
    _phase_cache: tuple | None = field(default=None, repr=False, compare=False)

    # ------------------------------------------------------------ evaluation
    def R(self, lam_nm):
        lam_nm = np.asarray(lam_nm, dtype=float)
        if np.ndim(self.reflectivity) == 0:
            r = np.full_like(lam_nm, float(cast(float, self.reflectivity)))
        else:
            x, y = cast(tuple, self.reflectivity)
            r = np.interp(lam_nm, x, y)
        if self.r_in_percent:
            r = r / 100.0
        return np.clip(r, 0.0, 1.0)

    def gdd(self, lam_nm):
        lam_nm = np.asarray(lam_nm, dtype=float)
        if np.ndim(self.gdd_fs2) == 0:
            return np.full_like(lam_nm, float(cast(float, self.gdd_fs2)))
        x, y = cast(tuple, self.gdd_fs2)
        return np.interp(lam_nm, x, y)

    def in_band(self, lam_nm):
        lam_nm = np.asarray(lam_nm, dtype=float)
        return (lam_nm >= self.band_nm[0]) & (lam_nm <= self.band_nm[1])

    def lidt(self, duration_fs):
        """LIDT [J/cm^2] at the given pulse duration (None if unknown)."""
        if self.lidt_J_cm2 is None:
            return None
        return self.lidt_J_cm2 * (duration_fs / self.lidt_ref_fs) ** self.lidt_exponent

    # ------------------------------------------------------------ solver hooks (SI, lambda in m)
    def reflectivity_si(self, lam_m):
        """Callable for MPC(mirror_reflectivity=...)."""
        return self.R(np.asarray(lam_m) * 1e9)

    def phase(self, lam_m, lambda0_m=1030e-9):
        """Spectral phase per bounce [rad], double integral of GDD(omega); the constant and
        linear (group delay) parts are removed at lambda0, so only the dispersion is kept.
        Callable for MPC(mirror_phase=...): use functools.partial to fix lambda0_m.
        Only meaningful with a tabulated GDD; a scalar GDD is better passed as mirror_gdd_fs2."""
        lam_m = np.asarray(lam_m, dtype=float)
        w = 2 * np.pi * C0 / np.where(lam_m > 0, lam_m, 1.0) * 1e-15      # rad/fs
        w0 = 2 * np.pi * C0 / lambda0_m * 1e-15
        if self._phase_cache is None or self._phase_cache[0] != lambda0_m:
            if np.ndim(self.gdd_fs2) == 0:
                wg = np.linspace(w0 * 0.5, w0 * 1.5, 4001)
                g = np.full_like(wg, float(cast(float, self.gdd_fs2)))
            else:
                x, y = cast(tuple, self.gdd_fs2)
                wg = 2 * np.pi * C0 / (np.asarray(x) * 1e-9) * 1e-15
                i = np.argsort(wg)
                wg, g = wg[i], np.asarray(y, dtype=float)[i]
            gd = cumulative_trapezoid(g, wg, initial=0.0)
            gd -= np.interp(w0, wg, gd)
            ph = cumulative_trapezoid(gd, wg, initial=0.0)
            ph -= np.interp(w0, wg, ph)
            self._phase_cache = (lambda0_m, wg, ph)
        _, wg, ph = self._phase_cache
        return np.interp(w, wg, ph)

    def with_(self, **changes):
        return replace(self, _phase_cache=None, **changes)

    # ------------------------------------------------------------ I/O
    @classmethod
    def from_files(cls, name, reflectivity_csv=None, gdd_csv=None, *, delimiter=None,
                   skip_header=0, wavelength_unit="nm", **kw):
        """Coating from vendor curves (two-column files: wavelength, value)."""
        if reflectivity_csv is not None:
            kw["reflectivity"] = load_curve(reflectivity_csv, delimiter, skip_header, wavelength_unit)
        if gdd_csv is not None:
            kw["gdd_fs2"] = load_curve(gdd_csv, delimiter, skip_header, wavelength_unit)
        return cls(name, **kw)


# ============================================================ the catalogue
MIRRORS: dict[str, Coating] = {
    "dielectric_1030nm": Coating(
        "dielectric_1030nm", kind="dielectric", role="cavity",
        band_nm=(955.0, 1095.0), reflectivity=0.9995, gdd_fs2=0.0, lidt_J_cm2=0.4,
        source="Optoman", notes="Custom mirror (ask V. Koltalo for the reference)"),
    "dielectric_800nm": Coating(
        "dielectric_800nm", kind="dielectric", role="cavity",
        band_nm=(720.0, 880.0), reflectivity=0.999, gdd_fs2=0.0, lidt_J_cm2=0.2,
        source="Eksma", notes="062-8xxxB"),
    "enhanced_silver": Coating(
        "enhanced_silver", kind="metallic", role="cavity",
        band_nm=(600.0, 1300.0), reflectivity=0.985, gdd_fs2=0.0, lidt_J_cm2=0.1,
        source="Eksma", notes="092-3125FR, LIDT underestimated for safety"),
}


def register_mirror(coating: Coating, overwrite=False):
    if coating.name in MIRRORS and not overwrite:
        raise KeyError(f"coating {coating.name!r} already registered (overwrite=True to replace)")
    MIRRORS[coating.name] = coating


def get_mirror(name) -> Coating:
    try:
        return MIRRORS[name]
    except KeyError:
        raise KeyError(f"unknown coating {name!r}; available: {sorted(MIRRORS)}") from None


def cavity_coatings(names=None):
    names = names if names is not None else MIRRORS.keys()
    return [get_mirror(n) for n in names if get_mirror(n).role in ("cavity", "both")]


def load_catalog(path, overwrite=True):
    """Register coatings from a JSON file. Curve files are resolved relative to the JSON file.

    [
      {"name": "my_hr_1030", "kind": "dielectric", "role": "cavity", "band_nm": [950, 1110],
       "reflectivity_csv": "hr_R.csv", "gdd_csv": "hr_GDD.csv", "r_in_percent": true,
       "lidt_J_cm2": 0.4, "vendor": "...", "source": "vendor theoretical"},
      ...
    ]
    """
    path = Path(path)
    entries = json.loads(path.read_text())
    out = []
    for e in entries:
        e = dict(e)
        name = e.pop("name")
        r_csv, g_csv = e.pop("reflectivity_csv", None), e.pop("gdd_csv", None)
        fmt = {k: e.pop(k) for k in ("delimiter", "skip_header", "wavelength_unit") if k in e}
        for key in ("band_nm", "diameters_inch"):
            if key in e:
                e[key] = tuple(e[key])
        c = Coating.from_files(name, path.parent / r_csv if r_csv else None,
                               path.parent / g_csv if g_csv else None, **fmt, **e)
        register_mirror(c, overwrite=overwrite)
        out.append(c)
    return out
