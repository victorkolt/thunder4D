"""Plate catalogue for the design helper (bulk and hybrid MPCs).

Adds to thunder4d.Solid what the design needs and the solver does not: a peak-intensity threshold in
the bulk and a damage threshold (fluence) of the faces. Dispersion and n2 still come from
thunder4d.materials (SOLIDS), overridable here with n2.

Thresholds
----------
intensity_threshold_W_cm2 : peak intensity inside the plate above which multiphoton absorption /
    free-carrier generation starts to matter. 1e12 W/cm^2 is an order of magnitude for dielectrics
    with a wide band gap and pulses of a few 100 fs (published bulk MPCs run at a few 1e11 W/cm^2),
    NOT a damage threshold: set it from your own experience for each material and pulse duration.
lidt_J_cm2 : damage fluence of the faces (coated or bare) at lidt_ref_fs, scaled with the duration as
    the mirror coatings (lidt_exponent). None = unknown: the plate fluence is then not checked (a
    warning says so) unless PlateConstraints.fluence has an explicit value.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

from ...materials import SOLIDS, Solid


@dataclass(frozen=True)
class PlateMaterial:
    name: str                                      # key in thunder4d.materials.SOLIDS
    axis: str = "o"                                # uniaxial crystals: "o" | "e"
    n2: float | None = None                        # None -> thunder4d.materials.SOLIDS
    intensity_threshold_W_cm2: float | None = 1e12
    lidt_J_cm2: float | None = None                # faces, at lidt_ref_fs
    lidt_ref_fs: float = 1000.0
    lidt_exponent: float = 0.0
    notes: str = ""

    @property
    def n2_value(self):
        return self.n2 if self.n2 is not None else SOLIDS[self.name].n2

    def solid(self, polarization_factor=1.0):
        """thunder4d.Solid (n2 scaled by the polarisation factor)."""
        return Solid(self.name, axis=self.axis, n2=self.n2_value * polarization_factor)

    def lidt(self, duration_fs):
        """Damage fluence of the faces [J/cm^2] at this pulse duration (None if unknown)."""
        if self.lidt_J_cm2 is None:
            return None
        return self.lidt_J_cm2 * (duration_fs / self.lidt_ref_fs) ** self.lidt_exponent

    def with_(self, **changes):
        """Copy with some fields changed, e.g. PLATES['FS'].with_(lidt_J_cm2=1.0, lidt_ref_fs=200)."""
        return replace(self, **changes)


PLATES: dict[str, PlateMaterial] = {
    "FS": PlateMaterial("FS"),
    "CaF2": PlateMaterial("CaF2"),
    "BaF2": PlateMaterial("BaF2"),
    "MgF2": PlateMaterial("MgF2", notes="uniaxial: ordinary index (axis='o'), c-cut plate"),
    "sapphire": PlateMaterial("sapphire", notes="uniaxial: ordinary index (axis='o'), c-cut plate"),
    "YAG": PlateMaterial("YAG"),
}


def register_plate(material: PlateMaterial, overwrite=False):
    if material.name not in SOLIDS:
        raise KeyError(f"{material.name!r} is not in thunder4d.materials.SOLIDS: add its Sellmeier data there first")
    if material.name in PLATES and not overwrite:
        raise KeyError(f"plate material {material.name!r} already registered (overwrite=True to replace)")
    PLATES[material.name] = material


def get_plate(name) -> PlateMaterial:
    try:
        return PLATES[name]
    except KeyError:
        raise KeyError(f"unknown plate material {name!r}; available: {sorted(PLATES)}") from None
