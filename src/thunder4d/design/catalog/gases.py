"""Gas catalogue for the design helper.

Adds to thunder4d.Material what the design needs and the solver does not: ionisation potential,
an ionisation threshold intensity, a cost class and a Raman flag. Dispersion and n2 still come
from thunder4d.materials (SELLMEIER, N2_1BAR), overridable here with n2_1bar.

Ionisation threshold
--------------------
The default threshold is a fraction of the barrier-suppression intensity
    I_BSI = 4e9 * Ip[eV]^4 / Z^2  W/cm^2   (Z = 1)
which is only an order of magnitude. The intensity at which ionisation starts to matter in an
MPC depends on the pulse duration and on how much loss/plasma you tolerate, so set
``ionization_threshold_W_cm2`` from your own experience or literature for each gas.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

from ...materials import N2_1BAR, SELLMEIER, Material


def barrier_suppression_intensity(ip_eV, Z=1):
    """Barrier-suppression intensity [W/cm^2]."""
    return 4e9 * ip_eV ** 4 / Z ** 2


@dataclass(frozen=True)
class Gas:
    name: str
    ip_eV: float
    raman_active: bool = False         # molecular gases: rotational Raman (not modelled yet)
    n2_1bar: float | None = None       # None -> thunder4d.materials.N2_1BAR
    ionization_threshold_W_cm2: float | None = None   # None -> bsi_fraction * I_BSI
    bsi_fraction: float = 0.2          # default threshold = bsi_fraction * I_BSI
    notes: str = ""

    @property
    def bsi_W_cm2(self):
        return barrier_suppression_intensity(self.ip_eV)

    @property
    def ionization_threshold(self):
        if self.ionization_threshold_W_cm2 is not None:
            return self.ionization_threshold_W_cm2
        return self.bsi_fraction * self.bsi_W_cm2

    @property
    def n2_per_bar(self):
        return self.n2_1bar if self.n2_1bar is not None else N2_1BAR[self.name]

    def material(self, pressure_bar, temperature_K=293.15, polarization_factor=1.0):
        """thunder4d.Material at this pressure (n2 scaled by the polarisation factor)."""
        return Material(self.name, pressure_bar, temperature_K,
                        n2_1bar=self.n2_per_bar * polarization_factor,
                        sellmeier=SELLMEIER.get(self.name))

    def with_(self, **changes):
        """Copy with some fields changed, e.g. GASES['Ar'].with_(n2_1bar=9.8e-24)."""
        return replace(self, **changes)


# Ionisation potentials: first ionisation energy (NIST). Air: O2 (lowest of the mixture).
GASES: dict[str, Gas] = {
    "He": Gas("He", 24.587),
    "Ne": Gas("Ne", 21.565),
    "Ar": Gas("Ar", 15.760),
    "Kr": Gas("Kr", 14.000),
    "Xe": Gas("Xe", 12.130),
    "N2": Gas("N2", 15.581, raman_active=True),
    "air": Gas("air", 12.07, raman_active=True,
               notes="Ip of O2; ionisation of air is dominated by O2"),
}


def register_gas(gas: Gas, overwrite=False):
    if gas.name in GASES and not overwrite:
        raise KeyError(f"gas {gas.name!r} already registered (overwrite=True to replace)")
    GASES[gas.name] = gas


def get_gas(name) -> Gas:
    try:
        return GASES[name]
    except KeyError:
        raise KeyError(f"unknown gas {name!r}; available: {sorted(GASES)}") from None
