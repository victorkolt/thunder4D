"""Catalogues used by the design helper: gases, mirror coatings and plate materials."""
from .gases import GASES, Gas, barrier_suppression_intensity, get_gas, register_gas
from .mirrors import MIRRORS, Coating, cavity_coatings, get_mirror, load_catalog, load_curve, register_mirror
from .solids import PLATES, PlateMaterial, get_plate, register_plate

__all__ = [
                      "GASES",
                      "MIRRORS",
                      "PLATES",
                      "Coating",
                      "Gas",
                      "PlateMaterial",
                      "barrier_suppression_intensity",
                      "cavity_coatings",
                      "get_gas",
                      "get_mirror",
                      "get_plate",
                      "load_catalog",
                      "load_curve",
                      "register_gas",
                      "register_mirror",
                      "register_plate",
]
