"""Catalogues used by the design helper: gases and mirror coatings."""
from .gases import GASES, Gas, barrier_suppression_intensity, get_gas, register_gas
from .mirrors import MIRRORS, Coating, cavity_coatings, get_mirror, load_catalog, load_curve, register_mirror

__all__ = [
                      "GASES",
                      "MIRRORS",
                      "Coating",
                      "Gas",
                      "barrier_suppression_intensity",
                      "cavity_coatings",
                      "get_gas",
                      "get_mirror",
                      "load_catalog",
                      "load_curve",
                      "register_gas",
                      "register_mirror",
]
