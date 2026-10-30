"""THUNDER4D - Transverse-resolved Herriott-cell UPPE solver for Nonlinear Dynamics close to Experimental Reality, 4D.

4D (x, y, z, t) UPPE simulation of multipass cells.
"""
from importlib.metadata import version

from . import diagnostics
from .backend import gpu_available, to_host
from .cell import MPC
from .grid import C0, Grid, suggest_grid
from .lensframe import LensFrameSimulation, LensFrameSolver, imaged_near_field_lens, lens_frame_scale, suggest_lens_grid
from .materials import Material
from .pulse import (
    Beam,
    Pulse,
    add_spectral_phase,
    gaussian,
    gaussian_spectrum,
    hermite_gauss,
    knife_edge,
    laguerre_gauss,
    measured_profile,
    measured_spectrum,
    mode_superposition,
    spectrum_from_temporal_field,
    super_gaussian,
    top_hat,
    zernike_wavefront,
)
from .simulation import MPCSimulation
from .solver import UPPESolver

__version__ = version("thunder4d")
