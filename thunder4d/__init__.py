"""THUNDER4D - Transverse-resolved Herriott-cell UPPE solver for Nonlinear Dynamics close to Experimental Reality, 4D.

4D (x, y, z, t) UPPE simulation of multipass cells.
"""
__version__ = "0.1.0"
from .grid import Grid, suggest_grid, C0
from .materials import Material
from .pulse import (Pulse, Beam, gaussian, super_gaussian, top_hat, knife_edge, measured_profile,
                    hermite_gauss, laguerre_gauss, mode_superposition,
                    gaussian_spectrum, measured_spectrum, spectrum_from_temporal_field,
                    add_spectral_phase, zernike_wavefront)
from .cell import MPC
from .solver import UPPESolver
from .simulation import MPCSimulation
from . import diagnostics
