API reference
=============

Everything useful is available at package level:

.. code-block:: python

   import thunder4d as m
   from thunder4d import diagnostics as dg

   m.Grid, m.suggest_grid, m.C0
   m.Material
   m.Pulse, m.Beam, m.gaussian, m.super_gaussian, m.top_hat, m.knife_edge,
   m.measured_profile, m.gaussian_spectrum, m.measured_spectrum,
   m.spectrum_from_temporal_field, m.add_spectral_phase, m.zernike_wavefront
   m.MPCCell, m.UPPESolver, m.MPCSimulation

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Module
     - Content
   * - :doc:`grid`
     - simulation grid, FFT conventions, automatic grid choice
   * - :doc:`materials`
     - gas dispersion and nonlinear index
   * - :doc:`pulse`
     - spectra, spatial profiles, aberrations, space-time couplings, pulse builders
   * - :doc:`cell`
     - cell geometry, eigenmodes, nonlinear mode matching, mirrors
   * - :doc:`solver`
     - UPPE propagator
   * - :doc:`simulation`
     - pass loop, recording, checkpoints, saving
   * - :doc:`diagnostics`
     - spectrograms, profiles, modal content, homogeneity, M², compression

.. toctree::
   :maxdepth: 2

   grid
   materials
   pulse
   cell
   solver
   simulation
   diagnostics
