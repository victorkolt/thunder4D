API reference
=============

Everything useful is available at package level:

.. code-block:: python

   import thunder4d as m
   from thunder4d import diagnostics as dg

   m.Grid, m.suggest_grid, m.C0, m.gpu_available, m.to_host
   m.Material
   m.Pulse, m.Beam, m.gaussian, m.super_gaussian, m.top_hat, m.knife_edge,
   m.measured_profile, m.gaussian_spectrum, m.measured_spectrum,
   m.spectrum_from_temporal_field, m.add_spectral_phase, m.zernike_wavefront
   m.MPC, m.UPPESolver, m.MPCSimulation
   m.LensFrameSimulation, m.LensFrameSolver, m.suggest_lens_grid, m.imaged_near_field_lens, m.lens_frame_scale
   m.default_input, m.load_input, m.save_input, m.build_simulation, m.run_input

The design helper is a sub-package: ``from thunder4d.design import ...``.

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
   * - :doc:`lensframe`
     - beam-following (lens-frame) solver, grid and injection (recommended)
   * - :doc:`diagnostics`
     - spectrograms, profiles, modal content, homogeneity, M², compression
   * - :doc:`inputfile`
     - standard input file (JSON): load, check, save, build and run
   * - :doc:`design`
     - design helper: problem description, search, quick solvers, maps, export and 4D validation

.. toctree::
   :maxdepth: 2

   grid
   materials
   pulse
   cell
   solver
   simulation
   lensframe
   diagnostics
   inputfile
   design
