Quickstart
==========

Install
-------

.. code-block:: bash

   pip install thunder4d               # or: uv add thunder4d
   pip install "thunder4d[examples]"   # matplotlib, for the examples and the plots of the design helper
   pip install "thunder4d[gui]"        # graphical interface: thunder4d-gui
   pip install "thunder4d[gpu]"        # CuPy, for an NVIDIA GPU
   pip install "thunder4d[numba]"      # Numba, faster CPU runs
   pip install "thunder4d[fftw]"       # pyFFTW, FFTW transforms on the CPU

Python 3.10 or newer, NumPy and SciPy. With CuPy installed, grids run on the GPU by default
(``device="auto"``; RTX 50-series cards need CUDA 12.8 or newer). ``thunder4d.gpu_available()`` says
whether it works. On the CPU, Numba and pyFFTW make runs faster (:ref:`numba-kernels`,
:ref:`fftw-backend`).

Units and conventions
---------------------

.. list-table::
   :widths: 25 75

   * - Units
     - SI (m, s, J, W), unless the name says otherwise: ``_fs``, ``_fs2``, ``_nm``, ``_mm``, ``_bar``
   * - Field
     - complex envelope, :math:`|A|^2` is the intensity in W/m²; ``grid.energy(Aw)`` in J
   * - Arrays
     - shape ``(Nx, Ny, Nt)`` in :math:`(x, y, \Omega)`, **FFT order** (index 0 is :math:`x = 0`,
       :math:`\Omega = 0`); sorted axes for plotting: ``grid.xs``, ``grid.ts``, ``grid.lams``, ...
   * - Spectral phase
     - :math:`\varphi = +\frac{\text{GDD}}{2}\Omega^2`: positive GDD is an up-chirp
   * - Transforms
     - :math:`\tilde A(\Omega) = \sum_t A(t)\,e^{+i\Omega t}` (``grid.fft.ft_t``); plane waves
       :math:`e^{+ik_x x}`, propagation :math:`e^{+ik_z z}`
   * - Passes
     - a pass is a transit over the mirror separation :math:`d` followed by a bounce; the run starts
       right after a mirror and diagnostics are taken at the cell centre

Every function of :mod:`thunder4d.diagnostics` returns sorted axes, and densities per unit wavelength.

From an input file
------------------

One JSON file describes a simulation completely, so that the design helper, the
:doc:`graphical interface <gui>` and scripts all run the same thing.

.. code-block:: python

   import thunder4d as th
   from thunder4d import diagnostics as dg

   cfg = th.default_input(
       laser={"energy_J": 1e-3, "duration_fs": 200},
       medium={"gas": "Ar", "pressure_bar": 0.7},
       cell={"roc_m": 0.6, "separation_m": 1.183, "n_passes": 26},
       grid={"lambda_min_nm": 900, "lambda_max_nm": 1180},
   )
   th.save_input(cfg, "cell.json")                   # defaults filled in, checked
   b = th.run_input("cell.json")                     # b.grid, b.cell, b.pulse, b.sim
   print(dg.compress(b.grid, b.sim.Aw)["fwhm_fs"])   # 23.6 fs (coarse grid)
   b.sim.save("run.npz")

The format, every key and its default are in :doc:`api_reference/inputfile`.

From a script
-------------

The same in five steps, with every object at hand:

.. code-block:: python

   import thunder4d as m
   from thunder4d import diagnostics as dg

   # 1) medium and cell: near-concentric Herriott cell, 20 passes
   cell = m.MPC.herriott(R=0.5, N=20, k=1, medium=m.Material("Ar", pressure_bar=1.0))

   # 2) grid that follows the beam (lens frame): the box only holds the waist
   grid = m.suggest_lens_grid(cell, lambda0=1030e-9, lambda_min=900e-9, lambda_max=1200e-9,
                              pulse_fwhm=200e-15, box_factor=6)

   # 3) pulse, matched to the Kerr-lens eigenmode of the cell
   E = m.gaussian_spectrum(grid, fwhm_fs=200)
   probe = m.Pulse.at_focus(grid, E, m.Beam(m.gaussian(cell.linear_mode(1030e-9)["w0"])), 0.6e-3)
   mm = cell.match(probe)                      # prints P/Pc, matched waist, B per pass
   pulse = m.Pulse.at_focus(grid, E, m.Beam(m.gaussian(mm["w0"]), chromatic_size="eigenmode"), 0.6e-3)

   # 4) run
   sim = m.LensFrameSimulation(pulse, cell, mode_reference=mm["w0"])
   sim.run()
   sim.save("run.npz")

   # 5) analysis
   c = dg.compress(grid, sim.Aw)
   print(f"{c['fwhm_fs']:.1f} fs with {c['gdd_fs2']:.0f} fs^2")

One line is printed per pass; watch the energy column (a loss of more than about 1 % means the box is
too small):

.. code-block:: text

   pass  10/20 | B =  10.68 rad | E = 0.5948 mJ | eta00 = 0.8617 | V_far = 0.9775 | steps 656 | 162 s

``sim.history`` holds the diagnostics of every pass (:ref:`history-keys`) and ``sim.save`` writes them,
with the final field, to a ``.npz``. The :doc:`user guide <user_guide/index>` explains each step.

Examples
--------

``examples/tutorial.ipynb`` and ``examples/design_tutorial.ipynb`` are the interactive introductions.
The scripts of ``examples/`` compare a Gaussian with a top-hat, follow the profiles of single colours
pass after pass, and clip a Gaussian with a knife edge; each takes a few minutes on a CPU:

.. code-block:: bash

   uv run --extra examples python examples/compare_gauss_tophat.py
