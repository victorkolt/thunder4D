Installation
============

Requirements
------------

- Python ≥ 3.10
- NumPy ≥ 2.0 (single-precision FFTs)
- SciPy (installed automatically): ``scipy.fft`` is about 8× faster than
  ``numpy.fft`` for the 2D transforms and can use several cores; ``numpy.fft`` is used
  as a fallback if SciPy is missing
- Matplotlib (only for the example scripts)

No GPU and no compiled extension are needed. For an NVIDIA GPU, install CuPy
(``pip install cupy-cuda12x``; RTX 50-series cards need CUDA 12.8 or newer): grids created with
``device="auto"`` (the default) then run on the GPU, see :func:`thunder4d.gpu_available`.

Installing with pip
-------------------

.. code-block:: bash

   pip install thunder4d                 # numpy + scipy
   pip install "thunder4d[examples]"     # + matplotlib, for the example scripts

Installing from source
----------------------

To get the example scripts or modify the code, clone the repository and install it in
editable mode:

.. code-block:: bash

   git clone https://github.com/victorkolt/thunder4d.git
   cd thunder4d
   pip install -e ".[examples]"

Checking the installation
-------------------------

.. code-block:: python

   import thunder4d as m

   cell = m.MPC.herriott(R=0.5, N=20, k=3, branch="planar", medium=m.Material("Ar", 1.0))
   grid = m.suggest_grid(cell, 1030e-9, 900e-9, 1200e-9, 200e-15)
   print(grid.describe())
   print(grid.fft.backend)      # 'scipy' if SciPy was found, else 'numpy'

Repository layout
-----------------

.. code-block:: text

   thunder4d/
   ├── README.md
   ├── thunder4d/
   │   ├── __init__.py        public API
   │   ├── grid.py            Grid, FFT wrapper, suggest_grid
   │   ├── materials.py       Material (Sellmeier + n2)
   │   ├── pulse.py           spectra, profiles, aberrations, STCs, Beam, Pulse
   │   ├── cell.py            MPC (geometry, eigenmode, nonlinear matching, mirrors)
   │   ├── solver.py          UPPESolver
   │   ├── simulation.py      MPCSimulation (pass loop, recording, checkpoints, saving)
   │   └── diagnostics.py     analysis functions
   ├── examples/
   │   ├── compare_gauss_tophat.py
   │   ├── tophat_profiles_vs_pass.py
   │   └── clipped_gauss_profiles.py
   └── docs/                  this documentation (Sphinx)
