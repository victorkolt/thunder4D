Installation
============

Requirements
------------

- Python ≥ 3.10
- NumPy ≥ 2.0 (single-precision FFTs)
- SciPy (optional, strongly recommended): ``scipy.fft`` is about 8× faster than
  ``numpy.fft`` for the 2D transforms and can use several cores
- Matplotlib (only for the example scripts)

No GPU and no compiled extension are needed.

Getting the code
----------------

Clone the repository (or unzip the archive) and either add its root folder to your
``PYTHONPATH``, or install it in editable mode:

.. code-block:: bash

   git clone https://github.com/<your-account>/thunder4d.git
   cd thunder4d
   pip install numpy scipy matplotlib
   pip install -e .          # optional, if you add a pyproject.toml

Without installation, the examples add the parent folder to ``sys.path`` themselves:

.. code-block:: python

   import sys
   sys.path.insert(0, "path/to/thunder4d")
   import thunder4d as m

Checking the installation
-------------------------

.. code-block:: python

   import thunder4d as m

   cell = m.MPCCell.herriott(R=0.5, N=10, k=3, medium=m.Material("Ar", 1.0))
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
   │   ├── cell.py            MPCCell (geometry, eigenmode, nonlinear matching, mirrors)
   │   ├── solver.py          UPPESolver
   │   ├── simulation.py      MPCSimulation (pass loop, recording, checkpoints, saving)
   │   └── diagnostics.py     analysis functions
   ├── examples/
   │   ├── compare_gauss_tophat.py
   │   ├── tophat_profiles_vs_pass.py
   │   └── clipped_gauss_profiles.py
   └── docs/                  this documentation (Sphinx)
