Installation
============

Requirements
------------

- Python ≥ 3.10
- NumPy ≥ 2.0 (single-precision FFTs)
- SciPy ≥ 1.10 (installed automatically): ``scipy.fft`` is about 8× faster than
  ``numpy.fft`` for the 2D transforms and can use several cores
- Matplotlib (only for the example scripts)

No GPU and no compiled extension are needed. For an NVIDIA GPU, install CuPy
(``uv pip install cupy-cuda12x``; RTX 50-series cards need CUDA 12.8 or newer): grids created with
``device="auto"`` (the default) then run on the GPU, see :func:`thunder4d.gpu_available`.

Installing with uv
------------------

The commands use `uv <https://docs.astral.sh/uv/>`_. In a uv project:

.. code-block:: bash

   uv add thunder4d                 # numpy + scipy
   uv add "thunder4d[examples]"     # + matplotlib, for the example scripts
   uv add "thunder4d[gpu]"          # + CuPy, for an NVIDIA GPU

In an existing environment (venv, conda), use ``uv pip install`` with the same arguments.

Installing from source
----------------------

To get the example scripts or modify the code, clone the repository and let uv create
the environment:

.. code-block:: bash

   git clone https://github.com/victorkolt/thunder4d.git
   cd thunder4d
   uv sync --extra examples          # .venv with thunder4d in editable mode
   uv run --extra examples python examples/compare_gauss_tophat.py
   uv run --extra examples --with jupyter jupyter lab    # tutorial and benchmark notebooks

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
   ├── CONTRIBUTING.md        development, documentation, releases
   ├── src/thunder4d/
   │   ├── __init__.py        public API
   │   ├── backend.py         CPU/GPU selection (NumPy or CuPy)
   │   ├── grid.py            Grid, FFT wrapper, suggest_grid
   │   ├── materials.py       Material (Sellmeier + n2)
   │   ├── pulse.py           spectra, profiles, aberrations, STCs, Beam, Pulse
   │   ├── cell.py            MPC (geometry, eigenmode, nonlinear matching, mirrors)
   │   ├── solver.py          UPPESolver
   │   ├── simulation.py      MPCSimulation (pass loop, recording, checkpoints, saving)
   │   └── diagnostics.py     analysis functions
   ├── examples/
   │   ├── tutorial.ipynb
   │   ├── compare_gauss_tophat.py
   │   ├── tophat_profiles_vs_pass.py
   │   └── clipped_gauss_profiles.py
   ├── benchmark/             comparisons with published experiments (notebooks)
   └── docs/                  this documentation (Sphinx)
