Performance and memory
======================

Lens frame
----------

The lens frame mostly saves **grid points**: the box holds the waist instead of the mirror spot,
so the number of points per axis is independent of the mirror separation (75 instead of 480 for
the tutorial cell, 75 instead of 1125 for an 18 mJ cell at :math:`d = 0.995 \cdot 2R`; the cost
scales with the square of that ratio). Each step costs one 2D FFT pair, 8 temporal 1D FFTs (RK4)
and a few phase multiplications built from 1D arrays (one fused pass each with Numba). The step is limited by both ``phi_max``
and ``gouy_step``: a pass takes at least :math:`\theta/\texttt{gouy\_step}` steps
(≈ 57 for :math:`\theta = 162°`), and more when the B-integral per pass exceeds
:math:`\theta\,\varphi_\text{max}/\texttt{gouy\_step}`.

Memory: the field, the RK4 buffers and FFT temporaries, about 8–10 field arrays. No linear
operator ``D`` and no propagator cache are stored. One ``complex64`` array is
``Nx·Ny·Nt·8`` bytes.

Standard solver
---------------

Each step costs 2 spatial 2D FFTs, 8 temporal 1D FFTs (RK4) and a few array operations.
With SciPy on one core, a 150 × 150 × 135 grid takes about 0.1–0.4 s per step. The number
of steps per pass scales as the B-integral per pass divided by ``phi_max``: a 20-pass run
at B ≈ 16 rad needs about 900 steps.

About 8–10 field arrays are held in memory: the field, the RK4 buffers, the linear
operator ``D``, up to ``cache_size`` cached propagators and the cached mirror transfer.
One ``complex64`` array is ``Nx·Ny·Nt·8`` bytes, so 256 × 256 × 256 is 134 MB.

.. _numba-kernels:

Numba kernels (CPU)
-------------------

On the CPU, the FFTs use all cores (``scipy.fft``, ``workers=-1``) but the NumPy array
operations between them (Kerr product, RK4 accumulation, phase multiplications) use one core
and allocate temporaries; in the RK4 they cost as much as the FFTs. With Numba installed
(``thunder4d[numba]``), both solvers replace them by fused kernels: one multithreaded pass over
memory per operation (Kerr product and peak intensity; RK4 stage; lens-frame kinetic and
potential phases, combined from their 1D factors; standard-solver propagator). The FFTs are
unchanged and the results are the same to rounding.

- ``kernels="auto"`` (default) picks Numba on a CPU grid when it is installed, NumPy otherwise;
  GPU grids always use CuPy. ``kernels="numpy"`` / ``"numba"`` force one or the other
  (solvers, ``LensFrameSimulation``, ``solver.kernels`` in input files). ``solver.kernels``
  tells which one runs.
- The first call compiles the kernels (a few seconds); they are then cached on disk.
- Threads: ``NUMBA_NUM_THREADS`` (default: all cores), independent of the FFT ``workers``.
- The gain grows with the core count, since only the FFT part was multithreaded before: on one
  core, about 1.6× on the RK4 sub-step and 1.3× on a lens-frame step (96 × 96 × 512, ``complex64``).
- A solver whose ``nonlinear`` is overridden keeps the NumPy RK4 (it calls ``nonlinear``);
  the linear steps still use the kernels.

.. _fftw-backend:

FFTW (CPU)
----------

With pyFFTW installed (``thunder4d[fftw]``), ``fft_backend="fftw"`` on the grid
(``suggest_lens_grid(..., fft_backend="fftw")``, ``grid.fft_backend`` in input files,
``--fft fftw`` for the benchmarks) runs the solvers' FFTs with FFTW instead of pocketfft. FFTW
measures the fastest algorithm for each transform of the grid once (``FFTW_MEASURE``: a few
seconds per grid, during the first step) and reuses it. Same results to rounding.

- On one core, a lens-frame step (96 × 96 × 512, ``complex64``, Numba kernels) goes from about
  520 to 410 ms; the spatial 2D transforms gain the most. Measure on your machine with the
  benchmarks (``ms/step``).
- Only the 3D fields of the solvers use FFTW; the diagnostics keep ``scipy.fft``. GPU grids use
  cuFFT whatever the setting.
- Threads: ``workers`` of the grid (default: all cores).
- Licence: pyFFTW is BSD, but its wheels include FFTW, which is GPL. That is why it is an
  optional extra and not the default.

Speed-ups (both solvers)
------------------------

- Numba installed, on the CPU (see above).
- SciPy backend with all cores (``fft_backend="scipy"``, ``workers=-1``; this is the
  default with ``"auto"``), or FFTW (``fft_backend="fftw"``, see above).
- ``record_every > 1``, or ``near_field=False``: near-field (collimated beam) diagnostics use per-colour
  matrix transforms and cost a few seconds per record.
- Fewer ``profile_wavelengths``.
- A smaller ``cache_size`` if memory is tight (standard solver).

Long runs
---------

Use ``run(max_seconds=…)`` together with ``checkpoint`` and ``resume``
(see :doc:`../how_to_run`).
