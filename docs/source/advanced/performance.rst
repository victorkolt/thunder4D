Performance and memory
======================

Lens frame
----------

The lens frame mostly saves **grid points**: the box holds the waist instead of the mirror spot,
so the number of points per axis is independent of the mirror separation (75 instead of 480 for
the tutorial cell, 75 instead of 1125 for an 18 mJ cell at :math:`d = 0.995 \cdot 2R`; the cost
scales with the square of that ratio). Each step costs one 2D FFT pair, 8 temporal 1D FFTs (RK4)
and a few phase multiplications built from 1D arrays. The step is limited by both ``phi_max``
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

Speed-ups (both solvers)
------------------------

- SciPy backend with all cores (``fft_backend="scipy"``, ``workers=-1``; this is the
  default with ``"auto"``).
- ``record_every > 1``, or ``near_field=False``: near-field (collimated beam) diagnostics use per-colour
  matrix transforms and cost a few seconds per record.
- Fewer ``profile_wavelengths``.
- A smaller ``cache_size`` if memory is tight (standard solver).

Long runs
---------

Use ``run(max_seconds=…)`` together with ``checkpoint`` and ``resume``
(see :doc:`../how_to_run`).
