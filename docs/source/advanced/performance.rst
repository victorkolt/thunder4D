Performance and memory
======================

Cost
----

Each step costs 2 spatial 2D FFTs, 8 temporal 1D FFTs (RK4) and a few array operations.
With SciPy on one core, a 150 × 150 × 135 grid takes about 0.1–0.4 s per step. The number
of steps per pass scales as the B-integral per pass divided by ``phi_max``: a 20-pass run
at B ≈ 16 rad needs about 900 steps.

Memory
------

About 8–10 field arrays are held in memory: the field, the RK4 buffers, the linear
operator ``D``, up to ``cache_size`` cached propagators and the cached mirror transfer.
One ``complex64`` array is ``Nx·Ny·Nt·8`` bytes, so 256 × 256 × 256 is 134 MB.

Speed-ups
---------

- SciPy backend with all cores (``fft_backend="scipy"``, ``workers=-1``; this is the
  default with ``"auto"`` when SciPy is installed).
- ``record_every > 1``, or ``far_field=False``: far-field diagnostics use per-colour
  matrix transforms and cost a few seconds per record.
- Fewer ``profile_wavelengths``.
- A smaller ``cache_size`` if memory is tight.

Long runs
---------

Use ``run(max_seconds=…)`` together with ``checkpoint`` and ``resume``
(see :doc:`../how_to_run`).
