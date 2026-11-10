Accuracy and performance
========================

Choosing the grid
-----------------

- **Box.** ``box_factor`` of :func:`~thunder4d.lensframe.suggest_lens_grid` counts waists at
  ``lambda_max``: 6–8 for clean Gaussians, 10–14 for top-hats, hard edges, aberrations or a breathing
  beam. (With :func:`~thunder4d.grid.suggest_grid` it counts mirror spots: 8–10, or 12–14.)
- **Step.** ``points_per_waist`` (8) sets :math:`dx` from the waist at ``lambda_min``.
- **Spectral window.** The Kerr source is switched off near its edges: make it 30–50 % wider than the
  expected −30 dB width.
- **Time window.** It must hold the chirped pulse at the end of the cell.

.. warning::

   The box absorber is a spatial filter. A box that is too small removes the high orders of a top-hat
   or clipped beam at every pass, which looks like spatial cleaning but is an artefact.

Checklist
---------

Before trusting a new configuration:

1. **Linear imaging.** With ``Material(..., n2=0)`` in a re-entrant cell, the output overlaps the input
   to better than 0.99, whatever the profile.
2. **Energy.** A loss of more than ~1 % during the run means the box or the spectral window is too small.
3. **B-integral.** For a matched Gaussian, the final B is close to ``cell.match(pulse)["B_total"]``.
4. **Convergence.** Halve ``phi_max`` and ``gouy_step``: the output must not change. ``complex64`` is
   enough for a few tens of rad; check with ``complex128``.
5. **n₂.** Replace the tabulated values with yours (``Material(..., n2_1bar=...)``).
6. **Solvers.** In a new regime, run a few passes with both solvers: they agree to a fraction of a
   percent.

Cost
----

- **Lens frame:** one 2D FFT pair, 8 temporal FFTs (RK4) and a few phase products per step. A pass
  takes at least :math:`\theta/\texttt{gouy\_step}` steps (57 for :math:`\theta = 162°`), more when the
  B-integral per pass exceeds :math:`\theta\,\varphi_\text{max}/\texttt{gouy\_step}`.
- **Standard solver:** two 2D FFTs and 8 temporal FFTs per step, about 0.1–0.4 s on one core for
  150 × 150 × 135; a 20-pass run at B ≈ 16 rad takes about 900 steps.
- **Memory:** 8–10 field arrays of ``Nx·Ny·Nt·8`` bytes (``complex64``): 256³ is 134 MB each. The
  standard solver also stores its linear operator and up to ``cache_size`` propagators.
- **Faster:** a GPU, Numba and FFTW on the CPU (below), SciPy FFTs on all cores (the default),
  ``record_every > 1``, ``near_field=False``, fewer ``profile_wavelengths``.

.. _numba-kernels:

Numba kernels (CPU)
-------------------

With Numba installed (``thunder4d[numba]``), both solvers replace the NumPy operations between the FFTs
(Kerr product, RK4 stages, phase products) by fused multithreaded kernels. The results are the same to
rounding.

- ``kernels="auto"`` (default) uses them on a CPU grid when Numba is installed. ``"numpy"`` or
  ``"numba"`` forces one, also as ``solver.kernels`` in input files. ``solver.kernels`` says which runs.
- The first call compiles them in a few seconds. They are then cached on disk.
- Threads: ``NUMBA_NUM_THREADS`` (default all cores).
- On one core, 1.6× on an RK4 sub-step and 1.3× on a lens-frame step (96 × 96 × 512). The gain grows
  with the core count.
- A solver with an overridden ``nonlinear`` keeps the NumPy RK4. Its linear steps still use the kernels.

.. _fftw-backend:

FFTW (CPU)
----------

With pyFFTW installed (``thunder4d[fftw]``), ``fft_backend="fftw"`` on the grid (``grid.fft_backend`` in
input files, ``--fft fftw`` for the benchmarks) runs the solvers' FFTs with FFTW. The results are the
same to rounding.

- FFTW plans each transform once, in a few seconds during the first step.
- On one core, a lens-frame step (96 × 96 × 512, Numba kernels) goes from 520 to 410 ms.
- Only the 3D fields of the solvers use FFTW. GPU grids use cuFFT.
- Threads: ``workers`` of the grid (default all cores).
- The pyFFTW wheels include FFTW, which is GPL.
