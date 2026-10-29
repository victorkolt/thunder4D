Grid choice, accuracy and validation
====================================

Choosing the grid
-----------------

- **Box (lens frame).** Start from :func:`~thunder4d.lensframe.suggest_lens_grid`; ``box_factor``
  is in units of the waist at ``lambda_max``. Use ≈ 6–8 for clean Gaussians and 10–14 for
  top-hats, hard edges, aberrations or a strongly mismatched (breathing) beam. On the mirror the
  same box is :math:`s_m` times larger in physical units.
- **Transverse step (lens frame).** ``points_per_waist`` (default 8) sets :math:`dx` from the
  waist at ``lambda_min``. There is no mirror-curvature constraint in the frame.
- **Standard solver.** With :func:`~thunder4d.grid.suggest_grid`, ``box_factor`` is in units
  of the mirror spot (≈ 8–10 for Gaussians, ≥ 12–14 otherwise), and :math:`dx` must also
  sample the mirror curvature phase at the edge of the box; ``suggest_grid`` handles both.
- **Spectral window.** It must contain the broadened spectrum with margin, because the
  Kerr source is switched off near the window edges. Set ``lambda_min`` and
  ``lambda_max`` at least 30–50 % wider than the expected −30 dB width.
- **Time window.** It must hold the chirped pulse at the end of the cell.

Accuracy
--------

- ``phi_max = 0.02`` rad is a good default. Check convergence by halving it.
- In the lens frame, ``gouy_step = 0.05`` rad (Gouy phase per step) bounds the diffraction
  step. Check it by halving it too, together with a larger ``points_per_waist``.
- ``complex64`` is accurate enough for typical runs (B of a few tens of rad). Use
  ``complex128`` to check, or for very long runs.

Validation checklist
--------------------

Run these checks on every new configuration before interpreting results:

1. **Linear imaging.** With ``Material(..., n2=0)`` and a re-entrant cell, the output
   after :math:`2N` passes must overlap the input to better than 0.99, for any input
   profile.
2. **Energy.** Watch the ``E = …`` column during the run. A loss of more than ~1 %
   means the box absorber is filtering the beam spatially or spectrally. Enlarge the box
   or the spectral window before interpreting any "cleaning".
3. **B-integral.** For a matched Gaussian, the final ``B`` should be close to
   ``cell.match(pulse)["B_total"]``.
4. **Convergence.** Halve ``phi_max`` (and ``gouy_step`` in the lens frame) and check that
   the output doesn't change.
5. **Nonlinear index.** Replace the tabulated :math:`n_2` with your own values.
6. **Solver cross-check.** For a new regime, run a few passes with both solvers
   (``benchmark/lens_frame_check.py`` is a template): B, energy, HG₀₀ fraction and spectrum
   should agree to a fraction of a percent.

Benchmarks against experiments
------------------------------

``benchmark/`` contains standalone scripts that reproduce published MPC experiments with the
lens-frame solver (Grebing *et al.* 2020, Kaumanns *et al.* 2018, Daniault *et al.* 2024) and
store a standard set of metrics next to the published values. ``benchmark/analyze.py`` prints
the comparison tables, compares two runs (convergence) and plots parameter sweeps. See
``benchmark/README.md`` for the options, the output format and how to add a benchmark.

.. warning::

   The boundary absorber acts as a spatial filter. With a box that is too small, the
   high-order content of top-hat or clipped beams is removed at each pass, which looks
   like spatial cleaning but is a numerical artefact.
