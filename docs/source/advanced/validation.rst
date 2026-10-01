Grid choice, accuracy and validation
====================================

Choosing the grid
-----------------

- **Box.** Start from :func:`~thunder4d.grid.suggest_grid`. Use ``box_factor`` ≈ 8–10
  for clean Gaussians and ≥ 12–14 for top-hats, hard edges and aberrations.
- **Transverse step.** It must resolve both the waist and the curvature phase applied
  by the mirror at the edge of the box; ``suggest_grid`` handles both.
- **Spectral window.** It must contain the broadened spectrum with margin, because the
  Kerr source is switched off near the window edges. Set ``lambda_min`` and
  ``lambda_max`` at least 30–50 % wider than the expected −30 dB width.
- **Time window.** It must hold the chirped pulse at the end of the cell.

Accuracy
--------

- ``phi_max = 0.02`` rad is a good default. Check convergence by halving it.
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
4. **Convergence.** Halve ``phi_max`` and check that the output doesn't change.
5. **Nonlinear index.** Replace the tabulated :math:`n_2` with your own values.

.. warning::

   The boundary absorber acts as a spatial filter. With a box that is too small, the
   high-order content of top-hat or clipped beams is removed at each pass, which looks
   like spatial cleaning but is a numerical artefact.
