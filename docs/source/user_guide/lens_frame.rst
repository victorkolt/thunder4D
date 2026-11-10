The lens frame
==============

:class:`~thunder4d.lensframe.LensFrameSimulation` propagates the field on a grid that expands from the
cell centre to the mirrors exactly like the cell eigenmode. It is the recommended solver.

Why
---

A fixed grid must hold the beam on the mirrors and resolve it at the waist. The ratio of the two is

.. math::

   s_m = \frac{w_\text{mirror}}{w_0} = \sqrt{1 + \left(\frac{d}{2 z_R}\right)^2},\qquad
   z_R = \frac{\sqrt{d(2R-d)}}{2},

which grows without bound towards the concentric limit, i.e. for the high-energy cells. A fixed grid
needs 480 points per axis for the tutorial cell (:math:`s_m = 6.4`) and 1125 for the 18 mJ Kaumanns cell
(:math:`s_m = 14.6`); the lens frame needs 75 for both. Its size, about ``box_factor`` ×
``points_per_waist`` × :math:`\sqrt{\lambda_\max/\lambda_\min}` points per axis, does not depend on
:math:`d`.

The transformation
------------------

With :math:`z` measured from the cell centre,

.. math::

   A(x, y, z) = \frac{1}{s}\, B\!\left(\frac{x}{s}, \frac{y}{s}, \zeta\right)
   \exp\!\left(\frac{i k r^2}{2\mathcal{R}}\right),\qquad
   s = \sqrt{1 + z^2/z_R^2},\quad \mathcal{R} = z + \frac{z_R^2}{z},\quad \zeta = z_R\arctan\frac{z}{z_R}.

- **Exact for every colour:** all wavelengths share the same :math:`z_R`, so the same :math:`s` and
  :math:`\mathcal{R}`.
- **The mirror is the identity:** at the mirror :math:`\mathcal{R} = R`, and the reflection turns the
  frame wavefront into the frame wavefront of the way back. A bounce is only the spectral response of
  the mirror and its aperture, taken at the physical radius :math:`s_m\rho`.
- **The centre is physical:** there :math:`s = 1` and :math:`B = A`, so every diagnostic recorded at the
  centre is unchanged. Mirror records are in frame units (``sim.mirror_scale`` converts them).

In the frame the paraxial envelope equation becomes, exactly,

.. math::

   \frac{\partial B}{\partial \zeta}
   = \frac{i}{2k}\nabla_\perp^2 B - \frac{i k \rho^2}{2 z_R^2} B + i\beta(\Omega)\,s^2 B
   + i\gamma(\Omega)\,\mathcal{F}_t\!\left[|B|^2 B\right].

Diffraction and the harmonic potential have the Hermite–Gauss modes of the cell as eigenstates, so the
Gouy phase per pass is built in. Dispersion keeps its physical length (:math:`s^2 d\zeta = dz`) and the
Kerr term, and with it the B-integral, is unchanged.

Scheme
------

Each step :math:`h` in :math:`\zeta`:

.. code-block:: text

   [diffraction + dispersion](h/2)  [potential(h/2)  Kerr-RK4(h)  potential(h/2)]  [diffraction + dispersion](h/2)

with :math:`h = \min(\texttt{gouy\_step}\cdot z_R,\ \varphi_\text{max}/(k_0 n_2 I_\text{peak}))`.
``gouy_step`` (0.05 rad) sets the accuracy of diffraction, ``phi_max`` (0.02 rad) that of the Kerr step.
The phase factors are rebuilt from 1D arrays, so no full-size operator is stored.

Limits
------

- The frame equation is paraxial; the standard solver is not. For MPC beams (a few mrad) the
  difference is negligible.
- It assumes the symmetric two-mirror cell, unfolded and on axis.
- The box absorber acts in the frame: ``box_factor`` waists at the centre, :math:`s_m` times more on the
  mirror.
- ``tests/test_lensframe.py`` checks that one pass leaves the eigenmode unchanged (overlap > 0.9999) and
  that, for a Kerr-matched 1 mJ pulse in a near-concentric cell, both solvers agree within 0.5 % on the
  B-integral, the energy, the HG₀₀ fraction and the beam size.
