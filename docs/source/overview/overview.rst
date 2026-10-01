Overview of the code
====================

This page describes what thunder4d computes and how. The conventions used throughout
the code (Fourier signs, array ordering, units) are in :doc:`conventions`.

.. toctree::
   :maxdepth: 1

   conventions

Representation of the field
---------------------------

The laser field is a complex envelope :math:`A(x, y, t)` with carrier
:math:`\exp[i(k_0 z - \omega_0 t)]`, normalised so that :math:`|A|^2` is the intensity
in W/m². It is stored on a 3D grid and propagated along :math:`z`, which gives a 4D
:math:`(x, y, z, t)` simulation. No symmetry is assumed: astigmatism, clipping, spatial
chirp or tilts are all represented exactly.

Propagation equation
--------------------

The field is propagated with the unidirectional pulse propagation equation (UPPE),
in the frame moving at the group velocity at :math:`\lambda_0`:

.. math::

   \frac{\partial \tilde A(k_x, k_y, \Omega)}{\partial z}
   = i\left[k_z(k_x, k_y, \omega) - k_0 - k_1\Omega\right]\tilde A
   + i\,\gamma(\Omega)\,\mathcal{F}_t\!\left[|A|^2 A\right]

with

.. math::

   k_z = \sqrt{k(\omega)^2 - k_x^2 - k_y^2},\qquad
   k(\omega) = \frac{n(\omega)\,\omega}{c},\qquad
   \omega = \omega_0 + \Omega,\qquad
   \gamma(\Omega) = \frac{n_2\,\omega}{c}\,\frac{n(\omega_0)}{n(\omega)} .

- **Linear operator.** It is exact: diffraction without the paraxial approximation,
  dispersion of the gas to all orders, and evanescent components (which decay).
- **Kerr term.** Instantaneous electronic response. The frequency dependence of
  :math:`\gamma` gives self-steepening; it can be switched off
  (:math:`\gamma = n_2\omega_0/c`).
- **Spectral window.** The nonlinear source is tapered to zero at the edges of the
  spectral window to avoid aliasing.

Numerical scheme
----------------

The solver keeps the field in the mixed domain :math:`(x, y, \Omega)`:

- the **linear step** only needs 2D spatial FFTs, since :math:`\Omega` is already a
  coordinate;
- the **Kerr term** does not depend on :math:`(k_x, k_y)`, so it only needs 1D
  temporal FFTs.

The two are combined with symmetric **Strang splitting** (linear half-step, nonlinear
step, linear half-step), merging consecutive linear half-steps. The nonlinear step is a
fourth-order **Runge–Kutta** step. The step size is adaptive:

.. math::

   h = \frac{\varphi_\text{max}}{k_0\, n_2\, I_\text{peak}},

limited to ``dz_max`` (default: a quarter of each propagated segment) and quantised to
``dz_max · 2^(−n/4)`` so that the linear propagators :math:`e^{iDh}` can be cached and
reused.

The multipass cell
------------------

Unfolded picture
~~~~~~~~~~~~~~~~

The cell is treated in the **unfolded** picture. One pass is

.. code-block:: text

   cell centre ──(d/2)──▶ curved mirror ──(d/2)──▶ cell centre

so :math:`N` passes means :math:`N` mirror bounces. Diagnostics are recorded at the cell
centre after every pass. That plane is the waist of the cell eigenmode with a flat
wavefront, so a Hermite–Gauss decomposition there is exact. Data can also be recorded
at the mirror plane, just before each bounce.

Each bounce multiplies the field in :math:`(x, y, \Omega)` by

.. math::

   e^{-i\omega r^2/(cR)}\;
   \sqrt{R_\text{refl}(\lambda)}\;
   e^{\,i\left[\frac{\text{GDD}}{2}\Omega^2 + \frac{\text{TOD}}{6}\Omega^3
   + \varphi_\text{mirror}(\lambda)\right]}\;
   \mathcal{A}(r),

i.e. an achromatic curved mirror (focal length :math:`R/2`), its spectral response,
and an optional soft clear aperture :math:`\mathcal{A}(r)`.

Linear eigenmode
~~~~~~~~~~~~~~~~

For a symmetric cell with :math:`g = 1 - d/R`, the Gouy phase per pass is
:math:`\theta = \arccos g` and, in the medium (:math:`\lambda \to \lambda/n_0`):

.. math::

   w_0^2 = \frac{\lambda}{2\pi}\sqrt{d(2R-d)},\qquad
   w_\text{mirror}^2 = \frac{\lambda R}{\pi}\sqrt{\frac{d}{2R-d}},\qquad
   z_R = \frac{\sqrt{d(2R-d)}}{2}.

A Herriott cell with :math:`N` round trips and integer :math:`k` is **re-entrant**
when :math:`d = R\,[1-\cos(\pi k/N)]`. Each mode :math:`\mathrm{HG}_{mn}` picks up
:math:`(m+n)\theta` per pass, so after :math:`2N` passes every mode is back in phase.

.. important::

   A linear re-entrant cell **images any input onto the output**, top-hats included.
   Spatial cleaning can therefore only come from the nonlinearity, from losses
   (apertures) or from imperfections. Running a configuration with :math:`n_2 = 0` and
   checking that the output reproduces the input is the first validation to do.

Nonlinear mode matching
~~~~~~~~~~~~~~~~~~~~~~~

In the aberration-free Kerr-lens approximation, for a uniformly filled cell and a power
:math:`P` constant during a pass, the envelope equation of a Gaussian beam reduces to
the linear one with

.. math::

   \lambda \;\to\; \lambda_\text{eff} = \lambda\sqrt{1 - P/P_c},
   \qquad P_c = \frac{\lambda^2}{2\pi n_0 n_2}.

The matched waist and mirror spot are the linear ones multiplied by
:math:`(1-P/P_c)^{1/4}`. The on-axis peak B-integral per pass is

.. math::

   B_\text{pass} = \frac{8\pi n_2 P}{\lambda\,\lambda_\text{eff}}
   \arctan\!\frac{d}{2 z_R'},\qquad z_R' = \frac{\pi w_0^2}{\lambda_\text{eff}},
   \qquad B_\text{pass}\approx 2\theta\,\frac{P}{P_c}\ \ (P \ll P_c).

A pulse can only be matched at **one** power: the temporal wings see a mode closer to
the linear one. This time-dependent mismatch is one of the space-time couplings that
the 4D simulation resolves.

What is not modelled
--------------------

- ionisation and plasma (MPCs are usually operated well below the threshold),
- Raman response of molecular gases (N₂, air),
- the folded off-axis geometry (astigmatism from the incidence angle, real spot
  pattern),
- non-uniform media (plates, windows, pressure gradients),
- vector effects (the field is scalar).

See :doc:`../advanced/extending` for where to add them.
