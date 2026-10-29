The lens frame
==============

:class:`~thunder4d.lensframe.LensFrameSimulation` propagates the field on a grid that
**follows the beam** inside the cell: the grid expands from the cell centre to the mirrors
exactly like the cell eigenmode does. It is the recommended solver. This page explains
the transformation, why it works, how it is solved and what it changes for the user.

Why a moving grid
-----------------

In a fixed (laboratory) grid the box must hold the beam on the mirrors, while the
transverse step must resolve it at the waist. The ratio of the two sizes is

.. math::

   s_m = \frac{w_\text{mirror}}{w_0} = \sqrt{1 + \left(\frac{d}{2 z_R}\right)^2},
   \qquad z_R = \frac{\sqrt{d(2R-d)}}{2},

which grows without bound as the cell approaches the concentric limit
:math:`d \to 2R`, i.e. precisely the high-energy cells. On top of this, the step must also
sample the curvature phase :math:`\omega r^2/(cR)` applied by the mirror at the corner of
the box, which shrinks :math:`dx` further as the box grows. For the toy cell of the tutorial
(:math:`R = 0.5` m, :math:`d = 0.976` m, :math:`s_m = 6.4`) the fixed grid needs 480 points
per transverse axis for 75 in the lens frame, i.e. ~40 times fewer points per plane; for the
18 mJ cell of the Kaumanns benchmark (:math:`s_m = 14.6`) it is 1125 against 75, ~225 times
fewer. In the lens frame the number of points per axis is
:math:`\approx` ``box_factor`` × ``points_per_waist`` × :math:`\sqrt{\lambda_\max/\lambda_\min}`,
independent of :math:`d`.

The transformation
------------------

The physical envelope :math:`A(x, y, z, \Omega)` is written as

.. math::

   A(x, y, z) = \frac{1}{s(z)}\; B\!\left(\frac{x}{s(z)}, \frac{y}{s(z)}, \zeta\right)\;
   \exp\!\left(\frac{i\,k(\omega)\, r^2}{2\,\mathcal{R}(z)}\right),

with the geometric quantities of the cell eigenmode

.. math::

   s(z) = \sqrt{1 + z^2/z_R^2}, \qquad
   \mathcal{R}(z) = z + \frac{z_R^2}{z}, \qquad
   \zeta = z_R \arctan\frac{z}{z_R},

:math:`z` being measured from the cell centre. :math:`\xi = x/s` and :math:`\eta = y/s` are the
frame coordinates; :math:`\zeta` is the propagation variable (proportional to the Gouy phase of
the eigenmode). Three properties make this the natural frame of a two-mirror cell:

- **Same Rayleigh range for every colour.** The eigenmode of the cell at any wavelength has
  the same :math:`z_R` (it only depends on :math:`R` and :math:`d`), hence the same
  :math:`s(z)` and :math:`\mathcal{R}(z)`. Only the waist :math:`w_0^2 = 2 z_R/k` depends on
  the colour, and :math:`k(\omega)` in the phase factor makes the transformation exact for
  every frequency.
- **The mirror is the identity.** At the mirror, :math:`\mathcal{R}(d/2) = R`: the frame
  wavefront is that of the eigenmode, diverging with the mirror radius. Reflection on the
  curved mirror turns it into the converging wavefront :math:`\mathcal{R}(-d/2) = -R`, which is
  again the frame wavefront on the way back. In the frame, a bounce is therefore only the
  spectral response of the mirror (reflectivity, GDD, TOD, measured phase) and its clear
  aperture, evaluated at the physical radius :math:`r = s_m\,\rho`.
- **At the cell centre** (:math:`z = 0`): :math:`s = 1`, :math:`\mathcal{R} = \infty`,
  :math:`\zeta = 0`, so :math:`B = A`. Every quantity recorded at the centre is physical.

Equation in the frame
---------------------

Inserting the ansatz into the paraxial envelope equation gives, exactly,

.. math::

   \frac{\partial B}{\partial \zeta}
   = \underbrace{\frac{i}{2k}\nabla_\perp^2 B}_\text{diffraction}
   \;\underbrace{-\;\frac{i\,k\,\rho^2}{2 z_R^2}\, B}_\text{harmonic potential}
   \;+\; \underbrace{i\,\beta(\Omega)\,s^2\, B}_\text{dispersion}
   \;+\; \underbrace{i\,\gamma(\Omega)\,\mathcal{F}_t\!\left[|B|^2 B\right]}_\text{Kerr},
   \qquad \rho^2 = \xi^2 + \eta^2,

with :math:`\beta = k(\omega) - k_0 - k_1\Omega` (gas dispersion in the frame moving at the group
velocity at :math:`\lambda_0`) and the same :math:`\gamma(\Omega)` as the standard solver
(self-steepening included).

- **Diffraction + potential** form a 2D harmonic oscillator. Its stationary states are the
  Hermite–Gauss eigenmodes of the cell, and :math:`\mathrm{HG}_{mn}` acquires the phase
  :math:`(m+n+1)\,\zeta/z_R`. Over one pass, :math:`\Delta\zeta/z_R = 2\arctan(d/2z_R) = \theta`,
  the Gouy phase per pass: the linear physics of the cell (re-entrance, imaging) is built in.
- **Dispersion** keeps its physical length: :math:`s^2\,d\zeta = dz`.
- **Kerr** is unchanged: :math:`|A|^2 dz = |B|^2 s^{-2}\cdot s^2 d\zeta`. The B-integral in the
  frame is :math:`\int k_0 n_2 |B|^2_\text{max}\, d\zeta`, identical to the physical one.

Numerical scheme
----------------

:class:`~thunder4d.lensframe.LensFrameSolver` splits each step :math:`h` in :math:`\zeta` as

.. code-block:: text

   [diffraction + dispersion](h/2)  [potential(h/2)  Kerr-RK4(h)  potential(h/2)]  [diffraction + dispersion](h/2)

- diffraction and dispersion are applied in :math:`(k_x, k_y, \Omega)` (one 2D FFT pair per step,
  consecutive halves merged), dispersion with the exact :math:`\Delta z` of the half-step;
- the potential is a phase in :math:`(\xi, \eta, \Omega)`;
- the Kerr step is the same fourth-order Runge–Kutta as in the standard solver.

The step is :math:`h = \min\left(\texttt{gouy\_step}\cdot z_R,\;
\varphi_\text{max}/(k_0 n_2 I_\text{peak})\right)`: ``gouy_step`` (default 0.05 rad) bounds the
Gouy phase per step and controls the accuracy of diffraction, ``phi_max`` (default 0.02 rad)
bounds the nonlinear phase per step. A pass is

.. code-block:: text

   centre (zeta = 0) ──▶ mirror (zeta_m) │ bounce: spectral response + aperture │ (-zeta_m) ──▶ centre

with :math:`\zeta_m = z_R\arctan(d/2z_R)`, and the soft absorbers applied after the bounce and at
the centre, as in the standard solver. The phase factors are rebuilt each step from 1D arrays,
so no full-size linear operator or propagator cache is stored.

Using it
--------

.. code-block:: python

   grid = m.suggest_lens_grid(cell, lambda0=1030e-9, lambda_min=900e-9, lambda_max=1200e-9,
                              pulse_fwhm=200e-15, box_factor=6)       # box in waists
   pulse = m.Pulse.at_focus(grid, E, m.Beam(m.gaussian(w0)), energy)  # defined at the centre
   sim = m.LensFrameSimulation(pulse, cell, phi_max=0.02, gouy_step=0.05, mode_reference=w0)
   sim.run()

- **Grid.** :func:`~thunder4d.lensframe.suggest_lens_grid`: box = ``box_factor`` × waist at
  ``lambda_max`` (≈ 6 for matched Gaussians, 10–14 for top-hats, hard edges or strong mismatch),
  :math:`dx` = waist at ``lambda_min`` / ``points_per_waist``.
- **Inputs.** :meth:`Pulse.at_focus <thunder4d.pulse.Pulse.at_focus>` and
  :meth:`Pulse.from_near_field <thunder4d.pulse.Pulse.from_near_field>` define the field at the
  centre, where frame and physical fields coincide: they work unchanged. A near field imaged on
  the mirror uses :func:`~thunder4d.lensframe.imaged_near_field_lens`, whose profile is given in
  **physical** coordinates at the image plane; in the frame it is only rescaled,
  :math:`B(\rho) = s\,U(s\rho)`, then propagated back to the centre.
- **Records.** Everything recorded at the centre is physical (spectra, :math:`S(x,\lambda)`,
  near field :math:`S(\theta,\lambda)`, modal content, :math:`M^2`, profiles). Mirror records
  (``fluence_mirror``, ``prof_mirror``) are in frame coordinates:

  .. code-block:: python

     s = sim.mirror_scale                       # = s_m
     x_mirror = sim.prof_coord * s              # physical positions on the mirror
     F_mirror = sim.history["fluence_mirror"][p] / s**2     # physical fluence [J/m^2]

Validity and checks
-------------------

- The frame equation is **paraxial** (exact within the paraxial approximation); the standard
  :class:`~thunder4d.solver.UPPESolver` is non-paraxial. For MPC beams (divergence of a few
  mrad) the difference is negligible; the standard solver remains available for checks.
- It assumes the symmetric two-mirror cell of :class:`~thunder4d.cell.MPC`, in the unfolded
  on-axis picture (like the standard solver).
- The test suite (``tests/test_lensframe.py``) checks that the linear eigenmode is unchanged by one
  pass (overlap > 0.9999) and that, for a Kerr-matched 1 mJ pulse in a near-concentric argon cell,
  the B-integral, energy, HG₀₀ fraction and beam size agree with the standard solver within 0.5 %
  and the output spectra have a similarity > 0.9999. ``benchmark/lens_frame_check.py`` runs the
  same comparison on a finer grid, on the CPU or the GPU.
- The box absorber acts in the frame, i.e. on a box that is ``box_factor`` waists wide at the
  centre and ``box_factor`` × :math:`s_m` waists wide on the mirror. Check the energy column of
  the run and, in the benchmarks, ``edge_fraction``.
