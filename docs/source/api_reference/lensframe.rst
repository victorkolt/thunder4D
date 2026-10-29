thunder4d.lensframe
===================

.. py:module:: thunder4d.lensframe

Beam-following ("lens-frame") propagation: the grid expands with the cell eigenmode, so it only
has to hold the beam at the waist scale. See :doc:`../overview/lens_frame` for the theory.

LensFrameSimulation
-------------------

.. py:class:: LensFrameSimulation(pulse, cell, *, phi_max=0.02, gouy_step=0.05, **kw)

   :class:`~thunder4d.simulation.MPCSimulation` running on a
   :class:`LensFrameSolver`. It accepts every keyword argument of ``MPCSimulation``
   (``record_every``, ``near_field``, ``max_order``, ``mode_reference``, ``save_fields``,
   ``record_mirror``, ``slit``, ``profile_wavelengths``, ``profile_bandwidth``,
   ``profile_axis``) and has the same methods (``run``, ``checkpoint``, ``resume``, ``save``),
   history keys (:ref:`history-keys`) and saved file.

   :param pulse: input :class:`~thunder4d.pulse.Pulse` on a lens-frame grid
      (:func:`suggest_lens_grid`), defined at the cell centre
   :param cell: :class:`~thunder4d.cell.MPC`
   :param float phi_max: maximum peak nonlinear phase per step [rad]
   :param float gouy_step: maximum Gouy phase per step [rad] (step in :math:`\zeta` at most
      ``gouy_step`` × :math:`z_R`)

   .. py:attribute:: mirror_scale

      :math:`s_m = \sqrt{1 + (d/2z_R)^2}`: scale of the grid at the mirror. Records at the cell
      centre are physical; mirror records (``fluence_mirror``, ``prof_mirror``) are in frame
      coordinates: physical position = ``mirror_scale`` × frame position, physical fluence =
      frame fluence / ``mirror_scale``\ ².

LensFrameSolver
---------------

.. py:class:: LensFrameSolver(grid, cell, *, phi_max=0.02, gouy_step=0.05, self_steepening=True, absorber_frac=0.9, absorber_order=16)

   Split-step solver in :math:`\zeta`: diffraction + dispersion (half steps, merged), harmonic
   potential, Kerr (RK4). Subclass of :class:`~thunder4d.solver.UPPESolver`, from which it
   inherits ``nonlinear``, ``peak_intensity`` and ``absorb``; the 3D linear operator of the
   standard solver is not built.

   **State:** ``B`` (accumulated on-axis peak B-integral [rad]), ``nsteps``, ``z`` (physical
   distance propagated [m]), ``zR``, ``zeta_m`` (:math:`\zeta` at the mirror), ``s_m``,
   ``h_max`` (= ``gouy_step`` × ``zR``), ``k0``, ``k1``, ``n2``.

   .. py:method:: propagate_zeta(Bw, zeta_a, zeta_b)

      Nonlinear propagation from :math:`\zeta_a` to :math:`\zeta_b` within one pass
      (:math:`|\zeta| \le \zeta_m`), adaptive step. Returns the new field.

   .. py:method:: propagate_linear_zeta(Bw, zeta_a, zeta_b)

      Linear propagation (no Kerr), forwards or backwards.

   .. py:method:: mirror(Bw)

      Bounce in the frame: spectral response of the mirror (reflectivity, GDD, TOD, measured
      phase) and clear aperture at the physical radius :math:`s_m\rho`. The curvature cancels.

   .. py:method:: z_of_zeta(zeta)

      :math:`z = z_R \tan(\zeta/z_R)`.

   ``propagate(Aw, L)`` is not available (it raises): use ``propagate_zeta``.

Functions
---------

.. py:function:: suggest_lens_grid(cell, lambda0, lambda_min, lambda_max, pulse_fwhm, *, box_factor=6.0, points_per_waist=8.0, time_window_factor=8.0, **grid_kw)

   Grid for the lens frame: box = ``box_factor`` × linear waist at ``lambda_max``,
   :math:`dx` = waist at ``lambda_min`` / ``points_per_waist``, time step from the spectral
   window, time window = ``time_window_factor`` × ``pulse_fwhm``. Other keywords go to
   :class:`~thunder4d.grid.Grid` (``device``, ``dtype``, …).

   Use ``box_factor`` ≈ 6 for matched Gaussians and 10–14 for top-hats, hard edges or a strong
   mismatch (their higher-order content is larger than the fundamental).

.. py:function:: imaged_near_field_lens(grid, E, beam, energy, cell, image_z=None, verbose=True)

   Lens-frame version of :meth:`Pulse.imaged_near_field <thunder4d.pulse.Pulse.imaged_near_field>`:
   near field imaged (same size for every colour) onto the plane at distance ``image_z`` from
   the centre (default: the mirror, :math:`d/2`) with the wavefront of the cell eigenmode, then
   propagated back to the centre in the frame. ``beam`` is given in **physical** coordinates at
   the image plane, e.g. ``Beam(top_hat(1.12 * w_mirror))``. Returns a
   :class:`~thunder4d.pulse.Pulse`.

.. py:function:: lens_frame_scale(cell)

   Returns ``(zR, zeta_m, s_m)``: Rayleigh range of the cell eigenmode, :math:`\zeta` at the
   mirror and the mirror scale factor.
