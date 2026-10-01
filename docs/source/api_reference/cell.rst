thunder4d.cell
==============

.. py:module:: thunder4d.cell

The multipass cell. See :doc:`../overview/overview` for the unfolded picture, the
eigenmode and the nonlinear mode matching.

.. py:class:: MPCCell(R, d, medium, n_passes, *, mirror_gdd_fs2=0.0, mirror_tod_fs3=0.0, mirror_reflectivity=1.0, mirror_phase=None, mirror_diameter=None)

   Symmetric two-mirror cell.

   :param float R: mirror radius of curvature [m]
   :param float d: mirror separation [m], :math:`0 < d < 2R`
   :param medium: :class:`~thunder4d.materials.Material`
   :param int n_passes: passes = bounces (centre → mirror → centre)
   :param float mirror_gdd_fs2: GDD per bounce [fs²]
   :param float mirror_tod_fs3: TOD per bounce [fs³]
   :param mirror_reflectivity: intensity reflectivity, scalar or callable
      ``R(lambda_m)``
   :param callable mirror_phase: ``phi(lambda_m)`` [rad] per bounce (e.g. a measured
      GDD curve)
   :param float mirror_diameter: clear aperture [m] (soft super-Gaussian edge, order 30);
      ``None`` = infinite

   **Attributes:** ``R``, ``d``, ``medium``, ``n_passes``, ``g`` (:math:`1 - d/R`),
   ``theta`` (Gouy phase per pass [rad]).

   .. py:classmethod:: herriott(R, N, k, medium, n_passes=None, **kw)

      Re-entrant Herriott cell: ``N`` round trips (= spots per mirror),
      :math:`d = R[1-\cos(\pi k/N)]`. ``n_passes`` defaults to :math:`2N`.

   .. py:method:: linear_mode(lam)

      Linear eigenmode at wavelength ``lam``.

      :returns: dict with ``w0``, ``w_mirror``, ``zR``, ``gouy_per_pass``

   .. py:method:: nonlinear_mode(P, lam=None, verbose=False)

      Kerr-matched eigenmode for power ``P`` [W]. Raises ``ValueError`` if
      :math:`P \ge P_c`.

      :returns: dict

      .. list-table::
         :widths: 35 65

         * - ``P, Pc, P_over_Pc``
           - power, critical power, ratio
         * - ``w0, w_mirror``
           - matched waist and mirror spot [m]
         * - ``w0_linear, w_mirror_linear``
           - linear values [m]
         * - ``B_per_pass, B_total``
           - on-axis peak B-integral estimates [rad]
         * - ``I_peak_center_W_cm2``
           - peak intensity at the waist [W/cm²]

   .. py:method:: match(pulse, power="peak", verbose=True)

      Same as :meth:`nonlinear_mode`, with the power taken from a
      :class:`~thunder4d.pulse.Pulse`: ``"peak"`` uses ``peak_power()``, ``"weighted"``
      uses ``weighted_power()``. A number in W is also accepted.

   .. py:method:: report(m=None)

      Text summary of the cell and, optionally, of a matching dict.

   .. py:method:: bounce_factor(grid)

      Complex transfer of one mirror, shape ``grid.shape`` (cached).

   .. py:method:: bounce(Aw, grid)

      Apply one mirror to ``Aw`` in place and return it.
