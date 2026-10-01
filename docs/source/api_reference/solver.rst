thunder4d.solver
================

.. py:module:: thunder4d.solver

UPPE propagator. See :doc:`../overview/overview` for the equation and the scheme.

.. py:class:: UPPESolver(grid, medium, *, phi_max=0.02, dz_max=None, paraxial=False, self_steepening=True, absorber_frac=0.9, absorber_order=16, cache_size=4)

   :param float phi_max: maximum peak nonlinear phase per step [rad]; controls accuracy
   :param float dz_max: maximum step [m] (default: a quarter of each propagated segment)
   :param bool paraxial: use :math:`k_z \approx k - (k_x^2+k_y^2)/2k`
   :param bool self_steepening: frequency-dependent :math:`\gamma(\Omega)` or constant
      :math:`n_2\omega_0/c`
   :param float absorber_frac: soft boundary absorbers reach 1/e at this fraction of the
      half-window (spatial, spectral and temporal versions)
   :param int absorber_order: super-Gaussian order of the absorbers
   :param int cache_size: number of linear propagators :math:`e^{iDh}` kept in memory
      (each is one full field array)

   **State:** ``B`` (accumulated on-axis peak B-integral [rad]), ``nsteps``, ``z``
   (propagated distance [m]), ``k0``, ``k1``, ``n2``, ``D`` (linear operator), ``igamma``.

   .. py:method:: propagate(Aw, L)

      Propagate over ``L`` [m] with the adaptive split-step scheme and return the new
      field. If :math:`n_2 = 0`, a single exact linear step is used.

   .. py:method:: linear(Aw, h)

      Exact linear propagation over ``h`` [m]; ``h < 0`` propagates backwards.

   .. py:method:: nonlinear(Aw, track=False)

      Kerr right-hand side :math:`i\gamma\,\mathcal F_t[|A|^2A]`. With ``track=True`` it
      updates the peak intensity used for step control. Override it to add physics
      (see :doc:`../advanced/extending`).

   .. py:method:: peak_intensity(Aw)

      :math:`\max |A(x,y,t)|^2` [W/m²].

   .. py:method:: absorb(Aw, temporal=False)

      Apply the spatial and spectral absorbers (and the temporal one if requested).

   .. note::

      :class:`~thunder4d.simulation.MPCSimulation` applies the absorbers twice per pass
      (at the mirror and at the centre), not at every step, so their filtering effect is
      small and independent of the step count.
