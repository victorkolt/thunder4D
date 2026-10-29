Extending the code
==================

New nonlinear terms
-------------------

Override :meth:`~thunder4d.solver.UPPESolver.nonlinear`. Ionisation, Raman or higher-order Kerr
terms go there, in the :math:`(x, y, \Omega) \leftrightarrow (x, y, t)` representation
already used for the Kerr term. :class:`~thunder4d.lensframe.LensFrameSolver` inherits
``nonlinear`` from :class:`~thunder4d.solver.UPPESolver`, so the same override works in both
solvers. In the lens frame the field is :math:`B`, and :math:`|B|^2` is the physical intensity
times :math:`s^2`: a term that depends on the intensity other than through the Kerr form must
include that factor (and :math:`d z = s^2 d\zeta`).

.. code-block:: python

   import numpy as np
   import thunder4d as m

   class RamanSolver(m.UPPESolver):
       def nonlinear(self, Aw, track=False):
           out = super().nonlinear(Aw, track)
           # A = self.grid.fft.ift_t(Aw); ... compute the extra polarisation ...
           # out += self.grid.fft.ft_t(extra) * self.igamma
           return out

   sim = m.MPCSimulation(pulse, cell, solver=RamanSolver(grid, gas))     # standard solver

   class RamanLensSolver(m.LensFrameSolver):
       nonlinear = RamanSolver.nonlinear                                # same physics

   sim = m.LensFrameSimulation(pulse, cell)
   sim.solver = RamanLensSolver(grid, cell, phi_max=0.02)               # before sim.run()

Extra optical elements
----------------------

Elements between passes (a lens phase, an aperture, a spectral filter, a window) can be
inserted by running the simulation pass by pass and modifying the field. ``sim.Aw`` is at the
cell centre, where the lens-frame field is the physical one:

.. code-block:: python

   aperture = (grid.r2 < (1.5e-3) ** 2)[:, :, None]
   for p in range(cell.n_passes):
       sim.run(n_passes=1, verbose=False)
       sim.Aw *= aperture

With the standard solver, ``sim.solver.propagate(Aw, L)`` and ``sim.solver.linear(Aw, h)`` can
be called directly for arbitrary propagation distances (``h < 0`` propagates backwards). In the
lens frame use ``sim.solver.propagate_zeta(Bw, zeta_a, zeta_b)`` and
``propagate_linear_zeta`` (:math:`\zeta = z_R\arctan(z/z_R)`).

Arbitrary inputs
----------------

Any spectral phase (``add_spectral_phase(phase=...)``), spatial profile
(:class:`~thunder4d.pulse.Profile`) or space-time coupling
(``Beam(custom_phase=...)``) can be given as a function.

Not yet modelled
----------------

- ionisation and plasma,
- Raman response of molecular gases,
- the folded off-axis geometry (incidence-angle astigmatism, real spot pattern),
- non-uniform media (plates, windows, pressure gradients); the nonlinear mode matching
  formula also assumes a uniformly filled cell,
- vector effects.
