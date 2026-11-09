Design helper
=============

The design helper goes from an **input laser and a target output** to a **ready-to-run MPC
configuration**: gas, pressure, mirror radius of curvature, cell length, number of round trips and
passes, with the margin to every constraint. It works in three layers, from cheap to accurate:

.. list-table::
   :header-rows: 1
   :widths: 18 22 60

   * - Layer
     - Cost per cell
     - What it does
   * - Analytic estimates
     - microseconds
     - Closed-form mode, :math:`P/P_c`, B-integral, intensity, fluence and an SPM estimate of the
       duration. Used to discard geometries, to rank starting points and to draw maps.
   * - **fast1d** (1+1D solver)
     - 30-100 ms
     - Time envelope coupled to a Gaussian beam followed through its complex :math:`q` parameter, with
       the full dispersion of the gas. Used to rank, tune and check the candidates.
   * - thunder4d (4D solver)
     - seconds to minutes
     - The full solver, run on the chosen design from a standard input file, and compared with fast1d.

The pages of this section:

.. toctree::
   :maxdepth: 1

   problem
   search
   solvers
   maps
   validation_4d

The interactive introduction is the notebook ``examples/design_tutorial.ipynb``; the objects are
listed in :doc:`../api_reference/design`.

.. note::

   The plots (``design_map(...).plot()``, ``validate_4d(...).plot()`` and the ``plot()`` of the
   fast1d output) need matplotlib, which is an optional dependency:
   ``uv add "thunder4d[examples]"``. Nothing else of the helper needs it.

Minimal example
---------------

18 mJ, 500 fs pulses at 1030 nm, to be compressed to 25 fs in argon or krypton. The limits are the
ones your hardware takes (see :doc:`problem` for what each one means and how to set it):

.. code-block:: python

   from thunder4d.design import DesignProblem, Fast1DQuick, LaserSpec, Max, MPCDesigner, Target, starting_points

   problem = DesignProblem(
       laser=LaserSpec(energy_J=18e-3, duration_fs=500.0, wavelength_nm=1030.0, rep_rate_Hz=1e3),
       target=Target(objective="target_duration", target_duration_fs=25.0),
   )
   C = problem.constraints
   C.space.max_separation_m.value = 5.0                 # m
   C.mirrors.allowed_roc_m = (1.0, 2.0, 3.0)            # the mirrors you have or can buy
   C.mirrors.allowed_coatings = ["dielectric_1030nm"]
   C.mirrors.fluence.value = 0.7                        # J/cm2 (None: the coating LIDT)
   C.kerr.b_per_pass_rad.value = 3.0                    # rad, on axis, per pass
   C.ionization.peak_intensity_W_cm2 = Max(3.5e13)      # W/cm2
   C.gas.allowed = ("Ar", "Kr")
   problem.search.round_trips = (8, 25)
   problem.search.branches = ("concentric",)
   problem.refinement.n_candidates = 3

   if __name__ == "__main__":
       designer = MPCDesigner(problem, quick=Fast1DQuick(), workers=4)
       result = designer.refine(starting_points(problem))
       print(result.table())
       print(result.best.report(problem))
       result.export("design.json")                     # standard THUNDER4D input file

The ``if __name__ == "__main__":`` guard is needed on Windows when ``workers > 1`` (in a notebook it
is not). The table and the report of the best design are:

.. include:: example_output.txt

The best design is argon at 170 mbar, :math:`R = 2` m, 15 round trips, :math:`k = 2` and 29 passes:
the cell that the limits above were calibrated on. Its only violated limit is soft, the fraction of
the energy in the main peak (0.63 for a floor of 0.70): the pulse keeps a pedestal, which is typical
of a strongly broadened spectrum compressed with GDD only. The result is a ranked list; each design can be exported, simulated, and
compared with fast1d (:doc:`validation_4d`).

Geometry and pass convention
----------------------------

A Herriott cell is set by its radius of curvature :math:`R`, the number of round trips :math:`M` of
one re-entrant cycle and the index :math:`k` (coprime with :math:`M`, :math:`k < M/2`):

.. math::

   d = R\left(1 + \cos\frac{\pi k}{M}\right)\ \text{(concentric branch, } d > R\text{)},\qquad
   d = R\left(1 - \cos\frac{\pi k}{M}\right)\ \text{(planar branch, } d < R\text{)}.

One cycle is :math:`2M` passes and puts :math:`M` spots on each mirror. The number of passes
actually used, :math:`N`, can be anything from ``SearchSpace.min_passes`` to :math:`2M`: with an
odd :math:`N` the beam leaves on the other mirror. For example, "15 round trips, :math:`k = 2`" is
:math:`M = 15`, :math:`N \le 30`.

The pass convention is the same in fast1d and in the 4D solver (:doc:`../overview/overview`): the
simulation starts right after a mirror, and one pass is a transit over :math:`d` followed by a
bounce on the next mirror, so :math:`N` passes are :math:`N` transits and :math:`N` bounces.

Scope and limits
----------------

The helper is a **pre-design tool**: it proposes cells that are worth simulating, it does not
replace the 4D simulation. Know what it does not contain:

- one stage; the compressor is an ideal GDD removal (the optimal GDD is reported, not realised
  with a given mirror pair or grating pair),
- a Gaussian beam: no clipping on the mirror edges, no aberrations, no spatio-temporal couplings,
  no mode mismatch beyond the Kerr lens,
- no Raman response and no ionisation loss (the intensity at the waist is *limited*, not
  simulated), so N\ :sub:`2` and air are flagged with a warning,
- the margins of the ionisation and fluence limits are only as good as the thresholds you give
  (:doc:`problem`, "Catalogues").

:doc:`solvers` quantifies how far fast1d can be trusted and :doc:`validation_4d` shows the
comparison with the 4D solver on a real cell.
