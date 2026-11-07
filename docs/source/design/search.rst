Searching for a design
======================

The search has three steps, which :meth:`MPCDesigner.run <thunder4d.design.designer.MPCDesigner.run>`
chains and which can be called one by one:

.. code-block:: text

   starting_points()  ->  screen()  ->  refine()
   enumerate and       rank with the    run the quick solver, tune the
   prefilter           analytic tier    best candidates, return n designs

Starting points
---------------

:func:`~thunder4d.design.designer.starting_points` enumerates the geometries allowed by the
``search`` block and the constraints:

.. list-table::
   :header-rows: 1
   :widths: 28 16 56

   * - ``problem.search``
     - Default
     - Meaning
   * - ``gases``, ``roc_m``, ``diameters_inch``, ``coatings``
     - None
     - restrict the lists of the constraints (``None``: all that is allowed)
   * - ``branches``
     - both
     - ``"concentric"`` and/or ``"planar"``
   * - ``round_trips``
     - (3, 40)
     - inclusive range of the number of round trips :math:`M`
   * - ``k_range``
     - None
     - restrict :math:`k` (``None``: every :math:`k < M/2` coprime with :math:`M`)
   * - ``pass_offsets``
     - (0, -1)
     - passes tried per geometry: :math:`2M` + offset (the full cycle, and one pass less, leaving on
       the other mirror)
   * - ``min_passes``
     - 5
     - never fewer passes

``starting_points(problem, k="all")`` tries every valid :math:`k`; ``k="smallest"`` keeps one per
:math:`M` (far fewer cells, faster). ``gases=`` and ``round_trips=`` override the
search block for one call.

Before anything is simulated, a **prefilter** removes the geometries that a hard limit already
excludes, from the linear cell mode and the input pulse: the separation, the stability, the fluence
on the mirrors, the ionisation threshold and the two spot-pattern limits. It is conservative: the
Kerr effect is allowed to help (it can shrink the spots by :math:`(1 - P/P_c)^{1/4}`), so no cell that
could pass is rejected. The number of geometries removed by each limit is printed.

Each remaining geometry starts at an analytic guess of the pressure, the lowest of:

- the pressure that gives :math:`P/P_c = 0.3`,
- 90 % of the B-per-pass limit, if it is hard,
- for ``target_duration`` and ``target_compression``: the pressure whose total B-integral gives the
  compression that the target needs, from the SPM estimate.

Score and ranking
-----------------

A design is judged by :func:`~thunder4d.design.designer.evaluate`, the same for every solver:

.. math::

   \text{score} = \text{objective} + \sum_{\text{soft limits}} w\,(\text{margin})^2

A violated **hard** limit sets the score to infinity: the design is rejected (it stays in the list,
ranked after the feasible ones by how far it is from feasibility). The objective of each
``Target.objective``:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Objective
     - Value
   * - ``"shortest"``
     - :math:`\tau / \tau_\text{in}`
   * - ``"max_peak_power"``
     - :math:`-P_\text{out} / P_\text{in}`
   * - ``"target_duration"``
     - :math:`((\tau - \tau_t)/\tau_t)^2 + 10^{-3}(1 - T)`
   * - ``"target_compression"``
     - :math:`((\tau_\text{in}/\tau - C)/C)^2 + 10^{-3}(1 - T)`
   * - ``"weighted"``
     - :math:`w_\tau \tau/\tau_\text{in} - w_T T + w_L\, d/d_\text{max} + w_p\, p/p_\text{max}`

(:math:`T` is the throughput.) Only the ranking of the scores matters, not their value.

Screening and refinement
------------------------

:meth:`~thunder4d.design.designer.MPCDesigner.screen` evaluates every starting point with the
analytic tier, keeps the ``keep`` best (default ten times ``n_candidates``) while spreading them over
the gases and then over (gas, RoC, branch), and returns them without metrics.
:meth:`~thunder4d.design.designer.MPCDesigner.refine` then takes candidates in that order:

- the quick solver evaluates them,
- for each, the neighbours are tried: the same geometry with :math:`N \pm` ``neighbour_n_passes``
  passes (never beyond :math:`2M`) and, with ``neighbour_roc_steps``, the neighbouring radii of the
  allowed list (same :math:`M`, :math:`k`),
- the continuous parameters (``tune``: the pressure, and optionally the pre-chirp) are optimised for
  the score (Nelder-Mead), and the pressure is rounded to the resolution of your regulator,
  taking the better of the two neighbouring steps,
- a candidate that never becomes feasible is reported with the limits it violates.

Candidates that fall inside the neighbourhood of one already refined (the same geometry with a close
number of passes) are skipped, and two candidates that converge to the same design are listed once.
It stops when ``n_candidates`` feasible designs are found, or after ``max_attempts`` candidates
(default three times ``n_candidates``). Because the first ranking uses the *untuned* starting
pressure, a geometry that would win after tuning can lose to one that happened to start closer: ask
for 5 designs rather than 1.

.. list-table::
   :header-rows: 1
   :widths: 30 16 54

   * - ``problem.refinement``
     - Default
     - Meaning
   * - ``n_candidates``
     - 5
     - number of designs returned
   * - ``diversity``
     - True
     - spread the designs over gases first, then over (gas, RoC, branch)
   * - ``tune``
     - ("pressure_bar",)
     - continuous parameters optimised: ``"pressure_bar"``, ``"prechirp_gdd_fs2"``
   * - ``neighbour_n_passes``
     - 2
     - also try :math:`N \pm` this many passes
   * - ``neighbour_roc_steps``
     - 0
     - also try this many neighbouring radii on each side
   * - ``method``
     - ``"nelder-mead"``
     - ``"nelder-mead"`` or ``"powell"``
   * - ``max_evaluations``, ``xtol_rel``
     - 60, 1e-3
     - optimiser budget and tolerance
   * - ``keep_traces``
     - True
     - keep every evaluation in ``result.traces``

Parallel runs
~~~~~~~~~~~~~

``MPCDesigner(problem, quick=..., workers=N)`` runs the quick solver in ``N`` processes: the
evaluation of the starting points and the refinement of the candidates are distributed. The quick
solver must be picklable (a module-level function or a class instance such as
:class:`~thunder4d.design.fast1d_quick.Fast1DQuick`, not a lambda), and a script must put its work
under ``if __name__ == "__main__":`` on Windows. The gas and mirror catalogues, custom entries
included, are copied to the workers. In a notebook neither precaution is needed.

Reading a report
----------------

``result.best.report(problem)`` (or any candidate's) prints everything about a design:

.. code-block:: text

   Ar 0.170 bar | R = 2.0 m, M = 15 round trips, k = 2, concentric, d = 3.827 m, 29 passes | dielectric_1030nm, 3"
     output : 25.4 fs (from 500 fs, x19.7), throughput 98.6 %, peak power 435 GW
     score  : 0.00924  = objective 0.000229 + penalties 0.00901   (lower is better)
     limit                    value           limit   margin  mode
     main_peak               0.6335  >=         0.7      -9%  soft  <-- violated
     fluence [J/cm2]         0.5582  <=         0.7      20%  hard
     ...
     spots  : 15 per mirror on r = 33.7 mm, 7.9 w apart
     align  : d +- 1 mm -> w0 0.1 %, w_mirror 0.2 % | RoC +- 0.5 % -> w0 3.0 %, w_mirror 2.9 %

- **output**: the compressed pulse (pure GDD removal), the throughput and the peak power.
- **score**: split into the objective and the penalties of the soft limits.
- **limits**: value, limit actually applied, margin and mode, sorted from the tightest to the
  loosest: the first line is what constrains the design. The limits are:

  .. list-table::
     :header-rows: 1
     :widths: 24 76

     * - Name
       - Checks
     * - ``separation``, ``chamber_length``
       - mirror separation; separation + margin against the chamber
     * - ``p_over_pcr``, ``b_per_pass``, ``b_total``
       - Kerr limits
     * - ``ionization``
       - peak intensity at the waist against the gas threshold
     * - ``fluence``
       - worst bounce, against the LIDT at the pulse duration of that bounce
     * - ``bandwidth``
       - output energy outside the coating band
     * - ``spot_separation``, ``edge_clearance``
       - spot pattern on the mirror
     * - ``avg_power_mirror``
       - average power on one mirror
     * - ``throughput``, ``main_peak``
       - quality floors of the result
     * - ``pressure``
       - against the ceiling of the chamber (a margin of 0 % means the ceiling is the limit)
     * - ``coating``, ``diameter``
       - the design uses a coating or a mirror size you did not allow
     * - ``simulation``
       - the quick solver stopped: beam collapse or optical shock (see :doc:`solvers`)

- **spots** and **align**: the spot pattern, and the sensitivity to the alignment errors of
  ``problem.tolerance``.

Results
-------

:class:`~thunder4d.design.designer.DesignResult` holds the designs (``refined``, ranked by
``result.table()``), ``result.best``, the evaluated starting points (``screened``) and the tuning
traces. ``result.save_json(path)`` stores the problem and all the designs; ``result.export(path)``
writes one design as a standard THUNDER4D input file (:doc:`validation_4d`); ``candidate.to_cell()``
gives the :class:`~thunder4d.cell.MPC` of a design directly.

When nothing is feasible
------------------------

``refine()`` prints, for each candidate it gives up on, the limits it violates, and ends with
"only n feasible configuration(s)". Then, in this order:

1. Read the first lines of the reports of the best rejected designs: the limit that binds is the one to
   revisit. A ``pressure`` margin of 0 % means the pressure ceiling is the limit.
2. **Calibrate the limits on a cell you know works**: evaluate it (see the tutorial notebook) and set
   B per pass, fluence and ionisation so that it passes. The helper then searches around what your
   hardware really takes. The defaults are conservative starting points, not recommendations.
3. Widen the search (more round trips, more radii, both branches) or raise ``max_attempts``.
4. Draw a :doc:`map <maps>`: it shows at a glance which limit excludes each region.
