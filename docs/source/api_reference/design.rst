thunder4d.design
================

.. py:module:: thunder4d.design

The design helper (see :doc:`../user_guide/design`). Everything below is importable from
``thunder4d.design``; the sections name the submodule that defines it. Every field of the problem
blocks, with its default, is defined in :mod:`thunder4d.design.specs`.

.. code-block:: python

   from thunder4d.design import (
       DesignProblem, LaserSpec, Target, Max, Min, Off,
       MPCDesigner, Candidate, starting_points, evaluate,
       Fast1DQuick, AnalyticQuick, design_map, validate_4d, to_input, export_input,
   )
   from thunder4d.design.catalog import GASES, MIRRORS, Coating, Gas, register_mirror, load_catalog

Problem description
-------------------

.. py:module:: thunder4d.design.specs

.. py:class:: DesignProblem(laser=LaserSpec(), target=Target(), constraints=Constraints(), losses=Losses(), search=SearchSpace(), refinement=Refinement(), tolerance=Tolerance())

   Everything the helper needs, in one object. ``to_dict()`` gives a plain dictionary (written in
   the JSON export).

.. py:class:: LaserSpec(energy_J=1e-3, duration_fs=300.0, wavelength_nm=1030.0, tl_duration_fs=None, shape="gaussian", prechirp_gdd_fs2=0.0, prechirp_tod_fs3=0.0, rep_rate_Hz=1e3, polarization="linear", name="laser")

   The injected pulse. Properties ``lambda0`` [m], ``peak_power_W``, ``average_power_W``,
   ``n2_polarization_factor`` (1 for linear, 2/3 for circular polarisation).

   .. py:classmethod:: preset(name, **overrides)

      A laser from :data:`LASER_PRESETS`, any field overridable.

.. py:data:: LASER_PRESETS

   Datasheet presets (PHAROS-PH2, CARBIDE, Dira, and ``mpc1_18mJ_350fs``), as a dictionary of
   ``LaserSpec`` fields.

.. py:function:: register_laser_preset(name, **fields)

   Add a preset.

.. py:class:: Target(objective="max_peak_power", target_duration_fs=None, target_compression=None, weights={...}, min_throughput=Min(0.8, mode="soft"), min_main_peak_fraction=Min(0.7, mode="soft"))

   What the design should achieve. ``objective`` is ``"shortest"``, ``"max_peak_power"``,
   ``"target_duration"``, ``"target_compression"`` or ``"weighted"``.

.. py:class:: Limit(value=None, bound="max", mode="hard", safety=1.0, weight=1.0)

   One constraint. ``allowed(default=None)`` is the limit actually applied (``value * safety`` for a
   maximum, ``value / safety`` for a minimum, ``default`` if ``value`` is ``None``);
   ``slack(x, default=None)`` the margin relative to it (negative: violated); ``penalty(x,
   default=None)`` is 0 if satisfied, infinity for a violated hard limit and ``weight * margin**2``
   for a violated soft one.

.. py:function:: Max(value=None, mode="hard", safety=1.0, weight=1.0)
.. py:function:: Min(value=None, mode="hard", safety=1.0, weight=1.0)
.. py:function:: Off()

   A maximum, a minimum, a disabled limit.

.. py:class:: Constraints(space, mirrors, kerr, ionization, spots, pressure, gas)

   What the hardware takes, in seven groups: :class:`SpaceConstraints`, :class:`MirrorConstraints`,
   :class:`KerrConstraints`, :class:`IonizationConstraints`, :class:`SpotConstraints`,
   :class:`PressureConstraints` and :class:`GasConstraints`.

.. py:class:: SpaceConstraints(max_separation_m=Max(4.0), chamber_margin_m=0.2, max_chamber_length_m=Off(), fixed_chamber_length_m=None)

   Mirror separation and chamber length.

.. py:class:: MirrorConstraints(allowed_roc_m=(0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0), allowed_diameters_inch=(3.0,), allowed_coatings=None, fluence=Max(None), bandwidth=Max(0.01), max_average_power_W=Off())

   The mirrors you can use, the fluence they take (``None``: the LIDT of the coating at the local
   pulse duration), the fraction of the output energy allowed outside the coating band and the
   average power on one mirror.

.. py:class:: KerrConstraints(p_over_pcr=Max(0.5), b_per_pass_rad=Max(1.0), b_total_rad=Off())

   Peak power over the critical power, on-axis B-integral per pass and in total.

.. py:class:: IonizationConstraints(peak_intensity_W_cm2=Max(None, safety=0.5))

   Peak intensity at the waist (``None``: the ionisation threshold of the gas).

.. py:class:: SpotConstraints(min_spot_separation_w=Min(3.0), edge_clearance_w=Min(2.5), pattern_radius_m=None)

   Circular pattern of :math:`M` spots on each mirror, in units of the spot radius on the mirror; the
   pattern radius follows the mirror size unless ``pattern_radius_m`` is given.

.. py:class:: PressureConstraints(pressure_bar=(0.0, 3.0), resolution_bar=0.01)

   The pressure range the chamber takes, and the resolution of the regulator.

.. py:class:: GasConstraints(allowed=("He", "Ne", "Ar", "Kr", "Xe", "N2"))

   The gases that may be used.

.. py:class:: Losses(mode_matching_efficiency=1.0)

   Fraction of the energy injected in the cell mode.

.. py:class:: Tolerance(separation_error_mm=1.0, roc_tolerance_rel=0.005)

   Alignment errors for which the reports give the sensitivity (information only).

.. py:class:: SearchSpace(gases=None, roc_m=None, diameters_inch=None, coatings=None, branches=("concentric", "planar"), round_trips=(3, 40), k_range=None, pass_offsets=(0, -1), min_passes=5)

   Which geometries :func:`~thunder4d.design.designer.starting_points` enumerates.

.. py:class:: Refinement(n_candidates=5, diversity=True, tune=("pressure_bar",), neighbour_n_passes=2, neighbour_roc_steps=0, method="nelder-mead", max_evaluations=60, xtol_rel=1e-3, keep_traces=True)

   How the best candidates are tuned.

Search
------

.. py:module:: thunder4d.design.designer

.. py:class:: Candidate(gas, pressure_bar, roc_m, round_trips, k, branch="concentric", n_passes=None, diameter_inch=3.0, coating="dielectric_1030nm", prechirp_gdd_fs2=0.0)

   One configuration: radius of curvature, number of round trips :math:`M` of the re-entrant cycle
   and index :math:`k` give :math:`d = R(1 \pm \cos(\pi k/M))`; ``n_passes`` is the number of passes
   actually used (default :math:`2M`). After an evaluation it carries ``metrics``, ``slacks`` (margin
   of each limit), ``limits`` (value, operator, applied limit, mode), ``warnings``, ``score``,
   ``objective``, ``penalty`` and ``violation``.

   .. py:attribute:: separation_m

      Mirror separation :math:`d` [m].

   .. py:method:: report(problem=None)

      The text report of :doc:`../user_guide/design`.

   .. py:method:: tolerances(problem=None, lambda0_m=None)

      Relative change of :math:`w_0` and of the mirror spot for the alignment errors of
      ``problem.tolerance``: ``{"separation": (dw0, dwm), "roc": (dw0, dwm)}``.

   .. py:method:: linear_mode(lambda0_m=1030e-9)

      ``(w0, w_mirror, zR)`` of the empty cell [m].

   .. py:method:: to_cell(polarization_factor=1.0, lambda0_m=1030e-9, **kw)

      The :class:`~thunder4d.cell.MPC` of the design, with the gas, the coating curves and the mirror
      diameter.

.. py:class:: QuickResult(duration_fs, peak_power_W, throughput, main_peak_fraction=None, compressor_gdd_fs2=None, b_total_rad=None, b_per_pass_rad=None, p_over_pcr=None, peak_intensity_W_cm2=None, mirror_fluence_J_cm2=None, duration_per_bounce_fs=None, out_of_band_fraction=None, spectrum_nm=None, extras={})

   What a quick solver returns (:doc:`../user_guide/design`).

.. py:class:: MPCDesigner(problem, quick=None, verbose=True, workers=None)

   :param problem: a :class:`~thunder4d.design.specs.DesignProblem`
   :param quick: a quick solver, ``quick(problem, candidate) -> QuickResult``
   :param int workers: processes for the quick solver (picklable solver needed)

   .. py:method:: screen(keep=None, **starting_kw)

      Evaluate the starting points with the analytic estimates and keep the ``keep`` best diverse
      ones (default ten times ``n_candidates``).

   .. py:method:: refine(candidates, max_attempts=None)

      Evaluate, tune and rank the candidates until ``n_candidates`` feasible designs are found, or
      ``max_attempts`` (default three times ``n_candidates``) have been tried. Returns a
      :class:`DesignResult`.

   .. py:method:: run(keep=None, **starting_kw)

      ``refine(screen(...))``.

.. py:class:: DesignResult(problem, screened, refined, traces)

   ``refined`` (the designs), ``screened`` (every evaluated starting point), ``traces`` (tuning
   evaluations), ``best`` (lowest score).

   .. py:method:: table(which="refined")

      Text table, ranked.

   .. py:method:: export(path, which=0, **kw)

      The design of rank ``which`` as a standard input file (see :func:`~thunder4d.design.export.to_input`).

   .. py:method:: save_json(path)

      Problem and designs as JSON.

.. py:function:: starting_points(problem, gases=None, round_trips=None, k="all", p_over_pcr=0.3, verbose=True)

   Geometries of the search space, prefiltered, each with an analytic starting pressure.

.. py:function:: start_pressure(problem, cand, p_over_pcr=0.3)

   The analytic first guess of the pressure of one geometry.

.. py:function:: prefilter(problem, cand)

   The list of hard limits that exclude a geometry analytically (empty: it can pass).

.. py:function:: herriott_k(round_trips, k_range=None)

   Valid indices :math:`k` for :math:`M` round trips: coprime with :math:`M` and :math:`k < M/2`.

.. py:function:: evaluate(problem, cand, metrics)

   Fill ``cand`` (margins, limits, warnings, score) from a dictionary of metrics: the same judgement
   for every solver.

Quick solvers
-------------

.. py:module:: thunder4d.design.fast1d_quick

.. py:class:: Fast1DQuick(accuracy="fast", matching="linear", keep_spectrum=True, options={})

   Quick solver backed by fast1d (:doc:`../user_guide/design`).

   :param str accuracy: ``"fast"``, ``"default"`` or ``"fine"``
   :param str matching: ``"linear"`` or ``"nonlinear"`` injection
   :param dict options: any other :class:`~thunder4d.design.fast1d.MPCInput` field

   .. py:method:: simulate(problem, cand, accuracy=None)

      The full :class:`~thunder4d.design.fast1d.MPCOutput` of a design (fields, spectra, ``plot()``).

   .. py:method:: validate(problem, cand, accuracy=None, rtol=0.02, checks=("steps", "grid", "precision"))

      Convergence check of fast1d for this design.

   .. py:method:: build_input(problem, cand, accuracy=None)

      The :class:`~thunder4d.design.fast1d.MPCInput` of a design.

.. py:module:: thunder4d.design.scan

.. py:class:: AnalyticQuick(matching="linear")

   Closed-form quick solver, microseconds per cell, optimistic durations.

.. py:function:: design_map(problem, *, gas, roc_m, k=1, round_trips=None, branch="concentric", x=None, y=None, quick=None, coating=None, diameter_inch=None, pass_offset=0, workers=None)

   Scan two parameters and find the limiting constraint at each point (:doc:`../user_guide/design`).

   :param x: ``(name, values)``, default pressure from the pressure range
   :param y: ``(name, values)``, default round trips from the search space
   :param quick: ``None`` for the analytic estimates, or a solver such as ``Fast1DQuick()``
   :param round_trips: fixed :math:`M`, required unless ``round_trips`` is a scanned axis
   :returns: a :class:`DesignMap`

.. py:class:: DesignMap

   ``x``, ``y``, ``x_name``, ``y_name``, ``limiting``, ``tightest``, ``margin``, ``values``,
   ``candidates``, ``feasible``.

   .. py:method:: best()

      The feasible candidate with the lowest score, or ``None``.

   .. py:method:: summary()

      A short text description.

   .. py:method:: plot(ax=None, contour="duration_fs", levels=None, mark=(), show_best=True)

      Draw the map: regions coloured by the limiting constraint, the feasible region outlined,
      contours of ``contour``, and the points of ``mark`` as ``(x, y, label)``.

fast1d
------

.. py:module:: thunder4d.design.fast1d

The 1+1D solver, usable on its own.

.. py:class:: MPCInput(energy, fwhm, wavelength=1030e-9, gas=..., pressure=..., roc=..., length=..., n_passes=20, accuracy=None, **numerics)

   Input of a run: SI units, one field per quantity (see the module docstring for the complete
   list). ``MPCInput.herriott(roc, N, k, branch="concentric", **kwargs)`` builds the geometry from
   :math:`R`, :math:`N` and :math:`k`; ``with_accuracy(accuracy, **overrides)`` copies the input
   with the numerical settings of a preset. ``mirror_response`` takes the complex amplitude transfer
   of a bounce on the absolute angular frequency.

.. py:function:: simulate(inp)

   Propagate and compress. Returns an :class:`MPCOutput` (``fwhm_compressed``,
   ``peak_power_compressed``, ``gdd_compression``, ``b_integral``, ``transmission``,
   ``intensity_max``, ``fluence_mirror_max``, per-pass arrays, fields and spectra, ``summary()``,
   ``plot()``). Raises :class:`SimulationStopped` if the model breaks down.

.. py:class:: MPCOutput

   Result of :func:`simulate`: the input, the time and frequency axes, the fields and spectra (input,
   output, compressed, Fourier limit), the energies and durations, ``gdd_compression``,
   ``b_integral``, ``peak_power_over_pcr``, ``intensity_max``, ``fluence_mirror_max`` and the
   per-pass arrays (``pass_energy``, ``pass_b_integral``, ``pass_w_mirror``, ``pass_fwhm``...).

.. py:function:: validate(inp, checks=("steps", "grid", "precision"), rtol=0.02)

   Re-run with halved steps, a larger grid and double precision; returns a ``ValidationReport``
   (``ok``, ``summary()``).

.. py:exception:: SimulationStopped

   The run was stopped by a guard (optical shock, step budget). ``p_over_pcr``, ``passes_done`` and
   ``b_integral`` tell where it stopped.

.. py:exception:: BeamCollapse

   Subclass of :exc:`SimulationStopped`, raised when :math:`P` exceeds the critical power.

.. py:data:: ACCURACY

   The settings of the ``"fast"``, ``"default"`` and ``"fine"`` presets.

From a design to the 4D solver
------------------------------

.. py:module:: thunder4d.design.export

.. py:function:: to_input(problem, cand, *, preset="coarse", injection="linear", name=None, window=None, precision="single", device="auto", run=None)

   The standard input file (:doc:`inputfile`) of a design, as a dictionary, with a ``design`` block
   recording its origin.

.. py:function:: export_input(problem, cand, path, **kw)

   ``to_input`` written to ``path`` as JSON.

.. py:class:: Validation4D(cfg, rows, extra, built, fast1d, runtime_s)

   Result of :func:`validate_4d`: the input used, the comparison ``rows`` (quantity, fast1d, 4D,
   relative difference), the 4D-only results in ``extra``, and the built simulation and the fast1d
   output. ``summary()`` returns the table as text and ``plot()`` the comparison figure.

.. py:function:: validate_4d(problem, cand, *, quick=None, preset="coarse", verbose=True, max_seconds=None, **to_input_kw)

   Run the design with the 4D lens-frame solver and compare it with fast1d
   (:doc:`../user_guide/design`). Returns a :class:`Validation4D`.

Catalogues
----------

.. py:module:: thunder4d.design.catalog.gases

.. py:class:: Gas(name, ip_eV, raman_active=False, n2_1bar=None, ionization_threshold_W_cm2=None, bsi_fraction=0.2, notes="")

   A gas of the design catalogue. ``ionization_threshold`` is the given threshold, or
   ``bsi_fraction`` times the barrier-suppression intensity. ``material(pressure_bar,
   temperature_K=293.15, polarization_factor=1.0)`` returns the :class:`~thunder4d.materials.Material`;
   ``with_(**changes)`` a modified copy.

.. py:data:: GASES

   The catalogue, a dictionary of :class:`Gas`.

.. py:function:: register_gas(gas, overwrite=False)
.. py:function:: get_gas(name)

   Add a gas; get one (with an informative error).

.. py:module:: thunder4d.design.catalog.mirrors

.. py:class:: Coating(name, kind="dielectric", role="cavity", band_nm=(0, inf), reflectivity=1.0, gdd_fs2=0.0, lidt_J_cm2=None, lidt_ref_fs=1000.0, lidt_exponent=0.0, pair=None, diameter_inch=(), r_in_percent=False, vendor="", source="", notes="")

   A mirror coating. ``reflectivity`` and ``gdd_fs2`` are scalars or ``(wavelength_nm, values)``
   tables. ``R(lam_nm)``, ``gdd(lam_nm)`` and ``in_band(lam_nm)`` evaluate them; ``lidt(duration_fs)``
   is the damage threshold at a pulse duration.

   .. py:classmethod:: from_files(name, reflectivity_csv=None, gdd_csv=None, *, delimiter=None, skip_header=0, wavelength_unit="nm", **kw)

      A coating from vendor curves (two columns: wavelength, value).

.. py:data:: MIRRORS

   The catalogue, a dictionary of :class:`Coating`.

.. py:function:: register_mirror(coating, overwrite=False)
.. py:function:: get_mirror(name)
.. py:function:: load_catalog(path, overwrite=True)

   Add a coating, get one, or register all the coatings of a JSON file (curve files are read
   relative to it).

.. py:function:: load_curve(path, delimiter=None, skip_header=0, wavelength_unit="nm")

   Read a two-column curve file; returns ``(wavelength_nm, values)``.
