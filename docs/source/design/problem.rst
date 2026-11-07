Describing the problem
======================

Everything the helper needs is in one object, :class:`~thunder4d.design.specs.DesignProblem`, made
of seven blocks. All of them are dataclasses with documented defaults: set only what matters to you.

.. code-block:: python

   problem = DesignProblem(laser=..., target=...)       # the two required blocks
   problem.constraints.kerr.p_over_pcr.value = 0.3      # any other knob, by attribute
   problem.search.round_trips = (5, 30)

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Block
     - Role
   * - ``laser``
     - the pulse injected in the cell
   * - ``target``
     - what "best" means, and the quality floors of the result
   * - ``constraints``
     - what your hardware takes: space, mirrors, Kerr effect, ionisation, spot pattern, pressure, gas
   * - ``losses``
     - losses on the way in
   * - ``search``
     - which geometries are tried (:doc:`search`)
   * - ``refinement``
     - how the best candidates are tuned (:doc:`search`)
   * - ``tolerance``
     - alignment tolerances reported for a design (information only)

The laser
---------

.. list-table::
   :header-rows: 1
   :widths: 28 14 58

   * - Field
     - Default
     - Meaning
   * - ``energy_J``
     - 1e-3
     - pulse energy [J]
   * - ``duration_fs``
     - 300
     - intensity FWHM of the injected pulse [fs]
   * - ``wavelength_nm``
     - 1030
     - central wavelength [nm]
   * - ``tl_duration_fs``
     - None
     - transform-limited duration, if the input is chirped (``duration_fs`` > ``tl_duration_fs``):
       the chirp is added as a positive GDD
   * - ``shape``
     - ``"gaussian"``
     - ``"gaussian"`` or ``"sech2"`` (peak power factor)
   * - ``prechirp_gdd_fs2``, ``prechirp_tod_fs3``
     - 0
     - spectral phase added before injection
   * - ``rep_rate_Hz``
     - 1000
     - repetition rate, for the average power on the mirrors
   * - ``polarization``
     - ``"linear"``
     - ``"linear"`` or ``"circular"`` (:math:`n_2` times 2/3)

``LaserSpec.preset(name, **overrides)`` returns a laser from the datasheet presets (PHAROS-PH2,
CARBIDE and Dira families, plus your own with ``register_laser_preset``). The values are the
manufacturers' upper bounds for the duration, and the repetition rate at which the maximum energy
is specified. Any field can be overridden:

.. code-block:: python

   from thunder4d.design import LaserSpec
   from thunder4d.design.specs import LASER_PRESETS

   print(sorted(LASER_PRESETS))
   laser = LaserSpec.preset("pharos_ph2_sp_2mJ", duration_fs=170.0, rep_rate_Hz=1e3)

The target
----------

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - ``objective``
     - the score is low when...
   * - ``"target_duration"``
     - the compressed duration is close to ``target_duration_fs`` (not shorter)
   * - ``"target_compression"``
     - input duration / output duration is close to ``target_compression``
   * - ``"shortest"``
     - the compressed pulse is short
   * - ``"max_peak_power"`` (default)
     - the compressed peak power is high (duration *and* throughput)
   * - ``"weighted"``
     - a weighted sum of the duration, the throughput, the cell length and the pressure
       (``weights``: ``duration``, ``throughput``, ``length``, ``pressure``)

For the two "target" objectives, candidates that all reach the target are told apart by their
throughput. Two quality floors apply to the result, both *soft* by default (they penalise the score
instead of rejecting the design):

.. list-table::
   :header-rows: 1
   :widths: 34 12 54

   * - Field
     - Default
     - Meaning
   * - ``min_throughput``
     - 0.80
     - energy out / energy in (cell and compressor losses)
   * - ``min_main_peak_fraction``
     - 0.70
     - energy in the main peak of the compressed pulse; with a GDD-only compressor it falls for
       very broad spectra, so lower it or switch it off (``Off()``) for few-cycle targets

Limits
------

Every constraint is a :class:`~thunder4d.design.specs.Limit`:

.. list-table::
   :header-rows: 1
   :widths: 14 86

   * - Field
     - Meaning
   * - ``value``
     - the limit. ``None`` means *take it from the catalogue*: the LIDT of the coating (at the local
       pulse duration) or the ionisation threshold of the gas
   * - ``bound``
     - ``"max"`` (the quantity must stay below the limit) or ``"min"`` (above)
   * - ``mode``
     - ``"hard"``: a violation rejects the design. ``"soft"``: it adds ``weight * margin**2`` to the
       score. ``"off"``: ignored
   * - ``safety``
     - the limit actually applied is ``value * safety`` for a maximum and ``value / safety`` for a
       minimum
   * - ``weight``
     - weight of the penalty of a soft limit

``Max(value, mode=..., safety=..., weight=...)``, ``Min(...)`` and ``Off()`` build limits. The
**margin** shown in the reports is the distance to the applied limit, relative to it: 20 % means
20 % of headroom, a negative margin is a violation.

.. warning::

   Setting only ``.value`` keeps the existing ``safety``. The ionisation limit has ``safety = 0.5``
   by default, so ``peak_intensity_W_cm2.value = 3.5e13`` applies 1.75e13 W/cm². Replace the whole
   limit (``Max(3.5e13)``) to reset it. The reports always print the limit actually applied.

Constraints
-----------

Space
~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Field
     - Default
     - Meaning
   * - ``max_separation_m``
     - hard, 4 m
     - mirror separation :math:`d`
   * - ``chamber_margin_m``
     - 0.2
     - chamber length = :math:`d` + this margin (mounts, windows)
   * - ``max_chamber_length_m``
     - off
     - limit on the chamber length
   * - ``fixed_chamber_length_m``
     - None
     - an existing chamber: its length is the default value of ``max_chamber_length_m``

Mirrors
~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Field
     - Default
     - Meaning
   * - ``allowed_roc_m``
     - 0.5 ... 7 m
     - the radii of curvature you can use (tuple)
   * - ``allowed_diameters_inch``
     - (3.0,)
     - mirror diameters
   * - ``allowed_coatings``
     - None
     - names of the cavity coatings to use (``None``: all of the catalogue)
   * - ``fluence``
     - hard, None
     - peak fluence on the mirrors, per bounce. ``None``: the LIDT of the coating at the pulse
       duration of that bounce, ``lidt * (tau / lidt_ref_fs) ** lidt_exponent``
   * - ``bandwidth``
     - hard, 0.01
     - fraction of the output energy outside the band of the coating
   * - ``max_average_power_W``
     - off
     - average power on one mirror (pulse energy × repetition rate × :math:`N/2` bounces)

Kerr effect
~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Field
     - Default
     - Meaning
   * - ``p_over_pcr``
     - hard, 0.5
     - peak power over the critical power, maximum over the passes
   * - ``b_per_pass_rad``
     - hard, 1 rad
     - on-axis B-integral of one pass (maximum over the passes)
   * - ``b_total_rad``
     - off
     - total on-axis B-integral

Ionisation
~~~~~~~~~~

``peak_intensity_W_cm2``: hard, ``value = None`` (the threshold of the gas), ``safety = 0.5``.
It limits the peak intensity at the waist, which is a proxy for ionisation: the helper does not
simulate it.

Spot pattern
~~~~~~~~~~~~

Each mirror carries :math:`M` spots on a circle. By default the circle is as large as the mirror
allows, so the pattern depends on the mirror size: its radius is the mirror radius minus
``edge_clearance_w`` times the spot size on the mirror. Adjacent spots are then
:math:`2r\sin(\pi/M)` apart.

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Field
     - Default
     - Meaning
   * - ``min_spot_separation_w``
     - hard, 3
     - minimum distance between adjacent spots, in spot radii on the mirror
   * - ``edge_clearance_w``
     - hard, 2.5
     - minimum distance between a spot centre and the mirror edge, in spot radii
   * - ``pattern_radius_m``
     - None
     - impose the radius of the circle (the edge clearance is then checked instead of used)

The spot size on the mirror is the most favourable one the Kerr effect allows in the analytic
prefilter (so that no cell that could pass is rejected), and the largest one of the run in the
evaluation.

Pressure and gas
~~~~~~~~~~~~~~~~

``pressure.pressure_bar`` = (0, 3): the range the chamber and the windows take.
``pressure.resolution_bar`` = 0.01: the resolution of your regulator, to which the refined
pressures are rounded. ``gas.allowed``: ``("He", "Ne", "Ar", "Kr", "Xe", "N2")``.

Losses and tolerances
---------------------

``losses.mode_matching_efficiency`` (default 1): fraction of the energy injected in the cell mode,
applied to the input energy.

``tolerance`` does not constrain anything: it sets the alignment errors for which each report gives
the change of waist and mirror spot (``separation_error_mm`` = 1 mm, ``roc_tolerance_rel`` = 0.5 %).
Near-concentric cells are the sensitive ones.

Catalogues
----------

Gases
~~~~~

The gas catalogue (``thunder4d.design.catalog.GASES``) takes the dispersion and :math:`n_2` from
:mod:`thunder4d.materials`, the same data as the 4D solver, and adds the ionisation potential and
an **ionisation threshold** intensity. The default threshold is only 0.2 times the
barrier-suppression intensity, which is an order of magnitude: **set your own values**.

.. code-block:: python

   from thunder4d.design.catalog import GASES

   GASES["Ar"] = GASES["Ar"].with_(ionization_threshold_W_cm2=4e13)

``register_gas(Gas(...))`` adds a gas. N\ :sub:`2` and air are flagged because the Raman response is
not modelled.

Mirrors
~~~~~~~

A :class:`~thunder4d.design.catalog.mirrors.Coating` has a reflectivity :math:`R(\lambda)` and a
group-delay dispersion :math:`\mathrm{GDD}(\lambda)` per bounce (scalars or tables), a usable band,
a damage threshold with its reference duration and an exponent, and information on the vendor. The
catalogue contains three cavity coatings (``dielectric_1030nm``, ``dielectric_800nm``,
``enhanced_silver``) with the vendor and notes recorded in ``catalog/mirrors.py``: check their
values against your own mirrors. Add your own, or load vendor curves:

.. code-block:: python

   from thunder4d.design.catalog import Coating, register_mirror, load_catalog

   register_mirror(Coating("my_hr_1030", kind="dielectric", band_nm=(960.0, 1100.0),
                           reflectivity=0.9995, lidt_J_cm2=0.7, vendor="...", source="datasheet"))
   register_mirror(Coating.from_files("my_hr_curves", "R.csv", "GDD.csv",   # two columns: nm, value
                                      lidt_J_cm2=0.7, band_nm=(950.0, 1110.0)))
   load_catalog("coatings.json")        # several coatings, curve files next to the JSON

The damage threshold scales with the pulse duration as ``lidt * (tau / lidt_ref_fs) ** lidt_exponent``
(exponent 0 by default: no scaling). The helper evaluates it for the duration the pulse has *at each
bounce*, so the last passes, with the shortest pulses, are the critical ones.

.. note::

   Custom gases and coatings are copied to the worker processes of a parallel run, so they work with
   ``workers > 1``.
