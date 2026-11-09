Quick solvers and their validity
================================

The helper judges a cell from a handful of numbers (duration, throughput, :math:`P/P_c`, B per pass,
intensity, fluence per bounce...). A **quick solver** produces them: any callable
``quick(problem, candidate) -> QuickResult``. Two are provided; the search, the maps and the
reports work the same with both.

Analytic estimates
------------------

:class:`~thunder4d.design.scan.AnalyticQuick` uses closed forms only (microseconds per cell): the
Kerr-matched mode of :meth:`MPC.nonlinear_mode <thunder4d.cell.MPC.nonlinear_mode>`, :math:`P/P_c`, the
B-integral per pass and in total, the peak intensity at the waist, the fluence on the mirrors, the
throughput :math:`R^N` and an SPM estimate of the duration,

.. math::

   \tau = \frac{\tau_\text{in}}{\sqrt{1 + \frac{4}{3\sqrt 3}\left(B/2\right)^2}} .

It has no dispersion and no notion of compressibility, so its durations are **optimistic**: use it
to discard geometries (``screen()``), to rank starting points and to draw maps, and decide on fast1d.

fast1d
------

:class:`~thunder4d.design.fast1d_quick.Fast1DQuick` wraps the 1+1D solver of
:mod:`thunder4d.design.fast1d`, which can also be used on its own
(:class:`~thunder4d.design.fast1d.MPCInput`, :func:`~thunder4d.design.fast1d.simulate`).

**Model.** A scalar envelope :math:`A(z, t)` propagated with the full dispersion of the gas
(Sellmeier) and an instantaneous Kerr effect with self-steepening, by an interaction-picture RK4
scheme whose step follows the nonlinear phase. The transverse dimension is reduced to a Gaussian
beam followed through its complex :math:`q` parameter: free propagation, a thin Kerr lens at every
step (aberration-free Gaussian ansatz, which balances diffraction at :math:`P = P_c`) and the
curved mirror at each bounce. The self-phase modulation uses the intensity-weighted mean phase of
the Gaussian profile; the reported B-integral is the on-axis one. The beam is followed with
sub-steps (at least 8 per Rayleigh length) so that the focus of near-concentric cells is resolved.
The compressor is an ideal GDD removal that maximises the peak power; the GDD is reported.

**Mapping from the design problem.**

- gas data from :mod:`thunder4d.materials` (the same as the 4D solver), with the polarisation factor,
- the cell from the candidate: :math:`R`, :math:`d`, :math:`N` passes, same pass convention as the
  4D code,
- the mirrors: :math:`R(\lambda)` and :math:`\mathrm{GDD}(\lambda)` of the coating, per bounce,
- the input: the energy times ``losses.mode_matching_efficiency``, the pre-chirp, and the chirp
  implied by ``duration_fs`` > ``tl_duration_fs``,
- ``matching="linear"`` (default) injects at the linear eigenmode of the cell;
  ``matching="nonlinear"`` at the Kerr-matched waist,
- ``options={...}`` passes any other :class:`~thunder4d.design.fast1d.MPCInput` field
  (``{"self_steepening": False}``).

**Per-bounce results.** The fluence and the pulse duration are returned for every bounce, so the
fluence limit takes the LIDT at the duration of that bounce.

Accuracy presets
~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 14 14 20 52

   * - Preset
     - Time per run
     - Grid
     - Use
   * - ``"fast"``
     - 0.1-0.2 s
     - compact, single precision, enlarged automatically if the pulse reaches its edges
     - the design loop (default of ``Fast1DQuick``)
   * - ``"default"``
     - 0.5-0.7 s
     - 4096 points, double precision
     - reference runs
   * - ``"fine"``
     - 2-8 s
     - 8192 points, finer steps, double precision
     - convergence reference

Measured against ``"fine"`` on a standard cell and on a near-concentric one (He, 2.7 bar, 40 passes),
the ``"fast"`` preset is within 0.4 % on the compressed duration, 0.1 % on the compressed peak
power, the B-integral and the mirror fluence, and 0.4 % on the peak intensity. The times are for
one core of an ordinary computer.

Guards
~~~~~~

When the model breaks down, the run stops instead of crawling: beam collapse (:math:`P` above the
critical power somewhere), a peak power 20 times the input one (optical shock), or an average of
more than 2000 steps per pass. The cell is then rejected through the ``p_over_pcr`` limit or a hard ``simulation``
limit, and the reason is in the warnings of its report. Calling ``fast1d.simulate`` directly raises
:class:`~thunder4d.design.fast1d.SimulationStopped`.

Checking the convergence of a result
------------------------------------

:meth:`Fast1DQuick.validate <thunder4d.design.fast1d_quick.Fast1DQuick.validate>` (or
:func:`fast1d.validate <thunder4d.design.fast1d.validate>` on an input) re-runs a design three
times and reports how much each output moved, with a verdict at 2 % by default:

- ``steps``: half the nonlinear phase per step, twice the minimum steps and beam sub-steps,
- ``grid``: a spectral window and a time window 1.5 times larger,
- ``precision``: double precision (only if the run is in single precision).

It also checks that the energy (or the photon number, with self-steepening) is conserved and that
the spectrum does not reach the edges of the grid. It costs three or four runs: use it on the
designs you keep, not inside the search. ``accuracy="default"`` validates a reference run instead
of the ``"fast"`` settings.

.. code-block:: python

   quick = Fast1DQuick()
   print(quick.validate(problem, result.best).summary())
   out = quick.simulate(problem, result.best, accuracy="default")   # full output: fields, spectra
   out.plot()

Where fast1d stops being reliable
---------------------------------

Converging fast1d only shows that it solves *its own model* correctly. How well the model describes
the cell is a different question, answered by the 4D solver (:doc:`validation_4d`). Expect fast1d
to be least reliable when

- :math:`P/P_c` is not small: the Gaussian ansatz is the weak point. On a cell at :math:`P/P_c = 0.34` and
  B = 63 rad, the 4D solver finds a B-integral 7.5 % higher and a compressed peak power 7 % lower,
  while the durations agree within 4 %. On a cell at B = 14 rad the agreement is within 3 % on every
  quantity,
- the spectrum is very broad (few-cycle targets): a GDD-only compressor leaves a gap between the
  compressed duration and the Fourier limit, and the energy in the main peak falls,
- the beam or the input is not Gaussian, the mirrors clip it, or the cell is mismatched: none of
  this is in the model.

Treat the duration as a prediction good to a few percent on moderate cells and to about 10 % on
strongly nonlinear ones, and the margins of the constraints as guides. Always simulate the design
you keep.

Using your own solver
---------------------

Any callable ``quick(problem, candidate) -> QuickResult`` can drive the search. Only the first three
fields are required; every other field you fill is checked against the matching limit
automatically, and the ones you leave out are not checked.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - :class:`~thunder4d.design.designer.QuickResult` field
     - Meaning
   * - ``duration_fs``, ``peak_power_W``, ``throughput``
     - compressed pulse and energy out / energy in (required)
   * - ``main_peak_fraction``, ``compressor_gdd_fs2``
     - energy in the main peak; the GDD applied
   * - ``b_total_rad``, ``b_per_pass_rad``, ``p_over_pcr``
     - Kerr limits (maximum over the passes for the last two)
   * - ``peak_intensity_W_cm2``
     - maximum at the waist
   * - ``mirror_fluence_J_cm2``, ``duration_per_bounce_fs``
     - arrays with one value per bounce
   * - ``out_of_band_fraction``
     - output energy outside the coating band
   * - ``spectrum_nm``
     - ``(wavelength, intensity)``, kept for the export (window of the 4D grid)
   * - ``extras``
     - anything else; ``solver_warnings`` are shown in the report, ``w_mirror_max_m`` sets the spot
       size of the spot-pattern checks, ``average_power_mirror_W`` feeds its limit

``examples/design_fast1d.py`` runs the whole loop with fast1d, and :doc:`../api_reference/design`
lists the interfaces.
