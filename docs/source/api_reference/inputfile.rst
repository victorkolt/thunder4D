thunder4d.inputfile
===================

.. py:module:: thunder4d.inputfile

Standard THUNDER4D input file (JSON): one file describes one simulation completely (laser, gas,
cell, mirrors, grid, solver, run), so that a design exported by the design helper, a GUI session
or a script can be re-run from the file alone.

.. code-block:: python

   import thunder4d as th

   cfg = th.load_input("cell.json")        # defaults filled in, checked
   b = th.run_input(cfg)                   # build + run; b.sim holds the results
   th.save_input(cfg, "copy.json")

Units are in the key names (``_J``, ``_fs``, ``_nm``, ``_m``, ``_bar``, ``_K``). Tables (measured
spectrum, mirror reflectivity and GDD curves) are stored inline as
``{"wavelength_nm": [...], "value": [...]}``. Unknown keys are rejected (a typo would otherwise be
ignored silently), except in the free-form ``design`` and ``notes`` blocks. Every missing required
field is listed in one error message.

.. list-table::
   :header-rows: 1
   :widths: 15 85

   * - Block
     - Keys (required in bold)
   * - ``laser``
     - **energy_J**, **duration_fs** (transform-limited FWHM), wavelength_nm, shape
       (``"gaussian"`` | ``"measured"``), spectrum, gdd_fs2, tod_fs3
   * - ``beam``
     - injection (``"linear"`` | ``"nonlinear"`` Kerr-matched | ``"waist"``), w0_m
   * - ``medium``
     - **gas** (a gas, or ``"vacuum"`` for a bulk cell), **pressure_bar** (not needed in vacuum), temperature_K,
       n2_1bar (None: thunder4d.materials)
   * - ``cell``
     - **roc_m**, **separation_m**, **n_passes**, herriott (information: round_trips, k, branch),
       plates, regions, wavelength_nm (bulk and hybrid cells, see below)
   * - ``mirrors``
     - reflectivity and gdd_fs2 (scalar or table, per bounce), tod_fs3, diameter_m, coating
       (information)
   * - ``grid``
     - preset (``"coarse"`` | ``"medium"`` | ``"fine"``), box_factor, points_per_waist,
       time_window_factor (override the preset), **lambda_min_nm**, **lambda_max_nm**,
       precision (``"single"`` | ``"double"``), device (``"auto"`` | ``"cpu"`` | ``"gpu"``),
       fft_backend (``"auto"`` | ``"scipy"`` | ``"fftw"``)
   * - ``solver``
     - phi_max, gouy_step, self_steepening, kernels (``"auto"`` | ``"numba"`` | ``"numpy"``)
   * - ``run``
     - record_every, near_field, record_mirror
   * - ``design``
     - free form: provenance written by the design helper (predictions, margins)

Bulk and hybrid cells
---------------------

``cell.plates`` is a list of plates and ``cell.regions`` a list of gas regions, with the positions of
:doc:`../user_guide/bulk_hybrid` (from the cell centre, positive towards the mirror hit at the end of pass 1).
``cell.wavelength_nm`` is the reference wavelength of such a cell (None: the laser wavelength). The kind of cell
is derived: no plates is a gas cell, plates with nothing but vacuum around them a bulk cell, plates and a gas a
hybrid cell.

.. list-table::
   :header-rows: 1
   :widths: 15 85

   * - Item
     - Keys
   * - plate
     - **thickness_m**, **position_m**, material (``"FS"``, ``"CaF2"``, ``"BaF2"``, ``"MgF2"``, ``"sapphire"``,
       ``"YAG"``; default ``"FS"``), coating_a and coating_b (``"bare"`` | ``"ar"``; default ``"ar"``), n2
       (None: the table value)
   * - region
     - **z0_m**, **z1_m**, gas (default ``"vacuum"``), pressure_bar (required for a gas), temperature_K

.. code-block:: json

   {"cell": {"roc_m": 0.2, "separation_m": 0.382, "n_passes": 14,
             "plates": [{"material": "FS", "thickness_m": 0.001, "position_m": 0.14},
                        {"material": "FS", "thickness_m": 0.001, "position_m": -0.14}],
             "regions": [{"z0_m": -0.14, "z1_m": 0.14, "gas": "vacuum"}]}}

With plates, the injected beam of ``beam.injection`` ``"linear"`` or ``"nonlinear"`` is built by
:meth:`~thunder4d.pulse.Pulse.cell_matched`, and the stability check applies to the equivalent separation.

.. py:function:: default_input(**blocks)

   A complete input with the defaults; keyword blocks are merged in, e.g.
   ``default_input(laser={"energy_J": 1e-3, "duration_fs": 200}, medium={"gas": "Ar", ...})``.

.. py:function:: load_input(path_or_dict)

   Read a file (or take a dict), fill in the defaults and check it (``ValueError`` listing every
   problem, ``KeyError`` for an unknown key).

.. py:function:: save_input(cfg, path)

   Write an input, defaults filled in, as indented JSON.

.. py:function:: build_simulation(cfg, verbose=False)

   Input -> object with ``cell``, ``grid``, ``pulse`` and ``sim`` (a
   :class:`~thunder4d.lensframe.LensFrameSimulation` ready to run), plus ``w0_injected_m``.

.. py:function:: run_input(cfg, verbose=True, max_seconds=None)

   :func:`build_simulation` then ``sim.run()``; returns the built object.

.. py:function:: input_problems(cfg)

   Every problem of a merged input, as a list of ``(path, message)``, where ``path`` is the key at
   fault (``"cell.separation_m"``); empty when the input is valid. :func:`load_input` raises with
   these messages.

.. py:function:: describe_input(cfg)

   Quick checks of an input before launching it, in milliseconds (neither the pulse nor the solver
   is built): linear eigenmode (``w0_m``, ``w_mirror_m``, ``zR_m``, ``gouy_per_pass_rad``), grid size
   (``grid_shape``, ``field_MB``), Gaussian estimates of the peak power, of ``p_over_pcr`` (gas) and
   ``plates_p_over_pcr`` (one value per plate), and of the peak fluence on the mirrors
   (``mirror_fluence_J_cm2``). ``kind`` is ``"gas"``, ``"bulk"`` or ``"hybrid"``.

.. py:data:: FIELDS

   How each key is shown and edited (label, displayed unit, kind of editor, help, conditions), as
   ``{"block.key": Field}``. The GUI builder is generated from it, and a test checks that every key
   of the input file has an entry: a new key gets its :class:`Field` in the same change.
