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
     - **gas**, **pressure_bar**, temperature_K, n2_1bar (None: thunder4d.materials)
   * - ``cell``
     - **roc_m**, **separation_m**, **n_passes**, herriott (information: round_trips, k, branch)
   * - ``mirrors``
     - reflectivity and gdd_fs2 (scalar or table, per bounce), tod_fs3, diameter_m, coating
       (information)
   * - ``grid``
     - preset (``"coarse"`` | ``"medium"`` | ``"fine"``), box_factor, points_per_waist,
       time_window_factor (override the preset), **lambda_min_nm**, **lambda_max_nm**,
       precision (``"single"`` | ``"double"``), device (``"auto"`` | ``"cpu"`` | ``"gpu"``)
   * - ``solver``
     - phi_max, gouy_step, self_steepening
   * - ``run``
     - record_every, near_field, record_mirror
   * - ``design``
     - free form: provenance written by the design helper (predictions, margins)

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
