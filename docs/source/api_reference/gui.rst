thunder4d.gui
=============

.. py:module:: thunder4d.gui

Graphical interface (optional: ``uv add "thunder4d[gui]"``), see :doc:`../gui/gui`. Importing
``thunder4d.gui`` and ``thunder4d.gui.registry`` does not import Qt.

.. py:function:: main(argv=None)

   Entry point of the ``thunder4d-gui`` command (``thunder4d-gui [input.json]``). Returns 1, with the
   install command, when PySide6 is missing.

Registries
----------

.. py:module:: thunder4d.gui.registry

.. py:data:: PANELS
             PLOTS
             METRICS

   Ordered collections of specs, indexed by key: ``register(spec, replace=False)``, ``unregister(key)``,
   iteration by ``order`` then registration order.

.. py:class:: PanelSpec(key, title, factory, order=100.0, tooltip="")

   A tab of the main window; ``factory(session)`` returns its widget. The session holds the current input
   (``config``, signal ``config_changed``) and results (``results``, ``results_changed``);
   ``session.show_panel(key)`` brings another tab to the front. A panel with a ``can_close()`` method is
   asked before the window closes.

.. py:class:: PlotSpec(key, title, draw, requires=(), order=100.0, group="", per_pass=False, data=None)

   A figure of the results viewer. ``draw(results, fig, options)`` fills a matplotlib ``Figure``;
   ``options["index"]`` is the selected record when ``per_pass``. Offered only when every key of
   ``requires`` is in the results. ``data(results, options)`` returns the arrays written by *Export data*
   (default: the ``requires`` arrays).

.. py:class:: MetricSpec(key, label, unit="", scale=1.0, order=100.0, fmt="{:.4g}", warn_above=None, plot=True)

   A quantity followed live: the last value of ``sim.history[key]`` times ``scale``, flagged above
   ``warn_above``, drawn versus pass when ``plot``.

.. py:function:: available_plots(results, plots=PLOTS)

   The registered figures that the results can draw.

Results and background tasks
----------------------------

.. py:class:: thunder4d.gui.results.Results(data, path=None)

   The arrays of a saved run, by key (a ``dict``), with ``path``, ``grid()`` (the simulation grid,
   rebuilt on the CPU) and ``cache`` (derived quantities). ``load_results(path)`` reads a ``.npz``.

.. py:class:: thunder4d.gui.worker.Task(work)

   ``work(task)`` run on a background thread (a ``QThread``). Signals: ``output`` (each printed line),
   ``progress`` (``task.report(value)``), ``succeeded(result)``, ``failed(traceback)``, ``cancelled``.
   ``task.cancel()`` makes the next ``task.check()`` raise ``Cancelled``.
