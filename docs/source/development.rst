Development
===========

The project uses `uv <https://docs.astral.sh/uv/>`_.

.. code-block:: bash

   git clone https://github.com/victorkolt/thunder4D
   cd thunder4D
   uv sync --extra examples
   uv run pytest                 # about 40 s; GUI tests run offscreen
   uv run ruff check             # lint
   uv run ty check               # types

Build the documentation, with warnings as errors:

.. code-block:: bash

   uv run --group docs sphinx-build -b html -W --keep-going docs/source docs/_build/html

Releases and CI are described in ``CONTRIBUTING.md``.

Benchmarks
----------

``benchmark/`` has one script per published experiment, built on ``benchmark/framework.py``.
``--quick`` runs 2 passes on a coarse grid; the test suite runs it for every benchmark and fails if the
results move by more than 0.1 %.

.. code-block:: bash

   cd benchmark
   python bench_grebing2020_1mJ.py --quick
   python bench_grebing2020_1mJ.py               # full run, finest grid that fits in memory
   python analyze.py --latest                    # tables against the papers

``benchmark/README.md`` explains the options, the output and how to add a benchmark.

Extending the interface
-----------------------

The graphical interface is built from registries, so a new feature reaches it without editing the
windows:

.. list-table::
   :header-rows: 1
   :widths: 30 34 36

   * - To show
     - Add
     - In
   * - a new key of the input file
     - its :class:`~thunder4d.inputfile.Field` in ``FIELDS`` (a test fails without it)
     - ``thunder4d/inputfile.py``
   * - a new kind of field
     - a ``FieldEditor`` in ``EDITORS``
     - ``thunder4d/gui/fields.py``
   * - a knob of the design helper
     - a ``Field`` in ``DESIGN_FIELDS``
     - ``thunder4d/gui/panels/design.py``
   * - a quantity followed during a run
     - a ``MetricSpec`` in ``METRICS``
     - ``thunder4d/gui/metrics.py``
   * - a figure of the results
     - a ``PlotSpec`` in ``PLOTS``, offered when the run holds its ``requires`` keys
     - ``thunder4d/gui/plots.py``
   * - a new tool
     - a ``PanelSpec`` in ``PANELS``
     - ``thunder4d/gui/panels/``

.. code-block:: python

   from thunder4d.gui.registry import PLOTS, PlotSpec

   def draw_stokes(results, fig, options):
       ax = fig.subplots()
       ax.plot(results["pass"], results["S3"] / results["S0"], ".-")

   PLOTS.register(PlotSpec("stokes", "Stokes parameters", draw_stokes, requires=("pass", "S0", "S3")))

The interface holds no physics: it calls the package (:func:`~thunder4d.inputfile.describe_input`,
:func:`~thunder4d.build_simulation`, :mod:`~thunder4d.diagnostics`, :mod:`thunder4d.design`). Long
computations run in a ``thunder4d.gui.worker.Task``. See :doc:`api_reference/gui`.
