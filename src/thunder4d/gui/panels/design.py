"""Design helper panel: from the laser and the limits to ranked cell designs, one click to the builder.

The form edits a thunder4d.design.DesignProblem through DESIGN_FIELDS (attribute path -> Field), so a new
knob of the design helper is one more entry there. The search (screen + refine) runs in the background;
the chosen design goes to the simulation builder as a standard input file (design.to_input).
"""
from __future__ import annotations

import copy
import csv
import json
import os
import re
import time
from typing import Any

from ...design import (
    LASER_PRESETS,
    AnalyticQuick,
    Candidate,
    DesignProblem,
    DesignResult,
    Fast1DQuick,
    LaserSpec,
    MPCDesigner,
    design_map,
    to_input,
)
from ...design.catalog.gases import GASES
from ...design.catalog.mirrors import cavity_coatings
from ...inputfile import GRID_PRESETS, Field
from .. import prefs
from .._qt import QtCore, QtGui, QtWidgets
from ..fields import FieldEditor, editor_for
from ..plotting import FigureWidget
from ..registry import PANELS, PanelSpec
from ..session import Session
from ..splitting import RatioSplitter
from ..worker import Task, notify_done

# Sections of the form, then the fields: attribute path in DesignProblem -> Field.
# A Limit is edited through its value (".value") or its safety factor (".safety").
DESIGN_SECTIONS = {"laser": "Laser", "target": "Target", "constraints": "Limits", "search": "Search"}
DESIGN_FIELDS: dict[str, Field] = {
    "laser.energy_J": Field("Pulse energy", unit="mJ", scale=1e-3, help="Energy of the pulse from the laser."),
    "laser.duration_fs": Field("Duration (FWHM)", unit="fs", help="Duration of the injected pulse."),
    "laser.tl_duration_fs": Field("Transform limit", unit="fs", optional=True, none_text="= duration",
                                  help="If the injected pulse is chirped: its transform-limited duration."),
    "laser.wavelength_nm": Field("Central wavelength", unit="nm", help="Central wavelength of the pulse."),
    "laser.rep_rate_Hz": Field("Repetition rate", unit="kHz", scale=1e3,
                                help="Only sets the average power on the mirrors (thermal limit, off by default)."),
    "laser.polarization": Field("Polarisation", "choice", choices=("linear", "circular"),
                                 help="Circular gives a Kerr index of 2/3 of the linear one."),
    "target.objective": Field("Objective", "choice",
                              choices=("max_peak_power", "shortest", "target_duration", "target_compression"),
                              help="What best means. max_peak_power: highest compressed peak power. shortest: "
                                   "shortest compressed pulse. The two targets: reach the value, not beyond."),
    "target.target_duration_fs": Field("Target duration", unit="fs", optional=True,
                                       help="Compressed duration to reach, not shorter.",
                                       show_if=("target.objective", ("target_duration",))),
    "target.target_compression": Field("Target compression", optional=True,
                                          help="Input duration / compressed duration to reach.",
                                       show_if=("target.objective", ("target_compression",))),
    "target.min_throughput.value": Field("Min throughput", help="Soft limit: penalised below."),
    "constraints.gas.allowed": Field("Gases", "multichoice", choices=tuple(GASES), help="Gases the search may use."),
    "constraints.mirrors.allowed_roc_m": Field("Mirror RoCs available", "float_list", unit="m",
        help="Radii of curvature you can use. Give one value to fix the mirrors you own."),
    "constraints.mirrors.allowed_diameters_inch": Field("Mirror diameters", "float_list", unit="inch",
        help="Diameters available: they set how many spots fit on a mirror."),
    "constraints.space.max_separation_m.value": Field("Max mirror separation", unit="m",
        help="Longest distance allowed between the two mirrors."),
    "constraints.pressure.pressure_bar": Field("Pressure range (min, max)", "float_list", unit="bar",
        help="Lowest and highest pressure tried (rating of the chamber and windows)."),
    "constraints.kerr.p_over_pcr.value": Field("Max P / Pcr",
        help="Peak power over the critical power of self-focusing."),
    "constraints.kerr.b_per_pass_rad.value": Field("Max B per pass", unit="rad",
        help="Kerr phase (B integral) on axis, per pass."),
    "constraints.mirrors.allowed_coatings": Field("Mirror coatings", "multichoice",
                                                  choices=tuple(m.name for m in cavity_coatings()), optional=True,
                                                  help="Coatings tried. None checked: all the cavity coatings."),
    "constraints.mirrors.fluence.value": Field("Max fluence", unit="J/cm²", optional=True, none_text="LIDT of coating",
                                               help="Peak fluence on the mirrors. Empty: the LIDT of each coating."),
    "constraints.mirrors.fluence.safety": Field("Fluence: fraction of the LIDT",
                                                help="Peak fluence <= this x LIDT of the coating."),
    "constraints.ionization.peak_intensity_W_cm2.safety": Field("Intensity: fraction of ionisation",
                                                                help="Peak intensity <= this x threshold."),
    "losses.mode_matching_efficiency": Field("Mode-matching efficiency",
        help="Fraction of the energy injected in the cell mode (1 = perfect)."),
    "search.round_trips": Field("Round trips M (min, max)", "float_list",
        help="Range of M tried: round trips of one re-entrant cycle."),
    "search.k_range": Field("Index k (min, max)", "float_list", optional=True, none_text="all k < M/2",
                            help="Re-entrant index k of the cell. Max k is the second value."),
    "search.branches": Field("Branches", "multichoice", choices=("concentric", "planar"),
                            help="concentric: d > R. planar: d < R."),
    "refinement.n_candidates": Field("Designs to find", "int",
        help="Number of designs returned after the Fast 1+1D refinement."),
}
RANGES = ("constraints.pressure.pressure_bar", "search.round_trips", "search.k_range")     # (min, max) pairs


def checked(path: str, field: Field, value: Any) -> Any:
    """The value of a design field, or ValueError if the search could not use it."""
    if (value is None or value == []) and not field.optional:
        raise ValueError("required" if field.kind != "multichoice" else "choose at least one")
    if value == [] and field.kind == "multichoice":
        return None                                           # nothing checked: no restriction
    if path in RANGES and value is not None and (len(value) != 2 or value[0] > value[1]):
        raise ValueError("give two values: min, max")
    return value


SOLVERS = {"Fast 1+1D (accurate, minutes)": Fast1DQuick, "Analytic (instant, rough)": AnalyticQuick}
COLUMNS = ("Gas", "p [bar]", "R [m]", "M", "k", "Passes", "Branch", "d [m]", "τ out [fs]", "Throughput", "Score")
HELP = {
    "Gas": "Gas filling the cell.",
    "p [bar]": "Gas pressure, tuned by the refinement.",
    "R [m]": "Radius of curvature of the two cavity mirrors.",
    "M": "Round trips of one re-entrant cycle of the Herriott cell (M spots on each mirror).",
    "k": "Re-entrant index, coprime with M. With R and M it sets the separation d = R (1 ± cos(π k / M)).",
    "Passes": "Passes through the cell (at most 2 M). One pass = transit over d + one mirror bounce.",
    "Branch": "concentric: d > R. planar: d < R.",
    "d [m]": "Distance between the two mirrors.",
    "τ out [fs]": "FWHM duration after the output compressor (lossless, optimal GDD).",
    "Throughput": "Fraction of the injected energy that comes out: mirror reflectivity over all bounces, "
                  "mode matching.",
    "Score": "Ranking, lower is better: objective + penalties of the soft limits. "
             "With max_peak_power it is minus the compressed peak power over the input peak power. "
             "Only comparable between designs of the same problem. Breakdown in Details.",
}


def section_of(path: str) -> str:
    head = path.split(".")[0]
    return head if head in DESIGN_SECTIONS else "search"


def get_attr(obj: Any, path: str) -> Any:
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj


def set_attr(obj: Any, path: str, value: Any) -> None:
    *parents, last = path.split(".")
    for part in parents:
        obj = getattr(obj, part)
    old = getattr(obj, last)
    if last == "k_range" and value is not None:               # None or (min, max) of integers
        value = tuple(round(v) for v in value)
    elif isinstance(old, tuple) and value is not None:          # keep the types of the dataclass
        if old and all(isinstance(v, int) for v in old):
            value = tuple(round(v) for v in value)
        else:
            value = tuple(value)
    setattr(obj, last, value)


class StoppableDesigner(MPCDesigner):
    """Stop also works with several workers: it takes effect before the next batch of refinements."""

    def __init__(self, *args: Any, task: Task, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.task = task

    def _refine(self, candidates: Any, max_attempts: Any, pmap: Any) -> DesignResult:
        def stoppable(fn: Any, items: Any) -> Any:
            self.task.check()
            return pmap(fn, items)
        return super()._refine(candidates, max_attempts, stoppable)


class CancellableQuick:
    """Wraps a quick solver so that Stop takes effect at the next evaluation."""

    def __init__(self, inner: Any, task: Task):
        self.inner, self.task = inner, task

    def __call__(self, problem: DesignProblem, cand: Candidate, /) -> Any:
        self.task.check()
        return self.inner(problem, cand)


SORT_KEY = QtCore.Qt.ItemDataRole.UserRole           # what a column is sorted on
INDEX = QtCore.Qt.ItemDataRole.UserRole + 1          # position of the design in DesignPanel.candidates


class SortItem(QtWidgets.QTableWidgetItem):
    """A table cell sorted by its number (not by its text) when it is one."""

    def __lt__(self, other: QtWidgets.QTableWidgetItem) -> bool:
        a, b = self.data(SORT_KEY), other.data(SORT_KEY)
        return a < b if type(a) is type(b) else str(a) < str(b)


def sort_key(text: str) -> float | str:
    try:
        return float(text)
    except ValueError:
        return float("inf") if text == "–" else text


def candidate_row(c: Candidate) -> list[str]:
    m = c.metrics
    tau, thr = m.get("duration_fs"), m.get("throughput")
    return [c.gas, f"{c.pressure_bar:.3f}", f"{c.roc_m:g}", str(c.round_trips), str(c.k), str(c.n_passes), c.branch,
            f"{c.separation_m:.3f}", "–" if tau is None else f"{tau:.1f}", "–" if thr is None else f"{thr:.3f}",
            f"{c.score:.3g}"]


class DesignPanel(QtWidgets.QWidget):
    def __init__(self, session: Session, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self.session = session
        self.problem = DesignProblem()
        self.result: DesignResult | None = None
        self.candidates: list[Candidate] = []
        self.task: Task | None = None
        self.editors: dict[str, FieldEditor] = {}
        self.rows: dict[str, tuple[QtWidgets.QFormLayout, FieldEditor]] = {}
        self.errors: dict[str, str] = {}

        splitter = RatioSplitter(0.4)
        splitter.addWidget(self._form())
        splitter.addWidget(self._results())
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(splitter)
        self.load_problem(self.problem)
        context = QtCore.Qt.ShortcutContext.WidgetWithChildrenShortcut
        for keys, slot in (("Ctrl+Return", self.start), ("Ctrl+Enter", self.start), ("Esc", self.stop),
                           ("Ctrl+O", self.open_problem), ("Ctrl+S", self.save_problem)):
            shortcut = QtGui.QShortcut(QtGui.QKeySequence(keys), self, slot)
            shortcut.setContext(context)

    # ------------------------------------------------------------ form
    def _form(self) -> QtWidgets.QScrollArea:
        content = QtWidgets.QWidget()
        column = QtWidgets.QVBoxLayout(content)
        files = QtWidgets.QHBoxLayout()
        for text, tip, slot in (("Open…", "Load a design problem (JSON), or the problem of saved results (Ctrl+O)",
                                 self.open_problem),
                                ("Save…", "Save the form as a design problem (JSON, Ctrl+S)", self.save_problem),
                                ("Reset", "Back to the default problem", self.reset)):
            button = QtWidgets.QPushButton(text)
            button.setToolTip(tip)
            button.clicked.connect(slot)
            files.addWidget(button)
        column.addLayout(files)
        cell_type = QtWidgets.QComboBox()
        cell_type.addItems(["Gas-filled cell", "Bulk (not available yet)", "Hybrid (not available yet)"])
        model = cell_type.model()
        assert isinstance(model, QtGui.QStandardItemModel)
        for i in (1, 2):                                       # only the gas-filled cell has a design helper
            model.item(i).setEnabled(False)
        cell_type.setToolTip("Kind of cell to design. Bulk and hybrid cells will come with their own design helper.")
        cell_row = QtWidgets.QFormLayout()
        cell_row.addRow("Cell type", cell_type)
        column.addLayout(cell_row)
        forms = {}
        for key, title in DESIGN_SECTIONS.items():
            group = QtWidgets.QGroupBox(title)
            forms[key] = QtWidgets.QFormLayout(group)
            column.addWidget(group)
        self.preset = QtWidgets.QComboBox()
        self.preset.addItem("custom")
        self.preset.addItems(sorted(LASER_PRESETS))
        self.preset.setToolTip("Fill the laser from a datasheet (thunder4d.design.LASER_PRESETS)")
        self.preset.currentTextChanged.connect(self.apply_preset)
        forms["laser"].addRow("Preset", self.preset)
        for path, field in DESIGN_FIELDS.items():
            editor = editor_for(field)
            editor.changed.connect(lambda p=path: self._edited(p))
            label = QtWidgets.QLabel(field.label)
            label.setToolTip(field.help or path)
            forms[section_of(path)].addRow(label, editor)
            self.editors[path] = editor
            self.rows[path] = (forms[section_of(path)], editor)
        self.solver = QtWidgets.QComboBox()
        self.solver.addItems(list(SOLVERS))
        self.solver.setToolTip("Model used to rank and refine the designs (the screen is always analytic)")
        self.solver.currentTextChanged.connect(lambda _: self._update_form())
        forms["search"].addRow("Solver", self.solver)
        self.workers = QtWidgets.QSpinBox()
        cores = getattr(os, "process_cpu_count", os.cpu_count)() or 1       # the cores this process may use
        self.workers.setRange(1, cores)
        self.workers.setValue(cores)
        self.workers.setToolTip("Processes running the Fast 1+1D simulations in parallel.\nDefault: all the cores.")
        forms["search"].addRow("Workers (Fast 1+1D)", self.workers)
        column.addStretch(1)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(content)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setMinimumWidth(content.sizeHint().width() + scroll.verticalScrollBar().sizeHint().width() + 4)
        return scroll

    def load_problem(self, problem: DesignProblem) -> None:
        self.problem = problem
        for path, editor in self.editors.items():
            editor.blockSignals(True)
            editor.set_value(get_attr(problem, path))
            editor.blockSignals(False)
        self.errors.clear()
        self._update_form()

    def reset(self) -> None:
        if self.task is not None:
            return
        self.preset.blockSignals(True)
        self.preset.setCurrentText("custom")
        self.preset.blockSignals(False)
        self.load_problem(DesignProblem())

    def open_problem(self) -> None:
        if self.task is not None:
            return
        path = prefs.open_file(self, "Open a design problem", "JSON (*.json)")
        if path:
            self.open_file(path)

    def open_file(self, path: str) -> bool:
        if self.task is not None:
            return False
        try:
            with open(path) as f:
                problem = DesignProblem.from_dict(json.load(f))
        except (OSError, ValueError, TypeError, AttributeError) as err:
            self.state.setText(f"Cannot open {path}: {err}")
            return False
        self.preset.blockSignals(True)
        self.preset.setCurrentText("custom")
        self.preset.blockSignals(False)
        self.load_problem(problem)
        prefs.add_recent("problem", path)
        self.state.setText(f"Opened {path}")
        return True

    def save_problem(self) -> None:
        path = prefs.save_file(self, "Save the design problem", "design_problem.json", "JSON (*.json)")
        if path:
            with open(path, "w") as f:
                json.dump(self.problem.to_dict(), f, indent=2)
            prefs.add_recent("problem", path)
            self.state.setText(f"Saved {path}")

    def apply_preset(self, name: str) -> None:
        if name == "custom":
            return
        self.problem.laser = LaserSpec.preset(name)
        for path, editor in self.editors.items():
            if path.startswith("laser."):
                editor.set_value(get_attr(self.problem, path))
        self._update_form()

    def _edited(self, path: str) -> None:
        try:
            set_attr(self.problem, path, checked(path, DESIGN_FIELDS[path], self.editors[path].value()))
            self.errors.pop(path, None)
        except ValueError as err:
            self.errors[path] = f"{DESIGN_FIELDS[path].label}: {err}"
        if path.startswith("laser.") and self.preset.currentText() != "custom":
            self.preset.blockSignals(True)
            self.preset.setCurrentText("custom")        # edited: no longer the datasheet values
            self.preset.blockSignals(False)
        self._update_form()

    def _update_form(self) -> None:
        for path, (form, editor) in self.rows.items():
            field = editor.field
            if field.show_if is not None:
                other, values = field.show_if
                form.setRowVisible(editor, get_attr(self.problem, other) in values)
            editor.set_problem(self.errors.get(path))
        idle = self.task is None
        self.workers.setEnabled(idle and SOLVERS[self.solver.currentText()] is Fast1DQuick)
        self.run_button.setEnabled(idle and not self.errors)
        self.run_button.setToolTip("\n".join(self.errors.values()))

    # ------------------------------------------------------------ results side
    def _results(self) -> QtWidgets.QWidget:
        box = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(box)
        bar = QtWidgets.QHBoxLayout()
        self.run_button = QtWidgets.QPushButton("Find designs")
        self.run_button.clicked.connect(self.start)
        self.stop_button = QtWidgets.QPushButton("Stop")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop)
        self.run_button.setToolTip("Ctrl+Enter")
        self.progress = QtWidgets.QProgressBar()
        self.progress.setFixedWidth(160)
        self.progress.setTextVisible(False)
        self.progress.hide()
        self.elapsed = QtWidgets.QLabel()
        self.clock = QtCore.QTimer(self)
        self.clock.timeout.connect(self._tick)
        self.state = QtWidgets.QLabel("Set the laser and the limits, then Find designs.")
        self.state.setWordWrap(True)
        self.state.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Preferred)
        for w in (self.run_button, self.stop_button):
            bar.addWidget(w)
        bar.addWidget(self.progress)
        bar.addWidget(self.elapsed)
        bar.addWidget(self.state, 1)
        layout.addLayout(bar)

        self.table = QtWidgets.QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels(list(COLUMNS))
        for i, name in enumerate(COLUMNS):
            header = self.table.horizontalHeaderItem(i)
            assert header is not None
            header.setToolTip(HELP[name])
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setToolTip("Click a column to sort on it")
        self.table.itemSelectionChanged.connect(self._selected)
        self.table.doubleClicked.connect(lambda _: self.use_in_builder())

        export = QtWidgets.QHBoxLayout()
        self.use_button = QtWidgets.QPushButton("Use in simulation builder")
        self.use_button.clicked.connect(self.use_in_builder)
        self.grid_preset = QtWidgets.QComboBox()
        self.grid_preset.addItems(list(GRID_PRESETS))
        self.injection = QtWidgets.QComboBox()
        self.injection.addItems(["linear", "nonlinear"])
        self.save_button = QtWidgets.QPushButton("Save results…")
        self.save_button.clicked.connect(self.save_results)
        self.copy_button = QtWidgets.QPushButton("Copy table")
        self.copy_button.setToolTip("Copy the designs to the clipboard (paste in a spreadsheet)")
        self.copy_button.clicked.connect(self.copy_table)
        self.csv_button = QtWidgets.QPushButton("CSV…")
        self.csv_button.setToolTip("Save the designs as a CSV file")
        self.csv_button.clicked.connect(self.export_csv)
        for w in (self.use_button, QtWidgets.QLabel("grid"), self.grid_preset, QtWidgets.QLabel("injection"),
                  self.injection):
            export.addWidget(w)
        export.addStretch(1)
        export.addWidget(self.copy_button)
        export.addWidget(self.csv_button)
        export.addWidget(self.save_button)

        self.details = QtWidgets.QPlainTextEdit()
        self.details.setReadOnly(True)
        self.details.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.FixedFont))
        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setFont(self.details.font())
        self.log_box = prefs.log_box(self.log)
        map_box = QtWidgets.QWidget()
        map_layout = QtWidgets.QVBoxLayout(map_box)
        self.map_button = QtWidgets.QPushButton("Map pressure × round trips around the selected design")
        self.map_button.setToolTip("Analytic estimates: which limit is binding where (instant)")
        self.map_button.clicked.connect(self.compute_map)
        self.map_figure = FigureWidget()
        map_layout.addWidget(self.map_button)
        map_layout.addWidget(self.map_figure, 1)
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(self.details, "Details")
        self.tabs.addTab(map_box, "Design map")
        self.tabs.addTab(self.log_box, "Log")

        lower = QtWidgets.QWidget()
        lower_layout = QtWidgets.QVBoxLayout(lower)
        lower_layout.setContentsMargins(0, 0, 0, 0)
        lower_layout.addLayout(export)
        lower_layout.addWidget(self.tabs, 1)
        split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        split.addWidget(self.table)
        split.addWidget(lower)
        layout.addWidget(split, 1)
        self._set_selection_actions()
        return box

    def _set_selection_actions(self) -> None:
        has = self.selected() is not None
        for w in (self.use_button, self.map_button):
            w.setEnabled(has and self.task is None)
        self.save_button.setEnabled(self.result is not None and self.task is None)
        for w in (self.copy_button, self.csv_button):
            w.setEnabled(bool(self.candidates))

    def selected(self) -> Candidate | None:
        rows = self.table.selectionModel().selectedRows() if self.table.selectionModel() else []
        if not rows:
            return None
        item = self.table.item(rows[0].row(), 0)
        assert item is not None
        return self.candidates[item.data(INDEX)]

    def _selected(self) -> None:
        c = self.selected()
        self.details.setPlainText(c.report(self.problem) if c is not None else "")
        self._set_selection_actions()

    # ------------------------------------------------------------ search
    def start(self) -> None:
        if self.task is not None or self.errors:
            return
        problem = copy.deepcopy(self.problem)
        inner = SOLVERS[self.solver.currentText()]()
        workers = self.workers.value() if isinstance(inner, Fast1DQuick) else 1

        def work(task: Task) -> DesignResult:
            quick = CancellableQuick(inner, task) if workers == 1 else inner     # a Task cannot go to a process
            designer = StoppableDesigner(problem, quick=quick, verbose=True, workers=workers, task=task)
            starts = designer.screen()
            task.check()
            return designer.refine(starts)

        self.log.clear()
        self.tabs.setCurrentWidget(self.log_box)
        self._run(Task(work, self), "Searching…", self._done)

    def _run(self, task: Task, message: str, on_success: Any) -> None:
        self.task = task
        task.output.connect(self.log.appendPlainText)
        task.output.connect(self._progress)
        self.t0 = time.monotonic()
        self.clock.start(1000)
        self._tick()
        self.progress.setRange(0, 0)                       # busy until the search reports how many designs it wants
        self.progress.show()
        task.succeeded.connect(on_success)
        task.failed.connect(self._failed)
        task.cancelled.connect(lambda: self.state.setText("Stopped."))
        task.finished.connect(self._finished)
        self.state.setText(message)
        self.stop_button.setEnabled(True)
        self._update_form()
        self._set_selection_actions()
        task.start()

    def _tick(self) -> None:
        minutes, seconds = divmod(int(time.monotonic() - self.t0), 60)
        self.elapsed.setText(f"{minutes}:{seconds:02d}")

    def _progress(self, line: str) -> None:
        found = re.match(r"\[(\d+)/(\d+), try", line)      # "[2/5, try 7] ..." printed by the designer
        if found:
            self.progress.setRange(0, int(found[2]))
            self.progress.setValue(int(found[1]) - ("no feasible" in line))     # failed tries print the next index

    def stop(self) -> None:
        if self.task is not None:
            self.task.cancel()
            self.state.setText("Stopping after the current evaluation…")

    def _finished(self) -> None:
        if self.task is not None:
            self.task.deleteLater()
        self.task = None
        self.stop_button.setEnabled(False)
        self.progress.hide()
        self.clock.stop()
        self._update_form()
        self._set_selection_actions()

    def _failed(self, trace: str) -> None:
        self.log.appendPlainText(trace)
        self.tabs.setCurrentWidget(self.log_box)
        self.state.setText("Failed: see the log.")

    def _done(self, result: DesignResult) -> None:
        self.show_result(result)
        notify_done(self, f"Design search finished: {len(self.candidates)} design(s).")

    def show_result(self, result: DesignResult) -> None:
        self.result = result
        self.candidates = sorted(result.refined, key=lambda c: c.score)
        self.table.setSortingEnabled(False)
        self.table.setRowCount(0)
        for i, c in enumerate(self.candidates):
            r = self.table.rowCount()
            self.table.insertRow(r)
            for col, text in enumerate(candidate_row(c)):
                item = SortItem(text)
                item.setData(SORT_KEY, sort_key(text))
                item.setData(INDEX, i)
                if col not in (0, 6):
                    item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
                self.table.setItem(r, col, item)
        self.table.setSortingEnabled(True)
        self.table.sortByColumn(len(COLUMNS) - 1, QtCore.Qt.SortOrder.AscendingOrder)       # the score
        if self.candidates:
            self.state.setText(f"{len(self.candidates)} design(s) found. Double-click one to simulate it.")
            self.table.selectRow(0)
            self.tabs.setCurrentWidget(self.details)
        else:
            self.state.setText("No feasible design: relax the limits (the log says which ones bind).")
        self._set_selection_actions()

    # ------------------------------------------------------------ actions on a design
    def use_in_builder(self) -> None:
        c = self.selected()
        if c is None or self.result is None:
            return
        cfg = to_input(self.result.problem, c, preset=self.grid_preset.currentText(),
                       injection=self.injection.currentText())
        self.session.set_config(cfg)
        self.session.show_panel("builder")

    def compute_map(self) -> None:
        c = self.selected()
        if c is None or self.result is None or self.task is not None:
            return
        problem = self.result.problem

        def work(task: Task) -> Any:
            return design_map(problem, gas=c.gas, roc_m=c.roc_m, k=c.k, branch=c.branch, coating=c.coating,
                              diameter_inch=c.diameter_inch)

        def draw(dmap: Any) -> None:
            self.map_figure.clear()
            ax = self.map_figure.figure.add_subplot()
            dmap.plot(ax=ax, mark=[(c.pressure_bar, c.round_trips, "selected")])
            self.map_figure.draw()
            self.state.setText(dmap.summary().splitlines()[0])

        self.tabs.setCurrentIndex(1)
        self._run(Task(work, self), "Mapping…", draw)

    def can_close(self) -> bool:
        if self.task is not None:
            self.task.cancel()
            self.task.wait()
        return True

    def table_rows(self) -> list[list[str]]:
        """The header, then the designs in the order shown."""
        rows = [list(COLUMNS)]
        for r in range(self.table.rowCount()):
            rows.append([getattr(self.table.item(r, c), "text", str)() for c in range(len(COLUMNS))])
        return rows

    def copy_table(self) -> None:
        QtWidgets.QApplication.clipboard().setText("\n".join("\t".join(row) for row in self.table_rows()))
        self.state.setText("Designs copied to the clipboard.")

    def export_csv(self) -> None:
        path = prefs.save_file(self, "Save the designs", "designs.csv", "CSV (*.csv)")
        if path:
            with open(path, "w", newline="") as f:
                csv.writer(f).writerows(self.table_rows())
            self.state.setText(f"Saved {path}")

    def save_results(self) -> None:
        if self.result is None:
            return
        path = prefs.save_file(self, "Save design results", "design.json", "JSON (*.json)")
        if path:
            self.result.save_json(path)
            self.state.setText(f"Saved {path}")


PANELS.register(PanelSpec("design", "Design helper", DesignPanel, order=5,
                          tooltip="Find a gas-filled cell for your laser and limits"))
