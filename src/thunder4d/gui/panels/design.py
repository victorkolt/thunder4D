"""Design helper panel: from the laser and the limits to ranked cell designs, one click to the builder.

The form edits a thunder4d.design.DesignProblem through DESIGN_FIELDS (attribute path -> Field), so a new
knob of the design helper is one more entry there. The search (screen + refine) runs in the background;
the chosen design goes to the simulation builder as a standard input file (design.to_input).
"""
from __future__ import annotations

import copy
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
from ...inputfile import GRID_PRESETS, Field
from .._qt import QtCore, QtGui, QtWidgets
from ..fields import FieldEditor, editor_for
from ..plotting import FigureWidget
from ..registry import PANELS, PanelSpec
from ..session import Session
from ..worker import Task

# Sections of the form, then the fields: attribute path in DesignProblem -> Field.
# A Limit is edited through its value (".value") or its safety factor (".safety").
DESIGN_SECTIONS = {"laser": "Laser", "target": "Target", "constraints": "Limits", "search": "Search"}
DESIGN_FIELDS: dict[str, Field] = {
    "laser.energy_J": Field("Pulse energy", unit="mJ", scale=1e-3),
    "laser.duration_fs": Field("Duration (FWHM)", unit="fs", help="Duration of the injected pulse."),
    "laser.tl_duration_fs": Field("Transform limit", unit="fs", optional=True, none_text="= duration",
                                  help="If the injected pulse is chirped: its transform-limited duration."),
    "laser.wavelength_nm": Field("Central wavelength", unit="nm"),
    "laser.rep_rate_Hz": Field("Repetition rate", unit="kHz", scale=1e3),
    "laser.polarization": Field("Polarisation", "choice", choices=("linear", "circular")),
    "target.objective": Field("Objective", "choice",
                              choices=("max_peak_power", "shortest", "target_duration", "target_compression")),
    "target.target_duration_fs": Field("Target duration", unit="fs", optional=True,
                                       show_if=("target.objective", ("target_duration",))),
    "target.target_compression": Field("Target compression", optional=True,
                                       show_if=("target.objective", ("target_compression",))),
    "target.min_throughput.value": Field("Min throughput", help="Soft limit: penalised below."),
    "constraints.gas.allowed": Field("Gases", "multichoice", choices=tuple(GASES)),
    "constraints.mirrors.allowed_roc_m": Field("Mirror RoCs available", "float_list", unit="m"),
    "constraints.mirrors.allowed_diameters_inch": Field("Mirror diameters", "float_list", unit="inch"),
    "constraints.space.max_separation_m.value": Field("Max mirror separation", unit="m"),
    "constraints.pressure.pressure_bar": Field("Pressure range (min, max)", "float_list", unit="bar"),
    "constraints.kerr.p_over_pcr.value": Field("Max P / Pcr"),
    "constraints.kerr.b_per_pass_rad.value": Field("Max B per pass", unit="rad"),
    "constraints.mirrors.fluence.safety": Field("Fluence: fraction of the LIDT",
                                                help="Peak fluence <= this x LIDT of the coating."),
    "constraints.ionization.peak_intensity_W_cm2.safety": Field("Intensity: fraction of ionisation",
                                                                help="Peak intensity <= this x threshold."),
    "losses.mode_matching_efficiency": Field("Mode-matching efficiency"),
    "search.round_trips": Field("Round trips M (min, max)", "float_list"),
    "search.branches": Field("Branches", "multichoice", choices=("concentric", "planar")),
    "refinement.n_candidates": Field("Designs to find", "int"),
}
SOLVERS = {"Fast 1+1D (accurate, minutes)": Fast1DQuick, "Analytic (instant, rough)": AnalyticQuick}
COLUMNS = ("Gas", "p [bar]", "R [m]", "M", "k", "Passes", "Branch", "d [m]", "τ out [fs]", "Throughput", "Score")


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
    if isinstance(old, tuple) and value is not None:          # keep the types of the dataclass
        if old and all(isinstance(v, int) for v in old):
            value = tuple(round(v) for v in value)
        else:
            value = tuple(value)
    setattr(obj, last, value)


class CancellableQuick:
    """Wraps a quick solver so that Stop takes effect at the next evaluation."""

    def __init__(self, inner: Any, task: Task):
        self.inner, self.task = inner, task

    def __call__(self, problem: DesignProblem, cand: Candidate, /) -> Any:
        self.task.check()
        return self.inner(problem, cand)


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

        splitter = QtWidgets.QSplitter()
        splitter.addWidget(self._form())
        splitter.addWidget(self._results())
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(splitter)
        self.load_problem(self.problem)

    # ------------------------------------------------------------ form
    def _form(self) -> QtWidgets.QScrollArea:
        content = QtWidgets.QWidget()
        column = QtWidgets.QVBoxLayout(content)
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
        forms["search"].addRow("Solver", self.solver)
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
            set_attr(self.problem, path, self.editors[path].value())
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
        self.state = QtWidgets.QLabel("Set the laser and the limits, then Find designs.")
        self.state.setWordWrap(True)
        self.state.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Preferred)
        for w in (self.run_button, self.stop_button):
            bar.addWidget(w)
        bar.addWidget(self.state, 1)
        layout.addLayout(bar)

        self.table = QtWidgets.QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels(list(COLUMNS))
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
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
        for w in (self.use_button, QtWidgets.QLabel("grid"), self.grid_preset, QtWidgets.QLabel("injection"),
                  self.injection):
            export.addWidget(w)
        export.addStretch(1)
        export.addWidget(self.save_button)

        self.details = QtWidgets.QPlainTextEdit()
        self.details.setReadOnly(True)
        self.details.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.FixedFont))
        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setFont(self.details.font())
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
        self.tabs.addTab(self.log, "Log")

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

    def selected(self) -> Candidate | None:
        rows = self.table.selectionModel().selectedRows() if self.table.selectionModel() else []
        return self.candidates[rows[0].row()] if rows else None

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

        def work(task: Task) -> DesignResult:
            designer = MPCDesigner(problem, quick=CancellableQuick(inner, task), verbose=True)
            starts = designer.screen()
            task.check()
            return designer.refine(starts)

        self.log.clear()
        self.tabs.setCurrentWidget(self.log)
        self._run(Task(work, self), "Searching…", self._done)

    def _run(self, task: Task, message: str, on_success: Any) -> None:
        self.task = task
        task.output.connect(self.log.appendPlainText)
        task.succeeded.connect(on_success)
        task.failed.connect(self._failed)
        task.cancelled.connect(lambda: self.state.setText("Stopped."))
        task.finished.connect(self._finished)
        self.state.setText(message)
        self.stop_button.setEnabled(True)
        self._update_form()
        self._set_selection_actions()
        task.start()

    def stop(self) -> None:
        if self.task is not None:
            self.task.cancel()
            self.state.setText("Stopping after the current evaluation…")

    def _finished(self) -> None:
        if self.task is not None:
            self.task.deleteLater()
        self.task = None
        self.stop_button.setEnabled(False)
        self._update_form()
        self._set_selection_actions()

    def _failed(self, trace: str) -> None:
        self.log.appendPlainText(trace)
        self.tabs.setCurrentWidget(self.log)
        self.state.setText("Failed: see the log.")

    def _done(self, result: DesignResult) -> None:
        self.show_result(result)

    def show_result(self, result: DesignResult) -> None:
        self.result = result
        self.candidates = sorted(result.refined, key=lambda c: c.score)
        self.table.setRowCount(0)
        for c in self.candidates:
            r = self.table.rowCount()
            self.table.insertRow(r)
            for col, text in enumerate(candidate_row(c)):
                item = QtWidgets.QTableWidgetItem(text)
                if col not in (0, 6):
                    item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
                self.table.setItem(r, col, item)
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

    def save_results(self) -> None:
        if self.result is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save design results", "design.json",
                                                        "JSON (*.json)")
        if path:
            self.result.save_json(path)
            self.state.setText(f"Saved {path}")


PANELS.register(PanelSpec("design", "Design helper", DesignPanel, order=5,
                          tooltip="Find a gas-filled cell for your laser and limits"))
