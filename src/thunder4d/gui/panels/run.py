"""Run control: launch the input of the builder, follow it pass by pass, stop it, save the results.

The run is MPCSimulation.run(n_passes=1) in a loop on a background thread, so Stop takes effect after the
current pass. What is followed comes from registry.METRICS (values and curves versus pass), so a new
quantity of the history is followed by registering it there.
"""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ...inputfile import build_simulation, describe_input
from .. import metrics as _builtin_metrics  # noqa: F401  (registers the built-in metrics)
from .._qt import QtGui, QtWidgets
from ..plotting import FigureWidget
from ..registry import METRICS, PANELS, PanelSpec
from ..results import load_results
from ..session import Session
from ..worker import Task

WARNING_STYLE = "color: #d32f2f; font-weight: bold;"


@dataclass
class Snapshot:
    """What the run thread sends after each pass (copies: the GUI never reads the live history)."""
    passes_done: int
    n_passes: int
    elapsed_s: float
    recorded: bool
    values: dict[str, float]
    lam: np.ndarray | None = None
    spectrum: np.ndarray | None = None


@dataclass
class Outcome:
    path: Path
    passes_done: int
    n_passes: int

    @property
    def stopped(self) -> bool:
        return self.passes_done < self.n_passes


def snapshot(sim: Any, t0: float, recorded: bool) -> Snapshot:
    h = sim.history
    values = {}
    if recorded:
        for key, series in h.items():
            last = series[-1] if series else None
            if isinstance(last, (int, float, np.floating, np.integer)) and key != "pass":
                values[key] = float(last)
    spectrum = np.array(h["spectrum"][-1], copy=True) if recorded and h["spectrum"] else None
    return Snapshot(sim.passes_done, sim.cell.n_passes, time.time() - t0, recorded, values,
                    np.array(sim.lam, copy=True) if spectrum is not None else None, spectrum)


def simulate(cfg: dict[str, Any], path: Path, include_field: bool, task: Task) -> Outcome:
    """Build and run `cfg` pass by pass, reporting a Snapshot after each one; saves to `path`, also when stopped."""
    t0 = time.time()
    sim = build_simulation(cfg, verbose=True).sim
    task.report(snapshot(sim, t0, recorded=True))
    try:
        while not sim.done and not task.stop_requested:
            recorded_before = len(sim.history["pass"])
            sim.run(n_passes=1, verbose=True)
            task.report(snapshot(sim, t0, recorded=len(sim.history["pass"]) > recorded_before))
        if task.stop_requested:
            print(f"stopped after pass {sim.passes_done}/{sim.cell.n_passes}")
    finally:                                   # an error in a pass keeps the passes already computed
        path.parent.mkdir(parents=True, exist_ok=True)
        sim.save(path, include_field=include_field)
        print(f"saved {path} ({sim.passes_done}/{sim.cell.n_passes} passes)")
    return Outcome(path, sim.passes_done, sim.cell.n_passes)


class RunPanel(QtWidgets.QWidget):
    def __init__(self, session: Session, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self.session = session
        self.task: Task | None = None
        self.series: dict[str, tuple[list[int], list[float]]] = {}
        self.spectra: list[tuple[np.ndarray, np.ndarray]] = []
        self.outcome: Outcome | None = None
        self._output_edited = False

        layout = QtWidgets.QVBoxLayout(self)
        self.input_label = QtWidgets.QLabel()
        self.input_label.setWordWrap(True)
        layout.addWidget(self.input_label)

        out = QtWidgets.QHBoxLayout()
        out.addWidget(QtWidgets.QLabel("Results file"))
        self.output = QtWidgets.QLineEdit()
        self.output.textEdited.connect(self._output_changed)
        out.addWidget(self.output, 1)
        browse = QtWidgets.QPushButton("Browse…")
        browse.clicked.connect(self.browse)
        out.addWidget(browse)
        self.include_field = QtWidgets.QCheckBox("Save the final field")
        self.include_field.setChecked(True)
        self.include_field.setToolTip("Needed for the compressed pulse and the profiles in the results viewer")
        out.addWidget(self.include_field)
        layout.addLayout(out)

        bar = QtWidgets.QHBoxLayout()
        self.run_button = QtWidgets.QPushButton("Run")
        self.run_button.clicked.connect(self.start)
        self.stop_button = QtWidgets.QPushButton("Stop")
        self.stop_button.setToolTip("Stops after the current pass and saves what was computed")
        self.stop_button.clicked.connect(self.stop)
        self.progress = QtWidgets.QProgressBar()
        self.progress.setFormat("pass %v / %m")
        self.timing = QtWidgets.QLabel()
        self.open_button = QtWidgets.QPushButton("Open in results viewer")
        self.open_button.clicked.connect(lambda: self.session.show_panel("results"))
        for w in (self.run_button, self.stop_button):
            bar.addWidget(w)
        bar.addWidget(self.progress, 1)
        bar.addWidget(self.timing)
        bar.addWidget(self.open_button)
        layout.addLayout(bar)

        self.cards = QtWidgets.QGridLayout()
        self.card_labels: dict[str, QtWidgets.QLabel] = {}
        layout.addLayout(self.cards)

        self.figure = FigureWidget()
        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.FixedFont))
        self.log.setMaximumBlockCount(5000)
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(self.figure, "Live plots")
        self.tabs.addTab(self.log, "Log")
        layout.addWidget(self.tabs, 1)

        session.config_changed.connect(self._config_changed)
        self._config_changed()

    # ------------------------------------------------------------ input
    def _config_changed(self, _: Any = None) -> None:
        cfg = self.session.config          # the current input (a panel may have replaced the one signalled)
        if cfg is None:
            self.input_label.setText("<b>No valid input.</b> Fill in the Simulation tab (or open an input file).")
        else:
            try:
                s = describe_input(cfg)
                nx, ny, nt = s["grid_shape"]
                detail = (f"{cfg['medium']['gas']} {cfg['medium']['pressure_bar']:.4g} bar, "
                          f"{cfg['cell']['n_passes']} passes, grid {nx} × {ny} × {nt} ({cfg['grid']['preset']})")
            except Exception as err:  # noqa: BLE001
                detail = f"cannot be built: {err}"
            self.input_label.setText(f"<b>Input:</b> {cfg.get('name') or 'unnamed'}: {detail}")
        if not self._output_edited:
            self.output.setText(str(self.default_output(cfg)))
        self._update_buttons()

    def default_output(self, cfg: dict[str, Any] | None) -> Path:
        name = (cfg or {}).get("name") or "run"
        folder = self.session.config_path.parent if self.session.config_path else Path.cwd()
        return folder / f"{name}.npz"

    def _output_changed(self) -> None:
        self._output_edited = True

    def browse(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Results file", self.output.text(),
                                                        "NumPy archive (*.npz)")
        if path:
            self.output.setText(path if path.endswith(".npz") else path + ".npz")
            self._output_edited = True

    def _update_buttons(self) -> None:
        running = self.task is not None
        self.run_button.setEnabled(not running and self.session.config is not None)
        self.stop_button.setEnabled(running)
        self.open_button.setEnabled(not running and self.outcome is not None)

    # ------------------------------------------------------------ run
    def start(self) -> None:
        cfg = self.session.config
        if cfg is None or self.task is not None:
            return
        path = Path(self.output.text()).expanduser()
        if path.exists():
            answer = QtWidgets.QMessageBox.question(self, "Run", f"{path} exists. Overwrite it?")
            if answer != QtWidgets.QMessageBox.StandardButton.Yes:
                return
        self.start_run(copy.deepcopy(cfg), path)

    def start_run(self, cfg: dict[str, Any], path: Path) -> None:
        include_field = self.include_field.isChecked()
        self.series.clear()
        self.spectra.clear()
        self.outcome = None
        self.log.clear()
        self.progress.setRange(0, int(cfg["cell"]["n_passes"]))
        self.progress.setValue(0)
        self.timing.setText("building…")
        task = Task(lambda t: simulate(cfg, path, include_field, t), self)
        task.output.connect(self.log.appendPlainText)
        task.progress.connect(self.on_snapshot)
        task.succeeded.connect(self.on_done)
        task.failed.connect(self.on_failed)
        task.finished.connect(self._finished)
        self.task = task
        self._update_buttons()
        task.start()

    def stop(self) -> None:
        if self.task is not None:
            self.task.cancel()
            self.timing.setText("stopping after this pass…")

    def _finished(self) -> None:
        if self.task is not None:
            self.task.deleteLater()
        self.task = None
        self._update_buttons()

    def on_failed(self, trace: str) -> None:
        self.log.appendPlainText(trace)
        self.tabs.setCurrentWidget(self.log)
        self.timing.setText("failed: see the log")

    def on_done(self, outcome: Outcome) -> None:
        self.outcome = outcome
        state = "stopped" if outcome.stopped else "done"
        self.timing.setText(f"{state}: {outcome.passes_done}/{outcome.n_passes} passes, saved {outcome.path.name}")
        try:
            self.session.set_results(load_results(outcome.path))
        except (OSError, ValueError) as err:
            self.log.appendPlainText(f"cannot read back {outcome.path}: {err}")

    # ------------------------------------------------------------ live view
    def on_snapshot(self, snap: Snapshot) -> None:
        self.progress.setValue(snap.passes_done)
        if snap.passes_done:
            per_pass = snap.elapsed_s / snap.passes_done
            eta = per_pass * (snap.n_passes - snap.passes_done)
            self.timing.setText(f"{snap.elapsed_s:.0f} s elapsed, {per_pass:.1f} s/pass, ~{eta:.0f} s left")
        if not snap.recorded:
            return
        for key, value in snap.values.items():
            passes, values = self.series.setdefault(key, ([], []))
            passes.append(snap.passes_done)
            values.append(value)
        if snap.spectrum is not None and snap.lam is not None:
            self.spectra.append((snap.lam, snap.spectrum))
        self._show_cards(snap.values)
        self._draw()

    def _show_cards(self, values: dict[str, float]) -> None:
        history = {k: [v] for k, v in values.items()}
        shown = [m for m in METRICS if m.key in values]
        for i, m in enumerate(shown):
            label = self.card_labels.get(m.key)
            if label is None:
                label = QtWidgets.QLabel()
                label.setToolTip(m.key)
                self.card_labels[m.key] = label
                self.cards.addWidget(label, i // 4, i % 4)
            value = m.value(history)
            label.setText(f"{m.label}: <b>{m.text(history)}</b>")
            label.setStyleSheet(WARNING_STYLE if m.warning(value) else "")

    def _draw(self) -> None:
        plotted = [m for m in METRICS if m.plot and m.key in self.series]
        n = len(plotted) + (1 if self.spectra else 0)
        if n == 0:
            return
        fig = self.figure.figure
        fig.clear()
        cols = min(3, n)
        rows = -(-n // cols)
        for i, m in enumerate(plotted):
            ax = fig.add_subplot(rows, cols, i + 1)
            passes, values = self.series[m.key]
            ax.plot(passes, np.asarray(values) * m.scale, ".-")
            ax.set_xlabel("pass")
            ax.set_title(f"{m.label}" + (f" [{m.unit}]" if m.unit else ""), fontsize="medium")
            if m.warn_above is not None:
                ax.axhline(m.warn_above, color="#d32f2f", ls="--", lw=1)
        if self.spectra:
            ax = fig.add_subplot(rows, cols, n)
            for (lam, s), label in ((self.spectra[0], "input"), (self.spectra[-1], "now")):
                ax.semilogy(lam * 1e9, s / s.max(), label=label)
            ax.set_ylim(1e-4, 2)
            ax.set_xlabel("wavelength [nm]")
            ax.set_title("Spectrum", fontsize="medium")
            ax.legend(fontsize="small")
        self.figure.draw()

    # ------------------------------------------------------------ closing the window
    def can_close(self) -> bool:
        if self.task is None:
            return True
        answer = QtWidgets.QMessageBox.question(
            self, "Simulation running", "Stop the simulation after the current pass (results saved) and quit?")
        if answer != QtWidgets.QMessageBox.StandardButton.Yes:
            return False
        self.task.cancel()
        self.task.wait()
        return True


PANELS.register(PanelSpec("run", "Run", RunPanel, order=20, tooltip="Launch the simulation and follow it live"))
