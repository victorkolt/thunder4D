"""Results viewer: every registered figure that the run can draw, a pass selector, export of figures and data.

The list of figures is registry.PLOTS filtered by the arrays of the results (PlotSpec.requires), so a new
output gets its figure by registering a PlotSpec (see gui/plots.py).
"""
from __future__ import annotations

import csv
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from .. import plots as _builtin_plots  # noqa: F401  (registers the built-in figures)
from .._qt import QtCore, QtWidgets
from ..plotting import FigureWidget
from ..registry import PANELS, PLOTS, PanelSpec, PlotSpec, available_plots
from ..results import Results, load_results
from ..session import Session

FIGURE_FILTER = "PNG (*.png);;PDF (*.pdf);;SVG (*.svg)"
DATA_FILTER = "NumPy archive (*.npz);;CSV, 1D data only (*.csv)"


def summary_text(r: Results) -> str:
    """One line on the run: gas, cell, passes, transmission, B."""
    parts = []
    if "gas" in r and "pressure_bar" in r:
        parts.append(f"{r['gas']} {float(r['pressure_bar']):.4g} bar")
    if "R" in r and "d" in r:
        parts.append(f"R = {float(r['R']):g} m, d = {float(r['d']):.4g} m")
    if "pass" in r:
        parts.append(f"{int(r['pass'][-1])} passes")
    if "energy" in r:
        parts.append(f"transmission {r['energy'][-1] / r['energy'][0]:.4f}")
    if "B" in r:
        parts.append(f"B = {float(r['B'][-1]):.3g} rad")
    return " | ".join(parts)


def write_csv(path: str | Path, data: dict[str, Any]) -> None:
    """1D arrays as columns (shorter ones padded with empty cells); scalars as one-row columns."""
    columns = {k: np.atleast_1d(np.asarray(v)) for k, v in data.items()}
    if any(v.ndim > 1 for v in columns.values()):
        raise ValueError("CSV holds 1D data only: export this figure as .npz")
    n = max(len(v) for v in columns.values())
    with open(path, "w", newline="") as f:
        out = csv.writer(f)
        out.writerow(list(columns))
        for i in range(n):
            out.writerow([v[i] if i < len(v) else "" for v in columns.values()])


class ResultsPanel(QtWidgets.QWidget):
    def __init__(self, session: Session, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self.session = session
        self.results: Results | None = None
        self.plots: list[PlotSpec] = []

        layout = QtWidgets.QVBoxLayout(self)
        bar = QtWidgets.QHBoxLayout()
        open_button = QtWidgets.QPushButton("Open results…")
        open_button.clicked.connect(self.open_dialog)
        self.title = QtWidgets.QLabel("No results: run a simulation, or open a .npz file.")
        self.title.setWordWrap(True)
        self.title.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Preferred)
        bar.addWidget(open_button)
        bar.addWidget(self.title, 1)
        self.export_figure_button = QtWidgets.QPushButton("Export figure…")
        self.export_figure_button.clicked.connect(self.export_figure_dialog)
        self.export_data_button = QtWidgets.QPushButton("Export data…")
        self.export_data_button.clicked.connect(self.export_data_dialog)
        bar.addWidget(self.export_figure_button)
        bar.addWidget(self.export_data_button)
        layout.addLayout(bar)

        self.list = QtWidgets.QTreeWidget()
        self.list.setHeaderHidden(True)
        self.list.setMaximumWidth(260)
        self.list.currentItemChanged.connect(lambda *_: self.redraw())
        right = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        pass_row = QtWidgets.QHBoxLayout()
        self.pass_label = QtWidgets.QLabel("Pass")
        self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider.valueChanged.connect(self._pass_changed)
        pass_row.addWidget(self.pass_label)
        pass_row.addWidget(self.slider, 1)
        right_layout.addLayout(pass_row)
        self.figure = FigureWidget()
        right_layout.addWidget(self.figure, 1)
        self.message = QtWidgets.QLabel()
        self.message.setWordWrap(True)
        self.message.setStyleSheet("color: #d32f2f;")
        right_layout.addWidget(self.message)
        split = QtWidgets.QSplitter()
        split.addWidget(self.list)
        split.addWidget(right)
        split.setStretchFactor(1, 1)
        layout.addWidget(split, 1)

        session.results_changed.connect(self.show_results)
        if session.results is not None:
            self.show_results(session.results)
        self._update_controls()

    # ------------------------------------------------------------ results
    def open_dialog(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open results", "", "Results (*.npz)")
        if path:
            self.open_file(path)

    def open_file(self, path: str | Path) -> bool:
        try:
            results = load_results(path)
        except (OSError, ValueError) as err:
            QtWidgets.QMessageBox.warning(self, "Open results", f"Cannot open {path}:\n{err}")
            return False
        self.session.set_results(results)
        return True

    def show_results(self, results: Results | None) -> None:
        current = self.current_plot()
        self.results = results
        self.list.clear()
        self.plots = available_plots(results, PLOTS) if results is not None else []
        if results is None:
            self.title.setText("No results.")
        else:
            self.title.setText(f"<b>{results.name}</b>: {summary_text(results)}")
        groups: dict[str, QtWidgets.QTreeWidgetItem] = {}
        select = None
        for spec in self.plots:
            parent = groups.get(spec.group)
            if parent is None:
                parent = QtWidgets.QTreeWidgetItem([spec.group or "Other"])
                parent.setFlags(parent.flags() & ~QtCore.Qt.ItemFlag.ItemIsSelectable)
                self.list.addTopLevelItem(parent)
                parent.setExpanded(True)
                groups[spec.group] = parent
            item = QtWidgets.QTreeWidgetItem([spec.title])
            item.setData(0, QtCore.Qt.ItemDataRole.UserRole, spec.key)
            parent.addChild(item)
            if select is None or (current is not None and spec.key == current.key):
                select = item
        n = len(results["pass"]) if results is not None and "pass" in results else 1
        self.slider.blockSignals(True)
        self.slider.setRange(0, max(n - 1, 0))
        self.slider.setValue(max(n - 1, 0))                   # last recorded pass
        self.slider.blockSignals(False)
        if select is not None:
            self.list.setCurrentItem(select)
        self.redraw()

    def current_plot(self) -> PlotSpec | None:
        item = self.list.currentItem() if hasattr(self, "list") else None
        key = item.data(0, QtCore.Qt.ItemDataRole.UserRole) if item is not None else None
        return next((p for p in self.plots if p.key == key), None)

    def options(self) -> dict[str, Any]:
        return {"index": self.slider.value()}

    def _pass_changed(self) -> None:
        plot = self.current_plot()
        if plot is not None and plot.per_pass:
            self.redraw()
        self._update_controls()

    def _update_controls(self) -> None:
        plot = self.current_plot()
        per_pass = plot is not None and plot.per_pass
        self.slider.setEnabled(per_pass)
        r = self.results
        if r is not None and "pass" in r and len(r["pass"]):
            self.pass_label.setText(f"Pass {int(r['pass'][self.slider.value()])}")
        for w in (self.export_figure_button, self.export_data_button):
            w.setEnabled(plot is not None)

    def redraw(self) -> None:
        self._update_controls()
        fig = self.figure.figure
        fig.clear()
        self.message.clear()
        plot = self.current_plot()
        if plot is None or self.results is None:
            self.figure.draw()
            return
        try:
            plot.draw(self.results, fig, self.options())
        except Exception:  # noqa: BLE001  (a broken figure must not break the viewer)
            fig.clear()
            self.message.setText(f"{plot.title} failed:\n{traceback.format_exc(limit=3)}")
        self.figure.draw()

    # ------------------------------------------------------------ export
    def _default_name(self, suffix: str) -> str:
        plot = self.current_plot()
        stem = self.results.path.stem if self.results is not None and self.results.path else "results"
        folder = self.results.path.parent if self.results is not None and self.results.path else Path.cwd()
        return str(folder / f"{stem}_{plot.key if plot else 'figure'}{suffix}")

    def export_figure_dialog(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export figure", self._default_name(".png"),
                                                        FIGURE_FILTER)
        if path:
            self.export_figure(path)

    def export_figure(self, path: str | Path) -> None:
        self.figure.figure.savefig(path, dpi=200)

    def export_data_dialog(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export data", self._default_name(".npz"), DATA_FILTER)
        if path:
            try:
                self.export_data(path)
            except (OSError, ValueError) as err:
                QtWidgets.QMessageBox.warning(self, "Export data", str(err))

    def export_data(self, path: str | Path) -> None:
        plot = self.current_plot()
        if plot is None or self.results is None:
            return
        data = dict(plot.export_data(self.results, self.options()))
        if str(path).endswith(".csv"):
            write_csv(path, data)
        else:
            arrays: dict[str, Any] = {k: np.asarray(v) for k, v in data.items()}
            np.savez_compressed(path, **arrays)


PANELS.register(PanelSpec("results", "Results", ResultsPanel, order=30,
                          tooltip="Figures of a finished run; export of figures and data"))
