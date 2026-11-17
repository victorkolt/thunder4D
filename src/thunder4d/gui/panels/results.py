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

from .. import (
    plots as _builtin_plots,  # noqa: F401  (registers the built-in figures)
    prefs,
)
from .._qt import QtCore, QtWidgets
from ..fields import ERROR_STYLE
from ..plotting import FigureWidget
from ..registry import PANELS, PLOTS, PanelSpec, PlotSpec, available_plots
from ..results import Results, load_results
from ..session import Session

FIGURE_FILTER = "PNG (*.png);;PDF (*.pdf);;SVG (*.svg)"
LIMITS = ("xmin", "xmax", "ymin", "ymax")
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
        self.axes: list[Any] = []                              # the axes of the figure shown (no colour bars)
        self.auto: list[dict[str, Any]] = []                   # their automatic limits and scales
        self.view: dict[tuple[str, int], dict[str, Any]] = {}  # what the user set: (plot, axes) -> limits, log

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
        self.copy_figure_button = QtWidgets.QPushButton("Copy figure")
        self.copy_figure_button.setToolTip("Copy the figure to the clipboard (paper colours)")
        self.copy_figure_button.clicked.connect(self.figure_copy)
        self.export_data_button = QtWidgets.QPushButton("Export data…")
        self.export_data_button.clicked.connect(self.export_data_dialog)
        bar.addWidget(self.copy_figure_button)
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
        right_layout.addLayout(self._axes_row())
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
        path = prefs.open_file(self, "Open results", "Results (*.npz)")
        if path:
            self.open_file(path)

    def open_file(self, path: str | Path) -> bool:
        try:
            results = load_results(path)
        except (OSError, ValueError) as err:
            QtWidgets.QMessageBox.warning(self, "Open results", f"Cannot open {path}:\n{err}")
            return False
        self.session.set_results(results)
        prefs.add_recent("results", path)
        return True

    def show_results(self, results: Results | None) -> None:
        current = self.current_plot()
        self.view.clear()                                  # limits set by the user are for the previous results
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

    # ------------------------------------------------------------ axis limits and log scales
    def _axes_row(self) -> QtWidgets.QHBoxLayout:
        row = QtWidgets.QHBoxLayout()
        self.axis_box = QtWidgets.QComboBox()
        self.axis_box.setToolTip("Axes of the figure to set")
        self.axis_box.currentIndexChanged.connect(lambda _: self._show_axis())
        row.addWidget(QtWidgets.QLabel("Axes"))
        row.addWidget(self.axis_box, 1)
        self.limit_edits: dict[str, QtWidgets.QLineEdit] = {}
        for name in LIMITS:
            edit = QtWidgets.QLineEdit()
            edit.setMaximumWidth(80)
            edit.setToolTip(f"{name[0]} {'min' if name.endswith('min') else 'max'}: empty = automatic")
            edit.editingFinished.connect(self._limits_edited)
            row.addWidget(QtWidgets.QLabel(f"{name[0]} {name[1:]}"))
            row.addWidget(edit)
            self.limit_edits[name] = edit
        self.log_boxes = {"x": QtWidgets.QCheckBox("log x"), "y": QtWidgets.QCheckBox("log y")}
        for box in self.log_boxes.values():
            box.toggled.connect(self._scale_edited)
            row.addWidget(box)
        self.db_box = QtWidgets.QCheckBox("dB")
        self.db_box.setChecked(True)
        self.db_box.setToolTip("Colour maps of spectra in dB or linear (normalised to the maximum)")
        self.db_box.toggled.connect(lambda _: self.redraw())
        row.addWidget(self.db_box)
        self.auto_button = QtWidgets.QPushButton("Auto")
        self.auto_button.setToolTip("Back to the automatic limits and scales of these axes")
        self.auto_button.clicked.connect(self._reset_axis)
        row.addWidget(self.auto_button)
        return row

    def _sync_axes(self) -> None:
        """After a figure was drawn: list its axes, keep their automatic values, apply what the user set."""
        plot = self.current_plot()
        self.axes = [] if plot is None else [a for a in self.figure.figure.axes if a.get_label() != "<colorbar>"]
        self.auto = [{"x": a.get_xlim(), "y": a.get_ylim(), "xlog": a.get_xscale() == "log",
                      "ylog": a.get_yscale() == "log"} for a in self.axes]
        for i, ax in enumerate(self.axes):
            ov = self.view.get((plot.key, i), {}) if plot is not None else {}
            for axis in "xy":
                if ov.get(f"{axis}log") is not None:
                    getattr(ax, f"set_{axis}scale")("log" if ov[f"{axis}log"] else "linear")
                lo, hi = ov.get(f"{axis}min"), ov.get(f"{axis}max")
                if lo is not None or hi is not None:
                    try:
                        getattr(ax, f"set_{axis}lim")(lo, hi)
                    except ValueError:
                        pass                                    # e.g. a limit <= 0 on a log axis
        index = max(self.axis_box.currentIndex(), 0)
        self.axis_box.blockSignals(True)
        self.axis_box.clear()
        for i, ax in enumerate(self.axes):
            self.axis_box.addItem(f"{i + 1}: {(ax.get_title() or ax.get_ylabel() or 'axes')[:40]}")
        self.axis_box.setCurrentIndex(min(index, len(self.axes) - 1))
        self.axis_box.blockSignals(False)
        self._show_axis()

    def _show_axis(self) -> None:
        """Fill the fields with what is set for the selected axes: the automatic values show as hints."""
        plot, i = self.current_plot(), self.axis_box.currentIndex()
        on = plot is not None and 0 <= i < len(self.axes)
        ov = self.view.get((plot.key, i), {}) if on else {}
        auto = self.auto[i] if on else {}
        for name, edit in self.limit_edits.items():
            edit.blockSignals(True)
            edit.setText("" if ov.get(name) is None else f"{ov[name]:g}")
            edit.setPlaceholderText(f"{auto[name[0]][name.endswith('max')]:.4g}" if on else "")
            edit.setStyleSheet("")
            edit.setEnabled(on)
            edit.blockSignals(False)
        for axis, box in self.log_boxes.items():
            box.blockSignals(True)
            box.setChecked(bool(ov.get(f"{axis}log") if ov.get(f"{axis}log") is not None else auto.get(f"{axis}log")))
            box.setEnabled(on)
            box.blockSignals(False)
        self.auto_button.setEnabled(on)

    def _current_view(self) -> dict[str, Any]:
        plot = self.current_plot()
        assert plot is not None
        return self.view.setdefault((plot.key, self.axis_box.currentIndex()), {})

    def _tidy_view(self) -> None:
        for key in [k for k, ov in self.view.items() if all(v is None for v in ov.values())]:
            del self.view[key]

    def _limits_edited(self) -> None:
        if self.current_plot() is None or not 0 <= self.axis_box.currentIndex() < len(self.axes):
            return
        values: dict[str, float | None] = {}
        for name, edit in self.limit_edits.items():
            text = edit.text().strip()
            try:
                values[name] = float(text) if text else None
            except ValueError:
                edit.setStyleSheet(ERROR_STYLE)
                return
            edit.setStyleSheet("")
        if self._current_view() == {**self._current_view(), **values}:
            return                                                # nothing changed
        self._current_view().update(values)
        self._tidy_view()
        self.redraw()

    def _scale_edited(self) -> None:
        plot, i = self.current_plot(), self.axis_box.currentIndex()
        if plot is None or not 0 <= i < len(self.axes):
            return
        ov = self._current_view()
        for axis, box in self.log_boxes.items():
            ov[f"{axis}log"] = box.isChecked() if box.isChecked() != self.auto[i][f"{axis}log"] else None
        self._tidy_view()
        self.redraw()

    def _reset_axis(self) -> None:
        plot = self.current_plot()
        if plot is not None:
            self.view.pop((plot.key, self.axis_box.currentIndex()), None)
            self.redraw()

    def current_plot(self) -> PlotSpec | None:
        item = self.list.currentItem() if hasattr(self, "list") else None
        key = item.data(0, QtCore.Qt.ItemDataRole.UserRole) if item is not None else None
        return next((p for p in self.plots if p.key == key), None)

    def options(self) -> dict[str, Any]:
        return {"index": self.slider.value(), "db": self.db_box.isChecked()}

    def _pass_changed(self) -> None:
        plot = self.current_plot()
        if plot is not None and plot.per_pass:
            self.redraw()
        self._update_controls()

    def _update_controls(self) -> None:
        plot = self.current_plot()
        per_pass = plot is not None and plot.per_pass
        self.slider.setEnabled(per_pass)
        self.db_box.setEnabled(plot is not None and plot.db)
        r = self.results
        if r is not None and "pass" in r and len(r["pass"]):
            self.pass_label.setText(f"Pass {int(r['pass'][self.slider.value()])}")
        for w in (self.copy_figure_button, self.export_figure_button, self.export_data_button):
            w.setEnabled(plot is not None)

    def redraw(self) -> None:
        self._update_controls()
        fig = self.figure.figure
        fig.clear()
        self.message.clear()
        plot = self.current_plot()
        if plot is None or self.results is None:
            self._sync_axes()
            self.figure.draw()
            return
        try:
            plot.draw(self.results, fig, self.options())
        except Exception:  # noqa: BLE001  (a broken figure must not break the viewer)
            fig.clear()
            self.message.setText(f"{plot.title} failed:\n{traceback.format_exc(limit=3)}")
        self._sync_axes()
        self.figure.draw()

    # ------------------------------------------------------------ export
    def _default_name(self, suffix: str) -> str:
        plot = self.current_plot()
        stem = self.results.path.stem if self.results is not None and self.results.path else "results"
        folder = self.results.path.parent if self.results is not None and self.results.path else Path.cwd()
        return str(folder / f"{stem}_{plot.key if plot else 'figure'}{suffix}")

    def export_figure_dialog(self) -> None:
        path = prefs.save_file(self, "Export figure", self._default_name(".png"), FIGURE_FILTER)
        if path:
            self.export_figure(path)

    def figure_copy(self) -> None:
        self.figure.copy_to_clipboard()
        self.title.setText("Figure copied to the clipboard.")

    def export_figure(self, path: str | Path) -> None:
        self.figure.save(path, dpi=200)

    def export_data_dialog(self) -> None:
        path = prefs.save_file(self, "Export data", self._default_name(".npz"), DATA_FILTER)
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
