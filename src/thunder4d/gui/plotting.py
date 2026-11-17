"""matplotlib figures inside the window (the same plots as the notebooks)."""
from __future__ import annotations

import io
import weakref
from pathlib import Path
from typing import Any, ClassVar

from ._qt import QtGui, QtWidgets

PAPER = {"figure": "white", "axes": "white", "text": "#1f2328", "edge": "#555555", "grid": "#cccccc"}   # exports


class FigureWidget(QtWidgets.QWidget):
    """A matplotlib Figure with the zoom / pan / save toolbar. Draw on self.figure, then call draw()."""

    instances: ClassVar[weakref.WeakSet[FigureWidget]] = weakref.WeakSet()

    def __init__(self, parent: QtWidgets.QWidget | None = None, toolbar: bool = True):
        super().__init__(parent)
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
        from matplotlib.figure import Figure

        self.figure = Figure(layout="constrained")
        self.canvas = FigureCanvasQTAgg(self.figure)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        if toolbar:
            layout.addWidget(NavigationToolbar2QT(self.canvas, self))
        layout.addWidget(self.canvas, 1)
        FigureWidget.instances.add(self)

    def clear(self) -> None:
        self.figure.clear()
        self.retheme()

    def draw(self) -> None:
        self.canvas.draw_idle()

    # ------------------------------------------------------------ colours
    def retheme(self, colors: dict[str, str] | None = None) -> None:
        """Give the figure the colours of the theme (matplotlib's rcParams), or of `colors` (see PAPER)."""
        import matplotlib as mpl

        rc = mpl.rcParams
        c = colors or {"figure": rc["figure.facecolor"], "axes": rc["axes.facecolor"], "text": rc["text.color"],
                       "edge": rc["axes.edgecolor"], "grid": rc["grid.color"]}
        fig = self.figure
        fig.set_facecolor(c["figure"])
        for ax in fig.axes:
            ax.set_facecolor(c["axes"])
            for spine in ax.spines.values():
                spine.set_edgecolor(c["edge"])
            ax.tick_params(which="both", color=c["edge"], labelcolor=c["text"])
            for label in (ax.xaxis.label, ax.yaxis.label, ax.title, ax._left_title, ax._right_title):
                label.set_color(c["text"])
            legend = ax.get_legend()
            if legend is not None:
                legend.get_frame().set(facecolor=c["axes"], edgecolor=c["grid"])
                for text in legend.get_texts():
                    text.set_color(c["text"])
        self.canvas.draw_idle()

    @classmethod
    def retheme_all(cls) -> None:
        for widget in list(cls.instances):
            widget.retheme()

    # ------------------------------------------------------------ output (always on paper colours)
    def save(self, path: str | Path | io.BytesIO, dpi: int = 200, **kwargs: Any) -> None:
        self.retheme(PAPER)
        try:
            self.figure.savefig(path, dpi=dpi, **kwargs)
        finally:
            self.retheme()

    def copy_to_clipboard(self) -> None:
        buffer = io.BytesIO()
        self.save(buffer, dpi=150, format="png")
        image = QtGui.QImage.fromData(buffer.getvalue(), "PNG")
        QtWidgets.QApplication.clipboard().setImage(image)
