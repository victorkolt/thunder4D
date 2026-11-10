"""matplotlib figures inside the window (the same plots as the notebooks)."""
from __future__ import annotations

from ._qt import QtWidgets


class FigureWidget(QtWidgets.QWidget):
    """A matplotlib Figure with the zoom / pan / save toolbar. Draw on self.figure, then call draw()."""

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

    def clear(self) -> None:
        self.figure.clear()

    def draw(self) -> None:
        self.canvas.draw_idle()
