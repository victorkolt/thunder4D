"""A splitter whose first pane keeps the same share of the width on any screen."""
from __future__ import annotations

from ._qt import QtCore, QtWidgets


class RatioSplitter(QtWidgets.QSplitter):
    """The first pane gets `ratio` of the width (never less than its minimum width), whatever the size of the
    window or the screen, until the user drags the handle: then the split is theirs."""

    def __init__(self, ratio: float = 0.4, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self.ratio = ratio
        self.dragged = False
        self.splitterMoved.connect(lambda *_: setattr(self, "dragged", True))

    def resizeEvent(self, event: object) -> None:
        super().resizeEvent(event)
        if not self.dragged and self.orientation() == QtCore.Qt.Orientation.Horizontal and self.count() == 2:
            total = self.width()
            first = min(max(int(total * self.ratio), self.widget(0).minimumWidth()), total)
            self.setSizes([first, total - first])
