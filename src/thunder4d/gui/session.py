"""State shared between the panels.

Panels never talk to each other directly: the design helper sets the input, the builder shows it, the run
panel sets the results, the viewer shows them. Each change is signalled, so a new panel only has to
listen to the session.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ._qt import QtCore


class Session(QtCore.QObject):
    config_changed = QtCore.Signal(object)      # the input (dict of the standard input file), or None
    results_changed = QtCore.Signal(object)     # the results (mapping of arrays, by key), or None
    panel_requested = QtCore.Signal(str)        # a panel asks the window to show another one (registry key)

    def __init__(self, parent: QtCore.QObject | None = None):
        super().__init__(parent)
        self.config: dict[str, Any] | None = None
        self.config_path: Path | None = None
        self.results: Any = None

    def set_config(self, cfg: dict[str, Any] | None, path: str | Path | None = None) -> None:
        self.config = cfg
        self.config_path = Path(path) if path is not None else None
        self.config_changed.emit(cfg)

    def show_panel(self, key: str) -> None:
        self.panel_requested.emit(key)

    def set_results(self, results: Any) -> None:
        self.results = results
        self.results_changed.emit(results)
