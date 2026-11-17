"""What the window remembers between sessions (QSettings): last folder, recent files, active tab.

Also the file dialogs that start in the last folder, and the log view with its Save and Clear buttons.
"""
from __future__ import annotations

from pathlib import Path

from ._qt import QtCore, QtWidgets

MAX_RECENT = 8


def settings() -> QtCore.QSettings:
    return QtCore.QSettings("thunder4d", "gui")


# ------------------------------------------------------------ file dialogs
def last_dir() -> str:
    folder = str(settings().value("last_dir", ""))
    return folder if Path(folder).is_dir() else str(Path.cwd())


def _remember(path: str) -> None:
    settings().setValue("last_dir", str(Path(path).resolve().parent))


def open_file(parent: QtWidgets.QWidget, title: str, file_filter: str) -> str:
    path, _ = QtWidgets.QFileDialog.getOpenFileName(parent, title, last_dir(), file_filter)
    if path:
        _remember(path)
    return path


def save_file(parent: QtWidgets.QWidget, title: str, name: str, file_filter: str) -> str:
    """`name`: a full path, or a file name proposed in the last folder."""
    start = name if Path(name).is_absolute() else str(Path(last_dir()) / name)
    path, _ = QtWidgets.QFileDialog.getSaveFileName(parent, title, start, file_filter)
    if path:
        _remember(path)
    return path


# ------------------------------------------------------------ recent files
def recent() -> list[tuple[str, str]]:
    """[(kind, path)], newest first, files that still exist. kind: input, problem or results."""
    raw = settings().value("recent", [])
    raw = [raw] if isinstance(raw, str) else list(raw or [])
    pairs = [tuple(item.split("\t", 1)) for item in raw if "\t" in item]
    return [(kind, path) for kind, path in pairs if Path(path).is_file()]


def add_recent(kind: str, path: str | Path) -> None:
    entry = (kind, str(Path(path).resolve()))
    items = [entry] + [e for e in recent() if e != entry]
    settings().setValue("recent", [f"{k}\t{p}" for k, p in items[:MAX_RECENT]])


# ------------------------------------------------------------ active tab
def save_layout(window: QtWidgets.QMainWindow) -> None:
    s = settings()
    s.setValue("tab", window.tabs.currentIndex())


def restore_layout(window: QtWidgets.QMainWindow) -> None:
    s = settings()
    try:
        window.tabs.setCurrentIndex(int(s.value("tab", 0)))
    except (TypeError, ValueError):
        pass


# ------------------------------------------------------------ log view
def log_box(log: QtWidgets.QPlainTextEdit) -> QtWidgets.QWidget:
    """`log` with a Save and a Clear button above it. Use the returned widget in the layout."""
    box = QtWidgets.QWidget()
    layout = QtWidgets.QVBoxLayout(box)
    layout.setContentsMargins(0, 4, 0, 0)
    bar = QtWidgets.QHBoxLayout()
    save, clear = QtWidgets.QPushButton("Save log…"), QtWidgets.QPushButton("Clear")

    def save_log() -> None:
        path = save_file(box, "Save the log", "thunder4d.log", "Text (*.txt *.log)")
        if path:
            Path(path).write_text(log.toPlainText() + "\n")

    save.clicked.connect(save_log)
    clear.clicked.connect(log.clear)
    bar.addStretch(1)
    bar.addWidget(save)
    bar.addWidget(clear)
    layout.addLayout(bar)
    layout.addWidget(log, 1)
    return box
