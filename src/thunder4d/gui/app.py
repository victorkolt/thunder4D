"""Main window: the tabs of the registered panels, sharing one Session."""
from __future__ import annotations

import argparse
import sys
import traceback
from importlib.metadata import version

from . import panels as _builtin_panels  # noqa: F401  (registers the built-in panels)
from ._qt import QtCore, QtGui, QtWidgets
from .registry import PANELS, PanelSpec, Registry
from .session import Session

TITLE = "THUNDER4D"


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, session: Session | None = None, panels: Registry[PanelSpec] = PANELS):
        super().__init__()
        self.session = session or Session(self)
        self.setWindowTitle(TITLE)
        self.resize(1200, 800)
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setDocumentMode(True)
        self.setCentralWidget(self.tabs)
        self.panels: dict[str, QtWidgets.QWidget] = {}
        for spec in panels:
            self._add_panel(spec)
        if not self.panels:
            self.tabs.addTab(self._message("No panel registered yet."), "Welcome")
        self.session.panel_requested.connect(self.show_panel)
        self._menus()
        self.statusBar().showMessage(f"thunder4d {version('thunder4d')}")

    def show_panel(self, key: str) -> None:
        if key in self.panels:
            self.tabs.setCurrentWidget(self.panels[key])

    def _add_panel(self, spec: PanelSpec) -> None:
        try:
            widget = spec.factory(self.session)
        except Exception:  # noqa: BLE001  (a broken panel must not take the whole window down)
            widget = self._message(f"The panel {spec.key!r} failed to load:\n\n{traceback.format_exc()}")
        index = self.tabs.addTab(widget, spec.title)
        if spec.tooltip:
            self.tabs.setTabToolTip(index, spec.tooltip)
        self.panels[spec.key] = widget

    @staticmethod
    def _message(text: str) -> QtWidgets.QWidget:
        label = QtWidgets.QLabel(text)
        label.setWordWrap(True)
        label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        return label

    def _menus(self) -> None:
        file_menu = self.menuBar().addMenu("&File")
        quit_action = file_menu.addAction("&Quit")
        quit_action.setShortcut(QtGui.QKeySequence.StandardKey.Quit)
        quit_action.triggered.connect(self.close)
        help_menu = self.menuBar().addMenu("&Help")
        help_menu.addAction("&About").triggered.connect(self._about)

    def _about(self) -> None:
        QtWidgets.QMessageBox.about(
            self, f"About {TITLE}",
            f"<b>thunder4d {version('thunder4d')}</b><br>4D UPPE solver for multipass cells.<br>"
            '<a href="https://victorkolt.github.io/thunder4D/">Documentation</a>')


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="thunder4d-gui", description="Graphical interface of thunder4d.")
    parser.add_argument("input", nargs="?", help="standard THUNDER4D input file (JSON) to open")
    parser.add_argument("--version", action="version", version=f"thunder4d {version('thunder4d')}")
    return parser.parse_args(argv)


def build_window(args: argparse.Namespace) -> MainWindow:
    """The main window, with the input file of the command line (if any) loaded in its session."""
    from ..inputfile import load_input

    window = MainWindow()
    if args.input:
        window.session.set_config(load_input(args.input), args.input)
        window.statusBar().showMessage(f"Opened {args.input}")
    return window


def run(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
    app.setApplicationName(TITLE)
    try:
        window = build_window(args)
    except (OSError, ValueError, KeyError) as err:     # unreadable or invalid input file
        print(f"thunder4d-gui: cannot open {args.input}: {err}", file=sys.stderr)
        return 2
    window.show()
    return app.exec()
