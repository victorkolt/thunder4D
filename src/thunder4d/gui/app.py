"""Main window: the tabs of the registered panels, sharing one Session."""
from __future__ import annotations

import argparse
import sys
import traceback
from importlib.metadata import version

from . import (
    panels as _builtin_panels,  # noqa: F401  (registers the built-in panels)
    theme,
)
from ._qt import QtCore, QtGui, QtWidgets
from .registry import PANELS, PanelSpec, Registry
from .session import Session

TITLE = "THUNDER4D"


class NoWheelUnlessFocused(QtCore.QObject):
    """Scrolling over a combo box or a spin box scrolls the form, unless the box was clicked first."""

    def eventFilter(self, obj: QtCore.QObject, event: QtCore.QEvent) -> bool:
        boxes = (QtWidgets.QComboBox, QtWidgets.QAbstractSpinBox)
        if event.type() == QtCore.QEvent.Type.Wheel and isinstance(obj, boxes) and not obj.hasFocus():
            area = obj.parentWidget()
            while area is not None and not isinstance(area, QtWidgets.QAbstractScrollArea):
                area = area.parentWidget()
            if area is not None:
                QtWidgets.QApplication.sendEvent(area.viewport(), event)
            return True
        return False


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
        for i in range(min(self.tabs.count(), 9)):
            QtGui.QShortcut(QtGui.QKeySequence(f"Ctrl+{i + 1}"), self, lambda i=i: self.tabs.setCurrentIndex(i))
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

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        """Panels with a can_close() method (unsaved input, running computation) may refuse."""
        for widget in self.panels.values():
            can_close = getattr(widget, "can_close", None)
            if can_close is not None and not can_close():
                event.ignore()
                return
        event.accept()

    def _menus(self) -> None:
        file_menu = self.menuBar().addMenu("&File")
        quit_action = file_menu.addAction("&Quit")
        quit_action.setShortcut(QtGui.QKeySequence.StandardKey.Quit)
        quit_action.triggered.connect(self.close)
        view_menu = self.menuBar().addMenu("&View")
        size_menu = view_menu.addMenu("Text &size")
        size_group = QtGui.QActionGroup(self)
        for label, delta in theme.TEXT_SIZES.items():
            action = size_menu.addAction(label)
            action.setCheckable(True)
            action.setChecked(delta == theme.saved_text_size())
            action.triggered.connect(lambda _=False, d=delta: self.set_text_size(d))
            size_group.addAction(action)
        theme_menu = view_menu.addMenu("&Theme")
        group = QtGui.QActionGroup(self)
        current = theme.saved()
        for name in theme.THEMES:
            action = theme_menu.addAction(name)
            action.setCheckable(True)
            action.setChecked(name == current)
            action.triggered.connect(lambda _=False, n=name: self.set_theme(n))
            group.addAction(action)
        help_menu = self.menuBar().addMenu("&Help")
        help_menu.addAction("&About").triggered.connect(self._about)

    def set_text_size(self, delta: int) -> None:
        theme.apply_text_size(delta)
        theme.save_text_size(delta)

    def set_theme(self, name: str) -> None:
        theme.save(theme.apply(name))
        self.statusBar().showMessage(f"Theme: {name}", 5000)

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
    theme.apply(theme.saved())
    theme.apply_text_size(theme.saved_text_size())
    wheel_filter = NoWheelUnlessFocused(app)
    app.installEventFilter(wheel_filter)
    try:
        window = build_window(args)
    except (OSError, ValueError, KeyError) as err:     # unreadable or invalid input file
        print(f"thunder4d-gui: cannot open {args.input}: {err}", file=sys.stderr)
        return 2
    window.show()
    return app.exec()
