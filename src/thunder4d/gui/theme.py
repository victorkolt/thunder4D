"""Look of the window: the Fusion style, a colour theme (palette + a small style sheet) chosen in View > Theme.

A theme is six colours. Add one to THEMES and it appears in the menu. The choice is remembered (QSettings)
and the matplotlib colours follow it, for the plots drawn afterwards.
"""
from __future__ import annotations

from dataclasses import dataclass

from ._qt import QtCore, QtGui, QtWidgets

DEFAULT = "Nord"


@dataclass(frozen=True)
class Theme:
    window: str     # background of the window
    base: str       # background of inputs, tables and plots
    text: str
    muted: str      # disabled text, hints
    border: str
    accent: str     # selection, focus, default button


THEMES = {
    "Light": Theme("#f4f5f7", "#ffffff", "#1f2328", "#8b929a", "#d0d5db", "#2f6fed"),
    "Dark": Theme("#1e2024", "#14161a", "#e4e6ea", "#7c828b", "#383c43", "#5b9dff"),
    "Nord": Theme("#2e3440", "#272c36", "#e5e9f0", "#7b869b", "#434c5e", "#88c0d0"),
    "Solarized Light": Theme("#eee8d5", "#fdf6e3", "#586e75", "#93a1a1", "#d6cfb8", "#268bd2"),
}

STYLE = """
QMenuBar {{ background: {window}; border: none; }}
QMenuBar::item {{ padding: 4px 10px; background: transparent; }}
QMenuBar::item:selected {{ background: {border}; border-radius: 4px; }}
QMenu {{ background: {base}; border: 1px solid {border}; padding: 4px; }}
QMenu::item {{ padding: 5px 22px; border-radius: 4px; }}
QMenu::item:selected {{ background: {accent}; color: #ffffff; }}
QStatusBar {{ background: {window}; }}
QToolTip {{ background: {base}; color: {text}; border: 1px solid {border}; padding: 4px; }}
QGroupBox {{ border: 1px solid {border}; border-radius: 6px; margin-top: 10px; padding: 8px 6px 6px 6px; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 4px; color: {muted}; }}
QPushButton {{ border: 1px solid {border}; border-radius: 6px; padding: 5px 14px; background: {base}; }}
QPushButton:hover {{ border-color: {accent}; }}
QPushButton:pressed {{ background: {window}; }}
QPushButton:disabled {{ color: {muted}; }}
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QPlainTextEdit, QTextEdit {{
    border: 1px solid {border}; border-radius: 5px; padding: 3px 6px; background: {base}; }}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, QPlainTextEdit:focus {{
    border-color: {accent}; }}
QTabWidget::pane {{ border: none; border-top: 1px solid {border}; }}
QTabBar::tab {{ padding: 7px 16px; border: none; border-bottom: 2px solid transparent; color: {muted}; }}
QTabBar::tab:selected {{ color: {text}; border-bottom-color: {accent}; }}
QTabBar::tab:hover {{ color: {text}; }}
QHeaderView::section {{ background: {window}; border: none; border-bottom: 1px solid {border}; padding: 4px 8px; }}
QTableView {{ gridline-color: {border}; border: 1px solid {border}; border-radius: 5px; }}
QSplitter::handle {{ background: {window}; }}
QScrollBar:vertical {{ width: 10px; background: transparent; }}
QScrollBar:horizontal {{ height: 10px; background: transparent; }}
QScrollBar::handle {{ background: {border}; border-radius: 4px; min-height: 24px; min-width: 24px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
"""


def _app() -> QtWidgets.QApplication:
    app = QtWidgets.QApplication.instance()
    assert isinstance(app, QtWidgets.QApplication), "create the QApplication first"
    return app


def palette(t: Theme) -> QtGui.QPalette:
    p = QtGui.QPalette()
    Role, Group = QtGui.QPalette.ColorRole, QtGui.QPalette.ColorGroup
    for role, color in ((Role.Window, t.window), (Role.WindowText, t.text), (Role.Base, t.base),
                        (Role.AlternateBase, t.window), (Role.Text, t.text), (Role.Button, t.base),
                        (Role.ButtonText, t.text), (Role.ToolTipBase, t.base), (Role.ToolTipText, t.text),
                        (Role.Highlight, t.accent), (Role.HighlightedText, "#ffffff"), (Role.Link, t.accent),
                        (Role.PlaceholderText, t.muted), (Role.Light, t.window), (Role.Midlight, t.window),
                        (Role.Mid, t.window), (Role.Dark, t.window), (Role.Shadow, t.window)):
        p.setColor(role, QtGui.QColor(color))
    for role in (Role.WindowText, Role.Text, Role.ButtonText):
        p.setColor(Group.Disabled, role, QtGui.QColor(t.muted))
    return p


def matplotlib_colors(t: Theme) -> None:
    """Plots drawn from now on use the colours of the theme."""
    import matplotlib as mpl
    mpl.rcParams.update({"figure.facecolor": t.window, "axes.facecolor": t.base, "savefig.facecolor": "white",
                         "axes.edgecolor": t.muted, "axes.labelcolor": t.text, "text.color": t.text,
                         "xtick.color": t.text, "ytick.color": t.text, "grid.color": t.border})


def apply(name: str) -> str:
    """Apply a theme to the running application (unknown name: the default). Returns the name used."""
    name = name if name in THEMES else DEFAULT
    t = THEMES[name]
    app = _app()
    app.setStyle("Fusion")
    app.setPalette(palette(t))
    app.setStyleSheet(STYLE.format(**t.__dict__))
    try:
        from .plotting import FigureWidget

        matplotlib_colors(t)
        FigureWidget.retheme_all()
    except ImportError:
        pass
    return name


TEXT_SIZES = {"Smaller": -1, "Default": 0, "Larger": 2, "Largest": 4}      # points added to the system font
_base_size: list[float] = []


def apply_text_size(delta: int) -> None:
    app = _app()
    if not _base_size:
        _base_size.append(app.font().pointSizeF())
    font = app.font()
    font.setPointSizeF(_base_size[0] + delta)
    app.setFont(font)


def saved_text_size() -> int:
    try:
        return int(str(QtCore.QSettings("thunder4d", "gui").value("text_size", 0)))
    except (TypeError, ValueError):
        return 0


def save_text_size(delta: int) -> None:
    QtCore.QSettings("thunder4d", "gui").setValue("text_size", delta)


def saved() -> str:
    return str(QtCore.QSettings("thunder4d", "gui").value("theme", DEFAULT))


def save(name: str) -> None:
    QtCore.QSettings("thunder4d", "gui").setValue("theme", name)
