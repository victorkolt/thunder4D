"""PySide6 is optional: the GUI is a convenience, not a dependency of the solver."""

INSTALL_HINT = ('the graphical interface needs PySide6 and matplotlib: install them with  uv add "thunder4d[gui]"  '
                '(or  pip install "thunder4d[gui]")')

try:
    from PySide6 import QtCore, QtGui, QtWidgets
except ImportError as err:
    raise ImportError(INSTALL_HINT) from err

__all__ = ["INSTALL_HINT", "QtCore", "QtGui", "QtWidgets"]
