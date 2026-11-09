"""Built-in panels of the main window.

Each module of this package registers its panel in thunder4d.gui.registry.PANELS when imported, and is
imported here (the main window imports this package). A new panel = a new module + one import line.
"""
from . import builder, design, run
