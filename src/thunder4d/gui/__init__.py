"""Graphical interface of thunder4d (optional:  uv add "thunder4d[gui]").

    thunder4d-gui [input.json]          # or:  python -m thunder4d.gui [input.json]

Importing this package does not import Qt: the extension points (thunder4d.gui.registry) can be filled
from any module. See registry.py to add a panel, a results plot or a live metric.
"""
from __future__ import annotations

import sys

from .registry import METRICS, PANELS, PLOTS, MetricSpec, PanelSpec, PlotSpec, Registry, available_plots


def main(argv: list[str] | None = None) -> int:
    """Entry point of the thunder4d-gui command."""
    try:
        from . import _qt  # PySide6 installed?
    except ImportError as err:
        print(f"thunder4d-gui: {err}", file=sys.stderr)
        return 1
    from .app import run

    return run(argv)


__all__ = ["METRICS", "PANELS", "PLOTS", "MetricSpec", "PanelSpec", "PlotSpec", "Registry", "available_plots",
           "main"]
