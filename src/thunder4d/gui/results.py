"""Results of a run as the GUI sees them: the arrays written by MPCSimulation.save, by key."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


class Results(dict):
    """{key: array} of a saved run (.npz), plus the file it came from (`path`)."""

    def __init__(self, data: dict[str, Any], path: str | Path | None = None):
        super().__init__(data)
        self.path = Path(path) if path is not None else None

    @property
    def name(self) -> str:
        return self.path.name if self.path else "results"


def load_results(path: str | Path) -> Results:
    with np.load(path, allow_pickle=False) as f:
        return Results({k: f[k] for k in f.files}, path)
