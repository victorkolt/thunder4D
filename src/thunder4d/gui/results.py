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
        self.cache: dict[str, Any] = {}          # derived quantities (grid, compressed pulse, ...)

    def grid(self) -> Any:
        """The simulation grid, rebuilt on the CPU from the saved sizes and steps."""
        if "grid" not in self.cache:
            from ..grid import Grid

            nx, ny, nt, dx, dy, dt = (float(v) for v in self["grid"])
            dtype = self["Aw_final"].dtype if "Aw_final" in self else np.complex64
            self.cache["grid"] = Grid(int(nx), dx, int(nt), float(self["lambda0"]), dt=dt, Ny=int(ny), dy=dy,
                                      dtype=dtype, device="cpu")
        return self.cache["grid"]

    def passes(self) -> list[int]:
        return [int(p) for p in self.get("pass", [])]

    @property
    def name(self) -> str:
        return self.path.name if self.path else "results"


def load_results(path: str | Path) -> Results:
    with np.load(path, allow_pickle=False) as f:
        return Results({k: f[k] for k in f.files}, path)
