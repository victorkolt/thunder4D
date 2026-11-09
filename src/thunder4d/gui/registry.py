"""Extension points of the GUI.

A feature reaches the GUI by registering entries here, never by editing the main window:

    from thunder4d.gui.registry import PLOTS, PlotSpec
    PLOTS.register(PlotSpec("stokes", "Stokes parameters", draw_stokes, requires=("S0", "S1", "S2", "S3")))

- PANELS  : tabs of the main window (simulation builder, design helper, run, results, batch scans, ...)
- PLOTS   : figures of the results viewer, offered only when the results hold the keys they need
- METRICS : quantities followed live during a run (one value per recorded pass, read from sim.history)

The fields of the input file are described next to its schema, in thunder4d.inputfile.

This module does not import Qt, so any module can register entries without PySide6 installed.
Entries are listed by `order`, then in registration order.
"""
from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Generic, Protocol, TypeVar


class _Spec(Protocol):
    @property
    def key(self) -> str: ...

    @property
    def order(self) -> float: ...


S = TypeVar("S", bound=_Spec)


class Registry(Generic[S]):
    """Ordered collection of specs, indexed by their key."""

    def __init__(self, kind: str):
        self.kind = kind
        self._items: dict[str, S] = {}

    def register(self, spec: S, *, replace: bool = False) -> S:
        """Add `spec`; a key already registered is an error unless replace=True."""
        if spec.key in self._items and not replace:
            raise ValueError(f"{self.kind} {spec.key!r} is already registered (pass replace=True to override it)")
        self._items[spec.key] = spec
        return spec

    def unregister(self, key: str) -> S:
        return self._items.pop(key)

    def __getitem__(self, key: str) -> S:
        return self._items[key]

    def __contains__(self, key: object) -> bool:
        return key in self._items

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[S]:
        return iter(sorted(self._items.values(), key=lambda s: s.order))   # stable: ties keep registration order

    def keys(self) -> list[str]:
        return [s.key for s in self]

    def __repr__(self) -> str:
        return f"Registry({self.kind!r}, {self.keys()})"


@dataclass(frozen=True)
class PanelSpec:
    """A tab of the main window. factory(session) -> QWidget; the session (gui.session.Session) holds the
    state shared between panels (current input, current results) and signals its changes."""
    key: str
    title: str
    factory: Callable[[Any], Any]
    order: float = 100.0
    tooltip: str = ""


@dataclass(frozen=True)
class PlotSpec:
    """A figure of the results viewer. draw(results, fig) fills a matplotlib Figure from the results
    (the arrays of MPCSimulation.save, by key). Offered only when every key of `requires` is present,
    so a plot of a new output appears by itself for the runs that have it."""
    key: str
    title: str
    draw: Callable[[Mapping[str, Any], Any], None]
    requires: tuple[str, ...] = ()
    order: float = 100.0
    group: str = ""

    def available(self, results: Mapping[str, Any]) -> bool:
        return all(k in results for k in self.requires)


@dataclass(frozen=True)
class MetricSpec:
    """A quantity followed live during a run: the last value of history[key], times `scale`, in `unit`.
    warn_above: the monitor flags values beyond it (e.g. a damage threshold)."""
    key: str
    label: str
    unit: str = ""
    scale: float = 1.0
    order: float = 100.0
    fmt: str = "{:.4g}"
    warn_above: float | None = None

    def value(self, history: Mapping[str, Sequence[Any]]) -> float | None:
        values = history.get(self.key)
        if not values or values[-1] is None:
            return None
        return float(values[-1]) * self.scale

    def warning(self, value: float | None) -> bool:
        return value is not None and self.warn_above is not None and value > self.warn_above

    def text(self, history: Mapping[str, Sequence[Any]]) -> str:
        v = self.value(history)
        if v is None:
            return "–"
        return self.fmt.format(v) + (f" {self.unit}" if self.unit else "")


PANELS: Registry[PanelSpec] = Registry("panel")
PLOTS: Registry[PlotSpec] = Registry("plot")
METRICS: Registry[MetricSpec] = Registry("metric")


def available_plots(results: Mapping[str, Any], plots: Registry[PlotSpec] = PLOTS) -> list[PlotSpec]:
    """The registered plots that the results can draw, in order."""
    return [p for p in plots if p.available(results)]
