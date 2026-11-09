"""Long computations (design search, simulations) in a background thread, so the window stays responsive.

    task = Task(work)            # work(task) -> result; may call task.check() to honour Stop
    task.output.connect(log.appendPlainText)     # what work() prints, line by line
    task.progress.connect(...)                   # task.report(anything) from work()
    task.succeeded.connect(show_result); task.failed.connect(show_error)
    task.start();  task.cancel()

Signals are queued to the GUI thread, so slots may touch widgets.
"""
from __future__ import annotations

import sys
import threading
import traceback
from collections.abc import Callable
from typing import Any

from ._qt import QtCore


class Cancelled(Exception):
    """Raised by Task.check() once the task was asked to stop."""


class _Lines:
    """File-like object that emits complete lines (stdout of the task)."""

    def __init__(self, emit: Callable[[str], None]):
        self.emit, self.buffer = emit, ""

    def write(self, text: str) -> int:
        self.buffer += text
        *lines, self.buffer = self.buffer.split("\n")
        for line in lines:
            self.emit(line)
        return len(text)

    def flush(self) -> None:
        if self.buffer:
            self.emit(self.buffer)
            self.buffer = ""


class _ThreadStdout:
    """sys.stdout that sends what a task's thread prints to that task; other threads print as before.
    Installed once: tasks running at the same time each keep their own output."""

    def __init__(self, fallback: Any):
        self.fallback = fallback
        self.targets: dict[int, _Lines] = {}

    def _target(self) -> Any:
        return self.targets.get(threading.get_ident(), self.fallback)

    def write(self, text: str) -> int:
        return self._target().write(text)

    def flush(self) -> None:
        self._target().flush()

    def __getattr__(self, name: str) -> Any:        # encoding, isatty, ... of the real stdout
        return getattr(self.fallback, name)


def _thread_stdout() -> _ThreadStdout:
    if not isinstance(sys.stdout, _ThreadStdout):
        sys.stdout = _ThreadStdout(sys.stdout)
    return sys.stdout


class Task(QtCore.QThread):
    output = QtCore.Signal(str)          # one printed line
    progress = QtCore.Signal(object)     # whatever work() passes to report()
    succeeded = QtCore.Signal(object)    # the result of work()
    failed = QtCore.Signal(str)          # traceback of the error
    cancelled = QtCore.Signal()

    def __init__(self, work: Callable[[Task], Any], parent: QtCore.QObject | None = None):
        super().__init__(parent)
        self.work = work
        self._stop = threading.Event()

    def cancel(self) -> None:
        self._stop.set()

    @property
    def stop_requested(self) -> bool:
        return self._stop.is_set()

    def check(self) -> None:
        """Call from work() at safe points: raises Cancelled once cancel() was called."""
        if self._stop.is_set():
            raise Cancelled

    def report(self, value: Any) -> None:
        self.progress.emit(value)

    def run(self) -> None:
        lines = _Lines(self.output.emit)
        stdout = _thread_stdout()
        stdout.targets[threading.get_ident()] = lines
        try:
            outcome: tuple[Any, ...] = (self.succeeded, self.work(self))
        except Cancelled:
            outcome = (self.cancelled,)
        except Exception:  # noqa: BLE001  (shown to the user, the window keeps running)
            outcome = (self.failed, traceback.format_exc())
        finally:
            lines.flush()                      # the last printed lines come before the result
            del stdout.targets[threading.get_ident()]
        signal, *args = outcome
        signal.emit(*args)
