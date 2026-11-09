"""Long computations (design search, simulations) in a background thread, so the window stays responsive.

    task = Task(work)            # work(task) -> result; may call task.check() to honour Stop
    task.output.connect(log.appendPlainText)     # what work() prints, line by line
    task.progress.connect(...)                   # task.report(anything) from work()
    task.succeeded.connect(show_result); task.failed.connect(show_error)
    task.start();  task.cancel()

Signals are queued to the GUI thread, so slots may touch widgets.
"""
from __future__ import annotations

import contextlib
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
        # stdout is redirected for the whole process while the task runs (prints are rare in the GUI thread)
        lines = _Lines(self.output.emit)
        try:
            with contextlib.redirect_stdout(lines):  # type: ignore[arg-type]
                result = self.work(self)
            lines.flush()
        except Cancelled:
            lines.flush()
            self.cancelled.emit()
        except Exception:  # noqa: BLE001  (shown to the user, the window keeps running)
            lines.flush()
            self.failed.emit(traceback.format_exc())
        else:
            self.succeeded.emit(result)
