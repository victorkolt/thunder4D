"""Editors of the input-file fields, one per Field.kind (thunder4d.inputfile.FIELDS).

The builder never knows which keys exist: it asks editor_for(field) for each entry of FIELDS. A new kind
of field = a FieldEditor subclass registered in EDITORS (tests check every kind used in FIELDS has one).

An editor converts between the file value (unit of the key name) and what is shown (Field.unit):
value() returns the file value or raises ValueError on text that cannot be read.
"""
from __future__ import annotations

import re
from typing import Any

from ..inputfile import Field
from ._qt import QtCore, QtGui, QtWidgets

ERROR_STYLE = "border: 1px solid #d32f2f; border-radius: 2px;"


def format_number(v: float) -> str:
    return f"{v:.10g}"


class FieldEditor(QtWidgets.QWidget):
    """Base class: an input widget plus the unit; `changed` is emitted on every edit."""
    changed = QtCore.Signal()

    def __init__(self, field: Field, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self.field = field
        self._layout = QtWidgets.QHBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self.input = self.make_input()
        self._layout.addWidget(self.input, 1)
        if field.unit:
            self._layout.addWidget(QtWidgets.QLabel(field.unit))
        if field.help:
            self.setToolTip(field.help)

    def make_input(self) -> QtWidgets.QWidget:
        raise NotImplementedError

    def value(self) -> Any:
        raise NotImplementedError

    def set_value(self, value: Any) -> None:
        raise NotImplementedError

    def set_problem(self, message: str | None) -> None:
        """Mark the editor (red frame, message as tooltip), or clear the mark with None."""
        self.input.setStyleSheet(ERROR_STYLE if message else "")
        self.input.setToolTip(message or self.field.help)


class FloatEditor(FieldEditor):
    cast: type = float

    def make_input(self) -> QtWidgets.QLineEdit:
        edit = QtWidgets.QLineEdit()
        edit.setPlaceholderText(self.field.none_text if self.field.optional else "required")
        edit.textEdited.connect(self.changed)
        return edit

    def value(self) -> float | int | None:
        assert isinstance(self.input, QtWidgets.QLineEdit)
        text = self.input.text().strip().replace(",", ".")
        if not text:
            return None
        try:
            v = float(text) * self.field.scale
        except ValueError:
            raise ValueError(f"{text!r} is not a number") from None
        if self.cast is int:
            if v != round(v):
                raise ValueError(f"{text!r} is not an integer")
            return round(v)
        return v

    def set_value(self, value: Any) -> None:
        assert isinstance(self.input, QtWidgets.QLineEdit)
        self.input.setText("" if value is None else format_number(value / self.field.scale))


class IntEditor(FloatEditor):
    cast = int


class BoolEditor(FieldEditor):
    def make_input(self) -> QtWidgets.QCheckBox:
        box = QtWidgets.QCheckBox()
        box.toggled.connect(self.changed)
        return box

    def value(self) -> bool:
        assert isinstance(self.input, QtWidgets.QCheckBox)
        return self.input.isChecked()

    def set_value(self, value: Any) -> None:
        assert isinstance(self.input, QtWidgets.QCheckBox)
        self.input.setChecked(bool(value))


class ChoiceEditor(FieldEditor):
    """A list of choices; None (not chosen yet) shows as an empty first entry."""

    def make_input(self) -> QtWidgets.QComboBox:
        combo = QtWidgets.QComboBox()
        combo.addItem("", None)
        for choice in self.field.choices:
            combo.addItem(str(choice), choice)
        combo.currentIndexChanged.connect(self.changed)
        return combo

    def value(self) -> Any:
        assert isinstance(self.input, QtWidgets.QComboBox)
        return self.input.currentData()

    def set_value(self, value: Any) -> None:
        assert isinstance(self.input, QtWidgets.QComboBox)
        index = self.input.findData(value)
        if index < 0:                         # a value from the file that is not a known choice: keep it
            self.input.addItem(str(value), value)
            index = self.input.count() - 1
        self.input.setCurrentIndex(index)


class TextEditor(FieldEditor):
    def make_input(self) -> QtWidgets.QLineEdit:
        edit = QtWidgets.QLineEdit()
        edit.textEdited.connect(self.changed)
        return edit

    def value(self) -> str:
        assert isinstance(self.input, QtWidgets.QLineEdit)
        return self.input.text()

    def set_value(self, value: Any) -> None:
        assert isinstance(self.input, QtWidgets.QLineEdit)
        self.input.setText(value or "")


class MultilineEditor(FieldEditor):
    def make_input(self) -> QtWidgets.QPlainTextEdit:
        edit = QtWidgets.QPlainTextEdit()
        edit.setFixedHeight(60)
        edit.textChanged.connect(self.changed)
        return edit

    def value(self) -> str:
        assert isinstance(self.input, QtWidgets.QPlainTextEdit)
        return self.input.toPlainText()

    def set_value(self, value: Any) -> None:
        assert isinstance(self.input, QtWidgets.QPlainTextEdit)
        self.input.blockSignals(True)
        self.input.setPlainText(value or "")
        self.input.blockSignals(False)


class InfoEditor(FieldEditor):
    """Shown, not edited (e.g. provenance written by the design helper); the value is kept as is."""

    def make_input(self) -> QtWidgets.QLabel:
        self._value: Any = None
        label = QtWidgets.QLabel("–")
        label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        return label

    def value(self) -> Any:
        return self._value

    def set_value(self, value: Any) -> None:
        assert isinstance(self.input, QtWidgets.QLabel)
        self._value = value
        if isinstance(value, dict):
            text = ", ".join(f"{k} = {v}" for k, v in value.items())
        else:
            text = "–" if value is None else str(value)
        self.input.setText(text)


# ------------------------------------------------------------ tables (spectra, mirror curves)
def parse_table(text: str, n_columns: int, n_required: int) -> list[list[float]]:
    """Rows of numbers from CSV / TSV / whitespace text; lines that are not numbers (headers) are skipped."""
    rows = []
    for line in text.splitlines():
        cells = [c for c in re.split(r"[,;\s]+", line.strip()) if c]
        try:
            numbers = [float(c) for c in cells]
        except ValueError:
            continue
        if not numbers:
            continue
        if not n_required <= len(numbers) <= n_columns:
            raise ValueError(f"expected {n_required} to {n_columns} columns, got {len(numbers)}: {line.strip()!r}")
        rows.append(numbers)
    if not rows:
        raise ValueError("no rows of numbers found")
    return rows


def rows_to_table(rows: list[list[float]], columns: tuple[str, ...]) -> dict[str, list[float]]:
    """{column: values}; a column missing from every row (an optional trailing one) is left out."""
    width = max(len(r) for r in rows)
    if any(len(r) != width for r in rows):
        raise ValueError("every row needs the same number of columns")
    return {col: [r[i] for r in rows] for i, col in enumerate(columns[:width])}


def table_to_rows(table: dict[str, list[float]], columns: tuple[str, ...]) -> list[list[float]]:
    present = [c for c in columns if c in table]
    return [list(r) for r in zip(*(table[c] for c in present))]


class TableDialog(QtWidgets.QDialog):
    """Edit a table: one row per wavelength; paste from a spreadsheet with Ctrl+V."""

    def __init__(self, columns: tuple[str, ...], rows: list[list[float]], parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Edit table")
        self.columns = columns
        self.grid = QtWidgets.QTableWidget(0, len(columns))
        self.grid.setHorizontalHeaderLabels(list(columns))
        self.grid.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        for row in rows:
            self._append(row)
        add = QtWidgets.QPushButton("Add row")
        add.clicked.connect(lambda: self._append([]))
        remove = QtWidgets.QPushButton("Remove selected rows")
        remove.clicked.connect(self._remove_selected)
        paste = QtWidgets.QPushButton("Paste")
        paste.setToolTip("Replace the table with the clipboard (columns: " + ", ".join(columns) + ")")
        paste.clicked.connect(self.paste)
        QtGui.QShortcut(QtGui.QKeySequence.StandardKey.Paste, self, self.paste)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Ok
                                             | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        tools = QtWidgets.QHBoxLayout()
        for b in (add, remove, paste):
            tools.addWidget(b)
        tools.addStretch(1)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(tools)
        layout.addWidget(self.grid)
        layout.addWidget(buttons)
        self.resize(480, 420)

    def _append(self, row: list[float]) -> None:
        r = self.grid.rowCount()
        self.grid.insertRow(r)
        for c, v in enumerate(row):
            self.grid.setItem(r, c, QtWidgets.QTableWidgetItem(format_number(v)))

    def _remove_selected(self) -> None:
        for r in sorted({i.row() for i in self.grid.selectedIndexes()}, reverse=True):
            self.grid.removeRow(r)

    def paste(self) -> None:
        try:
            rows = parse_table(QtGui.QGuiApplication.clipboard().text(), len(self.columns), 2)
        except ValueError as err:
            QtWidgets.QMessageBox.warning(self, "Paste", str(err))
            return
        self.grid.setRowCount(0)
        for row in rows:
            self._append(row)

    def rows(self) -> list[list[float]]:
        """The rows, empty trailing cells dropped; raises ValueError on a cell that is not a number."""
        rows = []
        for r in range(self.grid.rowCount()):
            cells = [self.grid.item(r, c) for c in range(self.grid.columnCount())]
            texts = [i.text().strip() if i is not None else "" for i in cells]
            while texts and not texts[-1]:
                texts.pop()
            if not texts:
                continue
            try:
                rows.append([float(t.replace(",", ".")) for t in texts])
            except ValueError:
                raise ValueError(f"row {r + 1}: {texts} are not all numbers") from None
        return rows

    def accept(self) -> None:
        try:
            rows = self.rows()
            if rows:
                rows_to_table(rows, self.columns)
        except ValueError as err:
            QtWidgets.QMessageBox.warning(self, "Table", str(err))
            return
        super().accept()


class TableEditor(FieldEditor):
    """{column: [...]} tables: a summary, import from a CSV / text file, edit in a dialog, clear."""

    def make_input(self) -> QtWidgets.QWidget:
        self._table: dict[str, list[float]] | None = None
        box = QtWidgets.QWidget()
        row = QtWidgets.QHBoxLayout(box)
        row.setContentsMargins(0, 0, 0, 0)
        self.summary = QtWidgets.QLabel()
        row.addWidget(self.summary, 1)
        for text, slot in (("Import…", self.import_file), ("Edit…", self.edit), ("Clear", self.clear)):
            button = QtWidgets.QPushButton(text)
            button.clicked.connect(slot)
            row.addWidget(button)
        self._show()
        return box

    def _show(self) -> None:
        t = self._table
        if not t:
            self.summary.setText("no table")
            return
        lam = t[self.field.columns[0]]
        self.summary.setText(f"{len(lam)} points, {min(lam):.6g}–{max(lam):.6g} nm")

    def value(self) -> dict[str, list[float]] | None:
        return self._table

    def set_value(self, value: Any) -> None:
        self._table = value if isinstance(value, dict) else None
        self._show()

    def set_table(self, table: dict[str, list[float]] | None) -> None:
        self.set_value(table)
        self.changed.emit()

    def import_file(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, f"Import {self.field.label}", "",
                                                        "Tables (*.csv *.txt *.dat *.tsv);;All files (*)")
        if path:
            self.load_file(path)

    def load_file(self, path: str) -> None:
        try:
            with open(path) as f:
                rows = parse_table(f.read(), len(self.field.columns), 2)
            self.set_table(rows_to_table(rows, self.field.columns))
        except (OSError, ValueError) as err:
            QtWidgets.QMessageBox.warning(self, f"Import {self.field.label}", f"{path}:\n{err}")

    def edit(self) -> None:
        rows = table_to_rows(self._table, self.field.columns) if self._table else []
        dialog = TableDialog(self.field.columns, rows, self)
        if dialog.exec():
            rows = dialog.rows()
            self.set_table(rows_to_table(rows, self.field.columns) if rows else None)

    def clear(self) -> None:
        self.set_table(None)


class FloatOrTableEditor(FieldEditor):
    """A constant, or a table versus wavelength."""

    def make_input(self) -> QtWidgets.QWidget:
        box = QtWidgets.QWidget()
        row = QtWidgets.QHBoxLayout(box)
        row.setContentsMargins(0, 0, 0, 0)
        self.mode = QtWidgets.QComboBox()
        self.mode.addItems(["constant", "table"])
        plain = Field(self.field.label, optional=self.field.optional, none_text=self.field.none_text)
        self.constant = FloatEditor(plain)
        self.table = TableEditor(Field(self.field.label, "table", columns=self.field.columns))
        self.stack = QtWidgets.QStackedWidget()
        self.stack.addWidget(self.constant)
        self.stack.addWidget(self.table)
        self.stack.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Maximum)
        row.addWidget(self.mode)
        row.addWidget(self.stack, 1)
        self.mode.currentIndexChanged.connect(self._mode_changed)
        self.constant.changed.connect(self.changed)
        self.table.changed.connect(self.changed)
        return box

    def _mode_changed(self, index: int) -> None:
        self.stack.setCurrentIndex(index)
        self.changed.emit()

    def value(self) -> Any:
        return self.table.value() if self.mode.currentIndex() == 1 else self.constant.value()

    def set_value(self, value: Any) -> None:
        is_table = isinstance(value, dict)
        self.mode.blockSignals(True)
        self.mode.setCurrentIndex(int(is_table))
        self.mode.blockSignals(False)
        self.stack.setCurrentIndex(int(is_table))
        self.table.set_value(value if is_table else None)
        self.constant.set_value(None if is_table else value)

    def set_problem(self, message: str | None) -> None:
        current = self.constant.input if self.mode.currentIndex() == 0 else self.table.summary
        current.setStyleSheet(ERROR_STYLE if message else "")
        self.setToolTip(message or self.field.help)


class HiddenEditor(FieldEditor):
    """Not shown; keeps the value of the file (e.g. the schema, the design provenance)."""

    def make_input(self) -> QtWidgets.QWidget:
        self._value: Any = None
        return QtWidgets.QWidget()

    def value(self) -> Any:
        return self._value

    def set_value(self, value: Any) -> None:
        self._value = value


EDITORS: dict[str, type[FieldEditor]] = {
    "float": FloatEditor, "int": IntEditor, "bool": BoolEditor, "choice": ChoiceEditor, "text": TextEditor,
    "multiline": MultilineEditor, "info": InfoEditor, "table": TableEditor, "float_or_table": FloatOrTableEditor,
    "hidden": HiddenEditor,
}


def editor_for(field: Field) -> FieldEditor:
    try:
        cls = EDITORS[field.kind]
    except KeyError:
        raise KeyError(f"no editor for fields of kind {field.kind!r} (known: {sorted(EDITORS)})") from None
    return cls(field)
