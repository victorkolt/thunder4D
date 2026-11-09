"""Simulation builder: every field of the standard input file, checked while typing.

The form is generated from thunder4d.inputfile.FIELDS (one section per BLOCKS entry): a key added to the
input file appears here with no change to this module. Checks: inputfile.input_problems (each problem is
shown on its field) and, once the input is valid, inputfile.describe_input (eigenmode, grid, P/Pcr, ...).
"""
from __future__ import annotations

import copy
import html
import json
from pathlib import Path
from typing import Any

from ...inputfile import BLOCKS, FIELDS, default_input, describe_input, get_path, input_problems, save_input, set_path
from .._qt import QtCore, QtWidgets
from ..fields import FieldEditor, editor_for
from ..registry import PANELS, PanelSpec
from ..session import Session

FILE_FILTER = "THUNDER4D input (*.json);;All files (*)"


def block_of(path: str) -> str:
    return path.split(".")[0] if "." in path else "general"


class BuilderPanel(QtWidgets.QWidget):
    def __init__(self, session: Session, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self.session = session
        self.path: Path | None = None
        self.modified = False
        self.cfg: dict[str, Any] = default_input()
        self.parse_errors: dict[str, str] = {}
        self.problems: list[tuple[str, str]] = []
        self.summary: dict[str, Any] | None = None
        self._loading = False            # set while the editors are filled from a config (no edit events)
        self._publishing = False         # set while this panel updates the session (ignore its own signal)

        self.editors: dict[str, FieldEditor] = {}
        self.rows: dict[str, tuple[QtWidgets.QFormLayout, FieldEditor]] = {}
        self.groups: dict[str, QtWidgets.QGroupBox] = {}

        self._refresh_timer = QtCore.QTimer(self)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.setInterval(150)
        self._refresh_timer.timeout.connect(self.refresh)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self._toolbar())
        splitter = QtWidgets.QSplitter()
        splitter.addWidget(self._form())
        splitter.addWidget(self._checks())
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        layout.addWidget(splitter, 1)

        session.config_changed.connect(self._on_session_config)
        if session.config is not None:
            self.load_config(session.config, session.config_path)
        else:
            self.load_config(self.cfg, None)

    # ------------------------------------------------------------ layout
    def _toolbar(self) -> QtWidgets.QToolBar:
        bar = QtWidgets.QToolBar()
        bar.addAction("New", self.new)
        bar.addAction("Open…", self.open_dialog)
        self.save_action = bar.addAction("Save", self.save)
        bar.addAction("Save as…", self.save_as_dialog)
        bar.addSeparator()
        self.advanced = QtWidgets.QCheckBox("Advanced settings")
        self.advanced.toggled.connect(self._update_visibility)
        bar.addWidget(self.advanced)
        spacer = QtWidgets.QWidget()
        spacer.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Preferred)
        bar.addWidget(spacer)
        self.file_label = QtWidgets.QLabel()
        bar.addWidget(self.file_label)
        return bar

    def _form(self) -> QtWidgets.QScrollArea:
        content = QtWidgets.QWidget()
        column = QtWidgets.QVBoxLayout(content)
        forms = {}
        for block, title in BLOCKS.items():
            group = QtWidgets.QGroupBox(title)
            forms[block] = QtWidgets.QFormLayout(group)
            forms[block].setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
            self.groups[block] = group
            column.addWidget(group)
        column.addStretch(1)
        for path, field in FIELDS.items():
            editor = editor_for(field)
            editor.setObjectName(path)
            editor.changed.connect(lambda p=path: self._edited(p))
            self.editors[path] = editor
            if field.kind == "hidden":
                continue
            label = QtWidgets.QLabel(field.label)
            label.setToolTip(field.help or path)
            forms[block_of(path)].addRow(label, editor)
            self.rows[path] = (forms[block_of(path)], editor)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(content)
        return scroll

    def _checks(self) -> QtWidgets.QWidget:
        box = QtWidgets.QGroupBox("Checks")
        layout = QtWidgets.QVBoxLayout(box)
        self.status = QtWidgets.QLabel()
        self.status.setWordWrap(True)
        self.status.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self.status.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        self.status.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.status, 1)
        return box

    # ------------------------------------------------------------ config <-> editors
    def load_config(self, cfg: dict[str, Any], path: str | Path | None) -> None:
        """Show `cfg` (filled in with the defaults, not checked: an incomplete input can be opened and fixed)."""
        self.cfg = default_input(**copy.deepcopy(cfg))
        self.path = Path(path) if path is not None else None
        self.parse_errors.clear()
        self._loading = True
        try:
            for p, editor in self.editors.items():
                editor.set_value(get_path(self.cfg, p))
        finally:
            self._loading = False
        self.modified = False
        self.refresh()

    def _edited(self, path: str) -> None:
        if self._loading:
            return
        try:
            set_path(self.cfg, path, self.editors[path].value())
            self.parse_errors.pop(path, None)
        except ValueError as err:
            self.parse_errors[path] = f"{path}: {err}"
        self.modified = True
        self._update_visibility()
        self._refresh_timer.start()

    def _on_session_config(self, cfg: dict[str, Any] | None) -> None:
        """An input sent by another panel (e.g. the design helper): asks before replacing unsaved edits;
        the new input is unsaved too (no file yet), so closing the window asks before losing it."""
        if self._publishing or cfg is None:
            return
        if not self.confirm_discard():
            self.refresh()                          # the session keeps showing this panel's input
            return
        self.load_config(cfg, self.session.config_path)
        if self.session.config_path is None:
            self.modified = True
            self.refresh()

    def _publish(self) -> None:
        self._publishing = True
        try:
            self.session.set_config(copy.deepcopy(self.cfg) if self.valid else None, self.path)
        finally:
            self._publishing = False

    # ------------------------------------------------------------ checks
    @property
    def valid(self) -> bool:
        return not self.problems

    def refresh(self) -> None:
        """Re-check the input and update the marks, the summary and the session."""
        self._refresh_timer.stop()
        problems = [(p, msg) for p, msg in self.parse_errors.items()]
        bad = set(self.parse_errors)
        try:
            problems += [(p, msg) for p, msg in input_problems(self.cfg) if p not in bad]
        except (TypeError, ValueError) as err:
            problems.append(("", f"cannot check the input: {err}"))
        self.summary = None
        if not problems:
            try:
                self.summary = describe_input(self.cfg)
            except Exception as err:  # noqa: BLE001  (any failure of the physics is reported, not raised)
                problems.append(("", f"cannot build the cell or the grid: {type(err).__name__}: {err}"))
        self.problems = problems
        messages: dict[str, list[str]] = {}
        for p, msg in problems:
            messages.setdefault(p, []).append(msg)
        for p, editor in self.editors.items():
            editor.set_problem("\n".join(messages[p]) if p in messages else None)
        self._update_visibility()
        self._show_status()
        self.save_action.setEnabled(self.valid)
        self.save_action.setToolTip("" if self.valid else "Fix the problems listed under Checks first")
        name = self.path.name if self.path else "unsaved input"
        self.file_label.setText(name + (" (modified)" if self.modified else ""))
        self._publish()

    def _show_status(self) -> None:
        if self.problems:
            items = "".join(f"<li>{html.escape(msg)}</li>" for _, msg in self.problems)
            self.status.setText(f"<p style='color:#d32f2f'><b>{len(self.problems)} problem(s)</b></p><ul>{items}</ul>")
            return
        s = self.summary or {}
        nx, ny, nt = s["grid_shape"]
        lines = [
            ("Waist at the centre", f"{s['w0_m'] * 1e6:.0f} µm"),
            ("Beam radius on the mirrors", f"{s['w_mirror_m'] * 1e3:.3g} mm"),
            ("Rayleigh length", f"{s['zR_m'] * 1e2:.3g} cm"),
            ("Gouy phase per pass", f"{s['gouy_per_pass_rad']:.4g} rad"),
            ("Path length in the cell", f"{s['path_length_m']:.3g} m"),
            ("Peak power (Gaussian)", f"{s['peak_power_W'] / 1e9:.3g} GW"),
            ("P / P<sub>cr</sub>", f"{s['p_over_pcr']:.3g}"),
            ("Peak fluence on the mirrors", f"{s['mirror_fluence_J_cm2'] * 1e3:.3g} mJ/cm²"),
            ("Grid", f"{nx} × {ny} × {nt} ({s['field_MB']:.3g} MB per field)"),
        ]
        rows = "".join(f"<tr><td>{k}</td><td>&nbsp;&nbsp;<b>{v}</b></td></tr>" for k, v in lines)
        warning = ""
        if s["p_over_pcr"] >= 1:
            warning = "<p style='color:#d32f2f'>The peak power exceeds the critical power: the beam self-focuses.</p>"
        self.status.setText(f"<p style='color:#2e7d32'><b>Ready to run</b></p><table>{rows}</table>{warning}")

    def _update_visibility(self) -> None:
        show_advanced = self.advanced.isChecked()
        shown_per_block: dict[str, int] = {}
        for path, (form, editor) in self.rows.items():
            field = editor.field
            visible = show_advanced or not field.advanced
            if field.show_if is not None:
                other, values = field.show_if
                visible = visible and get_path(self.cfg, other) in values
            form.setRowVisible(editor, visible)
            shown_per_block[block_of(path)] = shown_per_block.get(block_of(path), 0) + visible
        for block, group in self.groups.items():
            group.setVisible(shown_per_block.get(block, 0) > 0)

    # ------------------------------------------------------------ files
    def confirm_discard(self) -> bool:
        if not self.modified:
            return True
        answer = QtWidgets.QMessageBox.question(self, "Unsaved changes", "Discard the changes to the current input?")
        return answer == QtWidgets.QMessageBox.StandardButton.Yes

    def can_close(self) -> bool:
        return self.confirm_discard()

    def new(self) -> None:
        if self.confirm_discard():
            self.load_config(default_input(), None)

    def open_dialog(self) -> None:
        if not self.confirm_discard():
            return
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open input", "", FILE_FILTER)
        if path:
            self.open_file(path)

    def open_file(self, path: str | Path) -> bool:
        try:
            raw = json.loads(Path(path).read_text())
            if not isinstance(raw, dict):
                raise TypeError("not a JSON object")
            default_input(**raw)                       # unknown keys -> KeyError
        except (OSError, TypeError, ValueError, KeyError) as err:
            QtWidgets.QMessageBox.warning(self, "Open input", f"Cannot open {path}:\n{err}")
            return False
        self.load_config(raw, path)
        return True

    def save(self) -> bool:
        if self.path is None:
            return self.save_as_dialog()
        return self.save_to(self.path)

    def save_as_dialog(self) -> bool:
        start = str(self.path or Path.cwd() / f"{self.cfg.get('name') or 'mpc'}.json")
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save input", start, FILE_FILTER)
        return self.save_to(path) if path else False

    def save_to(self, path: str | Path) -> bool:
        self.refresh()
        if not self.valid:
            QtWidgets.QMessageBox.warning(self, "Save input", "The input has problems, see Checks.")
            return False
        try:
            save_input(self.cfg, path)
        except (OSError, ValueError) as err:
            QtWidgets.QMessageBox.warning(self, "Save input", f"Cannot save {path}:\n{err}")
            return False
        self.path = Path(path)
        self.modified = False
        self.refresh()
        return True


PANELS.register(PanelSpec("builder", "Simulation", BuilderPanel, order=10,
                          tooltip="Laser, gas, cell, mirrors and grid of the simulation"))
