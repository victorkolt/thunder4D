import json

import pytest

import thunder4d as th
from thunder4d.gui._qt import QtWidgets
from thunder4d.gui.fields import EDITORS, FloatOrTableEditor, editor_for, parse_table, rows_to_table
from thunder4d.gui.panels.builder import BuilderPanel
from thunder4d.gui.panels.run import RunPanel
from thunder4d.gui.session import Session
from thunder4d.inputfile import FIELDS, Field


def small_input(**over):
    cfg = th.default_input(
        name="small",
        laser={"energy_J": 1e-3, "duration_fs": 200.0},
        medium={"gas": "Ar", "pressure_bar": 0.5},
        cell={"roc_m": 1.0, "separation_m": 1.9, "n_passes": 2},
        grid={"lambda_min_nm": 990.0, "lambda_max_nm": 1075.0, "device": "cpu"},
    )
    for block, vals in over.items():
        cfg[block].update(vals)
    return cfg


@pytest.fixture
def builder(qtbot):
    session = Session()
    panel = BuilderPanel(session)
    qtbot.addWidget(panel)
    return panel


def line(editor):
    assert isinstance(editor.input, QtWidgets.QLineEdit)
    return editor.input


def type_in(panel, path, text):
    editor = panel.editors[path]
    line(editor).setText(text)
    editor.changed.emit()
    panel.refresh()


# ============================================================ editors
def test_every_kind_of_field_has_an_editor():
    assert {f.kind for f in FIELDS.values()} <= set(EDITORS)


def test_unknown_kind_says_which_kinds_exist(qtbot):
    with pytest.raises(KeyError, match="no editor for fields of kind 'spline'"):
        editor_for(Field("x", "spline"))


def test_numbers_are_shown_in_the_display_unit(qtbot):
    editor = editor_for(FIELDS["laser.energy_J"])
    qtbot.addWidget(editor)
    editor.set_value(1.5e-3)
    assert line(editor).text() == "1.5"
    line(editor).setText("2,5")                              # decimal comma accepted
    assert editor.value() == pytest.approx(2.5e-3)
    line(editor).setText("")
    assert editor.value() is None


def test_integer_field_rejects_fractions(qtbot):
    editor = editor_for(FIELDS["cell.n_passes"])
    qtbot.addWidget(editor)
    line(editor).setText("2.5")
    with pytest.raises(ValueError, match="not an integer"):
        editor.value()
    line(editor).setText("40")
    assert editor.value() == 40 and isinstance(editor.value(), int)


def test_tables_from_text_files():
    text = "wavelength (nm), R\n1000, 0.99\n1030;0.995\n1060\t0.98\n"
    rows = parse_table(text, 2, 2)
    assert rows_to_table(rows, ("wavelength_nm", "value")) == {"wavelength_nm": [1000, 1030, 1060],
                                                               "value": [0.99, 0.995, 0.98]}
    spectrum = rows_to_table(parse_table("1000 1\n1030 2\n", 3, 2), ("wavelength_nm", "intensity", "phase_rad"))
    assert set(spectrum) == {"wavelength_nm", "intensity"}           # optional phase column left out
    with pytest.raises(ValueError, match="columns"):
        parse_table("1000 1 2 3\n", 3, 2)
    with pytest.raises(ValueError, match="no rows"):
        parse_table("header only\n", 2, 2)


def test_table_editor_imports_a_csv(qtbot, tmp_path):
    editor = editor_for(FIELDS["mirrors.reflectivity"])
    qtbot.addWidget(editor)
    csv = tmp_path / "R.csv"
    csv.write_text("nm,R\n1000,0.99\n1060,0.98\n")
    assert isinstance(editor, FloatOrTableEditor)
    editor.mode.setCurrentIndex(1)
    editor.table.load_file(str(csv))
    assert editor.value() == {"wavelength_nm": [1000.0, 1060.0], "value": [0.99, 0.98]}


# ============================================================ builder
def test_round_trip_through_the_form_keeps_the_file(builder, tmp_path):
    cfg = small_input(laser={"shape": "measured", "gdd_fs2": 500.0,
                             "spectrum": {"wavelength_nm": [1000.0, 1030.0, 1060.0], "intensity": [0.1, 1.0, 0.1]}},
                      beam={"injection": "waist", "w0_m": 3.0e-4},
                      medium={"n2_1bar": 9.8e-24},
                      cell={"herriott": {"round_trips": 10, "k": 1, "branch": "concentric"}},
                      mirrors={"reflectivity": {"wavelength_nm": [990.0, 1080.0], "value": [0.99, 0.98]},
                               "gdd_fs2": -20.0, "diameter_m": 0.0254, "coating": "HR1030"})
    cfg["design"] = {"source": "test"}
    reference = th.save_input(cfg, tmp_path / "reference.json")
    assert builder.open_file(reference)
    assert builder.valid
    assert builder.save_to(tmp_path / "copy.json")
    assert json.loads((tmp_path / "copy.json").read_text()) == json.loads(reference.read_text())


def test_bad_values_are_flagged_and_block_saving(builder, tmp_path):
    builder.load_config(small_input(), None)
    assert builder.valid and builder.session.config is not None
    type_in(builder, "laser.energy_J", "abc")
    type_in(builder, "cell.separation_m", "3")
    assert {p for p, _ in builder.problems} == {"laser.energy_J", "cell.separation_m"}
    assert "border" in line(builder.editors["laser.energy_J"]).styleSheet()
    assert not builder.save_action.isEnabled()
    assert builder.session.config is None                    # nothing runnable is published
    assert "need 0 &lt; separation_m" in builder.status.text()
    type_in(builder, "laser.energy_J", "2")
    type_in(builder, "cell.separation_m", "1.9")
    assert builder.valid
    assert builder.session.config["laser"]["energy_J"] == pytest.approx(2e-3)


def test_summary_of_a_valid_input(builder):
    builder.load_config(small_input(), None)
    assert "Ready to run" in builder.status.text()
    assert builder.summary is not None and builder.summary["grid_shape"][0] > 0


def test_conditional_and_advanced_fields(builder):
    builder.load_config(small_input(), None)
    form, editor = builder.rows["beam.w0_m"]
    assert not form.isRowVisible(editor)
    builder.editors["beam.injection"].set_value("waist")
    builder.editors["beam.injection"].changed.emit()
    assert form.isRowVisible(editor)
    assert builder.groups["solver"].isHidden()                # only advanced fields
    builder.advanced.setChecked(True)
    assert not builder.groups["solver"].isHidden()


def test_input_from_another_panel_is_shown(builder):
    builder.session.set_config(small_input(medium={"gas": "Ne", "pressure_bar": 2.0}))
    assert builder.editors["medium.gas"].value() == "Ne"
    assert line(builder.editors["medium.pressure_bar"]).text() == "2"


def test_incomplete_file_opens_with_its_problems(builder, tmp_path, monkeypatch):
    path = tmp_path / "partial.json"
    path.write_text(json.dumps({"schema": "thunder4d-input", "medium": {"gas": "Ar"}}))
    assert builder.open_file(path)
    assert ("laser.energy_J", "laser.energy_J is required") in builder.problems
    warnings = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda *a: warnings.append(a))
    path.write_text(json.dumps({"laser": {"energy": 1}}))     # typo in a key: refused
    assert not builder.open_file(path) and warnings


def test_builder_is_a_tab_of_the_main_window(qtbot):
    from thunder4d.gui.app import MainWindow

    win = MainWindow()
    qtbot.addWidget(win)
    assert isinstance(win.panels["builder"], BuilderPanel)
    assert win.tabs.tabText(win.tabs.indexOf(win.panels["builder"])) == "Simulation"


def test_input_from_another_panel_asks_before_replacing_edits(qtbot, builder, monkeypatch):
    builder.load_config(small_input(), None)
    type_in(builder, "medium.pressure_bar", "0.7")
    answers = []

    def answer(reply):
        def ask(*a):
            answers.append(a)
            return reply
        return ask

    monkeypatch.setattr(QtWidgets.QMessageBox, "question", answer(QtWidgets.QMessageBox.StandardButton.No))
    builder.session.set_config(small_input(medium={"gas": "Ne", "pressure_bar": 2.0}))
    assert answers and builder.editors["medium.gas"].value() == "Ar"          # edits kept
    assert builder.session.config["medium"]["pressure_bar"] == pytest.approx(0.7)
    run = RunPanel(builder.session)                                          # listens after the builder
    qtbot.addWidget(run)
    builder.session.set_config(small_input(medium={"gas": "Ne", "pressure_bar": 2.0}))
    assert "Ar 0.7 bar" in run.input_label.text()

    monkeypatch.setattr(QtWidgets.QMessageBox, "question", answer(QtWidgets.QMessageBox.StandardButton.Yes))
    builder.session.set_config(small_input(medium={"gas": "Ne", "pressure_bar": 2.0}))
    assert builder.editors["medium.gas"].value() == "Ne"
    answers.clear()
    builder.session.set_config(small_input(medium={"gas": "Kr", "pressure_bar": 1.0}))   # next design
    assert not answers and builder.editors["medium.gas"].value() == "Kr"                # no edit: no question
