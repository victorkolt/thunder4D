import time

import pytest

from thunder4d.design import DesignProblem
from thunder4d.gui.fields import EDITORS
from thunder4d.gui.panels.design import DESIGN_FIELDS, DesignPanel, get_attr, set_attr
from thunder4d.gui.session import Session
from thunder4d.gui.worker import Task


@pytest.fixture
def panel(qtbot):
    p = DesignPanel(Session())
    qtbot.addWidget(p)
    return p


def edit(panel, path, value):
    editor = panel.editors[path]
    editor.set_value(value)
    editor.changed.emit()


def small_problem(panel):
    """One gas, few geometries, analytic solver: a search of a fraction of a second."""
    panel.preset.setCurrentText("pharos_ph2_sp_1mJ")
    edit(panel, "constraints.gas.allowed", ["Ar"])
    edit(panel, "search.round_trips", [8, 14])
    edit(panel, "refinement.n_candidates", 2)
    panel.solver.setCurrentIndex(1)


def wait_idle(qtbot, panel, timeout=60_000):
    qtbot.waitUntil(lambda: panel.task is None, timeout=timeout)


# ============================================================ form <-> DesignProblem
def test_every_design_field_exists_and_has_an_editor():
    problem = DesignProblem()
    for path, field in DESIGN_FIELDS.items():
        get_attr(problem, path)                                   # AttributeError if a knob was renamed
        assert field.kind in EDITORS, path


def test_tuples_keep_their_types():
    problem = DesignProblem()
    set_attr(problem, "search.round_trips", [5.0, 30.0])
    assert problem.search.round_trips == (5, 30) and isinstance(problem.search.round_trips[0], int)
    set_attr(problem, "constraints.mirrors.allowed_roc_m", [1.0, 2.0])
    assert problem.constraints.mirrors.allowed_roc_m == (1.0, 2.0)


def test_preset_fills_the_laser_and_an_edit_makes_it_custom(panel):
    panel.preset.setCurrentText("pharos_ph2_sp_2mJ")
    assert panel.problem.laser.energy_J == pytest.approx(2e-3)
    assert panel.editors["laser.energy_J"].input.text() == "2"
    panel.editors["laser.energy_J"].input.setText("1.5")
    panel.editors["laser.energy_J"].changed.emit()
    assert panel.problem.laser.energy_J == pytest.approx(1.5e-3)
    assert panel.preset.currentText() == "custom"


def test_bad_value_blocks_the_search(panel):
    panel.editors["constraints.mirrors.allowed_roc_m"].input.setText("1, two")
    panel.editors["constraints.mirrors.allowed_roc_m"].changed.emit()
    assert not panel.run_button.isEnabled()


@pytest.mark.parametrize(("path", "text"), [("constraints.pressure.pressure_bar", "3"),
                                            ("search.round_trips", "20, 5"),
                                            ("refinement.n_candidates", ""),
                                            ("constraints.space.max_separation_m.value", "")])
def test_incomplete_values_are_flagged_before_the_search(panel, path, text):
    panel.editors[path].input.setText(text)
    panel.editors[path].changed.emit()
    assert path in panel.errors and not panel.run_button.isEnabled()


def test_no_gas_is_flagged(panel):
    edit(panel, "constraints.gas.allowed", [])
    assert "choose at least one" in panel.errors["constraints.gas.allowed"]


def test_target_duration_shown_only_for_that_objective(panel):
    form, editor = panel.rows["target.target_duration_fs"]
    assert not form.isRowVisible(editor)
    edit(panel, "target.objective", "target_duration")
    assert form.isRowVisible(editor)


# ============================================================ search -> builder
def test_search_then_use_a_design_in_the_builder(qtbot, panel):
    small_problem(panel)
    panel.start()
    wait_idle(qtbot, panel)
    assert panel.candidates, panel.log.toPlainText()
    assert panel.table.rowCount() == len(panel.candidates)
    assert "bar" in panel.details.toPlainText()                    # report of the selected design
    with qtbot.waitSignal(panel.session.panel_requested) as shown:
        panel.use_in_builder()
    assert shown.args == ["builder"]
    cfg = panel.session.config
    best = panel.candidates[0]
    assert cfg["medium"]["gas"] == best.gas and cfg["cell"]["n_passes"] == best.n_passes


def test_design_map_of_the_selected_design(qtbot, panel):
    small_problem(panel)
    panel.start()
    wait_idle(qtbot, panel)
    panel.compute_map()
    wait_idle(qtbot, panel)
    assert panel.map_figure.figure.axes
    assert "map" in panel.state.text()


def test_stop_interrupts_the_search(qtbot, panel):
    small_problem(panel)
    panel.solver.setCurrentIndex(0)                                # fast 1+1D: seconds per evaluation
    panel.start()
    panel.stop()
    wait_idle(qtbot, panel)
    assert panel.state.text() == "Stopped." and not panel.candidates


# ============================================================ background task
def test_task_relays_prints_result_and_errors(qtbot):
    lines = []

    def work(task):
        print("first line")
        print("second", end="")
        return 42

    task = Task(work)
    task.output.connect(lines.append)
    with qtbot.waitSignal(task.succeeded) as done:
        task.start()
    assert done.args == [42] and lines == ["first line", "second"]
    task.wait()

    failing = Task(lambda t: 1 / 0)
    with qtbot.waitSignal(failing.failed) as failed:
        failing.start()
    assert "ZeroDivisionError" in failed.args[0]
    failing.wait()


def test_task_cancel(qtbot):
    def work(task):
        while True:
            task.check()
            time.sleep(0.01)

    task = Task(work)
    with qtbot.waitSignal(task.cancelled):
        task.start()
        task.cancel()
    task.wait()


def test_two_tasks_keep_their_own_output(qtbot):
    """A design search and a run at the same time: each log gets its own lines, even after the other ends."""
    import sys

    stdout = sys.stdout
    run_lines, design_lines = [], []

    def long(task):
        for i in range(5):
            print("run", i)
            time.sleep(0.05)

    def short(task):
        print("design")

    run = Task(long)
    run.output.connect(run_lines.append)
    design = Task(short)
    design.output.connect(design_lines.append)
    with qtbot.waitSignal(run.succeeded, timeout=10_000):
        run.start()
        design.start()
    design.wait()
    run.wait()
    qtbot.waitUntil(lambda: len(run_lines) == 5)
    assert run_lines == [f"run {i}" for i in range(5)] and design_lines == ["design"]
    sys.stdout.write("")                      # main thread still prints to the real stdout
    assert getattr(sys.stdout, "fallback", sys.stdout) is stdout
