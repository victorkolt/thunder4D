import numpy as np
import pytest

import thunder4d as th
from thunder4d.gui.panels.run import RunPanel
from thunder4d.gui.registry import METRICS
from thunder4d.gui.session import Session


def tiny_input(n_passes=3):
    return th.default_input(
        name="tiny", laser={"energy_J": 1e-3, "duration_fs": 200.0}, medium={"gas": "Ar", "pressure_bar": 0.5},
        cell={"roc_m": 1.0, "separation_m": 1.9, "n_passes": n_passes},
        grid={"lambda_min_nm": 990.0, "lambda_max_nm": 1075.0, "device": "cpu"},
        run={"near_field": False})


@pytest.fixture
def panel(qtbot):
    p = RunPanel(Session())
    qtbot.addWidget(p)
    return p


def wait_idle(qtbot, panel):
    qtbot.waitUntil(lambda: panel.task is None, timeout=120_000)


def test_run_needs_a_valid_input(panel):
    assert not panel.run_button.isEnabled()
    assert "No valid input" in panel.input_label.text()
    panel.session.set_config(tiny_input())
    assert panel.run_button.isEnabled()
    assert "3 passes" in panel.input_label.text()
    assert panel.output.text().endswith("tiny.npz")


def test_run_follows_every_pass_and_publishes_the_results(qtbot, panel, tmp_path):
    panel.session.set_config(tiny_input())
    with qtbot.waitSignal(panel.session.results_changed, timeout=120_000) as published:
        panel.start_run(tiny_input(), tmp_path / "tiny.npz")
    wait_idle(qtbot, panel)
    assert panel.outcome is not None and not panel.outcome.stopped
    assert panel.progress.value() == 3
    passes, energy = panel.series["energy"]
    assert passes == [0, 1, 2, 3] and energy[0] == pytest.approx(1e-3, rel=1e-3)
    assert "peak_intensity" in panel.series                  # recorded by the core, followed here
    assert set(panel.card_labels) <= set(METRICS.keys())
    results = published.args[0]
    assert results.path == tmp_path / "tiny.npz"
    assert np.array_equal(results["pass"], [0, 1, 2, 3]) and "Aw_final" in results
    assert panel.figure.figure.axes


def test_stop_saves_the_passes_done(qtbot, panel, tmp_path):
    panel.session.set_config(tiny_input(n_passes=40))
    panel.start_run(tiny_input(n_passes=40), tmp_path / "long.npz")
    assert panel.task is not None
    panel.task.progress.connect(lambda s: panel.stop() if s.passes_done >= 1 else None)
    wait_idle(qtbot, panel)
    assert panel.outcome is not None and panel.outcome.stopped
    with np.load(tmp_path / "long.npz") as f:
        assert len(f["energy"]) == panel.outcome.passes_done + 1 < 41


def test_failure_is_shown_in_the_log(qtbot, panel, tmp_path):
    cfg = tiny_input()
    cfg["medium"]["gas"] = "unobtainium"                       # passes the schema, fails when built
    panel.start_run(cfg, tmp_path / "bad.npz")
    wait_idle(qtbot, panel)
    assert "KeyError" in panel.log.toPlainText() and panel.outcome is None


def test_window_closes_freely_when_idle(panel):
    assert panel.can_close()


def test_error_in_a_pass_keeps_the_passes_done(qtbot, panel, tmp_path, monkeypatch):
    from thunder4d.lensframe import LensFrameSimulation

    real_run = LensFrameSimulation.run

    def failing_run(self, n_passes=None, verbose=True, max_seconds=None):
        if self.passes_done == 2:
            raise MemoryError("out of GPU memory")
        return real_run(self, n_passes, verbose=False, max_seconds=max_seconds)

    monkeypatch.setattr(LensFrameSimulation, "run", failing_run)
    panel.start_run(tiny_input(n_passes=5), tmp_path / "crash.npz")
    wait_idle(qtbot, panel)
    assert "MemoryError" in panel.log.toPlainText()
    with np.load(tmp_path / "crash.npz") as f:
        assert list(f["pass"]) == [0, 1, 2]
