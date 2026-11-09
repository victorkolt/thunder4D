import json
import subprocess
import sys

import pytest

from thunder4d.gui import main
from thunder4d.gui.registry import MetricSpec, PanelSpec, PlotSpec, Registry, available_plots


# ============================================================ optional install
def test_core_and_registry_do_not_import_qt():
    code = ("import sys, thunder4d, thunder4d.gui, thunder4d.gui.registry; "
            "print(any(m.split('.')[0] == 'PySide6' for m in sys.modules))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


def test_missing_pyside6_says_how_to_install(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "PySide6", None)            # import PySide6 -> ImportError
    monkeypatch.delitem(sys.modules, "thunder4d.gui._qt", raising=False)
    assert main([]) == 1
    assert "thunder4d[gui]" in capsys.readouterr().err


# ============================================================ registries (no Qt)
def test_registry_order_then_registration_order():
    reg = Registry("panel")
    for key, order in (("b", 20), ("a", 10), ("c", 20)):
        reg.register(PanelSpec(key, key.upper(), factory=lambda s: None, order=order))
    assert reg.keys() == ["a", "b", "c"]
    assert "b" in reg and len(reg) == 3 and reg["c"].title == "C"


def test_registry_rejects_duplicates_unless_replaced():
    reg = Registry("plot")
    reg.register(PlotSpec("p", "first", draw=lambda r, f, o: None))
    with pytest.raises(ValueError, match="already registered"):
        reg.register(PlotSpec("p", "second", draw=lambda r, f, o: None))
    reg.register(PlotSpec("p", "second", draw=lambda r, f, o: None), replace=True)
    assert reg["p"].title == "second"
    reg.unregister("p")
    assert len(reg) == 0


def test_plots_are_offered_only_when_their_data_is_there():
    reg = Registry("plot")
    reg.register(PlotSpec("spectra", "Spectra", draw=lambda r, f, o: None, requires=("lam", "spectrum")))
    reg.register(PlotSpec("stokes", "Stokes", draw=lambda r, f, o: None, requires=("S0",)))
    assert [p.key for p in available_plots({"lam": 0, "spectrum": 0}, reg)] == ["spectra"]


def test_metric_reads_the_last_recorded_value():
    m = MetricSpec("energy", "Energy", unit="mJ", scale=1e3, warn_above=1.5)
    assert m.value({"energy": []}) is None and m.text({}) == "–"
    h = {"energy": [1e-3, 2e-3]}
    assert m.value(h) == pytest.approx(2.0)
    assert m.text(h) == "2 mJ"
    assert m.warning(m.value(h)) and not m.warning(1.0) and not m.warning(None)


# ============================================================ main window (offscreen)
def test_window_shows_the_registered_panels_in_order(qtbot):
    from thunder4d.gui._qt import QtWidgets
    from thunder4d.gui.app import MainWindow

    reg = Registry("panel")
    reg.register(PanelSpec("results", "Results", factory=lambda s: QtWidgets.QLabel("r"), order=40))
    reg.register(PanelSpec("builder", "Simulation", factory=lambda s: QtWidgets.QLabel("b"), order=10))
    win = MainWindow(panels=reg)
    qtbot.addWidget(win)
    assert [win.tabs.tabText(i) for i in range(win.tabs.count())] == ["Simulation", "Results"]


def test_window_without_panels_has_a_welcome_tab(qtbot):
    from thunder4d.gui.app import MainWindow

    win = MainWindow(panels=Registry("panel"))
    qtbot.addWidget(win)
    assert win.tabs.count() == 1 and win.tabs.tabText(0) == "Welcome"


def test_a_broken_panel_does_not_break_the_window(qtbot):
    from thunder4d.gui._qt import QtWidgets
    from thunder4d.gui.app import MainWindow

    def broken(session):
        raise RuntimeError("boom")

    reg = Registry("panel")
    reg.register(PanelSpec("bad", "Bad", factory=broken))
    win = MainWindow(panels=reg)
    qtbot.addWidget(win)
    label = win.panels["bad"]
    assert isinstance(label, QtWidgets.QLabel) and "boom" in label.text()


def test_panels_share_the_session(qtbot):
    from thunder4d.gui._qt import QtWidgets
    from thunder4d.gui.app import MainWindow

    seen = {}

    def listener(session):
        session.config_changed.connect(seen.update)
        return QtWidgets.QWidget()

    reg = Registry("panel")
    reg.register(PanelSpec("a", "A", factory=listener))
    win = MainWindow(panels=reg)
    qtbot.addWidget(win)
    with qtbot.waitSignal(win.session.config_changed):
        win.session.set_config({"name": "cell"})
    assert seen == {"name": "cell"}


def test_input_file_of_the_command_line_is_loaded(qtbot, tmp_path):
    import thunder4d as th
    from thunder4d.gui.app import build_window, parse_args

    cfg = th.default_input(laser={"energy_J": 1e-3, "duration_fs": 200.0}, medium={"gas": "Ar", "pressure_bar": 0.5},
                           cell={"roc_m": 1.0, "separation_m": 1.9, "n_passes": 2},
                           grid={"lambda_min_nm": 990.0, "lambda_max_nm": 1075.0})
    path = th.save_input(cfg, tmp_path / "cell.json")
    win = build_window(parse_args([str(path)]))
    qtbot.addWidget(win)
    assert win.session.config is not None and win.session.config["medium"]["gas"] == "Ar"
    assert win.session.config_path == path


def test_invalid_input_file_is_reported(qtbot, tmp_path, capsys):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"schema": "thunder4d-input"}))
    assert main([str(bad)]) == 2
    assert "laser.energy_J is required" in capsys.readouterr().err
