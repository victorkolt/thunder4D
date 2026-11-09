import numpy as np
import pytest

import thunder4d as th
from thunder4d import diagnostics as dg
from thunder4d.gui import plots
from thunder4d.gui._qt import QtWidgets
from thunder4d.gui.panels.results import ResultsPanel, write_csv
from thunder4d.gui.registry import PLOTS, PlotSpec, available_plots
from thunder4d.gui.results import Results, load_results
from thunder4d.gui.session import Session
from thunder4d.inputfile import describe_input


def small_input(energy_J=1e-3):
    return th.default_input(
        name="small", laser={"energy_J": energy_J, "duration_fs": 200.0}, medium={"gas": "Ar", "pressure_bar": 0.5},
        cell={"roc_m": 1.0, "separation_m": 1.9, "n_passes": 3},
        grid={"lambda_min_nm": 990.0, "lambda_max_nm": 1075.0, "device": "cpu"},
        run={"near_field": True, "record_mirror": True})


@pytest.fixture(scope="module")
def saved(tmp_path_factory):
    """A short run with every output (near field, mirror records, final field), saved and reloaded."""
    built = th.build_simulation(small_input())
    built.sim.run(verbose=False)
    path = tmp_path_factory.mktemp("run") / "small.npz"
    built.sim.save(path)
    return built, path


@pytest.fixture
def panel(qtbot):
    p = ResultsPanel(Session())
    qtbot.addWidget(p)
    return p


def select(panel, key):
    root = panel.list.invisibleRootItem()
    for i in range(root.childCount()):
        group = root.child(i)
        for j in range(group.childCount()):
            if group.child(j).data(0, 256) == key:
                panel.list.setCurrentItem(group.child(j))
                return
    raise KeyError(key)


def test_rebuilt_grid_gives_the_same_compressed_pulse(saved):
    built, path = saved
    r = load_results(path)
    assert np.allclose(r.grid().Omega, built.grid.Omega)
    assert dg.compress(r.grid(), r["Aw_final"])["fwhm_fs"] == pytest.approx(
        dg.compress(built.grid, built.sim.Aw)["fwhm_fs"])


def test_every_builtin_figure_draws(qtbot, panel, saved):
    _, path = saved
    assert panel.open_file(path)
    builtin = {"transmission", "peak", "caustic", "quality", "mirror_fluence", "spectra", "pulse", "fluence",
               "sgram_x", "sgram_theta", "mode_content"}
    assert builtin <= {p.key for p in panel.plots}
    for key in builtin:
        select(panel, key)
        assert panel.message.text() == "", (key, panel.message.text())
        assert panel.figure.figure.axes, key


def test_figures_need_their_data(saved):
    _, path = saved
    r = load_results(path)
    partial = Results({k: v for k, v in r.items() if k not in ("Aw_final", "mirror_scale")})
    keys = {p.key for p in available_plots(partial, PLOTS)}
    assert "pulse" not in keys                          # no final field
    assert "mirror_fluence" not in keys                 # no mirror scale: would be in frame units
    assert "spectra" in keys


def test_mirror_spot_is_physical(tmp_path):
    """Linear run (1 nJ): the caustic on the mirrors is the eigenmode spot of the cell."""
    built = th.build_simulation(small_input(energy_J=1e-9))
    built.sim.run(verbose=False)
    built.sim.save(tmp_path / "linear.npz")
    d = plots.caustic_data(load_results(tmp_path / "linear.npz"), {})
    w_mirror = describe_input(small_input())["w_mirror_m"]
    assert d["w_mirror_x_m"] == pytest.approx(w_mirror, rel=0.05)


def test_pass_selector_only_for_per_pass_figures(panel, saved):
    _, path = saved
    panel.open_file(path)
    select(panel, "transmission")
    assert not panel.slider.isEnabled()
    select(panel, "spectra")
    assert panel.slider.isEnabled() and panel.slider.maximum() == 3
    panel.slider.setValue(1)
    assert panel.pass_label.text() == "Pass 1"
    legends = [ax.get_legend() for ax in panel.figure.figure.axes if ax.get_legend() is not None]
    assert "pass 1" in legends[0].get_texts()[-1].get_text()


def test_export_figure_and_data(panel, saved, tmp_path):
    _, path = saved
    panel.open_file(path)
    select(panel, "transmission")
    panel.export_figure(tmp_path / "fig.png")
    panel.export_data(tmp_path / "data.csv")
    assert (tmp_path / "fig.png").stat().st_size > 1000
    assert (tmp_path / "data.csv").read_text().splitlines()[0] == "pass,energy,B"
    select(panel, "sgram_x")
    panel.export_data(tmp_path / "sgram.npz")
    with np.load(tmp_path / "sgram.npz") as f:
        assert f["sgram_x"].ndim == 2 and int(f["pass"]) == 3
    with pytest.raises(ValueError, match="1D"):
        panel.export_data(tmp_path / "sgram.csv")


def test_csv_pads_shorter_columns(tmp_path):
    write_csv(tmp_path / "x.csv", {"a": [1, 2, 3], "b": [4], "c": 5.0})
    assert (tmp_path / "x.csv").read_text().splitlines() == ["a,b,c", "1,4,5.0", "2,,", "3,,"]


def test_broken_figure_is_reported_not_raised(panel, saved, monkeypatch):
    _, path = saved

    def broken(r, fig, o):
        raise RuntimeError("no luck")

    PLOTS.register(PlotSpec("zz_broken", "Broken", broken, ("pass",), order=999))
    try:
        panel.open_file(path)
        select(panel, "zz_broken")
        assert "no luck" in panel.message.text()
    finally:
        PLOTS.unregister("zz_broken")


def test_unreadable_file_is_refused(panel, tmp_path, monkeypatch):
    warnings = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda *a: warnings.append(a))
    (tmp_path / "bad.npz").write_text("not a zip")
    assert not panel.open_file(tmp_path / "bad.npz") and warnings
