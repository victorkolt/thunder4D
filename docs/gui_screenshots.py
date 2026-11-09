"""Screenshots of the GUI tutorial (docs/source/gui/gui.rst), taken offscreen.

    uv run python docs/gui_screenshots.py        # about a minute; writes docs/source/_static/figures/gui_*.png

The tutorial scenario: design a cell for a PHAROS 1 mJ / 190 fs laser, send the best design to the
simulation builder, run it, look at the results.
"""
import os
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from thunder4d.gui._qt import QtWidgets

OUT = Path(__file__).parent / "source" / "_static" / "figures"
SIZE = (1280, 800)


def wait(app, panel, timeout=600):
    t0 = time.time()
    while panel.task is not None and time.time() - t0 < timeout:
        app.processEvents()
        time.sleep(0.02)
    app.processEvents()


def shot(app, window, name):
    app.processEvents()
    window.grab().save(str(OUT / f"gui_{name}.png"))
    print("wrote", OUT / f"gui_{name}.png")


def main():
    app = QtWidgets.QApplication(sys.argv[:1])
    from thunder4d.gui.app import MainWindow

    window = MainWindow()
    window.resize(*SIZE)
    window.show()

    # 1) design helper
    window.show_panel("design")
    design = window.panels["design"]
    design.preset.setCurrentText("pharos_ph2_sp_1mJ")
    for path, value in (("constraints.gas.allowed", ["Ar", "Kr"]), ("search.round_trips", [8, 20]),
                        ("refinement.n_candidates", 3), ("constraints.space.max_separation_m.value", 2.0)):
        design.editors[path].set_value(value)
        design.editors[path].changed.emit()
    design.start()
    wait(app, design)
    shot(app, window, "design")
    design.compute_map()
    wait(app, design)
    shot(app, window, "design_map")

    # 2) builder, filled from the best design
    design.use_in_builder()
    shot(app, window, "builder")

    # 3) run
    run = window.panels["run"]
    window.show_panel("run")
    folder = Path(tempfile.mkdtemp())
    run.start_run(dict(window.session.config), folder / "tutorial.npz")
    wait(app, run)
    run.output.setText("tutorial.npz")          # no local path in the documentation
    shot(app, window, "run")

    # 4) results
    window.show_panel("results")
    results = window.panels["results"]
    for key in ("spectra", "pulse"):
        root = results.list.invisibleRootItem()
        items = [root.child(i).child(j) for i in range(root.childCount()) for j in range(root.child(i).childCount())]
        results.list.setCurrentItem(next(it for it in items if it.data(0, 256) == key))
        shot(app, window, f"results_{key}")


if __name__ == "__main__":
    main()
