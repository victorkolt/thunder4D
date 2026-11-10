"""Tests of the benchmark suite (benchmark/): regression on the quick cases, sanity of the runs, and
unit tests of the framework and of analyze.py.

The regression test reruns the quick case of every benchmark (2 passes on a coarse grid, a few seconds
on a CPU) and compares a set of physical outputs with the snapshots in benchmark/reference/quick/.
A failure means that a change of the code changed the physics or the numerics. If the change is
intended, update the snapshots and say so in the commit:

    cd benchmark && python analyze.py --update-quick
"""
import importlib
import json
import os
import sys

import numpy as np
import pytest

BENCH_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "benchmark")
RTOL = 1e-3          # quick runs are deterministic; this only absorbs FFT / BLAS rounding differences


def _load(name):
    """Import benchmark/<name>.py (benchmark/ is a folder of scripts, not a package)."""
    if BENCH_DIR not in sys.path:
        sys.path.insert(0, BENCH_DIR)
    return importlib.import_module(name)


fw = _load("framework")
an = _load("analyze")
BENCHMARKS = an.bench_classes()


@pytest.fixture(scope="module")
def quick_results(tmp_path_factory):
    """One quick run per benchmark, shared by the tests of this module."""
    out = str(tmp_path_factory.mktemp("results"))
    return {name: an.run_quick(cls, out) for name, cls in BENCHMARKS.items()}, out


# ---------------------------------------------------------------- quick runs
def test_every_benchmark_has_a_quick_snapshot():
    missing = [n for n in BENCHMARKS if not os.path.exists(os.path.join(an.QUICK_ROOT, f"{n}.json"))]
    assert not missing, f"no snapshot for {missing}: run `python analyze.py --update-quick {' '.join(missing)}`"


@pytest.mark.parametrize("name", sorted(BENCHMARKS))
def test_quick_run_is_sane(quick_results, name):
    r = quick_results[0][name]
    assert r["run"]["finished"]
    M = {k: m["value"] for k, m in r["metrics"].items()}
    for k, v in M.items():
        assert v is None or isinstance(v, str) or np.isfinite(v), f"{k} is not finite"
    assert 0 < M["transmission"] <= 1 + 1e-9                     # passive cell: no energy gain
    # field inside the box: the small quick box holds ~1e-3 of Gaussian tail; a blow-up gives > 1e-2
    assert M["edge_fraction"] < 5e-3
    assert M["M2x"] >= 1 - 1e-3 and M["M2y"] >= 1 - 1e-3          # M2 cannot be below 1
    # quick runs are not compared with the papers
    assert all(m["status"] in ("info", "nan") for m in r["metrics"].values())


@pytest.mark.parametrize("name", sorted(BENCHMARKS))
def test_quick_run_matches_snapshot(quick_results, name):
    path = os.path.join(an.QUICK_ROOT, f"{name}.json")
    if not os.path.exists(path):
        pytest.skip("no snapshot (see test_every_benchmark_has_a_quick_snapshot)")
    with open(path) as f:
        ref = json.load(f)
    new = an.quick_snapshot(quick_results[0][name])
    assert new["grid_shape"] == ref["grid_shape"], "the quick grid changed"
    changed = {k: (v, new["metrics"].get(k)) for k, v in ref["metrics"].items()
               if not np.isclose(new["metrics"].get(k, np.nan), v, rtol=RTOL, atol=0)}
    assert not changed, ("quick-run outputs changed (reference, now): "
                         + ", ".join(f"{k} {a:.6g} -> {b}" for k, (a, b) in changed.items())
                         + ". If intended: python analyze.py --update-quick")


# ---------------------------------------------------------------- framework units
def test_ref_status_and_deviation():
    assert fw.Ref(10, 0.1).status(10.5) == "ok"
    assert fw.Ref(10, 0.1).status(11.5) == "off"
    assert fw.Ref(10, 0.1).deviation(11) == pytest.approx(0.1)
    assert fw.Ref(0.96, 0.03, kind="abs").status(0.98) == "ok"
    assert fw.Ref(0.96, 0.03, kind="abs").status(0.92) == "off"
    assert fw.Ref(1.15, kind="max").status(1.1) == "ok"
    assert fw.Ref(1.15, kind="max").status(1.2) == "off"
    assert fw.Ref(6, kind="min").status(7) == "ok"
    assert fw.Ref(6, kind="min").status(5) == "off"
    assert fw.Ref(13, kind="info").status(25) == "info"


def test_fwhm_of_a_gaussian():
    x = np.linspace(-10, 10, 4001)
    assert fw.fwhm(x, np.exp(-4 * np.log(2) * (x / 3.0) ** 2)) == pytest.approx(3.0, rel=1e-4)


def test_fourier_limit_of_a_gaussian_spectrum():
    """Gaussian in frequency, given per unit wavelength: the Fourier limit is exact (TBP 0.441)."""
    c = 299.792458                                                 # nm/fs
    lam = np.linspace(700, 900, 4001)
    w, w0, dw = 2 * np.pi * c / lam, 2 * np.pi * c / 800, 0.05    # rad/fs, FWHM of the power spectrum
    S_lam = np.exp(-4 * np.log(2) * ((w - w0) / dw) ** 2) / lam ** 2
    assert fw.fourier_limit_of_spectrum(lam, S_lam) == pytest.approx(4 * np.log(2) / dw, rel=1e-3)


def test_load_digitized(tmp_path):
    p = tmp_path / "curve.csv"
    p.write_text("# source line\n# columns: x, y\n2.0,4.0\n1.0,2.0\n3.0,-1.0\n")
    x, y = fw.load_digitized(str(p))
    assert list(x) == [1.0, 2.0, 3.0]                              # sorted along x
    assert y.max() == pytest.approx(1.0)
    _, y_raw = fw.load_digitized(str(p), normalise=False)
    assert sorted(y_raw) == [-1.0, 2.0, 4.0]                       # phases keep their values
    assert fw.load_digitized(str(tmp_path / "missing.csv")) is None


# ---------------------------------------------------------------- analyze.py
def test_promote_and_report(quick_results, tmp_path, monkeypatch):
    results, root = quick_results
    monkeypatch.setattr(an, "REF_ROOT", str(tmp_path / "reference"))
    ids = [r["run"]["id"] for r in results.values()]
    with pytest.raises(SystemExit):                                # quick runs need --force
        an.promote(root, ids[:1])
    an.promote(root, ids, force=True)
    runs = an.load_reference()
    assert sorted(r["benchmark"] for r in runs) == sorted(BENCHMARKS)
    an.write_report(runs, out_dir=an.REF_ROOT)
    report = (tmp_path / "reference" / "REPORT.md").read_text()
    assert all(f"## {n}" in report for n in BENCHMARKS)
    assert (tmp_path / "reference" / "deviations.png").exists()
    for n in BENCHMARKS:
        assert (tmp_path / "reference" / n / "summary.png").exists()


def test_committed_reference_runs_are_complete():
    """Every committed reference run has its curves (needed to regenerate the figures and the docs)."""
    for d in sorted(os.listdir(an.REF_ROOT)):
        path = os.path.join(an.REF_ROOT, d)
        if d == "quick" or not os.path.isdir(path):
            continue
        assert os.path.exists(os.path.join(path, "results.json")), d
        assert os.path.exists(os.path.join(path, "curves.npz")), f"{d}: curves.npz not committed"


def test_peak_intensity_of_a_run_resumed_from_an_old_checkpoint():
    """Old checkpoints hold the first records as I_peak; resume() fills peak_intensity with None there."""
    fw = _load("framework")
    h = {"I_peak": [3.0, 2.0], "peak_intensity": [None, None, 1.0, 0.5]}
    np.testing.assert_array_equal(fw._peak_intensity(h), [3.0, 2.0, 1.0, 0.5])
    np.testing.assert_array_equal(fw._peak_intensity({"peak_intensity": [1.0]}), [1.0])
