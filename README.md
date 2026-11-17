# thunder4D

[![Tests](https://github.com/victorkolt/thunder4D/actions/workflows/tests.yml/badge.svg)](https://github.com/victorkolt/thunder4D/actions/workflows/tests.yml)
[![Docs](https://github.com/victorkolt/thunder4D/actions/workflows/docs.yml/badge.svg)](https://victorkolt.github.io/thunder4D)

**T**ransverse-resolved **H**erriott-cell **U**PPE solver for **N**onlinear **D**ynamics close to **E**xperimental
**R**eality, **4D**: a 4D (x, y, z, t) UPPE solver for multipass cells, gas-filled, bulk (solid plates) or hybrid,
with a design helper that goes from a laser to a ready-to-run gas cell, and a graphical interface.

**Documentation:** https://victorkolt.github.io/thunder4D

## Install

```bash
pip install thunder4d               # or: uv add thunder4d
pip install "thunder4d[gpu]"        # NVIDIA GPU (CuPy)
pip install "thunder4d[gui]"        # graphical interface: thunder4d-gui
pip install "thunder4d[numba]"      # faster CPU runs (Numba kernels)
pip install "thunder4d[fftw]"       # FFTW transforms on the CPU (fft_backend="fftw")
```

## Example

1 mJ, 200 fs at 1030 nm through 26 passes of a cell filled with 0.7 bar of argon, then the best uniform compression.

```python
import thunder4d as th
from thunder4d import diagnostics as dg

cfg = th.default_input(
    laser={"energy_J": 1e-3, "duration_fs": 200},
    medium={"gas": "Ar", "pressure_bar": 0.7},
    cell={"roc_m": 0.6, "separation_m": 1.183, "n_passes": 26},
    grid={"lambda_min_nm": 900, "lambda_max_nm": 1180},
)
b = th.run_input(cfg)                              # coarse grid: a few seconds on a laptop
print(dg.compress(b.grid, b.sim.Aw)["fwhm_fs"])    # 23.6 fs
```

## Validation

Against published experiments, on the benchmark grids. Values are *simulation (paper)*. Where the paper gives no
compressed duration, the value is the Fourier limit (FTL).

| experiment | compressed | transmission | run time (CPU) |
|---|---|---|---|
| [Grebing 2020](https://doi.org/10.1364/OL.408998), 1 mJ, 200 fs, Ar | 26.9 fs (31 fs) | 0.984 (0.96) | 1 min |
| [Kaumanns 2018](https://doi.org/10.1364/OL.43.005877), 18 mJ, 1.3 ps, Ar | 46.9 fs (41 fs) | 0.955 (0.957) | 16 min |
| [Rueda 2021](https://doi.org/10.1364/OE.435086), 0.29 mJ, 42 fs, Ar | 6.9 fs (8.1 fs) | 0.715 (0.737) | 6 min |
| [Westerberg 2025](https://doi.org/10.1063/5.0272968), 250 µJ, 180 fs, bulk, 2 × 1 mm FS | FTL 41.4 fs (39 fs) | 0.959 (> 0.94) | 1 min |
| [Koltalo 2025](https://doi.org/10.1063/5.0253348), 200 µJ, 180 fs, bulk, Gaussian beam | 49.9 fs (45 fs) | 0.959 (> 0.90) | 1 min |
| [Koltalo 2025](https://doi.org/10.1063/5.0253348), 350 to 610 µJ, 180 fs, bulk, LG<sub>0,1</sub> to LG<sub>0,3</sub> vortex | 48.4, 48.3, 45.9 fs (45 fs) | 0.958 to 0.959 (> 0.90) | 1 to 3 min |
| [Omar 2024](https://doi.org/10.1364/OE.513732), 400 µJ, 276 fs, hybrid air and FS | 28.0 fs (27 fs) | 0.954 (> 0.91) | 2 min |
| [Hu 2025](https://arxiv.org/abs/2511.08337), 0.5 µJ, 120 fs, bulk, oscillator | FTL 8.6 fs (8.1 fs) | 0.718 (0.750) | 9 min |
| [Hu 2025](https://arxiv.org/abs/2511.08337), 0.37 µJ, 100 fs, bulk, off-centre plate | FTL 8.1 fs (10.3 fs) | 0.760 | 14 min |

Details, and the calibrations of the bulk and hybrid runs, in the [benchmark report](benchmark/reference/REPORT.md).

## License

MIT
