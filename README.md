# thunder4D

[![Tests](https://github.com/victorkolt/thunder4D/actions/workflows/tests.yml/badge.svg)](https://github.com/victorkolt/thunder4D/actions/workflows/tests.yml)
[![Docs](https://github.com/victorkolt/thunder4D/actions/workflows/docs.yml/badge.svg)](https://victorkolt.github.io/thunder4D)

**T**ransverse-resolved **H**erriott-cell **U**PPE solver for **N**onlinear **D**ynamics close to **E**xperimental
**R**eality, **4D**: a 4D (x, y, z, t) UPPE solver for multipass cells, with a design helper that goes from a laser
to a ready-to-run cell, and a graphical interface.

**Documentation:** https://victorkolt.github.io/thunder4D

## Install

```bash
pip install thunder4d               # or: uv add thunder4d
pip install "thunder4d[gpu]"        # NVIDIA GPU (CuPy)
pip install "thunder4d[gui]"        # graphical interface: thunder4d-gui
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

Against published experiments, on the benchmark grids. Values are *simulation (paper)*.

| experiment | compressed | transmission | run time (CPU) |
|---|---|---|---|
| [Grebing 2020](https://doi.org/10.1364/OL.408998), 1 mJ, 200 fs, Ar | 26.9 fs (31 fs) | 0.984 (0.96) | 1 min |
| [Kaumanns 2018](https://doi.org/10.1364/OL.43.005877), 18 mJ, 1.3 ps, Ar | 46.9 fs (41 fs) | 0.955 (0.957) | 16 min |
| [Rueda 2021](https://doi.org/10.1364/OE.435086), 0.29 mJ, 42 fs, Ar | 6.9 fs (8.1 fs) | 0.715 (0.737) | 6 min |

Details in the [benchmark report](benchmark/reference/REPORT.md).

## License

MIT
