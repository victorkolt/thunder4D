# Benchmarks

Comparisons of thunder4d (lens-frame solver, `LensFrameSimulation`) with published gas-filled MPC
experiments. Every benchmark is a standalone script built on `framework.py`, so all of them are run,
stored and analysed the same way.

| script | paper | regime |
|---|---|---|
| `bench_grebing2020_1mJ.py` | Grebing *et al.*, Opt. Lett. 45, 6250 (2020) | 1 mJ, 200 fs, Ar, P/Pc ≈ 0.2 |
| `bench_kaumanns2018_18mJ.py` | Kaumanns *et al.*, Opt. Lett. 43, 5877 (2018) | 18 mJ, 1.3 ps, Ar, P/Pc ≈ 0.5, ~55 rad |
| `bench_rueda2021_8fs.py` | Rueda *et al.*, Opt. Express 29, 27004 (2021) | 0.29 mJ, 42 fs → 8 fs, Ar, 11 passes, P/Pc ≈ 0.5 (effective energy) |

## Running

```bash
cd benchmark
python bench_grebing2020_1mJ.py --quick          # 2 passes, coarse grid: checks that it runs (no paper check)
python bench_grebing2020_1mJ.py                  # full run, finest grid preset that fits in memory
python bench_grebing2020_1mJ.py -h               # all options
```

Options common to every benchmark:

| option | meaning |
|---|---|
| `--device auto\|cpu\|gpu`, `--gpu-id 1` | device; `--gpu-id` sets `CUDA_VISIBLE_DEVICES` (one benchmark per A100) |
| `--kernels auto\|numba\|numpy` | CPU array kernels between the FFTs (`auto`: Numba if installed); a forced value gets its own run directory, so `numba` and `numpy` runs can be timed side by side |
| `--fft auto\|scipy\|fftw` | CPU FFT library (`auto` = scipy; `fftw` needs pyFFTW and plans the transforms during the first steps); a forced value gets its own run directory |
| `--preset i` | force grid preset *i* (0 = finest); default: finest that fits (`--budget-gb` on the CPU, 80 % of free GPU memory) |
| `--phi-max`, `--gouy-step` | step control of the lens-frame solver (convergence checks) |
| `--set key=value` | change any entry of the benchmark's `params` (repeatable), e.g. `--set n2_1bar=1.1e-23` |
| `--label name` | prefix of the run directory |
| `--max-seconds S` | stop cleanly after S seconds; the same command resumes from the checkpoint |
| `--record-every n`, `--save-field`, `--no-resume`, `--out` | recording, final field in `run.npz`, fresh start, results root |

### Timing the CPU kernels

`--kernels numba` / `--kernels numpy` and `--fft scipy` / `--fft fftw` give separate run
directories for the same benchmark, so the same command with each value compares them. Compare `ms_per_step` (printed at the end and stored
in `results.json`): it times the propagation alone, without the diagnostics. `--label` keeps the
runs apart from earlier ones; `--no-resume` restarts a run that already exists.

```bash
cd benchmark
python bench_rueda2021_8fs.py --device cpu --kernels numba --quick --label jit   # compiles the kernels once (cached)
python bench_rueda2021_8fs.py --device cpu --kernels numpy --max-seconds 600 --label timing
python bench_rueda2021_8fs.py --device cpu --kernels numba --max-seconds 600 --label timing
python bench_rueda2021_8fs.py --device cpu --kernels numba --fft fftw --max-seconds 600 --label timing
```

With `--fft fftw`, the first steps include the FFTW planning (a few seconds per grid); over a
600 s run it is negligible.

The Numba threads follow `NUMBA_NUM_THREADS` (default: all cores, printed at the start of the run).

## Output

Each run goes to `results/<benchmark>/<label>_<fingerprint>/`. The fingerprint hashes the
parameters, the grid preset and the numerical settings: a different setting is a different run
directory, nothing is ever overwritten, and re-running the same command resumes or reuses the run.

| file | content |
|---|---|
| `results.json` | `run` (date, thunder4d version, git commit, device, kernels, cpu_count, grid, timing, settings; `propagation_s` and `ms_per_step` time the propagation alone, without the diagnostics, for this session of the run), `params`, `overrides`, `metrics` |
| `curves.npz` | standard curves (names below) used by `analyze.py` |
| `run.npz` | `sim.save()`: full per-pass history and axes |
| `checkpoint.pkl` | simulation state, for resuming |

Each metric is stored as

```json
"ftl_fs": {"value": 27.4, "unit": "fs", "label": "Fourier limit (integrated spectrum)",
           "ref": {"value": 27, "tol": 0.1, "kind": "rel", "note": ""},
           "status": "ok", "deviation": 0.015}
```

`kind` is `rel` (|v − ref| ≤ tol·|ref|), `abs` (|v − ref| ≤ tol), `max` (v ≤ ref, e.g. M² < 1.15),
`min` (v ≥ ref, e.g. V > 99 %) or `info` (shown, not checked). The tolerances are judgement calls
and live in each script's `reference` dict.

**Standard metrics** (computed identically for every benchmark): `w0_um`, `w_mirror_um`,
`w0_injected_um`, `P_over_Pc`, `B_estimate_rad`, `transmission`, `B_rad`, `I_peak_max_Wcm2` (exact,
cell centre), `F_mirror_max_Jcm2` (physical, from the lens-frame mirror records), `edge_fraction`
(numerical check: energy near the box absorber), `eta00_final`, `M2x`, `M2y`, `w_breathing`,
`span20_nm`, `rms_width_nm`, `lambda_centroid_nm`, `ftl_fs`, `comp_fs` and `comp_peak_ratio` (with the
paper's compressor), `best_gdd_fs2`, `best_comp_fs`, `best_peak_ratio`, `best_peak_power_GW`,
`V_weighted_x`, `V_min_fwhm`, `V_min_e2`, and `similarity_measured` if a digitised spectrum exists.

**Standard curves** in `curves.npz`: `pass`, `energy`, `B`, `eta00_total`, `V_near`, `V_far`,
`M2x_mean`, `M2y_mean`, `peak_power`, `I_peak`, `w_peak_x_um`, `w_peak_y_um`; `lam_nm`, `S_in`,
`S_out`; `t_tl_fs`/`P_tl`, `t_comp_fs`/`P_comp`, `t_best_fs`/`P_best`; `theta_mrad`, `lam_theta_nm`,
`S_theta`, `V_theta`; `mirror_x_mm`, `mirror_F_Jcm2`; optional `S_meas_lam_nm`/`S_meas` (measured) and
`S_ref_lam_nm`/`S_ref`/`S_ref_label` (a model from the paper).

## Analysing

```bash
python analyze.py                                    # all runs, tables against the papers
python analyze.py kaumanns2018_18mJ --all-metrics    # one benchmark, every metric
python analyze.py --latest --csv summary.csv         # one row per metric, for a spreadsheet
python analyze.py --latest --plot                    # summary.png in each run directory
python analyze.py --compare run_1d6084c5 phi01_8a2c11f0   # two runs side by side (convergence)
python analyze.py --sweep rueda2021_8fs energy ftl_fs comp_fs broadening_1e --plot
```

Quick runs are ignored unless `--quick` is given.

### Typical checks

```bash
# convergence of one benchmark: step size and grid
python bench_grebing2020_1mJ.py --label ref
python bench_grebing2020_1mJ.py --label phi01 --phi-max 0.01
python bench_grebing2020_1mJ.py --label fine  --preset 0 --gouy-step 0.025
python analyze.py --compare <ref run id> <phi01 run id>

# n2 uncertainty (±10 %)
python bench_kaumanns2018_18mJ.py --label n2lo --set n2_1bar=0.9e-23
python bench_kaumanns2018_18mJ.py --label n2hi --set n2_1bar=1.1e-23

# energy scan of Rueda: effective energy factor of the authors' simulation (0.5) up to nominal
for e in 1.0e-4 1.45e-4 1.9e-4 2.4e-4 2.9e-4; do
  python bench_rueda2021_8fs.py --set energy=$e --label e$e
done
python analyze.py --sweep rueda2021_8fs energy ftl_fs comp_fs broadening_1e I_peak_max_Wcm2 --plot
```

## Measured data

Curves digitised from the papers live in `data/`, one CSV per curve, named
`<author><year>_<quantity>.csv` and listed in each script's `measured` dict. Two comma-separated
columns, no header; lines starting with `#` give the source (paper, figure, curve), the units and the
date of digitisation:

```
# Rueda et al., Opt. Express 29, 27004 (2021), doi:10.1364/OE.435086 (open access)
# Fig. 3a, output spectrum after the MPC (solid blue), digitised 2026-11-01
# columns: wavelength (nm), intensity (linear, normalised to 1), as digitised (no baseline removed)
588.18,0.00289
```

| key in `measured` | columns | used for |
|---|---|---|
| `spectrum` | wavelength (nm), linear intensity | overlay, `similarity_measured`, `ftl_measured_fs` |
| `pulse` | time (fs), power | overlay on the compressed pulses, `fwhm_measured_pulse_fs` |

Spectra must be linear (convert log-scale figures before saving). Missing files are simply skipped.

## Reference runs and report

`results/` is not committed. The run that represents a benchmark is copied to `reference/`, which is:

```bash
python analyze.py --promote <run id> [<run id> ...]   # reference/<benchmark>/results.json + curves.npz
python analyze.py --report                            # reference/REPORT.md, deviations.png, */summary.png
```

`--promote` refuses unfinished runs and quick runs (`--force` to override) and replaces the previous
reference of that benchmark. `--report` re-checks the stored metrics against the `reference` values of
the current scripts, so changing a tolerance does not need a new run. The report contains a summary
table of all benchmarks, a figure of the deviation of every checked metric from the paper with its
tolerance, and one section per benchmark (run details, calibrated parameters, standard figure, every
metric with a reference). Commit `reference/` after running both commands.

## Tests

`tests/test_benchmarks.py` (run by `pytest` in CI) reruns the quick case of every benchmark on the CPU,
a few seconds each, and checks that:

- the run finishes, with finite outputs, no energy gain, the field inside the box and M² ≥ 1;
- the main physical outputs (transmission, B-integral, spectral widths, Fourier limit, M², homogeneity,
  ...) are within 0.1 % of the snapshots in `reference/quick/`;
- the framework helpers, `--promote` and `--report` work, and every committed reference run has its
  `curves.npz`.

Quick runs are not compared with the papers (all their metrics are `info`): the snapshots only detect
that a change of the code changed the results. When a change is intended (new physics, better
numerics), update the snapshots and say so in the commit message:

```bash
python analyze.py --update-quick            # all benchmarks, or give their names
```

A new benchmark needs its snapshot too (`python analyze.py --update-quick <name>`), otherwise the
tests fail.

## Adding a benchmark

Create `bench_<author><year>_<case>.py`; only the physics goes in it:

```python
import thunder4d as th
from framework import Benchmark, Ref, band_mirror

class Smith2027(Benchmark):
    name = "smith2027_10mJ"                    # results/<name>/, key for analyze.py
    title = "10 mJ, 300 fs, Kr (Smith et al., 2027)"
    paper = {"citation": "J. Smith et al., ...", "doi": "..."}
    params = dict(lambda0=1030e-9, energy=10e-3, fwhm=300e-15, gas="Kr", pressure_bar=0.5,
                  n2_1bar=2.8e-23, R=2.0, d=3.9, n_passes=30, compressor_gdd=-3000.0,
                  lambda_min=950e-9, lambda_max=1120e-9, injection="linear")
    reference = {"ftl_fs": Ref(40, 0.1), "transmission": Ref(0.95, 0.02, kind="abs"),
                 "M2x": Ref(1.2, kind="max")}
    measured = {"spectrum": "data/smith2027_spectrum.csv"}     # optional, see "Measured data"
    calibrated = {"energy": {"paper": 12e-3, "note": "why"}}    # only if a parameter is not the paper's

    def make_cell(self, P, n_passes):
        gas = th.Material(P["gas"], P["pressure_bar"], n2_1bar=P["n2_1bar"])
        return th.MPC(R=P["R"], d=P["d"], medium=gas, n_passes=n_passes)

if __name__ == "__main__":
    Smith2027().main()
```

Required `params`: `lambda0`, `energy`, `fwhm`, `n_passes`, `lambda_min`, `lambda_max`. Optional:
`compressor_gdd`, `gdd_fs2` / `tod_fs3` (input phase), `injection` (`"linear"` or `"nonlinear"`),
`window_fwhm` (time window when the output is strongly chirped).

Hooks to override when needed (all optional except `make_cell`):

| hook / attribute | default |
|---|---|
| `presets`, `quick_preset`, `n_fields` | grid presets for `suggest_lens_grid`, finest first; memory estimate |
| `record_every`, `compression_up`, `gdd_bounds` | diagnostics cadence; time oversampling; best-GDD search range |
| `grid_carrier(P)` | `lambda0` (Daniault puts it at the centre of a very wide window) |
| `spectrum(grid, P)` | Gaussian at `lambda0` with optional GDD/TOD |
| `pulse(ctx)` | Gaussian at the linear or Kerr-matched waist, `Pulse.at_focus` |
| `extra_metrics(ctx)` | returns `(metrics, curves)`; same names as standard ones override them |

Inside `extra_metrics`, build values with `metric(value, unit, label, ref)` and use the helpers of
`framework.py` (`beam_pixels`, `integrated_power`, `fourier_limit`, `best_phase`, `spectral_phase`,
`fwhm`, `span`, `rms_width`, `homogeneity_1d`, `spectral_similarity`). `ctx` holds `P`, `cell`,
`grid`, `sim`, `lin`, `mm`, `w0_ref`, `run_dir`.
