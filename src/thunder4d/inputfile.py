"""Standard THUNDER4D input file (JSON).

One file describes one simulation completely (laser, gas, cell, mirrors, grid, solver, run), so that
anything built by the design helper, the GUI or a script can be re-run from the file alone:

    cfg = th.load_input("cell.json")         # defaults filled in, checked
    sim = th.run_input(cfg)                  # build + run (lens-frame solver)
    th.save_input(cfg, "copy.json")

    b = th.build_simulation(cfg)              # or step by step: b.grid, b.cell, b.pulse, b.sim
    b.sim.run()

Units are in the key names (_J, _fs, _nm, _m, _bar, _K, ...). Tables (measured spectrum, mirror
reflectivity and GDD curves) are stored inline as {"wavelength_nm": [...], "value": [...]}, so a
file is self-contained. Unknown keys are rejected (typos would otherwise be silently ignored), except
in the free-form "design" and "notes" blocks.

Pass convention: the simulation starts right after a mirror and one pass = transit over d + bounce.
"""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = "thunder4d-input"
SCHEMA_VERSION = 1

# Grid presets of suggest_lens_grid: "coarse" = the benchmarks' quick grid (minutes),
# "fine" = their finest preset (benchmark grade).
GRID_PRESETS = {
    "coarse": {"box_factor": 5.0, "points_per_waist": 3.0, "time_window_factor": 4.0},
    "medium": {"box_factor": 6.0, "points_per_waist": 6.0, "time_window_factor": 6.0},
    "fine": {"box_factor": 8.0, "points_per_waist": 8.0, "time_window_factor": 8.0},
}

DEFAULTS: dict[str, Any] = {
    "schema": SCHEMA,
    "version": SCHEMA_VERSION,
    "name": "mpc",
    "notes": "",
    "laser": {
        "energy_J": None,                 # required
        "duration_fs": None,              # required: transform-limited FWHM (intensity)
        "wavelength_nm": 1030.0,
        "shape": "gaussian",              # "gaussian" | "measured" (spectrum below)
        "spectrum": None,                 # {"wavelength_nm": [...], "intensity": [...], "phase_rad": [...]}
        "gdd_fs2": 0.0,                   # spectral phase added to the input
        "tod_fs3": 0.0,
    },
    "beam": {
        "injection": "linear",            # "linear" | "nonlinear" (Kerr-matched) | "waist" (w0_m below)
        "w0_m": None,
    },
    "medium": {
        "gas": None,                      # required: a gas of thunder4d.materials
        "pressure_bar": None,             # required
        "temperature_K": 293.15,
        "n2_1bar": None,                  # None -> thunder4d.materials
    },
    "cell": {
        "roc_m": None,                    # required
        "separation_m": None,             # required
        "n_passes": None,                 # required
        "herriott": None,                 # optional, information: {"round_trips", "k", "branch"}
    },
    "mirrors": {
        "reflectivity": 1.0,              # scalar or {"wavelength_nm": [...], "value": [...]}
        "gdd_fs2": 0.0,                   # scalar or table, per bounce
        "tod_fs3": 0.0,                   # per bounce (scalar)
        "diameter_m": None,               # clear aperture, None = infinite
        "coating": None,                  # information: name in the design catalogue
    },
    "grid": {
        "preset": "coarse",               # "coarse" | "medium" | "fine"; explicit values below win
        "box_factor": None,
        "points_per_waist": None,
        "time_window_factor": None,
        "lambda_min_nm": None,            # required: spectral window of the grid
        "lambda_max_nm": None,
        "precision": "single",            # "single" | "double"
        "device": "auto",                 # "auto" | "cpu" | "gpu"
        "fft_backend": "auto",            # "auto" (= scipy) | "scipy" | "fftw" (pyFFTW, CPU)
    },
    "solver": {
        "phi_max": 0.02,
        "gouy_step": 0.05,
        "self_steepening": True,
        "kernels": "auto",                # "auto" | "numba" | "numpy" (CPU array kernels)
    },
    "run": {
        "record_every": 1,
        "near_field": True,
        "record_mirror": True,
    },
    "design": None,                       # free-form: provenance written by the design helper
}

REQUIRED = [("laser", "energy_J"), ("laser", "duration_fs"), ("medium", "gas"), ("medium", "pressure_bar"),
            ("cell", "roc_m"), ("cell", "separation_m"), ("cell", "n_passes"),
            ("grid", "lambda_min_nm"), ("grid", "lambda_max_nm")]
FREE_FORM = ("design", "notes")


# ============================================================ field descriptions
@dataclass(frozen=True)
class Field:
    """How one key of the input file is shown and edited (GUI forms, documentation).

    The file stores values in the unit of the key name; the form shows them in `unit`, with
    shown value = file value / scale (e.g. energy_J shown in mJ: scale = 1e-3).

    kind: float | int | bool | choice | text | multiline | table | float_or_table | info | hidden
          (the GUI also has float_list and multichoice, for forms other than the input file)
          table: {column: [...]} with the keys `columns` (the first one is the wavelength);
          float_or_table: a scalar, or a table as above; info: shown, not edited (design provenance);
          hidden: kept as is, not shown.
    optional: None is allowed and means `none_text` (e.g. "from the preset"); required keys are in REQUIRED.
    show_if: (path, values): shown only when the key at `path` has one of `values`.
    """
    label: str
    kind: str = "float"
    unit: str = ""
    scale: float = 1.0
    choices: tuple = ()
    columns: tuple[str, ...] = ()
    optional: bool = False
    none_text: str = ""
    help: str = ""
    show_if: tuple[str, tuple] | None = None
    advanced: bool = False


# Sections of the input, in display order ("general" = the top-level keys).
BLOCKS = {"general": "General", "laser": "Laser", "beam": "Beam injection", "medium": "Gas", "cell": "Cell",
          "mirrors": "Mirrors", "grid": "Grid", "solver": "Solver", "run": "Run"}


def _gases():
    from .materials import SELLMEIER
    return tuple(SELLMEIER)


# Every key of DEFAULTS is described here (tests/test_inputfile.py checks it): a new key of the input file
# gets its Field in the same change, and the GUI builder shows it with no other edit.
FIELDS: dict[str, Field] = {
    "schema": Field("Schema", "hidden"),
    "version": Field("Schema version", "hidden"),
    "name": Field("Name", "text", help="Name of the simulation (used for the output files)."),
    "notes": Field("Notes", "multiline", help="Free text."),
    "design": Field("Design provenance", "hidden"),
    # laser
    "laser.energy_J": Field("Pulse energy", unit="mJ", scale=1e-3),
    "laser.duration_fs": Field("Duration (FWHM)", unit="fs", help="Transform-limited FWHM of the intensity."),
    "laser.wavelength_nm": Field("Central wavelength", unit="nm"),
    "laser.shape": Field("Spectrum", "choice", choices=("gaussian", "measured"),
                         help="gaussian: from the duration; measured: the table below."),
    "laser.spectrum": Field("Measured spectrum", "table", columns=("wavelength_nm", "intensity", "phase_rad"),
                            show_if=("laser.shape", ("measured",)),
                            help="Intensity per wavelength; the phase column [rad] is optional."),
    "laser.gdd_fs2": Field("Input GDD", unit="fs²", help="Spectral phase added to the input pulse."),
    "laser.tod_fs3": Field("Input TOD", unit="fs³"),
    # beam
    "beam.injection": Field("Injected beam", "choice", choices=("linear", "nonlinear", "waist"),
                            help="linear: eigenmode of the empty cell; nonlinear: Kerr-matched eigenmode; "
                                 "waist: the waist below."),
    "beam.w0_m": Field("Waist at the centre", unit="µm", scale=1e-6, optional=True,
                       show_if=("beam.injection", ("waist",))),
    # medium
    "medium.gas": Field("Gas", "choice", choices=_gases()),
    "medium.pressure_bar": Field("Pressure", unit="bar"),
    "medium.temperature_K": Field("Temperature", unit="K"),
    "medium.n2_1bar": Field("n₂ at 1 bar", unit="m²/W", optional=True, none_text="from the gas table",
                            help="Kerr index at 1 bar and room temperature."),
    # cell
    "cell.roc_m": Field("Mirror radius of curvature", unit="m"),
    "cell.separation_m": Field("Mirror separation", unit="m"),
    "cell.n_passes": Field("Number of passes", "int", help="One pass = transit over the separation + bounce."),
    "cell.herriott": Field("Herriott pattern", "info", optional=True),
    # mirrors
    "mirrors.reflectivity": Field("Reflectivity", "float_or_table", columns=("wavelength_nm", "value"),
                                  help="Per bounce: a constant, or a table versus wavelength."),
    "mirrors.gdd_fs2": Field("GDD per bounce", "float_or_table", unit="fs²", columns=("wavelength_nm", "value")),
    "mirrors.tod_fs3": Field("TOD per bounce", unit="fs³"),
    "mirrors.diameter_m": Field("Clear aperture", unit="mm", scale=1e-3, optional=True, none_text="infinite"),
    "mirrors.coating": Field("Coating", "info", optional=True),
    # grid
    "grid.preset": Field("Resolution", "choice", choices=tuple(GRID_PRESETS),
                         help="coarse: minutes; fine: benchmark grade. The values below override it."),
    "grid.lambda_min_nm": Field("Shortest wavelength", unit="nm"),
    "grid.lambda_max_nm": Field("Longest wavelength", unit="nm"),
    "grid.box_factor": Field("Box / waist", optional=True, none_text="from the preset", advanced=True),
    "grid.points_per_waist": Field("Points per waist", optional=True, none_text="from the preset", advanced=True),
    "grid.time_window_factor": Field("Time window / duration", optional=True, none_text="from the preset",
                                     advanced=True),
    "grid.precision": Field("Precision", "choice", choices=("single", "double"), advanced=True),
    "grid.device": Field("Device", "choice", choices=("auto", "cpu", "gpu"), advanced=True),
    "grid.fft_backend": Field("FFT library", "choice", choices=("auto", "scipy", "numpy", "fftw"), advanced=True),
    # solver
    "solver.phi_max": Field("Max nonlinear phase per step", unit="rad", advanced=True),
    "solver.gouy_step": Field("Max Gouy phase per step", unit="rad", advanced=True),
    "solver.self_steepening": Field("Self-steepening", "bool", advanced=True),
    "solver.kernels": Field("CPU array kernels", "choice", choices=("auto", "numba", "numpy"), advanced=True),
    # run
    "run.record_every": Field("Record every", "int", unit="passes", advanced=True),
    "run.near_field": Field("Near-field diagnostics", "bool", advanced=True,
                            help="Angle-resolved spectrogram and near-field profiles (some seconds per record)."),
    "run.record_mirror": Field("Fluence on the mirrors", "bool", advanced=True),
}


def field_paths(defaults=DEFAULTS):
    """Every editable key of the input, as 'block.key' (or 'key' at the top level), in file order."""
    for key, val in defaults.items():
        if isinstance(val, dict) and key not in FREE_FORM:
            yield from (f"{key}.{sub}" for sub in val)
        else:
            yield key


def get_path(cfg, path):
    *block, key = path.split(".")
    return (cfg[block[0]] if block else cfg).get(key)


def set_path(cfg, path, value):
    *block, key = path.split(".")
    (cfg[block[0]] if block else cfg)[key] = value


# ============================================================ dict helpers
def default_input(**blocks):
    """A complete input with defaults; keyword blocks are merged in, e.g.
    default_input(laser={"energy_J": 1e-3, "duration_fs": 200}, medium={"gas": "Ar", ...})."""
    return _merge(copy.deepcopy(DEFAULTS), blocks)


def _merge(base, extra, path=""):
    for key, val in (extra or {}).items():
        where = f"{path}.{key}" if path else key
        if key not in base and path.split(".")[0] not in FREE_FORM and key not in FREE_FORM:
            raise KeyError(f"unknown input key {where!r}")
        if isinstance(val, dict) and isinstance(base.get(key), dict) and key not in FREE_FORM:
            _merge(base[key], val, where)
        else:
            base[key] = val
    return base


def _jsonable(v):
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, dict):
        return {k: _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, float) and not np.isfinite(v):
        return None
    return v


def input_problems(cfg):
    """Every problem of a (merged) input, as (path of the key at fault, message); [] if it is valid."""
    errors = []
    if cfg.get("schema") != SCHEMA:
        errors.append(("schema", f"schema must be {SCHEMA!r}"))
    if cfg.get("version", 0) > SCHEMA_VERSION:
        errors.append(("version", f"file version {cfg['version']} is newer than this thunder4d ({SCHEMA_VERSION})"))
    for block, key in REQUIRED:
        if cfg[block].get(key) is None:
            errors.append((f"{block}.{key}", f"{block}.{key} is required"))
    c = cfg["cell"]
    if c["roc_m"] and c["separation_m"] and not 0 < c["separation_m"] < 2 * c["roc_m"]:
        errors.append(("cell.separation_m", "cell: unstable, need 0 < separation_m < 2 roc_m"))
    if c["n_passes"] is not None and int(c["n_passes"]) < 1:
        errors.append(("cell.n_passes", "cell.n_passes must be >= 1"))
    g = cfg["grid"]
    if g["preset"] not in GRID_PRESETS:
        errors.append(("grid.preset", f"grid.preset must be one of {sorted(GRID_PRESETS)}"))
    lam0 = cfg["laser"]["wavelength_nm"]
    if g["lambda_min_nm"] and g["lambda_max_nm"] and not g["lambda_min_nm"] < lam0 < g["lambda_max_nm"]:
        errors.append(("grid.lambda_min_nm", "grid: need lambda_min_nm < laser.wavelength_nm < lambda_max_nm"))
    if cfg["laser"]["shape"] == "measured" and not cfg["laser"]["spectrum"]:
        errors.append(("laser.spectrum", "laser.shape = 'measured' needs laser.spectrum"))
    from .kernels import KERNELS

    if cfg["grid"]["fft_backend"] not in ("auto", "scipy", "numpy", "fftw"):
        errors.append(("grid.fft_backend", "grid.fft_backend must be 'auto', 'scipy', 'numpy' or 'fftw'"))
    if cfg["solver"]["kernels"] not in KERNELS:
        errors.append(("solver.kernels", f"solver.kernels must be one of {list(KERNELS)}"))
    if cfg["beam"]["injection"] not in ("linear", "nonlinear", "waist"):
        errors.append(("beam.injection", "beam.injection must be 'linear', 'nonlinear' or 'waist'"))
    if cfg["beam"]["injection"] == "waist" and not cfg["beam"]["w0_m"]:
        errors.append(("beam.w0_m", "beam.injection = 'waist' needs beam.w0_m"))
    for key in ("reflectivity", "gdd_fs2"):
        t = cfg["mirrors"][key]
        if isinstance(t, dict) and not {"wavelength_nm", "value"} <= set(t):
            errors.append((f"mirrors.{key}", f"mirrors.{key}: a table needs 'wavelength_nm' and 'value'"))
    return errors


def check_input(cfg):
    """Raise ValueError listing every problem of a (merged) input."""
    errors = [msg for _, msg in input_problems(cfg)]
    if errors:
        raise ValueError("invalid THUNDER4D input:\n  - " + "\n  - ".join(errors))
    return cfg


def save_input(cfg, path):
    """Write an input (defaults filled in) as JSON."""
    cfg = check_input(_merge(copy.deepcopy(DEFAULTS), cfg))
    Path(path).write_text(json.dumps(_jsonable(cfg), indent=2))
    return Path(path)


def load_input(path_or_dict):
    """Read an input file (or take a dict), fill in the defaults and check it."""
    if isinstance(path_or_dict, dict):
        raw = path_or_dict
    else:
        raw = json.loads(Path(path_or_dict).read_text())
    return check_input(_merge(copy.deepcopy(DEFAULTS), raw))


# ============================================================ building blocks
def phase_from_gdd(lam_m, table_lam_nm, table_gdd_fs2, lambda0_m):
    """Spectral phase [rad] per bounce from a GDD(lambda) table, in thunder4d's convention
    (+GDD/2 Omega^2); constant and linear parts removed at lambda0."""
    from scipy.constants import c as C0
    from scipy.integrate import cumulative_trapezoid

    wg = 2 * np.pi * C0 / (np.asarray(table_lam_nm, float) * 1e-9) * 1e-15
    i = np.argsort(wg)
    wg, g = wg[i], np.asarray(table_gdd_fs2, float)[i]
    w0 = 2 * np.pi * C0 / lambda0_m * 1e-15
    gd = cumulative_trapezoid(g, wg, initial=0.0)
    gd -= np.interp(w0, wg, gd)
    ph = cumulative_trapezoid(gd, wg, initial=0.0)
    ph -= np.interp(w0, wg, ph)
    w = 2 * np.pi * C0 / np.asarray(lam_m, float) * 1e-15
    return np.interp(w, wg, ph)


def _mirror_kwargs(cfg, lambda0_m):
    m = cfg["mirrors"]
    kw = {"mirror_tod_fs3": float(m["tod_fs3"] or 0.0), "mirror_diameter": m["diameter_m"]}
    r = m["reflectivity"]
    if isinstance(r, dict):
        r_lam, r_val = np.asarray(r["wavelength_nm"], float), np.asarray(r["value"], float)
        o = np.argsort(r_lam)
        r_lam, r_val = r_lam[o], r_val[o]

        def reflectivity(lam, r_lam=r_lam, r_val=r_val):
            return np.interp(np.asarray(lam) * 1e9, r_lam, r_val)
        kw["mirror_reflectivity"] = reflectivity
    else:
        kw["mirror_reflectivity"] = float(r)
    g = m["gdd_fs2"]
    if isinstance(g, dict):
        g_lam, g_val = np.asarray(g["wavelength_nm"], float), np.asarray(g["value"], float)

        def phase(lam, g_lam=g_lam, g_val=g_val):
            return phase_from_gdd(lam, g_lam, g_val, lambda0_m)
        kw["mirror_phase"] = phase
    else:
        kw["mirror_gdd_fs2"] = float(g or 0.0)
    return kw


def make_cell(cfg):
    from .cell import MPC
    from .materials import Material

    md, c = cfg["medium"], cfg["cell"]
    kw = {} if md["n2_1bar"] is None else {"n2_1bar": md["n2_1bar"]}
    gas = Material(md["gas"], md["pressure_bar"], md["temperature_K"], **kw)
    lam0 = cfg["laser"]["wavelength_nm"] * 1e-9
    return MPC(c["roc_m"], c["separation_m"], gas, int(c["n_passes"]), **_mirror_kwargs(cfg, lam0))


def grid_settings(cfg):
    g = dict(GRID_PRESETS[cfg["grid"]["preset"]])
    for key in g:
        if cfg["grid"].get(key) is not None:
            g[key] = float(cfg["grid"][key])
    return g


def input_duration_s(laser):
    """FWHM [s] of the injected Gaussian pulse: its transform limit stretched by the input GDD."""
    tl = laser["duration_fs"] * 1e-15
    gdd = (laser["gdd_fs2"] or 0.0) * 1e-30
    return tl * np.sqrt(1 + (4 * np.log(2) * gdd / tl ** 2) ** 2)


def make_grid(cfg, cell):
    from .lensframe import suggest_lens_grid

    las, g = cfg["laser"], cfg["grid"]
    kw = {"dtype": np.complex64 if g["precision"] == "single" else np.complex128}
    if g["device"] != "auto":
        kw["device"] = g["device"]
    if g["fft_backend"] != "auto":
        kw["fft_backend"] = g["fft_backend"]
    return suggest_lens_grid(cell, las["wavelength_nm"] * 1e-9, g["lambda_min_nm"] * 1e-9,
                             g["lambda_max_nm"] * 1e-9, input_duration_s(las), **grid_settings(cfg), **kw)


def make_pulse(cfg, grid, cell):
    from .pulse import Beam, Pulse, add_spectral_phase, gaussian, gaussian_spectrum, measured_spectrum

    las, beam = cfg["laser"], cfg["beam"]
    lam0 = las["wavelength_nm"] * 1e-9
    if las["shape"] == "measured":
        s = las["spectrum"]
        E = measured_spectrum(grid, s["wavelength_nm"], s["intensity"], s.get("phase_rad"))
    else:
        E = gaussian_spectrum(grid, fwhm_fs=las["duration_fs"], center_nm=las["wavelength_nm"])
    if las["gdd_fs2"] or las["tod_fs3"]:
        E = add_spectral_phase(grid, E, gdd_fs2=las["gdd_fs2"] or 0.0, tod_fs3=las["tod_fs3"] or 0.0)
    def at_grid(w):                     # beam sizes are given at lambda0; w ~ sqrt(lambda)
        return w * np.sqrt(grid.lambda0 / lam0)

    w0 = cell.linear_mode(lam0)["w0"]
    if beam["injection"] == "nonlinear":
        probe = Pulse.at_focus(grid, E, Beam(gaussian(at_grid(w0))), las["energy_J"])
        w0 = cell.nonlinear_mode(probe.peak_power(), lam0)["w0"]
    elif beam["injection"] == "waist":
        w0 = float(beam["w0_m"])
    return Pulse.at_focus(grid, E, Beam(gaussian(at_grid(w0)), chromatic_size="eigenmode"), las["energy_J"]), w0


def describe_input(cfg):
    """Quick checks of an input before launching it (milliseconds: neither the pulse nor the solver is built).

    Returns the linear eigenmode, the grid size and the Gaussian estimates of the peak power (versus the
    critical power of self-focusing) and of the peak fluence on the mirrors (linear mode)."""
    cfg = load_input(cfg)
    cell = make_cell(cfg)
    las = cfg["laser"]
    lam0 = las["wavelength_nm"] * 1e-9
    mode = cell.linear_mode(lam0)
    sizing = copy.deepcopy(cfg)
    sizing["grid"]["device"] = "cpu"                     # sizing only: never allocates on the GPU
    grid = make_grid(sizing, cell)
    from .design.specs import PEAK_POWER_FACTOR

    peak_power = PEAK_POWER_FACTOR["gaussian"] * las["energy_J"] / input_duration_s(las)
    critical = float(cell.medium.critical_power(lam0))
    n_bytes = np.prod(grid.shape) * grid.dtype.itemsize
    return {"w0_m": float(mode["w0"]), "w_mirror_m": float(mode["w_mirror"]), "zR_m": float(mode["zR"]),
            "gouy_per_pass_rad": float(mode["gouy_per_pass"]), "path_length_m": cell.d * cell.n_passes,
            "peak_power_W": float(peak_power), "critical_power_W": critical, "p_over_pcr": float(peak_power / critical),
            "mirror_fluence_J_cm2": float(2 * las["energy_J"] / (np.pi * mode["w_mirror"] ** 2) * 1e-4),
            "grid_shape": (grid.Nx, grid.Ny, grid.Nt), "field_MB": float(n_bytes / 1e6), "grid": grid.describe()}


@dataclass
class Built:
    cfg: dict
    cell: Any
    grid: Any
    pulse: Any
    sim: Any
    w0_injected_m: float


def build_simulation(cfg, verbose=False):
    """Input (dict or path) -> cell, grid, pulse and a LensFrameSimulation ready to run."""
    from .lensframe import LensFrameSimulation

    cfg = load_input(cfg)
    cell = make_cell(cfg)
    grid = make_grid(cfg, cell)
    pulse, w0 = make_pulse(cfg, grid, cell)
    s, r = cfg["solver"], cfg["run"]
    sim = LensFrameSimulation(pulse, cell, phi_max=s["phi_max"], gouy_step=s["gouy_step"],
                              record_every=int(r["record_every"]), near_field=r["near_field"],
                              record_mirror=r["record_mirror"], mode_reference=w0,
                              self_steepening=s["self_steepening"], kernels=s["kernels"])
    if verbose:
        print(f"[input] {cfg['name']}: grid {grid.Nx}x{grid.Ny}x{grid.Nt}, {cell.n_passes} passes")
    return Built(cfg, cell, grid, pulse, sim, w0)


def run_input(cfg, verbose=True, max_seconds=None):
    """Build and run an input (dict or path); returns the Built object (b.sim holds the results)."""
    b = build_simulation(cfg, verbose=verbose)
    b.sim.run(verbose=verbose, max_seconds=max_seconds)
    return b
