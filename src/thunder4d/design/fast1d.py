"""
fast1d - minimal 1+1D solver for gas-filled multipass cells (MPC).

Fast inner solver for the THUNDER4D design helper: refine a handful of
candidate cells before handing them to the full 4D solver.

Model
-----
* Scalar envelope A(z, t) in sqrt(W) (|A|^2 = power), frame moving at the
  group velocity of the carrier.
* Full gas dispersion (Sellmeier), instantaneous Kerr effect with optional
  self-steepening. No Raman, no ionisation. The input is a transform-limited pulse, or any
  field given through MPCInput.field (the design helper uses it for a chirped input).
* Transverse dimension reduced to a Gaussian beam followed through its
  complex q parameter: free propagation, thin Kerr lens per step (peak
  power, aberration-free Gaussian ansatz, balances diffraction at P = Pcr),
  curved mirror (f = R/2) at each bounce.
* SPM uses the intensity-weighted mean nonlinear phase over the Gaussian
  profile: A_eff = aeff_factor * pi * w^2 (aeff_factor = 1 -> half the
  on-axis phase; 0.5 -> on-axis). The reported B-integral is on-axis.
* Linear or circular polarisation (n2_circ = 2/3 n2_lin).
* Interaction-picture RK4 (RK4IP). Step size set by the nonlinear phase per
  step (B-integral driven) and a minimum number of steps per pass.
* Compression: pure GDD removal, optimised for peak power (or FWHM).

Accuracy presets (MPCInput(..., accuracy=...) or inp.with_accuracy(...))
-------------------------------------------------------------------------
* "fast"    : design helper. Coarser steps, compact grid, single precision,
              grid enlarged automatically if the pulse reaches its edges.
* "default" : the original fast1d settings, with the beam (q) followed by sub-steps (at least
              8 per Rayleigh length): the focus of near-concentric cells is resolved.
* "fine"    : finer steps and a larger grid, double precision.
validate(inp) re-runs inp with halved steps, a larger grid and double
precision and reports how much each output moved: use it to check that a
result (or the "fast" preset for a given design) is converged.

Conventions
-----------
E(z, t) = Re[A(z, t) exp(i(w0 t - beta0 z))], numpy FFT, Omega = 2 pi fftfreq.
A positive GDD multiplies the spectrum by exp(-i GDD Omega^2 / 2), so the
optimal compressor GDD after SPM is negative. Everything is SI.

Usage
-----
>>> from thunder4d.design.fast1d import MPCInput, simulate, validate
>>> inp = MPCInput.herriott(roc=2.0, N=15, k=2, gas="Ar", pressure=0.13,
...                         energy=18e-3, fwhm=350e-15)
>>> out = simulate(inp)
>>> print(out.summary())
>>> print(validate(inp).summary())        # convergence check, validation phase only
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
from scipy import fft as _fft

from ..materials import N2_1BAR, SELLMEIER

# One 1D transform per call: threads do not help, and the design helper runs several
# simulations in parallel processes (workers=-1 would oversubscribe the cores).
fft, ifft = _fft.fft, _fft.ifft


C0 = 299_792_458.0

# --------------------------------------------------------------------------
# Gas data: single source, thunder4d.materials (Sellmeier: Börzsönyi 2008,
# 273.15 K, 1 bar; n2: electronic, 1 bar, linear polarisation, 293.15 K)
# --------------------------------------------------------------------------
SELLMEIER_T_REF = 273.15
N2_T_REF = 293.15


def _gas_key(gas: str) -> str:
    for key in SELLMEIER:
        if key.lower() == gas.lower():
            return key
    raise ValueError(f"unknown gas {gas!r}, available: {list(SELLMEIER)}")


def _pol_factor(polarization: str) -> float:
    p = polarization.lower()
    if p.startswith("lin"):
        return 1.0
    if p.startswith("circ"):
        return 2.0 / 3.0
    raise ValueError("polarization must be 'linear' or 'circular'")


def refractive_index(gas, wavelength, pressure, temperature=293.15):
    """Linear index of a gas. wavelength [m], pressure [bar], temperature [K]."""
    b1, c1, b2, c2 = SELLMEIER[_gas_key(gas)]
    l2 = (np.asarray(wavelength, dtype=float) * 1e6) ** 2
    chi = b1 * l2 / (l2 - c1) + b2 * l2 / (l2 - c2)
    rho = pressure * SELLMEIER_T_REF / temperature
    return np.sqrt(1.0 + rho * chi)


def kerr_index(gas, pressure, temperature=293.15, polarization="linear"):
    """Effective n2 [m^2/W] at the working pressure and polarisation."""
    n2 = N2_1BAR[_gas_key(gas)] * pressure * N2_T_REF / temperature
    return n2 * _pol_factor(polarization)


# --------------------------------------------------------------------------
# Cell geometry
# --------------------------------------------------------------------------
def herriott_length(roc, N, k, branch="concentric"):
    """Mirror separation of a re-entrant Herriott cell (N round trips, index k).

    branch = "concentric": d = R (1 + cos(pi k / N))   (d > R)
    branch = "planar"    : d = R (1 - cos(pi k / N))   (d < R)
    """
    c = np.cos(np.pi * k / N)
    if branch == "concentric":
        return roc * (1.0 + c)
    if branch == "planar":
        return roc * (1.0 - c)
    raise ValueError("branch must be 'concentric' or 'planar'")


def cell_mode(roc, length, wavelength):
    """Linear eigenmode of a symmetric two-mirror cell -> (z_R, w0, w_mirror)."""
    if not 0.0 < length < 2.0 * roc:
        raise ValueError("unstable cell: need 0 < length < 2 R")
    zr = 0.5 * np.sqrt(length * (2.0 * roc - length))
    w0 = np.sqrt(wavelength * zr / np.pi)
    wm = w0 * np.sqrt(1.0 + (0.5 * length / zr) ** 2)
    return zr, w0, wm


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------
def fwhm(x, y):
    """FWHM of the main peak of y(x), linear interpolation at half maximum."""
    y = np.asarray(y, dtype=float)
    i = int(np.argmax(y))
    half = 0.5 * y[i]
    il = i
    while il > 0 and y[il] > half:
        il -= 1
    ir = i
    while ir < len(y) - 1 and y[ir] > half:
        ir += 1
    if y[il] > half or y[ir] > half:
        return np.nan  # peak touches the window edge
    xl = x[il] + (half - y[il]) * (x[il + 1] - x[il]) / (y[il + 1] - y[il])
    xr = x[ir - 1] + (half - y[ir - 1]) * (x[ir] - x[ir - 1]) / (y[ir] - y[ir - 1])
    return xr - xl


def _rms(x, w):
    s = w.sum()
    m = (x * w).sum() / s
    return np.sqrt(((x - m) ** 2 * w).sum() / s)


def _golden_max(f, a, b, n_iter=40):
    g = 0.5 * (np.sqrt(5.0) - 1.0)
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = f(c), f(d)
    for _ in range(n_iter):
        if fc > fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a)
            fd = f(d)
    return 0.5 * (a + b)


# --------------------------------------------------------------------------
# Accuracy presets
# --------------------------------------------------------------------------
ACCURACY = {
    "fast": {"dphi_max": 0.2, "min_steps_per_pass": 8, "q_substeps": 8, "q_steps_per_zr": 8,
             "n_points": 256, "time_window_factor": 10.0, "lambda_min_factor": 0.6,
             "precision": "single", "fast_len": True, "auto_grid": True},
    "default": {"dphi_max": 0.05, "min_steps_per_pass": 20, "q_substeps": 8, "q_steps_per_zr": 8,
                "n_points": 2**12, "time_window_factor": 20.0, "lambda_min_factor": 0.5,
                "precision": "double", "fast_len": False, "auto_grid": False},
    "fine": {"dphi_max": 0.02, "min_steps_per_pass": 40, "q_substeps": 8, "q_steps_per_zr": 16,
             "n_points": 2**13, "time_window_factor": 30.0, "lambda_min_factor": 0.4,
             "precision": "double", "fast_len": False, "auto_grid": True},
}


_AUTO: Any = None   # "take the value from the accuracy preset" (see MPCInput.__post_init__)


# --------------------------------------------------------------------------
# Input
# --------------------------------------------------------------------------
@dataclass
class MPCInput:
    # laser (transform-limited input, no pre-chirp)
    energy: float = 1e-3  # J
    fwhm: float = 300e-15  # s, intensity FWHM
    wavelength: float = 1030e-9  # m, carrier
    shape: str = "gaussian"  # "gaussian" | "sech2"
    polarization: str = "linear"  # "linear" | "circular"
    field: tuple[np.ndarray, np.ndarray] | None = None
    # (t [s], A [sqrt(W)]) on any grid; overrides energy / fwhm / shape.
    # Same carrier as `wavelength`. Use MPCOutput.as_field() to chain stages.

    # cell
    gas: str = "Ar"
    pressure: float = 1.0  # bar
    temperature: float = 293.15  # K
    roc: float = 1.0  # m, mirror radius of curvature
    length: float = 1.9  # m, mirror separation
    n_passes: int = 20  # passes = transit over d + bounce, starting right after a mirror
    mirror_reflectivity: float = 0.999  # energy, per bounce
    mirror_gdd: float = 0.0  # s^2 per bounce
    mirror_response: Callable[[np.ndarray], np.ndarray] | None = None
    # complex amplitude transfer of one bounce on the absolute angular frequency [rad/s]
    # (this module's convention: positive GDD = exp(-i GDD Omega^2 / 2)); replaces
    # mirror_reflectivity and mirror_gdd when given (measured R(lambda) and GDD curves).
    w0_in: float | None = None  # waist at cell centre [m]; None -> matched
    n2: float | None = None  # override, linear pol, at working pressure [m^2/W]

    # physics switches
    kerr_lens: bool = True
    self_steepening: bool = True
    aeff_factor: float = 1.0  # A_eff = aeff_factor * pi w^2

    # numerics (accuracy="fast" | "default" | "fine" sets all of them; explicit values win)
    # The numerical settings default to _AUTO (None): __post_init__ takes them from the accuracy preset,
    # so after construction they are never None (hence their plain types).
    accuracy: str | None = None
    n_points: int = _AUTO  # minimum; increased to fit time_window
    time_window: float | None = None  # [s]; default: time_window_factor x input FWHM
    time_window_factor: float = _AUTO
    lambda_min: float | None = None  # shortest wavelength on grid; default lambda_min_factor x wavelength
    lambda_min_factor: float = _AUTO
    dphi_max: float = _AUTO  # max mean nonlinear phase per step [rad]
    min_steps_per_pass: int = _AUTO
    q_substeps: int = _AUTO  # beam (q) sub-steps per field step: cheap, lets the field take long steps
    q_steps_per_zr: float = _AUTO  # and at least this many per Rayleigh length (0 = off)
    precision: str = _AUTO  # "double" | "single" (complex64: ~2x faster)
    fast_len: bool = _AUTO  # any 2^a 3^b 5^c grid size instead of a power of 2
    auto_grid: bool = _AUTO  # enlarge the grid and re-run if the pulse reaches its edges
    compression: str = "peak"  # "peak" | "fwhm"
    points_per_fwhm: int = 16  # minimum samples per FTL FWHM for the compression analysis
    # guards: stop (SimulationStopped) instead of crawling when the model breaks down
    max_peak_growth: float = 20.0      # peak power / input peak power (shock, collapse)
    max_steps_per_pass: int = 2000
    store_passes: bool = False  # keep the spectrum after every pass

    def __post_init__(self):
        preset = ACCURACY[self.accuracy or "default"]
        for key, value in preset.items():
            if getattr(self, key) is None:
                setattr(self, key, value)

    def with_accuracy(self, accuracy, **overrides):
        """Copy with all numerical settings taken from a preset (explicit overrides win)."""
        kw = {key: None for key in ACCURACY["default"]}
        kw.update(overrides)
        return replace(self, accuracy=accuracy, **kw)

    @classmethod
    def herriott(cls, roc, N, k, branch="concentric", **kwargs):
        """Re-entrant Herriott cell: length from (R, N, k), n_passes = 2N by default."""
        kwargs.setdefault("n_passes", 2 * N)
        return cls(roc=roc, length=herriott_length(roc, N, k, branch), **kwargs)


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------
@dataclass
class MPCOutput:
    inp: MPCInput
    # grids (omega is absolute and sorted)
    t: np.ndarray
    t_compressed: np.ndarray  # finer grid of the compressed / FTL fields (same window as t)
    omega: np.ndarray
    # fields [sqrt(W)] on t
    field_in: np.ndarray
    field_out: np.ndarray
    field_compressed: np.ndarray  # on t_compressed, peak centred at t = 0
    field_ftl: np.ndarray  # on t_compressed
    # spectral energy densities [J / (rad/s)] on omega
    spectrum_in: np.ndarray
    spectrum_out: np.ndarray
    # scalars
    energy_in: float
    energy_out: float
    transmission: float
    fwhm_in: float
    fwhm_out: float  # before compression
    fwhm_compressed: float
    fwhm_ftl: float
    gdd_compression: float  # s^2, to apply after the cell
    peak_power_in: float
    peak_power_compressed: float
    peak_power_ftl: float
    b_integral: float  # on-axis, total
    n2_eff: float
    p_critical: float
    peak_power_over_pcr: float  # max over the propagation
    intensity_max: float  # W/m^2, on-axis
    fluence_mirror_max: float  # J/m^2, peak fluence on the cavity mirrors
    w0_linear: float
    w_mirror_linear: float
    # per pass (value at the end of each pass)
    pass_energy: np.ndarray
    pass_b_integral: np.ndarray  # cumulative, on-axis
    pass_w_mirror: np.ndarray
    pass_w_min: np.ndarray
    pass_bandwidth_rms: np.ndarray  # rad/s
    pass_fwhm: np.ndarray  # s, pulse duration (uncompressed) at the end of each pass
    conservation_drift: float  # max relative change over a pass of the conserved quantity (numerics check):
    # photon number with self-steepening, energy without
    pass_spectra: np.ndarray | None = None
    warnings: list = field(default_factory=list)  # grid-edge diagnostics
    n_steps: int = 0
    runtime: float = 0.0

    # -- convenience -------------------------------------------------------
    @property
    def compression_factor(self) -> float:
        return self.fwhm_in / self.fwhm_compressed

    @property
    def strehl_t(self) -> float:
        """Compressed peak power / Fourier-limit peak power."""
        return self.peak_power_compressed / self.peak_power_ftl

    def spectrum_lambda(self, which="out"):
        """(wavelength [m], spectral energy density [J/m]) for 'in' or 'out'."""
        s = self.spectrum_out if which == "out" else self.spectrum_in
        m = self.omega > 0
        lam = 2 * np.pi * C0 / self.omega[m]
        sl = s[m] * 2 * np.pi * C0 / lam**2
        return lam[::-1], sl[::-1]

    def as_field(self, compressed=True, efficiency=1.0):
        """(t, A) to feed MPCInput(field=...) of a next stage."""
        if compressed:
            return self.t_compressed.copy(), self.field_compressed * np.sqrt(efficiency)
        return self.t.copy(), self.field_out * np.sqrt(efficiency)

    def summary(self) -> str:
        i = self.inp
        lines = [
            (f"{i.gas} {i.pressure * 1e3:.0f} mbar, {i.polarization}, R = {i.roc:g} m, "
            f"d = {i.length:.4f} m, {i.n_passes} passes"),
            f"w0 = {self.w0_linear * 1e6:.0f} um, w_mirror = {self.w_mirror_linear * 1e3:.3f} mm (linear)",
            (f"E in / out       : {self.energy_in * 1e3:.3f} / {self.energy_out * 1e3:.3f} mJ "
            f"(T = {self.transmission * 100:.1f} %)"),
            f"B-integral       : {self.b_integral:.1f} rad (on-axis)",
            f"P/Pcr max        : {self.peak_power_over_pcr:.3f} (Pcr = {self.p_critical * 1e-9:.1f} GW)",
            f"I max            : {self.intensity_max * 1e-4:.3e} W/cm^2",
            f"F mirror max     : {self.fluence_mirror_max * 1e-1:.1f} mJ/cm^2",
            f"FWHM in / out    : {self.fwhm_in * 1e15:.1f} / {self.fwhm_out * 1e15:.1f} fs",
            (f"FWHM comp / FTL  : {self.fwhm_compressed * 1e15:.2f} / {self.fwhm_ftl * 1e15:.2f} fs "
            f"(x{self.compression_factor:.1f})"),
            f"GDD compression  : {self.gdd_compression * 1e30:.0f} fs^2",
            (f"P peak comp / FTL: {self.peak_power_compressed * 1e-9:.1f} / "
            f"{self.peak_power_ftl * 1e-9:.1f} GW (ratio {self.strehl_t:.2f}, lossless compressor)"),
            (f"{self.n_steps} steps, {self.t.size} points, {self.inp.precision} precision, "
            f"{self.runtime:.3f} s, conservation drift {self.conservation_drift:.1e}"),
        ] + [f"WARNING: {w}" for w in self.warnings]
        return "\n".join(lines)

    def plot(self):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(2, 2, figsize=(10, 7))
        for which, lab in (("in", "input"), ("out", "output")):
            lam, s = self.spectrum_lambda(which)
            ax[0, 0].plot(lam * 1e9, s / s.max(), label=lab)
        ax[0, 0].set(xlabel="wavelength (nm)", ylabel="norm. spectrum")
        ax[0, 0].set_xlim(self.inp.wavelength * 1e9 * np.array([0.6, 1.5]))
        ax[0, 0].legend()
        tf = self.t_compressed * 1e15
        ax[0, 1].plot(tf, np.abs(self.field_compressed) ** 2 * 1e-9, label="compressed")
        ax[0, 1].plot(tf, np.abs(self.field_ftl) ** 2 * 1e-9, "--", label="FTL")
        ax[0, 1].set_xlim(-8 * self.fwhm_compressed * 1e15, 8 * self.fwhm_compressed * 1e15)
        ax[0, 1].set(xlabel="time (fs)", ylabel="power (GW)")
        ax[0, 1].legend()
        p = np.arange(1, self.inp.n_passes + 1)
        ax[1, 0].plot(p, self.pass_w_mirror * 1e3, "o-", label="w mirror")
        ax[1, 0].plot(p, self.pass_w_min * 1e3, "s-", label="w min")
        ax[1, 0].set(xlabel="pass", ylabel="beam radius (mm)")
        ax[1, 0].legend()
        ax[1, 1].plot(p, self.pass_b_integral, "o-")
        ax[1, 1].set(xlabel="pass", ylabel="cumulative B (rad)")
        fig.tight_layout()
        return fig


# --------------------------------------------------------------------------
# Solver
# --------------------------------------------------------------------------
def _initial_field(inp, t):
    if inp.field is not None:
        t_in, a_in = (np.asarray(x) for x in inp.field)
        re = np.interp(t, t_in, a_in.real, left=0.0, right=0.0)
        im = np.interp(t, t_in, a_in.imag, left=0.0, right=0.0)
        return re + 1j * im
    tau = inp.fwhm
    if inp.shape == "gaussian":
        a = np.exp(-2.0 * np.log(2.0) * (t / tau) ** 2)
    elif inp.shape == "sech2":
        a = 1.0 / np.cosh(t / (tau / 1.7627471740390860))
    else:
        raise ValueError("shape must be 'gaussian' or 'sech2'")
    dt = t[1] - t[0]
    return a * np.sqrt(inp.energy / (np.sum(a**2) * dt)) + 0j


def _make_grid(inp):
    lam0 = inp.wavelength
    lam_min = inp.lambda_min or inp.lambda_min_factor * lam0
    f_half = C0 / lam_min - C0 / lam0
    dt = 0.5 / f_half
    k = inp.time_window_factor
    if inp.field is not None:
        t_in, a_in = (np.asarray(x) for x in inp.field)
        tw = inp.time_window or max(t_in.max() - t_in.min(), k * fwhm(t_in, np.abs(a_in) ** 2))
    else:
        tw = inp.time_window or k * inp.fwhm
    if inp.fast_len:
        n = _fft.next_fast_len(max(inp.n_points, int(np.ceil(tw / dt))))
        n += n % 2
    else:
        n = max(inp.n_points, int(2 ** np.ceil(np.log2(tw / dt))))
    t = (np.arange(n) - n // 2) * dt
    om = 2.0 * np.pi * np.fft.fftfreq(n, dt)
    return t, om, dt


class SimulationStopped(RuntimeError):
    """The run was stopped because the model breaks down: beam collapse, optical shock / runaway
    peak power, or step budget exhausted. Carries what was known when it stopped."""

    def __init__(self, message, p_over_pcr=0.0, passes_done=0, b_integral=0.0):
        super().__init__(message)
        self.p_over_pcr, self.passes_done, self.b_integral = p_over_pcr, passes_done, b_integral


class BeamCollapse(SimulationStopped):
    """Self-focusing collapse of the Gaussian beam (P > Pcr somewhere in the cell)."""


def simulate(inp: MPCInput) -> MPCOutput:
    """Propagate inp through the cell and compress with pure GDD.

    With inp.auto_grid, the grid is enlarged (shorter lambda_min, longer time window) and the
    run repeated, up to 3 times, while the pulse reaches the edges of the grid."""
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):   # a collapsing beam raises BeamCollapse
        out = _simulate(inp)
    tries = 0
    while inp.auto_grid and tries < 3 and any("edge" in w for w in out.warnings):
        grow = {}
        if any("frequency-grid" in w for w in out.warnings):
            f_half = C0 / (inp.lambda_min or inp.lambda_min_factor * inp.wavelength) - C0 / inp.wavelength
            grow["lambda_min"] = C0 / (C0 / inp.wavelength + 1.5 * f_half)
        if any("time-window" in w for w in out.warnings):
            grow["time_window"] = 2.0 * (inp.time_window or inp.time_window_factor * out.fwhm_in)
        runtime = out.runtime
        inp = replace(inp, **grow)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            out = _simulate(inp)
        out.runtime += runtime
        tries += 1
    if tries:
        out.warnings.append(f"grid enlarged automatically ({tries}x), final: {out.t.size} points, "
                            f"lambda_min = {(inp.lambda_min or inp.lambda_min_factor * inp.wavelength) * 1e9:.0f} nm")
    return out


def _simulate(inp: MPCInput) -> MPCOutput:
    t_start = time.perf_counter()
    lam0 = inp.wavelength
    w0c = 2.0 * np.pi * C0 / lam0
    k0 = 2.0 * np.pi / lam0
    t, om, dt = _make_grid(inp)
    n = t.size
    omega = w0c + om
    single = inp.precision == "single"
    cdtype, _rdtype = (np.complex64, np.float32) if single else (np.complex128, np.float64)

    # --- dispersion ----------------------------------------------------------
    valid = omega > 0.1 * w0c
    nidx = np.ones(n)
    nidx[valid] = refractive_index(inp.gas, 2 * np.pi * C0 / omega[valid], inp.pressure, inp.temperature)

    def beta_of(w):
        return refractive_index(inp.gas, 2 * np.pi * C0 / w, inp.pressure, inp.temperature) * w / C0

    dw = 1e-4 * w0c
    beta0 = beta_of(w0c)
    beta1 = (beta_of(w0c + dw) - beta_of(w0c - dw)) / (2 * dw)
    lin = np.where(valid, -1j * (nidx * omega / C0 - beta0 - beta1 * om), 0.0)
    n0 = float(refractive_index(inp.gas, lam0, inp.pressure, inp.temperature))

    # --- nonlinearity ----------------------------------------------------------
    pol = _pol_factor(inp.polarization)
    n2 = (inp.n2 * pol) if inp.n2 is not None else kerr_index(inp.gas, inp.pressure, inp.temperature, inp.polarization)
    pcr = lam0**2 / (2.0 * np.pi * n0 * n2)
    ss = np.clip(1.0 + om / w0c, 0.0, None) if inp.self_steepening else np.ones(n)
    gamma_w2 = n2 * w0c / (C0 * inp.aeff_factor * np.pi)  # gamma = gamma_w2 / w^2
    mss = (-1j * ss).astype(cdtype)  # -i (1 + Omega / w0)

    def nl_t(at, g):
        """Nonlinear term from the time-domain field (|A|^2 without the sqrt of abs)."""
        return (g * mss) * fft((at.real * at.real + at.imag * at.imag) * at)

    # Half-step linear propagators, cached: steps are quantised to h_max 2^(-j/8)
    # (never longer than requested), so the same few factors are reused all along.
    exp_cache = {}

    def half_step(h, j):
        if j is None:
            return np.exp(lin * (0.5 * h)).astype(cdtype)
        e = exp_cache.get(j)
        if e is None:
            e = exp_cache[j] = np.exp(lin * (0.5 * h)).astype(cdtype)
        return e

    def rk4ip(aw, at, h, g, e):
        ai = e * aw
        k1 = e * nl_t(at, g)
        k2 = nl_t(ifft(ai + (0.5 * h) * k1), g)
        k3 = nl_t(ifft(ai + (0.5 * h) * k2), g)
        k4 = nl_t(ifft(e * (ai + h * k3)), g)
        return e * (ai + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3)) + (h / 6.0) * k4

    # --- beam ------------------------------------------------------------------
    d = inp.length
    zr_lin, w0_lin, wm_lin = cell_mode(inp.roc, d, lam0)
    zr_in = zr_lin if inp.w0_in is None else np.pi * inp.w0_in**2 / lam0
    q = -0.5 * d + 1j * zr_in  # at the entrance mirror plane

    def w_of(qq):
        return np.sqrt(-lam0 / (np.pi * np.imag(1.0 / qq)))

    if inp.mirror_response is not None:
        mirror = np.zeros(n, complex)
        mirror[valid] = inp.mirror_response(omega[valid])
    else:
        mirror = np.sqrt(inp.mirror_reflectivity) * np.exp(-0.5j * inp.mirror_gdd * om**2)
    mirror = mirror.astype(cdtype)
    # conserved by the propagation: photon number with self-steepening, energy without
    weight = np.where(valid, 1.0 / np.where(valid, omega, 1.0), 0.0) if inp.self_steepening else np.ones(n)

    # --- propagation -------------------------------------------------------------
    a0 = _initial_field(inp, t)
    aw = fft(a0.astype(cdtype))
    energy_in = np.sum(np.abs(a0) ** 2) * dt
    b_axis, i_max, ratio_max, f_max, nsteps, drift = 0.0, 0.0, 0.0, 0.0, 0, 0.0
    ppk0 = None
    np_ = inp.n_passes
    p_energy, p_b, p_wm, p_wmin, p_bw, p_fwhm = (np.zeros(np_) for _ in range(6))
    p_spec = np.zeros((np_, n)) if inp.store_passes else None
    h_max = d / inp.min_steps_per_pass
    nlev = 8  # step quantisation levels per octave

    for p in range(np_):
        z, wmin = 0.0, np.inf
        conserved = np.sum(np.abs(aw) ** 2 * weight)
        at = ifft(aw)
        while z < d * (1.0 - 1e-12):
            ppk = float(np.max(at.real * at.real + at.imag * at.imag))
            w_now = w_of(q)
            if not np.isfinite(w_now) or not np.isfinite(ppk):
                raise BeamCollapse(f"beam collapse in pass {p + 1} at z = {z:.3f} m "
                                   f"(P/Pcr reached {max(ratio_max, ppk / pcr):.2f})", max(ratio_max, ppk / pcr),
                                   p, b_axis)
            if ppk0 is None:
                ppk0 = ppk
            if ppk > inp.max_peak_growth * ppk0 or nsteps > inp.max_steps_per_pass * np_:
                why = (f"peak power x{ppk / ppk0:.0f}" if ppk > inp.max_peak_growth * ppk0
                       else f"more than {inp.max_steps_per_pass} steps per pass")
                raise SimulationStopped(f"stopped in pass {p + 1}: {why} (optical shock or collapse, "
                                        f"B = {b_axis:.0f} rad so far)", ratio_max, p, b_axis)
            g = gamma_w2 / w_now ** 2
            h_req = min(inp.dphi_max / (g * ppk), h_max)
            j = int(np.ceil(-nlev * np.log2(h_req / h_max) - 1e-9))
            h = h_max * 2.0 ** (-j / nlev)
            if h >= d - z:  # last (partial) step of the pass: not cached
                h, j = d - z, None
            # beam: q followed with q_substeps sub-steps (free propagation + thin Kerr lens),
            # the field step uses the exact mean of 1 / w^2 over the step
            m_sub = inp.q_substeps
            if inp.q_steps_per_zr:   # resolve the focus: beam sub-steps no longer than z_R / q_steps_per_zr
                m_sub = max(m_sub, int(np.ceil(h * inp.q_steps_per_zr / zr_lin)))
            hs, inv_w2 = h / m_sub, 0.0
            for _ in range(m_sub):
                q = q + 0.5 * hs
                wm = w_of(q)
                wmin = min(wmin, wm)
                inv_w2 += hs / wm**2
                if inp.kerr_lens:
                    q = 1.0 / (1.0 / q - hs * lam0**2 * ppk / (np.pi**2 * wm**4 * pcr))
                q = q + 0.5 * hs
            aw = rk4ip(aw, at, h, gamma_w2 * inv_w2 / h, half_step(h, j))
            at = ifft(aw)
            b_axis += k0 * n2 * 2.0 * ppk / np.pi * inv_w2
            i_max = max(i_max, 2.0 * ppk / (np.pi * wmin**2))
            ratio_max = max(ratio_max, ppk / pcr)
            z += h
            nsteps += 1
        spec = np.abs(aw).astype(float) ** 2
        drift = max(drift, abs(np.sum(spec * weight) / conserved - 1.0))
        p_energy[p] = spec.sum() * dt / n
        p_b[p] = b_axis
        p_wm[p] = w_of(q)
        p_wmin[p] = wmin
        p_bw[p] = _rms(om, spec)
        p_fwhm[p] = fwhm(t, np.abs(at).astype(float) ** 2)
        if p_spec is not None:
            p_spec[p] = np.fft.fftshift(spec) * dt**2 / (2 * np.pi)
        # bounce on the next mirror: one pass = transit + bounce (as the 4D code)
        f_max = max(f_max, 2.0 * p_energy[p] / (np.pi * p_wm[p] ** 2))
        aw = aw * mirror
        q = 1.0 / (1.0 / q - 2.0 / inp.roc)

    aw = aw.astype(complex)  # analysis and compression in double precision
    a_out = ifft(aw)
    spec = np.abs(aw) ** 2   # after the last bounce
    energy_out = spec.sum() * dt / n

    # --- grid diagnostics --------------------------------------------------------
    warnings = []
    spec_tot = np.abs(aw) ** 2
    edge_w = np.abs(om) > 0.9 * np.abs(om).max()
    if spec_tot[edge_w].sum() > 1e-4 * spec_tot.sum():
        warnings.append("spectrum reaches the frequency-grid edge: lower lambda_min")
    p_out = np.abs(a_out) ** 2
    if p_out[np.abs(t) > 0.45 * n * dt].sum() > 1e-4 * p_out.sum():
        warnings.append("pulse reaches the time-window edge: increase time_window")

    # --- compression: pure GDD ------------------------------------------------
    spec = np.abs(aw) ** 2
    t_rms = _rms(t, np.abs(a_out) ** 2)
    bw_rms = _rms(om, spec)
    g_span = 3.0 * t_rms / bw_rms

    # Analysed on a finer time grid (zero-padded spectrum, exact band-limited interpolation) so
    # that few-cycle pulses get >= points_per_fwhm samples: peak power, FWHM and optimal GDD do
    # not depend on the propagation grid.
    ftl_coarse = fwhm(t, np.abs(np.fft.fftshift(ifft(np.abs(aw)))) ** 2)
    pad = 1
    while pad < 16 and np.isfinite(ftl_coarse) and ftl_coarse < inp.points_per_fwhm * dt / pad:
        pad *= 2
    if pad > 1:
        n_c, h = n * pad, n // 2
        aw_c = np.zeros(n_c, complex)
        aw_c[:h], aw_c[n_c - (n - h):] = aw[:h], aw[h:]
        aw_c *= pad                       # ifft normalisation: same |A|^2 on the finer grid
        om_c = 2.0 * np.pi * np.fft.fftfreq(n_c, dt / pad)
        t_c = t[0] + np.arange(n_c) * (dt / pad)
    else:
        n_c, aw_c, om_c, t_c = n, aw, om, t
    om2 = om_c**2

    def compressed(gdd):
        return ifft(aw_c * np.exp(-0.5j * gdd * om2))

    if inp.compression == "peak":
        def score(gdd):
            return np.max(np.abs(compressed(gdd)) ** 2)
    elif inp.compression == "fwhm":
        def score(gdd):
            v = fwhm(t_c, np.abs(compressed(gdd)) ** 2)
            return -v if np.isfinite(v) else -np.inf
    else:
        raise ValueError("compression must be 'peak' or 'fwhm'")

    grid = np.linspace(-g_span, g_span, 161)
    vals = np.array([score(g) for g in grid])
    ib = int(np.clip(np.argmax(vals), 1, grid.size - 2))
    gdd_opt = _golden_max(score, grid[ib - 1], grid[ib + 1])

    a_c = compressed(gdd_opt)
    a_c = np.roll(a_c, n_c // 2 - int(np.argmax(np.abs(a_c))))
    a_ftl = np.fft.fftshift(ifft(np.abs(aw_c)))

    return MPCOutput(
        inp=inp,
        t=t,
        t_compressed=t_c,
        omega=np.fft.fftshift(omega),
        field_in=a0,
        field_out=a_out,
        field_compressed=a_c,
        field_ftl=a_ftl,
        spectrum_in=np.fft.fftshift(np.abs(fft(a0)) ** 2) * dt**2 / (2 * np.pi),
        spectrum_out=np.fft.fftshift(spec) * dt**2 / (2 * np.pi),
        energy_in=energy_in,
        energy_out=energy_out,
        transmission=energy_out / energy_in,
        fwhm_in=fwhm(t, np.abs(a0) ** 2),
        fwhm_out=fwhm(t, np.abs(a_out) ** 2),
        fwhm_compressed=fwhm(t_c, np.abs(a_c) ** 2),
        fwhm_ftl=fwhm(t_c, np.abs(a_ftl) ** 2),
        gdd_compression=gdd_opt,
        peak_power_in=np.max(np.abs(a0) ** 2),
        peak_power_compressed=np.max(np.abs(a_c) ** 2),
        peak_power_ftl=np.max(np.abs(a_ftl) ** 2),
        b_integral=b_axis,
        n2_eff=n2,
        p_critical=pcr,
        peak_power_over_pcr=ratio_max,
        intensity_max=i_max,
        fluence_mirror_max=f_max if np_ > 1 else np.nan,
        w0_linear=w0_lin,
        w_mirror_linear=wm_lin,
        pass_energy=p_energy,
        pass_b_integral=p_b,
        pass_w_mirror=p_wm,
        pass_w_min=p_wmin,
        pass_bandwidth_rms=p_bw,
        pass_fwhm=p_fwhm,
        conservation_drift=drift,
        pass_spectra=p_spec,
        warnings=warnings,
        n_steps=nsteps,
        runtime=time.perf_counter() - t_start,
    )


# --------------------------------------------------------------------------
# Validation (convergence of a run): validation phase, not used by the design loop
# --------------------------------------------------------------------------
VALIDATION_QUANTITIES = (
    # attribute, label, display scale, unit
    ("fwhm_compressed", "compressed FWHM", 1e15, "fs"),
    ("peak_power_compressed", "compressed peak power", 1e-9, "GW"),
    ("gdd_compression", "compressor GDD", 1e30, "fs^2"),
    ("b_integral", "B-integral", 1.0, "rad"),
    ("transmission", "transmission", 1.0, ""),
    ("peak_power_over_pcr", "P/Pcr max", 1.0, ""),
    ("intensity_max", "peak intensity", 1e-4, "W/cm^2"),
    ("fluence_mirror_max", "mirror fluence", 0.1, "mJ/cm^2"),
)


def spectral_distance(a: MPCOutput, b: MPCOutput) -> float:
    """Half L1 distance between the two normalised output spectra (0 = identical, 1 = disjoint)."""
    sa = a.spectrum_out / a.spectrum_out.sum()
    sb = np.interp(a.omega, b.omega, b.spectrum_out, left=0.0, right=0.0)
    sb = sb / sb.sum()
    return 0.5 * float(np.abs(sa - sb).sum())


@dataclass
class ValidationReport:
    reference: MPCOutput
    variants: dict  # check name -> MPCOutput
    rows: list  # (check, label, reference, variant, relative change, tolerance, ok), display units
    rtol: float

    @property
    def ok(self) -> bool:
        return all(r[-1] for r in self.rows)

    def summary(self) -> str:
        lines = [f"fast1d validation ({'PASSED' if self.ok else 'FAILED'}, tolerance {100 * self.rtol:.1f} %)",
                 (f"  reference: {self.reference.inp.accuracy or 'custom'} settings, {self.reference.t.size} points, "
                 f"{self.reference.n_steps} steps, {self.reference.inp.precision} precision"),
                 f"  {'check':<10}{'quantity':<24}{'reference':>12}{'variant':>12}{'change':>9}"]
        for check, label, ref, var, rel, tol, ok in self.rows:
            if isinstance(ref, str):
                lines.append(f"  {check:<10}{label:<24}{ref:>12}{var:>12}{'':>9}  {'ok' if ok else 'FAILED'}")
            else:
                lines.append(f"  {check:<10}{label:<24}{ref:>12.4g}{var:>12.4g}{100 * rel:>8.2f}%"
                             f"  {'ok' if ok else 'FAILED'}")
        return "\n".join(lines)


def validate(inp: MPCInput, checks=("steps", "grid", "precision"), rtol=0.02) -> ValidationReport:
    """Check that the result of inp is converged: re-run it with
        steps     : half the nonlinear phase per step, twice the minimum steps and beam sub-steps
        grid      : 1.5x wider spectral window and 1.5x longer time window
        precision : double precision (only if inp runs in single precision)
    and report the relative change of the main outputs (and the distance between the output
    spectra). Also checks the photon-number conservation and the grid-edge diagnostics of the
    reference run. Costs 3-4 runs: meant for the validation phase, not the design loop."""
    ref = simulate(inp)
    base = replace(ref.inp, auto_grid=False)  # the grid actually used (after auto-enlargement)
    lam_min = base.lambda_min or base.lambda_min_factor * base.wavelength
    f_half = C0 / lam_min - C0 / base.wavelength
    variants = {}
    if "steps" in checks:
        variants["steps"] = replace(base, dphi_max=0.5 * base.dphi_max,
                                    min_steps_per_pass=2 * base.min_steps_per_pass,
                                    q_substeps=2 * base.q_substeps,
                                    q_steps_per_zr=2 * base.q_steps_per_zr)
    if "grid" in checks:
        variants["grid"] = replace(base, lambda_min=C0 / (C0 / base.wavelength + 1.5 * f_half),
                                   time_window=1.5 * (ref.t[-1] - ref.t[0]))
    if "precision" in checks and base.precision == "single":
        variants["precision"] = replace(base, precision="double")
    outs = {name: simulate(v) for name, v in variants.items()}

    rows = []
    drift_tol = 1e-4 if base.precision == "single" else 1e-6
    rows.append(("numerics", "conservation drift", f"{ref.conservation_drift:.1e}", f"< {drift_tol:.0e}",
                 0.0, drift_tol, ref.conservation_drift < drift_tol))
    rows.append(("numerics", "grid edges", "reached" if any("edge" in w for w in ref.warnings) else "clear",
                 "", 0.0, 0.0, not any("edge" in w for w in ref.warnings)))
    for name, out in outs.items():
        for attr, label, scale, unit in VALIDATION_QUANTITIES:
            a, b = getattr(ref, attr), getattr(out, attr)
            if not (np.isfinite(a) and np.isfinite(b)):
                continue
            rel = abs(b - a) / abs(a) if a != 0 else abs(b - a)
            rows.append((name, f"{label} [{unit}]" if unit else label, a * scale, b * scale, rel, rtol, rel <= rtol))
        dist = spectral_distance(ref, out)
        rows.append((name, "spectrum (L1 distance)", 0.0, dist, dist, rtol, dist <= rtol))
    return ValidationReport(ref, outs, rows, rtol)


if __name__ == "__main__":
    inp = MPCInput.herriott(
        roc=2.0, N=15, k=2, gas="Ar", pressure=0.13,
        energy=18e-3, fwhm=350e-15, mirror_reflectivity=0.999,
    )
    print(simulate(inp).summary())
    print(validate(inp.with_accuracy("fast")).summary())
