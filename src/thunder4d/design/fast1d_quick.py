"""fast1d as the QuickSolver of the design helper.

    quick = Fast1DQuick()                                  # accuracy="fast" for the design loop
    result = MPCDesigner(problem, quick=quick, workers=4).refine(starts)
    report = quick.validate(problem, result.best)          # validation phase: convergence check
    print(report.summary())

Mapping design helper -> fast1d
-------------------------------
* gas data: n2 from the design gas catalogue (catalog.GASES, overridable), dispersion from
  thunder4d.materials: the same numbers as the 4D solver.
* cell: d = Candidate.separation_m (R, M round trips, k), n_passes = Candidate.n_passes. Same pass
  convention as the 4D code: start right after a mirror, one pass = transit over d + bounce.
* mirrors: R(lambda) and GDD(lambda) of the candidate's coating, per bounce. No clipping on the
  mirror edge (spot pattern constraints are evaluated separately).
* input: energy reduced by Losses.mode_matching_efficiency;
  pre-chirp = LaserSpec.prechirp + Candidate.prechirp, plus the GDD implied by duration_fs >
  tl_duration_fs (positive chirp assumed).
* output: lossless GDD compressor (the optimal GDD is reported);
  per-bounce fluence and pulse duration, so the LIDT is taken at the local duration.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from math import log

import numpy as np
from scipy.constants import c as C0

from . import fast1d
from .catalog.gases import get_gas
from .catalog.mirrors import get_mirror
from .designer import Candidate, QuickResult
from .specs import DesignProblem


def _opt(obj, name, default):
    """Optional knob: the default is used if the field was removed from specs.py."""
    return getattr(obj, name, default) if obj is not None else default


def main_peak_fraction(t, power):
    """Energy between the two minima that bound the main peak / total energy."""
    p = np.asarray(power, dtype=float)
    i = int(np.argmax(p))
    left = i
    while left > 0 and p[left - 1] <= p[left]:
        left -= 1
    right = i
    while right < p.size - 1 and p[right + 1] <= p[right]:
        right += 1
    return float(p[left:right + 1].sum() / p.sum())


@dataclass
class Fast1DQuick:
    """QuickSolver backed by fast1d. Module-level class: picklable, so MPCDesigner(workers > 1) works.

    accuracy : fast1d preset for the design loop ("fast" | "default" | "fine")
    matching : "linear" (injected at the linear eigenmode, as fast1d) | "nonlinear" (Kerr-matched
               waist w0 (1 - P/Pcr)^(1/4), as MPC.match)
    options  : any other MPCInput field (self_steepening=False, aeff_factor=..., dphi_max=...)
    keep_spectrum : store the output spectrum in the metrics (and the JSON export)
    """
    accuracy: str = "fast"
    matching: str = "linear"
    keep_spectrum: bool = True
    options: dict = field(default_factory=dict)

    # ---------------------------------------------------------------- input
    def build_input(self, problem: DesignProblem, cand: Candidate, accuracy=None) -> fast1d.MPCInput:
        las, L = problem.laser, _opt(problem, "losses", None)
        lam0 = las.lambda0
        gas, coat = get_gas(cand.gas), get_mirror(cand.coating)
        eta_in = _opt(L, "mode_matching_efficiency", 1.0)
        tl = (las.tl_duration_fs or las.duration_fs) * 1e-15
        n2_lin = gas.n2_per_bar * cand.pressure_bar      # linear polarisation, working pressure

        def mirror_response(omega):
            lam = 2 * np.pi * C0 / omega
            # Coating.phase is in the thunder4d convention (+GDD/2 Om^2); fast1d uses the opposite sign
            return np.sqrt(coat.R(lam * 1e9)) * np.exp(-1j * coat.phase(lam, lam0))

        kw = {"energy": las.energy_J * eta_in, "fwhm": tl, "wavelength": lam0,
                  "shape": "sech2" if las.shape == "sech2" else "gaussian", "polarization": las.polarization,
                  "gas": cand.gas, "pressure": cand.pressure_bar, "roc": cand.roc_m, "length": cand.separation_m,
                  "n_passes": cand.n_passes, "mirror_response": mirror_response, "n2": n2_lin,
                  "accuracy": accuracy or self.accuracy}
        if self.matching == "nonlinear":
            _, w0, _ = fast1d.cell_mode(cand.roc_m, cand.separation_m, lam0)
            pcr = lam0 ** 2 / (2 * np.pi * n2_lin * las.n2_polarization_factor)
            ratio = eta_in * las.peak_power_W / pcr
            kw["w0_in"] = w0 * max(1.0 - ratio, 1e-3) ** 0.25
        kw.update(self.options)
        inp = fast1d.MPCInput(**kw)

        # pre-chirp: explicit GDD/TOD + the chirp implied by duration_fs > tl_duration_fs
        gdd = (las.prechirp_gdd_fs2 + cand.prechirp_gdd_fs2) * 1e-30
        tod = las.prechirp_tod_fs3 * 1e-45
        if las.tl_duration_fs and las.duration_fs > las.tl_duration_fs:
            gdd += tl ** 2 / (4 * log(2)) * np.sqrt((las.duration_fs * 1e-15 / tl) ** 2 - 1)
        if gdd or tod:
            stretch = np.sqrt(1 + (4 * log(2) * gdd / tl ** 2) ** 2) + 4 * abs(tod) / tl ** 3   # rough
            grid_inp = replace(inp, time_window=inp.time_window_factor * tl * stretch)
            t, om, _ = fast1d._make_grid(grid_inp)
            a = np.fft.ifft(np.fft.fft(fast1d._initial_field(grid_inp, t))
                            * np.exp(-1j * (0.5 * gdd * om ** 2 + tod * om ** 3 / 6)))
            inp = replace(inp, field=(t, a), time_window=grid_inp.time_window)
        return inp

    # ---------------------------------------------------------------- QuickSolver
    def __call__(self, problem: DesignProblem, cand: Candidate) -> QuickResult:
        inp = self.build_input(problem, cand)
        try:
            out = fast1d.simulate(inp)
        except fast1d.SimulationStopped as err:   # the model breaks down: rejected, not a crash
            collapse = isinstance(err, fast1d.BeamCollapse)
            extras = {"solver_warnings": [str(err)]}
            if not collapse:
                extras["solver_failed"] = str(err)
            return QuickResult(duration_fs=np.nan, peak_power_W=0.0, throughput=0.0,
                               p_over_pcr=max(err.p_over_pcr, 1.0) if collapse else None,
                               b_total_rad=err.b_integral,
                               b_per_pass_rad=err.b_integral / max(err.passes_done, 1), extras=extras)
        return self.to_quick_result(problem, cand, out)

    def to_quick_result(self, problem, cand, out: fast1d.MPCOutput) -> QuickResult:
        las = problem.laser
        gdd_fs2 = out.gdd_compression * 1e30
        throughput = out.energy_out / las.energy_J      # lossless compressor
        nb = out.inp.n_passes                                  # one bounce per pass
        fluence = 2 * out.pass_energy[:nb] / (np.pi * out.pass_w_mirror[:nb] ** 2) * 1e-4   # J/cm2
        coat = get_mirror(cand.coating)
        lam, s = out.spectrum_lambda("out")
        out_of_band = float(s[~coat.in_band(lam * 1e9)].sum() / s.sum())
        b_pass = np.diff(np.concatenate([[0.0], out.pass_b_integral]))
        extras = {"solver_warnings": list(out.warnings), "runtime_s": out.runtime,
                  "n_steps": out.n_steps, "n_points": int(out.t.size), "energy_out_J": out.energy_out,
                  "conservation_drift": out.conservation_drift,
                  "w_mirror_max_m": float(out.pass_w_mirror.max()),
                  "average_power_mirror_W": out.energy_in * las.rep_rate_Hz * out.inp.n_passes / 2,
                  "ftl_duration_fs": out.fwhm_ftl * 1e15}
        return QuickResult(
            duration_fs=out.fwhm_compressed * 1e15,
            peak_power_W=out.peak_power_compressed,
            throughput=throughput,
            main_peak_fraction=main_peak_fraction(out.t_compressed, np.abs(out.field_compressed) ** 2),
            compressor_gdd_fs2=gdd_fs2,
            b_total_rad=out.b_integral,
            b_per_pass_rad=float(b_pass.max()),
            p_over_pcr=out.peak_power_over_pcr,
            peak_intensity_W_cm2=out.intensity_max * 1e-4,
            mirror_fluence_J_cm2=fluence if nb else None,
            duration_per_bounce_fs=out.pass_fwhm[:nb] * 1e15 if nb else None,
            out_of_band_fraction=out_of_band,
            spectrum_nm=(lam * 1e9, s) if self.keep_spectrum else None,
            extras=extras,
        )

    # ---------------------------------------------------------------- validation phase
    def simulate(self, problem, cand, accuracy=None) -> fast1d.MPCOutput:
        """Full fast1d output (fields, spectra, per-pass data, plot()) for one candidate."""
        return fast1d.simulate(self.build_input(problem, cand, accuracy))

    def validate(self, problem, cand, accuracy=None, rtol=0.02, checks=("steps", "grid", "precision")):
        """Convergence check of fast1d for this candidate (fast1d.validate). accuracy=None checks the
        settings used in the design loop; accuracy="default" or "fine" checks a reference run."""
        return fast1d.validate(self.build_input(problem, cand, accuracy), checks=checks, rtol=rtol)
