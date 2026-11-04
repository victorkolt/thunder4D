"""Benchmark: 0.29 mJ, 42 fs -> 8 fs in a compact argon-filled MPC (lens-frame solver).

P. Rueda, F. Videla, T. Witting, G. A. Torchia, F. J. Furch, "8 fs laser pulses from a compact
gas-filled multi-pass cell", Opt. Express 29, 27004 (2021), doi:10.1364/OE.435086 (open access).

Few-cycle compression of a Ti:Sa CPA in a short, very concentric cell with only 11 passes, hence a
high B-integral per pass (the paper estimates 1.5 pi per round trip) and P/Pc ~ 1 at the nominal
energy. It complements the two Yb cases: 800 nm, a nearly octave-wide spectrum, self-steepening,
and visible space-time couplings, in a small cell.

                 paper                                          used here
laser            Ti:Sa CPA, 790 nm, 42 fs, nearly Fourier-      Gaussian 42 fs at 790 nm, flat phase
                 limited; 290 uJ at the first cavity mirror     (measured input not published)
energy           the authors' own 3D simulation multiplies the  energy = 0.5 x 290 uJ = 145 uJ
                 energy by 0.5 to reproduce the spectrum        (--set energy=290e-6: nominal)
                 (pedestals, wavefront aberrations)
mirrors          protected silver, RoC 300 mm, 2", R = 97 %     R = 0.97, flat, GDD = 0
                 flat in the authors' simulation
geometry         d = 584 mm, 11 passes through the focus         MPC(R=0.3, d=0.584, n_passes=11)
gas              Ar 1500 mbar, n2 from Wahlstrand 2012           Ar 1.5 bar, n2(1 bar) = 1.0e-23 m^2/W
compressor       12 bounces PC-70 + fused-silica wedges,         -480 fs^2 (assumption: -40 fs^2 per
                 after a 0.5 mm FS exit window                  bounce) + best fused-silica thickness

Measured: Fourier limit 5.0 fs in the text, but 6.2 fs for the spectrum of Fig. 3a (digitised in
data/; the simulation is compared with the latter), compressed 8.06 fs (SPIDER; limited by residual
third order), more than 6-fold broadening at 1/e, 85 % of the energy in the core of the focused
beam. The authors' simulation gives a space-time Strehl-like ratio of 0.91.
Transmission: the code counts one mirror bounce per pass (11), the experiment has 10 bounces on the
cavity mirrors; the 61.4 % chamber transmission of the paper also includes windows and steering
mirrors, so it is only shown for information.
Not modelled: ionisation (the paper suspects it explains the stronger blue shift of the measured
spectrum; the peak intensity is reported), the measured input spectrum and phase, chirped-mirror
GDD oscillations, the spectral response of the silver mirrors.

Usage: python bench_rueda2021_8fs.py [--quick] [--device cpu] [--set energy=290e-6] ...
"""
import numpy as np

import thunder4d as th

from framework import (
    Benchmark,
    Ref,
    beam_pixels,
    best_phase,
    fourier_limit,
    fwhm,
    integrated_power,
    metric,
    spectral_phase,
)


def n_fused_silica(lam_um):
    """Malitson 1965."""
    l2 = lam_um ** 2
    return np.sqrt(1 + 0.6961663 * l2 / (l2 - 0.0684043 ** 2) + 0.4079426 * l2 / (l2 - 0.1162414 ** 2)
                   + 0.8974794 * l2 / (l2 - 9.896161 ** 2))


EXP_M2 = np.exp(-2)


def width_at(lam, S, level):
    """Full width of S(lam) at `level` x max, outermost crossings [same unit as lam]."""
    above = np.nonzero(S >= level * S.max())[0]
    return lam[above[-1]] - lam[above[0]]


def half_width_at(x, p, level=EXP_M2):
    """Half width of a 1D profile at `level` x max, outermost crossings."""
    above = np.nonzero(p >= level * p.max())[0]
    return 0.5 * (x[above[-1]] - x[above[0]])


class Rueda2021(Benchmark):
    name = "rueda2021_8fs"
    title = "0.29 mJ, 42 fs -> 8 fs, Ar 1500 mbar, 11 passes (Rueda et al., Opt. Express 29, 27004, 2021)"
    paper = {"citation": "P. Rueda et al., Opt. Express 29, 27004 (2021)", "doi": "10.1364/OE.435086",
             "url": "https://doi.org/10.1364/OE.435086"}
    params = {
        "lambda0": 790e-9,           # paper
        "energy": 145e-6,            # 0.5 x 290 uJ, as in the authors' simulation (nominal: 290e-6)
        "fwhm": 42e-15,              # paper (nearly Fourier-limited)
        "gas": "Ar",
        "pressure_bar": 1.5,         # paper
        "n2_1bar": 1.0e-23,          # table value; the paper uses Wahlstrand et al., PRL 109, 113904 (2012)
        "R": 0.3, "d": 0.584,        # paper
        "n_passes": 11,              # paper: 11 passes through the focal plane
        "mirror_R": 0.97,            # paper (simulation): protected silver, flat
        "cm_gdd": -480.0,            # assumption: 12 bounces of PC-70 at ~-40 fs^2 each
        "fs_scan_mm": (-2.0, 10.0),  # fused silica scanned in the compressor (includes the 0.5 mm window)
        "lambda_min": 450e-9, "lambda_max": 1300e-9,
        "window_fwhm": 100e-15,      # sets the time window (x time_window_factor): the output is chirped
        "injection": "linear",       # paper: matched to the cavity eigenmode
    }
    reference = {
        "ftl_fs": Ref(6.2, 0.10, note="FTL of the measured spectrum (Fig. 3a); text: 5.0 fs"),
        "comp_fs": Ref(8.06, kind="info", note="SPIDER, limited by residual TOD"),
        "broadening_1e": Ref(6.0, kind="min", note="> 6-fold at 1/e"),
        "core_energy_fraction": Ref(0.85, kind="info", note="measured, focused beam"),
        "transmission": Ref(0.737, kind="info", note="0.97^10; chamber incl. windows: 0.614"),
        "I_peak_max_Wcm2": Ref(1e14, kind="max", note="Ar ionisation, order of magnitude"),
    }
    measured = {"spectrum": "data/rueda2021_spectrum.csv",      # Fig. 3a, digitised (nm, I)
                "pulse": "data/rueda2021_pulse.csv"}            # Fig. 3b, digitised (fs, P)
    calibrated = {"energy": {"paper": 290e-6, "note": "x 0.5, as in the authors' own simulation "
                                                      "(pedestals, wavefront aberrations)"}}
    presets = [{"box_factor": 8, "points_per_waist": 8, "time_window_factor": 8},
               {"box_factor": 7, "points_per_waist": 7, "time_window_factor": 7},
               {"box_factor": 6, "points_per_waist": 6, "time_window_factor": 6},
               {"box_factor": 5, "points_per_waist": 5, "time_window_factor": 5}]
    quick_preset = {"box_factor": 5, "points_per_waist": 3, "time_window_factor": 4}
    compression_up = 8
    gdd_bounds = (-1500, 500)

    def make_cell(self, P, n_passes):
        gas = th.Material(P["gas"], P["pressure_bar"], n2_1bar=P["n2_1bar"])
        return th.MPC(R=P["R"], d=P["d"], medium=gas, n_passes=n_passes, mirror_reflectivity=P["mirror_R"])

    def grid_carrier(self, P):
        return 2 / (1 / P["lambda_min"] + 1 / P["lambda_max"])     # centre of the window in frequency

    def extra_metrics(self, ctx):
        """Compression as in the experiment (chirped mirrors + fused-silica scan), spectral
        broadening at 1/e, and the energy in the core of the focused beam."""
        from thunder4d import diagnostics as dg
        grid, P, up, R = ctx.grid, ctx.P, self.compression_up, self.reference
        M, C = {}, {}

        # --- compressor: fixed chirped mirrors + fused-silica thickness scan
        wc = 2 * np.pi * th.C0 / P["lambda0"]
        w = np.where(grid.valid, grid.omega, grid.omega0)
        phi_mm = n_fused_silica(2 * np.pi * th.C0 / w * 1e6) * w / th.C0 * 1e-3     # 1 mm of fused silica
        Wc = grid.omega - wc
        sel = grid.valid & (np.abs(Wc) < 0.1 * wc)
        c1, c0 = np.polyfit(Wc[sel], phi_mm[sel], 1)
        phi_mm = (phi_mm - c1 * Wc - c0) * grid.valid                              # no delay, no constant
        phi_cm = spectral_phase(grid, wc, P["cm_gdd"])
        Epix = beam_pixels(ctx.sim.Aw)
        L, scan, pk = best_phase(grid, Epix, lambda L: phi_cm + L * phi_mm, P["fs_scan_mm"], n_scan=81, up=up)
        _, P_tl = fourier_limit(grid, Epix, up)
        t_e, P_e = integrated_power(grid, Epix, phi_cm + L * phi_mm, up)
        M["comp_fs"] = metric(fwhm(t_e, P_e), "fs", f"compressed, {P['cm_gdd']:.0f} fs2 + {L:.2f} mm FS",
                              R["comp_fs"])
        M["comp_peak_ratio"] = metric(P_e.max() / P_tl.max(), "", "peak power / Fourier limit (paper compressor)")
        M["fs_thickness_mm"] = metric(L, "mm", "fused silica needed on top of the chirped mirrors")
        C.update(t_comp_fs=t_e, P_comp=P_e, fs_scan_mm=scan, fs_scan_peak=pk / P_tl.max())

        # --- spectral broadening at 1/e (frequency widths, as a ratio)
        h = ctx.sim.history
        lam, S = dg.total_spectrum(grid, ctx.sim.Aw)
        nu = th.C0 / lam
        ratio = width_at(nu[::-1], S[::-1], np.exp(-1)) / width_at(nu[::-1], h["spectrum"][0][::-1], np.exp(-1))
        M["broadening_1e"] = metric(ratio, "", "spectral width at 1/e, output / input (frequency)",
                                    R["broadening_1e"])

        # --- far field (focus of the collimated beam): energy within a circle of diameter
        #     pi x the 1/e^2 half widths (x, y averaged) of the integrated profiles, as in the paper
        theta, Ef = dg.near_field(grid, ctx.sim.Aw)                      # coarse: find the beam
        F = np.sum(np.abs(Ef) ** 2, axis=2)
        w_max = max(half_width_at(theta, F.sum(axis=1)), half_width_at(theta, F.sum(axis=0)))
        theta, Ef = dg.near_field(grid, ctx.sim.Aw, theta_max=4 * w_max, n_theta=121)   # fine
        F = np.sum(np.abs(Ef) ** 2, axis=2)
        wx = half_width_at(theta, F.sum(axis=1))
        wy = half_width_at(theta, F.sum(axis=0))
        r = np.hypot(theta[:, None], theta[None, :])
        core = F[r <= 0.5 * np.pi * 0.5 * (wx + wy)].sum() / F.sum()
        M["core_energy_fraction"] = metric(core, "", "far-field energy within diameter pi x w (1/e2 half width)",
                                           R["core_energy_fraction"])
        C.update(far_theta_mrad=theta * 1e3, far_fluence=(F / F.max()).astype(np.float32))
        return M, C


if __name__ == "__main__":
    Rueda2021().main()
