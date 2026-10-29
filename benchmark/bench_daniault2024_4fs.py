"""Benchmark: single-stage compression to 4 fs (1.5 cycles) in a helium-filled MPC (lens-frame solver).

L. Daniault, J. Kaur, G. Gallé, C. Sire, F. Sylla, R. Lopez-Martens, "Sub-2-cycle post-compression
of multi-mJ energy Ti:sapphire laser pulses in a gas-filled multi-pass cell", Opt. Lett. (2024),
open access: arXiv:2410.12525.

Pushes the UPPE as far as an MPC goes: 35 fs -> 4 fs in one stage, nearly octave-spanning spectrum.
Self-steepening, the helium dispersion over 83 m and the Kerr lens at P/Pc ~ 0.74 all matter at once.

                 paper                                          used here
laser            Ti:Sa, 800 nm, 35 fs, 6.7 mJ into the cell      Gaussian 35 fs (or data/ spectrum)
mirrors          enhanced silver, RoC 1.5 m, 4", ~99 %/bounce,   R = 0.987, flat over 430-1250 nm,
                 69 % over 28 passes                            GDD = 0 (assumptions)
geometry         ~3 m, highly concentric, 28 passes              MPC(R=1.5, d=2.981): quoted eigenmode
eigenmode        350 um diameter at the waist, 4.3 mm on mirror  computed
gas              He 1800 mbar (scan 0-2200), circular pol.       n2_eff = 2/3 n2 (scalar code)
compressor       8 pairs PC70 (-640 fs^2) + fused-silica wedges  -640 fs^2 + best fused silica

Measured: 4.0 fs compressed, Fourier limit 3.7 fs, temporal Strehl 90 %, 69 % cell throughput,
homogeneity > 99 % within the FWHM and > 97 % within 1/e^2.

Grid: the carrier is put at the centre (in frequency) of the 450-1400 nm window (~681 nm). The pulse
spectrum is centred at 800 nm explicitly, beam sizes are scaled from 800 nm, and every spectral
phase (GDD, glass) is expanded about 800 nm (framework.spectral_phase).

Pressure scan (paper Fig. 2): run once per pressure, e.g.
    for p in 0.4 0.8 1.2 1.5 1.8 2.0 2.2; do python bench_daniault2024_4fs.py --set pressure_bar=$p --label p$p; done
    python analyze.py --sweep daniault2024_4fs pressure_bar rms_width_nm ftl_fs comp_fs comp_peak_ratio V_min_e2
Not modelled: ionisation (the peak intensity is reported against the He barrier-suppression value),
polarisation dynamics beyond the 2/3 factor, chirped-mirror GDD oscillations, residual input phase.
"""
import numpy as np
import thunder4d as th
from framework import (Benchmark, Ref, band_mirror, metric, spectral_phase, beam_pixels,
                       integrated_power, fourier_limit, best_phase, fwhm, load_digitized)


def n_fused_silica(lam_um):
    """Malitson 1965."""
    l2 = lam_um ** 2
    return np.sqrt(1 + 0.6961663 * l2 / (l2 - 0.0684043 ** 2) + 0.4079426 * l2 / (l2 - 0.1162414 ** 2)
                   + 0.8974794 * l2 / (l2 - 9.896161 ** 2))


class Daniault2024(Benchmark):
    name = "daniault2024_4fs"
    title = "6.7 mJ, 35 fs -> 4 fs, He 1800 mbar, 28 passes (Daniault et al., Opt. Lett. 2024)"
    paper = {"citation": "L. Daniault et al., Opt. Lett. (2024)", "url": "https://arxiv.org/abs/2410.12525"}
    params = dict(
        lambda0=800e-9,
        energy=6.7e-3,           # paper: energy sent into the MPC
        fwhm=35e-15,             # paper; residual input GDD and TOD were cancelled
        input_spectrum_csv="data/daniault2024_input_spectrum.csv",   # optional measured spectrum (nm, I)
        gas="He",
        pressure_bar=1.8,        # paper optimum; scan 0.4 ... 2.2 bar
        n2_1bar=3.5e-25,         # literature, linear polarisation (uncertain by ~15 %)
        circular=True,           # paper: circular polarisation -> n2_eff = 2/3 n2
        R=1.5, d=2.981,          # paper: RoC 1.5 m, ~3 m; d reproduces the quoted eigenmode
        n_passes=28,             # paper
        mirror_R=0.987,          # paper: 69 % over 28 passes
        band=(430e-9, 1250e-9),  # assumption: enhanced silver flat over the broadened spectrum
        R_out=0.9,               # assumption: reflectivity far outside the band
        mirror_gdd=0.0,          # assumption
        cm_gdd=-640.0,           # paper: 8 pairs of PC70
        fs_scan_mm=(-2.0, 6.0),  # fused-silica thickness scanned in the compressor
        lambda_min=450e-9, lambda_max=1400e-9,
        window_fwhm=120e-15,     # sets the time window (x time_window_factor): the output is chirped
        injection="linear",
    )
    reference = {
        "w0_um": Ref(175, 0.05, note="350 um diameter"),
        "w_mirror_um": Ref(2150, 0.10, note="~4.3 mm diameter"),
        "P_over_Pc": Ref(0.74, 0.05, kind="abs"),
        "transmission": Ref(0.69, 0.02, kind="abs"),
        "ftl_fs": Ref(3.7, 0.10),
        "comp_fs": Ref(4.0, 0.15),
        "comp_peak_ratio": Ref(0.90, 0.10, kind="abs", note="temporal Strehl"),
        "V_min_fwhm": Ref(0.99, kind="min"),
        "V_min_e2": Ref(0.97, kind="min"),
        "I_peak_max_Wcm2": Ref(1.5e15, kind="max", note="He barrier suppression"),
    }
    measured = {"spectrum": "data/daniault2024_spectrum.csv"}   # optional, digitised Fig. 3 (nm, I)
    presets = [dict(box_factor=8, points_per_waist=8, time_window_factor=8),
               dict(box_factor=7, points_per_waist=7, time_window_factor=7),
               dict(box_factor=6, points_per_waist=6, time_window_factor=7),
               dict(box_factor=5, points_per_waist=5, time_window_factor=6)]
    quick_preset = dict(box_factor=5, points_per_waist=3, time_window_factor=4)
    record_every = 2
    compression_up = 8
    gdd_bounds = (-1500, 500)

    def make_cell(self, P, n_passes):
        n2 = P["n2_1bar"] * (2 / 3 if P["circular"] else 1.0)
        gas = th.Material(P["gas"], P["pressure_bar"], n2_1bar=n2)
        return th.MPC(R=P["R"], d=P["d"], medium=gas, n_passes=n_passes, mirror_gdd_fs2=P["mirror_gdd"],
                      mirror_reflectivity=band_mirror(P["mirror_R"], *P["band"], R_out=P["R_out"]))

    def grid_carrier(self, P):
        return 2 / (1 / P["lambda_min"] + 1 / P["lambda_max"])     # centre of the window in frequency

    def spectrum(self, grid, P):
        meas = load_digitized(P.get("input_spectrum_csv"))
        if meas is not None:
            return th.measured_spectrum(grid, meas[0], meas[1]), "measured spectrum, flat phase"
        return super().spectrum(grid, P)

    def extra_metrics(self, ctx):
        """Compression as in the experiment: fixed chirped mirrors + fused-silica thickness scan."""
        grid, P, up = ctx.grid, ctx.P, self.compression_up
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
        t_tl, P_tl = fourier_limit(grid, Epix, up)
        t_e, P_e = integrated_power(grid, Epix, phi_cm + L * phi_mm, up)
        strehl = P_e.max() / P_tl.max()
        R = self.reference
        M = {"comp_fs": metric(fwhm(t_e, P_e), "fs", f"compressed, {P['cm_gdd']:.0f} fs2 + {L:.2f} mm FS",
                               R["comp_fs"]),
             "comp_peak_ratio": metric(strehl, "", "temporal Strehl ratio (paper compressor)", R["comp_peak_ratio"]),
             "fs_thickness_mm": metric(L, "mm", "fused silica needed on top of the chirped mirrors")}
        C = {"t_comp_fs": t_e, "P_comp": P_e, "fs_scan_mm": scan, "fs_scan_peak": pk / P_tl.max()}
        return M, C


if __name__ == "__main__":
    Daniault2024().main()
