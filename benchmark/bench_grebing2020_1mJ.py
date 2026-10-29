"""Benchmark: 1 mJ, 200 fs, argon-filled Herriott cell (lens-frame solver).

C. Grebing, M. Müller, J. Buldt, H. Stark, J. Limpert, "Kilowatt-average-power compression of
millijoule pulses in a gas-filled multi-pass cell", Opt. Lett. 45, 6250 (2020),
doi:10.1364/OL.408998 (open access: arXiv:2103.00893).

Clean, textbook SPM stage in the mJ range: moderate peak power (~4.5 GW, P/Pc ~ 0.2), dielectric
mirrors. The input follows the authors' own simulation of this stage (Müller et al., Opt. Lett. 46,
2678, 2021): Gaussian beam, Gaussian 200 fs pulse of 994 uJ with a TOD of -1e4 fs^3.

                 paper                                          used here
mirrors          RoC 0.6 m, R > 99.95 %, |GDD| < 50 fs^2         R = 0.9995, GDD = +35 fs^2, band
                 over 970-1130 nm                               edges 10 nm outside (assumption)
geometry         d = 1183 mm, 26 passes, re-entrant              herriott(R=0.6, N=26, k=1)
eigenmode        0.30 mm diameter at focus, 2.5 mm on mirror    computed
gas              Ar 700 mbar, n2 = 1e-23 m^2/W at 1 atm          same
injection        linear eigenmode                                same
compressor       6 chirped mirrors, -1923 fs^2                   same GDD

Measured: -20 dB width ~120 nm, Fourier limit 27 fs, compressed 31 fs (autocorrelation), 96 %
transmission (cell + compressor), M2 < 1.15, weighted homogeneity 98.7 % (x) / 99.4 % (y).
Not modelled: folded off-axis geometry, windows, measured input spectrum and phase.

Usage: python bench_grebing2020_1mJ.py [--quick] [--device gpu] [--set n2_1bar=1.1e-23] ...
       (python bench_grebing2020_1mJ.py -h for all options)
"""
import numpy as np
import thunder4d as th
from framework import Benchmark, Ref, band_mirror, metric, spectral_phase


class Grebing2020(Benchmark):
    name = "grebing2020_1mJ"
    title = "1 mJ, 200 fs, Ar 700 mbar, 26 passes (Grebing et al., Opt. Lett. 45, 6250, 2020)"
    paper = {"citation": "C. Grebing et al., Opt. Lett. 45, 6250 (2020)",
             "doi": "10.1364/OL.408998", "url": "https://arxiv.org/abs/2103.00893"}
    params = dict(
        lambda0=1030e-9,
        energy=994e-6,           # Müller 2021, their simulation of this stage (Grebing 2020: ~1 mJ)
        fwhm=200e-15,            # paper
        tod_fs3=-1e4,            # Müller 2021: reproduces the slight spectral asymmetry
        gas="Ar",
        pressure_bar=0.7,        # paper
        n2_1bar=1.0e-23,         # paper
        R=0.6, N=26, k=1,        # paper: RoC 0.6 m, d = 1183 mm, 26 passes, re-entrant
        n_passes=26,
        mirror_R=0.9995,         # paper: R > 99.95 % over 970-1130 nm
        band=(960e-9, 1140e-9),  # assumption: 50 % edges 10 nm outside the specified band
        mirror_gdd=35.0,         # within the paper's specs (|GDD| < 50 fs^2)
        compressor_gdd=-1923.0,  # paper
        lambda_min=920e-9, lambda_max=1160e-9,
        injection="linear",      # "linear" (paper) or "nonlinear" (Kerr-matched waist)
    )
    reference = {
        "w0_um": Ref(150, 0.05, note="0.30 mm diameter"),
        "w_mirror_um": Ref(1250, 0.05, note="2.5 mm diameter"),
        "span20_nm": Ref(120, 0.15),
        "ftl_fs": Ref(27, 0.10),
        "comp_fs": Ref(31, 0.15, note="autocorrelation"),
        "transmission": Ref(0.96, 0.03, kind="abs", note="cell + compressor in the paper"),
        "M2x": Ref(1.15, kind="max"),
        "M2y": Ref(1.15, kind="max"),
        "V_weighted_x": Ref(0.987, 0.015, kind="abs"),
        "B_rad": Ref(13, kind="info", note="~13 beam-averaged"),
    }
    measured = {"spectrum": "data/grebing2020_spectrum.csv"}    # optional, digitised Fig. 2 (nm, I)
    presets = [dict(box_factor=8, points_per_waist=8, time_window_factor=8),
               dict(box_factor=7, points_per_waist=7, time_window_factor=7),
               dict(box_factor=6, points_per_waist=6, time_window_factor=6)]

    def make_cell(self, P, n_passes):
        gas = th.Material(P["gas"], P["pressure_bar"], n2_1bar=P["n2_1bar"])
        return th.MPC.herriott(R=P["R"], N=P["N"], k=P["k"], medium=gas, branch="concentric",
                               n_passes=n_passes, mirror_gdd_fs2=P["mirror_gdd"],
                               mirror_reflectivity=band_mirror(P["mirror_R"], *P["band"]))

    def extra_metrics(self, ctx):
        """Output spectrum and on-axis phase after the paper's compressor: the input of a second
        stage (Müller et al. 2021)."""
        from thunder4d import diagnostics as dg
        grid, P = ctx.grid, ctx.P
        M = {"d_mm": metric(ctx.cell.d * 1e3, "mm", "mirror separation", Ref(1183, 0.005))}
        A = th.to_host(ctx.sim.Aw)
        lam, S = dg.total_spectrum(grid, A)
        idx = np.nonzero(grid.valid)[0]
        idx = idx[np.argsort(grid.lam[idx])]
        wc = 2 * np.pi * th.C0 / P["lambda0"]
        phase = np.angle(A[0, 0, :] * np.exp(1j * spectral_phase(grid, wc, P["compressor_gdd"])))[idx]
        C = {"stage_out_lam_nm": lam * 1e9, "stage_out_S": S, "stage_out_phase": np.unwrap(phase)}
        return M, C


if __name__ == "__main__":
    Grebing2020().main()
