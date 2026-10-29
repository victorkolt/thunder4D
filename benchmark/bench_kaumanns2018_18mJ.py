"""Benchmark: 18 mJ, 1.3 ps, argon-filled Herriott cell (lens-frame solver).

M. Kaumanns, V. Pervak, D. Kormin, V. Leshchenko, A. Kessel, M. Ueffing, Y. Chen, T. Nubbemeyer,
"Multipass spectral broadening of 18 mJ pulses compressible from 1.3 ps to 41 fs",
Opt. Lett. 43, 5877 (2018), doi:10.1364/OL.43.005877.

High-energy regime: 13 GW (P/Pc ~ 0.5), large total nonlinear phase (~55 rad from a fit of the
measured spectrum), compression factor 33, near-concentric cell close to its stability edge: the
mirror spot (2.7 mm) is 15 times the waist (185 um), which is exactly where the lens frame pays off,
the grid only has to hold the waist. The authors' SPM model of their output (Eq. 1, phi = 55 rad) is
stored as a second reference spectrum.

                 paper                                          used here
mirrors          300 x 130 mm, RoC 1.5 m, R > 99.9 % over        R = 0.999, GDD = 0, edges 10 nm
                 965-1100 nm, GDD = 0 at 1030 nm                outside the band (assumption)
geometry         d ~ 1.99 R ~ 3 m, 45 passes                     MPC(R=1.5, d=2.986): quoted eigenmode
eigenmode        0.37 mm diameter at focus, 5.4 mm on mirror    computed
gas              Ar 600 mbar, n2(1 bar) ~ (10 +- 1)e-24 m^2/W    same
input            18.6 mJ, 1.3 ps, 1.1 nm, 1030 nm               transform-limited Gaussian 1.3 ps
compressor       17 chirped mirrors, -9400 fs^2                  same GDD

Measured: 17.8 mJ out (95.7 %), Fourier limit 39 fs, compressed 41 fs with 62 % of the
Fourier-limited peak power (their model: 41.6 fs, 67 %), M2 1.17 x 1.18, weighted homogeneity
96.7 % (x) / 97.3 % (y). Variant: --set pressure_bar=0.7 --set energy=17e-3 --set lambda_min=920e-9.
Not modelled: ionisation (run just below breakdown, ~18.3 mJ at 600 mbar), ASE / pre-pulses.
"""
import numpy as np
import thunder4d as th
from framework import Benchmark, Ref, band_mirror, metric, spectral_similarity


class Kaumanns2018(Benchmark):
    name = "kaumanns2018_18mJ"
    title = "18 mJ, 1.3 ps, Ar 600 mbar, 45 passes (Kaumanns et al., Opt. Lett. 43, 5877, 2018)"
    paper = {"citation": "M. Kaumanns et al., Opt. Lett. 43, 5877 (2018)", "doi": "10.1364/OL.43.005877"}
    params = dict(
        lambda0=1030e-9,
        energy=18.6e-3,          # paper
        fwhm=1.3e-12,            # paper (Fourier-limited input)
        gas="Ar",
        pressure_bar=0.6,        # paper
        n2_1bar=1.0e-23,         # paper: (10 +- 1) x 1e-24 m^2/W at 1 bar
        R=1.5, d=2.986,          # paper: RoC 1.5 m, d ~ 1.99 R; 2.986 m gives the quoted eigenmode
        n_passes=45,             # paper
        mirror_R=0.999,          # paper: R > 99.9 % over 965-1100 nm
        band=(955e-9, 1110e-9),  # assumption
        compressor_gdd=-9400.0,  # paper
        phi_fit=55.0,            # paper: phase of the SPM model fitted to the measured spectrum
        lambda_min=960e-9, lambda_max=1110e-9,
        injection="linear",
    )
    reference = {
        "w0_um": Ref(185, 0.05, note="0.37 mm diameter"),
        "w_mirror_um": Ref(2700, 0.05, note="5.4 mm diameter"),
        "P_over_Pc": Ref(0.5, kind="info"),
        "transmission": Ref(0.957, 0.02, kind="abs"),
        "ftl_fs": Ref(39, 0.10),
        "comp_fs": Ref(41, 0.15, note="model 41.6"),
        "comp_peak_ratio": Ref(0.62, 0.10, kind="abs", note="model 0.67"),
        "best_gdd_fs2": Ref(-9400, 0.15),
        "M2x": Ref(1.17, kind="max"),
        "M2y": Ref(1.18, kind="max"),
        "V_weighted_x": Ref(0.967, 0.02, kind="abs"),
        "B_rad": Ref(55, kind="info", note="55 (fit), 59 +- 6"),
        "similarity_spm_model": Ref(1.0, 0.05, kind="abs"),
    }
    measured = {"spectrum": "data/kaumanns2018_spectrum.csv"}   # optional, digitised Fig. 2 (nm, I)
    presets = [dict(box_factor=8, points_per_waist=8, time_window_factor=6),
               dict(box_factor=7, points_per_waist=7, time_window_factor=5),
               dict(box_factor=6, points_per_waist=6, time_window_factor=4.5)]
    quick_preset = dict(box_factor=5, points_per_waist=3, time_window_factor=3.5)
    record_every = 5             # 45 passes: full diagnostics every 5 passes (and the last)
    gdd_bounds = (-20000, 0)

    def make_cell(self, P, n_passes):
        gas = th.Material(P["gas"], P["pressure_bar"], n2_1bar=P["n2_1bar"])
        return th.MPC(R=P["R"], d=P["d"], medium=gas, n_passes=n_passes,
                      mirror_reflectivity=band_mirror(P["mirror_R"], *P["band"]))

    def extra_metrics(self, ctx):
        """The paper's SPM model: A(t) exp(i phi |A(t)|^2) for the 1.3 ps Gaussian, on the same grid."""
        from thunder4d import diagnostics as dg
        grid, P = ctx.grid, ctx.P
        a = np.exp(-2 * np.log(2) * (grid.t / P["fwhm"]) ** 2)
        E = grid.hfft.ft_t(a * np.exp(1j * P["phi_fit"] * a ** 2))
        idx = np.nonzero(grid.valid)[0]
        idx = idx[np.argsort(grid.lam[idx])]
        lam_m = grid.lam[idx]
        S_m = np.abs(E[idx]) ** 2 / lam_m ** 2
        S_m /= S_m.max()
        lam, S = dg.total_spectrum(grid, ctx.sim.Aw)
        M = {"similarity_spm_model": metric(spectral_similarity(lam * 1e9, S / S.max(), lam_m * 1e9, S_m), "",
                                            f"spectral similarity with the paper's SPM model (phi = {P['phi_fit']:.0f})",
                                            self.reference["similarity_spm_model"])}
        C = {"S_ref_lam_nm": lam_m * 1e9, "S_ref": S_m,
             "S_ref_label": np.array(f"SPM model, phi = {P['phi_fit']:.0f} rad (paper)")}
        return M, C


if __name__ == "__main__":
    Kaumanns2018().main()
