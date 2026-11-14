"""Benchmark: hybrid multi-pass multi-plate MPC, 112 uJ, 1.24 ps -> 39 fs, six 1 mm FS plates in air.

M. Seidel, P. Balla, C. Li, G. Arisholm, L. Winkelmann, I. Hartl, C. M. Heyl, "Factor 30 pulse
compression by hybrid multipass multiplate spectral broadening", Ultrafast Science 2022, 9754919
(2022), doi:10.34133/2022/9754919 (open access, arXiv:2111.12834).

Thirty-fold compression in one bulk stage: six thin plates near the centre of a near-concentric cell,
64 passes (384 plate crossings), ~20 Pc in the plates, B per pass up to ~4 pi/5. With nonlinear
matching, the plates form a weak quasi-waveguide: larger beam at the centre, smaller spots on the
mirrors than the linear eigenmode. Tests several Kerr lenses in one pass (moment-method matching
through six plates), a picosecond input broadened 33 times (long time window, wide spectrum),
and dispersion-limited SPM (the plates' dispersion saturates the broadening).

                 paper                                          used here
laser            Yb:YAG, burst mode, 1030 nm, 112 uJ, near-FTL  gaussian 1240 fs, flat phase, 112 uJ
                 1240 fs with a pedestal, ~80 MW                (pedestal not modelled: ~85 MW)
beam             (input M2 not given); simulation: M2 = 1       ideal gaussian, as the authors' simulation
mirrors          RoC 200 mm (f = 100 mm), quarter-wave stacks,  MPC(R=0.2, d=0.385, 64 passes), GDD 0,
                 d ~385 mm, 32 round trips                      R = 99.9 % (assumption)
plates           six 1 mm FS, AR, centred, spaced by 16 mm      at -40, -24, -8, 8, 24, 40 mm (spacing
                                                                taken centre to centre), AR 0.02 %/face
                                                                (assumption: MPC transmission ~80 %)
medium           ambient air (Kerr of air in their simulation)  air 1 bar, n2 = 8e-24 m^2/W
nonlinearity     n2(FS) = 2.5e-16 cm^2/W in their simulation    2.5e-20 m^2/W
                 (Z-scan in the cell: 2.25e-16)
injection        nonlinear mode matching at 112 uJ (ABCD with   Kerr-matched (moment method) at the peak
                 Kerr lenses, kappa = 0.43)                     power
compressor       72 chirped-mirror bounces, -14 400 fs^2,       -14 400 fs^2 (and the best GDD),
                 78 % transmission, 5 % depolarization          compressor_T = 0.78 x 0.95

Measured at full power: FTL 36 fs (broadening factor 33; 35.7 fs mean over 17 h), SHG-FROG 39 fs at
-14 400 fs^2, 65 uJ usable -> 1.1 GW, MPC transmission ~80 %, spectral homogeneity V_y = 97.3 %,
M2 ~1.2 up to 90 uJ and 1.6 x 1.5 at 112 uJ (attributed to four-wave mixing with the input pedestal).
Authors' simulation (100 uJ, 1250 fs gaussian, nonlinear matching): peak irradiance in the plates below
500 GW/cm^2 (mean 285 GW/cm^2), up to 2 TW/cm^2 with linear matching.
Not modelled: the input pedestal (central spectral peak, four-wave-mixing side bands, the M2 rise at
full power), mirror and AR coating spectra.

Usage: python bench_seidel2022_112uJ.py [--quick] [--device gpu] [--set energy=90e-6] ...
"""
import numpy as np
import plate_mpc as pm

import thunder4d as th

from framework import Ref, metric


class Seidel2022_112uJ(pm.PlateMPCBenchmark):
    name = "seidel2022_112uJ"
    title = "multi-pass multi-plate MPC, 112 uJ, 1.24 ps -> 39 fs (Seidel et al., Ultrafast Sci. 2022)"
    paper = {"citation": "M. Seidel et al., Ultrafast Science 2022, 9754919 (2022)",
             "doi": "10.34133/2022/9754919", "url": "https://arxiv.org/abs/2111.12834"}
    params = {
        "lambda0": 1030e-9,
        "fwhm": 1240e-15,
        "window_fwhm": 1500e-15,       # time window (x time_window_factor): the output stays ~ps long
        "energy": 112e-6,
        "R": 0.2, "d": 0.385, "n_passes": 64,
        "plate_t": 1e-3,
        "n_plates": 6,
        "plate_spacing": 0.016,        # centre to centre
        "n2_fs": 2.5e-20,
        "n2_air": 8.0e-24,
        "ar_R": 2e-4,                  # assumption
        "mirror_R": 0.999,             # assumption
        "mirror_gdd": 0.0,
        "vortex": 0,
        "M2_input": 1.0,
        "m2_model": "none",            # the authors simulated a fundamental gaussian (M2 = 1)
        "matching_power": "peak",      # matched for the peak power at 112 uJ
        "injection": "kerr-matched",
        "lambda_min": 920e-9, "lambda_max": 1160e-9,
        "compressor_gdd": -14400.0,
        "compressor_T": 0.78 * 0.95,   # compressor transmission x (1 - depolarization)
    }
    gdd_bounds = (-30000.0, 0.0)
    reference = {
        "ftl_fs": Ref(36, 0.10, note="FTL at full power (35.7 fs mean over 17 h)"),
        "broadening_factor": Ref(33, 0.10, note="1.2 ps -> 36 fs Fourier limit"),
        "comp_fs": Ref(39, 0.10, note="SHG-FROG at -14 400 fs2 (72 x -200 fs2)"),
        "peak_power_out_GW": Ref(1.1, 0.25, note="65 uJ usable after the compressor"),
        "transmission": Ref(0.80, kind="info", note="~80 % measured; set here by the assumed coating losses"),
        "V_weighted_x": Ref(0.973, kind="min", note="V_y = 97.3 % measured"),
        "M2x": Ref(1.3, kind="max", note="1.2 up to 90 uJ; 1.6 x 1.5 at 112 uJ from the pedestal (not modelled)"),
        "M2y": Ref(1.3, kind="max", note="see M2x"),
        "I_peak_max_Wcm2": Ref(5.6e11, kind="max", note="authors' simulation < 500 GW/cm2 at 100 uJ, x 112/100"),
        "P_over_Pc_plate": Ref(20, kind="info", note="'approximately 20 times above the critical power'"),
        "mirror_spot_dev": Ref(0.05, kind="max", note="Kerr-matched mode held over the passes"),
    }

    def make_cell(self, P, n_passes):
        self.check_m2_model(P)
        k = self.kerr_scale(P)
        fs = th.Solid("FS", n2=P["n2_fs"] * k)
        ar = th.Coating(P["ar_R"], name=f"AR {P['ar_R']*100:.2g} %")
        n = int(P["n_plates"])
        z = (np.arange(n) - (n - 1) / 2) * P["plate_spacing"]
        plates = [th.Plate(fs, P["plate_t"], float(zi), coating=ar, name=f"FS plate {i + 1}")
                  for i, zi in enumerate(z)]
        air = th.Material("air", 1.0, n2_1bar=P["n2_air"] * k)
        return th.MPC(P["R"], P["d"], air, n_passes, plates=plates, wavelength=P["lambda0"],
                      mirror_reflectivity=P["mirror_R"], mirror_gdd_fs2=P["mirror_gdd"])

    def extra_metrics(self, ctx):
        M, C = super().extra_metrics(ctx)
        ftl = ctx.metrics["ftl_fs"]["value"]
        M["broadening_factor"] = metric(ctx.P["fwhm"] * 1e15 / ftl, "", "input FWHM / output Fourier limit",
                                        self.reference.get("broadening_factor"))
        return M, C


if __name__ == "__main__":
    Seidel2022_112uJ().main()
