"""Benchmark: hybrid air-bulk MPC, 400 uJ, 276 fs -> 27 fs, one 1 mm FS plate off-centre (lens frame).

A. Omar, M. Hoffmann, G. Galle, F. Sylla, C. J. Saraceno, "Hybrid air-bulk multi-pass cell compressor
for high pulse energies with full spatio-temporal characterization", Opt. Express 32, 13235 (2024),
doi:10.1364/OE.513732 (open access, arXiv:2311.18468).

A true hybrid: ambient air and one fused-silica plate both contribute to the nonlinearity (FS ~36 % of
the B-integral in the authors' simulation), at P ~ 350 Pc in the plate, with the plate 20 cm from one
mirror. Tests an off-centre plate (the plate is at +z on odd passes and -z on even ones), the air
of the whole cell as Kerr medium, and the Kerr-matched injection with a strong Kerr lens far from the
centre ("ignoring the strong self-focusing effect of the FS plate can lead to the collapse of the beam").

                 paper                                          used here
laser            Carbide, 400 uJ, 100 kHz, spectrum FWHM 9.4 nm  gaussian spectrum, FTL 200 fs, + GDD
                 (FTL 200 fs), chirped to 276 fs, 1.38 GW,      13 700 fs^2 -> 276 fs (sign of the chirp
                 lambda 1037 nm                                 not given: positive assumed), 1037 nm
beam             M2 1.14 x 1.14 (slightly asymmetric)            ideal gaussian, as the authors' simulation
                                                                ("fundamental transverse mode TEM00")
mirrors          RoC 2 m, HR (~0 GDD), d = 134 cm, 23 round     MPC(R=2, d=1.34, 46 passes), GDD 0,
                 trips = 46 passes, 1.4 mm spot on the mirrors  R = 99.95 % (assumption)
plate            1 mm FS, AR, 20 cm from one mirror, in air     plate at +0.47 m from the centre (20 cm
                                                                from mirror B), AR 0.025 %/face (assumpt.)
nonlinearity     Pcr(FS) from "350 x Pcr at 1.38 GW", 1037 nm   n2(FS) = 3.0e-20 m^2/W (same Pcr)
                 air: Schwarz et al. 2012 (value not quoted),   n2(air) = 2.4e-23 m^2/W: effective value,
                 delayed (Raman) response in their model         3 x the electronic Kerr (see below)
injection        curved mirrors, nonlinear matching not ideal   Kerr-matched (moment method), weighted
                 in both axes                                   power
compressor       17 dispersive mirrors, -3000 fs^2              -3000 fs^2 (and the best GDD)

Measured: SHG-FROG 27 fs at -3000 fs^2, 10.2 GW, 78 % of the energy in the main pulse (2 x FWHM),
overall efficiency > 91 %, output M2 1.17 x 1.23, V-parameter 98.1 / 98.4 % (x / y), Strehl 0.97.
Before compression: autocorrelation FWHM 621 fs (their simulation: pulse 616 fs, AC 635 fs).
Authors' 3D simulation (HR mirrors): B = 16.56 rad in total, of which 5.9 rad (35.6 %) in the plate,
FTL 25 fs, compressed 27 fs at ~12 GW, simulated M2 < 1.05, V ~99.8 %.

Air (calibrated): the authors' model includes the delayed (rotational Raman) response of air, which
thunder4d does not have; for ~250 fs pulses it adds to the instantaneous Kerr effect. With the
electronic value 8e-24 m^2/W the plate takes 67 % of the B-integral (authors: 36 %) and the Fourier
limit is 39 fs instead of 25 fs; the effective 2.4e-23 m^2/W gives the authors' split (41 % in the
plate) together with their spectrum, compression, peak power and stretched output.

B-integral: the authors' 16.56 rad is half of the on-axis peak value of the simulation for the same
spectrum (34 rad), i.e. their B is computed with P / (pi w^2) = I_peak / 2. This is inferred from
the agreement of everything else, not stated in the paper; the reference is 2 x 16.56 rad.

Usage: python bench_omar2024_400uJ.py [--quick] [--device gpu] [--set n2_air=...] ...
"""
import plate_mpc as pm

import thunder4d as th

from framework import Ref

FS_N2_OMAR = 3.0e-20      # n2 = lambda^2 / (2 pi n0 Pcr), Pcr = 1.38 GW / 350 at 1037 nm, n0 = 1.450


class Omar2024_400uJ(pm.PlateMPCBenchmark):
    name = "omar2024_400uJ"
    title = "hybrid air-bulk MPC, 400 uJ, 276 fs -> 27 fs (Omar et al., Opt. Express 32, 13235, 2024)"
    paper = {"citation": "A. Omar et al., Opt. Express 32, 13235 (2024)", "doi": "10.1364/OE.513732",
             "url": "https://arxiv.org/abs/2311.18468"}
    params = {
        "lambda0": 1037e-9,
        "fwhm": 200e-15,               # FTL of the measured spectrum
        "gdd_fs2": 13700.0,            # chirp to 276 fs (sign assumed)
        "window_fwhm": 450e-15,        # time window: the output is stretched to ~620 fs
        "energy": 400e-6,
        "R": 2.0, "d": 1.34, "n_passes": 46,
        "plate_t": 1e-3,
        "plate_z": 0.47,               # plate centre from the cell centre: 20 cm from mirror B
        "n2_fs": FS_N2_OMAR,
        "n2_air": 2.4e-23,             # effective (calibrated): electronic 8e-24 x 3 for the delayed Raman
        "ar_R": 2.5e-4,                # assumption
        "mirror_R": 0.9995,            # assumption (HR quarter-wave stack)
        "mirror_gdd": 0.0,
        "vortex": 0,
        "M2_input": 1.14,              # measured, but the authors simulated an ideal TEM00 beam
        "m2_model": "none",
        "matching_power": "weighted",
        "injection": "kerr-matched",
        "lambda_min": 920e-9, "lambda_max": 1170e-9,
        "compressor_gdd": -3000.0,
        "compressor_T": 1.0,           # not given (overall efficiency > 91 %)
    }
    gdd_bounds = (-8000.0, 0.0)
    # quick check: 5 waists left 6e-3 of the fluence at the edge (the plate's strong Kerr lens far from the
    # centre widens the beam on the first passes); 7 waists, same coarse sampling
    quick_preset = {"box_factor": 7, "points_per_waist": 3, "time_window_factor": 4}
    calibrated = {"n2_air": {"paper": 8.0e-24,
                             "note": "electronic Kerr of air x 3, standing for the delayed Raman response of "
                                     "the authors' model (not quoted); fitted on their plate / air B split"}}
    reference = {
        "comp_fs": Ref(27, 0.10, note="SHG-FROG at -3000 fs2"),
        "comp_peak_power_GW": Ref(10.2, 0.20, note="measured (their simulation: ~12 GW)"),
        "main_pulse_fraction": Ref(0.78, kind="min", note="measured (FROG); the measured input spectrum (cut at "
                                                          "1042 nm) and mirror phases are not modelled"),
        "ftl_fs": Ref(25, 0.10, note="authors' simulation (HR mirrors); measured spectrum close to it"),
        "transmission": Ref(0.91, kind="min", note="overall efficiency > 91 %"),
        "chirped_out_fwhm_fs": Ref(616, 0.15, note="authors' simulation (measured AC FWHM 621 fs)"),
        "B_rad": Ref(33.1, 0.20, note="2 x 16.56 rad: the authors' B uses I_peak / 2 (inferred, see docstring)"),
        "B_fraction_plates": Ref(0.356, 0.25, note="authors' simulation: 5.9 of 16.56 rad in FS"),
        "M2x": Ref(1.3, kind="max", note="measured 1.17 x 1.23 (input 1.14); simulated < 1.05"),
        "M2y": Ref(1.3, kind="max", note="measured 1.17 x 1.23"),
        "V_weighted_x": Ref(0.98, kind="min", note="measured 98.1 % (x), 98.4 % (y)"),
        "P_over_Pc_plate": Ref(350, kind="info", note="'350x higher than the critical power in FS'"),
        "mirror_spot_dev": Ref(0.25, kind="info", note="the pulse stretches ~2x in the cell (positive GDD only): "
                                                      "matched at the input power, the beam breathes later "
                                                      "(the authors also see the caustic change)"),
    }

    def make_cell(self, P, n_passes):
        self.check_m2_model(P)
        k = self.kerr_scale(P)
        fs = th.Solid("FS", n2=P["n2_fs"] * k)
        ar = th.Coating(P["ar_R"], name=f"AR {P['ar_R']*100:.2g} %")
        plate = th.Plate(fs, P["plate_t"], P["plate_z"], coating=ar)
        air = th.Material("air", 1.0, n2_1bar=P["n2_air"] * k)
        return th.MPC(P["R"], P["d"], air, n_passes, plates=[plate], wavelength=P["lambda0"],
                      mirror_reflectivity=P["mirror_R"], mirror_gdd_fs2=P["mirror_gdd"])


if __name__ == "__main__":
    Omar2024_400uJ().main()
