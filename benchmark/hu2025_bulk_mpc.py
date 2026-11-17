"""Shared cell of the Hu et al. benchmarks (not a benchmark itself): one 2 mm AR-coated fused-silica
plate in a near-concentric MPC driven at sub-uJ energies.

Z. Hu, K. Fritsch, O. Pronin, "Supercontinuum generation in the low-peak-power regime using a
single-stage multipass cell", arXiv:2511.08337 (2025).

Low peak power (P ~ Pc of fused silica, a few MW) and many passes (58-68), 120 fs -> ~8 fs Fourier
limit with B ~ 0.4 rad per pass: tests a long, gentle accumulation of SPM in one plate, a near-zero
net dispersion per pass (plate +38 fs^2 against mirrors -50 fs^2), self-steepening over a 400 nm wide
spectrum, and air that plays no role (~4e-3 rad in total, paper).

Assumptions common to both (not given in the paper): carrier 1030 nm, gaussian spectrum with a flat
phase, ideal TEM00 input (no M2 given for the input), n2(FS) calibrated (not quoted, see below),
cavity mirrors with GDD only (no TOD), losses: 0.2 % per plate crossing (authors'
simulation), mirrors see below, Kerr-matched injection at the weighted power (the paper
mode-matches to the eigenmode with a telescope).

Dispersion regime: plate +38 fs^2 against one mirror -50 fs^2 per pass leaves ~-10 fs^2 net, so the
pulse self-compresses inside the cell (output ~8 fs before any compressor) and the result depends
strongly on what the paper does not give: the mirrors' GDD curve, TOD and reflection band (68
bounces on dispersive mirrors). The measured spectrum (800-1200 nm at ~1e-3) is narrower than the
simulated one. With flat mirrors the simulated Fourier limit is 7.2 fs (paper 8.1 fs); a reflection
band of 800-1200 nm (edges at R/2, 68 bounces) gives 8.3 fs, so the band is on by default
(mirror_lmin, mirror_lmax; 0 turns it off). The band is an assumption: the paper only quotes the
measured spectrum.

n2(FS): with the band, n2 = 2.2e-20 m^2/W gives the measured Fourier limit (8.3 fs) but B = 19 rad
(paper 27.3), and 3.0e-20 gives B = 28.5 rad but a 9.1 fs Fourier limit (more energy pushed to the
band edges and lost); the default 2.6e-20 sits between the two.
"""
import numpy as np

import plate_mpc as pm

import thunder4d as th


class Hu2025(pm.PlateMPCBenchmark):
    calibrated = {"mirror_R": {"paper": 0.997, "note": "authors' simulation: 0.3 % per bounce averaged over a flat "
                                                      "band; with the 800-1200 nm band edges modelled, the in-band "
                                                      "value giving the measured 75 % transmission"},
                  "n2_fs": {"paper": 2.2e-20, "note": "not quoted in the paper (2.2e-20 = thunder4d table value); "
                                                      "2.2e-20 fits the FLINT Fourier limit (8.3 fs) but gives B 30 % "
                                                      "low, 3.0e-20 fits B (28.5 rad) but gives 9.1 fs: 2.6e-20 is "
                                                      "the compromise"}}
    paper = {"citation": "Z. Hu, K. Fritsch, O. Pronin, arXiv:2511.08337 (2025)",
             "url": "https://arxiv.org/abs/2511.08337"}
    base_params = {
        "lambda0": 1030e-9,
        "R": 0.05,
        "plate_t": 2e-3,
        "n2_fs": 2.6e-20,              # calibrated (paper: not quoted), see the docstring
        "n2_air": 8.0e-24,
        "ar_R": 1e-3,                  # 0.2 % loss per plate crossing (authors' simulation)
        "mirror_R": 0.9985,            # in-band reflectivity (calibrated: 75 % transmission with the band)
        "mirror_gdd": -50.0,           # dispersive cavity mirrors, -50 fs^2 each
        "mirror_lmin": 800e-9,         # reflection band of the cavity mirrors [m] (0 = flat R): super-
        "mirror_lmax": 1200e-9,        # gaussian edges of order 16, R = mirror_R / 2 at the edges
        "vortex": 0,
        "M2_input": 1.0,
        "m2_model": "none",
        "matching_power": "weighted",
        "injection": "kerr-matched",
        "lambda_min": 720e-9, "lambda_max": 1400e-9,
        "compressor_gdd": None,        # GDD of the compressor not given: best GDD, and best GDD + TOD
        "compressor_T": 1.0,
    }
    gdd_bounds = (-3000.0, 1000.0)

    def make_cell(self, P, n_passes):
        self.check_m2_model(P)
        k = self.kerr_scale(P)
        fs = th.Solid("FS", n2=P["n2_fs"] * k)
        ar = th.Coating(P["ar_R"], name=f"AR {P['ar_R']*100:.2g} %")
        plate = th.Plate(fs, P["plate_t"], P["plate_z"], coating=ar)
        air = th.Material("air", 1.0, n2_1bar=P["n2_air"] * k)
        refl = P["mirror_R"]
        if P["mirror_lmin"] > 0 and P["mirror_lmax"] > P["mirror_lmin"]:
            lc, hw, R0 = (P["mirror_lmin"] + P["mirror_lmax"]) / 2, (P["mirror_lmax"] - P["mirror_lmin"]) / 2, refl

            def refl(lam):
                return R0 * np.exp(-np.log(2) * ((np.asarray(lam) - lc) / hw) ** 16)   # R0/2 at the edges
        return th.MPC(P["R"], P["d"], air, n_passes, plates=[plate], wavelength=P["lambda0"],
                      mirror_reflectivity=refl, mirror_gdd_fs2=P["mirror_gdd"])
