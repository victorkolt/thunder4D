"""Shared cell of the Lund bulk-MPC benchmarks (not a benchmark itself: the bench_*.py scripts
subclass these classes, one script per energy / vortex order so that each runs independently).
The Kerr-matched injection and the metrics come from plate_mpc.py.

The same compact Herriott cell is used in both papers:
    Pharos (Light Conversion), 1030 nm, 180 fs; mirrors RoC 200 mm, d = 382 mm, 7 round trips (14
    passes); two 1 mm AR-coated fused-silica plates placed symmetrically around the centre; they are
    the windows of a small evacuated cell (5-10 mbar): vacuum between the plates, air between the
    plates and the mirrors.

  * Westerberg et al., "Influence of the laser pulse duration in high-order harmonic generation",
    APL Photonics 10, 096103 (2025), doi:10.1063/5.0272968: plates 280 mm apart, energy scan
    50-250 uJ (Gaussian beam), grating compressor tuned for every energy.
  * Koltalo et al., "Energy scaling in a compact bulk multi-pass cell enabled by Laguerre-Gaussian
    single-vortex beams", APL Photonics 10, 046110 (2025), doi:10.1063/5.0253348
    (arXiv:2412.12613): plates 285 mm apart, LG_0,l beams (l = 1-3) shaped by an SLM, energies giving
    the same broadening as a 200 uJ Gaussian beam, 12 bounces on dispersive mirrors (-2400 fs^2).

Physics specific to these cells (thunder4d >= 1.6): plates and gas regions in the cell, Kerr-matched
injection with the moment method (P ~ 100-300 Pc in the plates), LG_0,l injection matched with the
critical power of the mode (MPC.nonlinear_mode(vortex=l)).

Input beam quality: the laser is a Pharos (up to 700 uJ, attenuated by a half-wave plate and a thin-film
polarizer: same pulse and beam at every energy) with a measured input M2 of 1.1 x 1.1 (Westerberg et al.,
Sec. II C; assumed identical for Koltalo et al., same laser). As in the authors' own SISYFOS simulation
("we account for the measured M2 to determine the peak intensity"), the field stays the ideal mode
(gaussian or LG_0,l) and the Kerr coefficients of every medium are divided by M2x M2y: the intensity
of a beam M2 times larger in each axis for the same power. For the vortices M2_input is the factor
on the ideal LG_0,l (M2 = 1.1 (l + 1)). Powers, energies and transmission are unchanged. m2_model
'none' gives the ideal beam.

Assumptions common to both (the papers do not give them): Gaussian 180 fs input with a flat phase
(the measured Pharos pulse has less peak power for the same energy: Koltalo, Table 2 quotes 0.8 GW at
200 uJ, the Gaussian gives 1.04 GW, so the simulation broadens slightly more); cavity mirrors
R = 99.9 %, no GDD; residual AR reflectivity 0.05 % per face; air 1 bar outside the evacuated cell;
injection matched at the intensity-weighted power (the papers' telescopes are set for the measured
beam sizes, i.e. a Kerr-matched beam).
"""

import plate_mpc as pm

import thunder4d as th

FS_N2_WESTERBERG = 3.0e-20      # 3e-16 cm^2/W at 1030 nm, quoted by Westerberg et al.
FS_N2_KOLTALO = 2.8e-20         # from Pcrit = 4.16 MW at 1030 nm quoted by Koltalo et al. (n0 = 1.450)


class LundBulkMPC(pm.PlateMPCBenchmark):
    """The Lund cell (both papers). Subclasses set name, title, paper, params, reference."""

    base_params = {
        "lambda0": 1030e-9,
        "fwhm": 180e-15,               # Pharos, both papers
        "energy": 200e-6,
        "R": 0.2, "d": 0.382,          # both papers
        "n_passes": 14,                # 7 round trips
        "plate_t": 1e-3,               # two 1 mm FS plates
        "plate_sep": 0.28,             # distance between the plates [m]
        "n2_fs": FS_N2_WESTERBERG,
        "ar_R": 5e-4,                  # assumption: residual reflectivity per AR face
        "mirror_R": 0.999,             # assumption
        "mirror_gdd": 0.0,             # assumption
        "air": True,                   # air between the plates and the mirrors (evacuated centre)
        "vortex": 0,                   # LG_0,l order (0 = gaussian)
        "M2_input": 1.1,               # measured input M2 (per axis), Westerberg et al. Sec. II C
        "m2_model": "intensity",       # "intensity" (n2 / M2^2) | "none" (see the module docstring)
        "matching_power": "weighted",  # Kerr matching at the "weighted" or "peak" power
        "injection": "kerr-matched",   # label only (framework tables): moment method, see pulse()
        "lambda_min": 880e-9, "lambda_max": 1220e-9,
        "compressor_gdd": None,
        "compressor_T": 1.0,           # compressor transmission, for the peak power after it
    }

    def make_cell(self, P, n_passes):
        self.check_m2_model(P)
        k = self.kerr_scale(P)
        fs = th.Solid("FS", n2=P["n2_fs"] * k)
        ar = th.Coating(P["ar_R"], name=f"AR {P['ar_R']*100:.2g} %")
        z = P["plate_sep"] / 2
        plates = [th.Plate(fs, P["plate_t"], +z, coating=ar), th.Plate(fs, P["plate_t"], -z, coating=ar)]
        air = th.Material("air", 1.0, n2_1bar=8.0e-24 * k) if P["air"] else None
        regions = [(-z, z, th.vacuum())] if P["air"] else ()
        return th.MPC(P["R"], P["d"], air, n_passes, plates=plates, regions=regions, wavelength=P["lambda0"],
                      mirror_reflectivity=P["mirror_R"], mirror_gdd_fs2=P["mirror_gdd"])


# ====================================================================== the two papers
class Westerberg2025(LundBulkMPC):
    """Westerberg et al. 2025: gaussian beam, energy scan, grating compressor tuned per energy."""
    paper = {"citation": "S. Westerberg et al., APL Photonics 10, 096103 (2025)",
             "doi": "10.1063/5.0272968",
             "url": "https://pubs.aip.org/aip/app/article/10/9/096103/3361161"}
    params = {**LundBulkMPC.base_params, "plate_sep": 0.28, "n2_fs": FS_N2_WESTERBERG,
              "compressor_T": 0.90}        # grating compressor transmission > 90 % (paper)


class Koltalo2025(LundBulkMPC):
    """Koltalo et al. 2025: LG_0,l beams at the energies that give the broadening of a 200 uJ
    gaussian beam (Table 1), dispersive-mirror compressor -2400 fs^2."""
    paper = {"citation": "V. Koltalo et al., APL Photonics 10, 046110 (2025)",
             "doi": "10.1063/5.0253348", "url": "https://arxiv.org/abs/2412.12613"}
    params = {**LundBulkMPC.base_params, "plate_sep": 0.285, "n2_fs": FS_N2_KOLTALO,
              "compressor_gdd": -2400.0}
    gdd_bounds = (-6000.0, 0.0)
