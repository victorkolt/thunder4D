"""Benchmark: bulk MPC at 1 MHz, 0.37 uJ, 100 fs -> 10 fs, one 2 mm FS plate 4 mm off-centre, 58 passes
(lens-frame solver).

Z. Hu, K. Fritsch, O. Pronin, "Supercontinuum generation in the low-peak-power regime using a
single-stage multipass cell", arXiv:2511.08337 (2025): verification with an attenuated Pharos and a
D-scan, with the authors' own simulation. Shared cell and assumptions in hu2025_bulk_mpc.py.

                 paper                                          used here
laser            Pharos UP attenuated, 0.4 uJ, 1 MHz, 90 fs;     authors' simulation: gaussian 100 fs,
                 authors' simulation: 0.37 uJ, 100 fs            0.37 uJ, flat phase, 1030 nm
mirrors          RoC 50 mm, -50 fs^2 each, d = 98 mm, 58 passes  MPC(R=0.05, d=0.098, 58 passes)
plate            2 mm FS, AR, 4 mm from the centre              +4 mm (towards mirror B), 0.1 % per face
nonlinearity     B = 20.4 rad total (0.4 rad / pass)            n2(FS) table value
losses           0.3 % per bounce, 0.2 % per crossing (sim.)    same
compressor       D-scan; before it ~ +1.8e3 fs^2, -3e2 fs^3      best GDD, and best GDD + TOD

Measured (D-scan): Fourier limit 10.32 fs, compressed 10.0 fs with 78.5 % of the Fourier-limited peak
(pedestal from uncompensated higher orders). The authors' simulation reproduces the measured spectrum.

Usage: python bench_hu2025_pharos_370nJ.py [--quick] [--device gpu] [--set energy=0.4e-6,fwhm=90e-15] ...
"""
import hu2025_bulk_mpc as hu

from framework import Ref


class Hu2025_Pharos370nJ(hu.Hu2025):
    name = "hu2025_pharos_370nJ"
    title = "bulk MPC, 0.37 uJ, 100 fs -> 10 fs, off-centre plate (Hu et al., arXiv:2511.08337, 2025)"
    params = {**hu.Hu2025.base_params, "fwhm": 100e-15, "window_fwhm": 200e-15, "energy": 0.37e-6,
              "d": 0.098, "n_passes": 58, "plate_z": 0.004}
    # off-centre plate: the beam breathes more than in the FLINT case and its wings reach the edge of an
    # 8-waist box (edge_fraction 1.7e-3 at n2 = 2.6e-20): 11 waists, same sampling per waist
    presets = [{"box_factor": 11, "points_per_waist": 8, "time_window_factor": 8},
               {"box_factor": 10, "points_per_waist": 7, "time_window_factor": 7},
               {"box_factor": 9, "points_per_waist": 6, "time_window_factor": 6}]
    reference = {
        "ftl_fs": Ref(10.32, kind="info", note="D-scan Fourier limit; the simulation gives ~8.4 fs with the same cell "
                                                "and band as the FLINT case (its 8.3 fs matches): the D-scan spectrum "
                                                "or the authors' 0.37 uJ / 100 fs input may differ"),
        "gdd_tod_comp_fs": Ref(11.5, kind="max", note="D-scan 10.0 fs (78.5 % of the FTL peak); an ideal GDD + "
                                                      "TOD compressor does at least as well"),
        "B_rad": Ref(20.4, 0.25, note="paper"),
        "B_fraction_plates": Ref(0.99, kind="min", note="air negligible"),
        "M2x": Ref(1.2, kind="max", note="not measured here; same cell as the FLINT case (1.107)"),
        "mirror_spot_dev": Ref(0.10, kind="max", note="Kerr-matched mode held over the passes"),
    }


if __name__ == "__main__":
    Hu2025_Pharos370nJ().main()
