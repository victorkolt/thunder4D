"""Benchmark: oscillator-driven bulk MPC, 0.5 uJ at 10 MHz, 120 fs -> 8.1 fs Fourier limit, one 2 mm FS
plate at the centre, 68 passes (lens-frame solver).

Z. Hu, K. Fritsch, O. Pronin, "Supercontinuum generation in the low-peak-power regime using a
single-stage multipass cell", arXiv:2511.08337 (2025): main result. Shared cell and assumptions in
hu2025_bulk_mpc.py.

                 paper                                          used here
laser            FLINT oscillator, 0.5 uJ, 10 MHz, 5 W, 120 fs,  gaussian 120 fs, flat phase, 0.5 uJ,
                 ~5 MW (wavelength and spectrum not given)       1030 nm
mirrors          RoC 50 mm, -50 fs^2 each, d = 97 mm, 68 passes  MPC(R=0.05, d=0.097, 68 passes)
plate            2 mm FS, AR, at the centre                     same, 0.1 % per face
nonlinearity     B = 0.4 rad / pass, 27.3 rad total, air ~4e-3   n2(FS) table value
compressor       5 dispersive mirrors (-150 fs^3 each) + 22 cm  best GDD, and best GDD + TOD
                 CaF2 for the GDD

Measured: spectrum 800-1200 nm, Fourier limit 8.1 fs, compressed ~10 fs, transmission 75 %, output
M2 1.107, beam 4.2 x 4.0 mm (ellipticity 0.94).

Usage: python bench_hu2025_flint_500nJ.py [--quick] [--device gpu] [--set n2_fs=2.5e-20] ...
"""
import hu2025_bulk_mpc as hu

from framework import Ref


class Hu2025_Flint500nJ(hu.Hu2025):
    name = "hu2025_flint_500nJ"
    title = "oscillator-driven bulk MPC, 0.5 uJ, 120 fs -> 8 fs (Hu et al., arXiv:2511.08337, 2025)"
    params = {**hu.Hu2025.base_params, "fwhm": 120e-15, "window_fwhm": 200e-15, "energy": 0.5e-6,
              "d": 0.097, "n_passes": 68, "plate_z": 0.0}
    reference = {
        "ftl_fs": Ref(8.1, 0.10, note="Fourier limit of the measured spectrum"),
        "gdd_tod_comp_fs": Ref(11.5, kind="max", note="measured ~10 fs (CaF2 + TOD mirrors); an ideal GDD + TOD "
                                                      "compressor does at least as well"),
        "B_rad": Ref(27.3, 0.25, note="0.4 rad per pass (paper)"),
        "B_fraction_plates": Ref(0.99, kind="min", note="air ~4e-3 rad of 27.3"),
        "transmission": Ref(0.75, 0.05, kind="abs", note="measured 75 %"),
        "M2x": Ref(1.2, kind="max", note="measured M2 1.107"),
        "M2y": Ref(1.2, kind="max", note="measured M2 1.107"),
        "mirror_spot_dev": Ref(0.10, kind="max", note="Kerr-matched mode held over the passes"),
        "P_over_Pc_plate": Ref(1.0, kind="info", note="~5 MW, about the critical power of FS"),
    }


if __name__ == "__main__":
    Hu2025_Flint500nJ().main()
