"""Benchmark: bulk MPC, 250 uJ, 180 fs -> 42 fs, two 1 mm fused-silica plates (lens-frame solver).

S. Westerberg et al., "Influence of the laser pulse duration in high-order harmonic generation",
APL Photonics 10, 096103 (2025), doi:10.1063/5.0272968 (open access).

A uJ-class bulk MPC far above the critical power of the plates (P ~ 250-300 Pc in fused silica):
tests the plates of thunder4d (dispersion, Kerr, AR faces, group-velocity frames), the gas regions
(evacuated cell between the plates, air near the mirrors) and the Kerr-matched injection with the
moment method. Same cell as bench_westerberg2025_100uJ.py and bench_koltalo2025_lg*.py (shared setup
in lund_bulk_mpc.py).

                 paper                                          used here
laser            Pharos (700 uJ max, attenuated),              gaussian 180 fs, flat phase, 250 uJ
                 1030 nm, 180 fs, 250 uJ at the MPC
mirrors          RoC 200 mm, d = 382 mm, 7 round trips           MPC(R=0.2, d=0.382, 14 passes),
                                                                R = 99.9 %, GDD = 0 (assumptions)
plates           2 x 1 mm FS, AR, 280 mm apart, centred         same, AR 0.05 % per face (assumption)
nonlinearity     n2(FS) = 3e-16 cm^2/W                          3e-20 m^2/W
beam quality     input M2 1.1 x 1.1 (Westerberg Sec. II C; same   M2_input = 1.1, m2_model 'intensity':
                 Pharos, attenuated from 700 uJ)                n2 / (M2x M2y) (see lund_bulk_mpc.py)
medium           plates = windows of a cell at ~5 mbar           vacuum between the plates, air 1 bar
                                                                outside
injection        mode-matching telescope (Kerr lensing          Kerr-matched (moment method) at the
                 accounted for)                                 intensity-weighted power
compressor       transmission gratings, tuned per energy        best GDD

Measured at 250 uJ: 42 fs (SHG-FROG), FTL 39 fs, 4.8 GW peak power, MPC transmission > 94 %, output
M2 1.1 x 1.3, peak power up to ~300 Pc of fused silica.
Not modelled: measured input spectrum and phase (the measured Pharos pulse has less peak power for the
same energy than the gaussian), mirror coatings, residual air (5 mbar) in the evacuated cell.

Usage: python bench_westerberg2025_250uJ.py [--quick] [--device gpu] [--set n2_fs=2.5e-20] ...
"""
import lund_bulk_mpc as lund

from framework import Ref


class Westerberg2025_250uJ(lund.Westerberg2025):
    name = "westerberg2025_250uJ"
    title = "bulk MPC, 250 uJ, 180 fs, 2 x 1 mm FS (Westerberg et al., APL Photonics 10, 096103, 2025)"
    params = {**lund.Westerberg2025.params, "energy": 250e-6}
    gdd_bounds = (-6000.0, 0.0)
    reference = {
        "ftl_fs": Ref(39, 0.10, note="FTL of the measured spectrum"),
        "best_comp_fs": Ref(42, 0.10, note="SHG-FROG, grating compressor tuned"),
        "peak_power_out_GW": Ref(4.8, 0.15, note="from the FROG and the energy after the compressor (T > 90 %)"),
        "transmission": Ref(0.94, kind="min", note="MPC transmission > 94 %"),
        "M2x": Ref(1.3, kind="max", note="measured 1.1 x 1.3"),
        "M2y": Ref(1.3, kind="max", note="measured 1.1 x 1.3"),
        "P_over_Pc_plate": Ref(300, kind="info", note="'up to 300 times' Pc of FS"),
        "mirror_spot_dev": Ref(0.05, kind="max", note="Kerr-matched mode held over the passes"),
    }


if __name__ == "__main__":
    Westerberg2025_250uJ().main()
