"""Benchmark: bulk MPC with a gaussian beam, 200 uJ, 180 fs: reference of the vortex benchmarks.

V. Koltalo et al., "Energy scaling in a compact bulk multi-pass cell enabled by Laguerre-Gaussian
single-vortex beams", APL Photonics 10, 046110 (2025), doi:10.1063/5.0253348 (arXiv:2412.12613).

The gaussian beam that sets the "200 uJ gaussian-equivalent broadening" of the paper: the LG_0,l
beams of bench_koltalo2025_lg1/2/3.py (350, 451, 610 uJ, Table 1) were tuned in energy to give the
same spectrum. Comparing the four runs tests the energy scaling with the vortex order, e.g.

    python analyze.py --latest koltalo2025_gaussian_200uJ koltalo2025_lg1 koltalo2025_lg2 koltalo2025_lg3

Same cell and assumptions as the vortex benchmarks (lund_bulk_mpc.py); same cell as Westerberg et al.
with the plates 285 mm apart and Pcrit(FS) = 4.16 MW.

                 paper                                          used here
laser            Pharos, 1030 nm, 180 fs, 200 uJ, gaussian beam gaussian 180 fs, flat phase, 200 uJ
                 (input peak power 0.8 GW, Table 2)             (1.04 GW: the measured pulse has less
                                                                peak power than a gaussian)
cell             RoC 200 mm, d = 382 mm, 14 passes, 2 x 1 mm    same; R = 99.9 %, AR 0.05 %/face
                 AR FS plates 285 mm apart, < 10 mbar between   (assumptions); vacuum between the
                 them                                           plates, air 1 bar outside
nonlinearity     Pcrit(FS) = 4.16 MW at 1030 nm                 n2 = 2.8e-20 m^2/W (same Pcrit)
injection        Kerr matching found by Bayesian optimisation   moment method, intensity-weighted power
compressor       12 bounces on dispersive mirrors, -2400 fs^2   -2400 fs^2 (and the best GDD)

Measured: FTL ~42 fs, SHG-FROG 42-47 fs with a TOD side pulse (Fig. 6 a), output peak power 3.1 GW
after the compressor (Table 2, boost x 3.9), MPC transmission > 90 %.

Usage: python bench_koltalo2025_gaussian_200uJ.py [--quick] [--device gpu] [--set energy=...] ...
"""
import lund_bulk_mpc as lund

from framework import Ref


class Koltalo2025_Gaussian200uJ(lund.Koltalo2025):
    name = "koltalo2025_gaussian_200uJ"
    title = "bulk MPC, gaussian beam, 200 uJ, 180 fs (Koltalo et al., APL Photonics 10, 046110, 2025)"
    params = {**lund.Koltalo2025.params, "energy": 200e-6, "vortex": 0}
    reference = {
        "ftl_fs": Ref(42, 0.10, note="FTL around 42 fs"),
        "comp_fs": Ref(45, 0.10, note="SHG-FROG 42-47 fs (TOD side pulse)"),
        "comp_peak_power_GW": Ref(3.1, 0.25, note="Table 2, after the compressor (losses, TOD)"),
        "transmission": Ref(0.90, kind="min", note="MPC transmission > 90 %"),
        "M2x": Ref(1.3, kind="max", note="good beam quality (not quantified for this cell in the paper)"),
        "M2y": Ref(1.3, kind="max"),
        "P_over_Pc_plate": Ref(192, kind="info", note="Table 2 input peak power 0.8 GW / 4.16 MW"),
        "mirror_spot_dev": Ref(0.05, kind="max", note="Kerr-matched mode held over the passes"),
    }


if __name__ == "__main__":
    Koltalo2025_Gaussian200uJ().main()
