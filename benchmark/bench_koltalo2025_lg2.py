"""Benchmark: bulk MPC with an LG_0,2 vortex beam, 451 uJ, 180 fs (lens-frame solver).

V. Koltalo et al., "Energy scaling in a compact bulk multi-pass cell enabled by Laguerre-Gaussian
single-vortex beams", APL Photonics 10, 046110 (2025), doi:10.1063/5.0253348 (arXiv:2412.12613).

Vortex beams in a bulk MPC: an LG_0,2 beam spreads the energy on a ring, so the same broadening as a
200 uJ gaussian beam needs 451 uJ (Table 1). Tests a non-gaussian, phase-singular input through the
plates, the Kerr matching with the critical power of the mode (Pc x 8, Eq. 2) and the
conservation of the topological charge. Same cell as the Westerberg benchmarks (lund_bulk_mpc.py).
The three orders are three scripts: bench_koltalo2025_lg1/2/3.py.

                 paper                                          used here
laser            Pharos, 1030 nm, 180 fs                        gaussian 180 fs, flat phase
beam             LG_0,2 from an SLM (hot spots, diffraction    ideal LG_0,2 (exp(i l phi)), size of the
                 ring), 451 uJ                                    cell eigenmode, 451 uJ
cell             RoC 200 mm, d = 382 mm, 14 passes, 2 x 1 mm    same; R = 99.9 %, AR 0.05 %/face
                 AR FS plates 285 mm apart, < 10 mbar between   (assumptions); vacuum between the
                 them                                           plates, air 1 bar outside
nonlinearity     Pcrit(FS) = 4.16 MW at 1030 nm                 n2 = 2.8e-20 m^2/W (same Pcrit)
beam quality     input M2 1.1 x 1.1 (Westerberg Sec. II C; same   M2 = 1.1 (l+1) (M2_input = 1.1 on
                 Pharos, attenuated from 700 uJ)                the ideal LG_0,l): n2 / 1.1^2
injection        Kerr matching found by Bayesian optimisation   moment method, Pc of the LG_0,2 mode,
                 (Fig. 2)                                       intensity-weighted power
compressor       12 bounces on dispersive mirrors, -2400 fs^2   -2400 fs^2 (and the best GDD)

Measured (200 uJ gaussian-equivalent broadening): FTL ~42 fs for every order, SHG-FROG 42-47 fs with
a TOD side pulse, input / output peak power 1.9 / 6.9 GW (Table 2, after the compressor),
MPC transmission > 90 %, ring homogeneity of the collimated output IH = 0.1, WH = 0.025 (Table 3),
topological charge conserved (spatially resolved FTS). The ideal vortex of the simulation needs more
energy than the measured one for the same broadening (paper: hot spots from the SLM), so the
simulation is expected to broaden somewhat less than the measurement at the same energy.

Usage: python bench_koltalo2025_lg2.py [--quick] [--device gpu] [--set energy=...] ...
"""
import lund_bulk_mpc as lund

from framework import Ref


class Koltalo2025_LG2(lund.Koltalo2025):
    name = "koltalo2025_lg2"
    title = "bulk MPC, LG_0,2 vortex, 451 uJ, 180 fs (Koltalo et al., APL Photonics 10, 046110, 2025)"
    params = {**lund.Koltalo2025.params, "energy": 451e-6, "vortex": 2}
    reference = {
        "ftl_fs": Ref(42, 0.10, note="FTL around 42 fs for all orders"),
        "comp_fs": Ref(45, 0.10, note="SHG-FROG 42-47 fs (TOD side pulse)"),
        "comp_peak_power_GW": Ref(6.9, 0.25, note="Table 2, after the compressor (losses, TOD)"),
        "transmission": Ref(0.90, kind="min", note="MPC transmission > 90 %"),
        "charge_fraction": Ref(0.9, kind="min", note="topological charge conserved"),
        "IH_out": Ref(0.1, kind="max", note="measured (SLM hot spots): the ideal beam can only do better"),
        "WH_out": Ref(0.025, kind="max", note="measured: the ideal beam can only do better"),
        "P_over_Pc_plate": Ref(57, kind="info", note="Table 2 input peak power / Pc of LG_0,2 in FS"),
        "mirror_spot_dev": Ref(0.05, kind="max", note="Kerr-matched mode held over the passes"),
    }


if __name__ == "__main__":
    Koltalo2025_LG2().main()
