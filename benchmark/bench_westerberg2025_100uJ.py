"""Benchmark: bulk MPC, 100 uJ, 180 fs -> 75 fs, two 1 mm fused-silica plates (lens-frame solver).

S. Westerberg et al., "Influence of the laser pulse duration in high-order harmonic generation",
APL Photonics 10, 096103 (2025), doi:10.1063/5.0272968 (open access).

Lowest energy of the paper's duration scan: P ~ 100 Pc in the plates, B ~ 6 rad. Same cell and
assumptions as bench_westerberg2025_250uJ.py (see that script and lund_bulk_mpc.py), including the
input M2 of 1.1 x 1.1 (m2_model 'intensity' by default) and the same Pharos pulse at every energy
(attenuated from 700 uJ).

Measured: "changing the input energy in the MPC from 100 uJ to 225 uJ, with a 25 uJ spacing, resulting
in compressed pulses of 75 fs, 65 fs, 57 fs, 49 fs and 45 fs" (Sec. II D): six energies, five durations,
so assigning 75 fs to 100 uJ is an assumption (if it belongs to 125 uJ, the 100 uJ pulse is longer).
MPC transmission > 94 %. Fig. 3 also shows the simulated (SISYFOS, measured input spectrum, input M2
1.1) and measured Fourier limits versus energy, not tabulated in the text.

Usage: python bench_westerberg2025_100uJ.py [--quick] [--device gpu] ...
"""
import lund_bulk_mpc as lund

from framework import Ref


class Westerberg2025_100uJ(lund.Westerberg2025):
    name = "westerberg2025_100uJ"
    title = "bulk MPC, 100 uJ, 180 fs, 2 x 1 mm FS (Westerberg et al., APL Photonics 10, 096103, 2025)"
    params = {**lund.Westerberg2025.params, "energy": 100e-6}
    gdd_bounds = (-8000.0, 0.0)
    reference = {
        "best_comp_fs": Ref(75, 0.15, note="SHG-FROG; assumed first point of the 100-225 uJ scan (5 durations "
                            "quoted for 6 energies)"),
        "transmission": Ref(0.94, kind="min", note="MPC transmission > 94 %"),
        "mirror_spot_dev": Ref(0.05, kind="max", note="Kerr-matched mode held over the passes"),
    }


if __name__ == "__main__":
    Westerberg2025_100uJ().main()
