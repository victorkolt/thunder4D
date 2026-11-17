"""Bulk MPC of Westerberg et al., APL Photonics 10, 096103 (2025), doi:10.1063/5.0272968.

Pharos, 1030 nm, 180 fs, up to 250 uJ. Herriott cell R = 200 mm, d = 382 mm, 7 round trips
(14 passes); two 1 mm AR-coated fused-silica plates 280 mm apart, placed symmetrically around the
centre, windows of a small cell evacuated to ~5 mbar (vacuum region between the plates, air
between the plates and the mirrors). Reported at 250 uJ: 42 fs compressed (FTL 39 fs), 4.8 GW
peak power, P up to ~300 Pc.

Runs in ~10 s per energy on a laptop CPU (coarse lens-frame grid).
"""
import numpy as np

import thunder4d as t4
from thunder4d import diagnostics as dg

LAM0 = 1030e-9


def cell(n2=None, air=True):
    fs = t4.Solid("FS") if n2 is None else t4.Solid("FS", n2=n2)
    plates = [t4.Plate(fs, 1e-3, +0.14, coating="ar"), t4.Plate(fs, 1e-3, -0.14, coating="ar")]
    if not air:            # everything in vacuum
        return t4.MPC(0.2, 0.382, None, 14, plates=plates, wavelength=LAM0)
    return t4.MPC(0.2, 0.382, t4.Material("air", 1.0), 14, plates=plates,
                  regions=[(-0.14, 0.14, t4.vacuum())], wavelength=LAM0)


def run(energy, n2=None, air=True, verbose=False):
    c = cell(n2, air)
    g = t4.suggest_lens_grid(c, LAM0, 880e-9, 1220e-9, 180e-15, box_factor=6, points_per_waist=6,
                             time_window_factor=6, dtype=np.complex64)
    p = t4.Pulse.cell_matched(g, t4.gaussian_spectrum(g, fwhm_fs=180), c, energy, power="weighted",
                              verbose=verbose)
    sim = t4.LensFrameSimulation(p, c, phi_max=0.05)
    sim.run(verbose=verbose)
    out = dg.compress(g, sim.Aw)
    out["B"] = sim.solver.B
    out["eta00"] = sim.history["eta00_total"][-1]
    out["transmission"] = sim.history["energy"][-1] / sim.history["energy"][0]
    return sim, out


if __name__ == "__main__":
    print(cell().report(cell().nonlinear_mode(1.3e9, LAM0)))
    for n2, air in ((None, False), (None, True), (3e-20, True)):   # FS n2: table (2.2e-20) / paper (3e-20)
        label = ("n2 = 2.2e-20" if n2 is None else "n2 = 3.0e-20") + (", air + vacuum" if air else ", all vacuum ")
        for E in (100e-6, 150e-6, 250e-6):
            _, o = run(E, n2, air)
            print(f"{label} | {E*1e6:4.0f} uJ | B = {o['B']:5.2f} rad | compressed {o['fwhm_fs']:5.1f} fs "
                  f"(FTL {o['tl_fwhm_fs']:5.1f} fs) at {o['gdd_fs2']:7.0f} fs^2 | P = {o['peak_power']/1e9:4.2f} GW"
                  f" | eta00 = {o['eta00']:.3f}")
    print("paper, 250 uJ: 42 fs (FTL 39 fs), 4.8 GW")
