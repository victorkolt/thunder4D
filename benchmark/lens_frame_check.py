"""Lens frame vs standard solver: near-concentric Ar cell, 1 mJ, 200 fs, 2 passes.
Usage: python lens_frame_check.py [cpu|gpu|auto]"""
import sys, time
import numpy as np
import thunder4d as th
from thunder4d import diagnostics as dg

device = sys.argv[1] if len(sys.argv) > 1 else "auto"
lam0 = 1030e-9
cell = th.MPC(R=0.5, d=0.94, medium=th.Material("Ar", 1.0), n_passes=2)
grids = {
    "standard": th.suggest_grid(cell, lam0, 920e-9, 1150e-9, 200e-15, box_factor=6,
                                points_per_waist=6, time_window_factor=6, device=device),
    "lens frame": th.suggest_lens_grid(cell, lam0, 920e-9, 1150e-9, 200e-15, box_factor=8,
                                       points_per_waist=6, time_window_factor=6, device=device),
}
spec = {}
for name, g in grids.items():
    print(f"{name:10s}: {g.describe()}")
    E = th.gaussian_spectrum(g, fwhm_fs=200)
    probe = th.Pulse.at_focus(g, E, th.Beam(th.gaussian(1e-3)), 1e-3)
    w0 = cell.nonlinear_mode(probe.peak_power(), lam0)["w0"]
    pulse = th.Pulse.at_focus(g, E, th.Beam(th.gaussian(w0), chromatic_size="eigenmode"), 1e-3)
    Sim = th.LensFrameSimulation if name == "lens frame" else th.MPCSimulation
    t = time.time()
    sim = Sim(pulse, cell, phi_max=0.02, near_field=False, record_mirror=False)
    sim.run(verbose=False)
    lam, S = dg.total_spectrum(g, sim.Aw)
    c, h = dg.compress(g, sim.Aw), sim.history
    spec[name] = (lam, S / S.max())
    print(f"{'':10s}  B {sim.solver.B:.3f} rad | w_peak {h['w_peak_x'][-1]*1e6:.1f} um | "
          f"eta00 {h['eta00_total'][-1]:.4f} | V {h['V_far'][-1]:.5f} | TL {c['tl_fwhm_fs']:.1f} fs | "
          f"compressed {c['fwhm_fs']:.1f} fs | {time.time() - t:.1f} s")
(la, Sa), (lb, Sb) = spec["standard"], spec["lens frame"]
Sb = np.interp(la, lb, Sb)
sim = np.trapezoid(np.sqrt(Sa * Sb), la) ** 2 / (np.trapezoid(Sa, la) * np.trapezoid(Sb, la))
print(f"spectral similarity: {sim:.5f} (1 = identical)")