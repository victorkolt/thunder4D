"""Gaussian vs top-hat near field through the same MPC.

QUICK = True : toy cell (runs in a few minutes on a laptop) to check the workflow.
QUICK = False: template for a real stage (edit the numbers; expect a much larger grid).

Outputs: figures in ./out_<name>/ and a run_<name>.npz with the per-pass history.
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import thunder4d as m
from thunder4d import diagnostics as dg

QUICK = True

if QUICK:
    lam0, fwhm, energy = 1030e-9, 200e-15, 1.2e-3
    gas = m.Material("Ar", 1.0)
    cell = m.MPC.herriott(R=0.5, N=20, k=3, branch="planar", medium=gas)  # 20 passes, 54 deg Gouy/pass, re-entrant
    lam_min, lam_max = 900e-9, 1200e-9
    box_factor = 14                   # generous: top-hat -> Airy rings / higher-order modes
else:
    # ---- template: fill with your stage parameters ----
    lam0, fwhm, energy = 1030e-9, 350e-15, 18e-3
    gas = m.Material("Ar", 0.130)
    cell = m.MPC(R=2.0, d=3.6, medium=gas, n_passes=29, mirror_gdd_fs2=-50)
    lam_min, lam_max = 920e-9, 1160e-9
    box_factor = 12

grid = m.suggest_grid(cell, lam0, lam_min, lam_max, fwhm, box_factor=box_factor)
print(grid.describe())

# ---------------------------------------------------------------- 1) the pulse
E = m.gaussian_spectrum(grid, fwhm_fs=fwhm * 1e15)
# E = m.measured_spectrum(grid, wl_nm, I)                   # measured spectrum instead
# E = m.add_spectral_phase(grid, E, gdd_fs2=500, tod_fs3=2e3)

probe = m.Pulse.at_focus(grid, E, m.Beam(m.gaussian(1e-3)), energy)
match = cell.match(probe, power="peak")                      # 2) nonlinear mode matching
w0 = match["w0"]

beams = {
    # gaussian with a flat waist at the cell centre = perfectly (peak-power) matched
    "gauss": lambda: m.Pulse.at_focus(grid, E, m.Beam(m.gaussian(w0)), energy),
    # top-hat collimated near field (4 mm radius) focused by the lens that maximises the overlap
    # with the matched mode: this is what the experiment does
    "tophat": lambda: m.Pulse.from_near_field(grid, E, m.Beam(m.top_hat(4e-3)), energy,
                                              nf_size=12e-3, nf_points=192, target_waist=w0),
    # examples of aberrations / STCs (uncomment to run):
    # "gauss_astig": lambda: m.Pulse.from_near_field(
    #     grid, E, m.Beam(m.gaussian(4e-3), zernike={"astig_0": 0.1}), energy,
    #     nf_size=16e-3, nf_points=192, target_waist=w0),
    # "gauss_spatial_chirp": lambda: m.Pulse.from_near_field(
    #     grid, E, m.Beam(m.gaussian(4e-3), spatial_chirp_mm_per_nm=(0.02, 0)), energy,
    #     nf_size=16e-3, nf_points=192, target_waist=w0),
}

# Long runs: `python compare_gauss_tophat.py [max_seconds]` stops cleanly after the time budget,
# writes a checkpoint, and resumes from it when launched again.
MAX_SECONDS = float(sys.argv[1]) if len(sys.argv) > 1 else None

runs = {}
for name, make in beams.items():
    ckpt = f"ckpt_{name}.pkl"
    pulse = make()          # cheap; overwritten by the checkpoint if one exists
    sim = m.MPCSimulation(pulse, cell, phi_max=0.02, mode_reference=w0, max_order=8)
    if os.path.exists(ckpt):
        sim.resume(ckpt)
    if not sim.done:
        print(f"\n===== {name} (from pass {sim.passes_done}) =====")
        sim.run(max_seconds=MAX_SECONDS)
        sim.checkpoint(ckpt)
        if not sim.done:
            print("time budget reached: run the script again to continue")
            sys.exit(0)
        sim.save(f"run_{name}.npz", include_field=False)
    runs[name] = sim

# ---------------------------------------------------------------- 3) figures
um, nm = 1e6, 1e9


def show_sgram(ax, x, lam, S, title, xlabel):
    S = S / S.max()
    ax.pcolormesh(lam * nm, x, S, shading="auto", cmap="inferno")
    # normalised per wavelength version is often more telling for radial chirp:
    ax.set_xlabel("wavelength (nm)")
    ax.set_ylabel(xlabel)
    ax.set_title(title, fontsize=9)


for name, sim in runs.items():
    out = f"out_{name}"
    os.makedirs(out, exist_ok=True)
    h, g = sim.history, sim.grid
    P = np.array(h["pass"])

    # spatio-spectral traces: input / output, near field (cell centre) and far field (angle)
    fig, axs = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    lim = 4 * w0 * um
    for col, i in enumerate((0, -1)):
        show_sgram(axs[0, col], sim.x * um, sim.lam, h["sgram_x"][i],
                   f"{name}: near field S(x, lambda), pass {P[i]}", "x (um)")
        axs[0, col].set_ylim(-lim, lim)
        show_sgram(axs[1, col], sim.theta * 1e3, sim.lam, h["sgram_theta"][i],
                   f"{name}: far field S(theta, lambda), pass {P[i]}", "theta (mrad)")
        axs[1, col].set_ylim(-4 * g.lambda0 / (np.pi * w0) * 1e3, 4 * g.lambda0 / (np.pi * w0) * 1e3)
    fig.savefig(f"{out}/spectrograms.png", dpi=130)

    # cleaning / homogeneity versus pass
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.6), constrained_layout=True)
    axs[0].plot(P, h["eta00_total"], "o-", label="HG00 fraction")
    axs[0].plot(P, h["V_near"], "s-", label="V near field")
    axs[0].plot(P, h["V_far"], "^-", label="V far field")
    axs[0].set_xlabel("pass"); axs[0].legend(); axs[0].set_title("cleaning / homogeneity")
    of = np.array(h["order_frac"])
    for N in range(of.shape[1]):
        axs[1].semilogy(P, of[:, N] + 1e-8, label=f"m+n={N}" if N < of.shape[1] - 1 else "higher")
    axs[1].set_xlabel("pass"); axs[1].set_title("HG modal energy fractions"); axs[1].legend(fontsize=7)
    axs[2].plot(P, h["M2x_mean"], "o-", label="M2 x")
    axs[2].plot(P, h["M2y_mean"], "s-", label="M2 y")
    axs[2].plot(P, h["B"], "k--", label="B (rad)")
    axs[2].set_xlabel("pass"); axs[2].legend()
    fig.savefig(f"{out}/vs_pass.png", dpi=130)

    # fundamental-mode fraction vs wavelength, in and out
    fig, ax = plt.subplots(figsize=(6, 3.6), constrained_layout=True)
    S_out = h["spectrum"][-1] / h["spectrum"][-1].max()
    ax.fill_between(sim.lam * nm, 0, S_out, color="0.85", label="output spectrum")
    ax.plot(sim.lam * nm, h["eta00"][0], label="eta00 input")
    ax.plot(sim.lam * nm, h["eta00"][-1], label="eta00 output")
    ax.set_xlim(lam_min * nm, lam_max * nm); ax.set_ylim(0, 1.02); ax.legend(); ax.set_xlabel("nm")
    fig.savefig(f"{out}/eta_vs_lambda.png", dpi=130)

    # fluence in / out and radial GDD (radial chirp) at the output
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)
    ext = [g.xs[0] * um, g.xs[-1] * um, g.ys[0] * um, g.ys[-1] * um]
    for a, i, t in ((axs[0], 0, "input"), (axs[1], -1, "output")):
        a.imshow(h["fluence"][i].T, origin="lower", extent=ext, cmap="viridis")
        a.set_xlim(-lim, lim); a.set_ylim(-lim, lim); a.set_title(f"fluence {t} (cell centre)")
    gmap = dg.local_gdd_map(g, sim.Aw)
    r, gr = dg.radial_profile(g, gmap)
    axs[2].plot(r * um, gr)
    axs[2].set_xlim(0, 2 * w0 * um); axs[2].set_xlabel("r (um)"); axs[2].set_ylabel("local GDD (fs^2)")
    axs[2].set_title("radial chirp at output")
    fig.savefig(f"{out}/fluence_gdd.png", dpi=130)
    plt.close("all")

    c = dg.compress(g, sim.Aw)
    print(f"{name}: output eta00 = {h['eta00_total'][-1]:.4f} (input {h['eta00_total'][0]:.4f}), "
          f"V_far = {h['V_far'][-1]:.4f}, compressed {c['fwhm_fs']:.1f} fs with {c['gdd_fs2']:.0f} fs^2 "
          f"(TL {c['tl_fwhm_fs']:.1f} fs)")
