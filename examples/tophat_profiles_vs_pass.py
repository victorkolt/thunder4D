"""Evolution, pass after pass, of the 1D beam profile of selected colours (1010 / 1030 / 1050 nm)
and of the spectrally integrated ("global") profile, for a top-hat near field.

Same toy cell as compare_gauss_tophat.py. Resumable:  python tophat_profiles_vs_pass.py [max_seconds]
Figures in ./out_profiles_<injection>/
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import thunder4d as m

lam0, fwhm, energy = 1030e-9, 200e-15, 1.2e-3
gas = m.Material("Ar", 1.0)
cell = m.MPCCell.herriott(R=0.5, N=10, k=3, medium=gas)
grid = m.suggest_grid(cell, lam0, 900e-9, 1200e-9, fwhm, box_factor=14)

# "mirror"  : laser near field imaged onto the first mirror (flat-top on the mirror at pass 1)
# "fourier" : near field in the front focal plane of the matching lens (Airy at the cell centre)
INJECTION = "mirror"

WAVELENGTHS = [1010e-9, 1030e-9, 1050e-9]
BANDWIDTH = 2e-9          # bandpass width (like an interference filter); 0 = single spectral bin

E = m.gaussian_spectrum(grid, fwhm_fs=fwhm * 1e15)
probe = m.Pulse.at_focus(grid, E, m.Beam(m.gaussian(1e-3)), energy)
mm = cell.match(probe, verbose=False)
w0 = mm["w0"]
if INJECTION == "mirror":
    pulse = m.Pulse.imaged_near_field(grid, E, m.Beam(m.top_hat(1.12 * mm["w_mirror"])), energy, cell)
else:
    pulse = m.Pulse.from_near_field(grid, E, m.Beam(m.top_hat(4e-3)), energy,
                                    nf_size=12e-3, nf_points=192, target_waist=w0)

sim = m.MPCSimulation(pulse, cell, phi_max=0.02, mode_reference=w0,
                      profile_wavelengths=WAVELENGTHS, profile_bandwidth=BANDWIDTH)
ckpt = f"ckpt_profiles_{INJECTION}.pkl"
if os.path.exists(ckpt):
    sim.resume(ckpt)
if not sim.done:
    sim.run(max_seconds=float(sys.argv[1]) if len(sys.argv) > 1 else None)
    sim.checkpoint(ckpt)
    if not sim.done:
        print("time budget reached: run again to continue")
        sys.exit(0)
    sim.save(f"run_profiles_{INJECTION}.npz", include_field=False)

# ------------------------------------------------------------------ figures
h = sim.history
P = np.array(h["pass"])
x = sim.prof_coord * 1e6                      # um
th = sim.theta * 1e3                          # mrad
near = np.array(h["prof_near"])               # (npass, nlam+1, Nx)
far = np.array(h["prof_far"])
mir = np.array([v for v in h["prof_mirror"] if v is not None])   # passes 1..N
band = np.array(h["band_energy"])             # (npass, nlam+1)
labels = [f"{l*1e9:.0f} nm" for l in WAVELENGTHS] + ["global"]
ok = band[:, :-1] / band[:, -1:] > 1e-5       # colour not yet generated -> hidden

os.makedirs(f"out_profiles_{INJECTION}", exist_ok=True)
lim_x, lim_th = 4 * w0 * 1e6, 4 * lam0 / (np.pi * w0) * 1e3


def norm_rows(a):
    return a / np.maximum(a.max(axis=-1, keepdims=True), 1e-300)


for tag, data, coord, lim, xl, passes in (
        ("near_field_centre", near, x, lim_x, "x (um)", P),
        ("far_field", far, th, lim_th, "theta (mrad)", P),
        ("mirror_plane", mir, x, 1.6 * lim_x, "x (um)", P[1:])):
    n = data.shape[1]
    # (1) maps: profile (normalised per pass) vs pass
    fig, axs = plt.subplots(1, n, figsize=(3.4 * n, 3.6), sharey=True, constrained_layout=True)
    for i in range(n):
        d = norm_rows(data[:, i, :].astype(float))
        if i < n - 1:
            d[~ok[-len(passes):, i]] = np.nan
        axs[i].pcolormesh(coord, passes, d, shading="nearest", cmap="inferno", vmin=0, vmax=1)
        axs[i].set_xlim(-lim, lim)
        axs[i].set_title(labels[i])
        axs[i].set_xlabel(xl)
    axs[0].set_ylabel("pass")
    fig.suptitle(f"{tag}: 1D profile (cut), normalised per pass")
    fig.savefig(f"out_profiles_{INJECTION}/map_{tag}.png", dpi=130)

    # (2) line plots: selected passes, all colours + global on the same axes
    sel = [p for p in (0, 1, 5, 10, 15, 20) if p in passes]
    fig, axs = plt.subplots(1, len(sel), figsize=(3.2 * len(sel), 3.2), sharey=True,
                            constrained_layout=True)
    for a, p in zip(np.atleast_1d(axs), sel):
        k = list(passes).index(p)
        for i in range(n):
            if i < n - 1 and not ok[-len(passes):][k, i]:
                continue
            a.plot(coord, norm_rows(data[k, i].astype(float)), "k--" if i == n - 1 else "-",
                   lw=1.2, label=labels[i])
        a.set_xlim(-lim, lim)
        a.set_title(f"pass {p}")
        a.set_xlabel(xl)
    np.atleast_1d(axs)[0].legend(fontsize=7)
    fig.savefig(f"out_profiles_{INJECTION}/lines_{tag}.png", dpi=130)
    plt.close("all")

# rms widths vs pass for each colour (near field cut) + fraction of energy in each band
fig, axs = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
for i, lab in enumerate(labels):
    d = near[:, i, :].astype(float)
    w = 2 * np.sqrt((d * x ** 2).sum(1) / d.sum(1))
    if i < len(WAVELENGTHS):
        w[~ok[:, i]] = np.nan
    axs[0].plot(P, w, "o-" if i < len(WAVELENGTHS) else "k--", label=lab)
    if i < len(WAVELENGTHS):
        axs[1].semilogy(P, band[:, i] / band[:, -1], "o-", label=lab)
axs[0].set_xlabel("pass"); axs[0].set_ylabel("2 sigma width, cut (um)"); axs[0].legend()
axs[1].set_xlabel("pass"); axs[1].set_ylabel(f"energy fraction in {BANDWIDTH*1e9:.0f} nm band")
axs[1].legend()
fig.savefig(f"out_profiles_{INJECTION}/widths_and_band_energy.png", dpi=130)
print(f"figures written to out_profiles_{INJECTION}/")
