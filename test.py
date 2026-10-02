#%%
import thunder4d as m
from thunder4d import diagnostics as dg
import matplotlib.pyplot as plt
import numpy as np

# medium and cell
gas  = m.Material("Ar", pressure_bar=1.0)
cell = m.MPC.herriott(R=1.0, N=17, k=1, medium=gas)

#%%

# grid sized from the cell
grid = m.suggest_grid(cell, lambda0=1030e-9, lambda_min=1000e-9, lambda_max=1060e-9, pulse_fwhm=700e-15, box_factor=12)
print(grid.describe())

#%%
# 1) spectrum (+ phase)
E = m.gaussian_spectrum(grid, fwhm_fs=700)
E = m.add_spectral_phase(grid, E, gdd_fs2=0)

#%%
# 2) nonlinear mode matching, computed from a probe pulse of the right energy
probe = m.Pulse.at_focus(grid, E, m.Beam(m.gaussian(1e-3)), 7e-3)
mm = cell.match(probe)

#%%
# 3) the actual input beam
pulse = m.Pulse.at_focus(grid, E, m.Beam(m.gaussian(mm["w0"])), 7e-3)

#%%
# 4) run
sim = m.MPCSimulation(pulse, cell, phi_max=0.02, mode_reference=mm["w0"], profile_wavelengths=[1010e-9, 1030e-9, 1050e-9], profile_bandwidth=2e-9)
sim.run()
sim.save("run.npz")

#%%
# 5) analyse the output
x, lam, S = dg.spectrogram_x_lambda(grid, sim.Aw)
c = dg.compress(grid, sim.Aw)
print(c["fwhm_fs"], c["gdd_fs2"])

lam, spec = dg.total_spectrum(grid, sim.Aw)
plt.plot(lam*1e9, spec/np.max(spec))
plt.show()