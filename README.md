# THUNDER4D

**T**ransverse-resolved **H**erriott-cell **U**PPE solver for **N**onlinear **D**ynamics close to **E**xperimental **R**eality, **4D**: a 4D (x, y, z, t) UPPE solver for multipass cells. Python package: `thunder4d`.

Full documentation: `https://victorkolt.github.io/thunder4D/`.

`thunder4d` simulates nonlinear pulse propagation in a gas-filled Herriott-type multipass cell (MPC) with full spatial and temporal resolution: the field is A(x, y, t) and it is propagated along z. It is built to study spatial and spatio-spectral effects that 1D or radially symmetric codes cannot capture:

- spatial effects in non-Gaussian inputs (clipped, aberrated, high-order beams),
- spatio-spectral homogeneity, radial chirp,
- time-dependent nonlinear mode mismatch,
- influence of space-time couplings (spatial chirp, pulse-front tilt, …) on the output.

The code is pure Python. NumPy (≥ 2.0) is required. SciPy is optional but strongly recommended, because `scipy.fft` is about 8× faster than `numpy.fft` for the 2D transforms. Matplotlib is only needed for the examples. No GPU is used.

## Quick start

```python
import thunder4d as m
from thunder4d import diagnostics as dg

# medium and cell
gas  = m.Material("Ar", pressure_bar=1.0)
cell = m.MPCCell.herriott(R=0.5, N=10, k=3, medium=gas)

# grid sized from the cell
grid = m.suggest_grid(cell, lambda0=1030e-9, lambda_min=900e-9, lambda_max=1200e-9, pulse_fwhm=200e-15, box_factor=12)
print(grid.describe())

# 1) spectrum (+ phase)
E = m.gaussian_spectrum(grid, fwhm_fs=200)
E = m.add_spectral_phase(grid, E, gdd_fs2=0)

# 2) nonlinear mode matching, computed from a probe pulse of the right energy
probe = m.Pulse.at_focus(grid, E, m.Beam(m.gaussian(1e-3)), 1.2e-3)
mm = cell.match(probe)

# 3) the actual input beam
pulse = m.Pulse.at_focus(grid, E, m.Beam(m.gaussian(mm["w0"])), 1.2e-3)

# 4) run
sim = m.MPCSimulation(pulse, cell, phi_max=0.02, mode_reference=mm["w0"], profile_wavelengths=[1010e-9, 1030e-9, 1050e-9], profile_bandwidth=2e-9)
sim.run()
sim.save("run.npz")

# 5) analyse the output
x, lam, S = dg.spectrogram_x_lambda(grid, sim.Aw)
c = dg.compress(grid, sim.Aw)
print(c["fwhm_fs"], c["gdd_fs2"])
```

---

## Physical model

The field is a complex envelope A(x, y, t) with carrier exp[i(k₀z − ω₀t)], normalised so that |A|² is the intensity in W/m². It is propagated with the unidirectional pulse propagation equation (UPPE), written in the frame moving at the group velocity at λ₀:

```
∂A(kx, ky, Ω)/∂z = i [ kz(kx, ky, ω) − k₀ − k₁Ω ] A  +  i γ(Ω) FT_t[ |A|² A ]

kz   = sqrt( k(ω)² − kx² − ky² )        (non-paraxial, full gas dispersion)
k(ω) = n(ω) ω / c,   ω = ω₀ + Ω,   k₁ = dk/dω at ω₀
γ(Ω) = n₂ ω/c · n(ω₀)/n(ω)               (self-steepening included)
```

- **Linear operator:** it is exact. It includes diffraction without the paraxial approximation, gas dispersion to all orders, and evanescent components, which decay.
- **Kerr term:** it is the instantaneous electronic response. Its frequency dependence gives self-steepening. With `self_steepening=False` it becomes γ = n₂ω₀/c.
- **Edge taper:** the nonlinear source is tapered to zero at the edges of the spectral window to avoid aliasing.

## Numerical scheme

- **Home domain:** the solver keeps the field in the mixed domain (x, y, Ω).
  - The linear step needs only 2D spatial FFTs, because Ω is already a coordinate.
  - The Kerr term does not depend on (kx, ky), so it needs only 1D temporal FFTs.
- **Splitting:** symmetric Strang splitting (linear half-step, nonlinear step, linear half-step). Consecutive linear half-steps are merged.
- **Nonlinear step:** fourth-order Runge–Kutta.
- **Adaptive step size:** h = φ_max / (k₀ n₂ I_peak), where φ_max is the peak nonlinear phase allowed per step. It is limited to `dz_max` (default d/8, a quarter of a half-pass) and quantised to `dz_max · 2^(−n/4)`. The quantisation lets the linear propagators exp(iDh) be reused from a small cache.

## Performance and memory

- **Cost per step:** 2 spatial 2D FFTs, 8 temporal 1D FFTs (RK4) and a few array operations. With SciPy on one core, a 150 × 150 × 135 grid takes about 0.1–0.4 s per step. A 20-pass run at B ≈ 16 rad needs about 900 steps.
- **Steps per pass:** they scale with the B-integral per pass divided by `phi_max`.
- **Memory:** about 8–10 field arrays: the field, RK4 buffers, the linear operator `D`, up to `cache_size` cached propagators and the cached mirror transfer. One complex64 array is Nx·Ny·Nt·8 bytes, so 256 × 256 × 256 is 134 MB.
- **Speed-ups:**
  - SciPy backend with all cores (`fft_backend="scipy"`, `workers=-1`, the default with `"auto"`).
  - `record_every > 1`, or `far_field=False` (far-field diagnostics use per-colour matrix transforms).
  - Fewer `profile_wavelengths`.
  - A smaller `cache_size` if memory is tight.
- **Long runs:** use `run(max_seconds=…)` with `checkpoint` / `resume`.

## Limitations

**Not included**
- **Ionisation and plasma:** fine for MPCs operated well below ionisation thresholds.
- **Raman response:** relevant for N₂, air and other molecular gases.
- **Off-axis geometry:** the real cell is folded, so the beam hits the mirrors at small angles. The resulting astigmatism and the actual spot pattern are not modelled; the unfolded on-axis picture is used.
- **Non-uniform media:** gas plates, windows or pressure gradients are not built in. The nonlinear mode matching formula also assumes a uniformly filled cell with constant power during a pass.
- **Vectorial effects:** the field is scalar, with a single polarisation.
