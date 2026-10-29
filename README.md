# thunder4D

**T**ransverse-resolved **H**erriott-cell **U**PPE solver for **N**onlinear **D**ynamics close to **E**xperimental **R**eality, **4D**: a 4D (x, y, z, t) UPPE solver for multipass cells. Python package: `thunder4d`.

Full documentation: `https://victorkolt.github.io/thunder4D/`.

`thunder4D` simulates nonlinear pulse propagation in a gas-filled Herriott-type multipass cell (MPC) with full spatial and temporal resolution: the field is A(x, y, t) and it is propagated along z. It is built to study spatial and spatio-spectral effects that 1D or radially symmetric codes cannot capture:

- spatial effects in non-Gaussian inputs (clipped, aberrated, high-order beams),
- spatio-spectral homogeneity, radial chirp,
- time-dependent nonlinear mode mismatch,
- influence of space-time couplings (spatial chirp, pulse-front tilt, …) on the output.

The code is pure Python. NumPy (≥ 2.0) and SciPy (≥ 1.10) are required; `scipy.fft` is about 8× faster than `numpy.fft` for the 2D transforms. Matplotlib is only needed for the examples. An NVIDIA GPU is optional: with CuPy installed (`uv pip install cupy-cuda12x`), `suggest_lens_grid(..., device="auto")` (the default) runs the propagation on the GPU, and `device="cpu"` / `"gpu"` force one or the other. `thunder4d.gpu_available()` tells you whether CuPy works; `thunder4d.to_host(array)` converts a device array to NumPy. Everything returned by `thunder4d.diagnostics` is a NumPy array on either device.

## Quick start

```python
import thunder4d as m
from thunder4d import diagnostics as dg

# 1) medium and cell
gas  = m.Material("Ar", pressure_bar=1.0)
cell = m.MPC.herriott(R=0.5, N=20, k=1, medium=gas)   # near-concentric, 20 passes, 162 deg Gouy/pass

# 2) lens-frame grid: the box only has to hold the beam at the waist
grid = m.suggest_lens_grid(cell, lambda0=1030e-9, lambda_min=900e-9, lambda_max=1200e-9,
                           pulse_fwhm=200e-15, box_factor=6)
print(grid.describe())

# 3) pulse with nonlinear mode matching
E = m.gaussian_spectrum(grid, fwhm_fs=200)
probe = m.Pulse.at_focus(grid, E, m.Beam(m.gaussian(cell.linear_mode(1030e-9)["w0"])), 0.6e-3)
mm = cell.match(probe)                                  # P/Pc, matched waist, B per pass
pulse = m.Pulse.at_focus(grid, E, m.Beam(m.gaussian(mm["w0"]), chromatic_size="eigenmode"), 0.6e-3)

# 4) run
sim = m.LensFrameSimulation(pulse, cell, phi_max=0.02, mode_reference=mm["w0"],
                            profile_wavelengths=[1010e-9, 1030e-9, 1050e-9], profile_bandwidth=2e-9)
sim.run()
sim.save("run.npz")

# 5) analyse the output
x, lam, S = dg.spectrogram_x_lambda(grid, sim.Aw)
c = dg.compress(grid, sim.Aw)
print(c["fwhm_fs"], c["gdd_fs2"])
```

A full walkthrough with figures is in [`examples/tutorial.ipynb`](examples/tutorial.ipynb); comparisons with published experiments are in [`benchmark/`](benchmark/README.md). To contribute, see [`CONTRIBUTING.md`](CONTRIBUTING.md).

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

## The lens frame

The recommended solver (`LensFrameSimulation`, grid from `suggest_lens_grid`) propagates the field in a frame that follows the cell eigenmode: A(x, y, z) = (1/s) B(x/s, y/s, ζ) exp(i k r² / 2R(z)), with s(z) = sqrt(1 + z²/z_R²), R(z) = z + z_R²/z and ζ = z_R arctan(z/z_R). In this frame:

- **the grid only holds the waist:** the eigenmode does not change size, so the box is a few waists wide whatever the mirror separation (75 points per axis instead of 480 for the tutorial cell, instead of 1125 for an 18 mJ cell at d = 0.995·2R);
- **the mirror is the identity:** its curvature cancels the frame wavefront exactly; only its spectral response and aperture remain;
- **the cell centre is physical** (s = 1): every diagnostic recorded there is unchanged. Mirror records are in frame coordinates; convert them with `sim.mirror_scale` (x_phys = s_m ξ, F_phys = F / s_m²).

The frame equation is ∂B/∂ζ = (i/2k)∇²B − i k ρ² B / (2 z_R²) + i β s² B + i γ FT_t[|B|² B]: paraxial diffraction plus a harmonic potential (whose eigenstates are the Hermite–Gauss modes of the cell, so the Gouy phase per pass is built in), the gas dispersion over the physical length, and the unchanged Kerr term. The step is limited by both `phi_max` and `gouy_step` (maximum Gouy phase per step, default 0.05 rad). The standard non-paraxial fixed-grid solver below (`suggest_grid` + `MPCSimulation`) has the same interface and remains available for cross-checks; `tests/test_lensframe.py` checks that both agree within 0.5 %. Details: documentation, page *The lens frame*.

## Numerical scheme (standard solver)

- **Home domain:** the solver keeps the field in the mixed domain (x, y, Ω).
  - The linear step needs only 2D spatial FFTs, because Ω is already a coordinate.
  - The Kerr term does not depend on (kx, ky), so it needs only 1D temporal FFTs.
- **Splitting:** symmetric Strang splitting (linear half-step, nonlinear step, linear half-step). Consecutive linear half-steps are merged.
- **Nonlinear step:** fourth-order Runge–Kutta.
- **Adaptive step size:** h = φ_max / (k₀ n₂ I_peak), where φ_max is the peak nonlinear phase allowed per step. It is limited to `dz_max` (default d/8, a quarter of a half-pass) and quantised to `dz_max · 2^(−n/4)`. The quantisation lets the linear propagators exp(iDh) be reused from a small cache.

## Performance and memory

- **Lens frame:** the gain is mostly in grid points (see above). A step costs one 2D FFT pair, 8 temporal 1D FFTs (RK4) and phase multiplications built from 1D arrays; no linear operator or propagator cache is stored. A pass takes at least θ/`gouy_step` steps.
- **Cost per step (standard solver):** 2 spatial 2D FFTs, 8 temporal 1D FFTs (RK4) and a few array operations. With SciPy on one core, a 150 × 150 × 135 grid takes about 0.1–0.4 s per step. A 20-pass run at B ≈ 16 rad needs about 900 steps.
- **Steps per pass:** they scale with the B-integral per pass divided by `phi_max`.
- **Memory (standard solver):** about 8–10 field arrays: the field, RK4 buffers, the linear operator `D`, up to `cache_size` cached propagators and the cached mirror transfer. One complex64 array is Nx·Ny·Nt·8 bytes, so 256 × 256 × 256 is 134 MB.
- **Speed-ups:**
  - SciPy backend with all cores (`fft_backend="scipy"`, `workers=-1`, the default with `"auto"`).
  - `record_every > 1`, or `near_field=False` (near-field diagnostics, i.e. of the collimated beam, use per-colour matrix transforms).
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
