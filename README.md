# THUNDER4D

**T**ransverse-resolved **H**erriott-cell **U**PPE solver for **N**onlinear **D**ynamics close to **E**xperimental **R**eality, **4D**: a 4D (x, y, z, t) UPPE solver for multipass cells. Python package: `thunder4d`.

Full documentation: `https://victorkolt.github.io/thunder4D/`.

`thunder4d` simulates nonlinear pulse propagation in a gas-filled Herriott-type multipass cell (MPC) with full spatial and temporal resolution: the field is A(x, y, t) and it is propagated along z. It is built to study spatial and spatio-spectral effects that 1D or radially symmetric codes cannot capture:

- spatial effects in non-Gaussian inputs (clipped, aberrated, high-order beams),
- spatio-spectral homogeneity, radial chirp,
- time-dependent nonlinear mode mismatch,
- influence of space-time couplings (spatial chirp, pulse-front tilt, …) on the output.

The code is pure Python. NumPy (≥ 2.0) is required. SciPy is optional but strongly recommended, because `scipy.fft` is about 8× faster than `numpy.fft` for the 2D transforms. Matplotlib is only needed for the examples. No GPU is used.

---

## Contents

1. [Installation and file layout](#1-installation-and-file-layout)
2. [Quick start](#2-quick-start)
3. [Physical model](#3-physical-model)
4. [Conventions and units](#4-conventions-and-units)
5. [Module reference](#5-module-reference)
   - [5.1 `grid` — grid and FFTs](#51-grid--grid-and-ffts)
   - [5.2 `materials` — the nonlinear medium](#52-materials--the-nonlinear-medium)
   - [5.3 `pulse` — pulse initialisation](#53-pulse--pulse-initialisation)
   - [5.4 `cell` — the multipass cell](#54-cell--the-multipass-cell)
   - [5.5 `solver` — the UPPE propagator](#55-solver--the-uppe-propagator)
   - [5.6 `simulation` — the driver and recorder](#56-simulation--the-driver-and-recorder)
   - [5.7 `diagnostics` — analysis functions](#57-diagnostics--analysis-functions)
6. [Recorded history and saved files](#6-recorded-history-and-saved-files)
7. [Example scripts](#7-example-scripts)
8. [Choosing the grid, accuracy and validation](#8-choosing-the-grid-accuracy-and-validation)
9. [Performance and memory](#9-performance-and-memory)
10. [Limitations and extension points](#10-limitations-and-extension-points)

---

## 1. Installation and file layout

No installation step is needed. Put the `thunder4d` folder next to your scripts, or add its parent folder to `sys.path`.

```
thunder4d/
├── README.md
├── thunder4d/
│   ├── __init__.py        public API (see below)
│   ├── grid.py            Grid, FFT wrapper, suggest_grid
│   ├── materials.py       Material (Sellmeier + n2)
│   ├── pulse.py           spectra, spatial profiles, aberrations, STCs, Beam, Pulse
│   ├── cell.py            MPCCell (geometry, eigenmode, nonlinear matching, mirrors)
│   ├── solver.py          UPPESolver (propagation)
│   ├── simulation.py      MPCSimulation (pass loop, recording, checkpoints, saving)
│   └── diagnostics.py     spectrograms, mode content, homogeneity, M², compression, …
└── examples/
    ├── compare_gauss_tophat.py
    ├── tophat_profiles_vs_pass.py
    └── clipped_gauss_profiles.py
```

Everything useful is exposed at package level:

```python
import thunder4d as m
from thunder4d import diagnostics as dg

m.Grid, m.suggest_grid, m.C0
m.Material
m.Pulse, m.Beam, m.gaussian, m.super_gaussian, m.top_hat, m.knife_edge, m.measured_profile
m.gaussian_spectrum, m.measured_spectrum, m.spectrum_from_temporal_field, m.add_spectral_phase
m.zernike_wavefront
m.MPCCell, m.UPPESolver, m.MPCSimulation
```

---

## 2. Quick start

```python
import thunder4d as m
from thunder4d import diagnostics as dg

# medium and cell
gas  = m.Material("Ar", pressure_bar=1.0)
cell = m.MPCCell.herriott(R=0.5, N=10, k=3, medium=gas)      # 20 passes, re-entrant

# grid sized from the cell
grid = m.suggest_grid(cell, lambda0=1030e-9, lambda_min=900e-9, lambda_max=1200e-9,
                      pulse_fwhm=200e-15, box_factor=12)
print(grid.describe())

# 1) spectrum (+ phase)
E = m.gaussian_spectrum(grid, fwhm_fs=200)
E = m.add_spectral_phase(grid, E, gdd_fs2=0)

# 2) nonlinear mode matching, computed from a probe pulse of the right energy
probe = m.Pulse.at_focus(grid, E, m.Beam(m.gaussian(1e-3)), 1.2e-3)
mm = cell.match(probe)                    # prints P/Pc, matched waist, B per pass

# 3) the actual input beam
pulse = m.Pulse.at_focus(grid, E, m.Beam(m.gaussian(mm["w0"])), 1.2e-3)

# 4) run
sim = m.MPCSimulation(pulse, cell, phi_max=0.02, mode_reference=mm["w0"],
                      profile_wavelengths=[1010e-9, 1030e-9, 1050e-9], profile_bandwidth=2e-9)
sim.run()
sim.save("run.npz")

# 5) analyse the output
x, lam, S = dg.spectrogram_x_lambda(grid, sim.Aw)
c = dg.compress(grid, sim.Aw)
print(c["fwhm_fs"], c["gdd_fs2"])
```

---

## 3. Physical model

### 3.1 Propagation equation

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

### 3.2 Numerical scheme

- **Home domain:** the solver keeps the field in the mixed domain (x, y, Ω).
  - The linear step needs only 2D spatial FFTs, because Ω is already a coordinate.
  - The Kerr term does not depend on (kx, ky), so it needs only 1D temporal FFTs.
- **Splitting:** symmetric Strang splitting (linear half-step, nonlinear step, linear half-step). Consecutive linear half-steps are merged.
- **Nonlinear step:** fourth-order Runge–Kutta.
- **Adaptive step size:** h = φ_max / (k₀ n₂ I_peak), where φ_max is the peak nonlinear phase allowed per step. It is limited to `dz_max` (default d/8, a quarter of a half-pass) and quantised to `dz_max · 2^(−n/4)`. The quantisation lets the linear propagators exp(iDh) be reused from a small cache.

### 3.3 Cell model

The cell is treated in the **unfolded** picture. One pass is:

```
cell centre ──(d/2)──▶ curved mirror ──(d/2)──▶ cell centre
```

So N passes means N mirror bounces. Diagnostics are recorded at the cell centre after every pass. That plane is the waist of the cell eigenmode and has a flat wavefront, so a Hermite–Gauss decomposition there is exact. Optionally, data are also recorded at the mirror plane before each bounce.

Each bounce multiplies the field in (x, y, Ω) by:

```
exp( −i ω r² / (c R) )                     curved mirror (focal length R/2), achromatic
× sqrt(Refl(λ)) · exp( i [GDD/2 Ω² + TOD/6 Ω³ + φ_mirror(λ)] )
× aperture(r)                               optional soft circular clear aperture
```

### 3.4 Linear eigenmode

For a symmetric cell with g = 1 − d/R, the Gouy phase per pass is θ = arccos(g). In the medium (λ → λ/n₀):

```
w₀²       = (λ / 2π) · sqrt( d (2R − d) )        waist at the centre
w_mirror² = (λ R / π) · sqrt( d / (2R − d) )     spot on the mirrors
z_R       = sqrt( d (2R − d) ) / 2
```

A Herriott cell with N round trips and integer k is re-entrant when d = R(1 − cos(πk/N)). Each Hermite–Gauss mode HG_mn then picks up (m+n)θ per pass, so after 2N passes every mode is back in phase. **A linear re-entrant cell images any input onto the output**, top-hats included. Any cleaning must therefore come from the nonlinearity, from losses (apertures), or from imperfections.

### 3.5 Nonlinear mode matching

Use the aberration-free Kerr lens approximation for a uniformly filled cell, with the power P constant during a pass. The envelope equation of a Gaussian beam then becomes the linear one with

```
λ → λ_eff = λ · sqrt(1 − P/Pc),      Pc = λ² / (2π n₀ n₂)
```

The matched waist and mirror spot are therefore the linear values multiplied by (1 − P/Pc)^¼. The on-axis peak B-integral per pass is

```
B_pass = 8π n₂ P / (λ λ_eff) · arctan( d / (2 z_R') ),   z_R' = π w₀² / λ_eff
       ≈ 2 θ P/Pc   for P ≪ Pc
```

A pulse can only be matched at one power. The wings of the pulse in time see a mode closer to the linear one. This time-dependent mismatch is one of the space-time couplings the 4D code resolves.

---

## 4. Conventions and units

| Quantity | Convention |
|---|---|
| Field | complex envelope, |A|² = intensity [W/m²] |
| Temporal transform | Ã(Ω) = Σₜ A(t) e^{+iΩt} (`grid.fft.ft_t`); A(t) = (1/Nt) Σ_Ω Ã e^{−iΩt} (`grid.fft.ift_t`) |
| Spatial transform | standard `fft2`, plane waves e^{+ikx·x}, propagation e^{+ikz·z} |
| Spectral phase | φ = +GDD/2 Ω²: **positive GDD = up-chirp**; e^{+iΩt₀} delays by t₀ |
| Field array shape | `(Nx, Ny, Nt)`, home domain (x, y, Ω) |
| Array ordering | **FFT order** everywhere: index 0 is x = 0, t = 0, Ω = 0 |
| SI units | m, s, J, W, unless the argument name says otherwise (`_fs`, `_nm`, `_mm`, `_urad`, `_bar`) |
| Energy | `grid.energy(Aw)` in J (Parseval with dx·dy·dt/Nt) |

FFT ordering is used internally because all solver operations are local, so ordering doesn't matter there and no shifts are needed. For plotting, use the sorted axes `grid.xs`, `grid.ys`, `grid.ts`, `grid.lams`, `grid.Omegas`, `grid.kxs` and the helpers `grid.shift_xy(a)` and `grid.shift_w(a)`. **All functions in `diagnostics` return sorted axes.**

---

## 5. Module reference

### 5.1 `grid` — grid and FFTs

#### `C0`
Speed of light in vacuum [m/s].

#### `class Grid`

```python
Grid(Nx, dx, Nt, lambda0, *, dt=None, lambda_min=None, lambda_max=None,
     Ny=None, dy=None, dtype=np.complex64, fft_backend="auto", workers=None)
```

| Argument | Meaning |
|---|---|
| `Nx, dx` | number of transverse points and step [m]. `Ny, dy` default to the same values. |
| `Nt` | number of temporal points |
| `lambda0` | carrier wavelength [m]. Defines Ω = 0 and the moving frame. |
| `dt` | time step [s]. Alternatively give `lambda_min` (and optionally `lambda_max`) and `dt` is chosen so the spectral window covers them. |
| `dtype` | `np.complex64` (default, half the memory) or `np.complex128` |
| `fft_backend` | `"auto"` (SciPy if installed, else NumPy), `"scipy"` or `"numpy"` |
| `workers` | number of threads for the SciPy backend (default: all cores) |

**Attributes**

| Attribute | Content |
|---|---|
| `x, y` | coordinates [m], FFT order |
| `kx, ky` | transverse wavenumbers [rad/m], FFT order |
| `t` | time [s], FFT order |
| `Omega` | angular frequency offset Ω [rad/s], FFT order |
| `omega` | absolute angular frequency ω₀ + Ω |
| `lam` | wavelength [m] for each Ω (`inf` where ω ≤ 0) |
| `valid` | boolean mask, ω > 0 |
| `omega0, lambda0` | carrier |
| `dtype, rdtype` | complex and matching real dtype |
| `fft` | `FFT` object (see below) |
| `shape` | `(Nx, Ny, Nt)` |
| `X, Y` | broadcastable coordinate arrays `(Nx,1)` and `(1,Ny)` |
| `r2` | x² + y², shape `(Nx, Ny)` |
| `dA` | dx·dy |
| `xs, ys, ts, kxs, lams, Omegas` | sorted versions, for plotting |

**Methods**

| Method | Description |
|---|---|
| `energy(Aw)` | energy [J] of a field in (x, y, Ω) |
| `energy_t(At)` | energy [J] of a field in (x, y, t) |
| `shift_xy(a)` | `fftshift` over the two spatial axes (static) |
| `shift_w(a, axis=-1)` | `fftshift` over the spectral/temporal axis (static) |
| `describe()` | one-line summary: sizes, memory per field, box, time window, λ window |

#### `class FFT`

```python
FFT(backend="auto", workers=None)
```
A thin wrapper that applies the conventions of §4.

| Method | Transform |
|---|---|
| `fft2(a)` | x, y → kx, ky (axes 0, 1) |
| `ifft2(a)` | kx, ky → x, y |
| `ft_t(a)` | t → Ω (last axis) |
| `ift_t(a)` | Ω → t |

#### `suggest_grid`

```python
suggest_grid(cell, lambda0, lambda_min, lambda_max, pulse_fwhm, *, box_factor=8.0,
             points_per_waist=8.0, time_window_factor=8.0, **grid_kw) -> Grid
```

Chooses a grid adapted to a cell:

- **Box width:** `box_factor` × the largest mirror spot radius, taken at `lambda_max`.
- **Step dx:** the smaller of two limits.
  - The smallest waist (at `lambda_min`) divided by `points_per_waist`.
  - The Nyquist limit of the mirror curvature phase at the corner of the box: 0.8 · λ_min R / (√2 L).
- **Time step:** dt covers [`lambda_min`, `lambda_max`] around `lambda0`.
- **Point counts:** Nt covers `time_window_factor` × the input FWHM. Nx and Nt are rounded up to fast FFT sizes (2ᵃ3ᵇ5ᶜ).
- **Extra arguments:** `grid_kw` (e.g. `dtype`, `fft_backend`) are passed to `Grid`.

Use `box_factor` ≥ 12–14 for top-hat, clipped or aberrated inputs: their higher-order content is much wider than the fundamental mode.

---

### 5.2 `materials` — the nonlinear medium

#### `class Material`

```python
Material(name, pressure_bar=1.0, temperature_K=293.15, *, n2_1bar=None,
         sellmeier=None, n_func=None, n2=None)
```

| Argument | Meaning |
|---|---|
| `name` | `"He"`, `"Ne"`, `"Ar"`, `"Kr"`, `"Xe"`, `"N2"`, `"air"`, or any label if `n_func` is given |
| `pressure_bar, temperature_K` | gas state. n² − 1 scales as p/T; n₂ scales as p · 293.15/T. |
| `n2_1bar` | override the tabulated n₂ at 1 bar [m²/W] |
| `sellmeier` | override the coefficients `(B1, C1, B2, C2)` (λ in µm) |
| `n_func` | callable `n(omega)` replacing the Sellmeier formula entirely |
| `n2` | absolute n₂ [m²/W] at the given conditions (no pressure scaling) |

- **Dispersion:** Börzsönyi et al., Appl. Opt. 47, 4856 (2008): n² − 1 = (p/p₀)(T₀/T)[B₁λ²/(λ² − C₁) + B₂λ²/(λ² − C₂)], with p₀ = 1 bar and T₀ = 273.15 K.
- **Tabulated n₂ (`N2_1BAR`):** approximate literature values (electronic Kerr only, no Raman). Replace them with your own values.

| Method / attribute | Description |
|---|---|
| `n(omega)` | refractive index (returns 1 for ω ≤ 0) |
| `k(omega)` | wavenumber n ω / c [rad/m] |
| `gvd(lam)` | GVD [fs²/m] at wavelength `lam` |
| `critical_power(lam)` | Pc = λ² / (2π n₀ n₂) [W] (Gaussian beam, aberration-free) |
| `n2` | n₂ at the current pressure and temperature [m²/W] |
| `p, T, name` | state and label |

Module-level dictionaries: `SELLMEIER`, `N2_1BAR`.

---

### 5.3 `pulse` — pulse initialisation

A pulse is built as

```
A(x, y, Ω) = E(Ω) · U(x, y; Ω)
```

- **E(Ω):** a 1D complex spectral amplitude.
- **U:** a spatial field, evaluated separately at every frequency (so it can be chromatic) and normalised to unit power at each frequency. The spatially integrated spectrum is therefore exactly |E(Ω)|².

#### 5.3.1 Spectral amplitude

All functions return a complex 1D array on `grid.Omega` (FFT order).

```python
gaussian_spectrum(grid, fwhm_fs=None, fwhm_nm=None, center_nm=None)
```
Transform-limited Gaussian. Give either the intensity FWHM duration [fs] or the spectral FWHM [nm]. `center_nm` defaults to λ₀.

```python
measured_spectrum(grid, wavelength_nm, intensity, phase_rad=None, background=0.0)
```
A spectrum measured against wavelength. It applies the λ → ω Jacobian (S_ω = S_λ λ² / 2πc), subtracts `background`, clips negative values and interpolates onto the grid. An optional spectral phase (e.g. from a d-scan or FROG retrieval) can be given on the same wavelength axis; it is unwrapped before interpolation.

```python
spectrum_from_temporal_field(grid, t_fs, field)
```
A complex temporal envelope (e.g. a retrieved pulse, |E|·e^{iφ}), relative to `grid.lambda0`, interpolated onto the time grid and transformed.

```python
add_spectral_phase(grid, E, gdd_fs2=0.0, tod_fs3=0.0, fod_fs4=0.0, delay_fs=0.0, phase=None)
```
Returns E · e^{iφ} with φ = GDD/2 Ω² + TOD/6 Ω³ + FOD/24 Ω⁴ + delay · Ω, where Ω is in rad/fs. `phase` adds an arbitrary term: either a callable of Ω [rad/s] or an array on `grid.Omega`.

#### 5.3.2 Spatial profiles

A `Profile` is a callable `f(X, Y)` returning a complex amplitude. It also has a `radius` (used as the default Zernike pupil) and a `name`.

| Function | Profile |
|---|---|
| `gaussian(w)` | exp(−r²/w²) (1/e² intensity radius w) |
| `super_gaussian(w, order=10)` | exp(−(r/w)^order); order 2 is a Gaussian |
| `top_hat(radius, order=24)` | super-Gaussian with soft edges. Higher orders give harder edges and need a finer near-field sampling (e.g. order 100 with 384 points). |
| `knife_edge(profile, edge, axis="x", side="+", width=None)` | clips any profile with a hard edge at `edge` [m] along `axis`. `side="+"` blocks x > edge, `"-"` blocks x < edge, `"both"` keeps \|x\| < \|edge\| (slit). `width` gives a tanh edge (10–90 % width); `None` gives a step. |
| `measured_profile(image, pixel_size, center="centroid", background=0.0)` | camera image (rows = y, columns = x) → amplitude √I, bilinear interpolation, zero outside the image. `center` is `"centroid"` or `(cx, cy)` [m]. |
| `Profile(func, radius, name)` | wrap your own function |

#### 5.3.3 Aberrations

```python
zernike_wavefront(X, Y, coeffs_waves, pupil_radius, lambda0) -> W [m]
```
Wavefront error built from RMS-normalised Zernike terms, with coefficients in waves at λ₀. It is applied in `Beam` as the phase (ω/c)·W, i.e. an achromatic path error that is correct for every colour.

Available terms: `piston`, `tilt_x`, `tilt_y`, `defocus`, `astig_0`, `astig_45`, `coma_x`, `coma_y`, `trefoil_0`, `trefoil_30`, `spherical`.

#### 5.3.4 `class Beam` — profile + aberrations + space-time couplings

```python
Beam(profile, zernike=None, pupil_radius=None,
     spatial_chirp_mm_per_nm=(0, 0), angular_dispersion_urad_per_nm=(0, 0),
     pft_fs_per_mm=(0, 0), pfc_fs_per_mm2=0.0, radial_gdd_fs2_per_mm2=0.0,
     tilt_urad=(0, 0), offset_m=(0, 0), custom_phase=None)
```

| Argument | Effect at a given wavelength λ (Δλ = λ − λ₀, Ω in rad/fs) |
|---|---|
| `profile` | spatial amplitude |
| `zernike` | dict, e.g. `{"astig_0": 0.1, "coma_x": 0.05}` (RMS waves at λ₀) |
| `pupil_radius` | Zernike normalisation radius [m] (default `profile.radius`) |
| `spatial_chirp_mm_per_nm` | (sx, sy): each colour is shifted by s·Δλ |
| `angular_dispersion_urad_per_nm` | (ax, ay): propagation angle a·Δλ |
| `pft_fs_per_mm` | (px, py): pulse-front tilt, delay = p·x |
| `pfc_fs_per_mm2` | pulse-front curvature, delay = a·r² (lens chromatism) |
| `radial_gdd_fs2_per_mm2` | GDD that varies as r² (radial chirp) |
| `tilt_urad` | pointing (same for all colours) |
| `offset_m` | transverse position |
| `custom_phase` | callable `(X, Y, Omega) -> phase [rad]`, anything else |

Method: `field(X, Y, omega, omega0)` returns the (unnormalised) complex field at one frequency.

Pulse-front tilt and angular dispersion are related but kept separate: `pft_fs_per_mm` is a pure x-dependent delay, while `angular_dispersion_urad_per_nm` is a wavelength-dependent propagation direction.

#### 5.3.5 `class Pulse`

```python
Pulse(grid, Aw)        # wraps a field A(x, y, Ω) of shape grid.shape
```

**Builders.** The three builders differ in **where the beam is defined** relative to the cell. This matters: a top-hat can be imaged on the mirror or Fourier transformed at the centre, and the two inputs evolve differently.

| Builder | Where the beam is defined | Typical use |
|---|---|---|
| `Pulse.at_focus(grid, E, beam, energy)` | directly at the cell centre | ideal matched Gaussian, test inputs, STCs defined at the focus |
| `Pulse.from_near_field(grid, E, beam, energy, nf_size, nf_points=256, focal_length=None, target_waist=None, verbose=True)` | collimated near field in the front focal plane of an achromatic lens; the cell centre is the focal plane (exact per-colour Fraunhofer transform) | top-hat near field → Airy pattern at the cell centre; chromatic focusing (spot ∝ λ) |
| `Pulse.imaged_near_field(grid, E, beam, energy, cell, image_z=None, curvature_radius=None, verbose=True)` | near field imaged (same size for every colour) onto the plane at `image_z` from the centre (default: the mirror, d/2), with the mode's wavefront curvature; then propagated back to the centre | laser near field relay-imaged onto the input mirror: flat-top on the mirror at pass 1 |

Details:

- **`from_near_field`**
  - The near field is sampled on `nf_points` × `nf_points` over a full width `nf_size` [m].
  - If `focal_length` is `None`, the lens is chosen by golden-section search to maximise the overlap at λ₀ with a Gaussian of waist `target_waist`, typically the matched `w0`. The chosen value is stored in `pulse.focal_length`.
  - A warning is printed if less than 99 % of the focused power fits in the box.
- **`imaged_near_field`**
  - `curvature_radius` defaults to R(z) = z + z_R²/z of the linear mode, which equals the mirror ROC at the mirror plane.
  - The profile should be sized to the mode at that plane. For a top-hat, a radius of 1.12 × `w_mirror` maximises the overlap with the Gaussian (η = 0.815).
- **All builders** normalise each colour to unit power before multiplying by E(Ω), then scale the total to `energy` [J].

**Methods and properties**

| Member | Description |
|---|---|
| `set_energy(energy)` | rescale to `energy` [J] (returns self) |
| `energy` | energy [J] (property) |
| `At()` | field in (x, y, t) |
| `power()` | spatially integrated power P(t) [W] (FFT order) |
| `peak_power()` | max P(t) [W] |
| `weighted_power()` | ∫P²dt / ∫P dt (= P_peak/√2 for a Gaussian) |
| `add_spectral_phase(**kw)` | same arguments as the function; applied to every pixel |
| `copy()` | independent copy |
| `grid, Aw` | the grid and the field array |

---

### 5.4 `cell` — the multipass cell

#### `class MPCCell`

```python
MPCCell(R, d, medium, n_passes, *, mirror_gdd_fs2=0.0, mirror_tod_fs3=0.0,
        mirror_reflectivity=1.0, mirror_phase=None, mirror_diameter=None)
```

| Argument | Meaning |
|---|---|
| `R` | mirror radius of curvature [m] |
| `d` | mirror separation [m]; must satisfy 0 < d < 2R |
| `medium` | `Material` filling the cell |
| `n_passes` | number of passes = number of bounces (see §3.3) |
| `mirror_gdd_fs2, mirror_tod_fs3` | dispersion per bounce |
| `mirror_reflectivity` | intensity reflectivity: a scalar, or a callable `R(lambda_m)` |
| `mirror_phase` | callable `phi(lambda_m)` [rad] per bounce (e.g. a measured GDD curve) |
| `mirror_diameter` | clear aperture [m], soft super-Gaussian edge (order 30); `None` means infinite |

```python
MPCCell.herriott(R, N, k, medium, n_passes=None, **kw)
```
Re-entrant Herriott cell: N round trips (= number of spots per mirror), d = R(1 − cos(πk/N)). `n_passes` defaults to 2N, one full re-entrant cycle.

**Attributes:** `R, d, medium, n_passes, g` (= 1 − d/R), `theta` (Gouy phase per pass [rad]).

**Methods**

| Method | Returns |
|---|---|
| `linear_mode(lam)` | dict `w0`, `w_mirror`, `zR`, `gouy_per_pass` of the linear eigenmode at wavelength `lam` |
| `nonlinear_mode(P, lam=None, verbose=False)` | Kerr-matched mode for power P [W] (see below) |
| `match(pulse, power="peak", verbose=True)` | same, with P taken from a `Pulse`: `"peak"` uses `peak_power()`, `"weighted"` uses `weighted_power()`. A number in W is also accepted. |
| `report(m=None)` | text summary of the cell and, optionally, of a matching dict |
| `bounce_factor(grid)` | complex transfer of one mirror, shape `grid.shape` (cached) |
| `bounce(Aw, grid)` | applies one mirror to `Aw` in place and returns it |

The dict returned by `nonlinear_mode` / `match` contains:

| Key | Content |
|---|---|
| `P, Pc, P_over_Pc` | power, critical power, ratio |
| `w0, w_mirror` | matched waist and mirror spot [m] |
| `w0_linear, w_mirror_linear` | linear values [m] |
| `B_per_pass, B_total` | on-axis peak B-integral estimates [rad] |
| `I_peak_center_W_cm2` | peak intensity at the waist [W/cm²] |

A `ValueError` is raised if P ≥ Pc (no matched mode: self-focusing).

---

### 5.5 `solver` — the UPPE propagator

#### `class UPPESolver`

```python
UPPESolver(grid, medium, *, phi_max=0.02, dz_max=None, paraxial=False,
           self_steepening=True, absorber_frac=0.9, absorber_order=16, cache_size=4)
```

| Argument | Meaning |
|---|---|
| `phi_max` | maximum peak nonlinear phase per step [rad]. Controls accuracy; halve it to check convergence. |
| `dz_max` | maximum step [m] (default: a quarter of each propagated segment) |
| `paraxial` | use kz ≈ k − (kx² + ky²)/2k instead of the exact square root |
| `self_steepening` | frequency-dependent γ(Ω) (True) or constant γ = n₂ω₀/c |
| `absorber_frac, absorber_order` | soft boundary absorbers: super-Gaussian of the given order, reaching 1/e at `absorber_frac` × half-window. Spatial (x, y), spectral (Ω) and temporal (t) versions exist. |
| `cache_size` | number of linear propagators exp(iDh) kept in memory (each is one full field array) |

**State:** `B` (accumulated on-axis peak B-integral [rad]), `nsteps`, `z` (total propagated distance [m]), `k0`, `k1`, `n2`, `D` (linear operator array), `igamma`.

**Methods**

| Method | Description |
|---|---|
| `propagate(Aw, L)` | propagate over L [m] with the adaptive split-step scheme; returns the new field. If n₂ = 0, a single exact linear step is used. |
| `linear(Aw, h)` | exact linear propagation over h [m] (h < 0 propagates backwards) |
| `nonlinear(Aw, track=False)` | the Kerr right-hand side i γ FT_t[\|A\|²A]; with `track=True` it updates the peak intensity used for step control |
| `peak_intensity(Aw)` | max \|A(x, y, t)\|² [W/m²] |
| `absorb(Aw, temporal=False)` | apply the spatial and spectral absorbers (and the temporal one if requested) |

The absorbers are applied by `MPCSimulation` twice per pass (at the mirror and at the centre), not at every step. That keeps their filtering effect small and independent of the step count.

---

### 5.6 `simulation` — the driver and recorder

#### `class MPCSimulation`

```python
MPCSimulation(pulse, cell, *, solver=None, record_every=1, far_field=True, max_order=6,
              mode_reference="linear", save_fields=(), save_dir=".", record_mirror=True,
              slit=None, profile_wavelengths=(), profile_bandwidth=0.0, profile_axis="x",
              **solver_kw)
```

| Argument | Meaning |
|---|---|
| `pulse` | input `Pulse` (copied), defined at the cell centre |
| `cell` | `MPCCell` |
| `solver` | an existing `UPPESolver`; otherwise one is created with `**solver_kw` (e.g. `phi_max=0.02`) |
| `record_every` | record diagnostics every n passes (pass 0 and the last pass are always recorded) |
| `far_field` | also record far-field quantities (angle-resolved spectrogram, far-field homogeneity, far-field profiles). Costs a few seconds per record. |
| `max_order` | highest Hermite–Gauss order in the modal decomposition |
| `mode_reference` | `"linear"`: HG basis of the linear eigenmode w₀(λ); or a float: a reference waist at λ₀ (e.g. the matched `w0`), scaled as √(λ/λ₀) |
| `save_fields` | iterable of pass numbers whose full 3D field is written to `save_dir/field_passNNN.npy` |
| `record_mirror` | record the fluence (and 1D profiles) at the mirror plane before each bounce |
| `slit` | slit width [m] integrated around y = 0 for the recorded S(x, λ) (`None` = one row) |
| `profile_wavelengths` | wavelengths [m] whose 1D profiles are recorded every pass, together with the global (spectrally integrated) profile |
| `profile_bandwidth` | bandpass width [m] around each of those wavelengths (0 = the nearest spectral bin, whose width is λ²/(cT)) |
| `profile_axis` | `"x"` or `"y"` |

**Methods and properties**

| Member | Description |
|---|---|
| `run(n_passes=None, verbose=True, max_seconds=None)` | propagate `n_passes` passes (default: up to `cell.n_passes`). `max_seconds` stops cleanly after the pass that exceeds the wall-clock budget. Prints one line per recorded pass. |
| `record(p, ...)` | record diagnostics for pass `p` (called automatically) |
| `checkpoint(path)` | pickle the full state (field, history, solver counters) |
| `resume(path)` | restore a checkpoint into a simulation built with the same grid and cell |
| `save(path, include_field=True)` | write the history and metadata (and the final field) to a compressed `.npz` (§6) |
| `done` | `True` when `cell.n_passes` passes have been made |
| `pulse` | the current field as a `Pulse` |
| `Aw` | the current field array |
| `history` | dict of per-pass records (§6) |
| `passes_done` | passes made so far |
| `lam, x, theta, prof_coord` | axes of the recorded quantities (sorted) |
| `solver` | the `UPPESolver` (gives access to `B`, `nsteps`, …) |

Typical long-run pattern (resumable):

```python
sim = m.MPCSimulation(pulse, cell, ...)
if os.path.exists("ckpt.pkl"):
    sim.resume("ckpt.pkl")
sim.run(max_seconds=3600)
sim.checkpoint("ckpt.pkl")
if sim.done:
    sim.save("run.npz")
```

---

### 5.7 `diagnostics` — analysis functions

All functions take `grid` and a field `Aw` in (x, y, Ω) (FFT order) and return **sorted** axes. Wavelength-resolved densities are per unit wavelength unless `per_lambda=False`.

#### Basic quantities

| Function | Returns |
|---|---|
| `spectral_support(grid, Aw, rel=1e-4)` | indices of frequencies carrying more than `rel` × the peak spectral energy |
| `total_spectrum(grid, Aw, per_lambda=True)` | `(lam, S)` spatially integrated spectrum, ascending wavelength |
| `fluence(grid, Aw)` | F(x, y) [J/m²] |
| `temporal_power(grid, Aw)` | `(t, P)` spatially integrated power [W] |

#### Far field

```python
far_field(grid, Aw, theta_max=None, n_theta=None, idx=None, theta_y=None) -> (theta_x, Ef)
```
Angular spectrum E(θx, θy, Ω) on an angle grid that is **common to all colours**. It is computed exactly with a per-colour matrix Fourier transform. This is what you would see after collimating the output with a lens of focal length f, at position x = f·θ.

- `theta_max` defaults to half the largest kx at λ₀.
- `n_theta` defaults to an odd number, so θ = 0 is sampled.
- `theta_y` can restrict the calculation to given θy values, e.g. `[0.]`.
- `Ef` has shape `(n_theta, len(theta_y), Nt)`.

#### Spatio-spectral traces

| Function | Returns |
|---|---|
| `spectrogram_x_lambda(grid, Aw, y0=0.0, slit=None, axis="x", per_lambda=True)` | `(x, lam, S[x, lam])`: imaging-spectrometer trace along a line, optionally integrated over a slit of width `slit` around `y0` |
| `spectrogram_theta_lambda(grid, Aw, theta_max=None, n_theta=None, per_lambda=True)` | `(theta, lam, S[theta, lam])`: angle-resolved spectrum at θy = 0 |

#### Wavelength-resolved 1D profiles

| Function | Returns |
|---|---|
| `band_indices(grid, lam_c, bandwidth=0.0)` | frequency indices inside [λc − bw/2, λc + bw/2] (nearest bin if bw = 0) |
| `profiles_1d(grid, Aw, wavelengths, bandwidth=0.0, axis="x", mode="cut")` | `(coord, profiles[n_lambda+1, N], band_energy[n_lambda+1])`. One row per wavelength band plus a last row for the global (spectrally integrated) profile. `mode="cut"`: line through the axis; `"integrated"`: projection, like a 1D camera lineout. `band_energy` [J] is the energy in each band; the last entry is the total. |
| `far_profiles_1d(grid, Aw, wavelengths, bandwidth=0.0, theta_max=None, n_theta=None)` | `(theta, profiles[n_lambda+1, n_theta])`: same in the far field (cut at θy = 0) |

#### Homogeneity

```python
homogeneity(grid, Aw=None, S_xyw=None, rel=1e-3) -> (V_map, V_mean)
```
Spectral homogeneity of each pixel against the total spectrum S₀:

```
V(x, y) = ( Σ_λ sqrt(S(x, y, λ) S₀(λ)) )² / ( Σ_λ S · Σ_λ S₀ )
```

V_mean is the fluence-weighted average; V = 1 means perfectly homogeneous. Pixels below `rel` × the peak fluence are NaN. Instead of a field you can pass an intensity cube `S_xyw`, e.g. `abs(Ef)**2` from `far_field`, to get the far-field homogeneity.

#### Modal content

```python
mode_content(grid, Aw, w0_of_lambda, max_order=6, idx=None) -> dict
```
Hermite–Gauss decomposition, frequency by frequency, on modes with waist `w0_of_lambda(lam)` and a flat phase. It is exact at the cell centre.

| Key | Content |
|---|---|
| `lam` | wavelengths of the analysed bins |
| `eta00` | fraction of power in HG₀₀ at each wavelength |
| `eta00_total` | spectrally integrated HG₀₀ fraction |
| `order_frac` | energy fraction in modes of order m+n = 0 … `max_order`; the last bin holds everything above |

#### Beam size and beam quality

```python
beam_moments(grid, Aw, idx=None) -> dict
```
Per wavelength: second-moment radii `wx`, `wy` [m] and `M2x`, `M2y`, computed from the Wigner moments M² = 2·sqrt(⟨x²⟩⟨kx²⟩ − ⟨x kx⟩²) with centroids removed. Also returned: `lam`, and the spectrally weighted averages `M2x_mean`, `M2y_mean`. Like any second-moment quantity, M² is sensitive to far wings: hard edges and Airy rings inflate it.

#### Compression and chirp maps

```python
compress(grid, Aw, gdd_bounds_fs2=(-30000, 30000), tod_fs3=0.0, n_scan=61, rel=1e-4) -> dict
```
Finds the uniform GDD (e.g. chirped mirrors) that maximises the peak of the spatially integrated power: a coarse scan over the bounds, then golden-section refinement. `tod_fs3` adds a fixed TOD.

| Key | Content |
|---|---|
| `gdd_fs2` | optimal GDD |
| `t_fs, P` | compressed P(t) [W] |
| `fwhm_fs` | FWHM of P(t) |
| `tl_fwhm_fs` | FWHM of the spatially resolved transform limit (flat phase in every pixel) |
| `peak_power, peak_power_TL` | peak powers [W] |
| `gdd_scan_fs2, peak_scan` | the coarse scan |

```python
local_gdd_map(grid, Aw, rel_fluence=1e-2, rel_spectrum=1e-2) -> GDD[x, y] (fs²)
```
Pixel-wise GDD from a spectrally weighted quadratic fit of the unwrapped spectral phase. It shows radial chirp. Pixels outside the beam are NaN.

```python
radial_profile(grid, M2d, nbins=None) -> (r, value)
```
Azimuthal average of a sorted 2D map (e.g. a fluence map or `local_gdd_map`).

---

## 6. Recorded history and saved files

`sim.history` is a dict of lists, with one entry per recorded pass. Wavelength-resolved quantities are interpolated on `sim.lam` (sorted).

| Key | Shape per entry | Content |
|---|---|---|
| `pass` | scalar | pass number (0 = input) |
| `energy` | scalar | energy [J] |
| `B` | scalar | accumulated on-axis peak B-integral [rad] |
| `peak_power` | scalar | peak of P(t) [W] |
| `spectrum` | (Nλ,) | spatially integrated spectrum |
| `fluence` | (Nx, Ny) | fluence at the cell centre [J/m²] |
| `fluence_mirror` | (Nx, Ny) or None | fluence at the mirror before the bounce |
| `sgram_x` | (Nx, Nλ) | S(x, λ) at y = 0 (or over `slit`) |
| `sgram_theta` | (Nθ, Nλ) or None | S(θ, λ) at θy = 0 |
| `eta00` | (Nλ,) | HG₀₀ fraction vs wavelength |
| `eta00_total` | scalar | spectrally integrated HG₀₀ fraction |
| `order_frac` | (max_order+2,) | energy fraction per mode order |
| `V_near, V_far` | scalar | mean spectral homogeneity at the centre / in the far field |
| `wx, wy, M2x, M2y` | (Nλ,) | second-moment radii and M² vs wavelength |
| `M2x_mean, M2y_mean` | scalar | spectrally weighted M² |
| `prof_near` | (nλ+1, Nx) | 1D cuts at the centre (wavelength bands + global) |
| `prof_near_proj` | (nλ+1, Nx) | 1D projections at the centre |
| `prof_far` | (nλ+1, Nθ) | far-field 1D cuts |
| `prof_mirror` | (nλ+1, Nx) or None | mirror-plane 1D cuts (None at pass 0) |
| `band_energy` | (nλ+1,) | energy in each band (last = total) [J] |
| `wall_s` | scalar | elapsed wall-clock time [s] |

`sim.save(path)` writes a compressed `.npz` containing:

- every non-empty history key, stacked over passes;
- `prof_mirror` and `fluence_mirror` for passes 1 … N only (before each bounce);
- the axes `lam`, `x`, `t`, `theta`, and, if profiles were recorded, `prof_coord`, `prof_wavelengths`, `prof_bandwidth`;
- metadata: `R`, `d`, `n_passes`, `gas`, `pressure_bar`, `n2`, `lambda0`, and `grid` = [Nx, Ny, Nt, dx, dy, dt];
- `Aw_final`, the final field, if `include_field=True`.

Load it with `d = np.load("run.npz")`.

---

## 7. Example scripts

All examples use the same toy cell: Ar at 1 bar, R = 0.5 m, Herriott N = 10, k = 3 (20 passes, 54° Gouy phase per pass), and 200 fs pulses at 1030 nm. Each runs in a few minutes on one core. Each script accepts an optional time budget in seconds (`python script.py 600`), saves a checkpoint when the budget is reached, and resumes from it on the next launch.

| Script | What it shows | Output |
|---|---|---|
| `compare_gauss_tophat.py` | matched Gaussian vs top-hat near field (Fourier injection). Plots: S(x, λ) and S(θ, λ) in/out; HG₀₀ fraction, homogeneity, mode orders, M² and B vs pass; η₀₀(λ); fluence maps; radial GDD; compression. `QUICK = False` gives a template for a real stage. | `out_gauss/`, `out_tophat/`, `run_*.npz` |
| `tophat_profiles_vs_pass.py` | 1D profiles at 1010 / 1030 / 1050 nm and the global profile, pass by pass, at the centre, at the mirror and in the far field. `INJECTION = "mirror"` (near field imaged on the mirror) or `"fourier"` (Airy at the centre). | `out_profiles_<injection>/` |
| `clipped_gauss_profiles.py` | Gaussian with a hard knife edge in x (imaged on the mirror). 1D profiles (linear and log) at 1000–1060 nm and global, at centre / mirror / far field, for selected passes; η₀₀ of each colour vs pass; spectrum. | `out_clipped/` |

---

## 8. Choosing the grid, accuracy and validation

### Grid
- **Box:** start from `suggest_grid`. Use `box_factor` ≈ 8–10 for clean Gaussians and ≥ 12–14 for top-hats, hard edges and aberrations.
- **Step:** the transverse step must resolve both the waist and the curvature phase applied by the mirror at the edge of the box. `suggest_grid` handles both.
- **Spectral window:** it must contain the broadened spectrum with margin, because the Kerr source is switched off near the window edges. Set `lambda_min` and `lambda_max` at least 30–50 % wider than the expected −30 dB width.
- **Time window:** it must hold the chirped pulse at the end of the cell.

### Accuracy
- `phi_max = 0.02` rad is a good default. Check convergence by halving it.
- `complex64` is accurate enough for typical runs (B of a few tens of rad). Switch to `complex128` to check, or for very long runs.

### Validation checklist for a new configuration
1. **Linear imaging.** With `Material(..., n2=0)` and a re-entrant cell, the output after 2N passes must overlap the input with overlap > 0.99, for any input profile.
2. **Energy.** Watch the `E = …` column. A loss of more than ~1 % means the box absorber is filtering the beam spatially or spectrally. Enlarge the box or the spectral window before interpreting any "cleaning".
3. **B-integral.** The final `B` should be close to `cell.match(pulse)["B_total"]` for a matched Gaussian.
4. **Convergence.** Halve `phi_max` and check that the output does not change.
5. **n₂ values.** Replace the tabulated values with your own.

---

## 9. Performance and memory

- **Cost per step:** 2 spatial 2D FFTs, 8 temporal 1D FFTs (RK4) and a few array operations. With SciPy on one core, a 150 × 150 × 135 grid takes about 0.1–0.4 s per step. A 20-pass run at B ≈ 16 rad needs about 900 steps.
- **Steps per pass:** they scale with the B-integral per pass divided by `phi_max`.
- **Memory:** about 8–10 field arrays: the field, RK4 buffers, the linear operator `D`, up to `cache_size` cached propagators and the cached mirror transfer. One complex64 array is Nx·Ny·Nt·8 bytes, so 256 × 256 × 256 is 134 MB.
- **Speed-ups:**
  - SciPy backend with all cores (`fft_backend="scipy"`, `workers=-1`, the default with `"auto"`).
  - `record_every > 1`, or `far_field=False` (far-field diagnostics use per-colour matrix transforms).
  - Fewer `profile_wavelengths`.
  - A smaller `cache_size` if memory is tight.
- **Long runs:** use `run(max_seconds=…)` with `checkpoint` / `resume`.

---

## 10. Limitations and extension points

**Not included**
- **Ionisation and plasma:** fine for MPCs operated well below ionisation thresholds.
- **Raman response:** relevant for N₂, air and other molecular gases.
- **Off-axis geometry:** the real cell is folded, so the beam hits the mirrors at small angles. The resulting astigmatism and the actual spot pattern are not modelled; the unfolded on-axis picture is used.
- **Uniform medium:** gas plates, windows or pressure gradients are not built in. The nonlinear mode matching formula also assumes a uniformly filled cell with constant power during a pass.
- **Vector effects:** the field is scalar, with a single polarisation.

**Extension points**
- **New nonlinear terms:** subclass `UPPESolver` and override `nonlinear(Aw, track=False)`. Ionisation, Raman or higher-order Kerr terms go there, in the (x, y, Ω) ↔ (x, y, t) representation already used for the Kerr term.
- **Other elements:** custom mirrors and optics can be inserted between passes by running `sim.run(n_passes=1)` in a loop and modifying `sim.Aw` (e.g. a lens phase, an aperture, a spectral filter). `sim.solver.propagate` and `sim.solver.linear` can be called directly for arbitrary propagation distances.
- **Arbitrary inputs:** a spectral phase (`add_spectral_phase(phase=...)`), a spatial profile (`Profile`), or a space-time coupling (`Beam(custom_phase=...)`) can be any function.
