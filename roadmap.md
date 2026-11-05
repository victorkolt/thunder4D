## Overview

| Release | Content | Status |
|---|---|---|
| v1.1.0 | Standardized benchmark suite | Planned |
| v1.2.0 | Design helper for gas-filled MPCs | Planned |
| v1.3.0 | Graphical interface | Planned |
| v1.4.0 | Multi-stage simulations | Planned |
| v1.5.0 | Raman response | Planned |
| v1.6.0 | Bulk and hybrid MPCs | Planned |
| v1.7.0 | Design helper for bulk and hybrid MPCs | Planned |
| v1.8.0 | Realistic mirrors | Planned |
| v1.9.0 | Performance and scaling | Planned |
| v2.0.0 | Vectorial solver | Planned |

Items marked *(possible extension)* are candidates for the release but not committed.

---

## v1.1.0 — Standardized benchmark suite

Goal: make validation against published results reproducible and easy to extend.

- Standard benchmark format: one script per case, shared input schema, shared
  analysis and plotting routines, so new cases are added and compared the same way
- Benchmarks against published gas-filled MPCs, from the µJ/mJ regime up to
  high-energy cells
- Reference data (spectra, temporal profiles, energy transmission) stored alongside
  each benchmark, with its source
- Summary report: one table/figure comparing all benchmarks to their references
- Low-resolution versions of the benchmarks run as regression tests in CI
- Benchmark pages in the documentation

## v1.2.0 — Design helper for gas-filled MPCs

Goal: go from an input laser and a target output to a ready-to-run MPC configuration.

- Inputs: pulse energy, duration, central wavelength, beam size, and the target
  output (duration or spectral broadening)
- Constraints: fluence on the mirrors, cell length, peak power relative to the
  critical power, B-integral per pass, ionization threshold
- Proposed design: gas and pressure, mirror radius of curvature, cell length,
  number of round trips, mode-matched beam size, with the margin on each constraint
- Scan of the design space, showing which constraint limits each region
- Export of the chosen design as a standard THUNDER4D input file
- Optional quick validation run of the proposed design with the full solver
- Documentation example: designing a cell, then simulating it

## v1.3.0 — Graphical interface

Goal: set up, run and inspect simulations without writing scripts.

- Optional install, so the core package stays lightweight:
  `pip install thunder4d[gui]`
- Simulation builder: laser, gas/medium, cell geometry and mirrors, with checks of
  the cell geometry and the input parameters before launching
- Design helper panel (from v1.2.0), feeding the simulation builder
- Save/load of the configuration as a standard THUNDER4D input file, so anything
  built in the GUI can be rerun from a script
- Run control with live monitoring (pass number, energy, B-integral, peak intensity)
- Results viewer: caustic, B-integral, spectra and temporal profiles, compressed
  pulse, spatio-spectral maps and (x, λ) spectrograms
- Export of figures and data
- GUI tutorial in the documentation

## v1.4.0 — Multi-stage simulations

Goal: simulate cascaded compressors (for example two MPC stages) in a single run.

- Chaining of several MPC stages, the output field of one stage being the input of
  the next
- Inter-stage elements: dispersion compensation (chirped mirrors, GDD/TOD, plates),
  free-space propagation, windows, telescopes for mode matching
- Per-stage outputs and diagnostics, plus a summary over the whole chain
- Multi-stage configurations in the input file, the design helper and the GUI
- Multi-stage benchmark (two-stage compressor)

## v1.5.0 — Raman response

Goal: simulate molecular gases where the delayed nonlinear response matters.

- Delayed (Raman) nonlinear polarization term in the UPPE, with a response
  function and a Raman fraction
- Rotational Raman parameters for molecular gases (N₂, O₂, air, …) in the
  material database
- Option to switch Raman on/off per stage, with no cost when off
- Raman taken into account in the gas design helper
- Validation case(s) against published Raman-affected MPC results, added to the
  benchmark suite
- *(possible extension)* Vibrational Raman contribution

## v1.6.0 — Bulk and hybrid MPCs

Goal: handle solid nonlinear media inside the cell.

- Bulk MPC: propagation through solid plates, with the dispersion and n₂ of
  common materials (fused silica, sapphire, …) in the material database
- Hybrid MPC: gas filling plus one or more plates, with free placement in the
  cell
- Plate geometry parameters: thickness, position, angle of incidence
- Benchmark case(s) for bulk / hybrid MPCs
- *(possible extension)* Surface (Fresnel) losses and AR coatings on the plates

## v1.7.0 — Design helper for bulk and hybrid MPCs

Goal: extend the design helper to cells containing solid plates.

- Plate parameters: material, thickness, number of plates, position in the cell
- Constraints specific to plates: fluence and B-integral per plate, self-focusing
  in the plate, damage threshold
- Combined gas + plate designs, with the split of nonlinearity between gas and plates
- Export to input files and integration in the GUI

## v1.8.0 — Realistic mirrors

Goal: bring the cell mirrors closer to the hardware.

- Measured or vendor reflectivity and GDD curves applied at each bounce
- Mirror database (dielectric / metallic) shared across simulations
- Per-mirror fluence monitoring with damage-threshold warnings
- *(possible extension)* Astigmatism from non-normal incidence and mirror misalignment

## v1.9.0 — Performance and scaling

Goal: run larger grids and parameter scans faster.

- Multi-GPU execution
- Checkpointing and restart of long runs
- Batch parameter scans (one input file → many runs)

## v2.0.0 — Vectorial solver

Goal: resolve the polarization state of the field, i.e. propagate the two transverse
components (E<sub>x</sub>, E<sub>y</sub>) instead of a single scalar field.

- Two-component field with coupled nonlinear propagation (Kerr effect for
  arbitrary polarization, including cross-coupling between components)
- Input pulses with any polarization state: linear, circular, elliptical, and
  spatially varying polarization
- Polarization-dependent mirror response: s/p reflectivity and phase at
  non-normal incidence
- Polarization-resolved outputs and diagnostics (Stokes parameters, polarization
  ellipse vs. x, y, t / λ)
- Vectorial benchmarks: circular vs. linear polarization in gas MPCs
- Raman, bulk/hybrid media and multi-stage simulations extended to the vectorial case
- *(possible extension)* Nonlinear polarization rotation and ellipse rotation studies