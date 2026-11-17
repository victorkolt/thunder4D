"""Design helper for bulk and hybrid MPCs: Pharos-like laser (250 uJ, 180 fs) -> ~40 fs.

Same laser as the bulk MPC of Westerberg et al., APL Photonics 10, 096103 (2025): R = 200 mm,
7 round trips, two 1 mm AR-coated FS plates at +-140 mm that are the windows of a small evacuated
cell (vacuum between the plates, air between the plates and the mirrors); 42 fs measured.

Two searches:
  1. bulk cell in vacuum, n identical plates from the stock list (FS or CaF2),
  2. that published layout (air at the mirrors, vacuum section between two FS windows) and a
     variant with one FS and one CaF2 window, given as your own Layouts.
fast1d follows the plates and gas sections; the chosen design is then exported and checked with the
4D solver (validate_4d, coarse grid: seconds to minutes on a CPU).

    python examples/design_bulk_mpc.py
"""
from thunder4d.design import (
    DesignProblem,
    Fast1DQuick,
    LaserSpec,
    Layout,
    MPCDesigner,
    PlateSpec,
    RegionSpec,
    SearchSpace,
    Target,
    validate_4d,
)

if __name__ == "__main__":
    problem = DesignProblem(laser=LaserSpec(energy_J=250e-6, duration_fs=180.0, rep_rate_Hz=1e3),
                            target=Target(objective="target_duration", target_duration_fs=40.0))
    C = problem.constraints
    C.mirrors.allowed_roc_m = (0.2, 0.3, 0.5)                 # short cells
    C.space.max_separation_m.value = 1.0
    C.plates.allowed_materials = ("FS", "CaF2")
    C.plates.allowed_thicknesses_mm = (0.5, 1.0, 2.0, 3.0)    # what the supplier stocks
    C.plates.ar_reflectivity = 2e-3                           # residual R per AR face

    # 1. bulk cell in vacuum, identical plates
    problem.search = SearchSpace(gases=("vacuum",), round_trips=(4, 15), plate_counts=(1, 2),
                                 plate_positions_rel=(0.5, 0.75))
    bulk = MPCDesigner(problem, quick=Fast1DQuick(), workers=4).run()
    print(bulk.table())

    # 2. your own layouts: windows of an evacuated section, in air at a fixed 1 bar
    windows = RegionSpec(-0.73, 0.73, "vacuum")
    problem.search = SearchSpace(
        gases=("air",), roc_m=(0.2,), round_trips=(5, 9),
        layouts=(Layout((PlateSpec("FS", 1.0, -0.73, ar_reflectivity=2e-3),
                         PlateSpec("FS", 1.0, 0.73, ar_reflectivity=2e-3)), (windows,), name="2 FS windows"),
                 Layout((PlateSpec("FS", 1.0, -0.73, ar_reflectivity=2e-3),
                         PlateSpec("CaF2", 1.0, 0.73, ar_reflectivity=2e-3)), (windows,), name="FS + CaF2")))
    C.pressure.pressure_bar = (1.0, 1.0)                     # the air is not tuned
    problem.refinement.tune = ()
    windowed = MPCDesigner(problem, quick=Fast1DQuick(), workers=4).run()
    print(windowed.table())
    best = windowed.best
    print(best.report(problem))

    # 4D check of the chosen design (Kerr-matched injection, coarse grid)
    print(validate_4d(problem, best, verbose=False).summary())
    windowed.export("windowed_bulk_design.json")
