"""MPC design helper (gas-filled, bulk and hybrid Herriott cells).

    from thunder4d.design import DesignProblem, Fast1DQuick, LaserSpec, MPCDesigner, starting_points
    problem = DesignProblem(laser=LaserSpec.preset("mpc1_18mJ_350fs"))
    result = MPCDesigner(problem, quick=Fast1DQuick()).refine(starting_points(problem))
    print(result.table());  result.save_json("design.json")
    cell = result.best.to_cell()          # -> thunder4d MPC for 4D verification

Bulk / hybrid cells: plates (each its own material, thickness, position, coating) and gas sections
(each its own gas and pressure) described by a Layout; the search tries n identical plates
(SearchSpace.plate_counts, stock thicknesses of Constraints.plates) and your own layouts
(SearchSpace.layouts). fast1d and the 4D export follow the layout. examples/design_bulk_mpc.py.

Tutorial: examples/design_tutorial.ipynb
"""
from . import catalog
from .designer import (
    Candidate,
    DesignResult,
    MPCDesigner,
    QuickResult,
    QuickSolver,
    evaluate,
    starting_points,
)
from .export import Validation4D, export_input, to_input, validate_4d
from .fast1d_quick import Fast1DQuick
from .scan import AnalyticQuick, DesignMap, design_map
from .specs import (
    LASER_PRESETS,
    Constraints,
    DesignProblem,
    GasConstraints,
    IonizationConstraints,
    KerrConstraints,
    LaserSpec,
    Layout,
    Limit,
    Losses,
    Max,
    Min,
    MirrorConstraints,
    Off,
    PlateConstraints,
    PlateSpec,
    PressureConstraints,
    Refinement,
    RegionSpec,
    SearchSpace,
    SpaceConstraints,
    SpotConstraints,
    Target,
    Tolerance,
    register_laser_preset,
)

__all__ = [
    "LASER_PRESETS",
    "AnalyticQuick",
    "Candidate",
    "Constraints",
    "DesignMap",
    "DesignProblem",
    "DesignResult",
    "Fast1DQuick",
    "GasConstraints",
    "IonizationConstraints",
    "KerrConstraints",
    "LaserSpec",
    "Layout",
    "Limit",
    "Losses",
    "MPCDesigner",
    "Max",
    "Min",
    "MirrorConstraints",
    "Off",
    "PlateConstraints",
    "PlateSpec",
    "PressureConstraints",
    "QuickResult",
    "QuickSolver",
    "Refinement",
    "RegionSpec",
    "SearchSpace",
    "SpaceConstraints",
    "SpotConstraints",
    "Target",
    "Tolerance",
    "Validation4D",
    "catalog",
    "design_map",
    "evaluate",
    "export_input",
    "register_laser_preset",
    "starting_points",
    "to_input",
    "validate_4d",
]
