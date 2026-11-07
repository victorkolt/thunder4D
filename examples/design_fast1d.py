"""Design a gas MPC with the fast1d quick solver, then check the best design (validation phase).

Catalogue changes (register_mirror, load_catalog, GASES[...]) go at the top level of this file,
outside the ``if __name__ == "__main__":`` block, so that the worker processes see them.
"""
from thunder4d.design import DesignProblem, Fast1DQuick, LaserSpec, MPCDesigner, Target, starting_points

if __name__ == "__main__":
    problem = DesignProblem(laser=LaserSpec(energy_J=2e-3, duration_fs=190.0, rep_rate_Hz=10e3),
                            target=Target(objective="target_duration", target_duration_fs=40.0))
    problem.constraints.space.max_separation_m.value = 2.0      # [m]
    problem.constraints.mirrors.allowed_coatings = ["dielectric_1030nm"]
    problem.refinement.n_candidates = 5
    problem.refinement.neighbour_n_passes = 1
    problem.refinement.max_evaluations = 40

    # 1) design loop: fast1d with its "fast" preset, in parallel processes
    quick = Fast1DQuick(accuracy="fast")
    starts = starting_points(problem, gases=("He", "Ne", "Ar", "Kr"))
    result = MPCDesigner(problem, quick=quick, workers=4).refine(starts)
    print(result.table())
    for c in sorted(result.refined, key=lambda c: c.score):
        print("\n" + c.report(problem))
    result.save_json("design_fast1d.json")

    # 2) validation phase (not in the loop): is fast1d converged for the chosen design?
    best = result.best
    print("\n" + quick.validate(problem, best).summary())                     # the "fast" settings
    print("\n" + quick.validate(problem, best, accuracy="default").summary())  # a reference run
    # full output (fields, spectra, per-pass data): quick.simulate(problem, best, "default").plot()
