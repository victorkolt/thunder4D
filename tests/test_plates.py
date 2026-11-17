"""Bulk / hybrid MPCs: solids, coatings, plates, equivalent cell, Kerr matching, propagation."""
import numpy as np
import pytest

import thunder4d as m
from thunder4d import diagnostics as dg

from helpers import LAM0, overlap

AR = m.Material("Ar", 1.0)
FS = m.Solid("FS")


# ------------------------------------------------------------ solids
@pytest.mark.parametrize("name, lam, n_ref, gvd_ref", [
    ("FS", 800e-9, 1.4533, 36.16), ("CaF2", 800e-9, 1.4306, 27.87), ("sapphire", 800e-9, 1.7601, 58.0)])
def test_solid_index_and_gvd_match_reference_values(name, lam, n_ref, gvd_ref):
    s = m.Solid(name)
    assert float(s.n_lam(lam)) == pytest.approx(n_ref, abs=2e-4)
    assert s.gvd(lam) / 1e3 == pytest.approx(gvd_ref, rel=0.01)       # fs^2/mm


def test_uniaxial_axes_differ_and_unknown_axis_is_rejected():
    assert float(m.Solid("MgF2", axis="e").n_lam(LAM0)) > float(m.Solid("MgF2").n_lam(LAM0))
    with pytest.raises(ValueError):
        m.Solid("FS", axis="e")


def test_custom_solid_needs_n2_and_aliases_resolve():
    with pytest.raises(ValueError):
        m.Solid("myglass", sellmeier=((1.0, 0.1),))
    assert m.Solid("fused silica").name == "FS"


# ------------------------------------------------------------ surfaces
def test_bare_plate_is_fresnel():
    p = m.Plate(FS, 1e-3)
    n = float(FS.n_lam(LAM0))
    R = ((n - 1) / (n + 1)) ** 2
    assert float(p.transmission(m.vacuum(), LAM0)) == pytest.approx((1 - R) ** 2, rel=1e-9)


def test_coatings_reflectivity_and_phase():
    lam = np.array([900e-9, 1000e-9, 1100e-9])
    c = m.Coating.from_table([900e-9, 1100e-9], [0.01, 0.03])
    assert c.R(lam) == pytest.approx([0.01, 0.02, 0.03])
    p = m.Plate(FS, 1e-3, coating=("ar", m.Coating(0.005, gdd_fs2=10.0)))
    assert np.abs(p.face_transfer("A", AR, lam, LAM0)) == pytest.approx(1.0)
    assert np.abs(p.face_transfer("B", AR, lam, LAM0)) ** 2 == pytest.approx(0.995)
    assert float(np.angle(p.face_transfer("B", AR, LAM0, LAM0))) == pytest.approx(0, abs=1e-12)


# ------------------------------------------------------------ geometry
def test_plate_shortens_the_cell_by_t_one_minus_one_over_n():
    t = 3e-3
    cell = m.MPC(1.0, 1.5, None, 10, plates=m.Plate(FS, t, 0.2), wavelength=LAM0)
    n = float(FS.n_lam(LAM0))
    assert cell.equivalent_separation() == pytest.approx(1.5 - t * (1 - 1 / n), rel=1e-12)


def _abcd_two_passes(cell, lam):
    """Independent check: physical ABCD (real slopes) over two passes, plates with their interface
    matrices. Returns cos of the two-pass Gouy phase."""
    M = np.eye(2)
    for pss in (1, 2):
        for seg in cell.layout(pss, lam):
            L = seg["z1"] - seg["z0"]
            if seg["kind"] == "gas":
                M = np.array([[1.0, L], [0.0, 1.0]]) @ M
                continue
            n1, n2 = float(cell.medium.n_lam(lam)), float(seg["medium"].n_lam(lam))
            M = (np.diag([1, n2 / n1]) @ np.array([[1.0, L], [0.0, 1.0]]) @ np.diag([1, n1 / n2])) @ M
        M = np.array([[1.0, 0.0], [-2 / cell.R, 1.0]]) @ M
    return (M[0, 0] + M[1, 1]) / 2


@pytest.mark.parametrize("pos", [0.0, 0.25, -0.4])
def test_equivalent_cell_matches_explicit_abcd(pos):
    plates = [m.Plate(FS, 5e-3, pos), m.Plate(m.Solid("CaF2"), 2e-3, pos + 0.1)]
    cell = m.MPC(1.0, 1.6, AR, 10, plates=plates, wavelength=LAM0)
    assert _abcd_two_passes(cell, LAM0) == pytest.approx(np.cos(2 * cell.theta), abs=1e-10)


def test_off_centre_plate_alternates_side_and_faces():
    cell = m.MPC(1.0, 1.6, AR, 10, plates=m.Plate(FS, 2e-3, 0.3), wavelength=LAM0)
    s1 = next(s for s in cell.layout(1) if s["kind"] == "plate")
    s2 = next(s for s in cell.layout(2) if s["kind"] == "plate")
    assert (s1["z0"] + s1["z1"]) / 2 == pytest.approx(0.3)
    assert (s2["z0"] + s2["z1"]) / 2 == pytest.approx(-0.3)
    assert (s1["entry_face"], s2["entry_face"]) == ("A", "B")
    assert sum(s["z1"] - s["z0"] for s in cell.layout(1)) == pytest.approx(cell.d)
    assert cell.layout(1)[-1]["u1"] == pytest.approx(cell.equivalent_separation())


def test_centre_of_equivalent_cell():
    sym = m.MPC(1.0, 1.6, AR, 10, plates=[m.Plate(FS, 2e-3, 0.3), m.Plate(FS, 2e-3, -0.3)], wavelength=LAM0)
    assert sym.centre_z(1) == pytest.approx(0.0, abs=1e-15)
    off = m.MPC(1.0, 1.6, None, 10, plates=m.Plate(FS, 10e-3, 0.3), wavelength=LAM0)
    shift = 10e-3 * (1 - 1 / float(FS.n_lam(LAM0))) / 2       # half the shortening, away from the plate
    assert off.centre_z(1) == pytest.approx(-shift, rel=1e-9)
    assert off.centre_z(2) == pytest.approx(+shift, rel=1e-9)


def test_layout_errors():
    with pytest.raises(ValueError):          # outside the cell
        m.MPC(1.0, 1.0, AR, 10, plates=m.Plate(FS, 2e-3, 0.4999), wavelength=LAM0)
    with pytest.raises(ValueError):          # overlap
        m.MPC(1.0, 1.0, AR, 10, plates=[m.Plate(FS, 2e-3, 0.0), m.Plate(FS, 2e-3, 1e-3)], wavelength=LAM0)
    with pytest.raises(ValueError):          # no reference wavelength
        m.MPC(1.0, 1.0, AR, 10, plates=m.Plate(FS, 2e-3))


def test_herriott_with_plates_is_reentrant():
    cell = m.MPC.herriott(0.5, 20, 3, AR, plates=m.Plate(FS, 3e-3, 0.05), wavelength=LAM0)
    turns = 20 * cell.theta / (2 * np.pi)
    assert turns == pytest.approx(round(turns), abs=1e-9)
    assert cell.d > 0.5 * (1 + np.cos(2 * np.pi * 3 / 20))       # the plate shortens the cell


# ------------------------------------------------------------ Kerr-matched mode (moments)
@pytest.mark.parametrize("ratio", [0.0, 0.3, 0.7])
def test_moment_matching_reproduces_the_analytic_gas_mode(ratio):
    cell = m.MPC(0.5, 0.9, m.Material("Ar", 1.0), 10)
    P = ratio * cell.medium.critical_power(LAM0)
    ref = cell.nonlinear_mode(P, LAM0)
    num = cell._plate_mode(P, LAM0)
    assert num["w_mirror"] == pytest.approx(ref["w_mirror"], rel=1e-6)
    assert num["w0"] == pytest.approx(ref["w0"], rel=1e-6)
    # the analytic B uses the vacuum wavelength in zR' (n - 1 ~ 3e-4 for 1 bar Ar)
    assert num["B_per_pass"] == pytest.approx(ref["B_per_pass"], rel=1e-3, abs=1e-12)


def test_plate_kerr_lens_far_above_pc_has_a_matched_mode():
    """Westerberg et al., APL Photonics 10, 096103 (2025): P ~ 250 Pc in 1 mm FS plates."""
    cell = westerberg2025()
    lin = cell.linear_mode(LAM0)
    mode = cell.nonlinear_mode(1.3e9, LAM0)
    assert all(d["P_over_Pc"] > 200 for d in mode["plates"])
    assert mode["w_mirror"] > lin["w_mirror"] and mode["w0"] < lin["w0"]     # Kerr lenses focus
    assert mode["B_pass"][0] == pytest.approx(mode["B_pass"][1], rel=1e-9)   # symmetric layout
    _, _, smin = cell._segment_moments(1e-7, 0.0, 0.05, 190.0, LAM0)    # 5 cm of glass at 190 Pc
    assert smin <= 0                                                      # -> collapse detected


# ------------------------------------------------------------ propagation
def westerberg2025(n_passes=14, air=None):
    """Bulk MPC of Westerberg et al., APL Photonics 10, 096103 (2025): R = 200 mm, d = 382 mm,
    7 round trips, two 1 mm AR-coated FS plates 280 mm apart around the centre, windows of an
    evacuated cell (air between the plates and the mirrors if `air` is given, else all vacuum)."""
    plates = [m.Plate(FS, 1e-3, +0.14, coating="ar"), m.Plate(FS, 1e-3, -0.14, coating="ar")]
    regions = [(-0.14, 0.14, m.vacuum())] if air is not None else ()
    return m.MPC(0.2, 0.382, air, n_passes, plates=plates, regions=regions, wavelength=LAM0)


def _radius(grid, A, F=None):
    """1/e^2 radius along x (2 x rms) of the fluence (sorted, as dg.fluence returns it)."""
    F = dg.fluence(grid, A) if F is None else F
    X2 = (grid.xs ** 2)[:, None]
    return 2 * np.sqrt(np.sum(F * X2) / np.sum(F))


@pytest.mark.parametrize("solver", ["lens", "standard"])
def test_linear_eigenmode_is_preserved_through_an_off_centre_plate(solver):
    glass = m.Solid("FS", n2=0.0)
    cell = m.MPC(1.0, 0.5, None, 4, plates=m.Plate(glass, 20e-3, 0.1), wavelength=LAM0)
    if solver == "lens":
        g = m.suggest_lens_grid(cell, LAM0, 1000e-9, 1060e-9, 300e-15, dtype=np.complex128)
    else:
        g = m.Grid(64, 70e-6, 32, LAM0, dt=8e-15, dtype=np.complex128, device="cpu")
    E = m.gaussian_spectrum(g, fwhm_fs=300)
    p = m.Pulse.cell_matched(g, E, cell, 1e-6, nonlinear=False)
    Sim = m.LensFrameSimulation if solver == "lens" else m.MPCSimulation
    sim = Sim(p, cell, near_field=False, record_mirror=False)
    w_in = _radius(g, sim.Aw)
    sim.run(verbose=False)
    for k in (1, 2, 3, 4):
        assert sim.history["w_peak_x"][k] == pytest.approx(sim.history["w_peak_x"][0], rel=2e-3)
    assert w_in == pytest.approx(cell.linear_mode(LAM0)["w0"], rel=2e-2)
    assert overlap(sim.Aw, p.Aw) > 0.999 if solver == "lens" else True


def test_bare_plates_lose_the_fresnel_reflections():
    glass = m.Solid("FS", n2=0.0)
    cell = m.MPC(1.0, 0.5, None, 2, plates=[m.Plate(glass, 2e-3, 0.1), m.Plate(glass, 2e-3, -0.1)],
                 wavelength=LAM0)
    g = m.suggest_lens_grid(cell, LAM0, 1000e-9, 1060e-9, 300e-15, dtype=np.complex128)
    p = m.Pulse.cell_matched(g, m.gaussian_spectrum(g, fwhm_fs=300), cell, 1e-6, nonlinear=False)
    sim = m.LensFrameSimulation(p, cell, near_field=False)
    sim.run(verbose=False)
    T = float(cell.plate_transmission(LAM0))
    assert sim.history["energy"][1] / sim.history["energy"][0] == pytest.approx(T, rel=1e-3)
    assert sim.history["energy"][2] / sim.history["energy"][0] == pytest.approx(T ** 2, rel=1e-3)


def test_plate_dispersion_and_group_delay_frame():
    """Linear: after one pass the pulse carries the GDD of the plate and stays centred in time."""
    glass = m.Solid("FS", n2=0.0)
    t = 10e-3
    cell = m.MPC(1.0, 0.5, None, 2, plates=m.Plate(glass, t, 0.1), wavelength=LAM0)
    g = m.suggest_lens_grid(cell, LAM0, 950e-9, 1110e-9, 40e-15, time_window_factor=30, dtype=np.complex128)
    p = m.Pulse.cell_matched(g, m.gaussian_spectrum(g, fwhm_fs=40), cell, 1e-6, nonlinear=False)
    sim = m.LensFrameSimulation(p, cell, near_field=False)
    sim.run(1, verbose=False)
    c = dg.compress(g, sim.Aw, gdd_bounds_fs2=(-2000, 0), n_scan=201)
    assert -c["gdd_fs2"] == pytest.approx(glass.gvd(LAM0) * t, rel=0.03)
    tt, P = dg.temporal_power(g, sim.Aw)
    assert abs(np.sum(tt * P) / np.sum(P)) < 5e-15


def test_lens_frame_and_standard_solver_agree_with_a_kerr_plate():
    glass = m.Solid("FS")
    cell = m.MPC(1.0, 0.5, None, 2, plates=m.Plate(glass, 5e-3, 0.1, coating="ar"), wavelength=LAM0)
    gl = m.suggest_lens_grid(cell, LAM0, 960e-9, 1110e-9, 120e-15, box_factor=6, dtype=np.complex128)
    gs = m.Grid(80, 50e-6, gl.Nt, LAM0, dt=gl.dt, dtype=np.complex128, device="cpu")
    out = []
    for g, Sim in ((gl, m.LensFrameSimulation), (gs, m.MPCSimulation)):
        p = m.Pulse.cell_matched(g, m.gaussian_spectrum(g, fwhm_fs=120), cell, 40e-6, verbose=False)
        sim = Sim(p, cell, near_field=False, record_mirror=False, phi_max=0.005)
        sim.run(verbose=False)
        out.append(sim)
    a, b = out
    assert a.solver.B == pytest.approx(b.solver.B, rel=0.03)
    assert a.solver.B > 1.0                                       # a real Kerr effect in the plate
    Sa, Sb = a.history["spectrum"][-1], b.history["spectrum"][-1]
    assert overlap(Sa, Sb) > 0.995
    assert a.history["w_peak_x"][-1] == pytest.approx(b.history["w_peak_x"][-1], rel=0.03)


def test_westerberg2025_matched_beam_stays_on_the_matched_mode():
    """Matched at the intensity-weighted power, the fluence on the mirrors keeps the size of the
    Kerr-matched mode from pass to pass (P ~ 250 Pc in the plates)."""
    cell = westerberg2025(n_passes=4)
    g = m.suggest_lens_grid(cell, LAM0, 900e-9, 1180e-9, 180e-15, box_factor=6, points_per_waist=6,
                            time_window_factor=6, dtype=np.complex64)
    p = m.Pulse.cell_matched(g, m.gaussian_spectrum(g, fwhm_fs=180), cell, 250e-6, power="weighted",
                             verbose=False)
    mode = cell.nonlinear_mode(p.weighted_power(), LAM0)
    sim = m.LensFrameSimulation(p, cell, near_field=False, phi_max=0.05)
    sim.run(verbose=False)
    for F in sim.history["fluence_mirror"][1:]:
        assert _radius(g, None, F) * sim.mirror_scale == pytest.approx(mode["w_mirror"], rel=0.02)
    assert sim.history["B"][1] == pytest.approx(cell.nonlinear_mode(p.peak_power(), LAM0)["B_pass"][0], rel=0.1)
    assert min(sim.history["eta00_total"]) > 0.95


# ------------------------------------------------------------ imaged near field with plates
@pytest.mark.parametrize("solver", ["lens", "standard"])
def test_imaged_eigenmode_with_plates_is_the_matched_input(solver):
    """A gaussian of the mirror spot imaged on mirror B (pass 1) = the linear eigenmode input."""
    glass = m.Solid("FS", n2=0.0)
    cell = m.MPC(1.0, 0.5, None, 2, plates=m.Plate(glass, 20e-3, 0.1, coating="ar"), wavelength=LAM0)
    if solver == "lens":
        g = m.suggest_lens_grid(cell, LAM0, 1020e-9, 1040e-9, 1e-12, dtype=np.complex128)
        build = m.imaged_near_field_lens
    else:
        g = m.Grid(64, 70e-6, 32, LAM0, dt=40e-15, dtype=np.complex128, device="cpu")
        build = m.Pulse.imaged_near_field
    E = m.gaussian_spectrum(g, fwhm_fs=1000)
    wm = cell.linear_mode(LAM0)["w_mirror"]
    imaged = build(grid=g, E=E, beam=m.Beam(m.gaussian(wm)), energy=1e-6, cell=cell, verbose=False)
    ref = m.Pulse.cell_matched(g, E, cell, 1e-6, nonlinear=False)
    assert overlap(imaged.Aw, ref.Aw) > 0.999


def test_imaged_top_hat_lands_on_the_first_mirror_with_plates():
    """The top-hat imaged on mirror B arrives there after pass 1 exactly as in the empty cell
    with the same equivalent geometry (the plates only change the path in between)."""
    glass = m.Solid("FS", n2=0.0)
    plates = [m.Plate(glass, 10e-3, 0.1), m.Plate(glass, 10e-3, -0.15)]
    loaded = m.MPC(1.0, 0.5, None, 1, plates=plates, wavelength=LAM0)
    empty = m.MPC(1.0, loaded.equivalent_separation(), None, 1)
    radii = []
    for cell in (loaded, empty):
        g = m.suggest_lens_grid(cell, LAM0, 1020e-9, 1040e-9, 1e-12, box_factor=12, points_per_waist=12,
                                dtype=np.complex128)
        a = 1.12 * cell.linear_mode(LAM0)["w_mirror"]
        p = m.imaged_near_field_lens(g, m.gaussian_spectrum(g, fwhm_fs=1000), m.Beam(m.top_hat(a)), 1e-6, cell,
                                     verbose=False)
        sim = m.LensFrameSimulation(p, cell, near_field=False)
        sim.run(verbose=False)
        radii.append(_radius(g, None, sim.history["fluence_mirror"][1]) * sim.mirror_scale / a)
    assert radii[0] == pytest.approx(radii[1], rel=1e-3)


# ------------------------------------------------------------ gas regions
def test_region_filling_the_whole_cell_is_the_plain_cell():
    ar = m.Material("Ar", 2.0)
    plain = m.MPC(0.5, 0.9, ar, 4)
    region = m.MPC(0.5, 0.9, None, 4, regions=[(-0.45, 0.45, ar)], wavelength=LAM0)
    assert region.ref_medium is ar
    assert region.equivalent_separation() == pytest.approx(0.9, rel=1e-14)
    P = 0.5 * ar.critical_power(LAM0)
    a, b = plain.nonlinear_mode(P, LAM0), region.nonlinear_mode(P, LAM0)
    assert b["w_mirror"] == pytest.approx(a["w_mirror"], rel=1e-6)
    g = m.suggest_lens_grid(plain, LAM0, 960e-9, 1110e-9, 100e-15, dtype=np.complex128)
    E = m.gaussian_spectrum(g, fwhm_fs=100)
    sims = []
    for c in (plain, region):
        p = m.Pulse.at_focus(g, E, m.Beam(m.gaussian(a["w0"]), chromatic_size="eigenmode"), 0.2e-3)
        s = m.LensFrameSimulation(p, c, near_field=False, phi_max=0.01)
        s.run(verbose=False)
        sims.append(s)
    assert sims[1].solver.B == pytest.approx(sims[0].solver.B, rel=0.01)
    assert overlap(sims[1].Aw, sims[0].Aw) > 0.999


def test_vacuum_cell_between_window_plates_in_air():
    """Westerberg 2025 with the air: the two plates are the windows of an evacuated cell."""
    air = m.Material("air", 1.0)
    cell = westerberg2025(air=air)
    segs = cell.layout(1)
    assert [s["medium"].name for s in segs] == ["air", "FS", "vacuum", "FS", "air"]
    assert segs[1]["amb_in"] is air and segs[1]["amb_out"].name == "vacuum"
    assert cell.ref_medium is air
    n_air = float(air.n_lam(LAM0))
    vac = 0.28 - 1e-3                                   # vacuum between the inner faces
    glass = 2e-3 * n_air / float(FS.n_lam(LAM0))
    assert cell.equivalent_separation() == pytest.approx(0.382 - 2e-3 - vac + vac * n_air + glass, rel=1e-12)
    mode = cell.nonlinear_mode(1.3e9, LAM0)
    assert mode["B_per_pass"] > westerberg2025().nonlinear_mode(1.3e9, LAM0)["B_per_pass"]    # + air


def test_region_errors():
    ar, he = m.Material("Ar", 1.0), m.Material("He", 1.0)
    with pytest.raises(ValueError):          # mirrors in different gases
        m.MPC(0.5, 0.9, ar, 4, regions=[(0.2, 0.45, he)], wavelength=LAM0)
    with pytest.raises(ValueError):          # overlap
        m.MPC(0.5, 0.9, ar, 4, regions=[(-0.2, 0.1, he), (0.0, 0.2, he)], wavelength=LAM0)
    with pytest.raises(ValueError):          # beyond the mirrors
        m.MPC(0.5, 0.9, ar, 4, regions=[(-0.5, 0.1, he)], wavelength=LAM0)
