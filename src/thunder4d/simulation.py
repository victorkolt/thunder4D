"""Driver: propagates a Pulse through an MPC and records reduced diagnostics after every pass.

Pass convention (same as the 1+1D code of the design helper): the simulation starts right after a
mirror (injection plane) and one pass = transit over d + bounce on the next mirror, so N passes =
N transits and N bounces, and the simulation ends right after the last bounce. The input Pulse is
given at the cell centre, as built by Pulse.at_focus & co.; it is moved to the injection plane by
a linear half pass. Diagnostics are taken at the cell centre (waist plane, where the modal
decomposition on the cell eigenmodes is exact): after every pass, the field is imaged back to the
centre by a linear half pass, i.e. the far field of the beam that leaves the cell after that pass.
Full 3D fields are only written to disk for the passes you ask for.

Cells with plates / gas regions (bulk / hybrid MPC): one solver per medium (gases + plate materials),
seen by the driver as one (_SolverSet: B, nsteps, z summed). A pass walks the segments of
cell.layout(pass): face transfer (Fresnel / coating), propagation in the plate (its own dispersion,
group-velocity frame and n2), face transfer, and the filling medium between. The "centre" is the
centre of the equivalent cell (MPC.centre_z), cut out of a plate if it falls inside one; the linear
half passes to/from the centre walk the same segments (faces inverted backwards)."""

import os
import pickle
import time
from typing import Any

import numpy as np

from . import diagnostics as dg
from .backend import to_host
from .pulse import Pulse
from .solver import UPPESolver


class _SolverSet:
    """Solvers of a cell with plates, one per medium, seen as one solver by the driver: B, nsteps,
    z are summed; everything else (absorb, peak_intensity, mirror, s_m, ...) is the main solver's
    (filling medium)."""

    def __init__(self, main, by_medium):
        self.main, self.by_medium = main, by_medium        # {id(medium): solver}
        self.all = [main] + [s for s in by_medium.values() if s is not main]

    def for_medium(self, medium):
        return self.by_medium[id(medium)]

    def __getattr__(self, name):
        return getattr(self.main, name)

    def _sum(self, attr):
        return sum(getattr(s, attr) for s in self.all)

    def _set(self, attr, value):
        setattr(self.main, attr, value - sum(getattr(s, attr) for s in self.all[1:]))

    B = property(lambda self: self._sum("B"), lambda self, v: self._set("B", v))
    nsteps = property(lambda self: self._sum("nsteps"), lambda self, v: self._set("nsteps", v))
    z = property(lambda self: self._sum("z"), lambda self, v: self._set("z", v))

    @property
    def _Ipeak(self):
        return self.main._Ipeak

    @_Ipeak.setter
    def _Ipeak(self, value):
        for s in self.all:
            s._Ipeak = value


def make_solvers(grid, cell, cls, kw):
    """The solver of the cell (cls = UPPESolver or LensFrameSolver), or with plates / gas regions a
    _SolverSet with one solver per medium (main: the reference medium, at the mirrors)."""
    if not cell.segmented:
        return cls(grid, cell.medium, **kw) if cls is UPPESolver else cls(grid, cell, **kw)
    by = {}
    for seg in cell.layout(1, grid.lambda0):
        med = seg["medium"]
        if id(med) not in by:
            by[id(med)] = cls(grid, med, **kw) if cls is UPPESolver else cls(grid, cell, medium=med, **kw)
    return _SolverSet(by[id(cell.ref_medium)], by)


class Walker:
    """Walks the segments of a pass of a cell with plates / gas regions: face transfers (Fresnel or
    coatings, with the gas on each side), and propagation in each medium with its solver, in
    physical z (lens=False) or in zeta of the equivalent cell (lens=True). Positions are REDUCED
    coordinates u (0 at the mirror just left, d_eq at the next one); walking backwards (ub < ua)
    is linear, with inverted face transfers."""

    def __init__(self, grid, cell, solvers, lens=False):
        self.grid, self.cell, self.solvers, self.lens = grid, cell, solvers, lens
        self.ueq = cell.equivalent_separation(grid.lambda0)
        self.uc = self.ueq / 2
        self._faces = {}

    def face(self, Aw, plate, face, ambient, inverse=False):
        key = (id(plate), face, id(ambient))
        if key not in self._faces:
            g = self.grid
            lam = np.where(g.valid, g.lam, g.lambda0)
            t = np.where(g.valid, plate.face_transfer(face, ambient, lam, g.lambda0), 1.0)
            self._faces[key] = (g.asarray(t.astype(g.dtype)[None, None, :]),
                                g.asarray((1 / t).astype(g.dtype)[None, None, :]))
        Aw *= self._faces[key][1 if inverse else 0]
        return Aw

    def walk(self, Aw, p, ua, ub, nonlinear):
        segs = self.cell.layout(p, self.grid.lambda0, split_at=(ua, ub))
        eps = 1e-12 * self.cell.d
        fwd = ub >= ua
        lo, hi = (ua, ub) if fwd else (ub, ua)
        sel = [s for s in segs if s["u0"] >= lo - eps and s["u1"] <= hi + eps]
        for s in (sel if fwd else sel[::-1]):
            if fwd:
                first, a_first, last, a_last = s["entry_face"], s.get("amb_in"), s["exit_face"], s.get("amb_out")
            else:
                first, a_first, last, a_last = s["exit_face"], s.get("amb_out"), s["entry_face"], s.get("amb_in")
            if first:
                Aw = self.face(Aw, s["plate"], first, a_first, inverse=not fwd)
            Aw = self.segment(Aw, s, fwd, nonlinear)
            if last:
                Aw = self.face(Aw, s["plate"], last, a_last, inverse=not fwd)
        return Aw

    def segment(self, Aw, seg, forward, nonlinear):
        S = self.solvers.for_medium(seg["medium"])
        if self.lens:
            zr = S.zR
            ua, ub = (seg["u0"], seg["u1"]) if forward else (seg["u1"], seg["u0"])
            za, zb = zr * np.arctan((ua - self.uc) / zr), zr * np.arctan((ub - self.uc) / zr)
            if nonlinear:
                S._Ipeak = None            # new segment: fresh peak intensity for the step size
                return S.propagate_zeta(Aw, za, zb)
            return S.propagate_linear_zeta(Aw, za, zb)
        L = seg["z1"] - seg["z0"]
        if nonlinear:
            S._Ipeak = None
            return S.propagate(Aw, L)
        return S.linear(Aw, L if forward else -L)


class MPCSimulation:
    """
    sim = MPCSimulation(pulse, cell, phi_max=0.02)
    sim.run()                      # or sim.run(n_passes=5) to continue step by step
    sim.history["eta00"]           # per pass, etc.  sim.save("run.npz")

    record_every   : record diagnostics every n passes (pass 0 = input is always recorded)
    near_field     : also record the angle-resolved spectrogram, profiles and homogeneity of the
                     collimated beam (near field)
                     (per-colour matrix Fourier transforms: costs some seconds per record)
    mode_reference : 'linear' -> HG basis of the linear eigenmode w0(lambda);
                     or a float w0 at lambda0 (e.g. the matched waist), scaled as sqrt(lambda)
    save_fields    : iterable of pass numbers whose full field is written to save_dir as .npy
    record_mirror  : also keep the fluence at the mirror plane (before bounce)
    profile_wavelengths : wavelengths [m] whose 1D profiles (bandpass `profile_bandwidth`, 0 = one
                     spectral bin) are recorded every pass, together with the global profile:
                     far field (cell centre, cut + projection), near field (collimated beam: angle,
                     cut), mirror (cut)
    profile_axis   : 'x' or 'y'

    Conventions: a pass = transit over d + bounce, starting right after a mirror (see the module
    docstring). sim.Aw is the field after the last pass seen at the cell centre (linear half pass);
    the propagated state itself (right after the last bounce) is sim.state.
    FAR FIELD = the cell centre (focus / waist plane, where the diagnostics are taken);
    NEAR FIELD = the collimated beam, i.e. the angular spectrum of the cell-centre field
    (x_out = f * theta after a collimating lens of focal length f).

    history keys include wx, wy (second-moment radius per colour, on sim.lam), w_peak_x,
    w_peak_y (the same radius [m] at the spectral peak of the input pulse, one value per pass) and
    peak_intensity (max of |E(x, y, t)|^2 at the cell centre [W/m^2], one value per record).
    """

    _lens = False      # segments propagated in physical z (True: in the lens frame)

    def __init__(self, pulse, cell, *, solver=None, record_every=1, near_field=True, max_order=6,
                 mode_reference="linear", save_fields=(), save_dir=".", record_mirror=True,
                 slit=None, profile_wavelengths=(), profile_bandwidth=0.0, profile_axis="x",
                 **solver_kw):
        self.grid = pulse.grid
        self.cell = cell
        self.passes_done = 0
        self.solver = solver or self._make_solvers(UPPESolver, solver_kw)
        if cell.segmented:
            self._walker = Walker(self.grid, cell, self.solver, lens=self._lens)
            self._uc, self._ueq = self._walker.uc, self._walker.ueq
        centre = self.grid.asarray(pulse.Aw).copy()       # lives on the grid's device
        self.state = self._from_centre(centre.copy())     # right after a mirror (injection plane)
        self._centre = (0, centre)                        # (passes, field seen at the centre)
        self.record_every, self.near, self.max_order = record_every, near_field, max_order
        self.save_fields, self.save_dir = set(save_fields), save_dir
        self.record_mirror, self.slit = record_mirror, slit
        self.prof_lams = list(profile_wavelengths)
        self.prof_bw, self.prof_axis = profile_bandwidth, profile_axis
        lam0 = self.grid.lambda0
        if mode_reference == "linear":
            self.w0_of_lambda = lambda lam: cell.linear_mode(lam)["w0"]
        else:
            w_ref = float(mode_reference)
            self.w0_of_lambda = lambda lam: w_ref * np.sqrt(lam / lam0)
        self.history = {k: [] for k in (
            "pass", "energy", "B", "peak_power", "peak_intensity", "spectrum", "fluence", "fluence_mirror",
            "sgram_x", "sgram_theta", "eta00", "eta00_total", "order_frac", "V_near", "V_far",
            "wx", "wy", "w_peak_x", "w_peak_y", "M2x", "M2y", "M2x_mean", "M2y_mean", "wall_s",
            "prof_far", "prof_far_proj", "prof_near", "prof_mirror", "band_energy")}
        self._t0 = time.time()
        self._peak_lam = None   # colour tracked by w_peak_*: spectral peak of the input pulse
        self.record(0)

    def _mirror_profiles(self, A):
        _, pr, _ = dg.profiles_1d(self.grid, A, self.prof_lams, self.prof_bw, self.prof_axis)
        return pr.astype(np.float32)

    def record(self, p, mirror_fluence=None, mirror_profiles=None):
        g, h = self.grid, self.history
        Ad = self.Aw                  # device array (GPU or CPU)
        A = to_host(Ad)               # one host copy per record, shared by all diagnostics
        if self.prof_lams:
            xc, pr, en = dg.profiles_1d(g, A, self.prof_lams, self.prof_bw, self.prof_axis, "cut")
            _, prj, _ = dg.profiles_1d(g, A, self.prof_lams, self.prof_bw, self.prof_axis, "integrated")
            h["prof_far"].append(pr.astype(np.float32))            # cell centre = far field
            h["prof_far_proj"].append(prj.astype(np.float32))
            h["band_energy"].append(en)
            h["prof_mirror"].append(mirror_profiles)
            if self.near:
                _, pf = dg.near_profiles_1d(g, A, self.prof_lams, self.prof_bw)
                h["prof_near"].append(pf.astype(np.float32))       # collimated beam = near field
            self.prof_coord = xc
        else:
            for k in ("prof_far", "prof_far_proj", "band_energy", "prof_mirror", "prof_near"):
                h[k].append(None)
        h["pass"].append(p)
        h["energy"].append(g.energy(Ad))
        h["B"].append(self.solver.B)
        h["peak_power"].append(Pulse(g, Ad).peak_power())
        h["peak_intensity"].append(self.solver.peak_intensity(Ad))     # W/m^2, at the cell centre
        lam, S = dg.total_spectrum(g, A)
        h["spectrum"].append(S)
        h["fluence"].append(dg.fluence(g, A).astype(np.float32))
        h["fluence_mirror"].append(mirror_fluence)
        x, _, Sx = dg.spectrogram_x_lambda(g, A, slit=self.slit)
        h["sgram_x"].append(Sx)
        mc = dg.mode_content(g, A, self.w0_of_lambda, self.max_order)
        h["eta00"].append(np.interp(lam, mc["lam"], mc["eta00"], left=np.nan, right=np.nan))
        h["eta00_total"].append(mc["eta00_total"])
        h["order_frac"].append(mc["order_frac"])
        h["V_far"].append(dg.homogeneity(g, A)[1])            # cell centre = far field
        bm = dg.beam_moments(g, A)
        for k in ("wx", "wy", "M2x", "M2y"):
            h[k].append(np.interp(lam, bm["lam"], bm[k], left=np.nan, right=np.nan))
        if self._peak_lam is None:
            self._peak_lam = lam[np.nanargmax(S)]
        for k in ("wx", "wy"):
            h["w_peak_" + k[1]].append(float(np.interp(self._peak_lam, bm["lam"], bm[k])))
        h["M2x_mean"].append(bm["M2x_mean"])
        h["M2y_mean"].append(bm["M2y_mean"])
        if self.near:
            th, _, St = dg.spectrogram_theta_lambda(g, A)
            h["sgram_theta"].append(St)
            _, Ef = dg.near_field(g, A, n_theta=min(g.Nx // 2 * 2 + 1, 129))
            h["V_near"].append(dg.homogeneity(g, S_xyw=np.abs(Ef) ** 2)[1])    # collimated beam = near field
            self.theta = th
        else:
            h["sgram_theta"].append(None)
            h["V_near"].append(np.nan)
        h["wall_s"].append(time.time() - self._t0)
        self.lam, self.x = lam, x
        if p in self.save_fields:
            np.save(os.path.join(self.save_dir, f"field_pass{p:03d}.npy"), A)

    # ------------------------------------------------------------ field seen at the centre
    @property
    def Aw(self):
        """Field after the last pass, imaged back to the cell centre (linear half pass)."""
        if self._centre[0] != self.passes_done:
            self._centre = (self.passes_done, self._centre_view(self.state))
        return self._centre[1]

    @Aw.setter
    def Aw(self, value):
        centre = self.grid.asarray(value)
        self.state = self._from_centre(centre.copy())
        self._centre = (self.passes_done, centre)

    # ------------------------------------------------------------ run
    def run(self, n_passes=None, verbose=True, max_seconds=None):
        """Run n_passes (default: until cell.n_passes). max_seconds: stop cleanly after the pass
        that exceeds this wall-clock budget (combine with checkpoint/resume)."""
        n = n_passes or (self.cell.n_passes - self.passes_done)
        t_start = time.time()
        for _ in range(n):
            if max_seconds is not None and time.time() - t_start > max_seconds:
                break
            p = self.passes_done + 1
            A = self._half_pass(self.state, p, 0)
            A = self.solver.absorb(A)
            A = self._half_pass(A, p, 1)
            Fm = dg.fluence(self.grid, A).astype(np.float32) if self.record_mirror else None
            Pm = self._mirror_profiles(A) if (self.record_mirror and self.prof_lams) else None
            A = self._bounce(A)
            self.state = self.solver.absorb(A, temporal=True)
            self.passes_done = p
            if p % self.record_every == 0 or p == self.cell.n_passes:
                self.record(p, Fm, Pm)
                if verbose:
                    h = self.history
                    print(f"pass {p:3d}/{self.cell.n_passes} | B = {self.solver.B:6.2f} rad | "
                          f"E = {h['energy'][-1]*1e3:.4g} mJ | eta00 = {h['eta00_total'][-1]:.4f} | "
                          f"V_near = {h['V_near'][-1]:.4f} | V_far = {h['V_far'][-1]:.4f} | "
                          f"M2 = {h['M2x_mean'][-1]:.3f}/{h['M2y_mean'][-1]:.3f} | "
                          f"steps {self.solver.nsteps} | {h['wall_s'][-1]:.0f} s", flush=True)
        return self

    # ------------------------------------------------------------ cells with plates / gas regions
    def _make_solvers(self, cls, kw):
        return make_solvers(self.grid, self.cell, cls, kw)

    def _walk(self, Aw, p, ua, ub, nonlinear):
        return self._walker.walk(Aw, p, ua, ub, nonlinear)

    def _half_pass(self, Aw, p, half):
        """half 0: mirror -> centre, half 1: centre -> mirror (nonlinear)."""
        if not self.cell.segmented:
            return self._to_centre(Aw) if half == 0 else self._to_mirror(Aw)
        return self._walk(Aw, p, 0.0, self._uc, True) if half == 0 else self._walk(Aw, p, self._uc, self._ueq, True)

    # one pass = mirror -> centre, centre -> mirror, bounce (LensFrameSimulation overrides these),
    # plus the two linear half passes between the injection plane and the centre
    def _to_centre(self, Aw):
        return self.solver.propagate(Aw, self.cell.d / 2)

    def _to_mirror(self, Aw):
        return self.solver.propagate(Aw, self.cell.d / 2)

    def _bounce(self, Aw):
        return self.cell.bounce(Aw, self.grid)

    def _from_centre(self, Aw):
        """Cell centre -> plane right after a mirror (linear, backwards)."""
        if self.cell.segmented:
            return self._walk(Aw, self.passes_done + 1, self._uc, 0.0, False)
        return self.solver.linear(Aw, -self.cell.d / 2)

    def _centre_view(self, Aw):
        """Plane right after a mirror -> cell centre (linear, on a copy)."""
        if self.cell.segmented:
            return self._walk(Aw.copy(), self.passes_done + 1, 0.0, self._uc, False)
        return self.solver.linear(Aw.copy(), self.cell.d / 2)

    @property
    def done(self):
        return self.passes_done >= self.cell.n_passes

    # ------------------------------------------------------------ checkpoints (long runs)
    def checkpoint(self, path):
        """Save the full state so a long run can be resumed (sim.resume(path))."""
        st = {"state": to_host(self.state), "history": self.history, "passes_done": self.passes_done,
              "B": self.solver.B, "nsteps": self.solver.nsteps, "z": self.solver.z,
              "Ipeak": self.solver._Ipeak}
        if hasattr(self.solver, "all"):   # one solver per medium: keep each one's counters
            st["per_medium"] = [(s.B, s.nsteps, s.z) for s in self.solver.all]
        with open(path, "wb") as f:
            pickle.dump(st, f, protocol=pickle.HIGHEST_PROTOCOL)

    def resume(self, path):
        with open(path, "rb") as f:
            st = pickle.load(f)
        history = st["history"]
        for key in self.history:          # keys added to the history after the checkpoint was written
            history.setdefault(key, [None] * len(history["pass"]))
        self.history = history
        self.passes_done = st["passes_done"]
        if history["wall_s"]:             # wall time keeps counting from the checkpoint
            self._t0 = time.time() - history["wall_s"][-1]
        if "state" in st:
            self.state = self.grid.asarray(st["state"])
            self._centre = (-1, None)
        else:                     # checkpoint written before the injection-plane convention
            self.Aw = st["Aw"]
        per = st.get("per_medium")
        if hasattr(self.solver, "all") and per is not None and len(per) == len(self.solver.all):
            for s, (b, n, z) in zip(self.solver.all, per):
                s.B, s.nsteps, s.z = b, n, z
        else:                     # single solver, or a checkpoint without the split (all in the main solver)
            self.solver.B, self.solver.nsteps, self.solver.z = st["B"], st["nsteps"], st["z"]
        self.solver._Ipeak = st["Ipeak"]
        return self

    @property
    def pulse(self):
        return Pulse(self.grid, self.Aw)

    # ------------------------------------------------------------ I/O
    def save(self, path, include_field=True):
        """Everything in one .npz (reduced diagnostics + final field)."""
        h = self.history
        out: dict[str, Any] = {"lam": self.lam, "x": self.x, "t": self.grid.ts,
               "theta": getattr(self, "theta", np.array([])),
               "R": self.cell.R, "d": self.cell.d, "n_passes": self.cell.n_passes,
               "gas": self.cell.medium.name, "pressure_bar": self.cell.medium.p,
               "n2": self.cell.medium.n2, "lambda0": self.grid.lambda0,
               "grid": np.array([self.grid.Nx, self.grid.Ny, self.grid.Nt, self.grid.dx,
                                 self.grid.dy, self.grid.dt])}
        for k, v in h.items():
            if len(v) and v[0] is not None and k not in ("fluence_mirror", "prof_mirror"):
                out[k] = np.array(v)
        pm = [v for v in h["prof_mirror"] if v is not None]
        if pm:
            out["prof_mirror"] = np.array(pm)       # passes 1..N (before each bounce)
        if self.prof_lams:
            out["prof_wavelengths"] = np.array(self.prof_lams)
            out["prof_bandwidth"] = self.prof_bw
            out["prof_coord"] = self.prof_coord
        if self.cell.regions:
            out["regions"] = np.array([f"[{z0}, {z1}] m | {med}" for z0, z1, med in self.cell.regions])
        if self.cell.plates:
            out["plates"] = np.array([f"{p.name} | {p.material} | z = {p.position} m | "
                                      f"{p.coating_A.name}/{p.coating_B.name}" for p in self.cell.plates])
        fm = [v for v in h["fluence_mirror"] if v is not None]
        if fm:
            out["fluence_mirror"] = np.array(fm)
            # mirror records of a beam-following grid are in frame coordinates: x_phys = mirror_scale * x,
            # fluence_phys = fluence / mirror_scale^2 (1 on a fixed grid)
            out["mirror_scale"] = float(getattr(self, "mirror_scale", 1.0))
        if include_field:
            out["Aw_final"] = to_host(self.Aw)          # final field seen at the cell centre
        np.savez_compressed(path, **out)
