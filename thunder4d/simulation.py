"""Driver: propagates a Pulse through an MPC and records reduced diagnostics after every pass
(cell centre plane). Full 3D fields are only written to disk for the passes you ask for."""
import os
import pickle
import time
import numpy as np
from . import diagnostics as dg
from .backend import to_host
from .solver import UPPESolver
from .pulse import Pulse


class MPCSimulation:
    """
    sim = MPCSimulation(pulse, cell, phi_max=0.02)
    sim.run()                      # or sim.run(n_passes=5) to continue step by step
    sim.history["eta00"]           # per pass, etc.  sim.save("run.npz")

    record_every   : record diagnostics every n passes (pass 0 = input is always recorded)
    far_field      : also record the angle-resolved spectrogram and far-field homogeneity
                     (per-colour matrix Fourier transforms: costs some seconds per record)
    mode_reference : 'linear' -> HG basis of the linear eigenmode w0(lambda);
                     or a float w0 at lambda0 (e.g. the matched waist), scaled as sqrt(lambda)
    save_fields    : iterable of pass numbers whose full field is written to save_dir as .npy
    record_mirror  : also keep the fluence at the mirror plane (before bounce)
    profile_wavelengths : wavelengths [m] whose 1D profiles (bandpass `profile_bandwidth`, 0 = one
                     spectral bin) are recorded every pass, together with the global profile:
                     near field (cell centre, cut + projection), far field (angle, cut), mirror (cut)
    profile_axis   : 'x' or 'y'

    history keys include wx, wy (second-moment radius per colour, on sim.lam) and w_peak_x,
    w_peak_y (the same radius [m] at the spectral peak of the input pulse, one value per pass).
    """

    def __init__(self, pulse, cell, *, solver=None, record_every=1, far_field=True, max_order=6,
                 mode_reference="linear", save_fields=(), save_dir=".", record_mirror=True,
                 slit=None, profile_wavelengths=(), profile_bandwidth=0.0, profile_axis="x",
                 **solver_kw):
        self.grid = pulse.grid
        self.cell = cell
        self.Aw = self.grid.asarray(pulse.Aw).copy()      # lives on the grid's device
        self.solver = solver or UPPESolver(self.grid, cell.medium, **solver_kw)
        self.record_every, self.far, self.max_order = record_every, far_field, max_order
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
        self.passes_done = 0
        self.history = {k: [] for k in (
            "pass", "energy", "B", "peak_power", "spectrum", "fluence", "fluence_mirror",
            "sgram_x", "sgram_theta", "eta00", "eta00_total", "order_frac", "V_near", "V_far",
            "wx", "wy", "w_peak_x", "w_peak_y", "M2x", "M2y", "M2x_mean", "M2y_mean", "wall_s",
            "prof_near", "prof_near_proj", "prof_far", "prof_mirror", "band_energy")}
        self._t0 = time.time()
        self._peak_lam = None   # colour tracked by w_peak_*: spectral peak of the input pulse
        self.record(0)

    # ------------------------------------------------------------ recording
    def _mirror_profiles(self):
        _, pr, _ = dg.profiles_1d(self.grid, self.Aw, self.prof_lams, self.prof_bw, self.prof_axis)
        return pr.astype(np.float32)

    def record(self, p, mirror_fluence=None, mirror_profiles=None):
        g, h = self.grid, self.history
        Ad = self.Aw                  # device array (GPU or CPU)
        A = to_host(Ad)               # one host copy per record, shared by all diagnostics
        if self.prof_lams:
            xc, pr, en = dg.profiles_1d(g, A, self.prof_lams, self.prof_bw, self.prof_axis, "cut")
            _, prj, _ = dg.profiles_1d(g, A, self.prof_lams, self.prof_bw, self.prof_axis, "integrated")
            h["prof_near"].append(pr.astype(np.float32))
            h["prof_near_proj"].append(prj.astype(np.float32))
            h["band_energy"].append(en)
            h["prof_mirror"].append(mirror_profiles)
            if self.far:
                th, pf = dg.far_profiles_1d(g, A, self.prof_lams, self.prof_bw)
                h["prof_far"].append(pf.astype(np.float32))
            self.prof_coord = xc
        else:
            for k in ("prof_near", "prof_near_proj", "band_energy", "prof_mirror", "prof_far"):
                h[k].append(None)
        h["pass"].append(p)
        h["energy"].append(g.energy(Ad))
        h["B"].append(self.solver.B)
        h["peak_power"].append(Pulse(g, Ad).peak_power())
        lam, S = dg.total_spectrum(g, A)
        h["spectrum"].append(S)
        h["fluence"].append(dg.fluence(g, A).astype(np.float32))
        h["fluence_mirror"].append(mirror_fluence)
        x, lam_s, Sx = dg.spectrogram_x_lambda(g, A, slit=self.slit)
        h["sgram_x"].append(Sx)
        mc = dg.mode_content(g, A, self.w0_of_lambda, self.max_order)
        h["eta00"].append(np.interp(lam, mc["lam"], mc["eta00"], left=np.nan, right=np.nan))
        h["eta00_total"].append(mc["eta00_total"])
        h["order_frac"].append(mc["order_frac"])
        h["V_near"].append(dg.homogeneity(g, A)[1])
        bm = dg.beam_moments(g, A)
        for k in ("wx", "wy", "M2x", "M2y"):
            h[k].append(np.interp(lam, bm["lam"], bm[k], left=np.nan, right=np.nan))
        if self._peak_lam is None:
            self._peak_lam = lam[np.nanargmax(S)]
        for k in ("wx", "wy"):
            h["w_peak_" + k[1]].append(float(np.interp(self._peak_lam, bm["lam"], bm[k])))
        h["M2x_mean"].append(bm["M2x_mean"])
        h["M2y_mean"].append(bm["M2y_mean"])
        if self.far:
            th, _, St = dg.spectrogram_theta_lambda(g, A)
            h["sgram_theta"].append(St)
            _, Ef = dg.far_field(g, A, n_theta=min(g.Nx // 2 * 2 + 1, 129))
            h["V_far"].append(dg.homogeneity(g, S_xyw=np.abs(Ef) ** 2)[1])
            self.theta = th
        else:
            h["sgram_theta"].append(None)
            h["V_far"].append(np.nan)
        h["wall_s"].append(time.time() - self._t0)
        self.lam, self.x = lam, x
        if p in self.save_fields:
            np.save(os.path.join(self.save_dir, f"field_pass{p:03d}.npy"), A)

    # ------------------------------------------------------------ run
    def run(self, n_passes=None, verbose=True, max_seconds=None):
        """Run n_passes (default: until cell.n_passes). max_seconds: stop cleanly after the pass
        that exceeds this wall-clock budget (combine with checkpoint/resume)."""
        n = n_passes or (self.cell.n_passes - self.passes_done)
        d = self.cell.d
        t_start = time.time()
        for _ in range(n):
            if max_seconds is not None and time.time() - t_start > max_seconds:
                break
            p = self.passes_done + 1
            self.Aw = self.solver.propagate(self.Aw, d / 2)
            Fm = dg.fluence(self.grid, self.Aw).astype(np.float32) if self.record_mirror else None
            Pm = self._mirror_profiles() if (self.record_mirror and self.prof_lams) else None
            self.Aw = self.cell.bounce(self.Aw, self.grid)
            self.Aw = self.solver.absorb(self.Aw)
            self.Aw = self.solver.propagate(self.Aw, d / 2)
            self.Aw = self.solver.absorb(self.Aw, temporal=True)
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

    @property
    def done(self):
        return self.passes_done >= self.cell.n_passes

    # ------------------------------------------------------------ checkpoints (long runs)
    def checkpoint(self, path):
        """Save the full state so a long run can be resumed (sim.resume(path))."""
        st = {"Aw": to_host(self.Aw), "history": self.history, "passes_done": self.passes_done,
              "B": self.solver.B, "nsteps": self.solver.nsteps, "z": self.solver.z,
              "Ipeak": self.solver._Ipeak}
        with open(path, "wb") as f:
            pickle.dump(st, f, protocol=pickle.HIGHEST_PROTOCOL)

    def resume(self, path):
        with open(path, "rb") as f:
            st = pickle.load(f)
        self.Aw = self.grid.asarray(st["Aw"])
        self.history = st["history"]
        self.passes_done = st["passes_done"]
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
        out = {"lam": self.lam, "x": self.x, "t": self.grid.ts,
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
        fm = [v for v in h["fluence_mirror"] if v is not None]
        if fm:
            out["fluence_mirror"] = np.array(fm)
        if include_field:
            out["Aw_final"] = to_host(self.Aw)
        np.savez_compressed(path, **out)
