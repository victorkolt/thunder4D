"""Module 2: the multipass cell.

Unfolded picture used by the solver: the simulation starts right after a mirror (injection plane)
and one pass = transit over d (through the cell centre) + bounce on the next mirror, so N passes =
N transits = N bounces. Diagnostics are taken at the cell centre (waist plane of the eigenmode,
flat phase), where the modal decomposition on the cell eigenmodes is exact.

Nonlinear mode matching (aberration-free Kerr lens, uniform medium, power P constant along a pass):
the envelope equation of a gaussian becomes the linear one with lambda -> lambda sqrt(1 - P/Pc),
Pc = lambda^2/(2 pi n0 n2). Hence the matched waist and mirror spot are the linear ones scaled by
(1 - P/Pc)^(1/4), and the B-integral per pass (on axis, peak) is
    B = 8 pi n2 P / (lambda lambda_eff) * atan(d / 2 zR')      (-> 2 theta P/Pc for P << Pc).
A pulse can only be matched at one power: the wings of the pulse see the linear mode. This
time-dependent mismatch is precisely one of the couplings the 4D code resolves.
"""
import math
from itertools import pairwise
from typing import Any

import numpy as np
from scipy.constants import c as C0, femto as FS
from scipy.optimize import fsolve

from .materials import vacuum
from .plates import Plate


def lg_critical_factor(l):
    """Pc of an LG_0,l beam / Pc of a gaussian (variance method): (l+1) 4^l (l!)^2 / (2l)!."""
    a = abs(int(l))
    return (a + 1) * 4 ** a * math.factorial(a) ** 2 / math.factorial(2 * a)


def lg_peak_factor(l):
    """Peak intensity of an LG_0,l beam / on-axis intensity of the gaussian of the same w and
    power: l^l e^-l / l! (1 for l = 0)."""
    a = abs(int(l))
    return 1.0 if a == 0 else a ** a * np.exp(-a) / math.factorial(a)


class MPC:
    """Symmetric two-mirror cell.

    R            : mirror radius of curvature [m]
    d            : mirror separation [m]
    medium       : Material
    n_passes     : number of passes (transit over d + bounce = 1 pass; starts right after a mirror)
    mirror_gdd_fs2, mirror_tod_fs3 : dispersion per bounce
    mirror_reflectivity : scalar, or callable R(lambda[m]) (intensity)
    mirror_phase : optional callable phi(lambda[m]) [rad] per bounce (measured GDD curves, ...)
    mirror_diameter : clear aperture [m] (soft super-gaussian edge), None = infinite
    plates       : Plate(s) crossed once per pass (bulk / hybrid MPC), see thunder4d.plates
    regions      : gas regions [(z0, z1, Material), ...] (pass-1 coordinates: centre 0, + towards
                   mirror B) filled with another gas than `medium`, e.g. a vacuum cell between two
                   window plates in air: regions=[(-0.14, 0.14, vacuum())]
    wavelength   : reference wavelength [m] of a cell with plates / regions (equivalent cell, Gouy)
    medium       : filling Material outside the regions; None = vacuum
    """

    def __init__(self, R, d, medium, n_passes, *, mirror_gdd_fs2=0.0, mirror_tod_fs3=0.0,
                 mirror_reflectivity=1.0, mirror_phase=None, mirror_diameter=None, plates=(),
                 regions=(), wavelength=None):
        self.R, self.d, self.n_passes = float(R), float(d), int(n_passes)
        self.medium = vacuum() if medium is None else medium
        self.plates = sorted([plates] if isinstance(plates, Plate) else list(plates), key=lambda p: p.position)
        self.regions = sorted([(float(z0), float(z1), m) for z0, z1, m in regions], key=lambda r: r[0])
        self.wavelength = None if wavelength is None else float(wavelength)
        if self.segmented and self.wavelength is None:
            raise ValueError("a cell with plates or gas regions needs its reference wavelength: "
                             "MPC(..., wavelength=lam0)")
        self.gdd, self.tod = mirror_gdd_fs2, mirror_tod_fs3
        self.refl, self.mphase = mirror_reflectivity, mirror_phase
        self.diameter = mirror_diameter
        self._check_layout()
        de = self.equivalent_separation()
        if not 0 < de < 2 * R:
            raise ValueError("unstable cell: need 0 < d < 2R"
                             + (f" (equivalent separation {de:.4f} m)" if self.segmented else ""))
        self.g = 1 - de / R
        self.theta = np.arccos(self.g)  # Gouy phase per pass
        self._bounce_cache = None

    # ------------------------------------------------------------ plates, regions and layout
    @property
    def segmented(self):
        """True if the pass is cut into segments (plates and/or gas regions)."""
        return bool(self.plates or self.regions)

    @property
    def ref_medium(self):
        """Reference medium of the reduced coordinate: the medium at the mirrors."""
        return self._medium_at(self.d / 2) if self.regions else self.medium

    def _medium_at(self, z):
        """Gas at physical position z (pass-1 coordinates, centre 0, + towards mirror B)."""
        for z0, z1, m in self.regions:
            if z0 <= z <= z1:
                return m
        return self.medium

    def equivalent_separation(self, lam=None):
        """Separation [m] of the empty cell (filled with the reference medium) with the same
        diffraction: the sum of the reduced lengths of the segments (d without plates/regions)."""
        if not self.segmented:
            return self.d
        return self.layout(1, lam)[-1]["u1"]

    def _check_layout(self):
        half = self.d / 2
        spans = [(p.position - p.t / 2, p.position + p.t / 2, p.name) for p in self.plates]
        for lo, hi, name in spans:
            if lo <= -half or hi >= half:
                raise ValueError(f"{name} at {(lo + hi) / 2 * 1e3:+.1f} mm does not fit between the mirrors "
                                 f"(d/2 = {half*1e3:.1f} mm)")
        for (_, hi1, n1), (lo2, _, n2) in pairwise(spans):
            if lo2 < hi1:
                raise ValueError(f"plates {n1} and {n2} overlap")
        for z0, z1, _ in self.regions:
            if not -half <= z0 < z1 <= half:
                raise ValueError(f"gas region [{z0}, {z1}] m must lie between the mirrors (|z| <= d/2)")
        for (_, b1, _), (a2, _, _) in pairwise(self.regions):
            if a2 < b1:
                raise ValueError("gas regions overlap")
        if self.regions and self._medium_at(-half) is not self._medium_at(half):
            raise ValueError("the two mirrors must be in the same medium (reference of the equivalent cell)")

    def layout(self, pass_index=1, lam=None, split_at=(), split_at_centre=False):
        """Segments of one pass, from the mirror just left (z = -d/2, u = 0) to the next one
        (z = +d/2, u = d_eq). Odd passes go A -> B, even passes B -> A (plates and regions mirrored).
        Each segment: {'kind': 'gas'|'plate', 'z0', 'z1' (physical, in the coordinates of this pass),
        'u0', 'u1' (reduced: length x n_ref/n), 'medium', 'plate', 'entry_face', 'exit_face' ('A'/'B'
        at the plate surfaces, None elsewhere), 'amb_in', 'amb_out' (gas on each side of a plate)}.
        split_at: reduced positions where to cut segments (split_at_centre adds d_eq/2)."""
        lam = self.wavelength if lam is None else lam
        sgn = 1 if pass_index % 2 else -1
        half = self.d / 2
        f_in, f_out = ("A", "B") if sgn > 0 else ("B", "A")
        cuts = {-half, half}
        for z0, z1, _ in self.regions:
            cuts |= {sgn * z0, sgn * z1}
        plates = [(sgn * p.position - p.t / 2, sgn * p.position + p.t / 2, p) for p in self.plates]
        for lo, hi, _ in plates:
            cuts |= {lo, hi}
        cuts = sorted(c for c in cuts if -half <= c <= half)
        segs: list[dict[str, Any]] = []
        for z0, z1 in pairwise(cuts):
            if z1 - z0 <= 1e-15:
                continue
            zm = (z0 + z1) / 2
            plate = next((p for lo, hi, p in plates if lo <= zm <= hi), None)
            if plate is not None:
                seg = {"kind": "plate", "medium": plate.material, "plate": plate}
            else:
                seg = {"kind": "gas", "medium": self._medium_at(sgn * zm), "plate": None}
            prev = segs[-1] if segs else None
            if prev is not None and prev["medium"] is seg["medium"] and prev["plate"] is seg["plate"]:
                prev["z1"] = z1
                continue
            segs.append({**seg, "z0": z0, "z1": z1, "entry_face": None, "exit_face": None})
        nref = float(self.ref_medium.n_lam(lam))
        u = 0.0
        for i, s in enumerate(segs):
            if s["plate"] is not None:
                s["entry_face"], s["exit_face"] = f_in, f_out
                s["amb_in"] = segs[i - 1]["medium"]
                s["amb_out"] = segs[i + 1]["medium"]
            du = (s["z1"] - s["z0"]) * nref / float(s["medium"].n_lam(lam))
            s["u0"], s["u1"] = u, u + du
            u += du
        if split_at_centre:
            split_at = (*split_at, u / 2)
        for uc in split_at:
            out = []
            for s in segs:
                if s["u0"] + 1e-15 < uc < s["u1"] - 1e-15:
                    a = (uc - s["u0"]) / (s["u1"] - s["u0"])
                    zc = s["z0"] + a * (s["z1"] - s["z0"])
                    out.append({**s, "z1": zc, "u1": uc, "exit_face": None})
                    out.append({**s, "z0": zc, "u0": uc, "entry_face": None})
                else:
                    out.append(s)
            segs = out
        return segs

    def u_of_z(self, z, pass_index=1, lam=None):
        """Reduced position of the physical plane z (coordinates of that pass)."""
        for s in self.layout(pass_index, lam):
            if s["z0"] - 1e-15 <= z <= s["z1"] + 1e-15:
                return s["u0"] + (z - s["z0"]) / (s["z1"] - s["z0"]) * (s["u1"] - s["u0"])
        raise ValueError(f"z = {z} m is outside the cell")

    def centre_z(self, pass_index=1, lam=None):
        """Physical position of the centre of the equivalent cell on pass `pass_index` (0 for a
        symmetric layout)."""
        uc = self.equivalent_separation(lam) / 2
        for s in self.layout(pass_index, lam):
            if s["u0"] - 1e-15 <= uc <= s["u1"] + 1e-15:
                return s["z0"] + (uc - s["u0"]) / (s["u1"] - s["u0"]) * (s["z1"] - s["z0"])
        return 0.0

    def plate_transmission(self, lam):
        """Power transmission of all the plates of one pass (Fresnel / coatings)."""
        T = np.ones_like(np.asarray(lam, float))
        for s in self.layout(1):
            if s["plate"] is not None:
                p = s["plate"]
                T = T * (1 - p.face_R(s["entry_face"], s["amb_in"], lam))
                T = T * (1 - p.face_R(s["exit_face"], s["amb_out"], lam))
        return T

    @classmethod
    def herriott(cls, R, N, k, medium, branch="concentric", n_passes=None, **kw):
        """Re-entrant Herriott cell: N passes (twice the spots per mirror), d = R (1 +- cos(2*pi k/N)).
        N must be even for the concentric branch. Default n_passes = N (one full re-entrant cycle);
        pass n_passes to simulate any other number of passes in the same cell.
        With plates (and wavelength) in kw, d is the PHYSICAL separation whose equivalent cell is
        re-entrant."""
        value = np.cos(2 * np.pi * k / N)
        if branch == "concentric":
            if N % 2:
                raise ValueError(
                    f"N = {N} is odd: the concentric branch is only re-entrant for even N. N is the "
                    f"number of passes of one re-entrant cycle (e.g. N = {2 * N}, k = {2 * k}); "
                    f"to simulate a different number of passes, set n_passes separately.")
            d = R * (1 + value)
        elif branch == "planar":
            d = R * (1 - value)
        else:
            raise ValueError(f"unknown branch {branch!r}: use 'concentric' or 'planar'")
        if kw.get("plates") or kw.get("regions"):
            if kw.get("wavelength") is None:
                raise ValueError("herriott with plates / regions needs wavelength=lam0")
            target = d
            for _ in range(20):        # physical d whose equivalent cell has the re-entrant separation
                d += target - cls(R, d, medium, 1, **kw).equivalent_separation()
        return cls(R, d, medium, n_passes or N, **kw)

    # ------------------------------------------------------------ modes
    def linear_mode(self, lam):
        """Linear eigenmode at lam (with plates: of the equivalent cell; w0 at its centre, exact
        mirror spot)."""
        n0 = self.ref_medium.n(2 * np.pi * C0 / lam)
        lm = lam / n0
        d = self.equivalent_separation(lam)
        w0 = np.sqrt(lm / (2 * np.pi) * np.sqrt(d * (2 * self.R - d)))
        wm = np.sqrt(lm * self.R / np.pi * np.sqrt(d / (2 * self.R - d)))
        return {"w0": w0, "w_mirror": wm, "zR": np.pi * w0 ** 2 / lm,
                "gouy_per_pass": np.arccos(1 - d / self.R)}

    # ------------------------------------------------------------ Kerr-matched mode (moments)
    def _ratio(self, medium, P, lam):
        return 0.0 if medium.n2 == 0 else P / medium.critical_power(lam)

    def _segment_moments(self, S, Sp, du, r, lam):
        """(w^2, d(w^2)/du) after a reduced length du of a medium with P/Pc = r; also returns the
        minimum of w^2 inside the segment."""
        K = (lam / (np.pi * float(self.ref_medium.n_lam(lam)))) ** 2 * (1 - r)
        Spp = 2 * (K + Sp ** 2 / 4) / S
        S1, Sp1 = S + Sp * du + Spp * du ** 2 / 2, Sp + Spp * du
        smin = min(S, S1)
        if Spp > 0 and 0 < -Sp / Spp < du:
            smin = min(smin, S - Sp ** 2 / (2 * Spp))
        return S1, Sp1, smin

    def _walk_moments(self, S, Sp, P, lam, pass_index, u_stop=None, record=None):
        """Moments through the segments of one pass (up to u_stop if given, no bounce)."""
        for seg in self.layout(pass_index, lam, split_at_centre=True):
            if u_stop is not None and seg["u0"] >= u_stop - 1e-15:
                break
            r = self._ratio(seg["medium"], P, lam)
            if record is not None:
                record.append((seg, S, Sp, r))
            S, Sp, smin = self._segment_moments(S, Sp, seg["u1"] - seg["u0"], r, lam)
            if smin <= 0:
                raise ValueError("self-focusing collapse inside the cell (no matched mode at this power)")
        return S, Sp

    def _two_pass_map(self, v, P, lam):
        S, Sp = np.exp(v[0]), v[1]
        for p in (1, 2):
            S, Sp = self._walk_moments(S, Sp, P, lam, p)
            Sp -= 4 * S / self.R                 # curved mirror (focal length R/2)
        return S, Sp

    def _matched_moments(self, P, lam):
        """Fixed point (w^2, d(w^2)/du) right after mirror A, followed continuously from the linear
        eigenmode (P = 0) to P with an adaptive step; raises ValueError if the branch is lost
        (no matched mode: self-focusing collapse, or the mode jumps)."""
        wm = self.linear_mode(lam)["w_mirror"]
        v = np.array([np.log(wm ** 2), -2 * wm ** 2 / self.R])

        def F(x, Pk):
            try:
                S, Sp = self._two_pass_map(x, Pk, lam)
            except ValueError:
                return np.array([1e3, 1e3])
            return np.array([np.log(S) - x[0], (Sp - x[1]) * self.R / np.exp(x[0])])
        f, step = 0.0, 1 / 16
        while f < 1:
            fk = min(1.0, f + step)
            sol, _, ok, _ = fsolve(F, v, args=(fk * P,), full_output=True, xtol=1e-13)
            good = (ok == 1 and np.max(np.abs(F(sol, fk * P))) < 1e-8 and abs(sol[0] - v[0]) < 0.1
                    and abs(sol[1] - v[1]) * self.R < 0.2 * np.exp(v[0]))
            if good:
                v, f, step = sol, fk, min(step * 1.5, 0.25)
            else:
                step /= 2
                if step < 1e-4:
                    raise ValueError(f"no Kerr-matched mode beyond P = {f*P/1e6:.4g} MW "
                                     f"(self-focusing: the matched branch is lost)")
        return np.exp(v[0]), v[1]

    def _plate_mode(self, P, lam):
        """Kerr-matched mode of a cell with plates (moment method), same keys as nonlinear_mode
        plus per-plate data and the input beam to give at the centre."""
        S, Sp = self._matched_moments(P, lam)
        ng = float(self.ref_medium.n_lam(lam))
        kv = 2 * np.pi / lam
        lin = self.linear_mode(lam)
        uc = self.equivalent_separation(lam) / 2
        out = {"P": P, "Pc": self.medium.critical_power(lam) if self.medium.n2 else np.inf,
               "w_mirror": np.sqrt(S), "w0_linear": lin["w0"], "w_mirror_linear": lin["w_mirror"],
               "plates": [], "B_pass": []}
        out["P_over_Pc"] = P / out["Pc"]
        xg, wg = np.polynomial.legendre.leggauss(24)
        for p in (1, 2):
            rec = []
            S1, Sp1 = self._walk_moments(S, Sp, P, lam, p, record=rec)
            B, wmin = 0.0, np.inf
            for seg, S0, Sp0, r in rec:
                du = seg["u1"] - seg["u0"]
                med = seg["medium"]
                K = (lam / (np.pi * ng)) ** 2 * (1 - r)
                Spp = 2 * (K + Sp0 ** 2 / 4) / S0
                uu = (xg + 1) / 2 * du
                Su = S0 + Sp0 * uu + Spp * uu ** 2 / 2
                wmin = min(wmin, float(np.sqrt(Su.min())))
                scale = float(med.n_lam(lam)) / ng            # physical length per reduced length
                Bseg = kv * med.n2 * np.sum(wg * 2 * P / (np.pi * Su)) * du / 2 * scale
                B += Bseg
                if p == 1 and seg["plate"] is not None:
                    S1e, _, _ = self._segment_moments(S0, Sp0, du, r, lam)
                    q = seg["plate"]
                    if out["plates"] and out["plates"][-1]["plate"] is q:      # second half of a split plate
                        d = out["plates"][-1]
                        d["w_exit"] = np.sqrt(S1e)
                        d["B"] += Bseg
                        d["w_min"] = min(d["w_min"], float(np.sqrt(Su.min())))
                    else:
                        out["plates"].append({"plate": q, "w_entry": np.sqrt(S0), "w_exit": np.sqrt(S1e),
                                              "w_min": float(np.sqrt(Su.min())), "B": Bseg,
                                              "P_over_Pc": r, "Pc": med.critical_power(lam)})
            out["B_pass"].append(B)
            if p == 1:
                out["w_min"] = wmin
            S, Sp = S1, Sp1 - 4 * S1 / self.R
        for d in out["plates"]:
            d["I_peak_W_cm2"] = 2 * P / (np.pi * d["w_min"] ** 2) * 1e-4
        nb = self.n_passes
        out["B_per_pass"] = float(np.mean(out["B_pass"]))
        out["B_total"] = float(out["B_pass"][0] * ((nb + 1) // 2) + out["B_pass"][1] * (nb // 2))
        # matched beam at the centre, and the input beam: field at the centre that, propagated
        # LINEARLY back to the injection plane (as MPCSimulation does), gives the matched beam
        S, Sp = self._matched_moments(P, lam)
        Sc, _ = self._walk_moments(S, Sp, P, lam, 1, u_stop=uc)
        out["w0"] = np.sqrt(Sc)
        Si, Spi = self._walk_moments(S, Sp, 0.0, lam, 1, u_stop=uc)
        out["input_w"] = np.sqrt(Si)
        out["input_R"] = 2 * Si / Spi if Spi != 0 else np.inf
        out["I_peak_center_W_cm2"] = 2 * P / (np.pi * out["w0"] ** 2) * 1e-4
        return out

    def nonlinear_mode(self, P, lam, verbose=False, vortex=0):
        """Kerr-lens matched eigenmode for power P [W] at wavelength lam [m].

        vortex = l: Laguerre-Gauss LG_0,l beam (embedded-gaussian radius w, the radii returned; the
        second-moment radius is sqrt(l+1) w). The variance law is the gaussian one with
        Pc -> Pc f_l, f_l = (l+1) 4^l (l!)^2 / (2l)! (4, 8, 12.8 for l = 1, 2, 3), so the matched radii
        are those of a gaussian of power P/f_l; B and the peak intensities are taken on the ring
        maximum, g_l = l^l e^-l / l! times the gaussian on-axis value (Koltalo et al., APL Photonics
        10, 046110, 2025, Eqs. 2 and 4).

        With plates: moment method (see the module docstring). Keys: w_mirror, w0 (matched beam at
        the centre), w_min, B_pass (odd, even pass), B_per_pass, B_total, plates (per plate: w_entry,
        w_exit, w_min, B, P_over_Pc, I_peak_W_cm2), and input_w, input_R: radius and wavefront
        radius of curvature of the beam to give at the centre (Pulse.cell_matched builds it)."""
        if vortex:
            f, g = lg_critical_factor(vortex), lg_peak_factor(vortex)
            out = self.nonlinear_mode(P / f, lam)
            out["P"], out["vortex"] = P, int(vortex)
            for k in ("Pc",):
                out[k] = out[k] * f
            for k in ("B_per_pass", "B_total", "I_peak_center_W_cm2"):
                out[k] = out[k] * f * g
            if "B_pass" in out:
                out["B_pass"] = [b * f * g for b in out["B_pass"]]
            for d in out.get("plates", ()):
                d["Pc"], d["B"], d["I_peak_W_cm2"] = d["Pc"] * f, d["B"] * f * g, d["I_peak_W_cm2"] * f * g
            if verbose:
                print(self.report(out))
            return out
        if self.segmented:
            out = self._plate_mode(P, lam)
            if verbose:
                print(self.report(out))
            return out
        Pc = self.medium.critical_power(lam)
        r = P / Pc
        if r >= 1:
            raise ValueError(f"P/Pc = {r:.2f} >= 1: no matched mode (self-focusing)")
        lin = self.linear_mode(lam)
        s = (1 - r) ** 0.25
        w0, wm = lin["w0"] * s, lin["w_mirror"] * s
        lam_eff = lam * np.sqrt(1 - r)
        zRp = np.pi * w0 ** 2 / lam_eff
        B = 8 * np.pi * self.medium.n2 * P / (lam * lam_eff) * np.arctan(self.d / (2 * zRp))
        out = {"P": P, "Pc": Pc, "P_over_Pc": r, "w0": w0, "w_mirror": wm,
               "w0_linear": lin["w0"], "w_mirror_linear": lin["w_mirror"], "B_per_pass": B,
               "B_total": B * self.n_passes,
               "I_peak_center_W_cm2": 2 * P / (np.pi * w0 ** 2) * 1e-4}
        if verbose:
            print(self.report(out))
        return out

    def match(self, pulse, power="peak", verbose=True, lam=None, vortex=0):
        """Nonlinear mode matching from a Pulse (wavelength: its grid's lambda0), or from a power in W
        with its wavelength lam [m]. power: 'peak' | 'weighted'."""
        if hasattr(pulse, "grid"):
            P = pulse.peak_power() if power == "peak" else pulse.weighted_power()
            lam = pulse.grid.lambda0
        elif lam is None:
            raise ValueError("match(power): give the wavelength, e.g. cell.match(5e9, lam=800e-9)")
        else:
            P = float(pulse)
        return self.nonlinear_mode(P, lam, verbose=verbose, vortex=vortex)

    def report(self, m=None):
        s = (f"MPC: R = {self.R:.3f} m, d = {self.d:.4f} m, {self.n_passes} passes, "
             f"Gouy/pass = {np.degrees(self.theta):.2f} deg, {self.medium}")
        if self.segmented:
            s += (f"\n  equivalent separation {self.equivalent_separation():.4f} m, plate transmission "
                  f"{float(self.plate_transmission(self.wavelength))*100:.2f} %/pass")
            for z0, z1, med in self.regions:
                s += f"\n  region [{z0*1e3:+.1f}, {z1*1e3:+.1f}] mm: {med}"
            for p in self.plates:
                s += "\n  " + p.describe(self._medium_at(p.position + p.t), self.wavelength)
        if m and self.segmented:
            s += (f"\n  P = {m['P']/1e9:.3g} GW | mirror {m['w_mirror_linear']*1e6:.0f} um (linear) -> "
                  f"{m['w_mirror']*1e6:.0f} um (matched) | centre {m['w0_linear']*1e6:.0f} -> {m['w0']*1e6:.0f} um"
                  f" | B/pass ~ {m['B_per_pass']:.3f} rad, total ~ {m['B_total']:.2f} rad")
            for d in m["plates"]:
                s += (f"\n  {d['plate'].name}: w {d['w_entry']*1e6:.0f} -> {d['w_exit']*1e6:.0f} um, "
                      f"P/Pc = {d['P_over_Pc']:.3g}, I_peak = {d['I_peak_W_cm2']:.3g} W/cm^2, B = {d['B']:.3f} rad")
            return s
        if m:
            s += (f"\n  P = {m['P']/1e9:.3g} GW, Pc = {m['Pc']/1e9:.3g} GW, P/Pc = {m['P_over_Pc']:.3f}"
                  f"\n  waist  {m['w0_linear']*1e6:.0f} um (linear) -> {m['w0']*1e6:.0f} um (matched)"
                  f"\n  mirror {m['w_mirror_linear']*1e6:.0f} um (linear) -> {m['w_mirror']*1e6:.0f} um (matched)"
                  f"\n  B/pass ~ {m['B_per_pass']:.3f} rad, total ~ {m['B_total']:.2f} rad, "
                  f"I_peak(centre) ~ {m['I_peak_center_W_cm2']:.3g} W/cm^2")
        return s

    # ------------------------------------------------------------ mirror
    def mirror_spectrum(self, grid):
        """Spectral transfer of one bounce on grid.Omega: GDD, TOD, reflectivity and phase."""
        W = grid.Omega * FS
        lam = np.where(grid.valid, grid.lam, 1.0)
        spec = np.exp(1j * (self.gdd / 2 * W ** 2 + self.tod / 6 * W ** 3))
        spec *= np.sqrt(np.clip(self.refl(lam), 0, 1)) if callable(self.refl) else np.sqrt(self.refl)
        if self.mphase is not None:
            spec *= np.exp(1j * self.mphase(lam))
        return spec * grid.valid

    def mirror_aperture(self, r):
        """Amplitude transmission of the clear aperture at radius r (soft super-gaussian edge)."""
        if self.diameter is None:
            return np.ones_like(r)
        return np.exp(-(r / (self.diameter / 2)) ** 30)

    def bounce_factor(self, grid):
        """Complex transfer of one mirror in (x, y, Om) (cached)."""
        if self._bounce_cache is not None and self._bounce_cache[0] is grid:
            return self._bounce_cache[1]
        k = self.ref_medium.k(np.where(grid.valid, grid.omega, grid.omega0))   # medium at the mirror
        F = np.exp(-1j * (grid.r2[:, :, None] * (k / self.R)[None, None, :]))
        F *= self.mirror_spectrum(grid)[None, None, :]
        F *= self.mirror_aperture(np.sqrt(grid.r2))[:, :, None]
        F = grid.asarray(F.astype(grid.dtype))     # built on the host, stored on the grid's device
        self._bounce_cache = (grid, F)
        return F

    def bounce(self, Aw, grid):
        Aw *= self.bounce_factor(grid)
        return Aw
