"""Shared machinery of the benchmarks of MPCs with plates (bulk and hybrid): Kerr-matched injection
(moment method, gaussian or LG_0,l), input-M2 model, and the plate / compression / vortex metrics.
Not a benchmark itself: lund_bulk_mpc.py, bench_omar2024_400uJ.py and bench_seidel2022_112uJ.py
subclass PlateMPCBenchmark and provide make_cell and their parameters.

Parameters read here (subclasses give them all in `params`):
    vortex          LG_0,l order (0 = gaussian)
    M2_input        input M2 per axis
    m2_model        "intensity": the field stays the ideal mode and every n2 is divided by M2x M2y (the
                    intensity of a beam M2 times larger in each axis, as in SISYFOS when the authors
                    "account for the measured M2 to determine the peak intensity"); "none": ideal beam.
                    Each benchmark follows what the authors' own simulation did.
    matching_power  Kerr matching at the "weighted" or "peak" power of the pulse
    matching_scale  factor on that power (optional, default 1): a long pulse that stretches in the cell
                    stays closer to the matched mode when matched below its peak power
    compressor_gdd  the paper's compressor GDD [fs^2] (None: best GDD only)
    compressor_T    compressor transmission (peak power after the compressor)
"""
import numpy as np

import thunder4d as th

from framework import Benchmark, Ref, beam_pixels, fwhm, integrated_power, metric, spectral_phase


class PlateMPCBenchmark(Benchmark):
    """Kerr-matched injection and metrics of a cell with plates; subclasses define make_cell."""

    # grid presets for the gaussian; vortex classes scale the box with the beam (sqrt(l+1))
    base_presets = [{"box_factor": 8, "points_per_waist": 8, "time_window_factor": 8},
                    {"box_factor": 7, "points_per_waist": 7, "time_window_factor": 7},
                    {"box_factor": 6, "points_per_waist": 6, "time_window_factor": 6}]
    base_quick = {"box_factor": 5, "points_per_waist": 3, "time_window_factor": 4}

    def __init_subclass__(cls, **kw):
        super().__init_subclass__(**kw)
        l = abs(int(cls.params.get("vortex", 0))) if cls.params else 0
        s = np.sqrt(l + 1)                     # second-moment size of LG_0,l / gaussian
        # presets written in the subclass body are its base presets (else they would be overwritten here)
        if "presets" in cls.__dict__:
            cls.base_presets = cls.__dict__["presets"]
        if "quick_preset" in cls.__dict__:
            cls.base_quick = cls.__dict__["quick_preset"]
        cls.presets = [{**p, "box_factor": round(p["box_factor"] * s, 1)} for p in cls.base_presets]
        cls.quick_preset = {**cls.base_quick, "box_factor": round(cls.base_quick["box_factor"] * s, 1)}

    # ---------------------------------------------------------------- cell
    @staticmethod
    def check_m2_model(P):
        if P["m2_model"] not in ("intensity", "none"):
            raise ValueError(f"m2_model {P['m2_model']!r}: 'intensity' or 'none'")

    @staticmethod
    def kerr_scale(P):
        """Factor on every n2: 1 / (M2x M2y) for the input M2 ('intensity'), 1 for 'none'."""
        return 1.0 / P["M2_input"] ** 2 if P["m2_model"] == "intensity" else 1.0

    # ---------------------------------------------------------------- injection
    def pulse(self, ctx):
        """Kerr-matched gaussian or LG_0,l beam (Pulse.cell_matched), given at the cell centre."""
        P, grid, cell = ctx.P, ctx.grid, ctx.cell
        E, src = self.spectrum(grid, P)
        l = int(P["vortex"])
        probe = th.Pulse.cell_matched(grid, E, cell, P["energy"], nonlinear=False, verbose=False)
        Pm = probe.peak_power() if P["matching_power"] == "peak" else probe.weighted_power()
        Pm *= P.get("matching_scale", 1.0)
        pulse = th.Pulse.cell_matched(grid, E, cell, P["energy"], power=Pm, vortex=l, verbose=False)
        ctx.lin = cell.linear_mode(P["lambda0"])
        ctx.mm = cell.nonlinear_mode(Pm, P["lambda0"], vortex=l)
        ctx.mm_peak = cell.nonlinear_mode(pulse.peak_power(), P["lambda0"], vortex=l)
        ctx.w0_ref = pulse.input_w
        ctx.mode_reference = ctx.lin["w0"] * np.sqrt(grid.lambda0 / P["lambda0"])
        ctx.spectrum_source = src + (f", LG_0,{l}" if l else ", gaussian beam")
        ctx.input_Aw = pulse.Aw.copy()
        print(cell.report(ctx.mm))
        return pulse

    # ---------------------------------------------------------------- metrics
    def extra_metrics(self, ctx):
        from thunder4d import diagnostics as dg
        sim, grid, P, R = ctx.sim, ctx.grid, ctx.P, self.reference
        l = int(P["vortex"])
        M, C = {}, {}
        h = sim.history
        mp = ctx.mm_peak

        # --- beam quality: with the 'intensity' model the input M2 is not in the field (ideal mode),
        # so the M2 measured on the field is the degradation in the cell only; the M2 of the beam is
        # estimated as input M2 x field M2 (product rule, exact for an embedded-gaussian beam)
        m2_in = P["M2_input"] if P["m2_model"] == "intensity" else 1.0
        M["M2_input"] = metric(m2_in, "", "input M2 per axis (in the n2, not in the field)")
        for ax in ("x", "y"):
            f = h[f"M2{ax}_mean"][-1]
            M[f"M2{ax}"] = metric(f, "", f"M2 {ax} of the simulated field (ideal-mode input: 1 or l+1)",
                                  R.get(f"M2{ax}"))
            M[f"M2{ax}_beam"] = metric(f * m2_in, "", f"M2 {ax} of the output beam = input M2 x field M2",
                                       R.get(f"M2{ax}_beam"))

        # --- plates and Kerr matching
        if ctx.cell.medium.n2 > 0:                # filling gas (air near the mirrors for the Lund cell)
            M["P_over_Pc"] = metric(mp["P_over_Pc"], "", f"input peak power / Pc in the {ctx.cell.medium.name}")
        plate = mp["plates"][0]
        M["P_over_Pc_plate"] = metric(plate["P_over_Pc"], "", "input peak power / Pc of the mode in FS",
                                      R.get("P_over_Pc_plate"))
        M["I_plate_max_Wcm2"] = metric(plate["I_peak_W_cm2"], "W/cm2",
                                       "peak intensity in a plate (matched estimate)")
        Bpl = sum(d["B"] for d in mp["plates"]) * P["n_passes"]      # per crossing (pass 1) x passes
        M["B_matched_estimate_rad"] = metric(mp["B_total"], "rad",
                                             "B-integral, matched-mode estimate at the peak power (all media)")
        M["B_plates_estimate_rad"] = metric(Bpl, "rad", "same, plates only")
        M["B_fraction_plates_estimate"] = metric(Bpl / max(mp["B_total"], 1e-30), "",
                                                 "share of the plates in the matched-mode estimate (no pulse "
                                                 "stretching)", R.get("B_fraction_plates_estimate"))
        M["w_mirror_matched_um"] = metric(ctx.mm["w_mirror"] * 1e6, "um",
                                          f"Kerr-matched radius on the mirrors ({P['matching_power']} power)")
        # mirror spot from pass to pass, against the matched mode (2 rms = sqrt(l+1) w for LG_0,l)
        s = sim.mirror_scale
        x2 = (grid.xs ** 2)[:, None]
        wm = [2 * np.sqrt(np.sum(F * x2) / np.sum(F)) * s for F in h["fluence_mirror"] if F is not None]
        if wm:
            wm = np.array(wm) / np.sqrt(l + 1)
            M["mirror_spot_dev"] = metric(np.max(np.abs(wm / ctx.mm["w_mirror"] - 1)), "",
                                          "max deviation of the mirror spot from the matched mode",
                                          R.get("mirror_spot_dev"))
            C["w_mirror_pass_um"] = wm * 1e6

        # --- share of the plates in the nonlinearity (one solver per medium)
        S = sim.solver
        if hasattr(S, "all"):
            Bp = sum(s.B for s in S.all if isinstance(s.medium, th.Solid))
            M["B_fraction_plates"] = metric(Bp / max(S.B, 1e-30), "", "share of the plates in the total B-integral",
                                            R.get("B_fraction_plates"))

        # --- output before the compressor, and compression at the paper's GDD (else the best GDD)
        wc = 2 * np.pi * th.C0 / P["lambda0"]
        pix = beam_pixels(sim.Aw)
        t0, P0 = integrated_power(grid, pix, None, self.compression_up)
        M["chirped_out_fwhm_fs"] = metric(fwhm(t0, P0), "fs", "output pulse FWHM before the compressor",
                                          R.get("chirped_out_fwhm_fs"))
        gdd = P.get("compressor_gdd")
        if gdd is None:
            best = getattr(ctx, "metrics", {}).get("best_gdd_fs2")
            gdd = best["value"] if best else 0.0
        t, Pc = integrated_power(grid, pix, spectral_phase(grid, wc, gdd), self.compression_up)
        if P.get("compressor_gdd") is not None:
            M["comp_peak_power_GW"] = metric(Pc.max() / 1e9, "GW", f"compressed peak power, {gdd:.0f} fs2",
                                             R.get("comp_peak_power_GW"))
        k = int(np.argmax(Pc))
        main = np.abs(t - t[k]) <= fwhm(t, Pc)                  # 2 x FWHM around the peak
        M["main_pulse_fraction"] = metric(Pc[main].sum() / Pc.sum(), "",
                                          f"energy within 2 x FWHM of the compressed peak ({gdd:.0f} fs2)",
                                          R.get("main_pulse_fraction"))
        # best GDD + TOD (Nelder-Mead from the best GDD): does a TOD-matched compressor close the gap
        # to the Fourier limit? (diagnostic: the papers' compressors are given as GDD only)
        from scipy.optimize import minimize

        def neg_peak(x):
            return -integrated_power(grid, pix, spectral_phase(grid, wc, x[0] * 1e3, x[1] * 1e4),
                                     self.compression_up)[1].max()
        best = getattr(ctx, "metrics", {}).get("best_gdd_fs2")
        x0 = [(best["value"] if best else gdd) / 1e3, 0.0]
        res = minimize(neg_peak, x0, method="Nelder-Mead",
                       options={"xatol": 1e-3, "fatol": 1e-4 * -neg_peak(x0), "maxfev": 120})
        t3, P3 = integrated_power(grid, pix, spectral_phase(grid, wc, res.x[0] * 1e3, res.x[1] * 1e4),
                                  self.compression_up)
        M["gdd_tod_comp_fs"] = metric(fwhm(t3, P3), "fs", "compressed with the best GDD + TOD",
                                      R.get("gdd_tod_comp_fs"))
        M["gdd_tod_gdd_fs2"] = metric(res.x[0] * 1e3, "fs2", "GDD of that best GDD + TOD compression")
        M["gdd_tod_tod_fs3"] = metric(res.x[1] * 1e4, "fs3", "TOD of that best GDD + TOD compression")
        T = P["compressor_T"]
        M["peak_power_out_GW"] = metric(Pc.max() / 1e9 * T, "GW",
                                        f"compressed peak power ({gdd:.0f} fs2) x compressor T = {T:g}",
                                        R.get("peak_power_out_GW"))

        # --- vortex: topological charge, M2 / (l+1), ring homogeneity of the collimated beam
        if l:
            def charge(A):
                lg = dg.lg_mode_content(grid, A, lambda lam: ctx.mode_reference * np.sqrt(lam / grid.lambda0),
                                        p_max=3, l_max=l + 4)
                f = lg["frac"].sum(axis=0)
                return f[lg["l"] == l].sum() / f.sum(), np.sum(lg["l"] * f) / f.sum()
            _, oam_in = charge(ctx.input_Aw)
            q_out, oam_out = charge(sim.Aw)
            M["charge_fraction"] = metric(q_out, "", f"energy fraction with topological charge l = {l} (output)",
                                          R.get("charge_fraction"))
            M["oam_mean"] = metric(oam_out, "", "mean OAM per photon, output (units of hbar)")
            M["oam_change"] = metric(oam_out - oam_in, "", "change of the mean OAM through the cell",
                                     Ref(0.0, 0.02, kind="abs", note="charge conserved"))
            M["M2x_over_lp1"] = metric(h["M2x_mean"][-1] / (l + 1), "", "field M2 x / (l+1) (1 = ideal LG_0,l)")
            M["M2x_beam_over_lp1"] = metric(h["M2x_mean"][-1] * m2_in / (l + 1), "",
                                            "output beam M2 x / (l+1) (input: M2_input)")
            theta, Ef = dg.near_field(grid, sim.Aw)
            F = np.sum(np.abs(Ef) ** 2, axis=2)
            ih, wh = ring_homogeneity(theta, F)
            theta0, Ef0 = dg.near_field(grid, ctx.input_Aw)
            F0 = np.sum(np.abs(Ef0) ** 2, axis=2)
            ih0, wh0 = ring_homogeneity(theta0, F0)
            M["IH_out"] = metric(ih, "", "ring intensity homogeneity, collimated output (0 = ideal)",
                                 R.get("IH_out"))
            M["WH_out"] = metric(wh, "", "ring width homogeneity, collimated output (0 = ideal)",
                                 R.get("WH_out"))
            M["IH_in"] = metric(ih0, "", "ring intensity homogeneity, input (0 = ideal)")
            M["WH_in"] = metric(wh0, "", "ring width homogeneity, input (0 = ideal)")
            C["near_theta_mrad"], C["near_F_out"] = theta * 1e3, (F / F.max()).astype(np.float32)
        return M, C


def ring_homogeneity(theta, F, n_cuts=18):
    """(IH, WH) of Koltalo et al. (Eqs. 6-7): over n diameters through the beam centre, mean of
    |I1 - I2| / (I1 + I2) for the two ring maxima and of |s1 - s2| / (s1 + s2) for their FWHM.
    theta: sorted axis of the square map F (theta_x, theta_y)."""
    from scipy.ndimage import map_coordinates
    F = np.asarray(F, float)
    n = len(theta)
    ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    ic, jc = np.sum(F * ii) / F.sum(), np.sum(F * jj) / F.sum()
    r = np.linspace(-n / 2 + 1, n / 2 - 1, 4 * n)
    IH, WH = [], []
    for a in np.linspace(0, np.pi, n_cuts, endpoint=False):
        prof = map_coordinates(F, [ic + r * np.cos(a), jc + r * np.sin(a)], order=1)
        sides = [(prof[r < 0][::-1], r[r < 0][::-1]), (prof[r >= 0], r[r >= 0])]
        peaks, widths = [], []
        for p, rr in sides:
            k = int(np.argmax(p))
            half = p[k] / 2
            lo = k
            while lo > 0 and p[lo] > half:
                lo -= 1
            hi = k
            while hi < len(p) - 1 and p[hi] > half:
                hi += 1
            # half-maximum crossings, linearly interpolated
            r_lo = np.interp(half, [p[lo], p[lo + 1]], [rr[lo], rr[lo + 1]]) if lo < k else rr[lo]
            r_hi = np.interp(half, [p[hi], p[hi - 1]], [rr[hi], rr[hi - 1]]) if hi > k else rr[hi]
            peaks.append(p[k])
            widths.append(abs(r_hi - r_lo))
        IH.append(abs(peaks[0] - peaks[1]) / (peaks[0] + peaks[1]))
        WH.append(abs(widths[0] - widths[1]) / max(widths[0] + widths[1], 1e-30))
    return float(np.mean(IH)), float(np.mean(WH))


