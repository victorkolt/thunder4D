"""Built-in figures of the results viewer (registered in registry.PLOTS).

Each figure states the arrays it needs (`requires`); the viewer offers it only for runs that hold them.
A new output of the solver (per-stage diagnostics, Stokes parameters, ...) gets its figure by registering
one more PlotSpec, here or from its own module. draw(results, fig, options): results is a
gui.results.Results, fig a matplotlib Figure, options["index"] the selected record for per-pass figures.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from .. import diagnostics as dg
from .registry import PLOTS, PlotSpec

NM, FS, MM = 1e9, 1e15, 1e3


def _passes(r: Mapping[str, Any]) -> np.ndarray:
    return np.asarray(r["pass"])


def _index(r: Mapping[str, Any], o: Mapping[str, Any]) -> int:
    return int(np.clip(o.get("index", -1), -len(r["pass"]), len(r["pass"]) - 1)) % len(r["pass"])


def _pass_label(r: Mapping[str, Any], o: Mapping[str, Any]) -> str:
    return f"pass {int(_passes(r)[_index(r, o)])}"


def _xy_extent(r: Mapping[str, Any]) -> list[float]:
    nx, ny, _, dx, dy, _ = (float(v) for v in r["grid"])
    return [-nx * dx / 2 * MM, nx * dx / 2 * MM, -ny * dy / 2 * MM, ny * dy / 2 * MM]


def second_moment_radius(F: np.ndarray, dx: float, dy: float) -> tuple[float, float]:
    """(wx, wy) = 2 sqrt(<x^2>), 2 sqrt(<y^2>) of a sorted 2D map (centroid removed)."""
    F = np.asarray(F, dtype=float)
    x = (np.arange(F.shape[0]) - F.shape[0] / 2) * dx
    y = (np.arange(F.shape[1]) - F.shape[1] / 2) * dy
    total = F.sum()
    px, py = F.sum(axis=1) / total, F.sum(axis=0) / total
    mx, my = (px * x).sum(), (py * y).sum()
    return 2 * np.sqrt((px * (x - mx) ** 2).sum()), 2 * np.sqrt((py * (y - my) ** 2).sum())


# ------------------------------------------------------------ evolution versus pass
def draw_energy_b(r, fig, o):
    p = _passes(r)
    ax1, ax2 = fig.subplots(1, 2)
    ax1.plot(p, np.asarray(r["energy"]) / r["energy"][0], ".-")
    ax1.set(xlabel="pass", ylabel="energy / input energy", title="Transmission")
    ax2.plot(p, r["B"], ".-")
    ax2.set(xlabel="pass", ylabel="B [rad]", title="B-integral (accumulated)")


def draw_peak(r, fig, o):
    p = _passes(r)
    ax1, ax2 = fig.subplots(1, 2)
    ax1.plot(p, np.asarray(r["peak_power"]) * 1e-9, ".-")
    ax1.set(xlabel="pass", ylabel="peak power [GW]", title="Peak power (uncompressed)")
    if "peak_intensity" in r:
        ax2.plot(p, np.asarray(r["peak_intensity"]) * 1e-16, ".-")
        ax2.set(xlabel="pass", ylabel="[TW/cm²]", title="Peak intensity at the cell centre")
    else:
        ax2.set_axis_off()


def mirror_fluence_physical(r) -> np.ndarray:
    """Fluence on the mirrors [J/m^2] in physical units (mirror records may be in lens-frame units)."""
    return np.asarray(r["fluence_mirror"], dtype=float) / float(r["mirror_scale"]) ** 2


def caustic_data(r, o):
    _, _, _, dx, dy, _ = (float(v) for v in r["grid"])
    out = {"pass": _passes(r), "w_centre_x_m": np.asarray(r["w_peak_x"]), "w_centre_y_m": np.asarray(r["w_peak_y"])}
    if "fluence_mirror" in r and "mirror_scale" in r:
        s = float(r["mirror_scale"])
        w = np.array([second_moment_radius(F, s * dx, s * dy) for F in r["fluence_mirror"]])
        out |= {"mirror_pass": _passes(r)[1:len(w) + 1], "w_mirror_x_m": w[:, 0], "w_mirror_y_m": w[:, 1]}
    return out


def draw_caustic(r, fig, o):
    d = caustic_data(r, o)
    has_mirror = "w_mirror_x_m" in d
    axes = fig.subplots(1, 2 if has_mirror else 1, squeeze=False)[0]
    axes[0].plot(d["pass"], d["w_centre_x_m"] * 1e6, ".-", label="x")
    axes[0].plot(d["pass"], d["w_centre_y_m"] * 1e6, ".-", label="y")
    axes[0].set(xlabel="pass", ylabel="radius [µm]", title="Beam radius at the cell centre (spectral peak)")
    axes[0].legend()
    if has_mirror:
        axes[1].plot(d["mirror_pass"], d["w_mirror_x_m"] * MM, ".-", label="x")
        axes[1].plot(d["mirror_pass"], d["w_mirror_y_m"] * MM, ".-", label="y")
        axes[1].set(xlabel="pass", ylabel="radius [mm]", title="Beam radius on the mirrors (second moment)")
        axes[1].legend()


def draw_quality(r, fig, o):
    p = _passes(r)
    ax1, ax2 = fig.subplots(1, 2)
    ax1.plot(p, r["eta00_total"], ".-", label="fundamental-mode content")
    for key, label in (("V_far", "homogeneity V, centre"), ("V_near", "homogeneity V, collimated")):
        if key in r and np.any(np.isfinite(r[key])):
            ax1.plot(p, r[key], ".-", label=label)
    ax1.set(xlabel="pass", title="Spatio-spectral quality")
    ax1.legend(fontsize="small")
    ax2.plot(p, r["M2x_mean"], ".-", label="x")
    ax2.plot(p, r["M2y_mean"], ".-", label="y")
    ax2.set(xlabel="pass", ylabel="M²", title="Beam quality (spectrally averaged)")
    ax2.legend()


def mirror_fluence_data(r, o):
    F = mirror_fluence_physical(r)
    return {"pass": _passes(r)[1:len(F) + 1], "peak_fluence_J_m2": F.reshape(len(F), -1).max(axis=1)}


def draw_mirror_fluence(r, fig, o):
    d = mirror_fluence_data(r, o)
    peak = d["peak_fluence_J_m2"] * 1e-4
    ax = fig.subplots()
    ax.plot(d["pass"], peak * 1e3, ".-")
    ax.set(xlabel="pass (bounce at its end)", ylabel="peak fluence [mJ/cm²]", title="Peak fluence on the mirrors")


# ------------------------------------------------------------ spectra and pulses
def draw_spectra(r, fig, o):
    lam, S, p = np.asarray(r["lam"]) * NM, np.asarray(r["spectrum"]), _passes(r)
    i = _index(r, o)
    ax1, ax2 = fig.subplots(1, 2)
    Sn = S / S.max(axis=1, keepdims=True)
    mesh = ax1.pcolormesh(lam, p, 10 * np.log10(np.maximum(Sn, 1e-6)), vmin=-40, vmax=0, shading="nearest")
    ax1.axhline(p[i], color="w", lw=0.8, ls="--")
    ax1.set(xlabel="wavelength [nm]", ylabel="pass", title="Spectrum versus pass")
    fig.colorbar(mesh, ax=ax1, label="dB")
    ax2.semilogy(lam, Sn[0], label="input")
    ax2.semilogy(lam, Sn[i], label=_pass_label(r, o))
    ax2.set(xlabel="wavelength [nm]", ylim=(1e-4, 2), title="Spectrum (normalised)")
    ax2.legend()


GDD_BOUNDS_FS2 = (-30000.0, 30000.0)
OVERSAMPLED_SIZE = 2e6        # pixels x time samples x oversampling: about a second, a few 100 MB


def compressed(r: Any) -> dict[str, Any]:
    """Best uniform GDD of the final field (cached on the results): the GDD scan of dg.compress, and, on
    small grids, the compressed pulse on an oversampled time axis (they hold short pulses in a few
    samples). Large grids sample the pulse well enough and would take minutes oversampled."""
    if "compress" not in r.cache:
        from ..design.export import compress_oversampled

        g, A = r.grid(), r["Aw_final"]
        out = dg.compress(g, A, gdd_bounds_fs2=GDD_BOUNDS_FS2)
        F = np.sum(np.abs(A) ** 2, axis=2)
        up = min(8, int(OVERSAMPLED_SIZE // (np.count_nonzero(F > 1e-4 * F.max()) * g.Nt)))
        if up >= 2:
            fine = compress_oversampled(g, A, g.energy(A), GDD_BOUNDS_FS2, up=up)
            out |= {k: fine[k] for k in ("gdd_fs2", "t_fs", "P", "fwhm_fs", "tl_fwhm_fs")}
        r.cache["compress"] = out
    return r.cache["compress"]


def pulse_data(r, o):
    t, P = dg.temporal_power(r.grid(), r["Aw_final"])
    c = compressed(r)
    return {"t_fs": t * FS, "P_output_W": P, "t_compressed_fs": c["t_fs"], "P_compressed_W": c["P"],
            "gdd_fs2": c["gdd_fs2"], "fwhm_fs": c["fwhm_fs"], "tl_fwhm_fs": c["tl_fwhm_fs"],
            "gdd_scan_fs2": c["gdd_scan_fs2"], "peak_scan_W": c["peak_scan"]}


def draw_pulse(r, fig, o):
    d = pulse_data(r, o)
    ax1, ax2 = fig.subplots(1, 2)
    ax1.plot(d["t_fs"], d["P_output_W"] * 1e-9, label="output (uncompressed)")
    ax1.plot(d["t_compressed_fs"], d["P_compressed_W"] * 1e-9,
             label=f"compressed: {d['fwhm_fs']:.1f} fs (TL {d['tl_fwhm_fs']:.1f} fs)")
    on = d["t_fs"][d["P_output_W"] > 1e-3 * d["P_output_W"].max()]          # uncompressed pulse support
    half = max(abs(on).max() if on.size else 0.0, 10 * d["fwhm_fs"])
    ax1.set(xlabel="time [fs]", ylabel="power [GW]", title="Output pulse", xlim=(-half, half))
    ax1.legend(fontsize="small")
    ax2.plot(d["gdd_scan_fs2"], np.asarray(d["peak_scan_W"]) * 1e-9, ".-")
    ax2.axvline(d["gdd_fs2"], color="k", lw=0.8, ls="--")
    ax2.set(xlabel="compressor GDD [fs²]", ylabel="peak power [GW]",
            title=f"Best uniform GDD: {d['gdd_fs2']:.0f} fs²")


# ------------------------------------------------------------ spatio-spectral maps
def draw_fluence(r, fig, o):
    i = _index(r, o)
    ax = fig.subplots()
    F = np.asarray(r["fluence"][i]) * 1e-4 * 1e3
    im = ax.imshow(F.T, origin="lower", extent=_xy_extent(r), aspect="equal")
    fig.colorbar(im, ax=ax, label="mJ/cm²")
    ax.set(xlabel="x [mm]", ylabel="y [mm]", title=f"Fluence at the cell centre, {_pass_label(r, o)}")


def _spectrogram(fig, coord, lam, S, xlabel, title):
    ax = fig.subplots()
    S = np.asarray(S, dtype=float)
    mesh = ax.pcolormesh(np.asarray(lam) * NM, coord, 10 * np.log10(np.maximum(S / S.max(), 1e-6)),
                         vmin=-30, vmax=0, shading="nearest")
    fig.colorbar(mesh, ax=ax, label="dB")
    ax.set(xlabel="wavelength [nm]", ylabel=xlabel, title=title)


def draw_sgram_x(r, fig, o):
    i = _index(r, o)
    _spectrogram(fig, np.asarray(r["x"]) * MM, r["lam"], r["sgram_x"][i], "x [mm]",
                 f"(x, λ) spectrogram at the cell centre, {_pass_label(r, o)}")


def draw_sgram_theta(r, fig, o):
    i = _index(r, o)
    _spectrogram(fig, np.asarray(r["theta"]) * 1e3, r["lam"], r["sgram_theta"][i], "angle [mrad]",
                 f"(θ, λ) spectrogram of the collimated beam, {_pass_label(r, o)}")


def draw_mode_content(r, fig, o):
    i = _index(r, o)
    ax = fig.subplots()
    lam = np.asarray(r["lam"]) * NM
    S = np.asarray(r["spectrum"][i])
    ax.plot(lam, r["eta00"][i], label="fundamental-mode content")
    ax.plot(lam, S / S.max(), color="0.6", lw=0.8, label="spectrum (normalised)")
    ax.set(xlabel="wavelength [nm]", ylim=(0, 1.05), title=f"Mode content per colour, {_pass_label(r, o)}")
    ax.legend(fontsize="small")


def _index_data(*keys):
    def data(r, o):
        i = _index(r, o)
        out = {"pass": int(_passes(r)[i]), "lam_m": np.asarray(r["lam"])}
        for k in keys:
            out[k] = np.asarray(r[k][i])
        if "x" in r:
            out["x_m"] = np.asarray(r["x"])
        if "theta" in r and "sgram_theta" in keys:
            out["theta_rad"] = np.asarray(r["theta"])
        return out
    return data


for spec in (
    PlotSpec("transmission", "Transmission and B-integral", draw_energy_b, ("pass", "energy", "B"), 10, "Evolution"),
    PlotSpec("peak", "Peak power and intensity", draw_peak, ("pass", "peak_power"), 20, "Evolution"),
    PlotSpec("caustic", "Beam size (caustic)", draw_caustic, ("pass", "w_peak_x", "w_peak_y", "grid"), 30,
             "Evolution", data=caustic_data),
    PlotSpec("quality", "Beam quality", draw_quality, ("pass", "eta00_total", "M2x_mean", "M2y_mean"), 40, "Evolution"),
    PlotSpec("mirror_fluence", "Fluence on the mirrors", draw_mirror_fluence,
             ("pass", "fluence_mirror", "mirror_scale"), 50, "Evolution", data=mirror_fluence_data),
    PlotSpec("spectra", "Spectra", draw_spectra, ("pass", "lam", "spectrum"), 60, "Spectra and pulse", per_pass=True),
    PlotSpec("pulse", "Output and compressed pulse", draw_pulse, ("Aw_final", "grid", "lambda0"), 70,
             "Spectra and pulse", data=pulse_data),
    PlotSpec("fluence", "Fluence map", draw_fluence, ("pass", "fluence", "grid"), 80, "Spatio-spectral",
             per_pass=True, data=_index_data("fluence")),
    PlotSpec("sgram_x", "(x, λ) spectrogram", draw_sgram_x, ("pass", "x", "lam", "sgram_x"), 90, "Spatio-spectral",
             per_pass=True, data=_index_data("sgram_x")),
    PlotSpec("sgram_theta", "(θ, λ) spectrogram", draw_sgram_theta, ("pass", "theta", "lam", "sgram_theta"), 100,
             "Spatio-spectral", per_pass=True, data=_index_data("sgram_theta")),
    PlotSpec("mode_content", "Mode content per colour", draw_mode_content, ("pass", "lam", "eta00", "spectrum"), 110,
             "Spatio-spectral", per_pass=True, data=_index_data("eta00", "spectrum")),
):
    if spec.key not in PLOTS:
        PLOTS.register(spec)
