"""Quantities followed live during a run (registered in registry.METRICS), read from MPCSimulation.history.

A new recorded quantity (per-stage B, Stokes parameters, fluence per mirror, ...) is followed by adding
one MetricSpec here, or from its own module.
"""
from .registry import METRICS, MetricSpec

for spec in (
    MetricSpec("energy", "Energy", "mJ", scale=1e3, order=10),
    MetricSpec("B", "B-integral", "rad", order=20),
    MetricSpec("peak_power", "Peak power", "GW", scale=1e-9, order=30),
    MetricSpec("peak_intensity", "Peak intensity (centre)", "TW/cm²", scale=1e-16, order=40),
    MetricSpec("eta00_total", "Fundamental-mode content", order=50, fmt="{:.4f}"),
    MetricSpec("M2x_mean", "M² (x)", order=60, fmt="{:.3f}"),
    MetricSpec("M2y_mean", "M² (y)", order=61, fmt="{:.3f}", plot=False),
):
    if spec.key not in METRICS:
        METRICS.register(spec)
