"""Small helpers shared by the tests."""
import numpy as np

import thunder4d as m

LAM0 = 1030e-9


def vacuum(n2=0.0):
    """Medium with n = 1: no dispersion, so propagation tests isolate diffraction or Kerr."""
    return m.Material("vacuum", n_func=np.ones_like, n2=n2)


def moments(x, w):
    """Mean and rms width of the coordinate x weighted by w (broadcast together)."""
    x, w = np.broadcast_arrays(x, w)
    mean = np.sum(w * x) / np.sum(w)
    return mean, np.sqrt(np.sum(w * (x - mean) ** 2) / np.sum(w))


def overlap(a, b):
    """|<a, b>|^2 / (<a, a> <b, b>): 1 when the two fields are equal up to a global phase."""
    a, b = np.ravel(a).astype(complex), np.ravel(b).astype(complex)
    return abs(np.vdot(a, b)) ** 2 / (np.vdot(a, a).real * np.vdot(b, b).real)
