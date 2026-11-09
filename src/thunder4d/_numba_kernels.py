"""Numba kernels of thunder4d.kernels (imported only when they are used: importing Numba takes
a moment). See thunder4d.kernels for when they run."""
import numba as nb
import numpy as np

_jit = nb.njit(parallel=True, cache=True)


@_jit
def kerr_inplace(A):
    """A *= |A|^2 in place (time domain); returns max |A|^2 before the update (NaN if any)."""
    nx, ny, nt = A.shape
    pk = np.zeros(nx * ny)
    bad = np.zeros(nx * ny, np.bool_)
    for ij in nb.prange(nx * ny):
        i, j = ij // ny, ij % ny
        m = 0.0
        for k in range(nt):
            a = A[i, j, k]
            intensity = a.real * a.real + a.imag * a.imag
            if intensity > m:
                m = intensity
            elif np.isnan(intensity):
                bad[ij] = True
            A[i, j, k] = a * intensity
        pk[ij] = m
    if bad.any():
        return np.nan
    return pk.max()


@_jit
def peak_abs2(A):
    """max |A|^2 (NaN if any element is NaN)."""
    nx, ny, nt = A.shape
    pk = np.zeros(nx * ny)
    bad = np.zeros(nx * ny, np.bool_)
    for ij in nb.prange(nx * ny):
        i, j = ij // ny, ij % ny
        m = 0.0
        for k in range(nt):
            a = A[i, j, k]
            intensity = a.real * a.real + a.imag * a.imag
            if intensity > m:
                m = intensity
            elif np.isnan(intensity):
                bad[ij] = True
        pk[ij] = m
    if bad.any():
        return np.nan
    return pk.max()


@_jit
def rk_stage(K, ig, Aw, acc, tmp, ch, w, first):
    """One RK4 stage after ft_t: k = ig*K; acc (=|+=) w*k; tmp = Aw + ch*k."""
    nx, ny, nt = K.shape
    for ij in nb.prange(nx * ny):
        i, j = ij // ny, ij % ny
        for k in range(nt):
            kk = K[i, j, k] * ig[k]
            if first:
                acc[i, j, k] = kk * w
            else:
                acc[i, j, k] += kk * w
            tmp[i, j, k] = Aw[i, j, k] + kk * ch


@_jit
def rk_final(K, ig, Aw, acc, h6):
    """Last RK4 stage: Aw += h/6 (acc + ig*K)."""
    nx, ny, nt = K.shape
    for ij in nb.prange(nx * ny):
        i, j = ij // ny, ij % ny
        for k in range(nt):
            Aw[i, j, k] += (acc[i, j, k] + K[i, j, k] * ig[k]) * h6


@_jit
def mul_inplace(A, P):
    """A *= P (same shape)."""
    nx, ny, nt = A.shape
    for ij in nb.prange(nx * ny):
        i, j = ij // ny, ij % ny
        for k in range(nt):
            A[i, j, k] *= P[i, j, k]


@_jit
def mul_separable(A, fx, fy, fw):
    """A[i, j, k] *= fx[i, k] fy[j, k] fw[k] (lens-frame kinetic and potential factors)."""
    nx, ny, nt = A.shape
    for ij in nb.prange(nx * ny):
        i, j = ij // ny, ij % ny
        for k in range(nt):
            A[i, j, k] *= fx[i, k] * fy[j, k] * fw[k]


@_jit
def exp_i(D, ih, out):
    """out = exp(ih D), ih = 1j*h in the dtype of D (UPPE propagator, evanescent part decays)."""
    nx, ny, nt = D.shape
    for ij in nb.prange(nx * ny):
        i, j = ij // ny, ij % ny
        for k in range(nt):
            out[i, j, k] = np.exp(D[i, j, k] * ih)
