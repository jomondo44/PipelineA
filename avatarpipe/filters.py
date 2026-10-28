"""Signal cleanup for tracking data: gap filling + One Euro filter (Casiez et al. 2012)."""
from __future__ import annotations

import math

import numpy as np


def fill_gaps(x: np.ndarray) -> np.ndarray:
    """Linearly interpolate NaN runs along axis 0; hold first/last valid value at the ends.

    x: (T, ...) float array. Returns a copy. If nothing is valid, returns zeros.
    """
    x = np.array(x, dtype=np.float64, copy=True)
    T = x.shape[0]
    flat = x.reshape(T, -1)
    t = np.arange(T)
    for j in range(flat.shape[1]):
        col = flat[:, j]
        ok = np.isfinite(col)
        if not ok.any():
            col[:] = 0.0
            continue
        if ok.all():
            continue
        col[:] = np.interp(t, t[ok], col[ok])
    return flat.reshape(x.shape)


def _alpha(cutoff: float, dt: float) -> float:
    tau = 1.0 / (2 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


def one_euro(x: np.ndarray, fps: float, min_cutoff=1.0, beta=0.0, d_cutoff=1.0) -> np.ndarray:
    """Zero-lag-ish adaptive low-pass, applied independently to every channel of x (T, ...).

    Low speed -> heavy smoothing (kills jitter); high speed -> light smoothing (keeps snap).
    """
    x = np.asarray(x, dtype=np.float64)
    if x.shape[0] < 2 or fps <= 0:
        return x.copy()
    dt = 1.0 / fps
    out = np.empty_like(x)
    out[0] = x[0]
    dx_prev = np.zeros_like(x[0])
    a_d = _alpha(d_cutoff, dt)
    for i in range(1, x.shape[0]):
        dx = (x[i] - out[i - 1]) / dt
        dx_hat = a_d * dx + (1 - a_d) * dx_prev
        cutoff = min_cutoff + beta * np.abs(dx_hat)
        tau = 1.0 / (2 * np.pi * cutoff)
        a = 1.0 / (1.0 + tau / dt)
        out[i] = a * x[i] + (1 - a) * out[i - 1]
        dx_prev = dx_hat
    return out


def one_euro_bidirectional(x: np.ndarray, fps: float, **kw) -> np.ndarray:
    """Offline use: run forward and backward and average, cancelling the phase lag."""
    fwd = one_euro(x, fps, **kw)
    bwd = one_euro(x[::-1], fps, **kw)[::-1]
    return 0.5 * (fwd + bwd)
