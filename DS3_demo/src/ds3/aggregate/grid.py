"""Common wavelength grid and linear resampling. All wavelengths are in nm; nothing is ever extrapolated."""
from __future__ import annotations
import numpy as np


def median_step(wl) -> float:
    wl = np.sort(np.asarray(wl, float))
    return float(np.median(np.diff(wl))) if wl.size > 1 else 1.0


def common_grid(wl_list, mode: str = "intersection", step: float | None = None) -> np.ndarray:
    """Shared wavelength vector of several scenes.

    mode  'intersection'  only the range every scene covers (default; no gaps)
          'union'         the full range; a scene has no data (NaN) where it does not reach
    step  spacing in nm; default is the coarsest native spacing, so no detail is invented.
    """
    wls = [np.asarray(w, float) for w in wl_list if w is not None and len(w) > 1]
    if not wls:
        raise ValueError("no scene has wavelengths")
    lo, hi = [w.min() for w in wls], [w.max() for w in wls]
    if mode == "intersection":
        start, stop = max(lo), min(hi)
        if start >= stop:
            raise ValueError("the scenes have no wavelength range in common (use mode 'union' to keep all of it)")
    elif mode == "union":
        start, stop = min(lo), max(hi)
    else:
        raise ValueError(f"unknown grid mode {mode!r} (use 'intersection' or 'union')")
    step = float(step) if step else max(median_step(w) for w in wls)
    if step <= 0:
        raise ValueError("step must be positive")
    n = int(np.floor((stop - start) / step + 1e-9)) + 1
    return start + step * np.arange(n)


def interp_weights(wl_src, wl_dst):
    """Pre-computed linear interpolation from wl_src (any order) to wl_dst: (order, i0, i1, t, outside)."""
    src = np.asarray(wl_src, float)
    dst = np.asarray(wl_dst, float)
    order = np.argsort(src)
    s = src[order]
    if s.size < 2:
        raise ValueError("need at least two source wavelengths")
    j = np.clip(np.searchsorted(s, dst), 1, s.size - 1)
    i0, i1 = j - 1, j
    t = np.clip((dst - s[i0]) / (s[i1] - s[i0]), 0.0, 1.0)
    tol = 1e-6 * (s[-1] - s[0])
    outside = (dst < s[0] - tol) | (dst > s[-1] + tol)
    return order, i0, i1, t, outside


def resample_rows(X, weights) -> np.ndarray:
    """Resample the rows of X (N, bands) with weights from `interp_weights`; NaN outside the source range."""
    order, i0, i1, t, outside = weights
    X = np.asarray(X, np.float64)[:, order]
    out = X[:, i0] * (1 - t) + X[:, i1] * t
    out[:, outside] = np.nan
    return out
