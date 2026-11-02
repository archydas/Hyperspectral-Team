"""Bring scenes onto a common wavelength grid (all wavelengths in nm)."""
from __future__ import annotations

import numpy as np


def median_step(wl):
    wl = np.sort(np.asarray(wl, float))
    return float(np.median(np.diff(wl))) if wl.size > 1 else 1.0


def common_grid(wl_list, mode="intersection", step=None):
    """Return the shared wavelength vector.

    mode  'intersection' - only the range every scene covers (no extrapolation)
          'union'        - the full range; scenes get NaN where they have no data
    step  grid spacing in nm; default = the coarsest native spacing (never invents detail).
    """
    wl_list = [np.asarray(w, float) for w in wl_list if w is not None]
    if not wl_list:
        raise ValueError("no scene has wavelengths")
    mins = [w.min() for w in wl_list]
    maxs = [w.max() for w in wl_list]
    if mode == "intersection":
        start, stop = max(mins), min(maxs)
        if start >= stop:
            raise ValueError("scenes have no overlapping wavelength range")
    elif mode == "union":
        start, stop = min(mins), max(maxs)
    else:
        raise ValueError(f"unknown grid mode {mode!r}")
    if step is None:
        step = max(median_step(w) for w in wl_list)
    n = int(np.floor((stop - start) / step + 1e-9)) + 1
    return start + step * np.arange(n)


def resample_spectra(X, wl_src, wl_dst):
    """Linear interpolation of spectra X (N, B) from wl_src to wl_dst; NaN outside the range."""
    X = np.asarray(X, np.float32)
    wl_src = np.asarray(wl_src, float)
    wl_dst = np.asarray(wl_dst, float)
    order = np.argsort(wl_src)
    wl_src, X = wl_src[order], X[:, order]
    if wl_src.size < 2:
        raise ValueError("need at least two source wavelengths")
    j = np.clip(np.searchsorted(wl_src, wl_dst), 1, wl_src.size - 1)
    i0, i1 = j - 1, j
    t = np.clip((wl_dst - wl_src[i0]) / (wl_src[i1] - wl_src[i0]), 0.0, 1.0).astype(np.float32)
    out = X[:, i0] * (1 - t) + X[:, i1] * t
    tol = 1e-6 * (wl_src[-1] - wl_src[0])
    out[:, (wl_dst < wl_src[0] - tol) | (wl_dst > wl_src[-1] + tol)] = np.nan
    return out


def resample_cube(cube, wl_src, wl_dst):
    h, w, b = cube.shape
    return resample_spectra(cube.reshape(-1, b), wl_src, wl_dst).reshape(h, w, -1)
