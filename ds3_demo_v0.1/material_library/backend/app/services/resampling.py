"""Linear resampling of one spectrum onto a wavelength grid (nm). Shared by Library resampling and the Builder.

Rules: no extrapolation; a gap in the source wider than 3x its median spacing is not bridged; the caller decides
how much of the grid a spectrum has to cover."""
from __future__ import annotations
import numpy as np

WATER_NM = [(1340, 1460), (1790, 1960)]          # strong water-vapour absorption


def without_water_bands(grid):
    grid = np.asarray(grid, float)
    return grid[[not any(lo <= g <= hi for lo, hi in WATER_NM) for g in grid]]


def resample_spectrum(w, v, grid):
    """(values on grid with NaN where there is no data, coverage 0..1). Raises ValueError for unusable input."""
    w = np.asarray(w, float); v = np.asarray([np.nan if x is None else x for x in v], float)
    if w.size != v.size:
        raise ValueError("values and wavelengths differ in length")
    ok = np.isfinite(v) & np.isfinite(w)
    if ok.sum() < 2:
        raise ValueError("fewer than 2 valid points")
    w, v = w[ok], v[ok]
    o = np.argsort(w); w, v = w[o], v[o]
    if np.any(np.diff(w) <= 0):
        raise ValueError("wavelengths must be strictly increasing")
    grid = np.asarray(grid, float)
    inside = (grid >= w[0]) & (grid <= w[-1])
    gapmax = 3 * float(np.median(np.diff(w))) if len(w) > 2 else np.inf
    r = np.full(grid.size, np.nan)
    r[inside] = np.interp(grid[inside], w, v)
    idx = np.searchsorted(w, grid[inside])
    left = w[np.clip(idx - 1, 0, len(w) - 1)]; right = w[np.clip(idx, 0, len(w) - 1)]
    tmp = r[inside]; tmp[(right - left) > gapmax] = np.nan; r[inside] = tmp
    return r, float(np.isfinite(r).mean())
