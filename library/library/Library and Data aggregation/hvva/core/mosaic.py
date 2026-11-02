"""Mosaic (spatial merge) and stack/composite (same area, several dates) of scenes."""
from __future__ import annotations

import numpy as np

from .geo import GeoInfo
from .resample import resample_cube
from .scene import Scene


def _on_grid(scenes, grid):
    out = []
    for s in scenes:
        if grid is None:
            out.append(s.cube)
        else:
            out.append(resample_cube(s.cube, s.wavelengths, grid))
    return out


def mosaic(scenes, grid=None, blend="mean", name="mosaic") -> Scene:
    """Place scenes side by side on one pixel grid.

    * Georeferenced scenes (same CRS and pixel size) are placed by their map coordinates;
      overlaps are blended (blend = mean | first | last | max).
    * Scenes without georeferencing are laid out left-to-right (flagged as non-georeferenced).
    """
    if not scenes:
        raise ValueError("no scenes")
    cubes = _on_grid(scenes, grid)
    nb = cubes[0].shape[2]
    geos = [s.geo for s in scenes]
    if all(g is not None for g in geos):
        g0 = geos[0]
        xs0, ys0 = g0.pixel_size
        for s, g in zip(scenes, geos):
            if not g.same_crs(g0):
                raise ValueError(f"CRS mismatch: '{s.name}' is {g.crs}, expected {g0.crs}")
            if not g.is_north_up:
                raise ValueError(f"'{s.name}' is rotated; only north-up scenes can be mosaicked")
            px, py = g.pixel_size
            if abs(px - xs0) > 1e-6 * xs0 or abs(py - ys0) > 1e-6 * ys0:
                raise ValueError(f"pixel size mismatch: '{s.name}' has {px:g} x {py:g}, expected {xs0:g} x {ys0:g}")
        bounds = [g.bounds(*c.shape[:2]) for g, c in zip(geos, cubes)]
        X0, Y1 = min(b[0] for b in bounds), max(b[3] for b in bounds)
        W = int(round((max(b[2] for b in bounds) - X0) / xs0))
        H = int(round((Y1 - min(b[1] for b in bounds)) / ys0))
        offs = [(int(round((Y1 - b[3]) / ys0)), int(round((b[0] - X0) / xs0))) for b in bounds]
        geo = GeoInfo(g0.crs, (xs0, 0.0, X0, 0.0, -ys0, Y1))
    else:
        H = max(c.shape[0] for c in cubes)
        W = sum(c.shape[1] for c in cubes)
        offs, x = [], 0
        for c in cubes:
            offs.append((0, x))
            x += c.shape[1]
        geo = None
    acc = np.zeros((H, W, nb), np.float64)
    cnt = np.zeros((H, W, nb), np.float64)
    best = np.full((H, W, nb), -np.inf)
    out = np.full((H, W, nb), np.nan, np.float32)
    for c, (r0, c0) in zip(cubes, offs):
        h, w, _ = c.shape
        win = (slice(r0, r0 + h), slice(c0, c0 + w))
        valid = ~np.isnan(c)
        if blend == "mean":
            acc[win] += np.where(valid, c, 0.0)
            cnt[win] += valid
        elif blend == "max":
            best[win] = np.fmax(best[win], np.where(valid, c, -np.inf))
        elif blend in ("first", "last"):
            tgt = out[win]
            take = valid & (np.isnan(tgt) if blend == "first" else True)
            tgt[take] = c[take]
            out[win] = tgt
        else:
            raise ValueError(f"unknown blend {blend!r}")
    if blend == "mean":
        with np.errstate(invalid="ignore", divide="ignore"):
            out = np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan).astype(np.float32)
    elif blend == "max":
        out = np.where(np.isfinite(best), best, np.nan).astype(np.float32)
    wl = None if grid is None else np.asarray(grid, float)
    if wl is None and scenes[0].has_wavelengths and all(
            s.has_wavelengths and np.array_equal(s.wavelengths, scenes[0].wavelengths) for s in scenes):
        wl = scenes[0].wavelengths
    res = Scene(name, out, wl, geo)
    # carry labels over where scenes share class names (ids re-mapped by name)
    names = sorted({n for s in scenes for n in s.class_masks()})
    if names:
        lab = np.zeros((H, W), np.int32)
        for s, (r0, c0) in zip(scenes, offs):
            for cname, m in s.class_masks().items():
                h, w = m.shape
                lab[r0:r0 + h, c0:c0 + w][m] = names.index(cname) + 1
        res.labels = lab
        res.class_names = {i + 1: n for i, n in enumerate(names)}
    return res


def _common_window(scenes):
    """(r0, c0, h, w) of each scene covering the area every scene has (georeferenced), else top-left crop."""
    geos = [s.geo for s in scenes]
    if all(g is not None for g in geos) and all(g.same_crs(geos[0]) and g.is_north_up for g in geos):
        b = [g.bounds(*s.shape[:2]) for g, s in zip(geos, scenes)]
        xmin, xmax = max(v[0] for v in b), min(v[2] for v in b)
        ymin, ymax = max(v[1] for v in b), min(v[3] for v in b)
        if xmin >= xmax or ymin >= ymax:
            raise ValueError("scenes do not overlap")
        xs, ys = geos[0].pixel_size
        W, H = int((xmax - xmin) / xs), int((ymax - ymin) / ys)
        wins = []
        for g, s, bb in zip(geos, scenes, b):
            wins.append((int(round((bb[3] - ymax) / ys)), int(round((xmin - bb[0]) / xs)), H, W))
        return wins, geos[0].shifted(0, 0) and GeoInfo(geos[0].crs, (xs, 0.0, xmin, 0.0, -ys, ymax))
    H = min(s.shape[0] for s in scenes)
    W = min(s.shape[1] for s in scenes)
    return [(0, 0, H, W)] * len(scenes), None


def stack_cube(scenes, grid=None):
    """(n_scenes, rows, cols, bands) array of the common area + the matching GeoInfo (or None)."""
    wins, geo = _common_window(scenes)
    cubes = _on_grid(scenes, grid)
    arr = np.stack([c[r0:r0 + h, c0:c0 + w] for c, (r0, c0, h, w) in zip(cubes, wins)])
    return arr, geo


def composite(scenes, grid=None, method="mean", name="composite") -> Scene:
    """Collapse the stack along the scene axis (mean | median | max | min), ignoring NaN."""
    arr, geo = stack_cube(scenes, grid)
    fn = {"mean": np.nanmean, "median": np.nanmedian, "max": np.nanmax, "min": np.nanmin}[method]
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            out = fn(arr, axis=0).astype(np.float32)
    wl = None if grid is None else np.asarray(grid, float)
    if wl is None and scenes[0].has_wavelengths:
        wl = scenes[0].wavelengths
    return Scene(name, out, wl, geo)
