"""Mosaic / composite / stack built from the ROIs of ONE scene (no second scene needed).

Each selected ROI is cut out of the scene as a patch (its bounding box). Pixels outside the ROI shape are NaN
(outside="nan") or kept (outside="bbox"). The patches are then

* mosaic  - placed side by side in a new scene (one row, or a grid of N columns, with a gap of NaN pixels), or
            kept at their original place in the scene (placement="in_place": same size and georeferencing as the
            scene, everything outside the ROIs is NaN);
* composite - centred on one canvas and collapsed pixel by pixel (mean | median | max | min);
* stack   - centred on one canvas and kept as a 4-D array (n_rois, rows, cols, bands), e.g. for model training.

The class of every ROI is carried over as a label map, so the result can go straight to Aggregation.
Bad bands stay flagged (same settings as the source scene). No GUI code here."""
from __future__ import annotations

import warnings
from typing import List, Sequence

import numpy as np

from .rois import roi_bbox, roi_mask
from .scene import Scene

PLACEMENTS = ("side", "in_place")
OUTSIDE = ("nan", "bbox")


def _check(scene: Scene, rois: Sequence[dict]):
    if scene is None:
        raise ValueError("choose a scene")
    if not rois:
        raise ValueError("tick at least one ROI of the scene")
    ids = {r["id"] for r in scene.rois}
    for r in rois:
        if r["id"] not in ids:
            raise ValueError(f"ROI '{r['name']}' does not belong to scene '{scene.name}'")


def roi_patches(scene: Scene, rois: Sequence[dict], outside: str = "nan") -> List[dict]:
    """[{roi, r0, c0, cube (h, w, bands) float32, mask (h, w) bool}] in the order of *rois*; empty ROIs skipped."""
    if outside not in OUTSIDE:
        raise ValueError(f"unknown outside mode {outside!r}")
    _check(scene, rois)
    rows, cols, _ = scene.cube.shape
    out = []
    for r in rois:
        bb = roi_bbox(r, rows, cols)
        if bb is None:
            continue
        r0, r1, c0, c1 = bb
        m = roi_mask(r, rows, cols)[r0:r1 + 1, c0:c1 + 1]
        cube = np.array(scene.cube[r0:r1 + 1, c0:c1 + 1], dtype=np.float32, copy=True)
        if outside == "nan":
            cube[~m] = np.nan
        out.append({"roi": r, "r0": r0, "c0": c0, "cube": cube, "mask": m})
    if not out:
        raise ValueError("the ticked ROIs contain no pixels")
    return out


def _result(scene: Scene, name: str, cube: np.ndarray, geo, labels=None, class_names=None) -> Scene:
    res = Scene(name, np.ascontiguousarray(cube, np.float32),
                None if scene.wavelengths is None else np.array(scene.wavelengths, float), geo,
                unit=scene.unit, unit_note=f"from scene '{scene.name}'", synthetic=scene.synthetic)
    res.bad_file = None if scene.bad_file is None else np.array(scene.bad_file, copy=True)
    res.mask_water, res.mask_empty, res.water_windows = scene.mask_water, scene.mask_empty, scene.water_windows
    if labels is not None and labels.any():
        res.labels = labels
        res.class_names = dict(class_names or {})
    return res


def _class_ids(patches):
    names = []
    for p in patches:
        if p["roi"]["class"] not in names:
            names.append(p["roi"]["class"])
    return names, {i + 1: n for i, n in enumerate(names)}


def roi_mosaic(scene: Scene, rois: Sequence[dict], placement: str = "side", gap: int = 2, columns: int = 0,
               outside: str = "nan", name: str = "roi_mosaic") -> Scene:
    """New scene made of the ROI patches of *scene* (see the module docstring)."""
    if placement not in PLACEMENTS:
        raise ValueError(f"unknown placement {placement!r}")
    patches = roi_patches(scene, rois, outside)
    names, cls = _class_ids(patches)
    nb = scene.n_bands
    if placement == "in_place":
        rows, cols, _ = scene.cube.shape
        cube = np.full((rows, cols, nb), np.nan, np.float32)
        lab = np.zeros((rows, cols), np.int32)
        for p in patches:                                        # later ROIs win where ROIs overlap
            h, w = p["mask"].shape
            win = (slice(p["r0"], p["r0"] + h), slice(p["c0"], p["c0"] + w))
            keep = p["mask"] if outside == "nan" else np.ones_like(p["mask"])
            cube[win][keep] = p["cube"][keep]
            lab[win][p["mask"]] = names.index(p["roi"]["class"]) + 1
        return _result(scene, name, cube, scene.geo, lab, cls)
    gap = max(0, int(gap))
    ncol = len(patches) if int(columns or 0) <= 0 else max(1, int(columns))
    grid_rows = [patches[i:i + ncol] for i in range(0, len(patches), ncol)]
    row_h = [max(p["mask"].shape[0] for p in gr) for gr in grid_rows]
    row_w = [sum(p["mask"].shape[1] for p in gr) + gap * (len(gr) - 1) for gr in grid_rows]
    H = sum(row_h) + gap * (len(grid_rows) - 1)
    Wd = max(row_w)
    cube = np.full((H, Wd, nb), np.nan, np.float32)
    lab = np.zeros((H, Wd), np.int32)
    y = 0
    for gr, rh in zip(grid_rows, row_h):
        x = 0
        for p in gr:
            h, w = p["mask"].shape
            cube[y:y + h, x:x + w] = p["cube"]
            lab[y:y + h, x:x + w][p["mask"]] = names.index(p["roi"]["class"]) + 1
            p["placed_at"] = (y, x)
            x += w + gap
        y += rh + gap
    return _result(scene, name, cube, None, lab, cls)        # patches are no longer contiguous on the map


def roi_stack(scene: Scene, rois: Sequence[dict], outside: str = "nan"):
    """(array (n, H, W, bands) float32, masks (n, H, W) bool, meta list). Patches are centred on an H x W canvas
    (H, W = the largest ROI), NaN around them. meta = [{name, class, row0, col0, rows, cols, n_pixels}]."""
    patches = roi_patches(scene, rois, outside)
    H = max(p["mask"].shape[0] for p in patches)
    Wd = max(p["mask"].shape[1] for p in patches)
    arr = np.full((len(patches), H, Wd, scene.n_bands), np.nan, np.float32)
    masks = np.zeros((len(patches), H, Wd), bool)
    meta = []
    for i, p in enumerate(patches):
        h, w = p["mask"].shape
        oy, ox = (H - h) // 2, (Wd - w) // 2
        arr[i, oy:oy + h, ox:ox + w] = p["cube"]
        masks[i, oy:oy + h, ox:ox + w] = p["mask"]
        r = p["roi"]
        meta.append({"name": r["name"], "class": r["class"], "row0": p["r0"], "col0": p["c0"], "rows": h, "cols": w,
                     "n_pixels": int(p["mask"].sum())})
    return arr, masks, meta


def roi_composite(scene: Scene, rois: Sequence[dict], method: str = "mean", outside: str = "nan",
                  name: str = "roi_composite") -> Scene:
    """One cube: the centred ROI patches collapsed pixel by pixel (mean | median | max | min), NaN ignored."""
    arr, masks, _meta = roi_stack(scene, rois, outside)
    fn = {"mean": np.nanmean, "median": np.nanmedian, "max": np.nanmax, "min": np.nanmin}[method]
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        out = fn(arr, axis=0).astype(np.float32)
    classes = {r["class"] for r in rois}
    lab, cls = None, None
    if len(classes) == 1:                                        # one class: the composite keeps it
        lab = masks.any(axis=0).astype(np.int32)
        cls = {1: next(iter(classes))}
    return _result(scene, name, out, None, lab, cls)
