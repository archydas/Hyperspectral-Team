"""Regions of interest (ROIs) - ported from the DS3 viewer's ROI Analysis / ROI Manager, made multi-image.

No GUI code here. A ROI is a plain dict stored in Scene.rois:

    {"id":    "3f2a...",            stable id (uuid hex) - also the primary key of the rois table in the database
     "name":  "ROI 4",
     "class": "vegetation",         class name; classes are matched across images by this name
     "kind":  "rect" | "polygon",
     "rect":  (r0, c0, r1, c1),     rect only: row/col range, end-exclusive (HyperVista convention)
     "vertices": [(x, y), ...]}     polygon only: column, row of each vertex (pixel-centre coordinates)

Masks are always computed from the geometry (never stored), so a ROI survives being saved, reloaded, moved
between images of the same size or re-projected through map coordinates.  Mask rules are identical to the
viewer's rect_mask / polygon_mask, so ROIs give the same pixels in both tools.

Files: to_ds3_json / from_ds3_json read and write the viewer's "DS3-ROIs" format (version 2 adds the scene name
and id of every ROI; the viewer reads version-2 files too because it only uses name / class / kind / geometry).
"""
from __future__ import annotations

import uuid
import warnings
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from matplotlib.path import Path as MplPath

from .resample import common_grid, resample_spectra

KINDS = ("rect", "polygon")


# ================================================================== geometry
def new_id() -> str:
    return uuid.uuid4().hex


def normalize_roi(roi: dict, counter: Optional[int] = None) -> dict:
    """Fill defaults (id, name, kind) and coerce geometry types; returns a new dict."""
    r = dict(roi)
    r["id"] = str(r.get("id") or new_id())
    r["class"] = str(r.get("class") or "class_1").strip() or "class_1"
    kind = r.get("kind") or ("polygon" if r.get("vertices") else "rect")
    if kind not in KINDS:
        raise ValueError(f"unknown ROI kind {kind!r}")
    r["kind"] = kind
    if kind == "rect":
        r0, c0, r1, c1 = (int(v) for v in r["rect"])
        r["rect"] = (min(r0, r1), min(c0, c1), max(r0, r1), max(c0, c1))
        r.pop("vertices", None)
    else:
        r["vertices"] = [(float(x), float(y)) for x, y in r["vertices"]]
        if len(r["vertices"]) < 3:
            raise ValueError("a polygon ROI needs at least 3 vertices")
        r.pop("rect", None)
    r["name"] = str(r.get("name") or (f"ROI {counter}" if counter else f"ROI {r['id'][:6]}"))
    r.pop("mask", None)
    return r


def polygon_mask(vertices, rows: int, cols: int) -> np.ndarray:
    """Pixel centres inside a polygon [(x, y), ...] - same rule as the viewer."""
    verts = np.asarray(vertices, dtype=float)
    mask = np.zeros((rows, cols), dtype=bool)
    if verts.shape[0] < 3:
        return mask
    x0 = max(0, int(np.floor(verts[:, 0].min())))
    x1 = min(cols - 1, int(np.ceil(verts[:, 0].max())))
    y0 = max(0, int(np.floor(verts[:, 1].min())))
    y1 = min(rows - 1, int(np.ceil(verts[:, 1].max())))
    if x1 < x0 or y1 < y0:
        return mask
    xx, yy = np.meshgrid(np.arange(x0, x1 + 1), np.arange(y0, y1 + 1))
    inside = MplPath(verts).contains_points(np.column_stack([xx.ravel(), yy.ravel()]))
    mask[y0:y1 + 1, x0:x1 + 1] = inside.reshape(yy.shape)
    return mask


def roi_mask(roi: dict, rows: int, cols: int) -> np.ndarray:
    if roi.get("kind", "rect") == "polygon":
        return polygon_mask(roi["vertices"], rows, cols)
    r0, c0, r1, c1 = roi["rect"]
    m = np.zeros((rows, cols), dtype=bool)
    m[max(r0, 0):min(r1, rows), max(c0, 0):min(c1, cols)] = True
    return m


def roi_bbox(roi: dict, rows: int, cols: int) -> Optional[Tuple[int, int, int, int]]:
    """(row_min, row_max, col_min, col_max), inclusive, of the pixels inside the ROI (None if empty)."""
    ys, xs = np.nonzero(roi_mask(roi, rows, cols))
    if ys.size == 0:
        return None
    return int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())


def describe_geometry(roi: dict) -> str:
    if roi.get("kind") == "polygon":
        return f"polygon, {len(roi['vertices'])} vertices"
    r0, c0, r1, c1 = roi["rect"]
    return f"rect rows {r0}-{r1 - 1}, cols {c0}-{c1 - 1}"


def outline(roi: dict) -> np.ndarray:
    """Outline vertices (x, y) in display coordinates (pixel edges for rectangles)."""
    if roi.get("kind") == "polygon":
        return np.asarray(roi["vertices"], float)
    r0, c0, r1, c1 = roi["rect"]
    return np.array([(c0 - .5, r0 - .5), (c1 - .5, r0 - .5), (c1 - .5, r1 - .5), (c0 - .5, r1 - .5)], float)


# ================================================================== spectra / statistics
def roi_pixels(cube: np.ndarray, roi: dict) -> np.ndarray:
    """(N, bands) spectra of the ROI's pixels that are not entirely NaN."""
    h, w, _ = cube.shape
    X = cube[roi_mask(roi, h, w)].astype(float)
    return X[~np.all(np.isnan(X), axis=1)] if X.size else X.reshape(0, cube.shape[2])


def roi_stats(cube: np.ndarray, roi: dict) -> dict:
    X = roi_pixels(cube, roi)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean = np.nanmean(X, axis=0) if len(X) else np.full(cube.shape[2], np.nan)
        std = np.nanstd(X, axis=0, ddof=1) if len(X) > 1 else np.full(cube.shape[2], np.nan)
    return {"n_pixels": int(len(X)), "mean": mean, "std": std}


def class_order(rois: Iterable[dict]) -> List[str]:
    names: List[str] = []
    for r in rois:
        if r["class"] not in names:
            names.append(r["class"])
    return names


def pool_classes(items: Sequence[Tuple[object, dict]], grid=None, max_pixels: Optional[int] = None,
                 seed: int = 0) -> Dict[str, dict]:
    """Pool the pixels of every class over all its ROIs, in one or several images.

    items: [(scene, roi), ...] (scene needs .cube, .wavelengths, .name).  Overlapping ROIs of the same class in
    the same image count their shared pixels once.  When the images' wavelengths differ, every spectrum is
    resampled to *grid* (default: the common intersection grid of the images involved).
    Returns {class: {wavelengths, mean, std, n_pixels, n_rois, n_scenes, roi_ids, scenes}}.
    """
    by_class: Dict[str, Dict[int, list]] = {}
    scenes = {}
    for sc, roi in items:
        scenes[id(sc)] = sc
        by_class.setdefault(roi["class"], {}).setdefault(id(sc), []).append(roi)
    wls = [s.wavelengths for s in scenes.values()]
    same = all(w is not None for w in wls) and all(len(w) == len(wls[0]) and np.allclose(w, wls[0]) for w in wls)
    if grid is None and not same:
        if any(w is None for w in wls):
            raise ValueError("images without wavelengths cannot be pooled with others - assign wavelengths first")
        grid = common_grid(wls, "intersection")
    out: Dict[str, dict] = {}
    rng = np.random.default_rng(seed)
    for cname, per_scene in by_class.items():
        chunks, roi_ids, names = [], [], []
        for sid, rois in per_scene.items():
            sc = scenes[sid]
            h, w, b = sc.cube.shape
            m = np.zeros((h, w), bool)
            for r in rois:
                m |= roi_mask(r, h, w)
                roi_ids.append(r["id"])
            X = sc.cube[m].astype(float)
            X = X[~np.all(np.isnan(X), axis=1)]
            if max_pixels and len(X) > max_pixels:
                X = X[rng.choice(len(X), max_pixels, replace=False)]
            if grid is not None and not same:
                X = resample_spectra(X, sc.wavelengths, grid)
            if len(X):
                chunks.append(X)
                names.append(sc.name)
        if not chunks:
            continue
        X = np.vstack(chunks)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mean = np.nanmean(X, axis=0)
            std = np.nanstd(X, axis=0, ddof=1) if len(X) > 1 else np.full(X.shape[1], np.nan)
        wl = np.asarray(grid if (grid is not None and not same) else wls[0] if wls[0] is not None
                        else np.arange(X.shape[1]), float)
        out[cname] = {"wavelengths": wl, "mean": mean, "std": std, "n_pixels": int(len(X)),
                      "n_rois": len(roi_ids), "n_scenes": len(names), "roi_ids": roi_ids, "scenes": names}
    return out


def anomaly_map(cube: np.ndarray) -> np.ndarray:
    """L2 distance of every pixel from the global mean spectrum (the viewer's anomaly map)."""
    rows, cols, bands = cube.shape
    data = cube.reshape(-1, bands).astype(float)
    valid_px = ~np.all(np.isnan(data), axis=1)
    if not valid_px.any():
        raise ValueError("all pixels are NaN")
    d = data[valid_px]
    good = ~np.all(np.isnan(d), axis=0)
    d = d[:, good]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mu = np.nanmean(d, axis=0, keepdims=True)
    d = np.where(np.isnan(d), mu, d)
    out = np.full(rows * cols, np.nan)
    out[valid_px] = np.sqrt(((d - mu) ** 2).sum(axis=1))
    return out.reshape(rows, cols)


# ================================================================== map coordinates (georeferenced images)
def geometry_to_map(roi: dict, geo) -> Optional[dict]:
    """ROI outline in map coordinates, so it can be placed on another image of the same area."""
    if geo is None:
        return None
    if roi["kind"] == "rect":
        r0, c0, r1, c1 = roi["rect"]
        x0, y0 = geo.xy(r0, c0, center=False)
        x1, y1 = geo.xy(r1, c1, center=False)
        return {"crs": geo.crs, "kind": "rect", "corners": [[x0, y0], [x1, y1]]}
    pts = [list(geo.xy(y, x)) for x, y in roi["vertices"]]
    return {"crs": geo.crs, "kind": "polygon", "vertices": pts}


def geometry_from_map(gmap: dict, geo, rows: int, cols: int) -> Optional[dict]:
    """Pixel geometry on an image with *geo* from geometry_to_map() output (None if CRS differs / outside)."""
    if not gmap or geo is None or (gmap.get("crs") and geo.crs and str(gmap["crs"]) != str(geo.crs)):
        return None
    if gmap["kind"] == "rect":
        (x0, y0), (x1, y1) = gmap["corners"]
        ra, ca = geo.rowcol(x0, y0)
        rb, cb = geo.rowcol(x1, y1)
        r0, r1 = sorted((int(round(ra)), int(round(rb))))
        c0, c1 = sorted((int(round(ca)), int(round(cb))))
        r0, c0, r1, c1 = max(r0, 0), max(c0, 0), min(r1, rows), min(c1, cols)
        if r1 <= r0 or c1 <= c0:
            return None
        return {"kind": "rect", "rect": (r0, c0, r1, c1)}
    verts = []
    for x, y in gmap["vertices"]:
        r, c = geo.rowcol(x, y)                                  # edge-based -> pixel-centre coordinates
        verts.append((float(c) - .5, float(r) - .5))
    if not polygon_mask(verts, rows, cols).any():
        return None
    return {"kind": "polygon", "vertices": verts}


# ================================================================== files (viewer-compatible)
def to_ds3_json(items: Sequence[Tuple[object, dict]]) -> dict:
    """DS3-ROIs v2. rect is written in the viewer's convention [ymin, ymax, xmin, xmax] (inclusive)."""
    shapes = {tuple(sc.cube.shape[:2]) for sc, _ in items}
    data = {"format": "DS3-ROIs", "version": 2, "rois": []}
    if len(shapes) == 1:
        data["image_shape"] = list(next(iter(shapes)))
    for sc, roi in items:
        it = {"id": roi["id"], "name": roi["name"], "class": roi["class"], "kind": roi["kind"],
              "scene": sc.name, "image_shape": list(sc.cube.shape[:2])}
        if roi["kind"] == "rect":
            r0, c0, r1, c1 = roi["rect"]
            it["rect"] = [int(r0), int(r1 - 1), int(c0), int(c1 - 1)]
        else:
            it["vertices"] = [[float(x), float(y)] for x, y in roi["vertices"]]
        gm = geometry_to_map(roi, getattr(sc, "geo", None))
        if gm:
            it["map"] = gm
        data["rois"].append(it)
    return data


def from_ds3_json(data: dict, rows: int, cols: int) -> Tuple[List[Tuple[Optional[str], dict]], int]:
    """[(scene name or None, roi), ...] clipped to a rows x cols image, plus the number skipped."""
    if not isinstance(data, dict) or "rois" not in data:
        raise ValueError("not a DS3-ROIs file")
    out, skipped = [], 0
    for k, item in enumerate(data["rois"], 1):
        try:
            if item["kind"] == "rect":
                ymin, ymax, xmin, xmax = (int(v) for v in item["rect"])
                ymin, ymax = max(0, ymin), min(rows - 1, ymax)
                xmin, xmax = max(0, xmin), min(cols - 1, xmax)
                if ymax < ymin or xmax < xmin:
                    skipped += 1
                    continue
                roi = {"kind": "rect", "rect": (ymin, xmin, ymax + 1, xmax + 1)}
            else:
                roi = {"kind": "polygon", "vertices": [(float(x), float(y)) for x, y in item["vertices"]]}
            roi.update({"id": item.get("id"), "name": item.get("name") or f"ROI {k}",
                        "class": item.get("class") or "class_1"})
            roi = normalize_roi(roi)
            if not roi_mask(roi, rows, cols).any():
                skipped += 1
                continue
            out.append((item.get("scene"), roi))
        except (KeyError, TypeError, ValueError):
            skipped += 1
    return out, skipped


# ================================================================== database records (library tool)
LIBRARY_MODES = {
    "class": "Class mean ± std, pooled over all ROIs of the class (one spectrum per class)",
    "roi": "One mean ± std per ROI (each ROI a sample of its class material)",
    "pixels": "Individual pixel spectra of each ROI (random subset, capped per ROI)",
}


def _lst(a) -> list:
    return [None if not np.isfinite(v) else float(v) for v in np.asarray(a, float).ravel()]


def _geom(roi: dict) -> dict:
    if roi["kind"] == "rect":
        return {"rect": [int(v) for v in roi["rect"]]}
    return {"vertices": [[float(x), float(y)] for x, y in roi["vertices"]]}


def _uid(*parts) -> str:
    import hashlib
    import json
    return hashlib.sha1(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:24]


def roi_record(scene, roi: dict) -> dict:
    """The row the rois table stores for one ROI of one scene."""
    h, w, _ = scene.cube.shape
    m = roi_mask(roi, h, w)
    ys, xs = np.nonzero(m)
    geo = getattr(scene, "geo", None)
    gm = geometry_to_map(roi, geo)
    return {"uid": roi["id"], "name": roi["name"], "class": roi["class"], "kind": roi["kind"],
            "geometry": _geom(roi), "scene_name": scene.name, "scene_path": getattr(scene, "path", None),
            "image_rows": int(h), "image_cols": int(w), "geometry_map": gm, "crs": geo.crs if geo else None,
            "bbox": [int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())] if ys.size else None,
            "n_pixels": int(m.sum()),
            "metadata": {"created_by": "HyperVista ROI analysis", "geometry_units": "pixels",
                         "rect_convention": "[row_start, col_start, row_end, col_end), end-exclusive",
                         "vertex_convention": "[col, row] pixel centres"}}


def library_items(items: Sequence[Tuple[object, dict]], mode: str = "class", max_pixels: int = 200,
                  seed: int = 0) -> Tuple[List[dict], List[dict]]:
    """(roi records, spectrum items) for repository.save_roi_spectra / client.save_roi_spectra.

    mode "class"  : one pooled mean ± std per class over all its ROIs (several images allowed)
         "roi"    : one mean ± std per ROI
         "pixels" : up to max_pixels pixel spectra per ROI
    Every spectrum item carries the ROI ids it came from and a uid, so saving the same ROIs twice adds nothing.
    """
    if mode not in LIBRARY_MODES:
        raise ValueError(f"mode must be one of {', '.join(LIBRARY_MODES)}")
    if not items:
        raise ValueError("no ROIs to save")
    for sc, _ in items:
        if not getattr(sc, "has_wavelengths", False):
            raise ValueError(f"{sc.name} has no wavelengths - assign them first (Scenes tab)")
    records = [roi_record(sc, r) for sc, r in items]
    empty = [r["name"] for r in records if not r["n_pixels"]]
    if empty:
        raise ValueError("these ROIs contain no pixels: " + ", ".join(empty))
    common = {"created_by": "HyperVista", "source": "ROI analysis"}
    out: List[dict] = []
    if mode == "class":
        for cname, p in pool_classes(items).items():
            sig = sorted((sc.name, r["id"], json_geom(r)) for sc, r in items if r["class"] == cname)
            out.append({"name": cname, "class": cname, "wavelengths": _lst(p["wavelengths"]), "values": _lst(p["mean"]),
                        "std": _lst(p["std"]), "roi_uids": p["roi_ids"],
                        "sample_name": f"{cname} - class mean of {p['n_rois']} ROI(s) in {p['n_scenes']} image(s)",
                        "uid": _uid("class", sig),
                        "material_metadata": dict(common),
                        "spectrum_metadata": {"roi_mode": "class_mean", "n_pixels": p["n_pixels"],
                                              "n_rois": p["n_rois"], "n_scenes": p["n_scenes"],
                                              "scenes": p["scenes"]}})
    else:
        rng = np.random.default_rng(seed)
        for sc, r in items:
            h, w, _ = sc.cube.shape
            wl = _lst(sc.wavelengths)
            if mode == "roi":
                st = roi_stats(sc.cube, r)
                out.append({"name": r["class"], "class": r["class"], "wavelengths": wl, "values": _lst(st["mean"]),
                            "std": _lst(st["std"]), "roi_uids": [r["id"]],
                            "sample_name": f"{r['class']} - {r['name']} ({sc.name})",
                            "uid": _uid("roi", sc.name, r["id"], json_geom(r)),
                            "material_metadata": dict(common),
                            "spectrum_metadata": {"roi_mode": "roi_mean", "n_pixels": st["n_pixels"],
                                                  "roi_name": r["name"], "scene": sc.name}})
                continue
            ys, xs = np.nonzero(roi_mask(r, h, w))
            keep = ~np.all(np.isnan(sc.cube[ys, xs]), axis=1)
            ys, xs = ys[keep], xs[keep]
            if max_pixels and len(ys) > max_pixels:
                pick = np.sort(rng.choice(len(ys), max_pixels, replace=False))
                ys, xs = ys[pick], xs[pick]
            for y, x in zip(ys.tolist(), xs.tolist()):
                out.append({"name": r["class"], "class": r["class"], "wavelengths": wl,
                            "values": _lst(sc.cube[y, x]), "roi_uids": [r["id"]],
                            "sample_name": f"{r['class']} - {r['name']} pixel (r{y}, c{x})",
                            "uid": _uid("px", sc.name, r["id"], y, x),
                            "material_metadata": dict(common),
                            "spectrum_metadata": {"roi_mode": "pixel", "pixel": [y, x], "roi_name": r["name"],
                                                  "scene": sc.name}})
    return records, out


def json_geom(roi: dict) -> str:
    import json
    return json.dumps(_geom(roi), sort_keys=True)


def roi_from_record(rec: dict, scene=None) -> Optional[dict]:
    """A library rois row (repository.roi_dict) -> a ROI for *scene*.

    Same image (name and size match): the stored pixel geometry, same id.  Otherwise, when both are
    georeferenced in the same CRS, the outline is re-projected through geometry_map (new id).  None if the ROI
    cannot be placed on this image.
    """
    base = {"name": rec.get("name"), "class": rec.get("class_name") or "class_1"}
    g = rec.get("geometry") or {}
    if scene is None:
        geom = {"kind": rec["geometry_type"], **g}
        return normalize_roi({**base, **geom, "id": rec.get("roi_uid")})
    h, w, _ = scene.cube.shape
    same_image = (rec.get("image_rows"), rec.get("image_cols")) == (h, w) and \
        (not rec.get("scene_name") or rec.get("scene_name") == scene.name)
    if same_image:
        roi = normalize_roi({**base, "kind": rec["geometry_type"], **g, "id": rec.get("roi_uid")})
        return roi if roi_mask(roi, h, w).any() else None
    geom = geometry_from_map(rec.get("geometry_map"), getattr(scene, "geo", None), h, w)
    if geom is None:
        return None
    return normalize_roi({**base, **geom})
