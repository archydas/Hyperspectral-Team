"""Object-sample (ROI) image patches: crop the polygon's bounding box out of a cube and write it as a GeoTIFF.

* file name = the class name  ->  <folder>/<class name>.tif   (a second ROI of the same class gets _2, _3, ...)
* bands = the image's bands, float32 reflectance (or the image's own unit); pixels outside the polygon = NaN (no-data)
* wavelengths are written as ENVI-domain tags, so the patch opens again in DS3 (Library > Load image) with the same wavelength axis
* when the image is georeferenced (its own map information, or ground control points set in the Image module) the patch is
  georeferenced too (cropped transform).
"""
from __future__ import annotations
import re
from pathlib import Path

import numpy as np

from . import images as svc
from . import georef as gr

MAX_PATCH_VALUES = 80_000_000          # pixels x bands: ~320 MB as float32; larger ROIs should be split


def safe_name(name: str, default: str = "roi") -> str:
    s = re.sub(r"[^\w\-. ()]+", "_", (name or "").strip()).strip(" ._")
    return (s or default)[:80]


def unique_path(folder: Path, stem: str, ext: str = ".tif") -> Path:
    p = folder / (stem + ext)
    k = 2
    while p.exists():
        p = folder / f"{stem}_{k}{ext}"; k += 1
    return p


def _frame(img, cube):
    """(affine(a..f) in the file's crs, crs string) for pixel-edge coordinates, or (None, None)."""
    tr = getattr(cube, "transform", None)
    crs = getattr(cube, "crs", None) or svc._crs_from_map_info((getattr(cube, "metadata", None) or {}).get("envi_map_info"))
    if tr is not None and crs:
        return [float(x) for x in tr[:6]], crs
    g = gr.saved(img)
    if g:
        return g["affine"], "EPSG:4326"
    return None, None


def pixels_to_lonlat(img, points):
    """[[row, col], ...] -> [{lat, lon}, ...] (None when the image has no georeference)."""
    cube = svc.open_image(img)
    aff, crs = _frame(img, cube)
    if aff is None:
        return None
    pts = np.asarray(points, float).reshape(-1, 2)
    a, b, c, d, e, f = aff
    q, r = pts[:, 1] + 0.5, pts[:, 0] + 0.5
    xs, ys = a * q + b * r + c, d * q + e * r + f
    if crs != "EPSG:4326":
        from rasterio.crs import CRS
        from rasterio.warp import transform as wt
        xs, ys = wt(CRS.from_user_input(crs), "EPSG:4326", list(xs), list(ys))
    return [dict(lat=float(la), lon=float(lo)) for lo, la in zip(xs, ys)]


def default_rgb(wl_nm, rgb=None):
    """Band numbers (1-based) used as the default colour display: the ones the page shows, else the bands nearest to 650 / 550 / 470 nm."""
    n = len(wl_nm)
    if rgb and len(rgb) == 3 and all(1 <= int(b) <= n for b in rgb):
        return [int(b) for b in rgb]
    w = np.asarray(wl_nm, float)
    if n >= 3 and w[0] < 600 < w[-1]:
        return [int(np.argmin(np.abs(w - t))) + 1 for t in (650.0, 550.0, 470.0)]
    return [1, min(2, n), min(3, n)]


def write_patch(img, polygon, class_name: str, folder, *, filename_stem: str | None = None, rgb=None) -> dict:
    """Write the ROI patch; returns {file, path, folder, rows, cols, bands, n_pixels, lat, lon, georeferenced}."""
    import rasterio
    from rasterio.transform import Affine
    from matplotlib.path import Path as MPath
    cube = svc.open_image(img)
    pts = np.asarray(polygon, float)
    if pts.ndim != 2 or pts.shape[0] < 3 or pts.shape[1] != 2:
        raise ValueError("an ROI needs at least 3 corner points")
    r0 = max(0, int(np.floor(pts[:, 0].min()))); r1 = min(cube.rows, int(np.ceil(pts[:, 0].max())) + 1)
    c0 = max(0, int(np.floor(pts[:, 1].min()))); c1 = min(cube.cols, int(np.ceil(pts[:, 1].max())) + 1)
    if r1 <= r0 or c1 <= c0:
        raise ValueError("the ROI lies outside the image")
    if (r1 - r0) * (c1 - c0) * cube.bands > MAX_PATCH_VALUES:
        raise ValueError(f"this ROI is too large for one patch ({r1 - r0} x {c1 - c0} pixels x {cube.bands} bands): draw a smaller polygon")
    block = np.asarray(cube.read(rows=(r0, r1), cols=(c0, c1)), np.float32)          # (rows, cols, bands)
    gy, gx = np.meshgrid(np.arange(r0, r1), np.arange(c0, c1), indexing="ij")
    inside = MPath(pts).contains_points(np.c_[gy.ravel(), gx.ravel()]).reshape(gy.shape)
    block[~inside] = np.nan
    n_valid = int((inside & np.isfinite(block).any(axis=2)).sum())
    if n_valid == 0:
        raise ValueError("the ROI contains no valid pixels")
    folder = Path(folder).expanduser()
    folder.mkdir(parents=True, exist_ok=True)
    out = unique_path(folder, filename_stem or safe_name(class_name))
    aff, crs = _frame(img, cube)
    kw = {}
    if aff is not None:
        a, b, c, d, e, f = aff
        kw = dict(crs=crs, transform=Affine(a, b, a * c0 + b * r0 + c, d, e, d * c0 + e * r0 + f))
    wl = [float(x) for x in cube.wavelengths_nm]
    data = np.moveaxis(block, 2, 0)                                                   # (bands, rows, cols)
    with rasterio.open(out, "w", driver="GTiff", height=r1 - r0, width=c1 - c0, count=cube.bands, dtype="float32",
                       nodata=float("nan"), compress="deflate", **kw) as dst:
        dst.write(data)
        for i, w in enumerate(wl, 1):
            dst.set_band_description(i, f"{w:.2f} nm")
        dst.update_tags(ns="ENVI", wavelength="{" + ", ".join(f"{w:.3f}" for w in wl) + "}", wavelength_units="nm",
                        bbl="{" + ", ".join("0" if bad else "1" for bad in cube.bad_band_mask) + "}")
        dbands = default_rgb(wl, rgb)
        dst.update_tags(ns="ENVI", default_bands="{" + ", ".join(str(b) for b in dbands) + "}")     # ENVI shows these bands first
        dst.update_tags(display_bands_rgb=",".join(str(b) for b in dbands), unit=str(cube.unit), class_name=str(class_name), source_image=str(getattr(img, "name", "")),
                        source_rows=f"{r0}:{r1}", source_cols=f"{c0}:{c1}", created_by="DS3 object sample")
    cy, cx = float(pts[:, 0].mean()), float(pts[:, 1].mean())
    ll = pixels_to_lonlat(img, [[cy, cx]])
    return dict(file=out.name, path=str(out), folder=str(out.parent), rows=r1 - r0, cols=c1 - c0, bands=cube.bands,
                display_bands=dbands, n_pixels=n_valid, lat=ll[0]["lat"] if ll else None, lon=ll[0]["lon"] if ll else None,
                georeferenced=aff is not None)
