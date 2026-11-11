"""Ground-control-point georeferencing: fit an affine map  pixel (col, row) -> (lon, lat)  from >= 3 control points.

The control points are pairs  (row, col) in the image  <->  (lat, lon) on the reference map (Esri World Imagery in the
page). The fit and its residuals are stored in the image's metadata under "georef", so the image behaves like a
georeferenced file from then on (corners, ROI latitude / longitude, patch GeoTIFFs).

Pixel coordinates are continuous, measured from the top-left EDGE of the image: the centre of pixel (row, col) is
(col + 0.5, row + 0.5).
"""
from __future__ import annotations
import math
from typing import List

import numpy as np

M_PER_DEG = 111_320.0


def fit_gcps(points: List[dict]) -> dict:
    """points: [{row, col, lat, lon}, ...] (>= 3, not on one line). Returns the affine, per-point residuals and the RMSE (m)."""
    if len(points) < 3:
        raise ValueError("georeferencing needs at least 3 control points (4 or more gives a check on the fit)")
    x = np.array([p["col"] + 0.5 for p in points], float)
    y = np.array([p["row"] + 0.5 for p in points], float)
    lon = np.array([p["lon"] for p in points], float)
    lat = np.array([p["lat"] for p in points], float)
    if not (np.all(np.abs(lat) <= 90) and np.all(np.abs(lon) <= 180)):
        raise ValueError("latitude must be within +-90 and longitude within +-180")
    A = np.c_[x, y, np.ones_like(x)]
    if np.linalg.matrix_rank(A, tol=1e-6 * max(1.0, float(np.abs(A).max()))) < 3:
        raise ValueError("the control points lie on one line (or repeat): spread them over the image, ideally near the corners")
    cl, *_ = np.linalg.lstsq(A, lon, rcond=None)
    cf, *_ = np.linalg.lstsq(A, lat, rcond=None)
    a, b, c = cl; d, e, f = cf
    if abs(a * e - b * d) < 1e-18:
        raise ValueError("the fitted map is degenerate: check the control points")
    plon, plat = A @ cl, A @ cf
    cos = math.cos(math.radians(float(lat.mean())))
    res_m = np.hypot((plon - lon) * M_PER_DEG * cos, (plat - lat) * M_PER_DEG)
    rmse = float(np.sqrt(np.mean(res_m ** 2))) if len(points) > 3 else 0.0        # exactly 3 points always fit exactly
    # ground size of one pixel (m), for the user's sanity check
    px_x = math.hypot(a * M_PER_DEG * cos, d * M_PER_DEG)
    px_y = math.hypot(b * M_PER_DEG * cos, e * M_PER_DEG)
    return dict(affine=[float(a), float(b), float(c), float(d), float(e), float(f)], n=len(points), rmse_m=rmse,
                residuals_m=[float(r) for r in res_m], pixel_size_m=[float(px_x), float(px_y)],
                exact_fit=len(points) == 3)


def apply_affine(aff, rows, cols, centre=True):
    """(row, col) arrays -> (lon, lat) arrays with the stored affine."""
    a, b, c, d, e, f = aff
    r = np.asarray(rows, float) + (0.5 if centre else 0.0)
    q = np.asarray(cols, float) + (0.5 if centre else 0.0)
    return a * q + b * r + c, d * q + e * r + f


def saved(img):
    """The georeference stored on an Image row, or None."""
    g = (getattr(img, "image_metadata", None) or {}).get("georef")
    return g if g and g.get("affine") else None
