"""Pixel / bounding-box / polygon / mask extraction from a Cube."""
from __future__ import annotations
import numpy as np
from affine import Affine
from rasterio import features
from rasterio.warp import transform_geom
from .contract import ROIResult


def _as_geom(polygon):
    if isinstance(polygon, dict):
        if polygon.get("type") == "Feature":
            return polygon["geometry"]
        return polygon
    verts = [list(map(float, v)) for v in polygon]
    if verts[0] != verts[-1]:
        verts.append(verts[0])
    return {"type": "Polygon", "coordinates": [verts]}


def _coords(geom):
    def walk(x):
        if isinstance(x, (list, tuple)) and x and isinstance(x[0], (int, float)):
            yield x
        else:
            for y in x:
                yield from walk(y)
    return np.array(list(walk(geom["coordinates"])))[:, :2]


def extract_roi(cube, *, pixel=None, xy=None, bbox_px=None, polygon=None, polygon_coords="map",
                polygon_crs=None, mask=None, all_touched=False):
    """Extract spectra. Exactly one selector:

    pixel=(row, col)            single pixel
    xy=(x, y)                   single pixel from map coordinates (cube CRS)
    bbox_px=(r0, c0, r1, c1)    rectangle, r1/c1 exclusive
    polygon=GeoJSON geometry/Feature or [(x, y), ...]
        polygon_coords='map' (needs a georeferenced cube) or 'pixel' (x=col, y=row)
        polygon_crs='EPSG:4326' etc. -> reprojected to the cube CRS if they differ
    mask=bool array [rows, cols]
    """
    given = [k for k, v in dict(pixel=pixel, xy=xy, bbox_px=bbox_px, polygon=polygon, mask=mask).items() if v is not None]
    if len(given) != 1:
        raise ValueError(f"give exactly one ROI selector, got {given}")
    R, C = cube.rows, cube.cols
    if xy is not None:
        pixel = cube.xy_to_pixel(*xy)
    if pixel is not None:
        r, c = int(pixel[0]), int(pixel[1])
        if not (0 <= r < R and 0 <= c < C):
            raise ValueError(f"pixel {pixel} outside cube {R}x{C}")
        r0, r1, c0, c1 = r, r + 1, c, c + 1
        sel = np.ones((1, 1), bool)
    elif bbox_px is not None:
        r0, c0, r1, c1 = bbox_px
        r0, c0, r1, c1 = max(0, r0), max(0, c0), min(R, r1), min(C, c1)
        sel = np.ones((max(0, r1 - r0), max(0, c1 - c0)), bool)
    elif mask is not None:
        mask = np.asarray(mask, bool)
        if mask.shape != (R, C):
            raise ValueError(f"mask shape {mask.shape} != cube {(R, C)}")
        rr, cc = np.where(mask)
        if rr.size == 0:
            r0 = r1 = c0 = c1 = 0; sel = np.zeros((0, 0), bool)
        else:
            r0, r1, c0, c1 = rr.min(), rr.max() + 1, cc.min(), cc.max() + 1
            sel = mask[r0:r1, c0:c1]
    else:
        geom = _as_geom(polygon)
        if polygon_coords == "map":
            if cube.transform is None:
                raise ValueError("cube is not georeferenced; use polygon_coords='pixel'")
            if polygon_crs and cube.crs and polygon_crs != cube.crs:
                geom = transform_geom(polygon_crs, cube.crs, geom)
            inv = ~Affine(*cube.transform)
            pts = np.array([inv @ (x, y) for x, y in _coords(geom)])      # (col, row)
            full_tr = Affine(*cube.transform)
        else:
            pts = _coords(geom)
            full_tr = Affine.identity()
        c0 = max(0, int(np.floor(pts[:, 0].min()))); c1 = min(C, int(np.ceil(pts[:, 0].max())) + 1)
        r0 = max(0, int(np.floor(pts[:, 1].min()))); r1 = min(R, int(np.ceil(pts[:, 1].max())) + 1)
        if r1 <= r0 or c1 <= c0:
            sel = np.zeros((0, 0), bool)
        else:
            sel = features.geometry_mask([geom], out_shape=(r1 - r0, c1 - c0),
                                         transform=full_tr @ Affine.translation(c0, r0),
                                         invert=True, all_touched=all_touched)
    nb = cube.bands
    if sel.size == 0 or not sel.any():
        z = np.zeros((0, nb), np.float32)
        return ROIResult(z, np.zeros((0, 2), int), np.full(nb, np.nan, np.float32), np.full(nb, np.nan, np.float32),
                         cube.wavelengths_nm, cube.bad_band_mask, cube.unit, 0, cube.source_path)
    block = cube.read(rows=(r0, r1), cols=(c0, c1))
    rr, cc = np.where(sel)
    spectra = block[rr, cc, :]
    coords = np.stack([rr + r0, cc + c0], axis=1)
    valid = ~np.all(np.isnan(spectra), axis=1)
    n_ex = int((~valid).sum())
    spectra, coords = spectra[valid], coords[valid]
    with np.errstate(all="ignore"):
        mean = np.nanmean(spectra, axis=0) if len(spectra) else np.full(nb, np.nan)
        std = np.nanstd(spectra, axis=0) if len(spectra) else np.full(nb, np.nan)
    return ROIResult(spectra.astype(np.float32), coords, mean.astype(np.float32), std.astype(np.float32),
                     cube.wavelengths_nm, cube.bad_band_mask, cube.unit, n_ex, cube.source_path)
