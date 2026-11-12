"""Map background for georeferenced scenes: coordinate transforms, web-map tiles, image placement (no GUI code).

    to_lonlat / from_lonlat   any supported CRS <-> WGS84 lon/lat.  pyproj or rasterio when installed; built in
                              (no extra package) for EPSG:4326, EPSG:3857 and WGS84 UTM (EPSG:326xx / 327xx,
                              'UTM zone 44N' text, ENVI UTM map info)
    PROVIDERS                 OpenStreetMap, Esri World Imagery (satellite), OpenTopoMap - or any {z}/{x}/{y} URL
    fetch_basemap()           the tiles covering an extent, stitched into one image (web-mercator metres),
                              cached on disk in ~/.hvva/tiles (tile servers ask for caching and a User-Agent)
    scene_affine()            pixel (col, row edges) -> web-mercator metres for a scene: the image is drawn with this
                              affine on top of the tiles, so ROI outlines and clicks map both ways

Attribution must stay visible on the map (it is drawn in the corner).
"""
from __future__ import annotations

import io
import math
import os
import re
import threading
import urllib.request
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

R_EARTH = 6378137.0
TILE = 256
CACHE = os.path.join(os.path.expanduser("~"), ".hvva", "tiles")
USER_AGENT = "HyperVista-Library/3.4 (desktop hyperspectral library tool)"

PROVIDERS: Dict[str, dict] = {
    "OpenStreetMap": {"url": "https://tile.openstreetmap.org/{z}/{x}/{y}.png", "max_zoom": 19,
                      "attribution": "© OpenStreetMap contributors"},
    "Esri World Imagery (satellite)": {
        "url": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        "max_zoom": 19, "attribution": "Imagery © Esri, Maxar, Earthstar Geographics"},
    "OpenTopoMap": {"url": "https://tile.opentopomap.org/{z}/{x}/{y}.png", "max_zoom": 17,
                    "attribution": "© OpenStreetMap contributors, SRTM | © OpenTopoMap (CC-BY-SA)"},
}


# ================================================================== CRS handling
def epsg_of(crs) -> Optional[int]:
    """EPSG code from 'EPSG:32644', WKT / proj text / plain 'UTM zone 44N' (WGS 84 UTM), or None."""
    if crs is None:
        return None
    s = str(crs)
    m = re.search(r"EPSG[:\"', ]+(\d{4,5})(?!.*EPSG)", s, re.I | re.S)
    if m:
        return int(m.group(1))
    m = re.search(r"UTM[ _]?zone[ _]?(\d{1,2})\s*([NS])", s, re.I)
    # WGS 84 (or no datum named: NAD83 / ETRS89 agree with it to about a metre); old datums are not assumed
    if m and not re.search(r"NAD[ _]?27|ED[ _]?50|Pulkovo|Tokyo|Indian|Arc[ _]?19", s, re.I):
        z = int(m.group(1))
        return (32600 if m.group(2).upper() == "N" else 32700) + z
    if re.search(r"\+proj=longlat", s) or re.fullmatch(r"\s*(WGS ?84|CRS:84|lat/?lon)\s*", s, re.I):
        return 4326
    return None


def _utm_params(epsg):
    if 32601 <= epsg <= 32660:
        return epsg - 32600, True
    if 32701 <= epsg <= 32760:
        return epsg - 32700, False
    return None


_A, _F, _K0 = 6378137.0, 1 / 298.257223563, 0.9996
_E2 = _F * (2 - _F)
_EP2 = _E2 / (1 - _E2)


def _utm_to_lonlat(x, y, zone, north):
    x = np.asarray(x, float) - 500000.0
    y = np.asarray(y, float) - (0.0 if north else 10000000.0)
    lon0 = math.radians((zone - 1) * 6 - 180 + 3)
    m = y / _K0
    mu = m / (_A * (1 - _E2 / 4 - 3 * _E2 ** 2 / 64 - 5 * _E2 ** 3 / 256))
    e1 = (1 - math.sqrt(1 - _E2)) / (1 + math.sqrt(1 - _E2))
    p1 = (mu + (3 * e1 / 2 - 27 * e1 ** 3 / 32) * np.sin(2 * mu) + (21 * e1 ** 2 / 16 - 55 * e1 ** 4 / 32) * np.sin(4 * mu)
          + (151 * e1 ** 3 / 96) * np.sin(6 * mu) + (1097 * e1 ** 4 / 512) * np.sin(8 * mu))
    n1 = _A / np.sqrt(1 - _E2 * np.sin(p1) ** 2)
    t1 = np.tan(p1) ** 2
    c1 = _EP2 * np.cos(p1) ** 2
    r1 = _A * (1 - _E2) / (1 - _E2 * np.sin(p1) ** 2) ** 1.5
    d = x / (n1 * _K0)
    lat = p1 - (n1 * np.tan(p1) / r1) * (d ** 2 / 2 - (5 + 3 * t1 + 10 * c1 - 4 * c1 ** 2 - 9 * _EP2) * d ** 4 / 24
                                         + (61 + 90 * t1 + 298 * c1 + 45 * t1 ** 2 - 252 * _EP2 - 3 * c1 ** 2) * d ** 6 / 720)
    lon = lon0 + (d - (1 + 2 * t1 + c1) * d ** 3 / 6 + (5 - 2 * c1 + 28 * t1 - 3 * c1 ** 2 + 8 * _EP2 + 24 * t1 ** 2)
                  * d ** 5 / 120) / np.cos(p1)
    return np.degrees(lon), np.degrees(lat)


def _lonlat_to_utm(lon, lat, zone, north):
    lon, lat = np.radians(np.asarray(lon, float)), np.radians(np.asarray(lat, float))
    lon0 = math.radians((zone - 1) * 6 - 180 + 3)
    n = _A / np.sqrt(1 - _E2 * np.sin(lat) ** 2)
    t = np.tan(lat) ** 2
    c = _EP2 * np.cos(lat) ** 2
    a = np.cos(lat) * (lon - lon0)
    m = _A * ((1 - _E2 / 4 - 3 * _E2 ** 2 / 64 - 5 * _E2 ** 3 / 256) * lat
              - (3 * _E2 / 8 + 3 * _E2 ** 2 / 32 + 45 * _E2 ** 3 / 1024) * np.sin(2 * lat)
              + (15 * _E2 ** 2 / 256 + 45 * _E2 ** 3 / 1024) * np.sin(4 * lat) - (35 * _E2 ** 3 / 3072) * np.sin(6 * lat))
    x = _K0 * n * (a + (1 - t + c) * a ** 3 / 6 + (5 - 18 * t + t ** 2 + 72 * c - 58 * _EP2) * a ** 5 / 120) + 500000.0
    y = _K0 * (m + n * np.tan(lat) * (a ** 2 / 2 + (5 - t + 9 * c + 4 * c ** 2) * a ** 4 / 24
                                       + (61 - 58 * t + t ** 2 + 600 * c - 330 * _EP2) * a ** 6 / 720))
    if not north:
        y = y + 10000000.0
    return x, y


def _lib_transform(xs, ys, src, dst):
    try:
        from pyproj import Transformer
        t = Transformer.from_crs(src, dst, always_xy=True)
        return t.transform(np.asarray(xs, float), np.asarray(ys, float))
    except ImportError:
        pass
    from rasterio.warp import transform as rtransform                  # ImportError -> caller falls back
    a, b = rtransform(src, dst, list(np.ravel(xs)), list(np.ravel(ys)))
    return np.asarray(a).reshape(np.shape(xs)), np.asarray(b).reshape(np.shape(ys))


def to_lonlat(xs, ys, crs):
    """Map coordinates in *crs* -> (lon, lat) degrees."""
    code = epsg_of(crs)
    if code == 4326:
        return np.asarray(xs, float), np.asarray(ys, float)
    if code == 3857:
        return merc_to_lonlat(xs, ys)
    if code and _utm_params(code):
        return _utm_to_lonlat(xs, ys, *_utm_params(code))
    try:
        return _lib_transform(xs, ys, crs if not code else f"EPSG:{code}", "EPSG:4326")
    except Exception as e:                                               # noqa: BLE001
        raise ValueError(f"cannot convert coordinates from {crs!r} to longitude/latitude "
                         f"(install pyproj for this CRS): {e}") from None


def from_lonlat(lon, lat, crs):
    """(lon, lat) degrees -> map coordinates in *crs*."""
    code = epsg_of(crs)
    if code == 4326:
        return np.asarray(lon, float), np.asarray(lat, float)
    if code == 3857:
        return lonlat_to_merc(lon, lat)
    if code and _utm_params(code):
        return _lonlat_to_utm(lon, lat, *_utm_params(code))
    try:
        return _lib_transform(lon, lat, "EPSG:4326", crs if not code else f"EPSG:{code}")
    except Exception as e:                                               # noqa: BLE001
        raise ValueError(f"cannot convert longitude/latitude to {crs!r} (install pyproj for this CRS): {e}") from None


def can_place(crs) -> bool:
    """True when coordinates in *crs* can be put on the web map."""
    code = epsg_of(crs)
    if code in (4326, 3857) or (code and _utm_params(code)):
        return True
    try:
        to_lonlat(np.array([500000.0]), np.array([4000000.0]), crs)
        return True
    except Exception:                                                    # noqa: BLE001
        return False


# ================================================================== web mercator + tiles
def lonlat_to_merc(lon, lat):
    lon = np.asarray(lon, float)
    lat = np.clip(np.asarray(lat, float), -85.05112878, 85.05112878)
    return R_EARTH * np.radians(lon), R_EARTH * np.log(np.tan(np.pi / 4 + np.radians(lat) / 2))


def merc_to_lonlat(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    return np.degrees(x / R_EARTH), np.degrees(2 * np.arctan(np.exp(y / R_EARTH)) - np.pi / 2)


_HALF = math.pi * R_EARTH


def tile_bounds(z, x, y):
    """(xmin, ymin, xmax, ymax) of tile z/x/y in web-mercator metres."""
    size = 2 * _HALF / 2 ** z
    return -_HALF + x * size, _HALF - (y + 1) * size, -_HALF + (x + 1) * size, _HALF - y * size


def choose_zoom(xmin, xmax, width_px=900, max_zoom=19):
    span = max(xmax - xmin, 1.0)
    z = math.log2(2 * _HALF * width_px / (TILE * span))
    return int(np.clip(math.floor(z), 1, max_zoom))


def _tile_range(xmin, ymin, xmax, ymax, z):
    n = 2 ** z
    size = 2 * _HALF / n
    x0 = int(np.clip(math.floor((xmin + _HALF) / size), 0, n - 1))
    x1 = int(np.clip(math.floor((xmax + _HALF) / size), 0, n - 1))
    y0 = int(np.clip(math.floor((_HALF - ymax) / size), 0, n - 1))
    y1 = int(np.clip(math.floor((_HALF - ymin) / size), 0, n - 1))
    return x0, x1, y0, y1


_MEM: Dict[tuple, np.ndarray] = {}
_LOCK = threading.Lock()


def fetch_tile(provider: str, z: int, x: int, y: int, url: Optional[str] = None, timeout=10) -> np.ndarray:
    """RGB(A) tile as float array 0-1 (disk + memory cache)."""
    key = (provider, z, x, y)
    with _LOCK:
        if key in _MEM:
            return _MEM[key]
    from PIL import Image
    safe = re.sub(r"[^A-Za-z0-9]+", "_", provider)
    path = os.path.join(CACHE, safe, str(z), str(x), f"{y}.tile")
    data = None
    if os.path.exists(path):
        with open(path, "rb") as f:
            data = f.read()
    if data is None:
        tpl = url or PROVIDERS[provider]["url"]
        req = urllib.request.Request(tpl.format(z=z, x=x, y=y), headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = r.read()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(data)
    img = np.asarray(Image.open(io.BytesIO(data)).convert("RGB"), np.float32) / 255.0
    with _LOCK:
        _MEM[key] = img
    return img


def fetch_basemap(extent, provider="OpenStreetMap", width_px=900, url=None, max_tiles=64):
    """Tiles covering extent (xmin, ymin, xmax, ymax in web-mercator m) -> (image, (xmin, xmax, ymin, ymax), z).
    Tiles that cannot be downloaded stay light grey; raises only if none could be fetched."""
    xmin, ymin, xmax, ymax = extent
    info = PROVIDERS.get(provider, {"max_zoom": 19})
    z = choose_zoom(xmin, xmax, width_px, info.get("max_zoom", 19))
    while True:
        x0, x1, y0, y1 = _tile_range(xmin, ymin, xmax, ymax, z)
        if (x1 - x0 + 1) * (y1 - y0 + 1) <= max_tiles or z <= 1:
            break
        z -= 1
    mosaic = np.full(((y1 - y0 + 1) * TILE, (x1 - x0 + 1) * TILE, 3), 0.92, np.float32)
    ok, errors = 0, []
    for ty in range(y0, y1 + 1):
        for tx in range(x0, x1 + 1):
            try:
                t = fetch_tile(provider, z, tx, ty, url)
                h, w = min(t.shape[0], TILE), min(t.shape[1], TILE)
                mosaic[(ty - y0) * TILE:(ty - y0) * TILE + h, (tx - x0) * TILE:(tx - x0) * TILE + w] = t[:h, :w, :3]
                ok += 1
            except Exception as e:                                       # noqa: BLE001
                errors.append(str(e))
    if not ok:
        raise ConnectionError(f"no map tiles could be downloaded ({errors[0] if errors else 'no tiles'}) - "
                              "check the internet connection")
    bx0, _, _, by1 = tile_bounds(z, x0, y0)
    _, by0, bx1, _ = tile_bounds(z, x1, y1)
    return mosaic, (bx0, bx1, by0, by1), z


# ================================================================== placing a scene
def scene_affine(scene) -> np.ndarray:
    """2x3 matrix M: web-mercator [X, Y] = M @ [col, row, 1] with (col, row) pixel EDGE coordinates.
    Fitted through the four image corners (exact for the corners, metre-accurate inside a scene)."""
    if scene.geo is None:
        raise ValueError(f"{scene.name} is not georeferenced - set it in the Georeferencing tab")
    if not scene.geo.crs:
        raise ValueError(f"{scene.name} has no CRS - set it in the Georeferencing tab")
    h, w = scene.cube.shape[:2]
    pix = np.array([(0, 0), (w, 0), (0, h), (w, h)], float)              # (col, row) corners
    mx, my = [], []
    for c, r in pix:
        x, y = scene.geo.xy(r, c, center=False)
        mx.append(x)
        my.append(y)
    lon, lat = to_lonlat(np.array(mx), np.array(my), scene.geo.crs)
    X, Y = lonlat_to_merc(lon, lat)
    A = np.column_stack([pix, np.ones(4)])
    cx, *_ = np.linalg.lstsq(A, X, rcond=None)
    cy, *_ = np.linalg.lstsq(A, Y, rcond=None)
    return np.vstack([cx, cy])


def footprint(M, rows, cols):
    pts = np.array([(0, 0), (cols, 0), (cols, rows), (0, rows)], float)
    return (M @ np.column_stack([pts, np.ones(4)]).T).T


def merc_to_pixel(M, X, Y):
    """Inverse of scene_affine: web-mercator -> fractional (row, col) pixel-CENTRE coordinates."""
    A = M[:, :2]
    t = M[:, 2]
    col_e, row_e = np.linalg.solve(A, np.array([X, Y], float) - t)
    return row_e - 0.5, col_e - 0.5


# ================================================================== georeferencing helpers
def affine_from_corners(ul_lonlat, lr_lonlat, rows, cols, crs="EPSG:4326"):
    """North-up transform from the upper-left / lower-right corner lon/lat of the image (corners of the pixels)."""
    (lon0, lat0), (lon1, lat1) = ul_lonlat, lr_lonlat
    x0, y0 = from_lonlat(np.array([lon0]), np.array([lat0]), crs)
    x1, y1 = from_lonlat(np.array([lon1]), np.array([lat1]), crs)
    x0, y0, x1, y1 = float(x0[0]), float(y0[0]), float(x1[0]), float(y1[0])
    if x1 == x0 or y1 == y0:
        raise ValueError("the two corners must differ in both directions")
    return ((x1 - x0) / cols, 0.0, x0, 0.0, (y1 - y0) / rows, y0)


def affine_from_gcps(gcps: Sequence[Tuple[float, float, float, float]]):
    """Least-squares affine from ground control points [(col, row, x, y), ...] (pixel CENTRES, >= 3 points).
    -> (transform (a, b, c, d, e, f) for pixel corners as GeoInfo uses, rmse in map units, residuals)."""
    g = np.asarray(gcps, float)
    if g.shape[0] < 3:
        raise ValueError("at least 3 control points are needed")
    A = np.column_stack([g[:, 0] + 0.5, g[:, 1] + 0.5, np.ones(len(g))])   # centre -> corner-based coordinates
    if np.linalg.matrix_rank(A) < 3:
        raise ValueError("the control points lie on one line - spread them over the image")
    cx, *_ = np.linalg.lstsq(A, g[:, 2], rcond=None)
    cy, *_ = np.linalg.lstsq(A, g[:, 3], rcond=None)
    pred = np.column_stack([A @ cx, A @ cy])
    res = np.hypot(pred[:, 0] - g[:, 2], pred[:, 1] - g[:, 3])
    rmse = float(np.sqrt(np.mean(res ** 2)))
    return (float(cx[0]), float(cx[1]), float(cx[2]), float(cy[0]), float(cy[1]), float(cy[2])), rmse, res
