"""ROI files from other tools: GeoJSON and ENVI ASCII ROI exports (read and write; no GUI code).

GeoJSON   Polygon / MultiPolygon -> polygon ROIs, Point / MultiPoint -> pixel ROIs.  Coordinates are map
          coordinates of the scene's CRS, longitude/latitude (RFC 7946 default; converted when the scene has a
          projected CRS), or pixel coordinates (x = column, y = row) - 'auto' decides from the numbers.
          The class comes from the feature property class / class_name / label / type / name.
ENVI      'ENVI Output of ROIs' text: per ROI its name, colour and pixel list (ID, X, Y [, Map X, Map Y, Lat, Lon,
          band values]); X / Y are 1-based sample / line.  Read as pixel ROIs; written with map coordinates
          when the scene is georeferenced.
"""
from __future__ import annotations

import json
import re
from typing import List, Optional, Tuple

import numpy as np

from .rois import normalize_roi, outline, roi_mask

CLASS_KEYS = ("class", "class_name", "classname", "label", "type", "category", "name")


# ================================================================== GeoJSON
def _features(doc) -> list:
    t = doc.get("type")
    if t == "FeatureCollection":
        return doc.get("features") or []
    if t == "Feature":
        return [doc]
    if t in ("Polygon", "MultiPolygon", "Point", "MultiPoint"):
        return [{"type": "Feature", "geometry": doc, "properties": {}}]
    raise ValueError("not a GeoJSON FeatureCollection / Feature / geometry")


def _all_coords(feats):
    out = []
    for f in feats:
        g = f.get("geometry") or {}
        c = g.get("coordinates")
        if c is None:
            continue
        a = np.asarray(_flatten(c), float)
        if a.size:
            out.append(a.reshape(-1, 2))
    return np.vstack(out) if out else np.zeros((0, 2))


def _flatten(c):
    if isinstance(c, (list, tuple)) and c and isinstance(c[0], (int, float)):
        return [c[:2]]
    out = []
    for x in c or []:
        out.extend(_flatten(x))
    return out


def _space(doc, feats, scene, coords):
    if coords != "auto":
        return coords
    xy = _all_coords(feats)
    h, w = scene.cube.shape[:2]
    if xy.size and (xy[:, 0].min() >= -1 and xy[:, 0].max() <= w + 1 and xy[:, 1].min() >= -1 and xy[:, 1].max() <= h + 1):
        if scene.geo is None or (doc.get("properties") or {}).get("coordinate_space") == "pixel" or \
                str((doc.get("crs") or {}).get("properties", {}).get("name", "")).lower() == "pixel":
            return "pixel"
    if scene.geo is None:
        if xy.size and np.abs(xy).max() > max(h, w) * 4:
            raise ValueError("the GeoJSON has map coordinates but the image is not georeferenced - set its "
                             "georeferencing first (Georeferencing tab)")
        return "pixel"
    crs_name = str((doc.get("crs") or {}).get("properties", {}).get("name", ""))
    if crs_name and "4326" not in crs_name and "CRS84" not in crs_name.upper():
        return "map"
    if xy.size and np.abs(xy[:, 0]).max() <= 180 and np.abs(xy[:, 1]).max() <= 90:
        from .basemap import epsg_of
        if epsg_of(scene.geo.crs) != 4326:
            return "lonlat"
    return "map"


def _to_pixel(points, scene, space):
    """[(x, y), ...] in *space* -> [(col, row), ...] pixel-centre coordinates."""
    p = np.asarray(points, float).reshape(-1, 2)
    if space == "pixel":
        return p
    xs, ys = p[:, 0], p[:, 1]
    if space == "lonlat":
        from .basemap import from_lonlat
        xs, ys = from_lonlat(xs, ys, scene.geo.crs)
    out = []
    for x, y in zip(np.atleast_1d(xs), np.atleast_1d(ys)):
        r, c = scene.geo.rowcol(float(x), float(y))
        out.append((c - 0.5, r - 0.5))
    return np.asarray(out, float)


def _class_of(props, default):
    for k in CLASS_KEYS:
        for kk in (k, k.upper(), k.capitalize()):
            if props.get(kk) not in (None, ""):
                return str(props[kk])
    return default


def read_geojson(source, scene, coords: str = "auto", default_class: str = "class_1") -> Tuple[List[dict], int, str]:
    """(ROIs for *scene*, number skipped, coordinate space used).  *source*: path, text or parsed dict."""
    if isinstance(source, dict):
        doc = source
    elif isinstance(source, str) and source.lstrip().startswith("{"):
        doc = json.loads(source)
    else:
        with open(source, encoding="utf-8") as f:
            doc = json.load(f)
    feats = _features(doc)
    space = _space(doc, feats, scene, coords)
    h, w = scene.cube.shape[:2]
    rois, skipped = [], 0
    for k, f in enumerate(feats, 1):
        g = f.get("geometry") or {}
        props = f.get("properties") or {}
        cls = _class_of(props, default_class)
        name = str(props.get("name") or props.get("id") or f.get("id") or f"ROI {k}")
        t = g.get("type")
        try:
            if t == "Polygon":
                polys = [g["coordinates"]]
            elif t == "MultiPolygon":
                polys = g["coordinates"]
            elif t in ("Point", "MultiPoint"):
                pts = [g["coordinates"]] if t == "Point" else g["coordinates"]
                pix = _to_pixel([p[:2] for p in pts], scene, space)
                px = sorted({(int(round(r)), int(round(c))) for c, r in pix
                             if 0 <= round(r) < h and 0 <= round(c) < w})
                if not px:
                    skipped += 1
                    continue
                rois.append(normalize_roi({"kind": "pixels", "pixels": px, "class": cls, "name": name}))
                continue
            else:
                skipped += 1
                continue
            for j, poly in enumerate(polys):
                ring = [p[:2] for p in poly[0]]
                if len(ring) > 3 and ring[0] == ring[-1]:
                    ring = ring[:-1]
                verts = [(float(c), float(r)) for c, r in _to_pixel(ring, scene, space)]
                roi = normalize_roi({"kind": "polygon", "vertices": verts, "class": cls,
                                     "name": name if len(polys) == 1 else f"{name} ({j + 1})"})
                if roi_mask(roi, h, w).any():
                    rois.append(roi)
                else:
                    skipped += 1
        except (KeyError, TypeError, ValueError, IndexError):
            skipped += 1
    return rois, skipped, space


def write_geojson(items, path: str, coords: str = "auto") -> str:
    """ROIs of one or several scenes -> GeoJSON.  Map coordinates when the scene is georeferenced (CRS in the
    file's 'crs' member and in each feature), else pixel coordinates (property coordinate_space = 'pixel')."""
    feats, crs_names = [], set()
    for sc, roi in items:
        use_map = coords != "pixel" and sc.geo is not None
        if roi["kind"] == "pixels":
            pts = [(c, r) for r, c in roi["pixels"]]
            geom_pts = [list(sc.geo.xy(r, c)) for c, r in pts] if use_map else [[float(c), float(r)] for c, r in pts]
            geom = {"type": "MultiPoint", "coordinates": geom_pts}
        else:
            ring = outline(roi) if roi["kind"] == "rect" else np.asarray(roi["vertices"], float)
            if use_map:
                pts = [list(sc.geo.xy(y + 0.5, x + 0.5, center=False)) for x, y in ring]
            else:
                pts = [[float(x), float(y)] for x, y in ring]
            pts.append(pts[0])
            geom = {"type": "Polygon", "coordinates": [pts]}
        if use_map:
            crs_names.add(sc.geo.crs or "")
        feats.append({"type": "Feature", "geometry": geom,
                      "properties": {"class": roi["class"], "name": roi["name"], "id": roi["id"], "scene": sc.name,
                                     "coordinate_space": "map" if use_map else "pixel",
                                     "crs": sc.geo.crs if use_map else None}})
    doc = {"type": "FeatureCollection", "name": "HyperVista ROIs", "features": feats}
    if len(crs_names) == 1 and next(iter(crs_names)):
        doc["crs"] = {"type": "name", "properties": {"name": next(iter(crs_names))}}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=1)
    return path


# ================================================================== ENVI ASCII ROI
def read_envi_ascii(path, rows: int, cols: int) -> Tuple[List[dict], int]:
    """(pixel ROIs, pixels skipped as outside the image) from an ENVI 'Output of ROIs' text file."""
    text = open(path, encoding="utf-8", errors="replace").read()
    if "ROI name" not in text:
        raise ValueError("not an ENVI ASCII ROI file (no '; ROI name:' lines)")
    names = re.findall(r";\s*ROI name:\s*(.+)", text)
    colours = re.findall(r";\s*ROI rgb value:\s*\{([^}]*)\}", text)
    header = None
    blocks, cur = [], []
    for line in text.splitlines():
        s = line.strip()
        if s.startswith(";"):
            if re.search(r"\bID\b", s) and re.search(r"\bX\b", s) and re.search(r"\bY\b", s):
                header = re.split(r"\s{2,}|\t", s.lstrip(";").strip())
            continue
        if not s:
            if cur:
                blocks.append(cur)
                cur = []
            continue
        cur.append(s.split())
    if cur:
        blocks.append(cur)
    ix, iy = 1, 2
    if header:
        hl = [h.strip().lower() for h in header]
        if "x" in hl and "y" in hl:
            ix, iy = hl.index("x"), hl.index("y")
    rois, skipped = [], 0
    for k, block in enumerate(blocks):
        px = set()
        for row in block:
            try:
                x, y = int(float(row[ix])) - 1, int(float(row[iy])) - 1          # 1-based sample / line
            except (ValueError, IndexError):
                continue
            if 0 <= y < rows and 0 <= x < cols:
                px.add((y, x))
            else:
                skipped += 1
        if not px:
            continue
        name = names[k].strip() if k < len(names) else f"ROI {k + 1}"
        m = re.match(r"^(.*?)\s*\[(.+)\]$", name)                  # HyperVista writes 'name [class]'
        cls = m.group(2).strip() if m else name
        name = m.group(1).strip() if m else name
        roi = normalize_roi({"kind": "pixels", "pixels": sorted(px), "class": cls, "name": name})
        if k < len(colours):
            roi["envi_rgb"] = [int(float(v)) for v in colours[k].split(",")[:3]]
        rois.append(roi)
    return rois, skipped


def write_envi_ascii(scene, rois, path: str, with_values: bool = False) -> str:
    """ENVI 'Output of ROIs' text for ROIs of one scene (any kind: rectangles / polygons become pixel lists)."""
    h, w = scene.cube.shape[:2]
    masks = [roi_mask(r, h, w) for r in rois]
    lines = ["; ENVI Output of ROIs (HyperVista)", f"; Number of ROIs: {len(rois)}", f"; File Dimension: {w} x {h}",
             ";"]
    import matplotlib
    cmap = matplotlib.colormaps["tab10"]
    for k, (r, m) in enumerate(zip(rois, masks)):
        rgb = r.get("envi_rgb") or [int(255 * v) for v in cmap(k % 10)[:3]]
        lines += [f"; ROI name: {r['name']} [{r['class']}]" if r["name"] != r["class"] else f"; ROI name: {r['name']}",
                  f"; ROI rgb value: {{{rgb[0]}, {rgb[1]}, {rgb[2]}}}", f"; ROI npts: {int(m.sum())}"]
    cols_h = ["ID", "X", "Y"] + (["Map X", "Map Y"] if scene.geo else [])
    if with_values:
        cols_h += [f"B{i + 1}" for i in range(scene.n_bands)]
    lines.append(";" + "".join(f"{c:>12}" for c in cols_h))
    for m in masks:
        ys, xs = np.nonzero(m)
        for i, (y, x) in enumerate(zip(ys, xs), 1):
            vals = [f"{i:>12d}", f"{x + 1:>12d}", f"{y + 1:>12d}"]
            if scene.geo:
                mx, my = scene.geo.xy(int(y), int(x))
                vals += [f"{mx:>12.3f}", f"{my:>12.3f}"]
            if with_values:
                vals += [f"{v:>12.6g}" for v in scene.cube[y, x]]
            lines.append("".join(vals))
        lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return path
