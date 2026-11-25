"""Point ground truth for classification: tables of (lat, lon | x, y | row, col) + value / class, one table per
layer (e.g. one Excel sheet per material), placed on the pixels of an image and turned into a label map.

Typical input (one file per material, value = unmixing abundance 0..1):

    lat        lon        val
    16.665071  80.449649  0.4367
    ...

Steps
    read_table()      Excel (.xlsx / .xls), CSV / TSV / TXT -> DataFrame
    guess_columns()   which columns hold lat / lon / x / y / row / col / value / class
    place_points()    -> standard frame: layer, lat, lon, row, col, value, cls, offset (pixels from the centre)
    stack_layers()    -> {layer: (rows, cols) value image, NaN where the layer has no point}
    make_labels()     label rule -> (labels, names, counts); 0 = unlabelled

Label rules (rule["kind"])
    dominant   class = layer with the highest value, if >= threshold; covered pixels below it -> "none" (optional)
    presence   one layer: "<layer>" where value >= threshold, "not <layer>" elsewhere in its coverage
    levels     one layer binned by edges, e.g. [0.05, 0.15] -> "< 0.05", "0.05 to 0.15", ">= 0.15"
    class      a class column (text or codes) in the table(s)

The functions are GUI-free so they can be tested and reused (DS3 Viewer and Analysis, scripts).
"""
from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

TABLE_EXTS = (".xlsx", ".xlsm", ".xls", ".csv", ".tsv", ".txt")
RULE_KINDS = ("dominant", "presence", "levels", "class")

_ROLE_NAMES = {
    "lat": ("lat", "latitude", "lat_dd", "y_lat", "lat_deg"),
    "lon": ("lon", "long", "longitude", "lng", "lon_dd", "x_lon", "lon_deg"),
    "x": ("x", "easting", "east", "utm_x", "map_x", "xcoord", "x_coord"),
    "y": ("y", "northing", "north", "utm_y", "map_y", "ycoord", "y_coord"),
    "row": ("row", "line", "pixel_row", "r", "i"),
    "col": ("col", "column", "sample", "pixel_col", "c", "j"),
    "value": ("val", "value", "abundance", "fraction", "score", "conc", "concentration", "amount", "v"),
    "cls": ("class", "class_name", "label", "category", "material", "type", "name", "cls"),
}


# ------------------------------------------------------------------ reading
def table_sheets(path) -> List[str]:
    """Sheet names of an Excel workbook ([] for text tables)."""
    p = Path(path)
    if p.suffix.lower() not in (".xlsx", ".xlsm", ".xls"):
        return []
    import pandas as pd
    with pd.ExcelFile(p) as x:
        return list(x.sheet_names)


def read_table(path, sheet=None):
    """A point table as a pandas DataFrame (first sheet unless *sheet* is given). Column names are stripped."""
    import pandas as pd
    p = Path(path)
    ext = p.suffix.lower()
    if ext in (".xlsx", ".xlsm", ".xls"):
        try:
            df = pd.read_excel(p, sheet_name=sheet if sheet is not None else 0)
        except ImportError as e:                                   # openpyxl / xlrd missing
            raise RuntimeError(f"Reading {ext} files needs openpyxl (pip install openpyxl): {e}") from e
    elif ext in (".csv", ".tsv", ".txt"):
        df = pd.read_csv(p, sep=None, engine="python", comment="#")
    else:
        raise ValueError(f"{p.name}: not a table ({', '.join(TABLE_EXTS)})")
    df = df.rename(columns=lambda c: str(c).strip())
    df = df.dropna(how="all")
    if df.empty:
        raise ValueError(f"{p.name}: the table is empty")
    return df


def guess_columns(columns: Sequence[str]) -> Dict[str, Optional[str]]:
    """{role: column or None} for lat, lon, x, y, row, col, value, cls (case-insensitive name match)."""
    norm = {str(c).strip().lower().replace(" ", "_"): c for c in columns}
    out: Dict[str, Optional[str]] = {}
    used = set()
    for role, names in _ROLE_NAMES.items():
        hit = None
        for n in names:
            if n in norm and norm[n] not in used:
                hit = norm[n]
                break
        out[role] = hit
        if hit is not None:
            used.add(hit)
    return out


def layer_names(paths: Sequence) -> List[str]:
    """Readable layer names from file names: abundance_Cadmium.xlsx, abundance_Lead.xlsx -> Cadmium, Lead.
    Upload prefixes like '944cfd80-' and the common prefix / suffix of several files are removed."""
    stems = [re.sub(r"^[0-9a-f]{8}-", "", Path(p).stem) for p in paths]
    if len(stems) > 1:
        pre = _common_affix(stems)
        suf = _common_affix([s[::-1] for s in stems])[::-1]
        cut = []
        for s in stems:
            t = s[len(pre):len(s) - len(suf) if suf else None]
            cut.append(t if t.strip(" _-.") else s)
        stems = cut
    out = []
    for s in stems:
        s = s.strip(" _-.") or "layer"
        name, k = s, 2
        while name in out:
            name, k = f"{s} ({k})", k + 1
        out.append(name)
    return out


def _common_affix(strs):
    if not strs:
        return ""
    s0 = strs[0]
    n = 0
    while n < len(s0) and all(len(s) > n and s[n] == s0[n] for s in strs):
        n += 1
    pre = s0[:n]
    # cut back to a separator so 'abundance_Ch'/'abundance_Ca' -> 'abundance_'
    m = max(pre.rfind("_"), pre.rfind("-"), pre.rfind(" "), pre.rfind("."))
    return pre[:m + 1] if m >= 0 else ""


# ------------------------------------------------------------------ placing points on pixels
def _to_crs(xs, ys, src, dst):
    """Transform coordinates between CRSs (rasterio, else pyproj). Identity when src == dst."""
    if not src or not dst or str(src) == str(dst):
        return np.asarray(xs, float), np.asarray(ys, float)
    try:
        from rasterio.warp import transform as wt
        x2, y2 = wt(str(src), str(dst), list(map(float, xs)), list(map(float, ys)))
        return np.asarray(x2, float), np.asarray(y2, float)
    except ImportError:
        from pyproj import Transformer
        t = Transformer.from_crs(str(src), str(dst), always_xy=True)
        x2, y2 = t.transform(np.asarray(xs, float), np.asarray(ys, float))
        return np.asarray(x2, float), np.asarray(y2, float)


def geo_usable(geo) -> bool:
    crs = getattr(geo, "crs", None)
    return geo is not None and bool(crs) and "unknown" not in str(crs).lower()


def place_points(df, mapping: dict, geo, rows: int, cols: int, layer: str = "layer"):
    """Put the rows of *df* on the pixels of an image.

    mapping: {"lat", "lon"} column names, or {"x", "y", "crs"} (map coordinates in that CRS),
             or {"row", "col"} (pixel indices, 0-based); plus optional "value" and "cls" columns.
    geo:     hvva.core.geo.GeoInfo of the image (needed for lat/lon and x/y).
    Returns (frame, report). frame columns: layer, lat, lon, row, col, value, cls, offset, inside.
    offset = distance of the point from its pixel centre in pixels (0 when the table holds pixel centres).
    """
    import pandas as pd
    n = len(df)
    lat = lon = None
    if mapping.get("lat") and mapping.get("lon"):
        lat = pd.to_numeric(df[mapping["lat"]], errors="coerce").to_numpy(float)
        lon = pd.to_numeric(df[mapping["lon"]], errors="coerce").to_numpy(float)
        if not geo_usable(geo):
            raise ValueError("The image has no usable map projection, so latitude / longitude cannot be placed on "
                             "its pixels. Use a georeferenced image, or a table with row / col columns.")
        if np.nanmax(np.abs(lat)) > 90 or np.nanmax(np.abs(lon)) > 360:
            raise ValueError("The latitude / longitude columns hold values outside +-90 / +-360: are they in "
                             "decimal degrees? (For UTM or other map coordinates choose x / y and their CRS.)")
        mx, my = _to_crs(lon, lat, "EPSG:4326", geo.crs)
        fr, fc = _rowcol(geo, mx, my)
    elif mapping.get("x") and mapping.get("y"):
        x = pd.to_numeric(df[mapping["x"]], errors="coerce").to_numpy(float)
        y = pd.to_numeric(df[mapping["y"]], errors="coerce").to_numpy(float)
        if not geo_usable(geo):
            raise ValueError("The image has no usable map projection, so map x / y cannot be placed on its pixels.")
        src = mapping.get("crs") or geo.crs
        mx, my = _to_crs(x, y, src, geo.crs)
        fr, fc = _rowcol(geo, mx, my)
        try:
            lon, lat = _to_crs(mx, my, geo.crs, "EPSG:4326")
        except Exception:                                                       # noqa: BLE001
            lat = lon = None
    elif mapping.get("row") and mapping.get("col"):
        fr = pd.to_numeric(df[mapping["row"]], errors="coerce").to_numpy(float) + 0.5
        fc = pd.to_numeric(df[mapping["col"]], errors="coerce").to_numpy(float) + 0.5
        if geo_usable(geo):
            mx, my = _xy(geo, fr - 0.5, fc - 0.5)
            try:
                lon, lat = _to_crs(mx, my, geo.crs, "EPSG:4326")
            except Exception:                                                   # noqa: BLE001
                lat = lon = None
    else:
        raise ValueError("Choose latitude + longitude, x + y, or row + col columns.")
    ok = np.isfinite(fr) & np.isfinite(fc)
    r = np.where(ok, np.floor(np.where(ok, fr, 0)), -1).astype(np.int64)
    c = np.where(ok, np.floor(np.where(ok, fc, 0)), -1).astype(np.int64)
    inside = ok & (r >= 0) & (r < rows) & (c >= 0) & (c < cols)
    off = np.hypot(fr - (r + 0.5), fc - (c + 0.5))
    value = None
    if mapping.get("value"):
        value = pd.to_numeric(df[mapping["value"]], errors="coerce").to_numpy(float)
    cls = None
    if mapping.get("cls"):
        cls = df[mapping["cls"]].astype(str).str.strip().to_numpy(object)
    out = pd.DataFrame({
        "layer": layer,
        "lat": lat if lat is not None else np.full(n, np.nan),
        "lon": lon if lon is not None else np.full(n, np.nan),
        "row": r, "col": c,
        "value": value if value is not None else np.full(n, np.nan),
        "cls": cls if cls is not None else np.full(n, None, dtype=object),
        "offset": np.where(ok, off, np.nan),
        "inside": inside,
    })
    dup = out[inside].duplicated(["row", "col"]).sum()
    rep = {"layer": layer, "n": int(n), "placed": int(inside.sum()), "outside": int((ok & ~inside).sum()),
           "unreadable": int((~ok).sum()), "duplicates": int(dup),
           "max_offset_px": float(np.nanmax(out.loc[inside, "offset"])) if inside.any() else float("nan"),
           "pixel_size": _pixel_size(geo)}
    return out, rep


def _rowcol(geo, mx, my):
    a, b, c0, d, e, f = geo.transform
    det = a * e - b * d
    if abs(det) < 1e-18:
        raise ValueError("The image's geotransform is singular.")
    col = (e * (mx - c0) - b * (my - f)) / det
    row = (a * (my - f) - d * (mx - c0)) / det
    return row, col


def _xy(geo, row, col):
    a, b, c0, d, e, f = geo.transform
    col, row = np.asarray(col, float) + 0.5, np.asarray(row, float) + 0.5
    return a * col + b * row + c0, d * col + e * row + f


def _pixel_size(geo):
    if geo is None:
        return None
    a, b, _, d, e, _ = geo.transform
    return float(math.hypot(a, d))


def report_text(reps: Sequence[dict]) -> str:
    lines = []
    for r in reps:
        s = f"{r['layer']}: {r['placed']:,} of {r['n']:,} points placed"
        bits = []
        if r["outside"]:
            bits.append(f"{r['outside']:,} outside the image")
        if r["unreadable"]:
            bits.append(f"{r['unreadable']:,} without coordinates")
        if r["duplicates"]:
            bits.append(f"{r['duplicates']:,} on an already used pixel (averaged)")
        if bits:
            s += " (" + ", ".join(bits) + ")"
        mo = r.get("max_offset_px")
        if mo is not None and np.isfinite(mo):
            ps = r.get("pixel_size")
            s += f"; farthest from a pixel centre: {mo:.2f} px" + (f" ({mo * ps:.1f} map units)" if ps else "")
        lines.append(s)
    return "\n".join(lines)


# ------------------------------------------------------------------ layers -> images -> labels
def stack_layers(frames, rows: int, cols: int) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """({layer: value image}, {layer: class image (object, None = no point)}). Duplicate pixels: mean value,
    most frequent class."""
    import pandas as pd
    df = pd.concat(list(frames), ignore_index=True) if not isinstance(frames, pd.DataFrame) else frames
    df = df[df["inside"]]
    values, classes = {}, {}
    for layer, g in df.groupby("layer", sort=False):
        flat = g["row"].to_numpy(np.int64) * cols + g["col"].to_numpy(np.int64)
        v = g["value"].to_numpy(float)
        if np.isfinite(v).any():
            s = np.bincount(flat[np.isfinite(v)], weights=v[np.isfinite(v)], minlength=rows * cols)
            k = np.bincount(flat[np.isfinite(v)], minlength=rows * cols)
            img = np.full(rows * cols, np.nan)
            img[k > 0] = s[k > 0] / k[k > 0]
            values[layer] = img.reshape(rows, cols)
        cl = g["cls"].to_numpy(object)
        has = np.array([x is not None and str(x) not in ("", "nan", "None") for x in cl])
        if has.any():
            img = np.full(rows * cols, None, dtype=object)
            sub = pd.DataFrame({"p": flat[has], "c": cl[has]})
            mode = sub.groupby("p")["c"].agg(lambda s: s.value_counts().index[0])
            img[mode.index.to_numpy()] = mode.to_numpy()
            classes[layer] = img.reshape(rows, cols)
    return values, classes


def default_rule(values: Dict[str, np.ndarray], classes: Dict[str, np.ndarray]) -> dict:
    if classes and not values:
        return {"kind": "class"}
    if len(values) > 1:
        return {"kind": "dominant", "threshold": 0.05, "none_class": True, "none_name": "none"}
    layer = next(iter(values), None)
    return {"kind": "presence", "layer": layer, "threshold": 0.05}


def make_labels(values: Dict[str, np.ndarray], classes: Dict[str, np.ndarray], rule: dict, shape=None):
    """Label image from the layers with a rule. Returns (labels int32 (0 = unlabelled), names {id: name},
    counts {id: pixels})."""
    kind = rule.get("kind", "dominant")
    if shape is None:
        some = next(iter(values.values()), None)
        if some is None:
            some = next(iter(classes.values()))
        shape = some.shape
    rows, cols = shape
    lab = np.zeros((rows, cols), np.int32)
    names: Dict[int, str] = {}
    if kind == "dominant":
        layers = [k for k in values if k in (rule.get("layers") or values.keys())]
        if not layers:
            raise ValueError("No value layers to compare.")
        S = np.stack([values[k] for k in layers])
        covered = np.any(np.isfinite(S), axis=0)
        Z = np.where(np.isfinite(S), S, -np.inf)
        best = np.argmax(Z, axis=0)
        top = np.max(Z, axis=0)
        thr = float(rule.get("threshold", 0.0))
        hit = covered & (top >= thr) & np.isfinite(top)
        for i, k in enumerate(layers):
            names[i + 1] = k
            lab[hit & (best == i)] = i + 1
        if rule.get("none_class", True):
            nid = len(layers) + 1
            names[nid] = rule.get("none_name") or "none"
            lab[covered & ~hit] = nid
    elif kind == "presence":
        layer = rule.get("layer") or next(iter(values))
        v = values[layer]
        thr = float(rule.get("threshold", 0.0))
        cov = np.isfinite(v)
        names = {1: layer, 2: f"not {layer}"}
        lab[cov & (v >= thr)] = 1
        if rule.get("absent_class", True):
            lab[cov & (v < thr)] = 2
        else:
            names.pop(2)
    elif kind == "levels":
        layer = rule.get("layer") or next(iter(values))
        v = values[layer]
        edges = sorted(float(e) for e in (rule.get("edges") or [0.05, 0.15]))
        cov = np.isfinite(v)
        given = list(rule.get("level_names") or [])
        bins = np.digitize(np.where(cov, v, 0), edges)          # 0 .. len(edges)
        auto = [f"{layer} < {edges[0]:g}"] + [f"{layer} {a:g} to {b:g}" for a, b in zip(edges, edges[1:])] + \
               [f"{layer} >= {edges[-1]:g}"]
        for i in range(len(edges) + 1):
            names[i + 1] = given[i] if i < len(given) and given[i] else auto[i]
            lab[cov & (bins == i)] = i + 1
    elif kind == "class":
        cats: List[str] = []
        for layer, img in classes.items():
            flat = img.ravel()
            for v in flat[flat != None]:                             # noqa: E711
                if v not in cats:
                    cats.append(v)
        if not cats:
            raise ValueError("The tables have no class column.")
        cats = sorted(cats, key=_natural)
        for i, cname in enumerate(cats, 1):
            names[i] = str(cname)
        idx = {c: i for i, c in enumerate(cats, 1)}
        for img in classes.values():
            flat = img.ravel()
            m = np.array([x is not None for x in flat])
            out = lab.ravel()
            out[m & (out == 0)] = [idx[x] for x in flat[m & (out == 0)]]
            lab = out.reshape(rows, cols)
    else:
        raise ValueError(f"unknown label rule {kind!r}")
    counts = {i: int((lab == i).sum()) for i in names}
    return lab, names, counts


def _natural(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(s))]


def rule_text(rule: dict) -> str:
    k = rule.get("kind")
    if k == "dominant":
        s = f"dominant layer, value >= {rule.get('threshold', 0):g}"
        return s + (f", others '{rule.get('none_name') or 'none'}'" if rule.get("none_class", True) else "")
    if k == "presence":
        return f"{rule.get('layer')} present where value >= {rule.get('threshold', 0):g}"
    if k == "levels":
        return f"{rule.get('layer')} in levels split at {', '.join(f'{e:g}' for e in rule.get('edges') or [])}"
    if k == "class":
        return "class column"
    return str(k)


def small_classes(counts: Dict[int, int], min_pixels: int) -> List[int]:
    return [i for i, n in counts.items() if 0 < n < min_pixels]


# ------------------------------------------------------------------ to / from the library database
def points_to_records(frame) -> List[dict]:
    """Placed points (inside the image) -> rows for repository.save_ground_truth."""
    df = frame[frame["inside"]]
    out = []
    for t in df.itertuples(index=False):
        out.append({"layer": str(t.layer),
                    "lat": None if not np.isfinite(t.lat) else float(t.lat),
                    "lon": None if not np.isfinite(t.lon) else float(t.lon),
                    "row": int(t.row), "col": int(t.col),
                    "value": None if not np.isfinite(t.value) else float(t.value),
                    "cls": None if t.cls is None or str(t.cls) in ("", "nan", "None") else str(t.cls)})
    return out


def records_to_frame(points: Sequence[dict], geo, rows: int, cols: int, same_image: bool):
    """Stored points -> a placed frame for this image. Re-placed by lat/lon when the image is georeferenced,
    else by the stored row / col when it is the same image (size)."""
    import pandas as pd
    df = pd.DataFrame(list(points))
    if df.empty:
        raise ValueError("This ground truth has no points.")
    for k in ("lat", "lon", "value"):
        df[k] = pd.to_numeric(df.get(k), errors="coerce")
    if "cls" not in df:
        df["cls"] = None
    frames, reps = [], []
    use_geo = geo_usable(geo) and df["lat"].notna().all()
    for layer, g in df.groupby("layer", sort=False):
        g = g.reset_index(drop=True)
        if use_geo:
            fr, rep = place_points(g, {"lat": "lat", "lon": "lon", "value": "value", "cls": "cls"}, geo, rows, cols,
                                   layer)
        elif same_image:
            fr, rep = place_points(g, {"row": "row", "col": "col", "value": "value", "cls": "cls"}, None, rows, cols,
                                   layer)
        else:
            raise ValueError("The image is not georeferenced and has a different size than the image this ground "
                             "truth was made on, so its points cannot be placed.")
        frames.append(fr)
        reps.append(rep)
    return pd.concat(frames, ignore_index=True), reps
