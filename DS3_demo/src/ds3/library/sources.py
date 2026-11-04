"""Inputs for the library builder other than plain spectra files: ENVI ASCII ROI export, GeoJSON polygons."""
from __future__ import annotations
import json, re
from pathlib import Path
import numpy as np
from ..ingest import extract_roi, IngestError
from ..ingest.metadata import InvalidHeaderError, UnsupportedFormatError
from ..ingest.contract import Spectrum


def is_envi_roi(path) -> bool:
    try:
        with open(path, "r", errors="replace") as f:
            head = f.read(600)
    except OSError:
        return False
    return bool(re.search(r"^;\s*ENVI Output of ROIs", head, re.M | re.I))


def read_envi_roi_ascii(path, one_based=True):
    """ENVI 'Output ROIs to ASCII' text export -> list of (name, npts, [(row, col), ...]) with 0-based pixels.
    Columns after the header are: ID X Y ...; X = sample (col), Y = line (row), 1-based in ENVI."""
    rois, cur = [], None
    for line in Path(path).read_text(errors="replace").splitlines():
        s = line.strip()
        if not s: continue
        if s.startswith(";"):
            m = re.match(r";\s*ROI name:\s*(.+)$", s, re.I)
            if m:
                cur = dict(name=m.group(1).strip(), pix=[]); rois.append(cur)
            continue
        parts = s.split()
        if cur is None or len(parts) < 3: continue
        try:
            x, y = float(parts[1]), float(parts[2])
        except ValueError:
            continue
        d = 1 if one_based else 0
        cur["pix"].append((int(y) - d, int(x) - d))
    if not rois: raise InvalidHeaderError(f"{path}: no '; ROI name:' blocks - not an ENVI ASCII ROI export")
    return [(r["name"], len(r["pix"]), r["pix"]) for r in rois]


def spectra_from_roi_ascii(path, cube, one_based=True, category=None):
    """Mean spectrum (and std as uncertainty) per ROI, measured from `cube` at the ROI's pixels."""
    out = []
    for name, n, pix in read_envi_roi_ascii(path, one_based):
        mask = np.zeros((cube.rows, cube.cols), bool)
        for r, c in pix:
            if 0 <= r < cube.rows and 0 <= c < cube.cols: mask[r, c] = True
        if not mask.any():
            raise InvalidHeaderError(f"ROI '{name}': none of its {n} pixels fall inside the {cube.rows}x{cube.cols} cube "
                                     f"(wrong image, or pixel base - try one_based=False)")
        res = extract_roi(cube, mask=mask)
        sp = res.to_spectrum(name, purity="image_pixel", category=category)
        sp.metadata.update(roi_file=Path(path).name, roi_pixels_listed=n, roi_pixels_used=res.n_pixels, origin="image_pixel")
        out.append(sp)
    return out


def spectra_from_geojson(path, cube, name_key="name", category=None):
    """Polygon features in the CUBE's CRS (or EPSG:4326 if the file says so) -> mean spectrum per feature."""
    gj = json.loads(Path(path).read_text())
    feats = gj["features"] if gj.get("type") == "FeatureCollection" else [gj]
    out = []
    for i, ft in enumerate(feats):
        geom = ft.get("geometry", ft)
        props = ft.get("properties") or {}
        res = extract_roi(cube, polygon=geom, polygon_crs="EPSG:4326" if cube.crs else None)
        if res.n_pixels == 0:
            raise InvalidHeaderError(f"feature {i} of {Path(path).name} covers no pixels of the cube")
        sp = res.to_spectrum(str(props.get(name_key, f"{Path(path).stem}_{i}")), purity="image_pixel", category=category)
        sp.metadata.update(geojson=Path(path).name, origin="image_pixel")
        out.append(sp)
    return out


def read_evf(path, *_, **__):
    raise UnsupportedFormatError(f"{Path(path).name}: ENVI .evf vector files are not supported yet "
                                 "(format not verified). In ENVI export the vectors to a shapefile/GeoJSON, or the ROI to ASCII.")


# ---- label maps (ground-truth classes) -> one mean spectrum per class -------------------------------------------------
PAVIA_U_CLASSES = ["Asphalt", "Meadows", "Gravel", "Trees", "Painted metal sheets", "Bare soil", "Bitumen",
                   "Self-blocking bricks", "Shadows"]      # published class names of the Pavia University ground truth


def read_label_map(path, variable=None):
    """2-D integer class map from .mat (variable with 'gt'/'label'/'class' in its name, or the only 2-D array), .npy or
    single-band .tif. 0 = unlabelled. Returns (labels[int], variable_name)."""
    p = Path(path); ext = p.suffix.lower()
    if ext == ".npy":
        a, name = np.load(p), p.stem
    elif ext in (".tif", ".tiff"):
        import rasterio
        with rasterio.open(p) as ds:
            a, name = ds.read(1), p.stem
    elif ext == ".mat":
        from ..ingest.readers.mat import _load_all
        arrs = {k: v for k, v in _load_all(p).items() if v.ndim == 2}
        if variable and variable in arrs: name = variable
        else:
            pref = [k for k in arrs if any(t in k.lower() for t in ("gt", "label", "class", "truth"))]
            c = pref or list(arrs)
            if len(c) != 1 and not pref:
                raise InvalidHeaderError(f"{p.name}: cannot tell which 2-D variable is the label map: {[(k, v.shape) for k, v in arrs.items()]}")
            name = c[0]
        a = arrs[name]
    else:
        raise UnsupportedFormatError(f"{p.name}: label maps must be .mat, .npy or .tif")
    if a.ndim != 2:
        raise InvalidHeaderError(f"{p.name}: label map must be 2-D, got shape {a.shape}")
    return np.rint(a).astype(np.int64), name


def spectra_from_labels(cube, labels, names=None, category=None, min_pixels=1):
    """Mean spectrum (std as uncertainty) of every class id > 0. names: {id: name} or list (id 1 = first)."""
    if labels.shape != (cube.rows, cube.cols):
        raise InvalidHeaderError(f"label map is {labels.shape} but the image is {(cube.rows, cube.cols)}; they must cover the same pixels")
    if isinstance(names, (list, tuple)):
        names = {i + 1: n for i, n in enumerate(names)}
    names = names or {}
    out = []
    for cid in sorted(int(c) for c in np.unique(labels) if c > 0):
        mask = labels == cid
        if mask.sum() < min_pixels:
            continue
        res = extract_roi(cube, mask=mask)
        if res.n_pixels == 0:
            continue
        sp = res.to_spectrum(names.get(cid, f"class {cid}"), purity="image_pixel", category=category)
        sp.metadata.update(label_id=cid, origin="image_pixel", roi_pixels_used=res.n_pixels)
        out.append(sp)
    if not out:
        raise InvalidHeaderError("the label map has no labelled (>0) pixels with valid data")
    return out
