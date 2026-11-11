"""Registered hyperspectral images: register a file, re-open it, preview it, read pixels, classify with a library.

The heavy lifting is done by the ds3 core (`ds3.ingest`, `ds3.library.image_ops`); this module only connects it to the
database tables of this project. Everything needs the ds3 package (pip install -e ..).
"""
from __future__ import annotations
import base64, io, json, uuid
from collections import OrderedDict
from pathlib import Path

import numpy as np
from sqlalchemy.orm import Session

from ..models import Image, Library, Dataset, Material, Sample, Spectrum, DataFile
from .readers import HAVE_DS3

if HAVE_DS3:
    from ds3.ingest import open_cube
    from ds3.library import image_ops
    from ds3.pipeline import sha256_of

_CUBES: "OrderedDict[str, object]" = OrderedDict()      # small LRU of opened cubes (lazy readers, cheap to keep)
_MAX_OPEN = 6


def _need_ds3():
    if not HAVE_DS3:
        raise ValueError("image support needs the ds3 package: pip install -e <path>/ds3-platform")


def _jsonable(o):
    return json.loads(json.dumps(o, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x)))


def register_image(db: Session, path: Path, **reader_kw) -> dict:
    """Open `path` with the ds3 readers and store its description. Same file again -> 'duplicate'."""
    _need_ds3()
    path = Path(path)
    digest = sha256_of(path) if path.is_file() else sha256_of_dir(path)
    old = db.query(Image).filter(Image.sha256 == digest).first()
    if old is not None:
        return dict(file=path.name, status="duplicate", kind="image", spectra=0, id=str(old.id), detail="image already registered")
    kw = {k: v for k, v in reader_kw.items() if v is not None}
    cube = open_cube(path, qc=False, **kw)
    notes = cube.metadata.get("notes", [])
    img = Image(id=uuid.uuid4(), name=path.stem if path.is_file() else path.name, path=str(path.resolve()), sha256=digest,
                format=cube.source_format, rows=cube.rows, cols=cube.cols, bands=cube.bands, unit=cube.unit,
                sensor=cube.sensor, wl_min_nm=float(cube.wavelengths_nm[0]), wl_max_nm=float(cube.wavelengths_nm[-1]),
                wavelengths=[float(x) for x in cube.wavelengths_nm], bad_band_mask=[bool(x) for x in cube.bad_band_mask],
                reader_kw=_jsonable(kw), image_metadata=_jsonable({"notes": notes}))
    db.add(img)
    db.flush()
    return dict(file=path.name, status="imported", kind="image", spectra=0, id=str(img.id), detail=cube.describe())


def sha256_of_dir(p: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    for f in sorted(p.rglob("*")):
        if f.is_file():
            h.update(f.name.encode()); h.update(str(f.stat().st_size).encode())
    return h.hexdigest()


def open_image(img: Image):
    _need_ds3()
    key = str(img.id)
    if key in _CUBES:
        _CUBES.move_to_end(key)
        return _CUBES[key]
    kw = dict(img.reader_kw or {})
    kw.setdefault("unit", img.unit)
    if not Path(img.path).exists():
        raise ValueError(f"the image file is no longer at {img.path}")
    cube = open_cube(img.path, qc=False, **kw)
    if img.bad_band_mask and len(img.bad_band_mask) == cube.bands:
        cube.bad_band_mask = np.asarray(img.bad_band_mask, bool)
    _CUBES[key] = cube
    while len(_CUBES) > _MAX_OPEN:
        _CUBES.popitem(last=False)
    return cube


def forget(image_id):
    _CUBES.pop(str(image_id), None)


def _png(arr) -> str:
    import matplotlib.image as mpimg
    buf = io.BytesIO()
    mpimg.imsave(buf, arr, format="png")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def preview(img: Image, bands=None, max_px=300_000, alpha=False):
    """bands: three 1-based band numbers or None (true colour). alpha=True makes no-data (black) pixels transparent (for the map overlay)."""
    cube = open_image(img)
    idx = None
    if bands:
        idx = [int(b) - 1 for b in bands]
        if len(idx) != 3 or not all(0 <= b < cube.bands for b in idx):
            raise ValueError(f"bands must be three numbers between 1 and {cube.bands}")
    p = image_ops.preview_rgb(cube, idx, max_px)
    rgb = p["rgb"]
    if alpha:
        rgb = np.dstack([rgb, np.where(rgb.sum(axis=2) == 0, 0, 255).astype(np.uint8)])
    h, w = rgb.shape[:2]
    return dict(image=_png(rgb), width=w, height=h, step=p["step"], bands=[b + 1 for b in p["bands"]], label=p["label"],
                rows=cube.rows, cols=cube.cols)


def _crs_from_map_info(mi):
    try:
        p = [x.strip() for x in str(mi).strip("{} ").split(",")]
        if p[0].upper() == "UTM":
            return f"EPSG:{(32600 if p[8].lower().startswith('n') else 32700) + int(p[7])}"
        if p[0].lower().startswith("geographic"):
            return "EPSG:4326"
    except Exception:
        pass
    return None


def geo(img: Image):
    """Where the file says the image is: corner coordinates in lat/lon (None when the file carries no map information)."""
    cube = open_image(img)
    out = dict(has_geo=False, rows=cube.rows, cols=cube.cols, crs=None, corners=None, bounds=None, note=None)
    tr = getattr(cube, "transform", None)
    if tr is None:
        from . import georef as gr
        g = gr.saved(img)                                    # control points set in the Image module
        if g:
            lon, lat = gr.apply_affine(g["affine"], [0, 0, cube.rows, cube.rows], [0, cube.cols, cube.cols, 0], centre=False)
            out.update(has_geo=True, crs="EPSG:4326 (control points)", source="gcp", rmse_m=g.get("rmse_m"),
                       corners=[dict(lat=float(la), lon=float(lo)) for la, lo in zip(lat, lon)],
                       bounds=[[float(min(lat)), float(min(lon))], [float(max(lat)), float(max(lon))]])
            return out
        out["note"] = "this file carries no map information"
        return out
    crs = getattr(cube, "crs", None) or _crs_from_map_info((getattr(cube, "metadata", None) or {}).get("envi_map_info"))
    if not crs:
        out["note"] = "the file has a pixel transform but no coordinate system"
        return out
    try:
        from rasterio.crs import CRS
        from rasterio.warp import transform as wt
        src = CRS.from_user_input(crs)
        a, b, c, d, e, f = [float(x) for x in tr[:6]]
        px = [(0, 0), (cube.cols, 0), (cube.cols, cube.rows), (0, cube.rows)]       # top-left, top-right, bottom-right, bottom-left
        xs = [a * col + b * row + c for col, row in px]; ys = [d * col + e * row + f for col, row in px]
        lon, lat = wt(src, "EPSG:4326", xs, ys)
        out.update(has_geo=True, crs=src.to_string()[:60], corners=[dict(lat=float(la), lon=float(lo)) for la, lo in zip(lat, lon)],
                   bounds=[[float(min(lat)), float(min(lon))], [float(max(lat)), float(max(lon))]])
    except Exception as ex:
        out["note"] = f"coordinates could not be converted ({ex})"
    return out


def _roi_blocks(cube, polygon, max_px=120_000):
    """Yield (pixels inside the polygon as float64 (N, bands)) block by block, plus the stride used.
    Large ROIs are sub-sampled on a regular grid so that at most ~max_px pixels are read."""
    from matplotlib.path import Path as MPath
    pts = np.asarray(polygon, float)
    if pts.ndim != 2 or pts.shape[0] < 3 or pts.shape[1] != 2:
        raise ValueError("an ROI needs at least 3 corner points")
    r0 = max(0, int(np.floor(pts[:, 0].min()))); r1 = min(cube.rows, int(np.ceil(pts[:, 0].max())) + 1)
    c0 = max(0, int(np.floor(pts[:, 1].min()))); c1 = min(cube.cols, int(np.ceil(pts[:, 1].max())) + 1)
    if r1 <= r0 or c1 <= c0:
        raise ValueError("the ROI lies outside the image")
    stride = max(1, int(np.ceil(np.sqrt((r1 - r0) * (c1 - c0) / max_px))))
    path = MPath(pts)

    def gen():
        for a in range(r0, r1, 64 * stride):
            b = min(r1, a + 64 * stride)
            blk = cube.read(rows=(a, b), cols=(c0, c1))
            rr = np.arange(a, b, stride); cc = np.arange(c0, c1, stride)
            sub = blk[rr - a][:, cc - c0, :]
            gy, gx = np.meshgrid(rr, cc, indexing="ij")
            inside = path.contains_points(np.c_[gy.ravel(), gx.ravel()]).reshape(gy.shape)
            sel = sub[inside].astype(np.float64)
            if sel.size:
                yield sel
    return gen(), stride


def roi_spectrum(img: Image, polygon):
    """Mean spectrum of the pixels inside a polygon [[row, col], ...] given in full-resolution pixel coordinates.
    Also returns the per-band standard deviation (sample, n - 1), minimum and maximum over those pixels."""
    cube = open_image(img)
    blocks, stride = _roi_blocks(cube, polygon)
    total = np.zeros(cube.bands); total2 = np.zeros(cube.bands); cnt = np.zeros(cube.bands); n_pix = 0
    lo = np.full(cube.bands, np.inf); hi = np.full(cube.bands, -np.inf)
    for sel in blocks:
        ok = np.isfinite(sel)
        z = np.where(ok, sel, 0)
        total += z.sum(axis=0); total2 += (z * z).sum(axis=0); cnt += ok.sum(axis=0); n_pix += int(ok.any(axis=1).sum())
        lo = np.minimum(lo, np.where(ok, sel, np.inf).min(axis=0)); hi = np.maximum(hi, np.where(ok, sel, -np.inf).max(axis=0))
    if n_pix == 0:
        raise ValueError("the ROI contains no valid pixels")
    vals = np.where(cnt > 0, total / np.maximum(cnt, 1), np.nan)
    var = np.where(cnt > 1, (total2 - cnt * vals ** 2) / np.maximum(cnt - 1, 1), np.nan)
    sd = np.sqrt(np.maximum(var, 0))
    bad = np.asarray(cube.bad_band_mask, bool)
    clean = lambda a: [None if (not np.isfinite(x)) else float(x) for x in np.where(bad, np.nan, a)]
    v = np.where(bad, np.nan, vals)
    return dict(row=None, col=None, size=None, n_pixels=n_pix, stride=stride, unit=cube.unit,
                wavelengths=[float(x) for x in cube.wavelengths_nm],
                values=[None if not np.isfinite(x) else float(x) for x in v], std=clean(sd),
                min=clean(np.where(np.isfinite(lo), lo, np.nan)), max=clean(np.where(np.isfinite(hi), hi, np.nan)),
                bad_bands=[bool(x) for x in cube.bad_band_mask]), vals


def target(img: Image, *, row=None, col=None, size=1, polygon=None):
    """The spectrum an action works on: an ROI mean when a polygon is given, otherwise the pixel / window."""
    if polygon:
        return roi_spectrum(img, polygon)
    return pixel(img, row, col, size)


def pixel(img: Image, row: int, col: int, size: int = 1):
    cube = open_image(img)
    vals, n = image_ops.pixel_spectrum(cube, row, col, size)
    if n == 0:
        raise ValueError(f"pixel ({row}, {col}) is nodata in every band")
    v = np.where(np.asarray(cube.bad_band_mask, bool), np.nan, vals)
    return dict(row=row, col=col, size=size, n_pixels=n, unit=cube.unit, wavelengths=[float(x) for x in cube.wavelengths_nm],
                values=[None if not np.isfinite(x) else float(x) for x in v],
                bad_bands=[bool(x) for x in cube.bad_band_mask]), vals


def library_spectra(db: Session, *, library_id=None, spectrum_ids=None, limit=5000):
    """[{id, material, category, unit, wavelengths_nm, values}] for a library or a list of spectrum ids."""
    from ..routes._serialize import wavelengths_of
    q = (db.query(Spectrum, Material)
         .join(Sample, Spectrum.sample_id == Sample.id).join(Material, Sample.material_id == Material.id)
         .join(Dataset, Sample.dataset_id == Dataset.id))
    if spectrum_ids:
        q = q.filter(Spectrum.id.in_([uuid.UUID(str(i)) for i in spectrum_ids]))
    elif library_id:
        q = q.filter(Dataset.library_id == uuid.UUID(str(library_id)))
    else:
        raise ValueError("choose a library or some spectra")
    out = []
    for sp, m in q.order_by(Material.name).limit(limit).all():
        out.append(dict(id=str(sp.id), material=m.name, category=m.category, unit=sp.measurement_unit,
                        wavelengths_nm=np.asarray(wavelengths_of(sp), float),
                        values=np.array([np.nan if x is None else x for x in sp.values], float)))
    return out


def best_matches(img: Image, vals, spectra, *, ignore_unit=False, top_k=10, min_overlap=0.5):
    cube = open_image(img)
    q = np.where(np.asarray(cube.bad_band_mask, bool), np.nan, vals)
    if not ignore_unit:
        spectra = [s for s in spectra if s["unit"] == cube.unit]
    if not spectra:
        return [], (f"no library spectrum is in the image's unit ({cube.unit}); tick 'ignore unit differences' "
                    "to compare shapes anyway (spectral angle ignores overall brightness)")
    M = image_ops.to_cube_grid(spectra, cube.wavelengths_nm)
    res = []
    for s, row in zip(spectra, M):
        m = np.isfinite(q) & np.isfinite(row)
        if m.mean() < min_overlap or m.sum() < 5:
            continue
        a, b = q[m], row[m]
        d = np.linalg.norm(a) * np.linalg.norm(b)
        if d > 0:
            res.append(dict(spectrum_id=s["id"], material=s["material"], category=s["category"], unit=s["unit"],
                            angle_deg=float(np.degrees(np.arccos(np.clip(a @ b / d, -1, 1)))), overlap=float(m.mean())))
    res.sort(key=lambda r: r["angle_deg"])
    return res[:top_k], None


def sam_overlay(img: Image, spectra, max_angle_deg=None):
    from matplotlib import colormaps
    cube = open_image(img)
    M = image_ops.to_cube_grid(spectra, cube.wavelengths_nm)
    r = image_ops.sam_map(cube, M, [s["material"] for s in spectra], max_angle_deg=max_angle_deg)
    n = len(spectra)
    cm = colormaps["tab20" if n > 10 else "tab10"]
    cols = [cm(i % cm.N) for i in range(n)]
    idx = r["index"]
    h, w = idx.shape
    ov = np.zeros((h, w, 4), np.float32)
    for k, c in enumerate(cols):
        ov[idx == k] = (*c[:3], 0.62)
    legend = [dict(name=spectra[k]["material"], color="#%02x%02x%02x" % tuple(int(255 * x) for x in cols[k][:3]),
                   pixels=int((idx == k).sum()), unit=spectra[k]["unit"]) for k in range(n)]
    return dict(overlay=_png((ov * 255).astype(np.uint8)), width=w, height=h, step=r["step"], bands_used=r["bands_used"],
                legend=legend, unmatched=int((idx < 0).sum()), unit_mismatch=sorted({s["unit"] for s in spectra if s["unit"] != cube.unit}))


def save_pixel(db: Session, img: Image, *, row=None, col=None, size=1, library, name, category=None, polygon=None):
    cube = open_image(img)
    p, vals = target(img, row=row, col=col, size=size, polygon=polygon)
    from .importer import _get_library, slugify
    lib = _get_library(db, library)
    ds_name = f"pixels from {img.name}"
    ds = db.query(Dataset).filter(Dataset.library_id == lib.id, Dataset.slug == slugify(ds_name)).first()
    if ds is None:
        ds = Dataset(id=uuid.uuid4(), library_id=lib.id, name=ds_name, slug=slugify(ds_name), source_name=img.name,
                     description=f"Spectra measured from image {img.name}", dataset_metadata={"origin": "image_pixel", "image_id": str(img.id)})
        db.add(ds); db.flush()
    mat_name = f"{name} (ROI, {p['n_pixels']} px)" if polygon else f"{name} ({row},{col})"
    mat = Material(id=uuid.uuid4(), library_id=lib.id, name=mat_name, category=category,
                   material_metadata={"origin": "image_pixel", "image": img.name})
    db.add(mat); db.flush()
    s = Sample(id=uuid.uuid4(), material_id=mat.id, dataset_id=ds.id, name=mat_name, sample_number=1,
               sample_metadata={"image_id": str(img.id), "row": row, "col": col, "window": size, "roi": bool(polygon)})
    db.add(s); db.flush()
    wl = [float(x) for x in cube.wavelengths_nm]
    good = [i for i, b in enumerate(cube.bad_band_mask) if not b and np.isfinite(vals[i])]
    if len(good) < 3:
        raise ValueError("fewer than 3 usable bands at this pixel")
    w = [wl[i] for i in good]; v = [float(vals[i]) for i in good]
    from .importer import grid_info
    step, uniform = grid_info(w)
    db.add(Spectrum(id=uuid.uuid4(), sample_id=s.id, wavelength_start=w[0], wavelength_end=w[-1], wavelength_step=step,
                    measurement_unit=cube.unit, values=v, wavelengths=None if uniform else w,
                    spectrum_metadata={"purity": "image_pixel", "wavelength_unit": "nm", "n_pixels": p["n_pixels"],
                                       "source_image": img.name, "row": row, "col": col, "roi_polygon": polygon, "point_count": len(v)}))
    db.flush()
    return dict(library=lib.name, library_id=str(lib.id), material=mat_name, material_id=str(mat.id), bands=len(v))


def save_label_spectra(db: Session, img: Image, labels_path: Path, library: str, names=None):
    """Label map (ground truth) -> one mean spectrum per class, stored as materials of `library`."""
    cube = open_image(img)
    from ds3.library.sources import read_label_map, spectra_from_labels, PAVIA_U_CLASSES
    labels, var = read_label_map(labels_path)
    if names is None and labels.max() <= len(PAVIA_U_CLASSES) and "pavia" in (str(labels_path) + img.name).lower():
        names = PAVIA_U_CLASSES
    spectra = spectra_from_labels(cube, labels, names)
    from .importer import _get_library, slugify, grid_info
    lib = _get_library(db, library)
    ds_name = f"classes from {img.name}"
    ds = db.query(Dataset).filter(Dataset.library_id == lib.id, Dataset.slug == slugify(ds_name)).first()
    if ds is None:
        ds = Dataset(id=uuid.uuid4(), library_id=lib.id, name=ds_name, slug=slugify(ds_name), source_name=Path(labels_path).name,
                     description=f"Class-mean spectra of image {img.name} from {Path(labels_path).name}",
                     dataset_metadata={"origin": "image_pixel", "image_id": str(img.id)})
        db.add(ds); db.flush()
    out = []
    for sp in spectra:
        mat = Material(id=uuid.uuid4(), library_id=lib.id, name=sp.material, category=None,
                       material_metadata={"origin": "image_pixel", "image": img.name, "pixels": sp.metadata.get("roi_pixels_used")})
        db.add(mat); db.flush()
        smp = Sample(id=uuid.uuid4(), material_id=mat.id, dataset_id=ds.id, name=sp.material, sample_number=1,
                     sample_metadata={"image_id": str(img.id), "label_id": sp.metadata.get("label_id")})
        db.add(smp); db.flush()
        w = [float(x) for x in sp.wavelengths_nm]; v = [None if not np.isfinite(x) else float(x) for x in sp.values]
        keep = [i for i, x in enumerate(v) if x is not None]
        w = [w[i] for i in keep]; v = [v[i] for i in keep]
        step, uniform = grid_info(w)
        db.add(Spectrum(id=uuid.uuid4(), sample_id=smp.id, wavelength_start=w[0], wavelength_end=w[-1], wavelength_step=step,
                        measurement_unit=sp.unit, values=v, wavelengths=None if uniform else w,
                        spectrum_metadata={"purity": "image_pixel", "wavelength_unit": "nm", "source_image": img.name,
                                           "n_pixels": sp.metadata.get("roi_pixels_used"), "label_id": sp.metadata.get("label_id")}))
        out.append(sp.material)
    db.flush()
    return dict(library=lib.name, library_id=str(lib.id), classes=out)
