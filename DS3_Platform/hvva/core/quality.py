"""Data-quality vocabulary shared by scenes, the importer and the library database (no GUI code).

    bad bands   water-absorption windows, bands the file flags (ENVI ``bbl``), bands with no data at all.
                They are FLAGGED, never deleted: the cube keeps every band; spectra taken from a scene
                (pixel, ROI, class mean, aggregation, library saves) get NaN = a gap there.
    units       what the values are: reflectance | radiance | dn | transmittance | emissivity | unknown.
                A library holds one unit (reflectance and radiance are never mixed or converted).
    purity      where a spectrum comes from: ground_measured (lab / field / reference library),
                image_pixel (taken from an image: pixel, ROI, class mean), synthetic (made up / simulated).
    containers  a .zip holding an image product (EnMAP L2A ...) is unpacked to a cache folder and the
                spectral image inside is opened; ancillary rasters (AVIRIS-NG geometry, EnMAP quicklooks /
                quality masks) are recognised and refused with the reason.
"""
from __future__ import annotations

import hashlib
import os
import re
import zipfile
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

WATER_WINDOWS: Tuple[Tuple[float, float], ...] = ((1340.0, 1460.0), (1790.0, 1960.0))
UNITS = ("reflectance", "radiance", "dn", "transmittance", "emissivity", "unknown")
PURITY = ("ground_measured", "image_pixel", "synthetic")


# ================================================================== bad bands
def parse_windows(text: str) -> Tuple[Tuple[float, float], ...]:
    """'1340-1460, 1790-1960' -> ((1340, 1460), (1790, 1960)).  Empty text -> ()."""
    out = []
    for part in re.split(r"[;,]", text or ""):
        part = part.strip()
        if not part:
            continue
        m = re.match(r"^\s*(\d+(?:\.\d+)?)\s*(?:-|–|to)\s*(\d+(?:\.\d+)?)\s*$", part)
        if not m:
            raise ValueError(f"cannot read the range '{part}' - write it like 1340-1460")
        lo, hi = sorted((float(m.group(1)), float(m.group(2))))
        out.append((lo, hi))
    return tuple(out)


def format_windows(windows) -> str:
    return ", ".join(f"{lo:g}-{hi:g}" for lo, hi in windows or ())


def water_mask(wavelengths, windows=WATER_WINDOWS) -> Optional[np.ndarray]:
    if wavelengths is None:
        return None
    wl = np.asarray(wavelengths, float)
    m = np.zeros(wl.shape, bool)
    for lo, hi in windows or ():
        m |= (wl >= lo) & (wl <= hi)
    return m


def empty_bands(cube: np.ndarray, max_rows: int = 256) -> np.ndarray:
    """Bands with no finite value (checked on up to *max_rows* evenly spaced rows - fast on big cubes)."""
    h = cube.shape[0]
    rows = np.unique(np.linspace(0, h - 1, min(h, max_rows)).astype(int))
    return ~np.isfinite(cube[rows]).any(axis=(0, 1))


def spans(wavelengths, mask) -> List[Tuple[float, float]]:
    """Contiguous runs of masked bands as (first, last) wavelength - for shading plots and reports."""
    if wavelengths is None or mask is None or not np.any(mask):
        return []
    wl = np.asarray(wavelengths, float)
    m = np.asarray(mask, bool)
    out, start = [], None
    for i, v in enumerate(m):
        if v and start is None:
            start = i
        if start is not None and (not v or i == len(m) - 1):
            end = i if v else i - 1
            out.append((float(wl[start]), float(wl[end])))
            start = None
    return out


def apply_mask(X, mask):
    """Copy of spectra X (..., bands) with the masked bands set to NaN."""
    X = np.array(X, dtype=np.float32 if np.asarray(X).dtype != np.float64 else np.float64, copy=True)
    if mask is not None and np.any(mask):
        X[..., np.asarray(mask, bool)] = np.nan
    return X


def bbl_from_header(text: str, n_bands: int) -> Optional[np.ndarray]:
    """ENVI 'bbl' (1 = good, 0 = bad) -> bad-band mask, or None."""
    vals = [v.strip() for v in str(text).strip().strip("{}").replace("\n", " ").split(",") if v.strip()]
    if len(vals) != n_bands:
        return None
    try:
        return np.array([float(v) == 0 for v in vals], bool)
    except ValueError:
        return None


# ================================================================== units
_UNIT_WORDS = [
    ("reflectance", re.compile(r"reflect|\brfl\b|_rfl|_corr|\brefl\b|albedo|surface ?reflectance|l2a\b", re.I)),
    ("radiance", re.compile(r"radiance|\brdn\b|_rdn|\brad\b|w/?\s*\(?m\^?2|µw|uw/cm|mw/|l1b\b|l1c\b", re.I)),
    ("dn", re.compile(r"\bdn\b|digital ?number|counts?\b|raw ?counts|\badu\b", re.I)),
    ("transmittance", re.compile(r"transmitt", re.I)),
    ("emissivity", re.compile(r"emissiv", re.I)),
]


def normalize_unit(value) -> Optional[str]:
    """Free text ('Reflectance', 'percent reflectance', 'W/m2/sr/um', 'counts', 'rdn') -> a UNITS word."""
    if value is None:
        return None
    s = str(value).strip().lower()
    if not s:
        return None
    if s in UNITS:
        return s
    if s in ("refl", "rfl", "r", "fraction"):
        return "reflectance"
    if s in ("rad", "rdn"):
        return "radiance"
    if s in ("counts", "count", "digital number", "digital numbers"):
        return "dn"
    if s in ("mixed", "none", "n/a", "na", "-"):
        return "unknown"
    for unit, pat in _UNIT_WORDS:
        if pat.search(s):
            return unit
    return "unknown"


def unit_from_name(name: str) -> Optional[str]:
    """AVIRIS / EnMAP / generic file-name hints, or None."""
    n = os.path.basename(str(name)).lower()
    if re.search(r"(_|-)(rfl|corr|refl|reflectance)(_|-|\.|$)|l2a", n):
        return "reflectance"
    if re.search(r"(_|-)(rdn|rad|radiance)(_|-|\.|$)|l1b|l1c", n):
        return "radiance"
    return None


def unit_from_values(cube: np.ndarray, max_rows: int = 128) -> Tuple[str, str]:
    """(unit, reason) from the value range alone - only says reflectance for 0-1.5 floats."""
    h = cube.shape[0]
    rows = np.unique(np.linspace(0, h - 1, min(h, max_rows)).astype(int))
    v = np.asarray(cube[rows], float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return "unknown", "no finite values"
    lo, hi = np.percentile(v, [1, 99])
    if -0.2 <= lo and hi <= 1.5:
        return "reflectance", f"assumed from the values ({lo:.2f}-{hi:.2f}) - change it if wrong"
    return "unknown", f"values {lo:.4g}-{hi:.4g}: could be scaled reflectance, radiance or DN - set the unit"


def units_compatible(a: Optional[str], b: Optional[str]) -> bool:
    """Two known, different units do not mix; unknown / None mixes with anything (and is reported)."""
    a, b = normalize_unit(a), normalize_unit(b)
    return a in (None, "unknown") or b in (None, "unknown") or a == b


def common_unit(units: Iterable[Optional[str]], what: str = "these spectra") -> str:
    """One unit for a set of sources, or ValueError if two known units differ."""
    known = sorted({u for u in (normalize_unit(x) for x in units) if u not in (None, "unknown")})
    if len(known) > 1:
        raise ValueError(f"{what} mix {' and '.join(known)} - reflectance and radiance (or DN) are never "
                         "combined; set the unit of each scene in the Scenes tab or use separate libraries")
    return known[0] if known else "unknown"


def normalize_purity(value) -> Optional[str]:
    if value is None:
        return None
    s = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if not s:
        return None
    if s in PURITY:
        return s
    if s in ("ground", "measured", "lab", "field", "reference", "library", "ground_truth", "groundmeasured"):
        return "ground_measured"
    if s in ("image", "pixel", "roi", "image_derived", "imagepixel", "scene", "endmember"):
        return "image_pixel"
    if s in ("simulated", "synthetic_data", "model", "modelled", "modeled", "fake", "demo"):
        return "synthetic"
    raise ValueError(f"purity must be one of {', '.join(PURITY)} (got {value!r})")


# ================================================================== ancillary files / containers
_AVIRIS_ANC = re.compile(r"(_|-)(glt|igm|loc|obs|obs_ort|loc_ort|h2o)$", re.I)


def ancillary_reason(path) -> Optional[str]:
    """Why *path* is not a spectral image (AVIRIS-NG geometry rasters, EnMAP quicklooks / masks), or None."""
    base = os.path.basename(str(path).rstrip("/\\"))
    stem = re.sub(r"\.(hdr|img|dat|tif|tiff)$", "", base, flags=re.I)
    m = _AVIRIS_ANC.search(stem)
    if m and re.match(r"^(ang|f)\d{6}", base.lower()):
        kind = m.group(2).lower()
        what = {"glt": "geometry lookup table", "igm": "input geometry (map coordinates)", "loc": "location",
                "obs": "observation geometry", "obs_ort": "orthorectified observation geometry",
                "loc_ort": "orthorectified location", "h2o": "water-vapour retrieval"}.get(kind, kind)
        return f"AVIRIS {what} raster ({kind}), not a spectral image - open the *_rdn_* or *_rfl_/_corr_* file"
    up = base.upper()
    if "ENMAP" in up and re.search(r"QL_(VNIR|SWIR)|QUICKLOOK", up):
        return "EnMAP quicklook picture (8-bit RGB), not spectral data - open the SPECTRAL_IMAGE file"
    if "ENMAP" in up and re.search(r"QL_QUALITY|QL_PIXELMASK|PIXELMASK", up):
        return "EnMAP quality / mask layer, not spectral data - open the SPECTRAL_IMAGE file"
    return None


_IMG_EXT = (".tif", ".tiff", ".hdr", ".nc", ".nc4", ".h5", ".hdf5", ".he5", ".mat", ".npz", ".npy")
CACHE_DIR = os.path.join(os.path.expanduser("~"), ".hvva", "extracted")


def _safe_name(name: str) -> str:
    base = os.path.basename(name.replace("\\", "/"))
    if not base or base in (".", "..") or ":" in base:
        raise ValueError(f"unsafe file name in the zip: {name!r}")
    return base


def _pick_image(names: Sequence[str]) -> Tuple[Optional[str], List[str]]:
    """(main image member, members to extract with it) from the names inside a zip."""
    files = [n for n in names if not n.endswith("/") and not os.path.basename(n).startswith(".")]
    up = {n: os.path.basename(n).upper() for n in files}
    spectral = [n for n in files if "SPECTRAL_IMAGE" in up[n] and up[n].endswith((".TIF", ".TIFF", ".HDR"))]
    if spectral:
        hdr = [n for n in spectral if up[n].endswith(".HDR")]
        main = hdr[0] if hdr else spectral[0]
        folder = os.path.dirname(main)
        prefix = up[main].split("SPECTRAL_IMAGE")[0]
        extra = [n for n in files if os.path.dirname(n) == folder and up[n].startswith(prefix) and (
            "SPECTRAL_IMAGE" in up[n] or up[n].endswith("METADATA.XML"))]
        return main, sorted(set(extra) | {main})
    cands = [n for n in files if os.path.splitext(n)[1].lower() in _IMG_EXT and not ancillary_reason(n)]
    hdrs = [n for n in cands if n.lower().endswith(".hdr")]
    if hdrs:
        main = hdrs[0]
        base = os.path.splitext(main)[0]
        extra = [n for n in files if os.path.splitext(n)[0] == base or n.startswith(base + ".")]
        return main, sorted(set(extra) | {main})
    others = [n for n in cands if not n.lower().endswith(".hdr")]
    if others:
        return others[0], [others[0]]
    return None, []


def resolve_input(path) -> Tuple[str, str]:
    """(path to open, note).  A .zip with an image product is unpacked to ~/.hvva/extracted/ (once - a cached
    copy is reused) and the spectral image inside is returned.  Ancillary rasters raise ValueError."""
    path = str(path)
    if not path.lower().endswith(".zip"):
        why = ancillary_reason(path)
        if why:
            raise ValueError(why)
        return path, ""
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        main, extract = _pick_image(names)
        note_inner = ""
        if main is None:                                             # EnMAP deliveries can nest the product zip
            inner = [n for n in names if n.lower().endswith(".zip")]
            for n in inner:
                with z.open(n) as fh:
                    import io
                    data = io.BytesIO(fh.read())
                with zipfile.ZipFile(data) as zi:
                    m2, ex2 = _pick_image(zi.namelist())
                    if m2:
                        st = os.stat(path)
                        key = hashlib.sha1(f"{os.path.abspath(path)}|{st.st_size}|{st.st_mtime}|{n}".encode()).hexdigest()[:10]
                        out = os.path.join(CACHE_DIR, f"{os.path.splitext(os.path.basename(n))[0]}-{key}")
                        os.makedirs(out, exist_ok=True)
                        for member in ex2:
                            dst = os.path.join(out, _safe_name(member))
                            if not os.path.exists(dst):
                                with zi.open(member) as src, open(dst, "wb") as f:
                                    while True:
                                        b = src.read(1 << 20)
                                        if not b:
                                            break
                                        f.write(b)
                        return os.path.join(out, _safe_name(m2)), f"unpacked {len(ex2)} file(s) from {os.path.basename(path)} / {n}"
            raise ValueError(f"{os.path.basename(path)}: no spectral image inside (looked for *SPECTRAL_IMAGE*, "
                             "ENVI .hdr, GeoTIFF, NetCDF, HDF5, MAT)")
        st = os.stat(path)
        key = hashlib.sha1(f"{os.path.abspath(path)}|{st.st_size}|{st.st_mtime}".encode()).hexdigest()[:10]
        out = os.path.join(CACHE_DIR, f"{os.path.splitext(os.path.basename(path))[0]}-{key}")
        os.makedirs(out, exist_ok=True)
        for member in extract:
            dst = os.path.join(out, _safe_name(member))
            if os.path.exists(dst) and os.path.getsize(dst) == z.getinfo(member).file_size:
                continue
            with z.open(member) as src, open(dst, "wb") as f:
                while True:
                    b = src.read(1 << 20)
                    if not b:
                        break
                    f.write(b)
        note_inner = f"unpacked {len(extract)} file(s) from {os.path.basename(path)} to {out}"
        return os.path.join(out, _safe_name(main)), note_inner
