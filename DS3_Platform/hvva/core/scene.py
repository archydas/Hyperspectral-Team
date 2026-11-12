"""A Scene = cube + wavelengths + georeferencing + class labels/ROIs.  No GUI code here."""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from . import quality as Q
from .geo import GeoInfo
from .rois import normalize_roi, roi_mask


@dataclass
class Scene:
    name: str
    cube: np.ndarray                                   # (rows, cols, bands) float32
    wavelengths: Optional[np.ndarray] = None           # nm
    geo: Optional[GeoInfo] = None
    labels: Optional[np.ndarray] = None                # (rows, cols) int, 0 = unlabelled
    class_names: Dict[int, str] = field(default_factory=dict)
    rois: List[dict] = field(default_factory=list)     # see hvva.core.rois: {id, name, class, kind, rect|vertices}
    path: Optional[str] = None
    # data quality (hvva.core.quality) - bands are flagged, never deleted
    bad_file: Optional[np.ndarray] = None              # bands the file marks bad (ENVI bbl ...)
    mask_water: bool = True                            # flag the water-absorption windows
    water_windows: tuple = Q.WATER_WINDOWS
    mask_empty: bool = True                            # flag bands with no data at all
    unit: str = "unknown"                              # reflectance | radiance | dn | ... | unknown
    unit_note: str = ""                                # how the unit was decided
    synthetic: bool = False                            # made-up data (demo): spectra saved from it are 'synthetic'
    display: Optional[dict] = None                     # hvva.core.display: {"mode", "bands", "stretch"}

    def __post_init__(self):
        self.rois = [normalize_roi(r, k) for k, r in enumerate(self.rois, 1)]
        self.unit = Q.normalize_unit(self.unit) or "unknown"
        self._empty_cache = None

    # ---- bad bands ---------------------------------------------------------
    def _empty(self):
        key = (id(self.cube), self.cube.shape)
        if self._empty_cache is None or self._empty_cache[0] != key:
            self._empty_cache = (key, Q.empty_bands(self.cube))
        return self._empty_cache[1]

    def bad_band_parts(self) -> Dict[str, np.ndarray]:
        nb = self.n_bands
        parts = {}
        if self.bad_file is not None and len(self.bad_file) == nb:
            parts["file"] = np.asarray(self.bad_file, bool)
        if self.mask_water and self.has_wavelengths:
            parts["water"] = Q.water_mask(self.wavelengths, self.water_windows)
        if self.mask_empty:
            parts["empty"] = self._empty()
        return parts

    @property
    def bad_bands(self) -> np.ndarray:
        """True = leave this band out of spectra taken from the scene (the cube itself is not changed)."""
        m = np.zeros(self.n_bands, bool)
        for part in self.bad_band_parts().values():
            m |= part
        return m

    @property
    def good_bands(self) -> np.ndarray:
        return ~self.bad_bands

    def bad_band_summary(self) -> str:
        parts = self.bad_band_parts()
        total = int(self.bad_bands.sum())
        if not total:
            return "no bad bands"
        names = {"file": "from the file", "water": "in masked ranges", "empty": "without data"}
        bits = [f"{int(v.sum())} {names.get(k, k)}" for k, v in parts.items() if v.any()]
        rng = ""
        if self.has_wavelengths:
            sp = Q.spans(self.wavelengths, self.bad_bands)
            rng = "; " + ", ".join(f"{a:.0f}-{b:.0f}" if b > a else f"{a:.0f}" for a, b in sp[:8]) + " nm" + \
                (" ..." if len(sp) > 8 else "")
        return f"{total} bad bands ({', '.join(bits)}){rng}"

    def masked(self, X):
        """Spectra X (..., bands) from this scene with the bad bands set to NaN (a copy)."""
        return Q.apply_mask(X, self.bad_bands)

    def spectrum_at(self, row: int, col: int, size: int = 1):
        """Pixel spectrum (mean of a size x size window), bad bands -> NaN."""
        h, w, _ = self.cube.shape
        r0, r1 = max(0, row - size // 2), min(h, row + size // 2 + 1)
        c0, c1 = max(0, col - size // 2), min(w, col + size // 2 + 1)
        block = self.cube[r0:r1, c0:c1].reshape(-1, self.n_bands).astype(np.float64)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            spec = np.nanmean(block, axis=0) if len(block) > 1 else block[0]
        return self.masked(spec)

    def pixels(self, mask2d):
        """Spectra of the pixels in a boolean mask, bad bands -> NaN."""
        return self.masked(self.cube[mask2d])

    @property
    def purity(self) -> str:
        """Purity of spectra taken from this scene."""
        return "synthetic" if self.synthetic else "image_pixel"

    # ---- ROIs --------------------------------------------------------------
    def add_roi(self, roi: dict) -> dict:
        """Normalise (id, name, kind) and append; returns the stored ROI."""
        names = {r["name"] for r in self.rois}
        k = len(self.rois) + 1
        while f"ROI {k}" in names:
            k += 1
        r = normalize_roi(roi, k)
        if r["name"] in names:                           # loaded / appended ROIs keep unique names per image
            base, j = r["name"], 2
            while f"{base} ({j})" in names:
                j += 1
            r["name"] = f"{base} ({j})"
        self.rois.append(r)
        return r

    def roi_by_id(self, roi_id: str) -> Optional[dict]:
        return next((r for r in self.rois if r["id"] == roi_id), None)

    # ---- shape helpers -------------------------------------------------
    @property
    def shape(self):
        return self.cube.shape

    @property
    def n_bands(self):
        return self.cube.shape[2]

    @property
    def has_wavelengths(self):
        return self.wavelengths is not None and len(self.wavelengths) == self.n_bands

    # ---- class masks -----------------------------------------------------
    def class_masks(self) -> Dict[str, np.ndarray]:
        """{class name: boolean mask} from the label map and the ROIs (rectangles and polygons)."""
        h, w, _ = self.cube.shape
        out: Dict[str, np.ndarray] = {}
        if self.labels is not None:
            for cid in np.unique(self.labels):
                if cid == 0:
                    continue
                nm = self.class_names.get(int(cid), f"class_{int(cid)}")
                out[nm] = out.get(nm, np.zeros((h, w), bool)) | (self.labels == cid)
        for roi in self.rois:
            m = out.setdefault(roi["class"], np.zeros((h, w), bool))
            m |= roi_mask(roi, h, w)
        return {k: v for k, v in out.items() if v.any()}

    def rgb(self, bands=None, stretch=(2, 98)):
        """Quick-look RGB; picks bands near 650/550/450 nm when wavelengths are known."""
        if bands is None:
            if self.has_wavelengths:
                bands = [int(np.argmin(np.abs(self.wavelengths - t))) for t in (650, 550, 450)]
            else:
                n = self.n_bands
                bands = [int(n * 0.75), int(n * 0.45), int(n * 0.15)]
        img = self.cube[:, :, bands].astype(np.float32)
        out = np.zeros_like(img)
        for k in range(3):
            lo, hi = np.nanpercentile(img[:, :, k], stretch)
            out[:, :, k] = np.clip((img[:, :, k] - lo) / (hi - lo + 1e-12), 0, 1)
        return np.nan_to_num(out)

    def summary(self):
        h, w, b = self.shape
        wl = f"{self.wavelengths.min():.0f}-{self.wavelengths.max():.0f} nm" if self.has_wavelengths else "none"
        return {"name": self.name, "rows": h, "cols": w, "bands": b, "wavelengths": wl,
                "crs": (self.geo.crs if self.geo else None) or "-",
                "classes": ", ".join(self.class_masks().keys()) or "-",
                "unit": self.unit + (" (synthetic)" if self.synthetic else ""),
                "bad": int(self.bad_bands.sum())}


# ----------------------------------------------------------------------
# Native format: .npz keeps the cube, wavelengths, geo and labels together
# ----------------------------------------------------------------------
def _roi_meta(r):
    d = {"id": r["id"], "name": r["name"], "class": r["class"], "kind": r["kind"]}
    if r["kind"] == "rect":
        d["rect"] = [int(v) for v in r["rect"]]
    elif r["kind"] == "pixels":
        d["pixels"] = [[int(x), int(y)] for x, y in r["pixels"]]
    else:
        d["vertices"] = [[float(x), float(y)] for x, y in r["vertices"]]
    if r.get("saved_to"):
        d["saved_to"] = list(r["saved_to"])          # libraries the ROI was saved to (ROI Manager column)
    return d


def save_scene(scene: Scene, path: str):
    meta = {
        "name": scene.name,
        "geo": scene.geo.to_dict() if scene.geo else None,
        "class_names": {str(k): v for k, v in scene.class_names.items()},
        "rois": [_roi_meta(r) for r in scene.rois],
        "unit": scene.unit, "unit_note": scene.unit_note, "synthetic": bool(scene.synthetic),
        "mask_water": bool(scene.mask_water), "water_windows": [list(w) for w in scene.water_windows],
        "mask_empty": bool(scene.mask_empty),
        "bad_file": None if scene.bad_file is None else [bool(v) for v in scene.bad_file],
        "display": scene.display,
    }
    arrays = {"cube": scene.cube.astype(np.float32), "meta": np.array(json.dumps(meta))}
    if scene.wavelengths is not None:
        arrays["wavelengths"] = np.asarray(scene.wavelengths, float)
    if scene.labels is not None:
        arrays["labels"] = scene.labels.astype(np.int32)
    np.savez_compressed(path, **arrays)
    scene.path = path


def _load_npz(path):
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta"])) if "meta" in z.files else {}
    return Scene(
        name=meta.get("name") or os.path.splitext(os.path.basename(path))[0],
        cube=z["cube"].astype(np.float32),
        wavelengths=z["wavelengths"] if "wavelengths" in z.files else None,
        geo=GeoInfo.from_dict(meta.get("geo")),
        labels=z["labels"] if "labels" in z.files else None,
        class_names={int(k): v for k, v in meta.get("class_names", {}).items()},
        rois=list(meta.get("rois", [])),            # old files ({class, rect}) are upgraded by Scene.__post_init__
        path=path,
        bad_file=None if meta.get("bad_file") is None else np.array(meta["bad_file"], bool),
        mask_water=bool(meta.get("mask_water", True)),
        water_windows=tuple(tuple(w) for w in meta.get("water_windows", Q.WATER_WINDOWS)),
        mask_empty=bool(meta.get("mask_empty", True)),
        unit=meta.get("unit") or "unknown", unit_note=meta.get("unit_note", ""),
        synthetic=bool(meta.get("synthetic", False)), display=meta.get("display"))


# ----------------------------------------------------------------------
# ENVI
# ----------------------------------------------------------------------
_ENVI_DTYPES = {1: "u1", 2: "i2", 3: "i4", 4: "f4", 5: "f8", 12: "u2", 13: "u4", 14: "i8", 15: "u8"}


def parse_envi_header(path):
    text = open(path, "r", errors="replace").read()
    hdr = {}
    for m in re.finditer(r"^\s*([^=\n{]+?)\s*=\s*(\{.*?\}|[^\n]*)", text, re.M | re.S):
        hdr[m.group(1).strip().lower()] = m.group(2).strip()
    return hdr


def _hdr_list(v):
    return [s.strip() for s in v.strip().strip("{}").replace("\n", " ").split(",") if s.strip()]


def load_envi(path):
    hdr_path = path if path.lower().endswith(".hdr") else os.path.splitext(path)[0] + ".hdr"
    if not os.path.exists(hdr_path) and not path.lower().endswith(".hdr"):
        hdr_path = path + ".hdr"
    h = parse_envi_header(hdr_path)
    ns, nl, nb = int(h["samples"]), int(h["lines"]), int(h["bands"])
    dt = np.dtype(_ENVI_DTYPES[int(h["data type"])]).newbyteorder(">" if h.get("byte order", "0").strip() == "1" else "<")
    offset = int(h.get("header offset", 0))
    inter = h.get("interleave", "bsq").lower()
    data_path = None
    base = hdr_path[:-4]
    for cand in ([path] if not path.lower().endswith(".hdr") else []) + \
            [base + e for e in ("", ".img", ".dat", ".raw", ".bin", ".sli", ".bil", ".bip", ".bsq")]:
        if os.path.isfile(cand) and not cand.lower().endswith(".hdr"):
            data_path = cand
            break
    if data_path is None:
        raise FileNotFoundError(f"no data file next to {hdr_path}")
    raw = np.fromfile(data_path, dtype=dt, offset=offset, count=ns * nl * nb)
    if inter == "bsq":
        cube = raw.reshape(nb, nl, ns).transpose(1, 2, 0)
    elif inter == "bil":
        cube = raw.reshape(nl, nb, ns).transpose(0, 2, 1)
    else:
        cube = raw.reshape(nl, ns, nb)
    if "data ignore value" in h:                                  # no-data value -> gaps
        try:
            ign = float(h["data ignore value"])
            cube = np.where(cube == ign, np.nan, cube.astype(np.float32))
        except ValueError:
            pass
    notes = []
    for key in ("reflectance scale factor",):                     # AVIRIS / ENVI: stored = reflectance x factor
        if key in h:
            try:
                f = float(h[key])
                if f not in (0.0, 1.0):
                    cube = cube.astype(np.float32) / np.float32(f)
                    notes.append(f"divided by the reflectance scale factor {f:g}")
            except ValueError:
                pass
    wl = None
    if "wavelength" in h:
        wl = np.array([float(v) for v in _hdr_list(h["wavelength"])])
        units = h.get("wavelength units", "").lower()
        if "micro" in units or (not units and wl.max() < 10):
            wl = wl * 1000.0
    bad_file = Q.bbl_from_header(h["bbl"], nb) if "bbl" in h else None
    if wl is None:                                                # EnMAP delivered as ENVI: wavelengths in METADATA.XML
        from .loaders import _overlap_keep, _sensor_sidecar
        side = _sensor_sidecar(data_path, nb)
        if side is not None:
            wl = side["wavelengths"]
            nod = side["nodata"] if side["nodata"] is not None else (-32768.0 if side["sensor"] == "EnMAP" else None)
            cube = cube.astype(np.float32)
            if nod is not None:
                cube[cube == nod] = np.nan
            if side["level"] == "L2A":
                cube /= np.float32(10000.0)
            keep = _overlap_keep(wl)
            if not np.array_equal(keep, np.arange(nb)):
                cube, wl = cube[:, :, keep], wl[keep]
                bad_file = None if bad_file is None else bad_file[keep]
            notes.append(f"{side['sensor']} {side['level']} wavelengths from {side['file']}")
            h.setdefault("hvva unit", "reflectance" if side["level"] == "L2A" else "radiance")
    unit, unote = "unknown", ""
    for key in ("hvva unit", "data units", "z plot titles", "description"):
        if key in h:
            u = Q.normalize_unit(h[key].strip("{} "))
            if u and u != "unknown":
                unit, unote = u, f"from the header ({key})"
                break
    if unit == "unknown" and "reflectance scale factor" in h:
        unit, unote = "reflectance", "from the header (reflectance scale factor)"
    if unit == "unknown":
        u = Q.unit_from_name(hdr_path)
        if u:
            unit, unote = u, "from the file name"
    if notes:
        unote = "; ".join([unote] + notes) if unote else "; ".join(notes)
    geo = None
    if "map info" in h:
        try:
            geo = GeoInfo.from_envi_map_info(h["map info"], h.get("coordinate system string"))
        except Exception:
            geo = None
    return Scene(os.path.splitext(os.path.basename(hdr_path))[0], np.ascontiguousarray(cube, np.float32),
                 wl, geo, path=data_path, bad_file=bad_file, unit=unit, unit_note=unote)


def save_envi(scene: Scene, base_path: str):
    """Write cube (BSQ float32) + .hdr that includes wavelengths and map info."""
    cube = scene.cube.astype("<f4")
    h, w, b = cube.shape
    cube.transpose(2, 0, 1).tofile(base_path + ".img")
    lines = ["ENVI", f"description = {{{scene.name}}}", f"samples = {w}", f"lines = {h}", f"bands = {b}",
             "header offset = 0", "file type = ENVI Standard", "data type = 4", "interleave = bsq", "byte order = 0"]
    if scene.has_wavelengths:
        lines += ["wavelength units = Nanometers",
                  "wavelength = {" + ", ".join(f"{v:.4f}" for v in scene.wavelengths) + "}"]
    bad = scene.bad_bands
    if bad.any():                                                   # ENVI bad band list: 1 = good, 0 = bad
        lines.append("bbl = {" + ", ".join("0" if b else "1" for b in bad) + "}")
    if scene.unit != "unknown":
        lines.append(f"hvva unit = {scene.unit}")
    if scene.geo:
        lines.append(f"map info = {scene.geo.to_envi_map_info()}")
        if scene.geo.crs and scene.geo.crs.upper().startswith("EPSG:"):
            lines.append(f"coordinate system string = {{{scene.geo.crs}}}")
    open(base_path + ".hdr", "w").write("\n".join(lines) + "\n")


# ----------------------------------------------------------------------
# generic loader (any format - see loaders.py)
# ----------------------------------------------------------------------
def _is_envi(path):
    low = path.lower()
    return low.endswith(".hdr") or (os.path.isfile(path) and (os.path.exists(os.path.splitext(path)[0] + ".hdr")
                                                              or os.path.exists(path + ".hdr")))


def _is_native_npz(path):
    if not path.lower().endswith(".npz"):
        return False
    with np.load(path, allow_pickle=False) as z:
        return "meta" in z.files and "cube" in z.files


def load_scene(path, name=None, **options) -> Scene:
    """Load any supported file.  options (all optional, guessed when omitted): cube_key, band_axis,
    labels_key, wavelengths_key, wavelength_range=(start, stop), raw={samples, lines, bands, dtype, ...}."""
    from .loaders import open_source, raw_source
    path, container_note = Q.resolve_input(path)                 # .zip product -> the image inside; ancillary -> error
    if options.get("raw"):
        src = raw_source(path, **options.pop("raw"))
        options.pop("cube_key", None)
        sc = src.build(**{k: v for k, v in options.items() if v is not None}, name=name)
        return sc
    if _is_native_npz(path):
        sc = _load_npz(path)
    elif _is_envi(path):
        sc = load_envi(path)
        if options.get("wavelength_range") and not sc.has_wavelengths:
            assign_wavelengths(sc, *options["wavelength_range"])
    else:
        src = open_source(path)
        sc = src.build(**{k: v for k, v in options.items() if v is not None}, name=None)
    if name:
        sc.name = name
    finish_unit(sc)
    if container_note:
        sc.unit_note = "; ".join(x for x in (sc.unit_note, container_note) if x)
    return sc


def fingerprint(scene: Scene) -> str:
    """Stable id of an image: from the file (size + first/last MiB) when there is one, else from the pixels.
    Same file copied elsewhere -> same fingerprint; any change of the data -> a new one."""
    import hashlib
    h = hashlib.sha256()
    p = scene.path
    if p and os.path.isfile(p):
        size = os.path.getsize(p)
        h.update(f"file|{size}|".encode())
        with open(p, "rb") as f:
            h.update(f.read(1 << 20))
            if size > 2 << 20:
                f.seek(-(1 << 20), os.SEEK_END)
                h.update(f.read(1 << 20))
    else:
        c = scene.cube
        h.update(f"cube|{c.shape}|{scene.name}|".encode())
        step = max(1, c.shape[0] // 64)
        h.update(np.ascontiguousarray(c[::step, ::max(1, c.shape[1] // 64)]).tobytes())
    h.update(f"|{scene.n_bands}".encode())
    return h.hexdigest()[:40]


def image_info(scene: Scene) -> dict:
    """What the library's images table stores about a scene (the pixels stay in the file)."""
    if getattr(scene, "_fingerprint", None) is None or scene._fingerprint[0] != (scene.path, scene.cube.shape):
        scene._fingerprint = ((scene.path, scene.cube.shape), fingerprint(scene))
    h, w, b = scene.cube.shape
    ext = os.path.splitext(scene.path or "")[1].lower().lstrip(".") or ("npz" if scene.path else "memory")
    return {"fingerprint": scene._fingerprint[1], "name": scene.name,
            "path": os.path.abspath(scene.path) if scene.path else None, "format": ext,
            "rows": int(h), "cols": int(w), "bands": int(b), "unit": scene.unit,
            "sensor": "EnMAP" if "ENMAP" in (scene.name or "").upper() else None,
            "wavelengths": None if not scene.has_wavelengths else [float(v) for v in scene.wavelengths],
            "bad_band_mask": [bool(v) for v in scene.bad_bands],
            "crs": scene.geo.crs if scene.geo else None,
            "transform": list(scene.geo.transform) if scene.geo else None,
            "metadata": {"unit_note": scene.unit_note, "synthetic": bool(scene.synthetic)}}


def finish_unit(sc: Scene):
    """Last resort for scenes whose loader could not tell the unit: file name, then 0-1 values."""
    if sc.unit != "unknown":
        return
    u = Q.unit_from_name(sc.path or sc.name)
    if u:
        sc.unit, sc.unit_note = u, "from the file name"
        return
    sc.unit, sc.unit_note = Q.unit_from_values(sc.cube)


def assign_wavelengths(scene: Scene, start, stop):
    scene.wavelengths = np.linspace(start, stop, scene.n_bands)
