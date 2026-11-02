"""A Scene = cube + wavelengths + georeferencing + class labels/ROIs.  No GUI code here."""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .geo import GeoInfo


@dataclass
class Scene:
    name: str
    cube: np.ndarray                                   # (rows, cols, bands) float32
    wavelengths: Optional[np.ndarray] = None           # nm
    geo: Optional[GeoInfo] = None
    labels: Optional[np.ndarray] = None                # (rows, cols) int, 0 = unlabelled
    class_names: Dict[int, str] = field(default_factory=dict)
    rois: List[dict] = field(default_factory=list)     # {"class": str, "rect": (r0, c0, r1, c1)}
    path: Optional[str] = None

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
        """{class name: boolean mask} from the label map and the rectangular ROIs."""
        h, w, _ = self.cube.shape
        out: Dict[str, np.ndarray] = {}
        if self.labels is not None:
            for cid in np.unique(self.labels):
                if cid == 0:
                    continue
                nm = self.class_names.get(int(cid), f"class_{int(cid)}")
                out[nm] = out.get(nm, np.zeros((h, w), bool)) | (self.labels == cid)
        for roi in self.rois:
            r0, c0, r1, c1 = roi["rect"]
            m = out.setdefault(roi["class"], np.zeros((h, w), bool))
            m[max(r0, 0):min(r1, h), max(c0, 0):min(c1, w)] = True
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
                "classes": ", ".join(self.class_masks().keys()) or "-"}


# ----------------------------------------------------------------------
# Native format: .npz keeps the cube, wavelengths, geo and labels together
# ----------------------------------------------------------------------
def save_scene(scene: Scene, path: str):
    meta = {
        "name": scene.name,
        "geo": scene.geo.to_dict() if scene.geo else None,
        "class_names": {str(k): v for k, v in scene.class_names.items()},
        "rois": [{"class": r["class"], "rect": list(r["rect"])} for r in scene.rois],
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
        rois=[{"class": r["class"], "rect": tuple(r["rect"])} for r in meta.get("rois", [])],
        path=path)


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
    wl = None
    if "wavelength" in h:
        wl = np.array([float(v) for v in _hdr_list(h["wavelength"])])
        units = h.get("wavelength units", "").lower()
        if "micro" in units or (not units and wl.max() < 10):
            wl = wl * 1000.0
    geo = None
    if "map info" in h:
        try:
            geo = GeoInfo.from_envi_map_info(h["map info"], h.get("coordinate system string"))
        except Exception:
            geo = None
    return Scene(os.path.splitext(os.path.basename(hdr_path))[0], np.ascontiguousarray(cube, np.float32),
                 wl, geo, path=data_path)


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
    return sc


def assign_wavelengths(scene: Scene, start, stop):
    scene.wavelengths = np.linspace(start, stop, scene.n_bands)
