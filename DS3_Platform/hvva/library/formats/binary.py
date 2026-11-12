"""Instrument and array readers: ASD / SVC / Spectral Evolution, ENVI spectral library, image cubes, array files."""
from __future__ import annotations

import re
import struct
from pathlib import Path
from typing import List

import numpy as np

from .base import ReadOptions, Reader, SpectrumRecord, register, safe_float, stem_name, unit_from_text
from .text import _decode, records_from_table, _META_KV

ENVI_DTYPES = {1: "u1", 2: "i2", 3: "i4", 4: "f4", 5: "f8", 12: "u2", 13: "u4", 14: "i8", 15: "u8"}


# ================================================================== field spectrometers
_ASD_VERSIONS = {"ASD": False, "asd": False, "as6": True, "as7": True, "as8": True}   # has reference block
_ASD_TYPES = ["raw", "reflectance", "radiance", "no units", "irradiance", "qi", "transmittance", "unknown",
              "absorbance"]
_ASD_INSTR = {0: "unknown", 1: "PSII", 2: "LSVNIR", 3: "FSVNIR", 4: "FSFR", 5: "FSNIR", 6: "CHEM", 7: "FSFR unattended"}


def read_asd(path: Path):
    """ASD FieldSpec binary (versions 1-8). Returns (wavelengths_nm, spectrum, reference|None, meta)."""
    b = path.read_bytes()
    ver = b[0:3].decode("latin-1")
    if ver not in _ASD_VERSIONS or len(b) < 484:
        raise ValueError(f"{path.name}: not an ASD file")
    dtype_i = b[186]
    start = struct.unpack("<f", b[191:195])[0]
    step = struct.unpack("<f", b[195:199])[0]
    fmt_i = b[199]
    nch = struct.unpack("<H", b[204:206])[0]
    code, size = {0: ("f", 4), 1: ("i", 4), 2: ("d", 8)}.get(fmt_i, ("f", 4))
    spec = np.array(struct.unpack("<" + code * nch, b[484:484 + nch * size]), float)
    ref = None
    if _ASD_VERSIONS[ver] and len(b) >= 484 + nch * size + 20:
        p = 484 + nch * size
        dlen = struct.unpack("<H", b[p + 18:p + 20])[0]
        q = p + 20 + dlen
        if len(b) >= q + nch * size:
            ref = np.array(struct.unpack("<" + code * nch, b[q:q + nch * size]), float)
    comments = b[3:160].split(b"\x00")[0].decode("latin-1", "replace").strip()
    instr = _ASD_INSTR.get(b[431], "unknown") if len(b) > 431 else "unknown"
    meta = {"asd_version": ver, "data_type": _ASD_TYPES[dtype_i] if dtype_i < len(_ASD_TYPES) else str(dtype_i),
            "channels": nch, "instrument": f"ASD {instr}", "comments": comments,
            "integration_time_ms": struct.unpack("<I", b[390:394])[0]}
    wl = start + step * np.arange(nch)
    return wl, spec, ref, meta


@register
class AsdReader(Reader):
    name = "asd"
    label = "ASD FieldSpec (.asd)"
    extensions = (".asd",)

    def sniff(self, path: Path) -> float:
        if not path.is_file():
            return 0.0
        with open(path, "rb") as f:
            head = f.read(3)
        return 0.97 if head.decode("latin-1") in _ASD_VERSIONS else 0.0

    def read(self, path: Path, opts: ReadOptions):
        wl, spec, ref, meta = read_asd(path)
        kind = meta["data_type"]
        quantity, scale = kind, "auto"
        if kind == "reflectance":
            vals = spec
            if ref is not None and np.nanmax(np.abs(spec)) > 5 and np.nanmax(np.abs(ref)) > 0:
                vals = spec / np.where(ref == 0, np.nan, ref)            # DN stored with reflectance flag
            quantity, scale = "reflectance", "unit"
        elif kind == "raw" and ref is not None and np.nanmax(np.abs(ref)) > 0:
            vals = spec / np.where(ref == 0, np.nan, ref)               # target DN / white reference DN
            quantity, scale = "reflectance", "unit"
            meta["computed"] = "target / white reference"
        else:
            vals = spec
            scale = "none"
        return [SpectrumRecord(stem_name(path), wl, vals, acquisition=None, wl_unit="nm", scale=scale,
                               quantity=quantity, material_metadata={"source_file": path.name, **meta},
                               spectrum_metadata={"uid": f"asd:{path.name}", "source_file": path.name},
                               group=str(path))]


def _keyed_numeric(text: str, data_marker: str, sep_regex: str):
    meta, rows, header, in_data = {}, [], None, False
    for ln in text.splitlines():
        if not in_data:
            if ln.strip().lower().startswith(data_marker):
                in_data = True
                continue
            m = re.match(sep_regex, ln)
            if m:
                meta[m.group(1).strip().lower()] = m.group(2).strip()
            continue
        if not ln.strip():
            continue
        cells = re.split(r"\t|\s{2,}|\s+(?=[-\d.])", ln.strip())
        if header is None and not re.match(r"^[-\d.]", cells[0]):
            header = [c.strip() for c in ln.strip().split("\t")] if "\t" in ln else cells
            continue
        rows.append([safe_float(c) for c in ln.split()])
    return meta, header, rows


@register
class SvcSigReader(Reader):
    name = "sig"
    label = "Spectra Vista SVC (.sig)"
    extensions = (".sig",)

    def sniff(self, path: Path) -> float:
        if not path.is_file() or path.suffix.lower() != ".sig":
            return 0.0
        head = _decode(path.read_bytes()[:3000]).lower()
        return 0.95 if ("data=" in head or "/*** spectra vista" in head) else 0.5

    def read(self, path: Path, opts: ReadOptions):
        meta, _hdr, rows = _keyed_numeric(_decode(path.read_bytes()), "data=", r"^\s*([^=]+?)\s*=\s*(.*)$")
        rows = [r for r in rows if len(r) >= 2 and np.isfinite(r[0])]
        if not rows:
            raise ValueError(f"{path.name}: no data block")
        M = np.array([r + [np.nan] * (4 - len(r)) for r in rows], float)
        wl = M[:, 0]
        if M.shape[1] >= 4 and np.isfinite(M[:, 3]).any():
            vals, scale, qty = M[:, 3], "percent", "reflectance"
        else:
            vals, scale, qty = M[:, -1], "auto", None
        instr = meta.get("instrument", "SVC")
        nm = re.sub(r"\.(sig|asd|sed)$", "", meta.get("name") or "", flags=re.I).strip() or stem_name(path)
        return [SpectrumRecord(nm, wl, vals, wl_unit="nm", scale=scale, quantity=qty,
                               acquisition=" ".join(meta.get(k, "") for k in ("comm", "comment", "location")) or None,
                               material_metadata={"source_file": path.name, "instrument": f"SVC {instr}",
                                                  **{k: v[:200] for k, v in meta.items() if k in (
                                                      "units", "time", "integration", "comm", "longitude", "latitude",
                                                      "optic", "memory slot")}},
                               spectrum_metadata={"uid": f"sig:{path.name}", "source_file": path.name},
                               group=str(path))]


@register
class SedReader(Reader):
    name = "sed"
    label = "Spectral Evolution (.sed)"
    extensions = (".sed",)

    def sniff(self, path: Path) -> float:
        if not path.is_file() or path.suffix.lower() != ".sed":
            return 0.0
        head = _decode(path.read_bytes()[:4000])
        return 0.95 if "Data:" in head else 0.5

    def read(self, path: Path, opts: ReadOptions):
        text = _decode(path.read_bytes())
        meta, header, rows = {}, None, []
        lines = text.splitlines()
        i = 0
        for i, ln in enumerate(lines):
            if ln.strip().startswith("Data:"):
                break
            m = re.match(r"^\s*([^:]+?)\s*:\s*(.*)$", ln)
            if m:
                meta[m.group(1).strip().lower()] = m.group(2).strip()
        body = lines[i + 1:]
        header = [c.strip() for c in body[0].split("\t")] if body else []
        for ln in body[1:]:
            if ln.strip():
                rows.append([safe_float(c) for c in ln.split("\t")])
        if not rows:
            raise ValueError(f"{path.name}: no data block")
        M = np.array([r + [np.nan] * (len(header) - len(r)) for r in rows], float)
        col = {h: k for k, h in enumerate(header)}
        if "Reflect. [1.0]" in col:
            vals, scale, qty = M[:, col["Reflect. [1.0]"]], "unit", "reflectance"
        elif "Reflect. %" in col:
            vals, scale, qty = M[:, col["Reflect. %"]], "percent", "reflectance"
        elif "Tgt./Ref. %" in col:
            vals, scale, qty = M[:, col["Tgt./Ref. %"]], "percent", "reflectance"
        else:
            vals, scale, qty = M[:, -1], "none", meta.get("measurement", "").lower() or None
        wl = M[:, col.get("Wvl", 0)]
        return [SpectrumRecord(stem_name(path), wl, vals, wl_unit="nm", scale=scale, quantity=qty,
                               acquisition=meta.get("comment") or None,
                               material_metadata={"source_file": path.name,
                                                  "instrument": f"Spectral Evolution {meta.get('instrument', '')}".strip(),
                                                  **{k: v[:200] for k, v in meta.items() if k in (
                                                      "measurement", "date", "time", "integration", "latitude",
                                                      "longitude", "foreoptic", "comment")}},
                               spectrum_metadata={"uid": f"sed:{path.name}", "source_file": path.name},
                               group=str(path))]


# ================================================================== ENVI spectral library
def parse_envi_header(path: Path) -> dict:
    text = _decode(path.read_bytes())
    hdr = {}
    for m in re.finditer(r"^\s*([^=\n{]+?)\s*=\s*(\{.*?\}|[^\n]*)", text, re.M | re.S):
        hdr[m.group(1).strip().lower()] = m.group(2).strip()
    return hdr


def hdr_list(v):
    return [s.strip() for s in str(v or "").strip().strip("{}").replace("\n", " ").split(",") if s.strip()]


def _envi_data_file(hdr_path: Path):
    base = hdr_path.with_suffix("")
    for cand in [base.with_suffix(e) for e in (".sli", ".slb", ".dat", ".img", ".bin", ".raw", ".esl")] + [base]:
        if cand.is_file() and cand.suffix.lower() != ".hdr":
            return cand
    return None


def _hdr_for(path: Path):
    for cand in (path if path.suffix.lower() == ".hdr" else None, path.with_suffix(".hdr"),
                 Path(str(path) + ".hdr")):
        if cand is not None and cand.is_file():
            return cand
    return None


@register
class EnviSliReader(Reader):
    name = "envi_sli"
    label = "ENVI spectral library (.sli/.hdr)"
    extensions = (".sli", ".slb", ".esl", ".hdr")

    def sniff(self, path: Path) -> float:
        if not path.is_file():
            return 0.0
        hdr = _hdr_for(path)
        if hdr is None:
            return 0.0
        if path.suffix.lower() not in (".hdr", ".sli", ".slb", ".esl"):
            return 0.0
        h = parse_envi_header(hdr)
        stem = path.stem
        if stem.lower().endswith("_std") and any((path.parent / (stem[:-4] + e)).is_file() for e in (".sli", ".hdr")):
            return 0.0                                  # HyperVista's companion std library: read with the means
        if "spectral library" in h.get("file type", "").lower() or path.suffix.lower() in (".sli", ".slb", ".esl"):
            return 0.95
        return 0.0

    def describe(self, path: Path) -> str:
        h = parse_envi_header(_hdr_for(path))
        return f"{self.label}: {h.get('lines', '?')} spectra × {h.get('samples', '?')} bands"

    def read(self, path: Path, opts: ReadOptions):
        hdr_path = _hdr_for(path)
        h = parse_envi_header(hdr_path)
        data_path = path if path.suffix.lower() != ".hdr" else _envi_data_file(hdr_path)
        if data_path is None:
            raise FileNotFoundError(f"no data file next to {hdr_path.name}")
        ns, nl = int(h["samples"]), int(h["lines"])
        dt = np.dtype(ENVI_DTYPES.get(int(h.get("data type", 4)), "f4")).newbyteorder(
            ">" if h.get("byte order", "0").strip() == "1" else "<")
        data = np.fromfile(data_path, dtype=dt, offset=int(h.get("header offset", 0)), count=ns * nl)
        data = data.reshape(nl, ns).astype(float)
        if "data ignore value" in h:
            data[data == safe_float(h["data ignore value"])] = np.nan
        sf = safe_float(h.get("reflectance scale factor"))
        scale = "auto"
        if np.isfinite(sf) and sf > 0:
            data = data / sf
            scale = "unit"
        wl = np.array([safe_float(v) for v in hdr_list(h.get("wavelength"))], float)
        if wl.size != ns:
            if opts.wl_range:
                wl = np.linspace(opts.wl_range[0], opts.wl_range[1], ns)
            else:
                raise ValueError(f"{hdr_path.name}: no wavelength list - give a wavelength range")
        unit = unit_from_text(h.get("wavelength units", ""))
        names = hdr_list(h.get("spectra names")) or [f"{stem_name(path)} {i + 1}" for i in range(nl)]
        names += [f"{stem_name(path)} {i + 1}" for i in range(len(names), nl)]
        std = None
        std_path = data_path.with_name(data_path.stem + "_std" + data_path.suffix)
        if std_path.is_file():                            # HyperVista writes <base>_std.sli next to the means
            try:
                std = np.fromfile(std_path, dtype=dt, count=ns * nl).reshape(nl, ns).astype(float)
                if np.isfinite(sf) and sf > 0:
                    std = std / sf
            except ValueError:
                std = None
        g = str(path)
        return [SpectrumRecord(names[i], wl, data[i], std=None if std is None else std[i], wl_unit=unit, scale=scale,
                               material_metadata={"source_file": data_path.name,
                                                  "description": h.get("description", "").strip("{}")[:300]},
                               spectrum_metadata={"uid": f"envi_sli:{data_path.name}:{i}", "source_file": data_path.name},
                               unit=_hdr_unit(h), group=g) for i in range(nl)]


def _hdr_unit(h):
    """Value unit written in an ENVI header (HyperVista's 'hvva unit', 'data units'), or None."""
    from ...core.quality import normalize_unit
    for k in ("hvva unit", "data units"):
        if k in h:
            u = normalize_unit(h[k].strip("{} "))
            if u and u != "unknown":
                return u
    return None


# ================================================================== image cubes
def _scene_records(sc, path: Path, opts: ReadOptions, reader: str) -> List[SpectrumRecord]:
    if not sc.has_wavelengths:
        if opts.wl_range:
            from ...core.scene import assign_wavelengths
            assign_wavelengths(sc, opts.wl_range[0], opts.wl_range[1])
        else:
            raise ValueError(f"{path.name}: the image has no wavelengths - give a wavelength range")
    cube = sc.cube.astype(float)
    h, w, b = cube.shape
    bad = sc.bad_bands                                    # water windows / file bbl / empty bands -> gaps
    if bad.any():
        cube[:, :, bad] = np.nan
    flat = cube.reshape(-1, b)
    good = np.isfinite(flat).any(axis=0)                  # bands that are no-data everywhere (water bands) -> gaps
    valid = np.isfinite(flat[:, good]).all(axis=1) & (np.nansum(np.abs(flat), axis=1) > 0)
    import warnings
    warnings.filterwarnings("ignore", "Mean of empty slice", RuntimeWarning)
    warnings.filterwarnings("ignore", "Degrees of freedom", RuntimeWarning)
    mode = opts.image_mode
    masks = sc.class_masks()
    if mode == "auto":
        mode = "classes" if masks else "mean"
    out, g = [], str(path)
    base = {"source_file": path.name, "scene": sc.name, "image_shape": [h, w, b]}
    if mode == "classes":
        if not masks:
            raise ValueError(f"{path.name}: no label map / classes in this file - choose mean or sample")
        for cname, m in masks.items():
            px = cube[m]
            px = px[np.isfinite(px[:, good]).all(axis=1)]
            if px.shape[0] == 0:
                continue
            out.append(SpectrumRecord(cname, sc.wavelengths, np.nanmean(px, 0),
                                      std=np.nanstd(px, 0, ddof=1) if px.shape[0] > 1 else None,
                                      material_metadata={**base, "n_pixels": int(px.shape[0])},
                                      spectrum_metadata={"uid": f"{reader}:{path.name}:class:{cname}",
                                                         "n_pixels": int(px.shape[0]), "source_scenes": sc.name},
                                      group=g))
    elif mode == "mean":
        px = flat[valid]
        out.append(SpectrumRecord(f"{sc.name} mean", sc.wavelengths, np.nanmean(px, 0), std=np.nanstd(px, 0, ddof=1),
                                  material_metadata={**base, "n_pixels": int(px.shape[0])},
                                  spectrum_metadata={"uid": f"{reader}:{path.name}:mean", "n_pixels": int(px.shape[0]),
                                                     "source_scenes": sc.name}, group=g))
    else:
        idx = np.flatnonzero(valid)
        n = min(len(idx), max(1, opts.image_samples))
        if mode == "sample":
            idx = np.sort(np.random.default_rng(0).choice(idx, n, replace=False))
        else:
            idx = idx[:n]
        for k in idx:
            r, c = divmod(int(k), w)
            out.append(SpectrumRecord(f"{sc.name} pixel ({r}, {c})", sc.wavelengths, flat[k],
                                      material_metadata={**base, "pixel": [r, c]},
                                      spectrum_metadata={"uid": f"{reader}:{path.name}:px{r}_{c}", "n_pixels": 1,
                                                         "source_scenes": sc.name}, group=g))
    unit = sc.unit if sc.unit != "unknown" else None
    for r in out:
        r.unit, r.purity = unit, sc.purity
        if unit == "reflectance":
            r.scale = "unit"                              # already 0-1 (the loader scaled it)
        elif unit:
            r.scale, r.quantity = "none", unit            # radiance / DN: keep as measured
        if bad.any():
            r.spectrum_metadata["bad_bands_masked"] = int(bad.sum())
    return out


@register
class ImageReader(Reader):
    name = "image"
    label = "Hyperspectral image (ENVI, GeoTIFF, MAT, HDF5, NetCDF, NumPy)"
    extensions = (".hdr", ".img", ".dat", ".bsq", ".bil", ".bip", ".tif", ".tiff", ".mat", ".h5", ".hdf5", ".he5",
                  ".hdf", ".nc", ".nc4", ".npy", ".npz", ".zarr", ".zip")

    def sniff(self, path: Path) -> float:
        from ...core.quality import _pick_image, ancillary_reason
        ext = path.suffix.lower()
        if ext not in self.extensions or ancillary_reason(path):
            return 0.0
        if ext == ".zip":                                       # an image product (EnMAP L2A ...) in a zip
            try:
                import zipfile
                with zipfile.ZipFile(path) as z:
                    names = z.namelist()
            except Exception:                                               # noqa: BLE001
                return 0.0
            main, _ = _pick_image(names)
            return 0.9 if main and "SPECTRAL_IMAGE" in main.upper() else 0.0
        if ext in (".hdr", ".img", ".dat", ".bsq", ".bil", ".bip"):
            hdr = _hdr_for(path)
            if hdr is None:
                return 0.0
            h = parse_envi_header(hdr)
            if "spectral library" in h.get("file type", "").lower():
                return 0.0
            return 0.85 if int(h.get("bands", "1") or 1) > 1 else 0.0
        if ext in (".mat", ".h5", ".hdf5", ".he5", ".hdf", ".nc", ".nc4", ".npy", ".npz", ".zarr"):
            try:
                from ...core.loaders import open_source
                src = open_source(str(path))
            except Exception:                                               # noqa: BLE001
                return 0.0
            return 0.7 if any(len(s) == 3 for s, _ in src.arrays.values()) else 0.0
        if "QL_" in path.name.upper() or "QUICKLOOK" in path.name.upper():
            return 0.0
        return 0.5                                              # tif: a cube if it has many bands (checked on read)

    def describe(self, path: Path) -> str:
        return "Hyperspectral image → class means / image mean / sampled pixels"

    def read(self, path: Path, opts: ReadOptions):
        from ...core.scene import load_scene
        target = path
        if path.suffix.lower() in (".img", ".dat", ".bsq", ".bil", ".bip"):
            target = _hdr_for(path) or path
        sc = load_scene(str(target))
        if sc.cube.ndim != 3 or sc.cube.shape[2] < 3:
            raise ValueError(f"{path.name}: not a multi-band image")
        if sc.n_bands <= 4:
            v = sc.cube[::max(1, sc.shape[0] // 64), ::max(1, sc.shape[1] // 64)]
            v = v[np.isfinite(v)]
            if v.size and v.min() >= 0 and v.max() <= 255 and np.allclose(v, np.round(v)):
                raise ValueError(f"{path.name}: looks like an 8-bit RGB picture (quicklook), not spectral data")
        return _scene_records(sc, path, opts, self.name)


# ================================================================== array files holding a spectra matrix
def _mat_names(path: Path, n: int):
    """Spectrum names stored as a cell array / char matrix in a .mat file (the generic loaders skip those)."""
    try:
        import scipy.io
        d = scipy.io.loadmat(str(path), squeeze_me=True)
    except Exception:                                                       # noqa: BLE001
        return None
    pref = sorted(d, key=lambda k: 0 if re.search(r"name|label|class|id|material", k, re.I) else 1)
    for k in pref:
        if k.startswith("__"):
            continue
        a = np.asarray(d[k])
        if a.dtype.kind in "OUS" and a.size == n:
            return [str(x).strip() for x in a.ravel()]
        if a.dtype.kind in "US" and a.ndim == 2 and a.shape[0] == n:
            return ["".join(r).strip() for r in a]
    return None
def _kind(dt) -> str:
    try:
        return np.dtype(dt).kind
    except TypeError:
        s = str(dt).lower()
        return "U" if ("str" in s or "char" in s or "cell" in s or "object" in s) else "O"
@register
class ArrayLibraryReader(Reader):
    name = "arrays"
    label = "Spectra matrix in MAT / HDF5 / NPZ / NPY"
    extensions = (".mat", ".h5", ".hdf5", ".he5", ".npz", ".npy", ".nc")

    def _pick(self, src):
        arrays = {k: (tuple(s), _kind(dt)) for k, (s, dt) in src.arrays.items()}
        mats = [(k, s) for k, (s, kd) in arrays.items() if len(s) == 2 and min(s) >= 2 and max(s) >= 3 and kd in "fiu"]
        vecs = {k: s[0] for k, (s, kd) in arrays.items() if len(s) == 1 and kd in "fiu" and s[0] >= 3}
        vecs.update({k: max(s) for k, (s, kd) in arrays.items() if len(s) == 2 and min(s) == 1 and kd in "fiu"
                     and max(s) >= 3})
        best = None
        for k, s in sorted(mats, key=lambda kv: -kv[1][0] * kv[1][1]):
            for wk, n in vecs.items():
                if wk == k:
                    continue
                if n == s[1]:
                    best = (k, wk, 1)
                elif n == s[0]:
                    best = (k, wk, 0)
                if best and re.search(r"wave|wl|lambda|band|nm", wk, re.I):
                    return best
            if best:
                return best
        return None

    def sniff(self, path: Path) -> float:
        if path.suffix.lower() not in self.extensions or not path.exists():
            return 0.0
        try:
            from ...core.loaders import open_source
            src = open_source(str(path))
        except Exception:                                                   # noqa: BLE001
            return 0.0
        if any(len(s) == 3 for s, _ in src.arrays.values()):
            return 0.0
        return 0.75 if self._pick(src) else 0.0

    def read(self, path: Path, opts: ReadOptions):
        from ...core.loaders import open_source
        src = open_source(str(path))
        pick = self._pick(src)
        if pick is None:
            raise ValueError(f"{path.name}: no spectra matrix + wavelength vector found")
        mk, wk, band_axis = pick
        M = np.asarray(src.reader(mk), float)
        wl = np.asarray(src.reader(wk), float).ravel()
        if band_axis == 0:
            M = M.T
        names = None
        for k, (s, dt) in src.arrays.items():
            if k in (mk, wk):
                continue
            if len(s) >= 1 and max(s) == M.shape[0] and _kind(dt) in "USO":
                try:
                    names = [str(x).strip() for x in np.asarray(src.reader(k)).ravel()]
                except Exception:                                           # noqa: BLE001
                    names = None
                break
        if names is None and path.suffix.lower() == ".mat":
            names = _mat_names(path, M.shape[0])
        names = names or [f"{stem_name(path)} {i + 1}" for i in range(M.shape[0])]
        g = str(path)
        return [SpectrumRecord(names[i], wl, M[i], wl_unit=unit_from_text(wk),
                               material_metadata={"source_file": path.name, "array": mk},
                               spectrum_metadata={"uid": f"arrays:{path.name}:{mk}:{i}", "source_file": path.name},
                               group=g) for i in range(M.shape[0])]
