"""Universal loading: any array-based hyperspectral format -> Scene.

Supported: ENVI (.hdr + data), MATLAB .mat (v5 and v7.3), HDF5 (.h5/.hdf5/.he5/.hdf), NetCDF (.nc),
Zarr (folder/.zarr), GeoTIFF/TIFF, NumPy (.npy/.npz), and headerless raw binary (you give the geometry).

Two steps so a GUI can ask the user when a file is ambiguous:

    src = open_source(path)            # lists the arrays inside (name, shape, dtype)
    guess = src.guess()                # best guess of cube / band axis / labels / wavelengths
    scene = src.build(**guess)         # or override any of the choices

`load_scene(path)` in scene.py does both steps with the guesses (what the CLI uses).
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Tuple

import numpy as np

from .geo import GeoInfo

_WL_NAMES = re.compile(r"wave|wl|lambda|center|centre|band_?nm|spectral", re.I)
_LAB_NAMES = re.compile(r"gt|label|class|truth|mask|ground|map", re.I)
_RAW_DTYPES = {"uint8": "u1", "int16": "i2", "uint16": "u2", "int32": "i4", "uint32": "u4",
               "float32": "f4", "float64": "f8", "int64": "i8", "uint64": "u8"}


class UnsupportedFormat(ValueError):
    pass


@dataclass
class Source:
    path: str
    fmt: str
    arrays: Dict[str, Tuple[tuple, str]]                       # name -> (shape, dtype)
    reader: Callable[[str], np.ndarray]
    geo: Optional[GeoInfo] = None
    wavelengths: Optional[np.ndarray] = None                   # when the format stores them natively
    note: str = ""
    # per-file corrections applied by build() to the main cube (sensor products: EnMAP, GeoTIFF nodata ...)
    nodata: Optional[float] = None                             # this value -> NaN (gap)
    band_gain: Optional[np.ndarray] = None                     # value = DN * gain + offset, per band
    band_offset: Optional[np.ndarray] = None
    scale: Optional[float] = None                              # then divide by this (e.g. 10000 -> reflectance 0-1)
    band_keep: Optional[np.ndarray] = None                     # band indices to keep, in this order
    fwhm: Optional[np.ndarray] = None
    band_axis_hint: Optional[int] = None                       # known band axis (e.g. TIFF bands, already last)

    # ---- candidates ----------------------------------------------------
    def cube_candidates(self):
        return [k for k, (s, _) in self.arrays.items() if len(s) == 3 and min(s) >= 1 and np.prod(s) > 100]

    def label_candidates(self, cube_key=None, band_axis=None):
        out = []
        sp = None
        if cube_key:
            s = list(self.arrays[cube_key][0])
            if band_axis is not None:
                s.pop(band_axis)
                sp = tuple(s)
        for k, (s, dt) in self.arrays.items():
            if len(s) == 2 and k != cube_key and (sp is None or tuple(s) == sp or tuple(s[::-1]) == sp):
                if np.dtype(dt).kind in "iub" or _LAB_NAMES.search(k):
                    out.append(k)
        return out

    def wavelength_candidates(self, n_bands=None):
        return [k for k, (s, _) in self.arrays.items()
                if (len(s) == 1 or (len(s) == 2 and 1 in s)) and (n_bands is None or max(s) == n_bands)]

    # ---- guessing ----------------------------------------------------------
    def guess(self):
        cubes = self.cube_candidates()
        if not cubes:
            raise UnsupportedFormat(f"no 3-D array (rows x cols x bands) found in {os.path.basename(self.path)}; "
                                    f"arrays: {', '.join(self.arrays) or 'none'}")
        big = max(cubes, key=lambda k: np.prod(self.arrays[k][0]))
        axis = self.band_axis_hint if self.band_axis_hint is not None else guess_band_axis(self.arrays[big][0])
        nb = self.arrays[big][0][axis]
        labs = self.label_candidates(big, axis)
        wls = [k for k in self.wavelength_candidates(nb)]
        wls.sort(key=lambda k: (not _WL_NAMES.search(k)))
        wls = [k for k in wls if _WL_NAMES.search(k) or len(wls) == 1]
        labs.sort(key=lambda k: (not _LAB_NAMES.search(k)))
        return {"cube_key": big, "band_axis": axis, "labels_key": labs[0] if labs else None,
                "wavelengths_key": wls[0] if wls else None}

    # ---- build ---------------------------------------------------------------
    def build(self, cube_key=None, band_axis=None, labels_key=None, wavelengths_key=None,
              wavelength_range=None, name=None):
        from .scene import Scene
        g = self.guess() if cube_key is None else {}
        if g:                                                     # nothing chosen: use every guess
            cube_key = g["cube_key"]
            band_axis = g["band_axis"] if band_axis is None else band_axis
            labels_key = labels_key or g["labels_key"]
            wavelengths_key = wavelengths_key or g["wavelengths_key"]
        raw = np.asarray(self.reader(cube_key))
        if raw.ndim != 3:
            raise ValueError(f"'{cube_key}' is {raw.ndim}-D, a 3-D cube is needed")
        if band_axis is None:
            band_axis = g.get("band_axis", guess_band_axis(raw.shape))
        raw = np.moveaxis(raw, band_axis, 2)
        keep = self.band_keep
        if keep is not None and np.array_equal(keep, np.arange(raw.shape[2])):
            keep = None
        if keep is not None:
            raw = raw[:, :, keep]                                 # select / reorder bands on the raw (small) dtype
        cube = raw.astype(np.float32)
        if self.nodata is not None:                               # before any scaling: compare raw values
            cube[raw == self.nodata] = np.nan
        del raw
        if self.band_gain is not None:
            g = np.asarray(self.band_gain, np.float32)
            o = None if self.band_offset is None else np.asarray(self.band_offset, np.float32)
            if keep is not None and len(g) > cube.shape[2]:
                g = g[keep]
                o = None if o is None else o[keep]
            if len(g) == cube.shape[2]:
                cube *= g
                if o is not None:
                    cube += o
        if self.scale:
            cube /= np.float32(self.scale)
        spatial = cube.shape[:2]
        labels = None
        if labels_key:
            lab = np.squeeze(np.asarray(self.reader(labels_key)))
            if lab.shape == spatial[::-1] and lab.shape != spatial:
                lab = lab.T
            if lab.shape != spatial:
                raise ValueError(f"label array {lab.shape} does not match the cube {spatial}")
            labels = np.nan_to_num(lab).astype(np.int32)
        wl = self.wavelengths
        if wavelengths_key:
            wl = np.asarray(self.reader(wavelengths_key), float).ravel()
        if wl is not None and len(wl) != cube.shape[2]:
            raise ValueError(f"{len(wl)} wavelengths for {cube.shape[2]} bands")
        if wl is None and wavelength_range:
            wl = np.linspace(wavelength_range[0], wavelength_range[1], cube.shape[2])
        if wl is not None and np.nanmax(wl) < 10:                 # micrometres -> nm
            wl = wl * 1000.0
        sc = Scene(name or os.path.splitext(os.path.basename(self.path.rstrip("/\\")))[0], cube, wl, self.geo, labels,
                   {int(i): f"class_{int(i)}" for i in np.unique(labels) if i} if labels is not None else {}, path=self.path)
        return sc


def guess_band_axis(shape):
    """The band axis is normally the shortest (images are bigger than the spectrum is long) -
    but a small tile with many bands can fool this, which is why a GUI should let the user confirm."""
    return int(np.argmin(shape))


# ----------------------------------------------------------------------
# readers per format
# ----------------------------------------------------------------------
def _from_npy(path):
    a = np.load(path, mmap_mode="r")
    return Source(path, "npy", {"data": (a.shape, str(a.dtype))}, lambda k: np.load(path))


def _from_npz(path):
    z = np.load(path, allow_pickle=False)
    arrs = {k: (z[k].shape, str(z[k].dtype)) for k in z.files if z[k].dtype.kind in "iufb"}
    return Source(path, "npz", arrs, lambda k: z[k])


def _from_mat(path):
    import scipy.io as sio
    try:
        info = [(n, s, c) for n, s, c in sio.whosmat(path) if not n.startswith("__")]
        arrs = {n: (tuple(s), "float64") for n, s, c in info if c in ("double", "single", "int8", "int16", "int32", "int64",
                                                                      "uint8", "uint16", "uint32", "uint64", "logical")}
        return Source(path, "mat", arrs, lambda k: np.asarray(sio.loadmat(path, variable_names=[k])[k]))
    except (NotImplementedError, ValueError):                    # MATLAB v7.3 = HDF5 (scipy cannot read it)
        with open(path, 'rb') as fh:
            fh.seek(0)
            if b'\x89HDF' not in fh.read(2048):
                raise
        src = _from_h5(path)
        src.fmt = "mat (v7.3)"
        rd = src.reader
        src.reader = lambda k: rd(k).T                           # MATLAB stores column-major: axes come reversed
        src.arrays = {k: (tuple(s[::-1]), d) for k, (s, d) in src.arrays.items()}
        return src


def _h5_require():
    try:
        import h5py
        return h5py
    except ImportError as e:
        raise UnsupportedFormat("reading HDF5/MATLAB v7.3/NetCDF4 needs h5py: pip install h5py") from e


def _from_h5(path):
    h5py = _h5_require()
    arrs = {}
    with h5py.File(path, "r") as f:
        def visit(name, obj):
            if isinstance(obj, h5py.Dataset) and obj.dtype.kind in "iufb" and obj.ndim >= 1:
                arrs[name] = (tuple(obj.shape), str(obj.dtype))
        f.visititems(visit)

    def rd(k):
        with h5py.File(path, "r") as f:
            return f[k][()]
    return Source(path, "hdf5", arrs, rd)


def _from_nc(path):
    try:
        import netCDF4
        ds = netCDF4.Dataset(path)
        arrs = {}

        def walk(g, prefix=""):
            for n, v in g.variables.items():
                if v.dtype.kind in "iufb":
                    arrs[prefix + n] = (tuple(v.shape), str(v.dtype))
            for n, sub in g.groups.items():
                walk(sub, prefix + n + "/")
        walk(ds)

        def rd(k):
            g = ds
            *grp, v = k.split("/")
            for x in grp:
                g = g.groups[x]
            return np.ma.filled(g.variables[v][:].astype(float), np.nan)
        return Source(path, "netcdf", arrs, rd)
    except ImportError:
        try:
            return _from_h5(path)
        except UnsupportedFormat:
            from scipy.io import netcdf_file
            nc = netcdf_file(path, mmap=False)
            arrs = {n: (tuple(v.shape), str(v.data.dtype)) for n, v in nc.variables.items() if v.data.dtype.kind in "iuf"}
            return Source(path, "netcdf3", arrs, lambda k: np.array(nc.variables[k].data))


def _from_zarr(path):
    try:
        import zarr
    except ImportError as e:
        raise UnsupportedFormat("reading Zarr needs zarr: pip install zarr") from e
    z = zarr.open(path, mode="r")
    arrs = {}
    if hasattr(z, "shape"):
        arrs["data"] = (tuple(z.shape), str(z.dtype))
        return Source(path, "zarr", arrs, lambda k: np.asarray(z[:]))

    def walk(g, prefix=""):
        for k in g.keys():
            v = g[k]
            if hasattr(v, "shape"):
                arrs[prefix + k] = (tuple(v.shape), str(v.dtype))
            else:
                walk(v, prefix + k + "/")
    walk(z)
    return Source(path, "zarr", arrs, lambda k: np.asarray(z[k][:]))


_NUM_UNIT = re.compile(r"([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*(nm|nanomet|um|µm|micro)?", re.I)


def _wl_from_texts(texts):
    """Band descriptions / metadata values like '450.5', 'Band 3 (450.5 nm)', '0.4505 micrometers' -> nm array."""
    out = []
    for t in texts:
        if t is None:
            return None
        m = None
        for m_ in _NUM_UNIT.finditer(str(t)):
            if m_.group(2) or m is None:
                m = m_
            if m_.group(2):
                break
        if m is None:
            return None
        v = float(m.group(1))
        unit = (m.group(2) or "").lower()
        out.append(v * 1000.0 if unit.startswith(("u", "µ", "micro")) else v)
    wl = np.asarray(out, float)
    if len(wl) < 2 or not np.all(np.isfinite(wl)) or len(np.unique(wl)) < len(wl) * 0.9:
        return None
    if np.nanmax(wl) < 10:
        wl = wl * 1000.0
    return wl if 200 <= wl.min() and wl.max() <= 20000 else None


def _gdal_meta_items(xml_text):
    """GDALMetadata TIFF tag -> {(name, sample): value}."""
    import xml.etree.ElementTree as ET
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return {}
    return {(it.get("name", ""), it.get("sample")): (it.text or "").strip() for it in root.iter("Item")}


def _envi_list(v):
    return [x.strip() for x in str(v).strip().strip("{}").split(",") if x.strip()]


def _sensor_sidecar(path, n_bands):
    """Wavelengths / gains / nodata from a sensor metadata XML next to the image (EnMAP *-METADATA.XML).
    Returns dict or None."""
    import xml.etree.ElementTree as ET
    folder = os.path.dirname(os.path.abspath(path))
    base = os.path.basename(path)
    prefix = base.rsplit("-", 1)[0] if "-" in base else os.path.splitext(base)[0]
    try:
        names = os.listdir(folder)
    except OSError:
        return None
    cands = [n for n in names if n.upper().endswith("METADATA.XML")]
    cands.sort(key=lambda n: (not n.startswith(prefix), n))
    for n in cands:
        try:
            root = ET.parse(os.path.join(folder, n)).getroot()
        except (ET.ParseError, OSError):
            continue
        bands = []
        for el in root.iter():
            tag = el.tag.split("}")[-1]
            if tag != "bandID":
                continue
            kids = {c.tag.split("}")[-1]: (c.text or "").strip() for c in el}
            if "wavelengthCenterOfBand" not in kids:
                continue
            try:
                bands.append((int(el.get("number", len(bands) + 1)), float(kids["wavelengthCenterOfBand"]),
                              float(kids.get("FWHMOfBand") or "nan"), float(kids.get("GainOfBand") or "nan"),
                              float(kids.get("OffsetOfBand") or "nan")))
            except ValueError:
                continue
        if len(bands) != n_bands:
            continue
        bands.sort()
        arr = np.array([b[1:] for b in bands], float)
        level = "L2A" if "L2A" in base.upper() or "L2A" in n.upper() else ("L1C" if "L1C" in base.upper() else
                                                                                 ("L1B" if "L1B" in base.upper() else "?"))
        nod = None
        for el in root.iter():
            if el.tag.split("}")[-1].lower() in ("nodatavalue", "no_data_value", "nodata"):
                try:
                    nod = float((el.text or "").strip())
                    break
                except ValueError:
                    pass
        return {"file": n, "sensor": "EnMAP" if "ENMAP" in (n + base).upper() else "sensor", "level": level,
                "wavelengths": arr[:, 0], "fwhm": arr[:, 1], "gain": arr[:, 2], "offset": arr[:, 3],
                "nodata": nod}
    return None


def _overlap_keep(wl):
    """Band order for a sensor with two detectors listed one after the other (EnMAP: VNIR 418-993 nm, then
    SWIR 902-2445 nm).  Where the second detector starts below the end of the first, the first detector's bands
    in the overlap are dropped (EnMAP-Box default 'SWIR only'); the rest is sorted by wavelength."""
    wl = np.asarray(wl, float)
    keep = np.arange(len(wl))
    drops = np.nonzero(np.diff(wl) < 0)[0]
    if len(drops):
        k = drops[0] + 1                                       # first band of the second detector
        keep = np.array([i for i in range(len(wl)) if i >= k or wl[i] < wl[k]])
    return keep[np.argsort(wl[keep], kind="stable")]


def _tiff_rasterio(path):
    import rasterio
    import warnings
    geo, descs, meta_wl = None, None, None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")                       # NotGeoreferencedWarning on plain TIFFs
        ds = rasterio.open(path)
    with ds:
        arr = ds.read().transpose(1, 2, 0)
        t = ds.transform
        plain = (t.a, t.b, t.c, t.d, abs(t.e), t.f) == (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
        if ds.crs is not None or not plain:
            geo = GeoInfo(ds.crs.to_string() if ds.crs else None, (t.a, t.b, t.c, t.d, t.e, t.f))
        nodata = ds.nodata
        descs = list(ds.descriptions)
        per_band = [ds.tags(i + 1) for i in range(ds.count)]
        for key in ("wavelength", "WAVELENGTH", "Wavelength", "center_wavelength"):
            if all(key in tg for tg in per_band):
                meta_wl = _wl_from_texts([tg[key] for tg in per_band])
                break
        if meta_wl is None:
            envi = ds.tags(ns="ENVI") or {}
            if "wavelength" in envi:
                vals = _envi_list(envi["wavelength"])
                unit = str(envi.get("wavelength_units", "")).lower()
                meta_wl = _wl_from_texts([v + (" um" if unit.startswith("micro") else "") for v in vals])
    return arr, geo, nodata, descs, meta_wl


def _tiff_tifffile(path):
    import tifffile
    geo, nodata, descs, meta_wl, items = None, None, None, None, {}
    with tifffile.TiffFile(path) as tf:
        try:
            arr = tf.asarray()
        except ValueError as e:
            if "imagecodecs" in str(e):
                raise ValueError(f"{os.path.basename(path)} is compressed - install a TIFF codec: "
                                 "pip install imagecodecs   (or pip install rasterio)") from None
            raise
        try:
            axes = tf.series[0].axes
        except Exception:                                             # noqa: BLE001
            axes = ""
        page = tf.pages[0]
        sc, tie = page.tags.get("ModelPixelScaleTag"), page.tags.get("ModelTiepointTag")
        if sc is not None and tie is not None:
            sx, sy, tp = sc.value[0], sc.value[1], tie.value
            if not (sx == 1 and sy == 1 and not any(tp[3:5])):             # (1, 1) at (0, 0) = not georeferenced
                geo = GeoInfo("unknown (GeoTIFF)", (sx, 0.0, tp[3] - tp[0] * sx, 0.0, -sy, tp[4] + tp[1] * sy))
        nd = page.tags.get("GDAL_NODATA") or page.tags.get(42113)
        if nd is not None:
            try:
                nodata = float(str(nd.value).strip().strip("\x00"))
            except ValueError:
                nodata = None
        gm = page.tags.get("GDAL_METADATA") or page.tags.get(42112)
        if gm is not None:
            items = _gdal_meta_items(gm.value)
    if arr.ndim == 3:
        if len(axes) == 3 and axes[0] not in "YX" and axes[1:] == "YX":
            arr = arr.transpose(1, 2, 0)                          # bands-first (planar / multi-page) -> last
        elif len(axes) != 3 and arr.shape[0] < arr.shape[1] and arr.shape[0] < arr.shape[2]:
            arr = arr.transpose(1, 2, 0)
    if items and arr.ndim == 3:
        nb = arr.shape[2]
        for key in ("wavelength", "WAVELENGTH", "Wavelength"):
            vals = [items.get((key, str(i))) for i in range(nb)]
            if all(v is not None for v in vals):
                meta_wl = _wl_from_texts(vals)
                break
        if meta_wl is None:                                       # ENVI metadata domain: one {list}
            v = items.get(("wavelength", None))
            if v:
                unit = str(items.get(("wavelength_units", None), "")).lower()
                vals = _envi_list(v)
                if len(vals) == nb:
                    meta_wl = _wl_from_texts([x + (" um" if unit.startswith("micro") else "") for x in vals])
        if meta_wl is None:
            descs = [items.get(("DESCRIPTION", str(i))) for i in range(nb)]
    return arr, geo, nodata, descs, meta_wl


def _tiff_pages(path):
    try:
        import tifffile
        with tifffile.TiffFile(path) as tf:
            return len(tf.pages)
    except Exception:                                                 # noqa: BLE001
        return 1


def _from_tiff(path):
    """GeoTIFF / TIFF cube.  rasterio when installed (CRS, nodata, band metadata), else tifffile.
    Picks up: no-data value -> NaN, band wavelengths from GDAL / ENVI metadata or band descriptions, sensor
    metadata next to the image (EnMAP *-METADATA.XML: wavelengths, FWHM, gains, -32768, x10000), and warns
    about quicklooks / quality layers."""
    note = []
    try:
        arr, geo, nodata, descs, meta_wl = _tiff_rasterio(path)
        if arr.shape[2] == 1 and _tiff_pages(path) > 1:           # multi-page TIFF: GDAL only shows page 1
            arr, geo2, nodata, descs, meta_wl = _tiff_tifffile(path)
            geo = geo or geo2
    except ImportError:
        arr, geo, nodata, descs, meta_wl = _tiff_tifffile(path)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    nb = arr.shape[2]
    src = Source(path, "tiff", {"image": (arr.shape, str(arr.dtype))}, lambda k: arr, geo=geo, band_axis_hint=2)
    up = os.path.basename(path).upper()
    if nodata is not None and np.isfinite(nodata) and np.dtype(arr.dtype).kind in "iuf":
        src.nodata = float(nodata)
    wl = meta_wl if meta_wl is not None and len(meta_wl) == nb else None
    if wl is None and descs and all(descs):
        wl = _wl_from_texts(descs)
    if wl is not None:
        note.append(f"{nb} band wavelengths read from the file ({wl.min():.0f}-{wl.max():.0f} nm)")
    side = _sensor_sidecar(path, nb) if nb > 3 else None
    if side is not None:
        wl = side["wavelengths"]
        src.fwhm = side["fwhm"]
        if side["nodata"] is not None:
            src.nodata = side["nodata"]
        elif src.nodata is None and side["sensor"] == "EnMAP":
            src.nodata = -32768.0                                 # EnMAP products use -32768 for no data
        short = side["file"].rsplit("-", 1)[-1] if side["file"].startswith(os.path.basename(path).rsplit("-", 1)[0]) \
            else side["file"]
        note.append(f"{side['sensor']} {side['level']}: wavelengths and FWHM of {nb} bands from {short}")
        if side["level"] in ("L1B", "L1C") and np.all(np.isfinite(side["gain"])):
            src.band_gain, src.band_offset = side["gain"], np.nan_to_num(side["offset"])
            note.append("DN converted to radiance with the per-band gain / offset")
        elif side["level"] == "L2A":
            src.scale = 10000.0
            note.append("reflectance x 10000 scaled to 0-1")
    if src.nodata is not None:
        note.append(f"no-data value {src.nodata:g} -> gaps (NaN)")
    if wl is not None:
        keep = _overlap_keep(wl)
        if len(keep) != nb or not np.array_equal(keep, np.arange(nb)):
            n_drop = nb - len(keep)
            src.band_keep = keep
            if src.fwhm is not None:
                src.fwhm = src.fwhm[keep]
            note.append("bands sorted by wavelength" + (f"; {n_drop} overlapping bands of the first detector "
                                                         f"dropped ({len(keep)} bands kept)" if n_drop else ""))
            wl = wl[keep]
        src.wavelengths = wl
    if "QUALITY" in up or "PIXELMASK" in up or "MASK" in up and nb == 1:
        note.append("this is a quality / mask layer, not spectral data - open the product's SPECTRAL_IMAGE file")
    elif "QL_" in up or "QUICKLOOK" in up or (nb <= 4 and np.dtype(arr.dtype) == np.uint8):
        note.append(f"this looks like a QUICKLOOK picture ({nb} band(s), {arr.dtype}), not spectral data - open "
                    "the product's SPECTRAL_IMAGE file instead")
    src.note = "; ".join(note)
    return src


def read_raw(path, samples, lines, bands, dtype="uint16", interleave="bsq", offset=0, big_endian=False):
    """Headerless binary cube. interleave: bsq | bil | bip. Returns (rows, cols, bands)."""
    dt = np.dtype(_RAW_DTYPES.get(dtype, dtype)).newbyteorder(">" if big_endian else "<")
    n = samples * lines * bands
    a = np.fromfile(path, dtype=dt, offset=offset, count=n)
    if a.size < n:
        raise ValueError(f"file has {a.size} values, expected {n} - check samples/lines/bands/dtype/offset")
    il = interleave.lower()
    if il == "bsq":
        return a.reshape(bands, lines, samples).transpose(1, 2, 0)
    if il == "bil":
        return a.reshape(lines, bands, samples).transpose(0, 2, 1)
    if il == "bip":
        return a.reshape(lines, samples, bands)
    raise ValueError("interleave must be bsq, bil or bip")


def raw_source(path, **params):
    a = read_raw(path, **params)
    return Source(path, "raw", {"data": (a.shape, str(a.dtype))}, lambda k: a)


# ----------------------------------------------------------------------
def open_source(path) -> Source:
    low = path.lower().rstrip("/\\")
    if os.path.isdir(path) and (low.endswith(".zarr") or os.path.exists(os.path.join(path, ".zgroup"))
                                or os.path.exists(os.path.join(path, "zarr.json")) or os.path.exists(os.path.join(path, ".zarray"))):
        return _from_zarr(path)
    ext = os.path.splitext(low)[1]
    if ext == ".npy":
        return _from_npy(path)
    if ext == ".npz":
        return _from_npz(path)
    if ext == ".mat":
        return _from_mat(path)
    if ext in (".h5", ".hdf5", ".he5", ".hdf", ".h4"):
        return _from_h5(path)
    if ext in (".nc", ".nc4", ".cdf"):
        return _from_nc(path)
    if ext in (".tif", ".tiff"):
        return _from_tiff(path)
    if ext == ".zarr":
        return _from_zarr(path)
    # unknown extension: sniff the magic bytes
    with open(path, "rb") as f:
        head = f.read(8)
    if head[:4] == b"\x89HDF":
        return _from_h5(path)
    if head[:3] == b"CDF":
        return _from_nc(path)
    if head[:6] == b"\x93NUMPY":
        return _from_npy(path)
    if head[:4] in (b"II*\x00", b"MM\x00*"):
        return _from_tiff(path)
    if head[:6] == b"MATLAB":
        return _from_mat(path)
    raise UnsupportedFormat(f"cannot tell the format of {os.path.basename(path)}; if it is headerless binary use "
                            "the raw option (samples, lines, bands, data type, interleave)")
