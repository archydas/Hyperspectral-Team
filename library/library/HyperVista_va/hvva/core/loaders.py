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
        axis = guess_band_axis(self.arrays[big][0])
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
        cube = np.moveaxis(raw, band_axis, 2).astype(np.float32)
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


def _from_tiff(path):
    geo = None
    try:
        import rasterio
        with rasterio.open(path) as ds:
            arr = ds.read().transpose(1, 2, 0)
            t = ds.transform
            geo = GeoInfo(ds.crs.to_string() if ds.crs else None, (t.a, t.b, t.c, t.d, t.e, t.f))
    except ImportError:
        import tifffile
        with tifffile.TiffFile(path) as tf:
            arr = tf.asarray()
            page = tf.pages[0]
            sc, tie = page.tags.get("ModelPixelScaleTag"), page.tags.get("ModelTiepointTag")
            if sc is not None and tie is not None:
                sx, sy, tp = sc.value[0], sc.value[1], tie.value
                geo = GeoInfo("unknown (GeoTIFF)", (sx, 0.0, tp[3] - tp[0] * sx, 0.0, -sy, tp[4] + tp[1] * sy))
        if arr.ndim == 3 and arr.shape[0] < arr.shape[1] and arr.shape[0] < arr.shape[2]:
            arr = arr.transpose(1, 2, 0)                          # (bands, rows, cols) -> (rows, cols, bands)
    return Source(path, "tiff", {"image": (arr.shape, str(arr.dtype))}, lambda k: arr, geo=geo)


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
