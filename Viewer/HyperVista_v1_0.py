"""DS3 - Hyperspectral Viewer & Analysis Tool, version 1.0 (Kapih Team).

Version 1.0 combines the multi-format loading / Preprocessing work (HyperVistav3) with the
Data Correction work (HyperVistap5):

  * Loading: ENVI, MATLAB, IDL, HDF5/HDF4, NetCDF, Zarr, GeoTIFF/GDAL, FITS, NumPy, ASD spectra.
  * Preprocessing tab: normalisation, units, common wavelength grid, spectral indices, band depth,
    statistics, export, undo.
  * Data Correction tab (four equal-width panels):
      1. Radiometric calibration (DN -> radiance / reflectance, radiance -> reflectance, editable
         per-band parameters, bad-band handling)
      2. Atmospheric correction: empirical line, dark pixel subtraction, flat field, IARR
      3. Spectral correction (smile reduction)
      4. Noise reduction: NGmeet (validation mode with PSNR / SSIM, or denoise the active cube)
  * One processing history shared by every operation that changes the cube (View > Processing
    history), written into exported cube headers, and View > Revert to original cube.
"""
import os
import re
import sys
import json
import time
import math
import copy
import threading
import warnings
import traceback
import importlib.util
from pathlib import Path
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from PyQt5 import QtWidgets, QtCore, QtGui

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.widgets import RectangleSelector
from matplotlib import patches
from matplotlib.path import Path as MplPath
from matplotlib.widgets import PolygonSelector

import spectral

# ----------------------------------------------------------------------
# Themes: "dark" and "light" (the original look)
# ----------------------------------------------------------------------
SETTINGS_ORG = "Kapih Team"

THEMES = {
    "dark": {
        "label": "Dark",
        "mpl": {
            "figure.facecolor": "#252526",
            "figure.edgecolor": "#252526",
            "savefig.facecolor": "#252526",
            "axes.facecolor": "#1e1e1e",
            "axes.edgecolor": "#8a8a8a",
            "axes.labelcolor": "#dcdcdc",
            "axes.titlecolor": "#dcdcdc",
            "text.color": "#dcdcdc",
            "xtick.color": "#c8c8c8",
            "ytick.color": "#c8c8c8",
            "grid.color": "#555555",
            "legend.facecolor": "#2d2d30",
            "legend.edgecolor": "#555555",
        },
        "pane_border": "#3f3f46",
        "info_style": "background:#2d2d30; color:#dcdcdc; border:1px solid #3f3f46;",
    },
    "light": {
        "label": "Light (original)",
        "mpl": {
            "figure.facecolor": "white",
            "figure.edgecolor": "white",
            "savefig.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "black",
            "axes.labelcolor": "black",
            "axes.titlecolor": "black",
            "text.color": "black",
            "xtick.color": "black",
            "ytick.color": "black",
            "grid.color": "#b0b0b0",
            "legend.facecolor": "white",
            "legend.edgecolor": "0.8",
        },
        "pane_border": "#d6d6d6",
        "info_style": "background:#f7f7f7; border:1px solid #d0d0d0;",
    },
}
CURRENT_THEME = "dark"
_ORIG_STYLE = None          # the platform's native Qt style, captured in main()


def set_matplotlib_theme(name):
    """Make newly created matplotlib figures use the given theme."""
    global CURRENT_THEME
    CURRENT_THEME = name
    matplotlib.rcParams.update(THEMES[name]["mpl"])


def restyle_figure(fig):
    """Apply the current theme to an existing figure (background, axes, ticks, legends)."""
    t = THEMES[CURRENT_THEME]["mpl"]
    fig.set_facecolor(t["figure.facecolor"])
    for ax in fig.axes:
        ax.set_facecolor(t["axes.facecolor"])
        for sp in ax.spines.values():
            sp.set_edgecolor(t["axes.edgecolor"])
        ax.tick_params(axis="both", which="both", colors=t["xtick.color"])
        ax.xaxis.label.set_color(t["axes.labelcolor"])
        ax.yaxis.label.set_color(t["axes.labelcolor"])
        ax.title.set_color(t["axes.titlecolor"])
        for ln in ax.get_xgridlines() + ax.get_ygridlines():
            ln.set_color(t["grid.color"])
        leg = ax.get_legend()
        if leg is not None:
            frame = leg.get_frame()
            frame.set_facecolor(t["legend.facecolor"])
            frame.set_edgecolor(t["legend.edgecolor"])
            for txt in leg.get_texts():
                txt.set_color(t["text.color"])


set_matplotlib_theme("dark")


def apply_dark_theme(app):
    """Fusion style + dark palette for the whole Qt application."""
    app.setStyle("Fusion")
    C = QtGui.QColor
    P = QtGui.QPalette
    pal = P()
    pal.setColor(P.Window, C(37, 37, 38))
    pal.setColor(P.WindowText, C(220, 220, 220))
    pal.setColor(P.Base, C(30, 30, 30))
    pal.setColor(P.AlternateBase, C(45, 45, 48))
    pal.setColor(P.ToolTipBase, C(45, 45, 48))
    pal.setColor(P.ToolTipText, C(220, 220, 220))
    pal.setColor(P.Text, C(220, 220, 220))
    pal.setColor(P.Button, C(51, 51, 55))
    pal.setColor(P.ButtonText, C(220, 220, 220))
    pal.setColor(P.BrightText, C(255, 90, 90))
    pal.setColor(P.Link, C(86, 156, 214))
    pal.setColor(P.Highlight, C(38, 120, 200))
    pal.setColor(P.HighlightedText, C(255, 255, 255))
    for role in (P.Text, P.ButtonText, P.WindowText):
        pal.setColor(P.Disabled, role, C(127, 127, 127))
    pal.setColor(P.Disabled, P.Highlight, C(70, 70, 70))
    try:
        pal.setColor(P.PlaceholderText, C(120, 120, 120))
    except Exception:
        pass
    app.setPalette(pal)
    app.setStyleSheet(
        "QToolTip { color: #dcdcdc; background-color: #2d2d30; border: 1px solid #555555; }"
    )


# Optional sklearn for k-means and supervised classification
try:
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.decomposition import NMF, PCA
    from sklearn.model_selection import StratifiedKFold, GroupKFold
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import (
        accuracy_score, cohen_kappa_score, confusion_matrix, classification_report
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from sklearn.model_selection import StratifiedGroupKFold
except ImportError:                      # scikit-learn < 1.0
    StratifiedGroupKFold = None

# TensorFlow is optional (2D CNN). It is imported lazily because it is slow to load.
TF_AVAILABLE = importlib.util.find_spec("tensorflow") is not None

APP_NAME = "DS3"
APP_SUBTITLE = "Hyperspectral Viewer & Analysis Tool"
APP_CREDIT = "Created by Kapih Team"
APP_VERSION = "1.0"


def apply_app_theme(app, name):
    """Switch the whole Qt application between the dark theme and the native light look."""
    if name == "dark":
        apply_dark_theme(app)
    else:
        if _ORIG_STYLE:
            app.setStyle(_ORIG_STYLE)
        app.setPalette(app.style().standardPalette())
        app.setStyleSheet("")


def resource_path(filename):
    import os
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_path, filename)



# ======================================================================
# Multi-format hyperspectral loading (pure Python / NumPy, no GUI code)
#
# Supported (each reader is optional and only needed for its format):
#   ENVI .hdr + data (.img .dat .raw .bil .bip .bsq .bin .rfl ...)   -> spectral
#   MATLAB .mat v5-v7.2 -> scipy | v7.3 -> h5py
#   HDF5 / HDF-EOS5 .h5 .hdf5 .he5 (PRISMA, DESIS, ...)               -> h5py
#   NetCDF .nc .nc4 .cdf (EMIT, PACE, ...)                             -> h5py / netCDF4 / scipy
#   HDF4 .hdf .h4 .L1R (Hyperion L1R, ...)                             -> pyhdf or GDAL
#   GeoTIFF / TIFF, JPEG 2000, VRT, NITF, ISIS, PDS, KEA, ERDAS .img   -> rasterio or GDAL (tifffile for TIFF)
#   ERDAS .lan / .gis                                                  -> spectral
#   NumPy .npy / .npz                                                  -> numpy
#   FITS .fits .fit .fts                                               -> astropy
# ======================================================================

class HSILoadError(Exception):
    """A file could not be read as a hyperspectral cube."""


class HSILoadCancelled(Exception):
    """The user cancelled a dataset / axis choice while loading."""


def _try_import(name):
    try:
        return importlib.import_module(name)
    except Exception:
        return None


# (label, extensions without dot). Labels avoid parentheses (Qt filter syntax).
HSI_FORMATS = [
    ("ENVI - header or data file", ["hdr", "img", "dat", "raw", "bil", "bip", "bsq", "bin", "rfl", "cube", "sc"]),
    ("MATLAB v5 - v7.3", ["mat"]),
    ("HDF5 / HDF-EOS5 - PRISMA, DESIS, ...", ["h5", "hdf5", "he5"]),
    ("HDF4 / HDF-EOS - Hyperion L1R, ...", ["hdf", "h4", "hdf4", "he4", "l1r"]),
    ("NetCDF - EMIT, PACE, ...", ["nc", "nc4", "cdf", "netcdf"]),
    ("GeoTIFF / TIFF - EnMAP, Hyperion L1T, ...", ["tif", "tiff", "gtif", "gtiff"]),
    ("NumPy arrays", ["npy", "npz"]),
    ("FITS", ["fits", "fit", "fts"]),
    ("ERDAS - LAN, GIS, Imagine", ["lan", "gis", "img"]),
    ("Zarr - zipped store, or zarr.json / .zgroup inside a .zarr folder",
     ["zarr.zip", "zip", "!zarr.json", "!.zgroup", "!.zarray", "!.zmetadata"]),
    ("IDL save files", ["sav"]),
    ("ASD FieldSpec point spectra - imported as reference spectra", ["asd"]),
    ("Other GDAL rasters - VRT, JPEG 2000, NITF, ISIS, PDS, KEA, ER Mapper, PCIDSK, ECW, MrSID",
     ["vrt", "jp2", "j2k", "ntf", "nitf", "cub", "lbl", "pds", "kea", "rst", "ers", "pix", "ecw", "sid"]),
]

_ENVI_EXTS = {"." + e for e in HSI_FORMATS[0][1]}
_HDF5_EXTS = {".h5", ".hdf5", ".he5"}
_HDF4_EXTS = {".hdf", ".h4", ".hdf4", ".he4", ".l1r"}
_NETCDF_EXTS = {".nc", ".nc4", ".cdf", ".netcdf"}
_TIFF_EXTS = {".tif", ".tiff", ".gtif", ".gtiff"}
_FITS_EXTS = {".fits", ".fit", ".fts"}
# extensions that are never ENVI binaries, so a stray .hdr next to them is ignored
_NON_ENVI_EXTS = ({"." + e for _, exts in HSI_FORMATS for e in exts} - _ENVI_EXTS)

LARGE_CUBE_ELEMENTS = 150_000_000   # above this the cube is kept as float32 to save memory


def hsi_file_filter():
    """Qt file-dialog filter string covering every supported format."""
    def pats(exts):
        out = []
        for e in exts:
            if e.startswith("!"):          # literal file name, e.g. zarr.json
                out.append(e[1:])
                continue
            out += [f"*.{e}", f"*.{e.upper()}"]
        return " ".join(dict.fromkeys(out))

    all_exts = list(dict.fromkeys(e for _, exts in HSI_FORMATS for e in exts))
    parts = [f"All hyperspectral formats ({pats(all_exts)})"]
    parts += [f"{label} ({pats(exts)})" for label, exts in HSI_FORMATS]
    parts.append("All files (*)")
    return ";;".join(parts)


_NUM_RE = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")
_WL_NAME_RE = re.compile(
    r"wave\s*_?length|\bwl\b|(^|_)wl($|_)|wvl|lambda|band_?cent|cent(er|re)_?wav|list_cw|^cw", re.I
)
_FILL_KEYS = ("_FillValue", "fill_value", "fillvalue", "missing_value", "nodata", "nodata_value",
              "no_data_value", "data ignore value")
_SCALE_KEYS = ("scale_factor", "scalefactor", "scale")
_OFFSET_KEYS = ("add_offset", "addoffset", "offset")
_UNIT_KEYS = ("units", "unit", "wavelength units", "wavelength_units")


def _norm_key(k):
    return re.sub(r"[^a-z0-9]", "", str(k).lower())


def _attrs_lookup(attrs, keys):
    if not attrs:
        return None
    norm = {_norm_key(k): v for k, v in attrs.items()}
    for k in keys:
        if _norm_key(k) in norm:
            return norm[_norm_key(k)]
    return None


def _attr_to_text(v, max_items=4096):
    """Metadata value -> str or list of plain Python values (for the metadata dict)."""
    try:
        if isinstance(v, bytes):
            return v.decode("utf-8", "replace")
        a = np.asarray(v)
        if a.ndim == 0:
            x = a.item()
            return x.decode("utf-8", "replace") if isinstance(x, bytes) else x
        if a.size > max_items:
            return f"<array {a.shape} {a.dtype}>"
        return [x.decode("utf-8", "replace") if isinstance(x, bytes) else x for x in a.ravel().tolist()]
    except Exception:
        return str(v)


def _to_float_vector(v):
    """Numeric 1-D vector from arrays, lists or ENVI-style strings ("{450, 460, ...}")."""
    try:
        if isinstance(v, (bytes, str)):
            v = [v]
        a = np.asarray(v)
        if a.dtype.kind in "SUO":
            text = " ".join(x.decode("utf-8", "replace") if isinstance(x, bytes) else str(x)
                            for x in a.ravel())
            a = np.array([float(t) for t in _NUM_RE.findall(text)], dtype=float)
        a = np.squeeze(np.asarray(a, dtype=float))
    except Exception:
        return None
    if a.ndim != 1 or a.size < 2:
        return None
    return a


def _scalar(v):
    if v is None:
        return None
    try:
        if isinstance(v, (bytes, str)):
            s = v.decode("utf-8", "replace") if isinstance(v, bytes) else v
            m = _NUM_RE.findall(s)
            return float(m[0]) if m else None
        a = np.asarray(v).ravel()
        return float(a[0]) if a.size else None
    except Exception:
        return None


def normalize_wavelengths(wl, units=None):
    """Return wavelengths in nanometres (micrometres, Angstrom, metres are converted)."""
    if wl is None:
        return None
    wl = np.asarray(wl, dtype=float).ravel()
    finite = wl[np.isfinite(wl)]
    if finite.size == 0 or np.all(finite == 0):
        return None
    u = str(units or "").strip().lower()
    if not u or u in ("unknown", "none", "n/a", "index"):
        return wl * 1000.0 if np.nanmax(np.abs(finite)) < 30 else wl
    if "wavenumber" in u or "cm-1" in u or "cm^-1" in u or "1/cm" in u:
        return wl
    if "nano" in u or u == "nm":
        return wl
    if "micro" in u or u in ("um", "µm", "μm"):
        return wl * 1e3
    if "angstrom" in u or u in ("a", "å", "aa"):
        return wl / 10.0
    if "milli" in u or u == "mm":
        return wl * 1e6
    if u in ("m", "meter", "meters", "metre", "metres"):
        return wl * 1e9
    return wl


def _finalize_array(arr, fill=None, scale=None, offset=None):
    """Raw (rows, cols, bands) array -> C-contiguous float cube with fill->NaN and scaling applied."""
    arr = np.asarray(arr)
    dtype = np.float32 if arr.size > LARGE_CUBE_ELEMENTS else np.float64
    fill = _scalar(fill)
    mask = (arr == fill) if (fill is not None and np.isfinite(fill)) else None
    out = np.array(arr, dtype=dtype, order="C")
    del arr
    if mask is not None and mask.any():
        out[mask] = np.nan
    nb = out.shape[-1]
    for value, neutral, op in ((scale, 1.0, "mul"), (offset, 0.0, "add")):
        if value is None:
            continue
        v = np.asarray(value, dtype=float).ravel()
        if v.size == 1:
            v = v.reshape(())
        elif v.size == nb:
            v = v.reshape(1, 1, nb)
        else:
            continue
        if np.allclose(v, neutral):
            continue
        if op == "mul":
            out *= v.astype(dtype)
        else:
            out += v.astype(dtype)
    return out


# ---------------- ENVI -------------------------------------------------
def _is_envi_header(path):
    try:
        with open(path, "rb") as f:
            return f.read(256).lstrip().upper().startswith(b"ENVI")
    except Exception:
        return False


def find_envi_header(path):
    """The ENVI header belonging to a data file (foo.img.hdr or foo.hdr), or None."""
    p = Path(path)
    if p.suffix.lower() == ".hdr":
        return p
    for c in (Path(str(p) + ".hdr"), p.with_suffix(".hdr"),
              Path(str(p) + ".HDR"), p.with_suffix(".HDR")):
        if c.is_file() and c != p:
            return c
    return None


def find_envi_data_file(hdr):
    """The binary data file belonging to an ENVI header, or None."""
    hdr = Path(hdr)
    base = hdr.with_suffix("")          # foo.img.hdr -> foo.img ; foo.hdr -> foo
    exts = ["", ".img", ".dat", ".raw", ".bil", ".bip", ".bsq", ".bin", ".rfl", ".cube", ".sc"]
    for ext in exts + [e.upper() for e in exts if e]:
        c = Path(str(base) + ext)
        if c.is_file() and c != hdr:
            return c
    return None


_LOAD_PROGRESS_CB = None      # set by the GUI while a file loads: cb(percent, message)


def _load_progress(pct, msg=""):
    cb = _LOAD_PROGRESS_CB
    if cb is not None:
        try:
            cb(pct, msg)
        except Exception:
            pass


def _read_envi_with_progress(img):
    """Read an ENVI image in row chunks so the GUI can show real progress (falls back to img.load())."""
    try:
        mm = img.open_memmap(writable=False)
        if mm.ndim != 3:
            raise ValueError("unexpected memmap shape")
        out = np.empty(mm.shape, dtype=mm.dtype)
        rows = mm.shape[0]
        row_bytes = max(1, int(np.prod(mm.shape[1:])) * mm.dtype.itemsize)
        step = max(1, int(32 * 1024 * 1024 // row_bytes))
        for r0 in range(0, rows, step):
            out[r0:r0 + step] = mm[r0:r0 + step]
            _load_progress(10 + 60.0 * min(rows, r0 + step) / rows, "Reading pixel data")
        del mm
        return out
    except MemoryError:
        raise
    except Exception:
        return np.asarray(img.load())


def _load_envi(path):
    p = Path(path)
    hdr = find_envi_header(p)
    if hdr is None:
        raise HSILoadError(f"No ENVI header (.hdr) found for {p.name}.")
    data = p if p.suffix.lower() != ".hdr" else find_envi_data_file(hdr)
    try:
        img = spectral.envi.open(str(hdr), str(data) if data is not None else None)
    except Exception as e:
        raise HSILoadError(f"Failed to open ENVI file:\n{e}")
    if not hasattr(img, "load"):
        raise HSILoadError("This ENVI header describes a spectral library (.sli), not an image cube.")
    arr = _read_envi_with_progress(img)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    rows, cols, bands = arr.shape[-3], arr.shape[-2], arr.shape[-1]
    meta = dict(getattr(img, "metadata", None) or {})
    _load_progress(72, "Cleaning invalid values")
    cube = _finalize_array(arr.reshape(rows, cols, bands), fill=meta.get("data ignore value"))
    wl = _to_float_vector(meta["wavelength"]) if "wavelength" in meta else None
    if wl is not None and wl.size != bands:
        wl = None
    return {"cube": cube, "wavelengths": wl, "wavelength_units": meta.get("wavelength units"),
            "metadata": meta, "format": "ENVI", "dataset": None}


def _load_envi_or_fallback(p, choose, ask_axis):
    try:
        return _load_envi(p)
    except MemoryError:
        raise
    except Exception as envi_error:
        # GDAL's ENVI / EHdr drivers sometimes read files the spectral package cannot
        data = p if p.suffix.lower() != ".hdr" else find_envi_data_file(p)
        if data is not None:
            try:
                return _load_raster(data, choose, ask_axis)
            except HSILoadCancelled:
                raise
            except Exception:
                pass
        if isinstance(envi_error, HSILoadError):
            raise
        raise HSILoadError(f"Failed to open ENVI file:\n{envi_error}")


def _load_spectral_open(p):
    """spectral.open_image: ENVI headers, ERDAS .lan / .gis."""
    img = spectral.open_image(str(p))
    arr = np.asarray(img.load())
    if arr.ndim == 2:
        arr = arr[:, :, None]
    meta = dict(getattr(img, "metadata", None) or {})
    wl = None
    bands_info = getattr(img, "bands", None)
    if bands_info is not None and getattr(bands_info, "centers", None):
        wl = _to_float_vector(bands_info.centers)
    return {"cube": _finalize_array(arr), "wavelengths": wl,
            "wavelength_units": getattr(bands_info, "band_unit", None) if bands_info else None,
            "metadata": meta, "format": "spectral (" + Path(str(p)).suffix.lstrip(".").upper() + ")",
            "dataset": None}


# ---------------- generic array containers -----------------------------
def _add_entry(name, shape, dtype, cands, vecs, loader, attrs=None):
    """Register a 3-D cube candidate or a wavelength-like 1-D vector found in a container."""
    try:
        kind = np.dtype(dtype).kind
    except Exception:
        return
    if kind not in "biuf":
        return
    shape = tuple(int(s) for s in shape)
    nontrivial = tuple(s for s in shape if s > 1)
    leaf = str(name).replace("\\", "/").split("/")[-1]
    if len(nontrivial) == 3:
        cands.append({"name": str(name), "shape": nontrivial, "dtype": str(np.dtype(dtype)),
                      "load": loader, "attrs": dict(attrs or {})})
    elif len(nontrivial) == 1 and _WL_NAME_RE.search(leaf):
        try:
            v = _to_float_vector(loader())
        except Exception:
            v = None
        if v is not None:
            u = _attrs_lookup(attrs, _UNIT_KEYS)
            vecs.append((str(name), v, _attr_to_text(u) if u is not None else None))


def _add_attr_vector(label, key, value, vecs):
    if _WL_NAME_RE.search(str(key)):
        v = _to_float_vector(value)
        if v is not None:
            vecs.append((label, v, None))


_GOOD_WORDS = ("reflect", "radian", "cube", "hsi", "hyper", "spectr", "toa", "surface", "data", "image", "img")
_BAD_WORDS = ("label", "mask", "truth", "class", "error", "uncert", "qual", "flag", "cloud",
              "latitude", "longitude", "geoloc", "angle", "fwhm", "noise", "snr", "lat", "lon")


def _candidate_score(name):
    n = str(name).lower()
    leaf = re.split(r"[/:\\]", n)[-1]
    s = 0
    if any(w in n for w in _GOOD_WORDS):
        s += 2
    if any(w in leaf for w in _BAD_WORDS) or leaf == "gt" or leaf.endswith("_gt"):
        s -= 3
    return s


def _choose_index(labels, choose, title, label):
    if len(labels) <= 1 or choose is None:
        return 0
    i = choose(title, label, labels, 0)
    if i is None:
        raise HSILoadCancelled()
    return i


_BAND_DIM_RE = re.compile(r"band|wave|wl|lambda|spectr|channel", re.I)


def _band_axis_from_dims(dims, shape):
    """Spectral axis from named dimensions (xarray / NetCDF / Zarr), or None."""
    try:
        dims = [str(d) for d in dims]
    except Exception:
        return None
    if len(dims) != 3 or len(shape) != 3 or any(int(n) <= 1 for n in shape):
        return None
    hits = [i for i, d in enumerate(dims) if _BAND_DIM_RE.search(d)]
    return hits[0] if len(hits) == 1 else None


def guess_band_axis(shape, wl_lengths=()):
    """(axis, confident). Spectral axis guessed from wavelength lengths or the array shape."""
    matches = [ax for ax in range(3) if shape[ax] in set(wl_lengths)]
    if len(matches) == 1:
        return matches[0], True
    a, b, c = shape
    if a == b != c:
        return 2, True
    if b == c != a:
        return 0, True
    if a == c != b:
        return 1, True
    # remote-sensing scenes almost always have fewer bands than rows / columns
    return int(np.argmin(shape)), False


def _finish_container(fmt, cands, vecs, meta, closer=None, choose=None, ask_axis=None):
    try:
        if not cands:
            raise HSILoadError(f"No 3-D numeric array (hyperspectral cube) was found in this {fmt} file.")
        cands = sorted(cands, key=lambda c: (_candidate_score(c["name"]), int(np.prod(c["shape"]))),
                       reverse=True)
        labels = [f"{c['name']}    [{' x '.join(map(str, c['shape']))}, {c['dtype']}]" for c in cands]
        cand = cands[_choose_index(
            labels, choose, "Choose the hyperspectral cube",
            "This file contains several 3-D arrays. Which one is the hyperspectral cube?")]
        arr = np.squeeze(np.asarray(cand["load"]()))
    finally:
        if closer is not None:
            try:
                closer()
            except Exception:
                pass
    if arr.ndim != 3:
        raise HSILoadError(f"Dataset '{cand['name']}' is not 3-D after loading (shape {arr.shape}).")

    if cand.get("band_axis") is not None:
        axis, confident = int(cand["band_axis"]), True
    else:
        axis, confident = guess_band_axis(arr.shape, [v.size for _, v, _ in vecs])
    if not confident and ask_axis is not None:
        a = ask_axis(arr.shape, axis)
        if a is None:
            raise HSILoadCancelled()
        axis = int(a)
    attrs = cand["attrs"]
    cube = _finalize_array(np.moveaxis(arr, axis, -1),
                           fill=_attrs_lookup(attrs, _FILL_KEYS),
                           scale=_scalar(_attrs_lookup(attrs, _SCALE_KEYS)),
                           offset=_scalar(_attrs_lookup(attrs, _OFFSET_KEYS)))
    del arr
    nb = cube.shape[2]
    wl = units = None
    matching = [v for v in vecs if v[1].size == nb]
    if matching:
        matching.sort(key=lambda v: 0 if "wave" in v[0].lower() else 1)
        _, wl, units = matching[0]
    meta = dict(meta or {})
    meta["dataset"] = cand["name"]
    if axis != 2:
        meta["original band axis"] = axis
    return {"cube": cube, "wavelengths": wl, "wavelength_units": units,
            "metadata": meta, "format": fmt, "dataset": cand["name"]}


def _read_hdf5(p, matlab=False):
    h5py = _try_import("h5py")
    if h5py is None:
        raise HSILoadError("Reading HDF5 / NetCDF4 / MATLAB v7.3 files needs h5py:\n  pip install h5py")
    f = h5py.File(str(p), "r")
    cands, vecs, meta = [], [], {}

    def attrs_of(obj):
        out = {}
        try:
            for k, v in obj.attrs.items():
                out[str(k)] = v
        except Exception:
            pass
        return out

    for k, v in attrs_of(f).items():
        meta[k] = _attr_to_text(v)
        _add_attr_vector(f"@{k}", k, v, vecs)

    def visit(name, obj):
        a = attrs_of(obj)
        for k, v in a.items():
            _add_attr_vector(f"{name}@{k}", k, v, vecs)
        if isinstance(obj, h5py.Dataset):
            try:
                dt = obj.dtype
            except Exception:
                return
            if matlab:   # MATLAB stores column-major: reverse the axes
                _add_entry(name, obj.shape[::-1], dt, cands, vecs, lambda o=obj: np.transpose(o[()]), a)
            else:
                _add_entry(name, obj.shape, dt, cands, vecs, lambda o=obj: o[()], a)

    f.visititems(visit)
    fmt = "MATLAB v7.3" if matlab else ("NetCDF4" if Path(str(p)).suffix.lower() in _NETCDF_EXTS else "HDF5")
    return fmt, cands, vecs, meta, f.close


def _read_mat(p):
    sio = _try_import("scipy.io")
    if sio is not None:
        try:
            d = sio.loadmat(str(p))
        except NotImplementedError:          # MATLAB v7.3 files are HDF5
            return _read_hdf5(p, matlab=True)
        except Exception as e:
            h5py = _try_import("h5py")
            if h5py is not None and h5py.is_hdf5(str(p)):
                return _read_hdf5(p, matlab=True)
            raise HSILoadError(f"Failed to read MATLAB file:\n{e}")
        cands, vecs = [], []
        for k, v in d.items():
            if k.startswith("__") or not isinstance(v, np.ndarray):
                continue
            _add_entry(k, v.shape, v.dtype, cands, vecs, lambda v=v: v)
        meta = {"MATLAB header": _attr_to_text(d.get("__header__", ""))}
        return "MATLAB", cands, vecs, meta, None
    h5py = _try_import("h5py")
    if h5py is not None and h5py.is_hdf5(str(p)):
        return _read_hdf5(p, matlab=True)
    raise HSILoadError("Reading .mat files needs SciPy (and h5py for v7.3 files):\n  pip install scipy h5py")


def _read_numpy(p):
    obj = np.load(str(p), allow_pickle=False)
    if isinstance(obj, np.ndarray):
        items = {p.stem: obj}
    else:
        items = {k: obj[k] for k in obj.files}
        obj.close()
    cands, vecs = [], []
    for k, v in items.items():
        _add_entry(k, v.shape, v.dtype, cands, vecs, lambda v=v: v)
    return "NumPy", cands, vecs, {}, None


def _read_fits(p):
    fits = _try_import("astropy.io.fits")
    if fits is None:
        raise HSILoadError("Reading FITS files needs astropy:\n  pip install astropy")
    hdul = fits.open(str(p), memmap=False)
    cands, vecs = [], []
    meta = {}
    try:
        for k, v in hdul[0].header.items():
            if k:
                meta[str(k)] = _attr_to_text(v)
    except Exception:
        pass
    for i, hdu in enumerate(hdul):
        data = hdu.data
        if not isinstance(data, np.ndarray) or data.dtype.names:
            continue
        name = f"HDU {i}" + (f" ({hdu.name})" if hdu.name else "")
        _add_entry(name, data.shape, data.dtype, cands, vecs, lambda d=data: d)
        hdr = hdu.header
        if data.ndim == 3 and "CRVAL3" in hdr:       # spectral WCS on FITS axis 3 = NumPy axis 0
            n = data.shape[0]
            cdelt = hdr.get("CDELT3", hdr.get("CD3_3", 1.0))
            wl = hdr["CRVAL3"] + (np.arange(n) + 1 - hdr.get("CRPIX3", 1.0)) * cdelt
            vecs.append((f"{name} spectral WCS (wavelength)", np.asarray(wl, float), hdr.get("CUNIT3")))
    return "FITS", cands, vecs, meta, hdul.close


def _read_netcdf_classic(p):
    """NetCDF files that are not HDF5 based (NetCDF3 / classic)."""
    nc = _try_import("netCDF4")
    cands, vecs = [], []
    if nc is not None:
        ds = nc.Dataset(str(p), "r")
        ds.set_auto_maskandscale(False)       # fill values / scaling are applied by the loader

        def walk(group, prefix):
            for name, var in group.variables.items():
                attrs = {k: var.getncattr(k) for k in var.ncattrs()}
                n_before = len(cands)
                _add_entry(prefix + name, var.shape, var.dtype, cands, vecs,
                           lambda v=var: np.asarray(v[...]), attrs)
                if len(cands) > n_before:
                    cands[-1]["band_axis"] = _band_axis_from_dims(var.dimensions, var.shape)
            for gname, sub in group.groups.items():
                walk(sub, f"{prefix}{gname}/")

        walk(ds, "")
        meta = {k: _attr_to_text(ds.getncattr(k)) for k in ds.ncattrs()}
        return "NetCDF", cands, vecs, meta, ds.close
    sio = _try_import("scipy.io")
    if sio is not None:
        f = sio.netcdf_file(str(p), "r", mmap=False)
        for name, var in f.variables.items():
            attrs = dict(getattr(var, "_attributes", {}))
            _add_entry(name, var.shape, var.data.dtype, cands, vecs, lambda v=var: np.array(v.data), attrs)
        meta = {str(k): _attr_to_text(v) for k, v in getattr(f, "_attributes", {}).items()}
        return "NetCDF3", cands, vecs, meta, f.close
    raise HSILoadError("Reading NetCDF files needs h5py, netCDF4 or SciPy:\n  pip install h5py netCDF4")


def _read_hdf4(p):
    SD = _try_import("pyhdf.SD")
    if SD is None:
        return None
    types = {3: np.uint8, 20: np.int8, 21: np.uint8, 22: np.int16, 23: np.uint16,
             24: np.int32, 25: np.uint32, 5: np.float32, 6: np.float64}
    sd = SD.SD(str(p), SD.SDC.READ)
    cands, vecs = [], []
    for name, info in sd.datasets().items():
        dims = info[1] if isinstance(info[1], (list, tuple)) else (info[1],)
        dtype = types.get(info[2])
        if dtype is None:
            continue
        try:
            attrs = sd.select(name).attributes()
        except Exception:
            attrs = {}
        _add_entry(name, dims, dtype, cands, vecs, lambda n=name: np.asarray(sd.select(n).get()), attrs)
    meta = {}
    try:
        meta = {str(k): _attr_to_text(v) for k, v in sd.attributes().items()}
    except Exception:
        pass
    return "HDF4", cands, vecs, meta, sd.end


def _load_hdf_family(p, choose, ask_axis):
    errors = []
    h5py = _try_import("h5py")
    if h5py is not None and h5py.is_hdf5(str(p)):
        return _finish_container(*_read_hdf5(p), choose=choose, ask_axis=ask_axis)
    ext = p.suffix.lower()
    if ext in _NETCDF_EXTS:
        try:
            return _finish_container(*_read_netcdf_classic(p), choose=choose, ask_axis=ask_axis)
        except HSILoadCancelled:
            raise
        except Exception as e:
            errors.append(f"NetCDF reader: {e}")
    if ext in _HDF4_EXTS or ext not in (_HDF5_EXTS | _NETCDF_EXTS):
        try:
            c = _read_hdf4(p)
            if c is not None:
                return _finish_container(*c, choose=choose, ask_axis=ask_axis)
            errors.append("pyhdf not installed (pip install pyhdf) - trying GDAL instead.")
        except HSILoadCancelled:
            raise
        except Exception as e:
            errors.append(f"HDF4 reader: {e}")
    try:
        return _load_raster(p, choose, ask_axis)
    except HSILoadCancelled:
        raise
    except Exception as e:
        errors.append(str(e))
    hint = "" if h5py is not None else "\n\nFor HDF5 / NetCDF4 files install h5py:  pip install h5py"
    raise HSILoadError("Could not read this HDF / NetCDF file.\n\n" + "\n".join(errors) + hint)


# ---------------- Zarr ----------------------------------------------------
_ZARR_MARKERS = (".zgroup", ".zarray", ".zattrs", ".zmetadata", "zarr.json")


def _is_zarr_dir(d):
    d = Path(d)
    return d.is_dir() and (d.name.lower().endswith(".zarr")
                           or any((d / m).is_file() for m in _ZARR_MARKERS))


def _find_zarr_root(p):
    """Root folder of a Zarr store given the folder itself or any file inside it, else None."""
    p = Path(p)
    start = p if p.is_dir() else p.parent
    chain = [start, *start.parents]
    if p.is_file() and p.name not in _ZARR_MARKERS \
            and not any(d.name.lower().endswith(".zarr") for d in chain):
        return None
    found = None
    for d in chain:
        if d.name.lower().endswith(".zarr") and d.is_dir():
            return d
        if _is_zarr_dir(d):
            found = d
        elif found is not None:
            break
    return found


def _read_zarr(target):
    """Zarr v2 / v3 store: a .zarr folder or a zipped store (.zarr.zip / .zip)."""
    zarr = _try_import("zarr")
    if zarr is None:
        raise HSILoadError("Reading Zarr stores needs the zarr package:\n  pip install zarr")
    target = Path(target)
    closer = None
    try:
        if target.is_file():
            store = zarr.storage.ZipStore(str(target), mode="r")
            closer = store.close
            root = zarr.open(store, mode="r")
        else:
            root = zarr.open(str(target), mode="r")
    except Exception as e:
        if closer is not None:
            try:
                closer()
            except Exception:
                pass
        raise HSILoadError(f"Could not open '{target.name}' as a Zarr store:\n{e}")

    cands, vecs = [], []

    def attrs_of(node):
        try:
            return dict(node.attrs)
        except Exception:
            return {}

    def add_array(name, arr):
        attrs = attrs_of(arr)
        dims = attrs.get("_ARRAY_DIMENSIONS") or getattr(getattr(arr, "metadata", None), "dimension_names", None)
        # xarray keeps the fill value in the array metadata rather than in the attributes
        if dims and _attrs_lookup(attrs, _FILL_KEYS) is None and getattr(arr, "fill_value", None) is not None:
            attrs["_FillValue"] = arr.fill_value
        n_before = len(cands)
        _add_entry(name, arr.shape, arr.dtype, cands, vecs, lambda a=arr: np.asarray(a[...]), attrs)
        if len(cands) > n_before and dims:
            cands[-1]["band_axis"] = _band_axis_from_dims(dims, arr.shape)
        for k, v in attrs.items():
            _add_attr_vector(f"{name}@{k}", k, v, vecs)

    def walk(group, prefix, depth=0):
        if depth > 8:
            return
        for name, arr in group.arrays():
            add_array(prefix + name, arr)
        for name, sub in group.groups():
            walk(sub, f"{prefix}{name}/", depth + 1)

    meta = {}
    for k, v in attrs_of(root).items():
        meta[str(k)] = _attr_to_text(v)
        _add_attr_vector(f"@{k}", k, v, vecs)
    if hasattr(root, "groups") and hasattr(root, "arrays"):
        walk(root, "")
    else:                                      # the store itself is a single array
        add_array(target.name, root)
    return "Zarr", cands, vecs, meta, closer


# ---------------- IDL save files ---------------------------------------------
def _read_idl_sav(p):
    sio = _try_import("scipy.io")
    if sio is None:
        raise HSILoadError("Reading IDL .sav files needs SciPy:\n  pip install scipy")
    try:
        d = sio.readsav(str(p), python_dict=True, verbose=False)
    except Exception as e:
        raise HSILoadError(f"Failed to read IDL save file:\n{e}")
    cands, vecs = [], []

    def visit(name, v, depth=0):
        if depth > 4 or not isinstance(v, np.ndarray):
            return
        if v.dtype.names:                      # IDL structure: look inside its fields
            if v.size == 1:
                rec = v.ravel()[0]
                for field in v.dtype.names:
                    visit(f"{name}.{field}", np.asarray(rec[field]), depth + 1)
            return
        if v.dtype == object:
            if v.size == 1:
                visit(name, np.asarray(v.ravel()[0]), depth + 1)
            return
        _add_entry(name, v.shape, v.dtype, cands, vecs, lambda v=v: v)

    for k, v in d.items():
        visit(str(k), np.asarray(v) if not isinstance(v, np.ndarray) else v)
    return "IDL save", cands, vecs, {}, None


# ---------------- ASD FieldSpec spectra (.asd) ---------------------------------
# Binary layout per the ASD file-format spec, cross-checked with SpecDAL and asdreader:
# 484-byte header, spectrum (channels values), then for version >= 2 a reference block:
# flag (2 B), reference time (8 B), spectrum time (8 B), description (2-B length + text),
# white-reference spectrum (channels values, same data format as the spectrum).
ASD_DATA_TYPES = {0: "raw DN", 1: "reflectance", 2: "radiance", 3: "no units", 4: "irradiance",
                  5: "QI", 6: "transmittance", 7: "unknown", 8: "absorbance"}
_ASD_FORMATS = {0: ("<f4", 4), 1: ("<i4", 4), 2: ("<f8", 8)}


def _asd_detector_slices(wl, splice1, splice2):
    s1 = splice1 if splice1 is not None and np.isfinite(splice1) and 400 < splice1 < 2400 else 1000.0
    s2 = splice2 if splice2 is not None and np.isfinite(splice2) and s1 < splice2 < 2450 else 1800.0
    i1 = int(np.searchsorted(wl, s1, side="right"))   # first SWIR1 channel
    i2 = int(np.searchsorted(wl, s2, side="right"))   # first SWIR2 channel
    return i1, i2


def asd_splice_correction(wl, y, splice1=None, splice2=None):
    """Remove the detector jumps at ~1000 nm and ~1800 nm.

    SWIR1 is kept as the reference; VNIR and SWIR2 are shifted additively so that
    each meets the linear extrapolation of SWIR1 at the splice.
    """
    y = np.asarray(y, dtype=float).copy()
    wl = np.asarray(wl, dtype=float)
    n = len(y)
    i1, i2 = _asd_detector_slices(wl, splice1, splice2)
    if 1 <= i1 <= n - 2 and np.all(np.isfinite(y[i1 - 1:i1 + 2])):
        predicted = 2 * y[i1] - y[i1 + 1]
        y[:i1] += predicted - y[i1 - 1]
    if i1 + 2 <= i2 <= n - 1 and np.all(np.isfinite(y[i2 - 2:i2 + 1])):
        predicted = 2 * y[i2 - 1] - y[i2 - 2]
        y[i2:] += predicted - y[i2]
    return y


def read_asd(path, splice_correction=True):
    """Read an ASD FieldSpec binary spectrum file (file versions 1-8).

    Returns a dict with wavelengths (nm), values, kind, target, reference and meta.
    Whenever a white reference is stored (raw, reflectance and radiance files) the result is
    reflectance = target / white reference; otherwise the stored values are returned.
    """
    import struct
    p = Path(path)
    buf = p.read_bytes()
    if len(buf) < 484:
        raise HSILoadError(f"{p.name}: too small to be an ASD file.")
    magic = buf[:3]
    if magic in (b"ASD", b"asd"):
        version = 1
    elif magic[:2] == b"as" and magic[2:3].isdigit():
        version = int(magic[2:3])
    else:
        raise HSILoadError(f"{p.name}: not an ASD binary file (it starts with {magic!r}).")

    def u(fmt, off):
        return struct.unpack_from("<" + fmt, buf, off)[0]

    data_type = u("B", 186)
    start, step = u("f", 191), u("f", 195)
    fmt_code, n = u("B", 199), u("H", 204)
    it = u("I", 390)
    instrument_num = u("H", 400)
    swir1_gain, swir2_gain = u("H", 436), u("H", 438)
    splice1, splice2 = u("f", 444), u("f", 448)
    comments = buf[3:160].split(b"\0")[0].decode("latin-1", "replace").strip()
    if fmt_code not in _ASD_FORMATS:
        raise HSILoadError(f"{p.name}: unknown ASD data format code {fmt_code}.")
    if not (0 < n < 100000) or not np.isfinite(start) or not np.isfinite(step) or step <= 0:
        raise HSILoadError(f"{p.name}: invalid ASD header (channels={n}, start={start}, step={step}).")
    dt, size = _ASD_FORMATS[fmt_code]
    end = 484 + n * size
    if len(buf) < end:
        raise HSILoadError(f"{p.name}: file is truncated ({len(buf)} bytes, expected at least {end}).")
    target = np.frombuffer(buf, dtype=dt, count=n, offset=484).astype(float)
    wl = start + step * np.arange(n, dtype=float)

    reference = None
    if version >= 2 and len(buf) >= end + 20:
        off = end + 2 + 8 + 8                   # reference flag, reference time, spectrum time
        desc_len = u("H", off)
        off += 2 + desc_len
        if len(buf) >= off + n * size:
            ref = np.frombuffer(buf, dtype=dt, count=n, offset=off).astype(float)
            if np.all(np.isfinite(ref)) and np.nanmax(np.abs(ref)) > 0:
                reference = ref

    if data_type in (0, 1, 2) and reference is not None:   # raw, reflectance and radiance files
        with np.errstate(divide="ignore", invalid="ignore"):
            values = np.where(reference != 0, target / reference, np.nan)
        kind = "reflectance"
    else:
        values = target.copy()
        kind = ASD_DATA_TYPES.get(data_type, "unknown")
        if data_type == 0 and it > 0:           # raw counts: normalise each detector
            i1, i2 = _asd_detector_slices(wl, splice1, splice2)
            values[:i1] /= it
            if swir1_gain:
                values[i1:i2] *= swir1_gain / 2048.0
            if swir2_gain:
                values[i2:] *= swir2_gain / 2048.0
            kind = "raw DN (normalised)"
    applied = bool(splice_correction) and kind == "reflectance"   # never shift raw detector counts
    if applied:
        values = asd_splice_correction(wl, values, splice1, splice2)

    return {
        "name": p.name, "path": str(p), "wavelengths": wl, "values": values, "kind": kind,
        "target": target, "reference": reference,
        "meta": {"file version": version, "data type": ASD_DATA_TYPES.get(data_type, "unknown"),
                 "channels": n, "integration time (ms)": it, "instrument number": instrument_num,
                 "splice wavelengths": (splice1, splice2), "comments": comments,
                 "splice corrected": applied},
    }


# ---------------- GDAL-family rasters ----------------------------------
def _wavelengths_from_band_info(n, band_tags, descs, ds_tags, envi_tags):
    """Wavelengths from GDAL / rasterio band metadata, ENVI-domain tags or band descriptions."""
    vals, units = [], None
    for t in band_tags:
        tl = {str(k).lower(): v for k, v in (t or {}).items()}
        w = tl.get("wavelength", tl.get("center_wavelength", tl.get("centralwavelength")))
        x = _scalar(w)
        if x is None:
            vals = None
            break
        vals.append(x)
        units = units or tl.get("wavelength_units") or tl.get("wavelength units")
    if vals and len(vals) == n:
        return np.asarray(vals, float), units
    for tags in (envi_tags, ds_tags):
        tl = {str(k).lower(): v for k, v in (tags or {}).items()}
        if "wavelength" in tl:
            v = _to_float_vector(str(tl["wavelength"]))
            if v is not None and v.size == n:
                return v, tl.get("wavelength_units") or tl.get("wavelength units")
    if descs and len(descs) == n and all(descs):
        nums = []
        for d in descs:
            found = _NUM_RE.findall(str(d))
            if not found:
                return None, None
            nums.append(float(found[-1]))
        nums = np.asarray(nums)
        if not (np.allclose(nums, np.arange(1, n + 1)) or np.allclose(nums, np.arange(n))):
            joined = " ".join(map(str, descs)).lower()
            u = "nm" if "nm" in joined else ("um" if re.search(r"µm|μm|\bum\b|micro", joined) else None)
            return nums, u
    return None, None


def _raster_result(data_bfirst, nodata, scales, offsets, band_tags, descs, ds_tags, envi_tags, fmt, dataset):
    n = data_bfirst.shape[0]
    cube = _finalize_array(np.moveaxis(data_bfirst, 0, -1), fill=nodata, scale=scales, offset=offsets)
    wl, units = _wavelengths_from_band_info(n, band_tags, descs, ds_tags, envi_tags)
    meta = {}
    for tags in (ds_tags, envi_tags):
        meta.update({str(k): v for k, v in (tags or {}).items()})
    if dataset:
        meta["dataset"] = dataset
    return {"cube": cube, "wavelengths": wl, "wavelength_units": units,
            "metadata": meta, "format": fmt, "dataset": dataset}


def _load_with_rasterio(rio, path, choose):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with rio.open(path) as src:
            subs = list(src.subdatasets or [])
            if src.count == 0 or (subs and src.count < 2):
                if not subs:
                    raise HSILoadError("The file contains no raster bands.")
                order = sorted(range(len(subs)), key=lambda i: -_candidate_score(subs[i]))
                subs = [subs[i] for i in order]
                i = _choose_index(subs, choose, "Choose a subdataset",
                                  "This file contains several datasets. Which one is the hyperspectral cube?")
                return _load_with_rasterio(rio, subs[i], choose)
            if src.count < 2:
                raise HSILoadError("The raster has only one band, so it is not a spectral cube.")
            data = src.read()
            try:
                envi_tags = src.tags(ns="ENVI")
            except Exception:
                envi_tags = {}
            return _raster_result(
                data, src.nodata, src.scales, src.offsets,
                [src.tags(i) for i in range(1, src.count + 1)], list(src.descriptions),
                src.tags(), envi_tags, f"{src.driver} (rasterio)",
                None if Path(str(path)).is_file() else str(path))


def _load_with_gdal(gdal, path, choose):
    gdal.UseExceptions()
    ds = gdal.Open(str(path))
    if ds is None:
        raise HSILoadError("GDAL could not open the file.")
    if ds.RasterCount == 0 or (ds.RasterCount < 2 and ds.GetSubDatasets()):
        subs = ds.GetSubDatasets() or []
        if not subs:
            raise HSILoadError("The file contains no raster bands.")
        subs = sorted(subs, key=lambda s: -_candidate_score(s[0]))
        i = _choose_index([f"{d}" for _, d in subs], choose, "Choose a subdataset",
                          "This file contains several datasets. Which one is the hyperspectral cube?")
        return _load_with_gdal(gdal, subs[i][0], choose)
    n = ds.RasterCount
    if n < 2:
        raise HSILoadError("The raster has only one band, so it is not a spectral cube.")
    bands = [ds.GetRasterBand(i + 1) for i in range(n)]
    data = ds.ReadAsArray()
    scales = [b.GetScale() if b.GetScale() is not None else 1.0 for b in bands]
    offsets = [b.GetOffset() if b.GetOffset() is not None else 0.0 for b in bands]
    return _raster_result(
        data, bands[0].GetNoDataValue(), scales, offsets,
        [b.GetMetadata() or {} for b in bands], [b.GetDescription() for b in bands],
        ds.GetMetadata() or {}, ds.GetMetadata("ENVI") or {},
        f"{ds.GetDriver().ShortName} (GDAL)", None if Path(str(path)).is_file() else str(path))


def _load_raster(p, choose=None, ask_axis=None):
    """GeoTIFF / JPEG 2000 / VRT / NITF / ERDAS / HDF4 / ... via rasterio, GDAL, tifffile or spectral."""
    errors = []
    rio = _try_import("rasterio")
    if rio is not None:
        try:
            return _load_with_rasterio(rio, str(p), choose)
        except HSILoadCancelled:
            raise
        except Exception as e:
            errors.append(f"rasterio: {e}")
    gdal = _try_import("osgeo.gdal")
    if gdal is not None:
        try:
            return _load_with_gdal(gdal, str(p), choose)
        except HSILoadCancelled:
            raise
        except Exception as e:
            errors.append(f"GDAL: {e}")
    if Path(str(p)).suffix.lower() in _TIFF_EXTS:
        tf = _try_import("tifffile")
        if tf is not None:
            try:
                arr = tf.imread(str(p))
                cands = []
                _add_entry(Path(str(p)).name, arr.shape, arr.dtype, cands, [], lambda a=arr: a)
                return _finish_container("TIFF (tifffile)", cands, [], {}, None, choose=choose, ask_axis=ask_axis)
            except HSILoadCancelled:
                raise
            except Exception as e:
                errors.append(f"tifffile: {e}")
    try:
        return _load_spectral_open(Path(str(p)))
    except Exception as e:
        errors.append(f"spectral: {e}")
    hint = ""
    if rio is None and gdal is None:
        hint = ("\n\nNo GDAL-based reader is installed. Install one of:\n"
                "  pip install rasterio\n"
                "  conda install -c conda-forge gdal\n"
                "  pip install tifffile   (TIFF only)")
    raise HSILoadError(f"Could not read '{Path(str(p)).name}' as a hyperspectral cube."
                       + ("\n\n" + "\n".join(errors) if errors else "") + hint)


# ---------------- sidecar XML wavelengths (EnMAP, DESIS, GDAL .aux.xml) -
_XML_WL_RE = re.compile(
    r"<(wavelengthCenterOfBand|centralWavelength|centerWavelength|center_wavelength|wavelength)\b[^>]*>"
    r"\s*([-+]?[0-9.]+(?:[eE][-+]?\d+)?)\s*</\1>", re.I)
_PAM_WL_RE = re.compile(r'<MDI\s+key="wavelength"\s*>\s*([-+]?[0-9.]+(?:[eE][-+]?\d+)?)\s*</MDI>', re.I)


def _wavelengths_from_sidecar_xml(p, n_bands):
    p = Path(p)
    cands = [Path(str(p) + ".aux.xml"), p.with_suffix(".xml"), Path(str(p) + ".xml")]
    try:
        for i, q in enumerate(sorted(p.parent.iterdir())):
            if i > 500:
                break
            if q.suffix.lower() == ".xml" and "metadata" in q.name.lower():
                cands.append(q)
    except Exception:
        pass
    seen = set()
    for q in cands:
        if q in seen or not q.is_file():
            continue
        seen.add(q)
        try:
            if q.stat().st_size > 50_000_000:
                continue
            text = q.read_text(errors="ignore")
            for rx, grp in ((_PAM_WL_RE, 1), (_XML_WL_RE, 2)):
                vals = [float(m.group(grp)) for m in rx.finditer(text)]
                if len(vals) == n_bands:
                    return np.asarray(vals, float)
        except Exception:
            continue
    return None


# ---------------- public entry point -----------------------------------
def load_hyperspectral(path, choose=None, ask_axis=None):
    """Read a hyperspectral cube from any supported file format.

    choose(title, label, items, default) -> index or None   (several datasets in one file)
    ask_axis(shape, guessed_axis) -> axis or None            (spectral axis is ambiguous)
    Both callbacks are optional; without them the best guess is used.

    Returns a dict with: cube (rows, cols, bands float array), wavelengths (nm or None),
    wavelength_units, metadata (dict), format (str), dataset (str or None).
    """
    p = Path(path)
    if not p.exists():
        raise HSILoadError(f"File not found:\n{p}")
    _load_progress(5, f"Opening {p.name}")
    ext = p.suffix.lower()
    zarr_root = _find_zarr_root(p)

    if zarr_root is not None:
        res = _finish_container(*_read_zarr(zarr_root), choose=choose, ask_axis=ask_axis)
    elif p.is_dir():
        raise HSILoadError(f"'{p.name}' is a folder but not a Zarr store. Open a file inside it instead.")
    elif ext == ".zip":
        res = _finish_container(*_read_zarr(p), choose=choose, ask_axis=ask_axis)
    elif ext == ".sav":
        res = _finish_container(*_read_idl_sav(p), choose=choose, ask_axis=ask_axis)
    elif ext == ".asd":
        raise HSILoadError("ASD files hold single point spectra, not image cubes. "
                           "Use File > Import ASD field spectra to add them as reference spectra.")
    elif ext == ".hdr":
        if _is_envi_header(p):
            res = _load_envi_or_fallback(p, choose, ask_axis)
        else:                                   # e.g. ESRI BIL header -> GDAL EHdr driver
            data = find_envi_data_file(p)
            if data is None:
                raise HSILoadError("This .hdr is not an ENVI header and no matching data file was found.")
            res = _load_raster(data, choose, ask_axis)
    elif ext not in _NON_ENVI_EXTS and find_envi_header(p) is not None \
            and _is_envi_header(find_envi_header(p)):
        res = _load_envi_or_fallback(p, choose, ask_axis)
    elif ext == ".mat":
        res = _finish_container(*_read_mat(p), choose=choose, ask_axis=ask_axis)
    elif ext in (".npy", ".npz"):
        res = _finish_container(*_read_numpy(p), choose=choose, ask_axis=ask_axis)
    elif ext in _FITS_EXTS:
        res = _finish_container(*_read_fits(p), choose=choose, ask_axis=ask_axis)
    elif ext in (_HDF5_EXTS | _HDF4_EXTS | _NETCDF_EXTS):
        res = _load_hdf_family(p, choose, ask_axis)
    elif ext in (".lan", ".gis"):
        res = _load_spectral_open(p)
    else:
        h5py = _try_import("h5py")
        if h5py is not None and h5py.is_hdf5(str(p)):
            res = _finish_container(*_read_hdf5(p), choose=choose, ask_axis=ask_axis)
        else:
            res = _load_raster(p, choose, ask_axis)

    _load_progress(75, "Checking cube and wavelengths")
    cube = res["cube"]
    if cube.ndim != 3:
        raise HSILoadError(f"Expected a 3-D cube but got shape {cube.shape}.")
    nb = cube.shape[2]
    wl = res.get("wavelengths")
    if wl is None and res.get("format") != "ENVI":
        wl = _wavelengths_from_sidecar_xml(p, nb)
    if wl is not None:
        wl = normalize_wavelengths(wl, res.get("wavelength_units"))
        if wl is not None and len(wl) != nb:
            wl = None
    res["wavelengths"] = wl
    res["metadata"] = dict(res.get("metadata") or {})
    res["metadata"]["source file"] = str(p)
    res["metadata"]["source format"] = res.get("format")
    return res


def reader_status():
    """(package, formats, installed) for the Help > Supported formats dialog."""
    pkgs = [
        ("spectral", "ENVI (.hdr + data), ERDAS .lan / .gis"),
        ("scipy", "MATLAB .mat v5-v7.2, IDL .sav, NetCDF3"),
        ("zarr", "Zarr stores (.zarr folders, zipped .zarr.zip)"),
        ("h5py", "HDF5 / HDF-EOS5 (PRISMA, DESIS), NetCDF4 (EMIT, PACE), MATLAB v7.3"),
        ("netCDF4", "NetCDF (alternative reader)"),
        ("rasterio", "GeoTIFF, JPEG 2000, VRT, NITF, ERDAS .img, HDF4 and every GDAL format"),
        ("osgeo", "GDAL Python bindings (same formats as rasterio)"),
        ("tifffile", "TIFF (fallback without GDAL)"),
        ("astropy", "FITS"),
        ("pyhdf", "HDF4 (Hyperion L1R)"),
    ]
    return [(name, what, importlib.util.find_spec(name) is not None) for name, what in pkgs]



# ======================================================================
# Preprocessing backend (pure NumPy, no GUI code)
#   normalisation / standardisation, value + wavelength units, common
#   wavelength grid, band ratios / normalised-difference indices,
#   band depth, band gap (absorption edge) and spectral statistics.
# ======================================================================

def _as_float(cube):
    """Float view/copy of an array (float32 stays float32, everything else becomes float64)."""
    a = np.asarray(cube)
    return a if a.dtype.kind == "f" else a.astype(np.float64)


def _nonzero_spread(spread, level, dtype):
    """True where a spread (std / range) is real, not float rounding of a constant signal."""
    tol = 64 * np.finfo(dtype).eps * np.abs(level)
    return np.asarray(spread > np.maximum(tol, np.finfo(dtype).tiny))


def _quiet():
    """Context that silences the expected NaN / divide-by-zero warnings of nan-reductions."""
    class _Q:
        def __enter__(self):
            self._w = warnings.catch_warnings()
            self._w.__enter__()
            warnings.simplefilter("ignore", category=RuntimeWarning)
            self._e = np.errstate(divide="ignore", invalid="ignore", over="ignore")
            self._e.__enter__()

        def __exit__(self, *exc):
            self._e.__exit__(*exc)
            self._w.__exit__(*exc)
    return _Q()


NORMALIZATION_METHODS = [
    ("minmax", "Min-max to range"),
    ("snv", "SNV - standard normal variate (per pixel)"),
    ("zscore", "Z-score standardisation (per band)"),
    ("l2", "Unit vector / L2 norm (per pixel)"),
]
MINMAX_SCOPES = [("band", "Per band"), ("pixel", "Per pixel spectrum"), ("global", "Whole cube")]


def minmax_scale(cube, lo=0.0, hi=1.0, scope="band", robust=False):
    """Rescale values linearly to [lo, hi].

    scope: "band" (each band separately), "pixel" (each spectrum separately) or "global".
    robust: use the 2nd / 98th percentiles instead of min / max and clip to the range.
    Constant bands / spectra are mapped to lo; NaNs stay NaN.
    """
    X = _as_float(cube)
    axes = {"band": (0, 1), "pixel": (2,), "global": None}[scope]
    with _quiet():
        if robust:
            mn = np.nanpercentile(X, 2.0, axis=axes, keepdims=True)
            mx = np.nanpercentile(X, 98.0, axis=axes, keepdims=True)
        else:
            mn = np.nanmin(X, axis=axes, keepdims=True)
            mx = np.nanmax(X, axis=axes, keepdims=True)
        span = np.asarray(mx - mn, dtype=X.dtype)
        real = _nonzero_spread(span, np.maximum(np.abs(mx), np.abs(mn)), X.dtype)
        out = (X - mn) / np.where(real, span, 1).astype(X.dtype)
        out = np.where(real | ~np.isfinite(X), out, 0).astype(X.dtype, copy=False)
        if robust:
            np.clip(out, 0.0, 1.0, out=out)
        out = out * X.dtype.type(hi - lo) + X.dtype.type(lo)
    return out.astype(X.dtype, copy=False)


def snv(cube):
    """Standard normal variate: each pixel spectrum -> (x - mean) / std over its bands."""
    X = _as_float(cube)
    with _quiet():
        mu = np.nanmean(X, axis=2, keepdims=True)
        sd = np.nanstd(X, axis=2, ddof=1, keepdims=True)
        real = _nonzero_spread(sd, mu, X.dtype)
        out = (X - mu) / np.where(real, sd, 1).astype(X.dtype)
        out = np.where(real | ~np.isfinite(X), out, 0)
    return out.astype(X.dtype, copy=False)


def zscore_bands(cube):
    """Standardise each band to zero mean and unit standard deviation over all pixels."""
    X = _as_float(cube)
    with _quiet():
        mu = np.nanmean(X, axis=(0, 1), keepdims=True)
        sd = np.nanstd(X, axis=(0, 1), keepdims=True)
        real = _nonzero_spread(sd, mu, X.dtype)
        out = (X - mu) / np.where(real, sd, 1).astype(X.dtype)
        out = np.where(real | ~np.isfinite(X), out, 0)
    return out.astype(X.dtype, copy=False)


def l2_normalize(cube):
    """Scale each pixel spectrum to unit Euclidean length (removes brightness / illumination)."""
    X = _as_float(cube)
    with _quiet():
        norm = np.sqrt(np.nansum(X * X, axis=2, keepdims=True))
        out = X / np.where(norm > 0, norm, 1).astype(X.dtype)
    return out.astype(X.dtype, copy=False)


def apply_normalization(cube, method, lo=0.0, hi=1.0, scope="band", robust=False):
    if method == "minmax":
        return minmax_scale(cube, lo, hi, scope, robust)
    if method == "snv":
        return snv(cube)
    if method == "zscore":
        return zscore_bands(cube)
    if method == "l2":
        return l2_normalize(cube)
    raise ValueError(f"Unknown normalisation method: {method}")


# (label, factor, offset); None = custom factor / offset typed by the user
VALUE_UNIT_CONVERSIONS = [
    ("Reflectance scaled x10000  ->  0-1", 1e-4, 0.0),
    ("Reflectance %  (0-100)  ->  0-1", 1e-2, 0.0),
    ("Reflectance 0-1  ->  %  (0-100)", 100.0, 0.0),
    ("Reflectance 0-1  ->  scaled x10000", 1e4, 0.0),
    ("Radiance uW/(cm2 sr nm)  ->  W/(m2 sr um)", 10.0, 0.0),
    ("Radiance W/(m2 sr um)  ->  uW/(cm2 sr nm)", 0.1, 0.0),
    ("Radiance W/(m2 sr nm)  ->  W/(m2 sr um)", 1000.0, 0.0),
    ("Radiance W/(m2 sr um)  ->  W/(m2 sr nm)", 1e-3, 0.0),
    ("Custom:  value x factor + offset", None, None),
]


def scale_values(cube, factor=1.0, offset=0.0):
    X = _as_float(cube)
    return (X * X.dtype.type(factor) + X.dtype.type(offset)).astype(X.dtype, copy=False)


WAVELENGTH_UNITS = [("nm", "Nanometres (nm)"), ("um", "Micrometres (um)"), ("A", "Angstrom (A)"),
                    ("cm-1", "Wavenumber (cm-1)"), ("m", "Metres (m)")]


def wavelengths_to_nm(wl, unit):
    """Convert a wavelength / wavenumber vector to nanometres."""
    wl = np.asarray(wl, dtype=float)
    if unit == "nm":
        return wl.copy()
    if unit == "um":
        return wl * 1e3
    if unit == "A":
        return wl / 10.0
    if unit == "m":
        return wl * 1e9
    if unit == "cm-1":
        with _quiet():
            return np.where(wl > 0, 1e7 / wl, np.nan)
    raise ValueError(f"Unknown wavelength unit: {unit}")


def sort_bands(cube, wl):
    """Reorder bands so wavelengths increase (needed after a wavenumber -> nm conversion)."""
    wl = np.asarray(wl, dtype=float)
    order = np.argsort(wl, kind="stable")
    if np.all(order == np.arange(len(order))):
        return cube, wl
    return np.ascontiguousarray(np.asarray(cube)[:, :, order]), wl[order]


def uniform_grid(start, stop, step):
    if step <= 0:
        raise ValueError("The grid step must be positive.")
    if stop <= start:
        raise ValueError("The grid end must be larger than the start.")
    n = int(np.floor((stop - start) / step + 1e-9)) + 1
    if n < 2:
        raise ValueError("The grid needs at least two wavelengths; use a smaller step.")
    return start + step * np.arange(n, dtype=float)


def read_wavelength_grid(path):
    """Target wavelengths (nm) and optional FWHM (nm) from an ENVI .hdr or a text / CSV file.

    Text format: one band per line - wavelength, or wavelength and FWHM, or band number,
    wavelength and FWHM. A single line with a comma / space separated list also works.
    Header lines without numbers and lines starting with # are ignored.
    """
    p = Path(path)
    wl = fwhm = units = None
    if p.suffix.lower() == ".hdr":
        h = spectral.envi.read_envi_header(str(p))
        if h.get("wavelength") is not None:
            wl = _to_float_vector(h["wavelength"])
        if h.get("fwhm") is not None:
            fwhm = _to_float_vector(h["fwhm"])
        units = h.get("wavelength units")
    else:
        rows_ = []
        for line in p.read_text(errors="ignore").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            nums = [float(t) for t in _NUM_RE.findall(line)]
            if nums:
                rows_.append(nums)
        if len(rows_) == 1:
            wl = np.asarray(rows_[0], dtype=float)
        elif rows_:
            ncol = min(len(r) for r in rows_)
            cols_ = [np.asarray([r[c] for r in rows_], dtype=float) for c in range(ncol)]
            c0 = cols_[0]
            looks_like_index = (ncol >= 2 and np.allclose(c0, np.round(c0))
                                and np.allclose(np.diff(c0), 1))
            if looks_like_index:
                cols_ = cols_[1:]
            wl = cols_[0]
            if len(cols_) >= 2:
                fwhm = cols_[1]
    if wl is None or wl.size < 2:
        raise HSILoadError(f"No wavelength list with at least two values was found in {p.name}.")
    wl = np.asarray(wl, dtype=float)
    wl_nm = normalize_wavelengths(wl, units)
    if fwhm is not None:
        fwhm = np.asarray(fwhm, dtype=float)
        factor = wl_nm[0] / wl[0] if wl[0] != 0 else 1.0
        fwhm = fwhm * factor if fwhm.size == wl.size else None
    order = np.argsort(wl_nm)
    return wl_nm[order], (fwhm[order] if fwhm is not None else None)


GRID_METHODS = [("linear", "Linear interpolation"),
                ("gaussian", "Gaussian SRF convolution (FWHM)"),
                ("nearest", "Nearest band")]


def resample_spectral(cube, wl, target, method="linear", target_fwhm=None, chunk=50000):
    """Resample every pixel spectrum from wavelengths wl onto the target grid.

    linear / nearest: interpolation between neighbouring source bands.
    gaussian: each target band is a Gaussian-weighted average of the source bands
      (spectral response function with the given FWHM; default = local grid spacing).
      Use it when going to a coarser sensor; it never uses a FWHM below the source spacing.
    Target wavelengths outside the source range become NaN. NaN source values are ignored
    by the Gaussian method and propagate with the interpolation methods.
    """
    X = _as_float(cube)
    rows, cols, nb = X.shape
    wl = np.asarray(wl, dtype=float)
    target = np.asarray(target, dtype=float)
    if wl.size != nb:
        raise ValueError("Wavelength count does not match the number of bands.")
    good = np.isfinite(wl)
    order = np.argsort(np.where(good, wl, np.inf))[: int(good.sum())]
    src_wl, first = np.unique(wl[order], return_index=True)
    src_idx = order[first]
    if src_wl.size < 2:
        raise ValueError("At least two distinct finite source wavelengths are needed.")
    inside = (target >= src_wl[0] - 1e-9) & (target <= src_wl[-1] + 1e-9)
    flat = X.reshape(-1, nb)
    out = np.empty((flat.shape[0], target.size), dtype=X.dtype)

    if method in ("linear", "nearest"):
        j1 = np.clip(np.searchsorted(src_wl, target), 1, src_wl.size - 1)
        j0 = j1 - 1
        t = np.clip((target - src_wl[j0]) / (src_wl[j1] - src_wl[j0]), 0.0, 1.0)
        if method == "nearest":
            t = (t >= 0.5).astype(float)
        t = t.astype(X.dtype)
        for a in range(0, flat.shape[0], chunk):
            A = flat[a:a + chunk][:, src_idx]
            out[a:a + chunk] = A[:, j0] * (1 - t) + A[:, j1] * t
    elif method == "gaussian":
        if target_fwhm is not None and np.size(target_fwhm) == target.size:
            fwhm = np.asarray(target_fwhm, dtype=float).copy()
        elif target_fwhm is not None and np.size(target_fwhm) == 1 and float(np.ravel(target_fwhm)[0]) > 0:
            fwhm = np.full(target.size, float(np.ravel(target_fwhm)[0]))
        else:
            fwhm = np.abs(np.gradient(target)) if target.size > 1 else np.ones(1)
        src_spacing = np.interp(target, src_wl, np.abs(np.gradient(src_wl)))
        fwhm = np.maximum(np.where(np.isfinite(fwhm) & (fwhm > 0), fwhm, src_spacing), src_spacing)
        sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        W = np.exp(-0.5 * ((src_wl[:, None] - target[None, :]) / sigma[None, :]) ** 2)
        W[W < 1e-6] = 0.0
        W = W.astype(X.dtype)
        wsum = W.sum(axis=0)
        for a in range(0, flat.shape[0], chunk):
            A = flat[a:a + chunk][:, src_idx]
            M = np.isfinite(A)
            num = np.where(M, A, 0) @ W
            den = M.astype(X.dtype) @ W
            with _quiet():
                out[a:a + chunk] = np.where(den > 0.5 * wsum, num / den, np.nan)
    else:
        raise ValueError(f"Unknown resampling method: {method}")
    out[:, ~inside] = np.nan
    return out.reshape(rows, cols, target.size)


SPECTRAL_FEATURE_KINDS = [
    ("nd", "Normalised difference  (L1 - L2) / (L1 + L2)"),
    ("ratio", "Band ratio  L1 / L2"),
    ("diff", "Band difference  L1 - L2"),
    ("depth", "Band depth (continuum removed)"),
    ("gap", "Band gap / absorption edge"),
]

# (label, kind, wavelengths in nm)
SPECTRAL_FEATURE_PRESETS = [
    ("Custom", None, ()),
    ("NDVI - vegetation (860, 650)", "nd", (860, 650)),
    ("NDRE - red edge (790, 720)", "nd", (790, 720)),
    ("NDWI Gao - leaf water (860, 1240)", "nd", (860, 1240)),
    ("NDWI McFeeters - open water (560, 860)", "nd", (560, 860)),
    ("MNDWI - water (560, 1650)", "nd", (560, 1650)),
    ("NDBI - built-up (1650, 860)", "nd", (1650, 860)),
    ("NDII - canopy water (860, 1650)", "nd", (860, 1650)),
    ("NBR - burn ratio (860, 2200)", "nd", (860, 2200)),
    ("PRI - photochemical (531, 570)", "nd", (531, 570)),
    ("Simple ratio SR (860 / 650)", "ratio", (860, 650)),
    ("Iron oxide ratio (650 / 550)", "ratio", (650, 550)),
    ("Chlorophyll depth 670 (550, 670, 760)", "depth", (550, 670, 760)),
    ("Water depth 970 (890, 970, 1070)", "depth", (890, 970, 1070)),
    ("Water depth 1200 (1080, 1200, 1270)", "depth", (1080, 1200, 1270)),
    ("Al-OH clay depth 2200 (2120, 2200, 2250)", "depth", (2120, 2200, 2250)),
    ("Carbonate depth 2340 (2270, 2340, 2390)", "depth", (2270, 2340, 2390)),
    ("Red-edge position (680 - 750)", "gap", (680, 750)),
    ("Fe-oxide absorption edge / band gap (480 - 650)", "gap", (480, 650)),
]


def nearest_band(wl, value):
    """Index of the band nearest to a wavelength (or the band index itself when wl is None)."""
    if wl is None:
        return int(np.clip(round(float(value)), 0, 10 ** 9))
    wl = np.asarray(wl, dtype=float)
    return int(np.nanargmin(np.abs(wl - float(value))))


def band_ratio(a, b):
    with _quiet():
        r = np.asarray(a, dtype=np.float64) / np.asarray(b, dtype=np.float64)
    r[~np.isfinite(r)] = np.nan
    return r


def normalized_difference(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    with _quiet():
        r = (a - b) / (a + b)
    r[~np.isfinite(r)] = np.nan
    return r


def band_depth(cube, wl, left, center, right, deepest=False):
    """Continuum-removed band depth of an absorption feature.

    The continuum is the straight line between the left and right shoulder bands.
    depth = 1 - R / R_continuum, at the centre band, or (deepest=True) at the deepest
    point between the shoulders. Also returns the wavelength of the deepest point.
    """
    X = np.asarray(cube)
    wl = np.asarray(wl, dtype=float)
    iL, iC, iR = (nearest_band(wl, v) for v in (left, center, right))
    wL, wC, wR = wl[iL], wl[iC], wl[iR]
    if not (wL < wC < wR):
        raise ValueError(f"Need left shoulder < centre < right shoulder. The nearest bands are "
                         f"{wL:g}, {wC:g} and {wR:g}.")
    RL = X[:, :, iL].astype(np.float64)
    RR = X[:, :, iR].astype(np.float64)
    win = np.where((wl >= wL) & (wl <= wR))[0]
    win = win[np.argsort(wl[win])]
    min_cr = np.full(RL.shape, np.nan)
    pos = np.full(RL.shape, np.nan)
    with _quiet():
        for j in win:
            cont = RL + (RR - RL) * (wl[j] - wL) / (wR - wL)
            cr = X[:, :, j].astype(np.float64) / cont
            better = np.isfinite(cr) & ~(cr >= min_cr)
            min_cr = np.where(better, cr, min_cr)
            pos = np.where(better, wl[j], pos)
        if deepest:
            depth = 1.0 - min_cr
        else:
            cont_c = RL + (RR - RL) * (wC - wL) / (wR - wL)
            depth = 1.0 - X[:, :, iC].astype(np.float64) / cont_c
    depth[~np.isfinite(depth)] = np.nan
    return depth, pos, (wL, wC, wR)


def absorption_edge(cube, wl, lo, hi):
    """Absorption-edge (inflection) wavelength and the equivalent band-gap energy.

    Within [lo, hi] the edge is where reflectance rises fastest (maximum dR/dlambda),
    refined to sub-band precision with a parabola. Band gap Eg (eV) = 1239.84 / lambda(nm).
    For vegetation between ~680 and 750 nm the edge wavelength is the red-edge position.
    """
    X = np.asarray(cube)
    wl = np.asarray(wl, dtype=float)
    a, b = min(lo, hi), max(lo, hi)
    sel = np.where((wl >= a) & (wl <= b))[0]
    sel = sel[np.argsort(wl[sel])]
    if sel.size < 3:
        raise ValueError(f"At least three bands are needed between {a:g} and {b:g}; found {sel.size}.")
    w = wl[sel]
    R = X[:, :, sel].astype(np.float64)
    with _quiet():
        d = np.gradient(R, w, axis=2)
    d_f = np.where(np.isfinite(d), d, -np.inf)
    k = np.argmax(d_f, axis=2)
    dmax = np.take_along_axis(d_f, k[..., None], axis=2)[..., 0]
    k0 = np.clip(k, 1, w.size - 2)
    y0 = np.take_along_axis(d_f, (k0 - 1)[..., None], axis=2)[..., 0]
    y1 = np.take_along_axis(d_f, k0[..., None], axis=2)[..., 0]
    y2 = np.take_along_axis(d_f, (k0 + 1)[..., None], axis=2)[..., 0]
    with _quiet():
        denom = y0 - 2.0 * y1 + y2
        delta = np.where((k == k0) & np.isfinite(denom) & (denom < 0), 0.5 * (y0 - y2) / denom, 0.0)
    delta = np.clip(np.nan_to_num(delta), -0.5, 0.5)
    pos = w[k] + delta * (w[k0 + 1] - w[k0 - 1]) / 2.0
    pos[~np.isfinite(dmax) | (dmax <= 0)] = np.nan
    with _quiet():
        eg = 1239.84198 / pos
    return eg, pos, (w[0], w[-1])


PIXEL_STATISTICS = [("mean", "Mean"), ("std", "Standard deviation"), ("range", "Range (max - min)"),
                    ("min", "Minimum"), ("max", "Maximum"), ("median", "Median"),
                    ("cv", "Coefficient of variation (std / mean)")]


def pixel_statistic_map(cube, stat):
    """One statistic of every pixel spectrum (over the bands) as a 2-D image."""
    X = np.asarray(cube)
    with _quiet():
        if stat == "mean":
            return np.nanmean(X, axis=2).astype(np.float64)
        if stat == "std":
            return np.nanstd(X, axis=2, ddof=1).astype(np.float64)
        if stat == "range":
            return (np.nanmax(X, axis=2) - np.nanmin(X, axis=2)).astype(np.float64)
        if stat == "min":
            return np.nanmin(X, axis=2).astype(np.float64)
        if stat == "max":
            return np.nanmax(X, axis=2).astype(np.float64)
        if stat == "median":
            return np.nanmedian(X, axis=2).astype(np.float64)
        if stat == "cv":
            m = np.nanmean(X, axis=2).astype(np.float64)
            r = np.nanstd(X, axis=2, ddof=1) / m
            r[~np.isfinite(r)] = np.nan
            return r
    raise ValueError(f"Unknown statistic: {stat}")


def band_statistics(spectra, median_max=500000, seed=0):
    """Per-band statistics of a set of spectra (N pixels x bands).

    Returns mean, std (sample), min, max, range, median and the count of valid values.
    The median is computed on a random subset when N is larger than median_max.
    """
    S = np.asarray(spectra)
    n_px = S.shape[0]
    with _quiet():
        out = {
            "mean": np.nanmean(S, axis=0).astype(np.float64),
            "std": np.nanstd(S, axis=0, ddof=1 if n_px > 1 else 0).astype(np.float64),
            "min": np.nanmin(S, axis=0).astype(np.float64),
            "max": np.nanmax(S, axis=0).astype(np.float64),
        }
        sub = S
        if n_px > median_max:
            sub = S[np.random.default_rng(seed).choice(n_px, median_max, replace=False)]
        out["median"] = np.nanmedian(sub, axis=0).astype(np.float64)
    out["range"] = out["max"] - out["min"]
    out["n"] = np.isfinite(S).sum(axis=0)
    out["median_sampled"] = n_px > median_max
    return out


# ======================================================================
# Analysis helpers (pure NumPy / sklearn, no GUI code)
# ======================================================================

def class_colors(n):
    """Return n distinct RGB colours (tab10 for <=10 classes, tab20 beyond)."""
    cmap = matplotlib.colormaps["tab10" if n <= 10 else "tab20"]
    return [tuple(cmap(i % cmap.N)[:3]) for i in range(n)]


def prepare_features(cube_2d):
    """Flatten-ready (N, bands) matrix -> float32 copy with NaNs filled by band means.

    Returns (X, valid_px) where valid_px marks pixels that are not entirely NaN.
    """
    X = np.array(cube_2d, dtype=np.float32, copy=True)
    valid_px = ~np.all(np.isnan(X), axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        means = np.nanmean(X, axis=0)
    means = np.where(np.isnan(means), 0.0, means).astype(np.float32)
    nan_mask = np.isnan(X)
    if nan_mask.any():
        X[nan_mask] = np.broadcast_to(means, X.shape)[nan_mask]
    return X, valid_px


def polygon_mask(vertices, rows, cols):
    """Boolean (rows, cols) mask of pixel centres inside a polygon given as [(x, y), ...]."""
    verts = np.asarray(vertices, dtype=float)
    mask = np.zeros((rows, cols), dtype=bool)
    if verts.shape[0] < 3:
        return mask
    x0 = max(0, int(np.floor(verts[:, 0].min())))
    x1 = min(cols - 1, int(np.ceil(verts[:, 0].max())))
    y0 = max(0, int(np.floor(verts[:, 1].min())))
    y1 = min(rows - 1, int(np.ceil(verts[:, 1].max())))
    if x1 < x0 or y1 < y0:
        return mask
    xx, yy = np.meshgrid(np.arange(x0, x1 + 1), np.arange(y0, y1 + 1))
    pts = np.column_stack([xx.ravel(), yy.ravel()])
    inside = MplPath(verts).contains_points(pts)
    mask[y0:y1 + 1, x0:x1 + 1] = inside.reshape(yy.shape)
    return mask


def rect_mask(ymin, ymax, xmin, xmax, rows, cols):
    mask = np.zeros((rows, cols), dtype=bool)
    mask[ymin:ymax + 1, xmin:xmax + 1] = True
    return mask


def spectral_angle_map(X, E, chunk=200000):
    """Spectral Angle Mapper.

    X: (N, bands) pixel spectra, E: (M, bands) reference spectra.
    Returns (best_idx, best_angle_rad). Pixels with ~zero norm get idx -1 and NaN angle.
    """
    E = np.asarray(E, dtype=np.float32)
    En = E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-12)
    N = X.shape[0]
    best_idx = np.empty(N, dtype=np.int32)
    best_ang = np.empty(N, dtype=np.float32)
    for s in range(0, N, chunk):
        xb = np.asarray(X[s:s + chunk], dtype=np.float32)
        norms = np.linalg.norm(xb, axis=1)
        cos = (xb @ En.T) / np.maximum(norms, 1e-12)[:, None]
        np.clip(cos, -1.0, 1.0, out=cos)
        ang = np.arccos(cos)
        j = np.argmin(ang, axis=1)
        a = ang[np.arange(len(j)), j]
        zero = norms <= 1e-12
        a[zero] = np.nan
        j[zero] = -1
        best_idx[s:s + chunk] = j
        best_ang[s:s + chunk] = a
    return best_idx, best_ang


def build_label_images(rois, class_names, rows, cols):
    """Rasterise ROIs into a class-index image and a ROI-id image.

    Pixels covered by ROIs of *different* classes are ambiguous and flagged in
    the returned conflict mask (they are excluded from training).
    """
    cls_img = -np.ones((rows, cols), dtype=np.int32)
    roi_img = -np.ones((rows, cols), dtype=np.int32)
    conflict = np.zeros((rows, cols), dtype=bool)
    for roi_i, roi in enumerate(rois):
        ci = class_names.index(roi["class"])
        m = roi["mask"]
        conflict |= m & (cls_img != -1) & (cls_img != ci)
        cls_img[m] = ci
        roi_img[m] = roi_i
    return cls_img, roi_img, conflict


def split_train_test(y, roi_id, test_frac, mode, rng):
    """Return boolean test-mask over samples.

    mode "pixels": stratified random pixel split.
    mode "rois":   hold out whole ROIs per class (needs >=2 ROIs of a class,
                   otherwise that class falls back to a pixel split).
    """
    test = np.zeros(len(y), dtype=bool)
    if test_frac <= 0:
        return test
    for c in np.unique(y):
        in_c = np.flatnonzero(y == c)
        used_roi_split = False
        if mode == "rois":
            rois_c = np.unique(roi_id[in_c])
            if len(rois_c) >= 2:
                k = int(round(len(rois_c) * test_frac))
                k = min(max(k, 1), len(rois_c) - 1)
                chosen = rng.choice(rois_c, size=k, replace=False)
                test[in_c[np.isin(roi_id[in_c], chosen)]] = True
                used_roi_split = True
        if not used_roi_split:
            if len(in_c) < 2:
                continue
            k = int(round(len(in_c) * test_frac))
            k = min(max(k, 1), len(in_c) - 1)
            test[rng.choice(in_c, size=k, replace=False)] = True
    return test


def cap_per_class(idx, y, max_per_class, rng):
    """Randomly keep at most max_per_class samples of each class among idx."""
    if max_per_class <= 0:
        return idx
    keep = []
    for c in np.unique(y[idx]):
        ic = idx[y[idx] == c]
        if len(ic) > max_per_class:
            ic = rng.choice(ic, size=max_per_class, replace=False)
        keep.append(ic)
    return np.sort(np.concatenate(keep)) if keep else idx


# Registry of supervised classifiers. To plug in another model (e.g. a CNN
# wrapper) add an entry that returns an object with fit(X, y) / predict(X).
def _make_random_forest():
    return RandomForestClassifier(
        n_estimators=200, n_jobs=-1, random_state=0, class_weight="balanced"
    )


def _make_svm_rbf():
    return Pipeline([
        ("scale", StandardScaler()),
        ("svc", SVC(kernel="rbf", C=10.0, gamma="scale", class_weight="balanced")),
    ])


class PcaSvm:
    """PCA (up to 100 components) followed by an SVM, as in the reference SVM_HSIn.py."""

    def __init__(self, n_components=100, kernel="poly"):
        self.n_components = n_components
        self.kernel = kernel

    def fit(self, X, y):
        n = int(max(1, min(self.n_components, X.shape[1], X.shape[0] - 1)))
        self.pca = PCA(n_components=n, random_state=0)
        Xp = self.pca.fit_transform(X)
        self.svc = SVC(kernel=self.kernel, class_weight="balanced")
        self.svc.fit(Xp, y)
        return self

    def predict(self, X):
        return self.svc.predict(self.pca.transform(X))


def _make_pca_svm_poly():
    return PcaSvm(n_components=100, kernel="poly")


CLASSIFIERS = {
    "Random forest": _make_random_forest,
    "SVM (RBF kernel)": _make_svm_rbf,
    "SVM (PCA + poly kernel)": _make_pca_svm_poly,
}


def format_report(names, y_true, y_pred, n_train, n_test, split_desc):
    labels = list(range(len(names)))
    present = [i for i in labels if i in set(y_true.tolist()) | set(y_pred.tolist())]
    lines = [
        f"Split: {split_desc}",
        (f"Training samples: {n_train}    Test samples: {n_test}" if n_train is not None
         else f"Pooled test predictions: {n_test}"),
        "",
        f"Overall accuracy: {100.0 * accuracy_score(y_true, y_pred):.2f}%",
        f"Cohen's kappa:    {cohen_kappa_score(y_true, y_pred):.4f}",
        "",
        "Per-class precision / recall / F1 / support:",
        classification_report(
            y_true, y_pred, labels=present,
            target_names=[names[i] for i in present],
            digits=3, zero_division=0,
        ),
        "Confusion matrix (rows = true, columns = predicted):",
    ]
    cm = confusion_matrix(y_true, y_pred, labels=present)
    w = max(6, max(len(names[i]) for i in present) + 2)
    lines.append(" " * w + "".join(f"{names[i][:w - 1]:>{w}}" for i in present))
    for r, i in enumerate(present):
        lines.append(f"{names[i][:w - 1]:<{w}}" + "".join(f"{v:>{w}}" for v in cm[r]))
    return "\n".join(lines)


class TaskCancelled(Exception):
    pass


def _check_cancel(cancel):
    if cancel is not None and cancel.is_set():
        raise TaskCancelled()


def _get_features(cube, feature_source, pca_scores, pca_valid):
    """(N, F) float32 feature matrix and the mask of valid pixels."""
    rows, cols, bands = cube.shape
    if feature_source == "pca":
        k = pca_scores.shape[2]
        return pca_scores.reshape(-1, k).astype(np.float32), pca_valid.copy()
    return prepare_features(cube.reshape(-1, bands))


def _collect_samples(valid_px, rois, class_names, rows, cols):
    """Labelled pixels from the ROIs.

    Returns (flat pixel indices, class ids 0..K-1, ROI ids, class names used,
    number of ambiguous pixels that were dropped).
    """
    cls_img, roi_img, conflict = build_label_images(rois, class_names, rows, cols)
    ok = (cls_img.ravel() >= 0) & ~conflict.ravel() & valid_px
    samp_idx = np.flatnonzero(ok)
    y_all = cls_img.ravel()[samp_idx]
    rid = roi_img.ravel()[samp_idx]
    present = np.unique(y_all)
    if len(present) < 2:
        raise RuntimeError("Need labelled pixels from at least 2 different classes.")
    remap = -np.ones(len(class_names), dtype=int)
    remap[present] = np.arange(len(present))
    used_names = [class_names[i] for i in present]
    return samp_idx, remap[y_all], rid, used_names, int(conflict.sum())


def _split_notes(report, n_conflict, split_mode, has_test):
    if n_conflict:
        report += (f"\n\nNote: {n_conflict} pixels lie in ROIs of different classes "
                   "and were excluded from training.")
    if split_mode == "pixels" and has_test:
        report += ("\n\nNote: neighbouring pixels are highly correlated, so a random pixel split "
                   "tends to over-estimate accuracy. 'Hold out whole ROIs' is a stricter test.")
    return report


def run_supervised(cube, rois, class_names, feature_source, pca_scores, pca_valid,
                   clf_name, test_frac, split_mode, max_per_class,
                   progress=None, cancel=None):
    """Train a classifier on labelled ROI pixels and classify the whole image.

    Runs in a worker thread (no GUI calls). Returns a dict with the label image,
    class names that were used, and a text report.
    """
    def say(msg):
        if progress is not None:
            progress(msg)

    rows, cols, bands = cube.shape
    rng = np.random.default_rng(0)

    say("Preparing features...")
    X_all, valid_px = _get_features(cube, feature_source, pca_scores, pca_valid)
    samp_idx, y, rid_all, used_names, n_conflict = _collect_samples(
        valid_px, rois, class_names, rows, cols)
    _check_cancel(cancel)

    test_mask = split_train_test(y, rid_all, test_frac, split_mode, rng)
    sample_ids = np.arange(len(y))
    train_ids = cap_per_class(sample_ids[~test_mask], y, max_per_class, rng)
    test_ids = cap_per_class(sample_ids[test_mask], y, max_per_class, rng) \
        if test_mask.any() else np.array([], dtype=int)

    if len(np.unique(y[train_ids])) < 2:
        raise RuntimeError("Training set contains fewer than 2 classes after splitting.")

    say(f"Training {clf_name} on {len(train_ids)} pixels...")
    clf = CLASSIFIERS[clf_name]()
    clf.fit(X_all[samp_idx[train_ids]], y[train_ids])
    _check_cancel(cancel)

    split_desc = ("hold-out whole ROIs" if split_mode == "rois" else "random pixels")
    if len(test_ids) > 0:
        say("Evaluating on held-out samples...")
        y_pred = clf.predict(X_all[samp_idx[test_ids]])
        report = format_report(used_names, y[test_ids], y_pred,
                               len(train_ids), len(test_ids), split_desc)
    else:
        report = (f"No hold-out set (test fraction = 0). Trained on {len(train_ids)} pixels.\n"
                  "Set a test fraction above 0 to get accuracy figures.")
    report = _split_notes(report, n_conflict, split_mode, len(test_ids) > 0)

    labels_flat = -np.ones(rows * cols, dtype=np.int32)
    valid_idx = np.flatnonzero(valid_px)
    chunk = 100000
    for s0 in range(0, len(valid_idx), chunk):
        _check_cancel(cancel)
        say(f"Classifying image... {min(s0 + chunk, len(valid_idx))}/{len(valid_idx)} pixels")
        sel = valid_idx[s0:s0 + chunk]
        labels_flat[sel] = clf.predict(X_all[sel])

    return {
        "labels": labels_flat.reshape(rows, cols),
        "names": used_names,
        "report": report,
        "title": f"{clf_name}",
    }


def run_crossval(cube, rois, class_names, feature_source, pca_scores, pca_valid,
                 clf_name, n_folds, split_mode, max_per_class,
                 progress=None, cancel=None):
    """k-fold cross-validation on the labelled ROI pixels (as in the reference SVM script).

    split_mode "pixels": stratified random folds. "rois": whole ROIs are held out per fold.
    """
    def say(msg):
        if progress is not None:
            progress(msg)

    rows, cols, bands = cube.shape
    rng = np.random.default_rng(0)
    say("Preparing features...")
    X_all, valid_px = _get_features(cube, feature_source, pca_scores, pca_valid)
    samp_idx, y, rid, used_names, n_conflict = _collect_samples(
        valid_px, rois, class_names, rows, cols)

    ids = np.arange(len(y))
    if split_mode == "rois":
        rois_per_class = [len(np.unique(rid[y == c])) for c in np.unique(y)]
        k = min(n_folds, min(rois_per_class))
        if k < 2:
            raise RuntimeError(
                "Hold-out by ROI needs at least 2 ROIs in every class. "
                "Add more ROIs or switch to random pixels."
            )
        if StratifiedGroupKFold is not None:
            splitter = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=0)
        else:
            splitter = GroupKFold(n_splits=k)
        splits = list(splitter.split(ids, y, rid))
        desc = f"{k}-fold cross-validation, whole ROIs held out"
    else:
        k = min(n_folds, int(np.bincount(y).min()))
        if k < 2:
            raise RuntimeError("Each class needs at least 2 labelled pixels for cross-validation.")
        splits = list(StratifiedKFold(n_splits=k, shuffle=True, random_state=0).split(ids, y))
        desc = f"{k}-fold stratified cross-validation, random pixels"

    fold_acc, y_true, y_pred = [], [], []
    for f, (tr, te) in enumerate(splits):
        _check_cancel(cancel)
        say(f"Cross-validation fold {f + 1}/{k} ({clf_name})...")
        tr = cap_per_class(tr, y, max_per_class, rng)
        te = cap_per_class(te, y, max_per_class, rng)
        if len(np.unique(y[tr])) < 2 or len(te) == 0 or not set(y[te]) <= set(y[tr]):
            fold_acc.append(float("nan"))      # a test class never seen in training
            continue
        clf = CLASSIFIERS[clf_name]()
        clf.fit(X_all[samp_idx[tr]], y[tr])
        pred = clf.predict(X_all[samp_idx[te]])
        fold_acc.append(float(accuracy_score(y[te], pred)))
        y_true.append(y[te])
        y_pred.append(pred)

    if not y_true:
        raise RuntimeError("No fold could be evaluated (each training fold needs 2+ classes).")
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    accs = np.array(fold_acc)
    good = ~np.isnan(accs)

    lines = [f"Classifier: {clf_name}", f"{desc}", ""]
    for f, a in enumerate(accs):
        lines.append(f"  Fold {f + 1}: " + (
            "skipped (a test class had no training pixels)" if np.isnan(a) else f"{100.0 * a:.2f}%"))
    lines.append(f"Average accuracy: {100.0 * accs[good].mean():.2f}%  "
                 f"(+/- {100.0 * accs[good].std():.2f}%)")
    lines.append("")
    report = "\n".join(lines) + format_report(used_names, y_true, y_pred, None, len(y_true), desc)
    report = _split_notes(report, n_conflict, split_mode, False)
    return {"report": report}


# ----------------------------------------------------------------------
# Signal-to-noise ratio (reference: SNR.py)
# ----------------------------------------------------------------------
def compute_snr(cube, pixel_mask=None):
    """Spatial-domain SNR per band and spectral-domain SNR per pixel.

    band SNR  = mean / std of each band over the image (or over pixel_mask)
    pixel SNR = mean / std of each pixel's spectrum across all bands
    Zero standard deviation gives 0, as in the reference script.
    """
    rows, cols, bands = cube.shape
    flat = cube.reshape(-1, bands)
    sel = flat if pixel_mask is None else flat[np.asarray(pixel_mask).ravel()]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        bm, bs = np.nanmean(sel, axis=0), np.nanstd(sel, axis=0)
        pm, ps = np.nanmean(flat, axis=1), np.nanstd(flat, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        band_snr = np.where(bs > 0, bm / bs, 0.0)
        pixel_snr = np.where(ps > 0, pm / ps, 0.0)
    pixel_snr = np.where(np.isnan(pm), np.nan, pixel_snr)   # fully invalid pixels stay NaN
    return band_snr.astype(float), pixel_snr.reshape(rows, cols).astype(float)


# ----------------------------------------------------------------------
# Noise reduction: NGmeet (non-local meets global) denoising
# ----------------------------------------------------------------------
def nlm_filter_2d(img, sigma, patch=5, dist=6):
    """Non-local-means filter for one eigen-image (needs scikit-image).

    The image is rescaled to [0, 1] and sigma is rescaled with it, so the filter strength
    is the same as filtering the original values with `sigma`.
    """
    from skimage.restoration import denoise_nl_means
    img = np.asarray(img, dtype=np.float32)
    lo, hi = float(img.min()), float(img.max())
    scale = max(hi - lo, 1e-12)
    u = (img - lo) / scale
    s = float(sigma) / scale
    out = denoise_nl_means(u, patch_size=patch, patch_distance=dist,
                           h=0.8 * s, sigma=s, fast_mode=True, channel_axis=None)
    return (out * scale + lo).astype(np.float32)


def estimate_subspace_rank(eigvals, n_bands, max_rank=30):
    """Heuristic signal-subspace size from the band-covariance eigenvalues (descending).

    The noise variance is taken as the median of the smaller half of the eigenvalues;
    components clearly above it are kept. The user can override the result in the GUI.
    """
    w = np.clip(np.asarray(eigvals, dtype=float), 0.0, None)
    noise_var = float(np.median(w[len(w) // 2:])) if len(w) > 1 else 0.0
    k = int(np.sum(w > 4.0 * max(noise_var, 1e-12)))
    return int(np.clip(k, min(3, n_bands), min(max_rank, n_bands)))


def ngmeet_denoise(cube, sigma=0.01, n_iter=2, lam=0.9, rank=None,
                   progress=None, cancel=None):
    """NGmeet-style hyperspectral denoising. cube: (rows, cols, bands), normalised to [0, 1].

    Per iteration:
      1. spectral low-rank approximation of Y (eigen-decomposition of the band covariance)
         -> reduced cube M (k eigen-images) and orthogonal basis A
      2. non-local spatial filtering of every eigen-image (smoothing parameter `sigma`)
      3. inverse projection X = M_denoised A^T + mean
      4. iterative regularisation  Y_next = lam * X + (1 - lam) * Y
    The last X is the denoised cube. Pure NumPy / scikit-image, no GUI code.
    """
    def say(msg):
        if progress is not None:
            progress(msg)

    rows, cols, B = cube.shape
    Y = np.asarray(cube, dtype=np.float32).reshape(-1, B)
    X = Y
    used_rank = None
    for it in range(int(n_iter)):
        _check_cancel(cancel)
        mu = Y.mean(axis=0, keepdims=True)
        Yc = Y - mu
        cov = (Yc.T @ Yc).astype(np.float64) / Yc.shape[0]
        w, V = np.linalg.eigh(cov)
        w, V = w[::-1], V[:, ::-1]                      # descending eigenvalues
        k = int(rank) if rank else estimate_subspace_rank(w, B)
        k = int(np.clip(k, 1, B))
        used_rank = k
        A = V[:, :k].astype(np.float32)
        M = (Yc @ A).reshape(rows, cols, k)
        for j in range(k):
            _check_cancel(cancel)
            say(f"NGmeet iteration {it + 1}/{n_iter}: filtering eigen-image {j + 1}/{k}")
            M[:, :, j] = nlm_filter_2d(M[:, :, j], sigma)
        X = M.reshape(-1, k) @ A.T + mu
        Y = lam * X + (1.0 - lam) * Y
    return X.reshape(rows, cols, B).astype(np.float32), used_rank


def rgb_estimate(cube, wavelengths, targets=(640.0, 550.0, 460.0), fallback=(0.7, 0.5, 0.3)):
    """Contrast-stretched RGB (2-98 % per channel) from the bands nearest the target wavelengths."""
    B = cube.shape[2]
    if wavelengths is not None and len(wavelengths) == B and np.all(np.isfinite(wavelengths)):
        wl = np.asarray(wavelengths, dtype=float)
        idx = [int(np.argmin(np.abs(wl - t))) for t in targets]
    else:
        idx = [min(B - 1, int(B * f)) for f in fallback]
    out = np.zeros(cube.shape[:2] + (3,), dtype=np.float32)
    for c, b in enumerate(idx):
        ch = np.asarray(cube[:, :, b], dtype=np.float32)
        if np.all(np.isnan(ch)):
            continue
        lo, hi = np.nanpercentile(ch, (2, 98))
        out[:, :, c] = np.clip((np.nan_to_num(ch, nan=lo) - lo) / max(hi - lo, 1e-12), 0.0, 1.0)
    return out, idx


def spectral_moving_mean(cube, window, chunk_rows=32):
    """Moving average along the spectral axis (window shrinks at the two ends, like MATLAB movmean).

    NaN values are ignored. Output dtype: float64 stays float64, everything else becomes float32.
    """
    window = int(window)
    rows, cols, B = cube.shape
    if window < 1 or window > B:
        raise ValueError(f"The spectral window must be between 1 and the number of bands ({B}).")
    out_dtype = np.float64 if cube.dtype == np.float64 else np.float32
    out = np.empty(cube.shape, dtype=out_dtype)
    lo = window // 2                     # elements before the centre
    hi = window - 1 - lo                 # elements after the centre
    idx = np.arange(B)
    a = np.clip(idx - lo, 0, B)          # window start (inclusive)
    b = np.clip(idx + hi + 1, 0, B)      # window end (exclusive)
    for r in range(0, rows, chunk_rows):
        x = np.asarray(cube[r:r + chunk_rows], dtype=np.float64)
        valid = np.isfinite(x)
        cs = np.concatenate([np.zeros(x.shape[:2] + (1,)), np.cumsum(np.where(valid, x, 0.0), axis=2)], axis=2)
        cn = np.concatenate([np.zeros(x.shape[:2] + (1,)), np.cumsum(valid, axis=2)], axis=2)
        n = cn[:, :, b] - cn[:, :, a]
        with np.errstate(invalid="ignore", divide="ignore"):
            m = (cs[:, :, b] - cs[:, :, a]) / n
        m[n == 0] = np.nan
        out[r:r + chunk_rows] = m
    return out


def mnf_smile_reduction(cube, n_components, max_samples=200000, chunk=200000):
    """Smile reduction by MNF reconstruction: the smile is a smooth, low-energy spatial pattern that
    sits in the noisy high-order MNF components, so the cube is rebuilt from the first components only.

    Noise covariance is estimated from differences of horizontally adjacent pixels (shift difference).
    """
    rows, cols, B = cube.shape
    k = int(n_components)
    if k < 1 or k > B:
        raise ValueError(f"The number of MNF components must be between 1 and {B}.")
    X = np.asarray(cube, dtype=np.float64)
    if not np.all(np.isfinite(X)):
        X = np.where(np.isfinite(X), X, np.nanmean(np.where(np.isfinite(X), X, np.nan), axis=(0, 1), keepdims=True))
    flat = X.reshape(-1, B)
    if flat.shape[0] > max_samples:
        sel = np.sort(np.random.default_rng(0).choice(flat.shape[0], max_samples, replace=False))
        sample = flat[sel]
    else:
        sample = flat
    mean = sample.mean(axis=0)
    if cols < 2:
        raise ValueError("MNF needs at least two columns to estimate noise.")
    d = (X[:, 1:, :] - X[:, :-1, :]).reshape(-1, B)
    if d.shape[0] > max_samples:
        d = d[np.sort(np.random.default_rng(1).choice(d.shape[0], max_samples, replace=False))]
    Cn = np.cov(d, rowvar=False) / 2.0 + 1e-10 * np.eye(B)
    Cx = np.cov(sample, rowvar=False)
    # whiten noise: Cn = V diag(w) V^T
    w, V = np.linalg.eigh(Cn)
    W = V / np.sqrt(np.maximum(w, 1e-12))
    ev, U = np.linalg.eigh(W.T @ Cx @ W)
    order = np.argsort(ev)[::-1]
    T = W @ U[:, order]                          # forward MNF transform (columns = components)
    Tinv = np.linalg.pinv(T)                     # inverse (rows)
    out_dtype = np.float64 if cube.dtype == np.float64 else np.float32
    out = np.empty(cube.shape, dtype=out_dtype)
    flat_out = out.reshape(-1, B)
    for s0 in range(0, flat.shape[0], chunk):
        z = (flat[s0:s0 + chunk] - mean) @ T[:, :k]
        flat_out[s0:s0 + chunk] = z @ Tinv[:k, :] + mean
    return out


def _sample_pixels(cube, max_pixels, seed=0):
    """(N, bands) float64 matrix of finite pixel spectra, subsampled to at most max_pixels rows."""
    flat = cube.reshape(-1, cube.shape[2])
    if flat.shape[0] > max_pixels:
        idx = np.sort(np.random.default_rng(seed).choice(flat.shape[0], max_pixels, replace=False))
        flat = flat[idx]
    X = np.asarray(flat, dtype=np.float64)
    return X[np.all(np.isfinite(X), axis=1)]


def count_endmembers_hfc(cube, pfa=1e-3, noise_whiten=True, max_pixels=200000):
    """Virtual dimensionality by the HFC / NWHFC method (Harsanyi-Farrand-Chang; Chang & Du 2004).

    For every eigenvalue index l the sample correlation eigenvalue (lam_corr) is compared with the sample
    covariance eigenvalue (lam_cov). Without signal both are equal; with signal lam_corr > lam_cov. A
    Neyman-Pearson test with false-alarm probability `pfa` decides, and the number of indices where
    signal is detected is the number of endmembers. With noise_whiten=True (NWHFC) the data are first
    whitened with the noise covariance estimated by multiple regression (Roger & Arnold).
    Returns (n_endmembers, details dict).
    """
    from scipy.special import erfcinv
    X = _sample_pixels(cube, max_pixels)
    N, B = X.shape
    if N < B + 1:
        raise ValueError(f"Too few valid pixels ({N}) for {B} bands.")
    if not (0.0 < pfa < 0.5):
        raise ValueError("The false-alarm probability must be between 0 and 0.5.")
    if noise_whiten:
        K = np.cov(X, rowvar=False) + 1e-12 * np.eye(B)
        P = np.linalg.pinv(K)
        d = np.maximum(np.diag(P), 1e-30)
        Kn = (P / d[:, None]) / d[None, :]          # covariance of the regression residuals
        Kn = (Kn + Kn.T) / 2.0
        w, V = np.linalg.eigh(Kn)
        w = np.maximum(w, 1e-12 * w.max())
        X = X @ (V / np.sqrt(w))                    # noise-whitened data
    R = (X.T @ X) / N                               # correlation matrix
    Kc = np.cov(X, rowvar=False)                    # covariance matrix
    lam_corr = np.sort(np.linalg.eigvalsh(R))[::-1]
    lam_cov = np.sort(np.linalg.eigvalsh(Kc))[::-1]
    sigma = np.sqrt(2.0 * (lam_corr ** 2 + lam_cov ** 2) / N)
    tau = np.sqrt(2.0) * sigma * erfcinv(2.0 * pfa)
    detected = (lam_corr - lam_cov) > tau
    n = int(np.count_nonzero(detected))
    return n, {"lam_corr": lam_corr, "lam_cov": lam_cov, "threshold": tau, "detected": detected,
               "pfa": float(pfa), "noise_whiten": bool(noise_whiten), "pixels": int(N)}


def _reduce_dimensions(cube, X_idx_flat, k, method, max_noise_pixels=200000):
    """Scores (n_selected, k) of the selected pixels after PCA or MNF reduction to k components."""
    rows, cols, B = cube.shape
    flat = cube.reshape(-1, B)
    X = np.asarray(flat[X_idx_flat], dtype=np.float64)
    mean = X.mean(axis=0)
    Xc = X - mean
    if method == "PCA":
        sub = Xc[np.random.default_rng(0).choice(len(Xc), min(len(Xc), 100000), replace=False)]
        _, _, Vt = np.linalg.svd(sub, full_matrices=False)
        return Xc @ Vt[:k].T
    # MNF: noise covariance from differences of horizontally adjacent pixels
    if cols < 2:
        raise ValueError("MNF needs at least two columns; choose PCA instead.")
    c = np.asarray(cube, dtype=np.float64)
    d = (c[:, 1:, :] - c[:, :-1, :]).reshape(-1, B)
    d = d[np.all(np.isfinite(d), axis=1)]
    if d.shape[0] > max_noise_pixels:
        d = d[np.sort(np.random.default_rng(1).choice(d.shape[0], max_noise_pixels, replace=False))]
    Cn = np.cov(d, rowvar=False) / 2.0 + 1e-10 * np.eye(B)
    w, V = np.linalg.eigh(Cn)
    W = V / np.sqrt(np.maximum(w, 1e-12))
    Cx = np.cov(Xc, rowvar=False)
    ev, U = np.linalg.eigh(W.T @ Cx @ W)
    order = np.argsort(ev)[::-1]
    return Xc @ (W @ U[:, order[:k]])


def atgp_indices(Y, n):
    """Automatic Target Generation Process: first the largest-norm pixel, then repeatedly the pixel
    with the largest residual after projecting out the ones already chosen."""
    chosen = [int(np.argmax(np.einsum("ij,ij->i", Y, Y)))]
    Q = np.zeros((0, Y.shape[1]))
    for _ in range(n - 1):
        v = Y[chosen[-1]].copy()
        for q in Q:
            v -= (q @ v) * q
        nv = np.linalg.norm(v)
        if nv < 1e-12:
            break
        Q = np.vstack([Q, v / nv])
        R = Y - (Y @ Q.T) @ Q
        r = np.einsum("ij,ij->i", R, R)
        r[chosen] = -1.0
        chosen.append(int(np.argmax(r)))
    return chosen


def ppi_endmembers(cube, n_end, n_vectors=104, reduction="MNF", max_pixels=200000, seed=0,
                   progress=None, cancel=None):
    """Pixel Purity Index (Boardman, Kruse and Green, 1995).

    1) reduce the data to n_end components (MNF or PCA); 2) generate n_vectors random unit vectors
    (skewers); 3) project every pixel onto each skewer and count how often a pixel is the maximum or
    minimum (its PPI count); 4) the n_end pixels with the highest PPI counts are the endmembers.
    Returns (endmember spectra (n_end, bands), pixel indices, ppi counts of all sampled pixels).
    """
    n_end = int(n_end)
    n_vectors = int(n_vectors)
    rows, cols, B = cube.shape
    total = rows * cols
    if n_end < 2 or n_end > B:
        raise ValueError(f"The number of endmembers must be between 2 and {B}.")
    if n_vectors < 1:
        raise ValueError("At least one skewer is needed.")
    flat = cube.reshape(-1, B)
    idx_all = np.arange(total)
    if total > max_pixels:
        idx_all = np.sort(np.random.default_rng(seed).choice(total, max_pixels, replace=False))
    ok = np.all(np.isfinite(np.asarray(flat[idx_all], dtype=np.float64)), axis=1)
    idx_all = idx_all[ok]
    Y = _reduce_dimensions(cube, idx_all, n_end, reduction)
    rng = np.random.default_rng(seed)
    counts = np.zeros(len(Y))
    block = 256
    for s0 in range(0, n_vectors, block):
        if cancel is not None and cancel.is_set():
            raise TaskCancelled()
        k = min(block, n_vectors - s0)
        skew = rng.normal(size=(k, Y.shape[1]))
        skew /= np.maximum(np.linalg.norm(skew, axis=1, keepdims=True), 1e-12)
        proj = Y @ skew.T
        np.add.at(counts, np.argmax(proj, axis=0), 1)
        np.add.at(counts, np.argmin(proj, axis=0), 1)
        if progress is not None:
            progress(f"PPI: {s0 + k}/{n_vectors} skewers")
    order = np.argsort(-counts, kind="stable")
    sel = idx_all[order[:n_end]]
    return np.asarray(flat[sel], dtype=np.float64), sel, counts


def fippi_endmembers(cube, n_end, reduction="MNF", max_iter=50, max_pixels=200000, seed=0,
                     progress=None, cancel=None):
    """Fast Iterative Pixel Purity Index (Chang & Plaza, IEEE GRSL 3(1), 2006).

    1) reduce the data to n_end components (MNF or PCA); 2) ATGP gives the initial skewers (pixels);
    3) project all pixels onto the skewers, count how often each pixel is an extreme (PPI count);
    4) the pixels with the highest counts are added to the skewers; repeat until the skewer set no
    longer changes. If the final skewer set has more than n_end pixels, ATGP inside it keeps n_end.
    Returns (endmember spectra (n_end, bands), pixel indices, iterations used).
    """
    n_end = int(n_end)
    rows, cols, B = cube.shape
    total = rows * cols
    if n_end < 2 or n_end > B:
        raise ValueError(f"The number of endmembers must be between 2 and {B}.")
    flat = cube.reshape(-1, B)
    idx_all = np.arange(total)
    if total > max_pixels:
        idx_all = np.sort(np.random.default_rng(seed).choice(total, max_pixels, replace=False))
    ok = np.all(np.isfinite(np.asarray(flat[idx_all], dtype=np.float64)), axis=1)
    idx_all = idx_all[ok]
    Y = _reduce_dimensions(cube, idx_all, n_end, reduction)
    skewers = atgp_indices(Y, n_end)
    counts = np.zeros(len(Y))
    iters = 0
    for it in range(int(max_iter)):
        if cancel is not None and cancel.is_set():
            raise TaskCancelled()
        iters = it + 1
        S = Y[skewers]
        S = S / np.maximum(np.linalg.norm(S, axis=1, keepdims=True), 1e-12)
        proj = Y @ S.T                                       # (N, n_skewers)
        counts = np.zeros(len(Y))
        np.add.at(counts, np.argmax(proj, axis=0), 1)        # maximum extreme of every skewer
        np.add.at(counts, np.argmin(proj, axis=0), 1)        # minimum extreme of every skewer
        order = np.argsort(-counts, kind="stable")
        cand = [int(i) for i in order[:n_end] if counts[i] > 0]
        new_set = list(dict.fromkeys(skewers + cand))        # union, order kept
        if progress is not None:
            progress(f"FIPPI iteration {iters}: {len(new_set)} skewers")
        if set(new_set) == set(skewers):
            break
        skewers = new_set
    # The converged skewer set holds the purest pixels (usually a few more than n_end). Keep n_end of
    # them by running ATGP inside that set, which favours the pixels that span the largest simplex.
    cand_idx = list(dict.fromkeys(skewers))
    if len(cand_idx) > n_end:
        pick = atgp_indices(Y[cand_idx], n_end)
        final = [cand_idx[i] for i in pick]
    else:
        final = cand_idx
    if len(final) < n_end:                                   # degenerate data: pad with high-count pixels
        order = np.argsort(-counts, kind="stable")
        final += [int(i) for i in order if int(i) not in final][:n_end - len(final)]
    sel = idx_all[final]
    return np.asarray(flat[sel], dtype=np.float64), sel, iters


def nfindr_endmembers(cube, n_end, max_iter=3, max_pixels=200000, seed=0, progress=None, cancel=None):
    """N-FINDR (Winter 1999): find the pixels whose simplex has the largest volume.

    The data are reduced to n_end-1 dimensions by PCA. Replacing endmember j by a pixel x changes the
    simplex volume proportionally to |cofactor_j . [1; x]|, so every pixel can be tested at once.
    Returns (endmember spectra (n_end, bands), pixel indices into the sampled pixel set).
    """
    n_end = int(n_end)
    X = _sample_pixels(cube, max_pixels, seed)
    N, B = X.shape
    if n_end < 2 or n_end > min(B, N):
        raise ValueError(f"The number of endmembers must be between 2 and {min(B, N)}.")
    mean = X.mean(axis=0)
    U, S, Vt = np.linalg.svd(X[np.random.default_rng(seed).choice(N, min(N, 50000), replace=False)] - mean,
                             full_matrices=False)
    Y = (X - mean) @ Vt[:n_end - 1].T                      # (N, n_end-1)
    A = np.hstack([np.ones((N, 1)), Y])                    # homogeneous coordinates
    rng = np.random.default_rng(seed)
    idx = rng.choice(N, n_end, replace=False)
    for it in range(int(max_iter)):
        changed = False
        for j in range(n_end):
            if cancel is not None and cancel.is_set():
                raise TaskCancelled()
            M = A[idx]                                      # (n_end, n_end), rows are the endmembers
            rest = np.delete(M, j, axis=0)
            # cofactor vector c: det(M with row j replaced by a) = c . a
            _, _, vt = np.linalg.svd(rest)
            c = vt[-1]                                      # null direction of the other rows
            # scale so that c . a equals the determinant (up to a constant factor, same for all pixels)
            vol = np.abs(A @ c)
            best = int(np.argmax(vol))
            if best != idx[j] and vol[best] > vol[idx[j]] * (1 + 1e-12):
                idx[j] = best
                changed = True
        if progress is not None:
            progress(f"N-FINDR iteration {it + 1}/{max_iter}")
        if not changed:
            break
    return X[idx].copy(), idx


def rgb_pair_shared_stretch(cube_a, cube_b, wavelengths):
    """RGB of two cubes on ONE common display scale (2-98 % limits taken from cube_a).

    Using the same limits for both images keeps a before/after comparison honest: a correction that
    lowers the values shows up as a darker picture instead of being stretched back to look identical.
    """
    B = cube_a.shape[2]
    if wavelengths is not None and len(wavelengths) == B and np.all(np.isfinite(wavelengths)):
        wl = np.asarray(wavelengths, dtype=float)
        idx = [int(np.argmin(np.abs(wl - t))) for t in (640.0, 550.0, 460.0)]
    else:
        idx = [min(B - 1, int(B * f)) for f in (0.7, 0.5, 0.3)]
    outs = [np.zeros(cube_a.shape[:2] + (3,), dtype=np.float32) for _ in range(2)]
    for c, b in enumerate(idx):
        ca = np.asarray(cube_a[:, :, b], dtype=np.float32)
        cb = np.asarray(cube_b[:, :, b], dtype=np.float32)
        if np.all(np.isnan(ca)):
            continue
        lo, hi = np.nanpercentile(ca, (2, 98))
        span = max(float(hi - lo), 1e-12)
        for o, ch in zip(outs, (ca, cb)):
            o[:, :, c] = np.clip((np.nan_to_num(ch, nan=lo) - lo) / span, 0.0, 1.0)
    return outs[0], outs[1], idx


def psnr(ref, test, data_range=1.0):
    mse = float(np.mean((np.asarray(ref, dtype=np.float64) - np.asarray(test, dtype=np.float64)) ** 2))
    return float("inf") if mse == 0 else float(10.0 * np.log10(data_range ** 2 / mse))


def mean_ssim(ref, test, max_bands=20, data_range=1.0):
    """Mean SSIM over (up to max_bands evenly spaced) bands."""
    from skimage.metrics import structural_similarity
    B = ref.shape[2]
    idx = np.unique(np.linspace(0, B - 1, min(B, max_bands)).astype(int))
    vals = [structural_similarity(ref[:, :, b], test[:, :, b], data_range=data_range) for b in idx]
    return float(np.mean(vals))


def run_ngmeet_job(cube, validate, sigma, n_iter, lam, rank, noise_std, seed,
                   progress=None, cancel=None):
    """Normalise -> (optionally add Gaussian noise) -> NGmeet -> metrics.

    validate=True : noise is simulated on the normalised cube so the result can be compared with
                    the clean original (PSNR / SSIM). The user's cube is not modified.
    validate=False: the real cube is denoised and the result is mapped back to the original units.
    """
    def say(msg):
        if progress is not None:
            progress(msg)

    rows, cols, bands = cube.shape
    say("Normalising the cube...")
    X, valid_px = prepare_features(cube.reshape(-1, bands))        # fills NaNs with band means
    lo, hi = float(np.min(X[valid_px])), float(np.max(X[valid_px]))
    span = max(hi - lo, 1e-12)
    clean = ((X - lo) / span).reshape(rows, cols, bands).astype(np.float32)
    del X
    _check_cancel(cancel)

    res = {"lo": lo, "hi": hi, "validate": validate, "valid_px": valid_px,
           "params": {"sigma": sigma, "iterations": n_iter, "lambda": lam}}
    if validate:
        say("Adding Gaussian noise...")
        rng = np.random.default_rng(seed)
        noisy = clean + rng.normal(0.0, noise_std, size=clean.shape).astype(np.float32)
        noisy = np.clip(noisy, 0.0, 1.0)                           # as MATLAB's imnoise does
        res["params"]["noise_std"] = noise_std
        res["params"]["seed"] = seed
        src = noisy
    else:
        noisy = None
        src = clean

    den, used_rank = ngmeet_denoise(src, sigma=sigma, n_iter=n_iter, lam=lam, rank=rank,
                                    progress=progress, cancel=cancel)
    res["params"]["rank"] = used_rank
    res["clean"], res["noisy"], res["denoised"] = clean, noisy, den
    if validate:
        say("Computing PSNR / SSIM...")
        res["metrics"] = {
            "psnr_noisy": psnr(clean, noisy), "psnr_denoised": psnr(clean, den),
            "ssim_noisy": mean_ssim(clean, noisy), "ssim_denoised": mean_ssim(clean, den),
        }
    return res


# ----------------------------------------------------------------------
# Atmospheric correction: empirical line
# ----------------------------------------------------------------------
def load_field_spectrum(path):
    """Read (wavelength, reflectance) pairs from a CSV / TXT file.

    Header lines and non-numeric lines are skipped; the first two numbers of each line are used.
    Fill values (e.g. -1.2e34 in USGS / ECOSTRESS files) and non-finite points are dropped.
    Returns (wavelengths, values) sorted by wavelength, in the file's own units.
    """
    import re
    wl, val = [], []
    with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
        for line in f:
            parts = [q for q in re.split(r"[,;\t ]+", line.strip()) if q]
            if len(parts) < 2:
                continue
            try:
                a, b = float(parts[0]), float(parts[1])
            except ValueError:
                continue
            if np.isfinite(a) and np.isfinite(b) and b > -1e3:
                wl.append(a)
                val.append(b)
    if len(wl) < 2:
        raise ValueError("Need at least two numeric 'wavelength, reflectance' lines.")
    wl, val = np.asarray(wl, dtype=float), np.asarray(val, dtype=float)
    wl, idx = np.unique(wl, return_index=True)           # sorted, duplicates removed
    return wl, val[idx]


def auto_wavelengths_to_nm(wl):
    """Wavelengths below 100 are taken to be micrometres and converted to nm."""
    wl = np.asarray(wl, dtype=float)
    return wl * 1000.0 if np.nanmax(wl) < 100.0 else wl


def resample_field_spectrum(field_wl_nm, field_values, target_wl_nm):
    """Linear resampling of a field spectrum onto the cube wavelengths (NaN outside its range)."""
    return np.interp(target_wl_nm, field_wl_nm, field_values, left=np.nan, right=np.nan)


def empirical_line(measured, field):
    """Per-band empirical line:  reflectance = gain * measured + offset.

    measured, field: (n_ref, bands). `measured` is the image value of each reference material
    (DN, TOA radiance or TOA reflectance), `field` its known surface reflectance resampled to the
    image bands (NaN where the field spectrum has no value).
    Per band: >= 2 references -> least-squares line (gain and offset);
              1 reference    -> offset = 0, gain = field / measured;
              0 references   -> error.
    Returns gain, offset, n_refs_used (per band) and a boolean mask of bands that used the
    single-reference / through-origin solution.
    """
    measured = np.asarray(measured, dtype=np.float64)
    field = np.asarray(field, dtype=np.float64)
    valid = np.isfinite(measured) & np.isfinite(field)
    n = valid.sum(axis=0)
    if np.any(n == 0):
        bad = np.flatnonzero(n == 0)
        raise ValueError(
            f"{bad.size} band(s) have no field spectrum value (first band index {int(bad[0])}, "
            f"last {int(bad[-1])}). The field spectra must cover the image wavelengths; "
            "remove those bands in Band Management or add reference spectra that cover them.")
    x = np.where(valid, measured, 0.0)
    y = np.where(valid, field, 0.0)
    nn = n.astype(np.float64)
    mx, my = x.sum(axis=0) / nn, y.sum(axis=0) / nn
    dx = np.where(valid, measured - mx, 0.0)
    dy = np.where(valid, field - my, 0.0)
    sxx, sxy = (dx * dx).sum(axis=0), (dx * dy).sum(axis=0)
    regress = (n >= 2) & (sxx > 1e-10 * (x * x).sum(axis=0) + 1e-300)
    xx0 = (x * x).sum(axis=0)
    if np.any(~regress & (xx0 <= 0)):
        raise ValueError("A reference has a measured value of exactly zero in a band where the gain "
                         "cannot be estimated. Check the reference ROIs (dark or empty pixels).")
    with np.errstate(divide="ignore", invalid="ignore"):
        gain = np.where(regress, sxy / np.where(sxx > 0, sxx, 1.0), (x * y).sum(axis=0) / np.where(xx0 > 0, xx0, 1.0))
    offset = np.where(regress, my - gain * mx, 0.0)
    return gain, offset, n, ~regress


def apply_gain_offset(cube, gain, offset, chunk_rows=64):
    """cube * gain + offset per band, in float32 and in row blocks to limit memory."""
    out = np.empty(cube.shape, dtype=np.float32)
    g = np.asarray(gain, dtype=np.float32).reshape(1, 1, -1)
    o = np.asarray(offset, dtype=np.float32).reshape(1, 1, -1)
    for r0 in range(0, cube.shape[0], chunk_rows):
        out[r0:r0 + chunk_rows] = np.asarray(cube[r0:r0 + chunk_rows], dtype=np.float32) * g + o
    return out


# ----------------------------------------------------------------------
# Atmospheric correction: dark pixel subtraction, flat field, IARR
# ----------------------------------------------------------------------
def scene_mean_spectrum(cube, chunk_rows=64):
    """Mean spectrum over all pixels (NaN-aware, chunked so no full-size temporary is made)."""
    C = cube.shape[2]
    total = np.zeros(C, dtype=np.float64)
    count = np.zeros(C, dtype=np.float64)
    for r0 in range(0, cube.shape[0], chunk_rows):
        blk = np.asarray(cube[r0:r0 + chunk_rows], dtype=np.float64).reshape(-1, C)
        ok = np.isfinite(blk)
        total += np.where(ok, blk, 0.0).sum(axis=0)
        count += ok.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(count > 0, total / count, np.nan)


def resolve_dark_pixels(dark, shape):
    """Bring `darkPixels` to an array that broadcasts against an (M, N, C) cube.

    scalar            -> the same value from every pixel of every band
    C-element vector  -> one value per band
    (M, N) array      -> one value per pixel, subtracted from every band
    (Md, Nd, C) array:
        Md == M and Nd == N  -> pixel-by-pixel
        Md != M and Nd != N  -> mean over both spatial axes   (1 x 1 x C)
        Md != M and Nd == N  -> mean over rows                (1 x N x C)
        Md == M and Nd != N  -> mean over columns             (M x 1 x C)
    """
    M, N, C = shape
    d = np.asarray(dark, dtype=np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        if d.size == 1:
            return d.reshape(1, 1, 1)
        if d.ndim == 1:
            if d.size != C:
                raise ValueError(f"A dark-pixel vector must have one value per band ({C}); got {d.size}.")
            return d.reshape(1, 1, C)
        if d.ndim == 2:
            if d.shape != (M, N):
                raise ValueError(f"A 2-D dark-pixel array must match the image size ({M} x {N}); got {d.shape[0]} x {d.shape[1]}.")
            return d.reshape(M, N, 1)
        if d.ndim == 3:
            Md, Nd, Cd = d.shape
            if Cd != C:
                raise ValueError(f"The third dimension of the dark-pixel array must be {C} (bands); got {Cd}.")
            if Md == M and Nd == N:
                return d
            if Md != M and Nd != N:
                return np.nanmean(d, axis=(0, 1), keepdims=True)
            if Md != M and Nd == N:
                return np.nanmean(d, axis=0, keepdims=True)
            return np.nanmean(d, axis=1, keepdims=True)
    raise ValueError("darkPixels must be a scalar, a vector, a 2-D array or a 3-D array.")


def subtract_dark_pixels(cube, dark, clip_negative=False, chunk_rows=64):
    """Dark object / dark pixel subtraction (see resolve_dark_pixels for the accepted shapes).

    Reference: Souri, A. H. and M. A. Sharifi. "Evaluation of Scene-Based Empirical Approaches for
    Atmospheric Correction of Hyperspectral Imagery." 33rd Asian Conference on Remote Sensing,
    Pattaya, Thailand, November 2012.
    """
    d = resolve_dark_pixels(dark, cube.shape)
    out = np.empty(cube.shape, dtype=np.float32)
    for r0 in range(0, cube.shape[0], chunk_rows):
        dd = d if d.shape[0] == 1 else d[r0:r0 + chunk_rows]
        blk = np.asarray(cube[r0:r0 + chunk_rows], dtype=np.float64) - dd
        if clip_negative:
            blk = np.where(np.isnan(blk), np.nan, np.maximum(blk, 0.0))
        out[r0:r0 + chunk_rows] = blk
    return out


def _check_reference_spectrum(ref, what):
    ref = np.asarray(ref, dtype=np.float64).ravel()
    bad = ~np.isfinite(ref) | (ref <= 0)
    if np.any(bad):
        idx = np.flatnonzero(bad)
        raise ValueError(
            f"The {what} is zero, negative or invalid in {idx.size} band(s) (first band index {int(idx[0])}), "
            "so the division is undefined there. Remove those bands in Band Management first "
            "(or pick a different region / run dark subtraction with a smaller value).")
    return ref


def divide_by_spectrum(cube, ref, chunk_rows=64):
    """Divide every pixel spectrum by `ref` (per band), float32 output, chunked."""
    out = np.empty(cube.shape, dtype=np.float32)
    r = np.asarray(ref, dtype=np.float64).reshape(1, 1, -1)
    for r0 in range(0, cube.shape[0], chunk_rows):
        out[r0:r0 + chunk_rows] = np.asarray(cube[r0:r0 + chunk_rows], dtype=np.float64) / r
    return out


def flat_field_correct(cube, flat_spectrum):
    """Flat field correction: pixel spectrum / mean spectrum of a bright, spectrally flat area.

    The result is relative reflectance (the flat area itself becomes ~1 in every band).
    Reference: Roberts, D. A., Y. Yamaguchi, and R. J. P. Lyon. "Comparison of Various Techniques
    for Calibration of AIS Data." In Proceedings of the Second Airborne Imaging Spectrometer Data
    Analysis Workshop, ed. Gregg Vane and Alexander F. H. Goetz, 21-30. Pasadena: Jet Propulsion
    Laboratory, 1986.
    """
    ref = _check_reference_spectrum(flat_spectrum, "flat-field mean spectrum")
    if ref.size != cube.shape[2]:
        raise ValueError(f"The flat-field spectrum has {ref.size} values but the cube has {cube.shape[2]} bands.")
    return divide_by_spectrum(cube, ref), ref


def iarr_correct(cube):
    """Internal Average Relative Reflectance: pixel spectrum / mean spectrum of the whole scene.

    Reference: Kruse, Fred A. "Use of Airborne Imaging Spectrometer Data to Map Minerals Associated
    with Hydrothermally Altered Rocks in the Northern Grapevine Mountains, Nevada, and California."
    Remote Sensing of Environment 24, no. 1 (February 1988): 31-51.
    https://doi.org/10.1016/0034-4257(88)90004-1
    """
    ref = _check_reference_spectrum(scene_mean_spectrum(cube), "scene mean spectrum")
    return divide_by_spectrum(cube, ref), ref


def load_dark_array(path):
    """Read darkPixels from a .npy file (any shape) or a text/CSV file (scalar or per-band vector)."""
    import re
    if str(path).lower().endswith(".npy"):
        return np.load(path)
    vals = []
    with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
        for tok in re.split(r"[,;\s]+", f.read()):
            try:
                vals.append(float(tok))
            except ValueError:
                continue                                   # header text
    if not vals:
        raise ValueError("No numeric values found in the file.")
    return np.asarray(vals, dtype=np.float64)


# ----------------------------------------------------------------------
# Automatic unmixing with NMF (reference: Spectral Unmixing.py)
# ----------------------------------------------------------------------
def run_nmf_unmix(cube, n_end, band_lo, band_hi, max_fit_px, seed=42,
                  progress=None, cancel=None):
    def say(msg):
        if progress is not None:
            progress(msg)

    rows, cols, _ = cube.shape
    sub = cube[:, :, band_lo:band_hi + 1]
    nb = sub.shape[2]
    X, valid_px = prepare_features(sub.reshape(-1, nb))
    n_neg = int(np.sum(X < 0))
    if n_neg:
        X = np.clip(X, 0.0, None)          # NMF needs non-negative data
    Xv = X[valid_px]
    if Xv.shape[0] < n_end:
        raise RuntimeError("Fewer valid pixels than requested endmembers.")

    rng = np.random.default_rng(seed)
    if max_fit_px and Xv.shape[0] > max_fit_px:
        fit_idx = np.sort(rng.choice(Xv.shape[0], size=max_fit_px, replace=False))
    else:
        fit_idx = None

    say(f"Fitting NMF with {n_end} endmembers on "
        f"{Xv.shape[0] if fit_idx is None else len(fit_idx)} pixels...")
    nmf = NMF(n_components=n_end, init="random", random_state=seed, max_iter=300)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        W_fit = nmf.fit_transform(Xv if fit_idx is None else Xv[fit_idx])
    H = nmf.components_
    _check_cancel(cancel)

    if fit_idx is None:
        W = W_fit
    else:
        W = np.empty((Xv.shape[0], n_end), dtype=np.float32)
        chunk = 200000
        for s0 in range(0, Xv.shape[0], chunk):
            _check_cancel(cancel)
            say(f"Estimating abundances... {min(s0 + chunk, Xv.shape[0])}/{Xv.shape[0]} pixels")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                W[s0:s0 + chunk] = nmf.transform(Xv[s0:s0 + chunk])

    abundance = np.zeros((rows * cols, n_end), dtype=np.float32)
    abundance[valid_px] = W
    classified = -np.ones(rows * cols, dtype=np.int32)
    classified[valid_px] = np.argmax(W, axis=1)
    return {
        "abundance": abundance.reshape(rows, cols, n_end),
        "endmembers": H,
        "classified": classified.reshape(rows, cols),
        "band_range": (band_lo, band_hi),
        "n_negative": n_neg,
    }


# ----------------------------------------------------------------------
# 2D CNN on PCA patches (reference: ai_hsi.py)
# ----------------------------------------------------------------------
def gather_patches(windows, ids, cols):
    """(B, ps, ps, C) float32 patches centred on flat pixel indices ids."""
    r = ids // cols
    c = ids % cols
    return np.ascontiguousarray(windows[r, c].transpose(0, 2, 3, 1), dtype=np.float32)


def build_cnn(input_shape, n_classes, net_size="reference"):
    """Keras 2D CNN. 'reference' is the architecture from the notebook; 'compact' is lighter."""
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import tensorflow as tf
    from tensorflow.keras import layers, models

    tf.keras.utils.set_random_seed(42)
    if net_size == "compact":
        widths, dense = [32, 64, 128], 128
    else:
        widths, dense = [64, 128, 256, 512], 512

    stack = [layers.Input(shape=input_shape)]
    for bi, w in enumerate(widths):
        stack.append(layers.Conv2D(w, (3, 3), activation="relu", padding="same"))
        stack.append(layers.Conv2D(w, (3, 3), activation="relu", padding="same"))
        if bi < len(widths) - 1:
            stack.append(layers.MaxPooling2D((2, 2)))
            stack.append(layers.Dropout(0.3))
        else:
            stack.append(layers.GlobalAveragePooling2D())
    stack.append(layers.Dense(dense, activation="relu"))
    stack.append(layers.Dropout(0.5))
    stack.append(layers.Dense(n_classes, activation="softmax"))

    model = models.Sequential(stack)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def run_cnn(cube, rois, class_names, n_pca, patch_size, epochs, batch_size, net_size,
            test_frac, split_mode, max_per_class, predict_scope,
            progress=None, cancel=None):
    """Whitened-PCA -> patches -> 2D CNN trained on labelled ROI pixels."""
    def say(msg):
        if progress is not None:
            progress(msg)

    rows, cols, bands = cube.shape
    rng = np.random.default_rng(0)
    patch_size = int(patch_size) | 1               # force odd
    margin = patch_size // 2

    say("Preparing PCA features...")
    X, valid_px = prepare_features(cube.reshape(-1, bands))
    Xv = X[valid_px]
    nc = int(max(1, min(n_pca, bands, Xv.shape[0] - 1)))
    fit_n = min(Xv.shape[0], 200000)
    fit_idx = rng.choice(Xv.shape[0], size=fit_n, replace=False) if fit_n < Xv.shape[0] \
        else np.arange(Xv.shape[0])
    pca = PCA(n_components=nc, whiten=True, random_state=42).fit(Xv[fit_idx])
    feat = np.zeros((rows * cols, nc), dtype=np.float32)
    feat[valid_px] = pca.transform(Xv).astype(np.float32)
    del X, Xv
    padded = np.pad(feat.reshape(rows, cols, nc),
                    ((margin, margin), (margin, margin), (0, 0)), mode="constant")
    windows = sliding_window_view(padded, (patch_size, patch_size), axis=(0, 1))
    _check_cancel(cancel)

    samp_idx, y, rid, used_names, n_conflict = _collect_samples(
        valid_px, rois, class_names, rows, cols)
    test_mask = split_train_test(y, rid, test_frac, split_mode, rng)
    ids_all = np.arange(len(y))
    train_ids = cap_per_class(ids_all[~test_mask], y, max_per_class, rng)
    test_ids = cap_per_class(ids_all[test_mask], y, max_per_class, rng) \
        if test_mask.any() else np.array([], dtype=int)
    if len(np.unique(y[train_ids])) < 2:
        raise RuntimeError("Training set contains fewer than 2 classes after splitting.")

    say("Building the network...")
    model = build_cnn((patch_size, patch_size, nc), len(used_names), net_size)

    def predict_ids(flat_ids):
        out = np.empty(len(flat_ids), dtype=np.int32)
        bs = 512
        for b in range(0, len(flat_ids), bs):
            _check_cancel(cancel)
            xb = gather_patches(windows, flat_ids[b:b + bs], cols)
            out[b:b + bs] = np.argmax(model.predict(xb, verbose=0), axis=1)
        return out

    log = []
    n_tr = len(train_ids)
    for ep in range(int(epochs)):
        order = rng.permutation(train_ids)
        tot_loss = tot_acc = 0.0
        for b in range(0, n_tr, batch_size):
            _check_cancel(cancel)
            sel = order[b:b + batch_size]
            xb = gather_patches(windows, samp_idx[sel], cols)
            out = np.atleast_1d(np.asarray(model.train_on_batch(xb, y[sel]), dtype=float))
            tot_loss += float(out[0]) * len(sel)
            tot_acc += float(out[1] if len(out) > 1 else 0.0) * len(sel)
            say(f"CNN epoch {ep + 1}/{epochs}: batch {b // batch_size + 1}/"
                f"{math.ceil(n_tr / batch_size)}")
        line = f"Epoch {ep + 1:>3}/{epochs}  loss {tot_loss / n_tr:.4f}  train acc {tot_acc / n_tr:.4f}"
        if len(test_ids) > 0:
            val = float(np.mean(predict_ids(samp_idx[test_ids]) == y[test_ids]))
            line += f"  val acc {val:.4f}"
        log.append(line)
        say(line)

    split_desc = ("hold-out whole ROIs" if split_mode == "rois" else "random pixels")
    if len(test_ids) > 0:
        y_pred = predict_ids(samp_idx[test_ids])
        report = format_report(used_names, y[test_ids], y_pred, len(train_ids),
                               len(test_ids), split_desc)
    else:
        report = f"No hold-out set (test fraction = 0). Trained on {len(train_ids)} patches."
    report = _split_notes(report, n_conflict, split_mode, len(test_ids) > 0)
    report += (f"\n\nNetwork: {net_size}, patch {patch_size}x{patch_size}, {nc} whitened PCA "
               f"components, {epochs} epochs, batch {batch_size}\n\nTraining log:\n" + "\n".join(log))

    if predict_scope == "rois":
        cls_img, _, _ = build_label_images(rois, class_names, rows, cols)
        target = np.flatnonzero((cls_img.ravel() >= 0) & valid_px)
    else:
        target = np.flatnonzero(valid_px)
    labels_flat = -np.ones(rows * cols, dtype=np.int32)
    bs = 512
    for b in range(0, len(target), bs):
        _check_cancel(cancel)
        say(f"Classifying with the CNN... {min(b + bs, len(target))}/{len(target)} pixels")
        sel = target[b:b + bs]
        xb = gather_patches(windows, sel, cols)
        labels_flat[sel] = np.argmax(model.predict(xb, verbose=0), axis=1)

    return {
        "labels": labels_flat.reshape(rows, cols),
        "names": used_names,
        "report": report,
        "title": "2D CNN",
    }


class WorkerSignals(QtCore.QObject):
    progress = QtCore.pyqtSignal(str)
    finished = QtCore.pyqtSignal(object)
    failed = QtCore.pyqtSignal(str)


class Worker(QtCore.QRunnable):
    """Runs fn(progress_callback) on a thread-pool thread."""

    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self.signals = WorkerSignals()

    def run(self):
        try:
            result = self.fn(self.signals.progress.emit)
        except TaskCancelled:
            self.signals.failed.emit("__cancelled__")
        except Exception as e:
            traceback.print_exc()
            self.signals.failed.emit(str(e))
        else:
            self.signals.finished.emit(result)


class ImageCanvas(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = Figure()
        super().__init__(self.fig)
        self.after_clear = None
        self.ax = self.fig.add_subplot(111)
        self.ax.set_axis_off()
        try:
            self.fig.tight_layout()
        except Exception:
            pass

    def show_image(self, img, legend=None, cmap="gray"):
        """Show an image. legend: optional list of (label, (r, g, b)) shown in a key."""
        self.ax.clear()
        self.ax.set_axis_off()

        img = np.asarray(img)

        if img.ndim == 2:
            self.ax.imshow(img, cmap=cmap, aspect="equal")
        elif img.ndim == 3 and img.shape[2] == 3:
            self.ax.imshow(img, aspect="equal")
        else:
            self.ax.imshow(img, aspect="equal")

        # keep the view on the image even if overlays (ROI polygons) stray outside it
        self.ax.set_autoscale_on(False)

        if legend:
            handles = [patches.Patch(facecolor=c, edgecolor="#888888", label=lbl)
                       for lbl, c in legend]
            self.ax.legend(handles=handles, loc="upper right", fontsize=8,
                           framealpha=0.85)

        # lets the main window re-attach ROI selectors that ax.clear() detached
        if self.after_clear is not None:
            self.after_clear()

        restyle_figure(self.fig)
        self.draw()


class PlotCanvas(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = Figure()
        super().__init__(self.fig)
        self.ax = self.fig.add_subplot(111)
        try:
            self.fig.tight_layout()
        except Exception:
            pass

    def _safe_layout(self):
        try:
            self.fig.tight_layout()
        except np.linalg.LinAlgError:
            self.fig.subplots_adjust(left=0.15, bottom=0.15)
        except Exception:
            pass

    def show_lineplot(self, x, y, title=None, xlabel="Band", ylabel="Value"):
        self.ax.clear()

        x = np.asarray(x).ravel()
        y = np.asarray(y).ravel()

        self.ax.plot(x, y)
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        if title:
            self.ax.set_title(title)
        self.ax.grid(True, alpha=0.3)

        self._safe_layout()
        restyle_figure(self.fig)
        self.draw()

    def show_barplot(self, x, y, title=None, xlabel="Component", ylabel="Explained Variance (%)"):
        self.ax.clear()

        x = np.asarray(x).ravel()
        y = np.asarray(y).ravel()

        self.ax.bar(x, y, width=0.8)
        self.ax.set_xticks(x)
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        if title:
            self.ax.set_title(title)
        self.ax.grid(True, axis="y", alpha=0.3)

        if len(y) > 0:
            ymax = float(np.nanmax(y))
            self.ax.set_ylim(0, ymax * 1.1 if ymax > 0 else 1)

        self._safe_layout()
        restyle_figure(self.fig)
        self.draw()

    def show_spectrum(self, x, y, title=None, xlabel="Band"):
        self.show_lineplot(x, y, title=title, xlabel=xlabel, ylabel="Value")

    def show_multiline(self, x, series, title=None, xlabel="Band", ylabel="Value"):
        """series: list of (label, y_array, (r, g, b))."""
        self.ax.clear()
        x = np.asarray(x).ravel()
        for label, y, color in series:
            self.ax.plot(x, np.asarray(y).ravel(), label=label, color=color)
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        if title:
            self.ax.set_title(title)
        self.ax.grid(True, alpha=0.3)
        if series:
            self.ax.legend(fontsize=8)
        self._safe_layout()
        restyle_figure(self.fig)
        self.draw()




class HyperspectralViewer(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        icon_path = resource_path("hypervista.ico")
        if Path(icon_path).exists():
            self.setWindowIcon(QtGui.QIcon(icon_path))

        self.setWindowTitle(f"{APP_NAME} v{APP_VERSION} — {APP_SUBTITLE} | {APP_CREDIT}")
        self.resize(1340, 760)
        self.setAcceptDrops(True)
        self.setMinimumSize(1100, 640)
        self.theme = CURRENT_THEME
        self._apply_window_qss(self.theme)

        self.cube = None
        self.wavelengths = None
        self.metadata = {}
        self.radiometric_stage = "DN / unknown"
        self._radiometric_preview_data = None
        self._active_original_band_indices = None
        self.processing_history = []          # every operation that changed the active cube
        self.el_refs = []                     # empirical-line references: ROI mask + field spectrum
        self.el_result = None                 # last gain / offset and reference comparison
        self.original_cube = None             # cube as loaded (kept so it can be restored)
        self.original_wavelengths = None
        self.original_metadata = None
        self.display_mode = "single"

        self.roi_active = False
        self.roi_selector = None
        self.pending_sel = None          # current (not yet saved) selection
        self.roi_patches = []            # matplotlib artists of drawn ROI outlines
        self.last_roi_mean_full = None
        self.rois = []                   # saved, labelled ROIs
        self.roi_counter = 0
        self.show_roi_outlines = True

        self.sam_angle_map = None
        self.class_names_result = None
        self.last_report = None
        self._worker = None
        self._worker_cube = None
        self._busy = False
        self._cancel_event = None
        self._on_done = None
        self._figure_dialogs = []

        self.nmf_abundance = None
        self.nmf_endmembers = None
        self.nmf_band_range = None
        self.snr_band = None
        self.snr_pixel = None

        self.anomaly_map = None
        self.class_labels = None
        self.class_rgb_map = None
        self.endmembers = []

        self.pca_scores = None
        self.pca_explained_ratio = None
        self.pca_valid_mask = None
        self.pca_n_components = 0

        self.asd_library = []            # imported ASD field spectra (dicts from read_asd)

        self.prep_history = []           # preprocessing steps applied to the active cube
        self._prep_undo = None           # (cube, wavelengths, history) before the last step
        self._prep_compare = None        # sample spectra before / after the last step
        self._prep_map = None            # last derived feature / statistic image
        self._prep_grid_file = None      # (wavelengths, fwhm, file name) for the common grid

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        main_layout = QtWidgets.QVBoxLayout(central)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)

        top_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        main_layout.addWidget(top_splitter, 3)

        left_widget = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_widget)
        left_layout.setSpacing(8)

        self.image_canvas = ImageCanvas()
        left_layout.addWidget(self.image_canvas, 1)

        slider_layout = QtWidgets.QHBoxLayout()
        self.band_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.band_slider.setMinimum(0)
        self.band_slider.setMaximum(0)
        self.band_slider.setValue(0)
        self.band_slider.setEnabled(False)
        self.band_slider.valueChanged.connect(self.on_band_changed)

        self.band_label = QtWidgets.QLabel("Band: -")
        slider_layout.addWidget(self.band_slider, 1)
        slider_layout.addWidget(self.band_label)
        left_layout.addLayout(slider_layout)

        mode_group = QtWidgets.QGroupBox("Display")
        mode_layout = QtWidgets.QGridLayout(mode_group)

        self.radio_single = QtWidgets.QRadioButton("Single band")
        self.radio_rgb = QtWidgets.QRadioButton("RGB composite")
        self.radio_single.setChecked(True)
        self.radio_single.toggled.connect(self.on_mode_changed)
        self.radio_rgb.toggled.connect(self.on_mode_changed)
        mode_layout.addWidget(self.radio_single, 0, 0)
        mode_layout.addWidget(self.radio_rgb, 0, 1)

        mode_layout.addWidget(QtWidgets.QLabel("R band:"), 1, 0)
        self.spin_r = QtWidgets.QSpinBox()
        self.spin_r.setMinimum(0)
        self.spin_r.setMaximum(0)
        self.spin_r.valueChanged.connect(self.on_rgb_band_changed)
        mode_layout.addWidget(self.spin_r, 1, 1)

        mode_layout.addWidget(QtWidgets.QLabel("G band:"), 2, 0)
        self.spin_g = QtWidgets.QSpinBox()
        self.spin_g.setMinimum(0)
        self.spin_g.setMaximum(0)
        self.spin_g.valueChanged.connect(self.on_rgb_band_changed)
        mode_layout.addWidget(self.spin_g, 2, 1)

        mode_layout.addWidget(QtWidgets.QLabel("B band:"), 3, 0)
        self.spin_b = QtWidgets.QSpinBox()
        self.spin_b.setMinimum(0)
        self.spin_b.setMaximum(0)
        self.spin_b.valueChanged.connect(self.on_rgb_band_changed)
        mode_layout.addWidget(self.spin_b, 3, 1)

        left_layout.addWidget(mode_group)

        right_widget = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_widget)
        right_layout.setSpacing(8)

        self.plot_canvas = PlotCanvas()
        self.plot_canvas.setMinimumHeight(320)
        right_layout.addWidget(self.plot_canvas, 1)

        self.info_label = QtWidgets.QLabel(
            f"Welcome to {APP_NAME}. Open a hyperspectral image with File > Open (Ctrl+O) or drag a file onto the window. "
            "ENVI, MATLAB, IDL, HDF5/HDF4, NetCDF, Zarr, GeoTIFF, NumPy, FITS and other GDAL formats are supported. "
            "The tools are organized in the tabs below."
        )
        self.info_label.setWordWrap(True)
        self.info_label.setMinimumHeight(80)
        self.info_label.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)
        self._apply_window_qss(self.theme)
        right_layout.addWidget(self.info_label)

        top_splitter.addWidget(left_widget)
        top_splitter.addWidget(right_widget)
        top_splitter.setStretchFactor(0, 4)
        top_splitter.setStretchFactor(1, 2)
        top_splitter.setSizes([980, 480])

        # ---- Cube-history bar: visible on every tab ----
        hist_bar = QtWidgets.QFrame()
        hist_bar_l = QtWidgets.QHBoxLayout(hist_bar)
        hist_bar_l.setContentsMargins(8, 2, 8, 2)
        hist_bar_l.setSpacing(6)
        self.btn_revert_all = QtWidgets.QPushButton("Revert to original cube")
        self.btn_revert_all.setToolTip("Discard every processing step on every tab and restore the cube as loaded")
        self.btn_revert_all.clicked.connect(self.on_revert_to_original)
        hist_bar_l.addWidget(self.btn_revert_all)
        self.btn_show_history = QtWidgets.QPushButton("Show history")
        self.btn_show_history.clicked.connect(self.on_show_processing_history)
        hist_bar_l.addWidget(self.btn_show_history)
        self.btn_save_history = QtWidgets.QPushButton("Save history...")
        self.btn_save_history.setToolTip("Save the list of operations applied to the cube (JSON or text)")
        self.btn_save_history.clicked.connect(self.on_save_processing_history)
        hist_bar_l.addWidget(self.btn_save_history)
        self.history_bar_label = QtWidgets.QLabel("History: no operations yet")
        hist_bar_l.addWidget(self.history_bar_label, 1)
        main_layout.addWidget(hist_bar)

        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setDocumentMode(True)
        main_layout.addWidget(self.tabs, 2)

        tab1 = QtWidgets.QWidget()
        tab1_layout = QtWidgets.QHBoxLayout(tab1)
        roi_group = QtWidgets.QGroupBox("ROI Tools")
        roi_layout = QtWidgets.QVBoxLayout(roi_group)
        sel_row = QtWidgets.QHBoxLayout()
        self.cb_roi = QtWidgets.QCheckBox("Enable ROI selection")
        self.cb_roi.toggled.connect(self.on_roi_toggled)
        sel_row.addWidget(self.cb_roi)
        self.combo_roi_shape = QtWidgets.QComboBox()
        self.combo_roi_shape.addItems(["Rectangle (drag)", "Polygon (click points)"])
        self.combo_roi_shape.currentIndexChanged.connect(self.on_roi_shape_changed)
        sel_row.addWidget(self.combo_roi_shape)
        roi_layout.addLayout(sel_row)
        self.roi_label = QtWidgets.QLabel("ROI: none")
        self.roi_label.setWordWrap(True)
        roi_layout.addWidget(self.roi_label)
        class_row = QtWidgets.QHBoxLayout()
        class_row.addWidget(QtWidgets.QLabel("Class:"))
        self.combo_class = QtWidgets.QComboBox()
        self.combo_class.setEditable(True)
        self.combo_class.setInsertPolicy(QtWidgets.QComboBox.NoInsert)
        self.combo_class.setEditText("Class 1")
        self.combo_class.setMinimumWidth(110)
        class_row.addWidget(self.combo_class, 1)
        self.btn_add_roi = QtWidgets.QPushButton("Add selection to ROI list")
        self.btn_add_roi.clicked.connect(self.on_add_roi)
        class_row.addWidget(self.btn_add_roi)
        roi_layout.addLayout(class_row)
        self.btn_add_endmember = QtWidgets.QPushButton("Add current ROI mean as endmember")
        self.btn_add_endmember.clicked.connect(self.on_add_endmember_from_roi)
        roi_layout.addWidget(self.btn_add_endmember)
        tab1_layout.addWidget(roi_group)

        an_group = QtWidgets.QGroupBox("Anomaly Detection")
        an_layout = QtWidgets.QVBoxLayout(an_group)
        self.btn_anomaly = QtWidgets.QPushButton("Compute anomaly map")
        self.btn_anomaly.clicked.connect(self.on_compute_anomaly)
        an_layout.addWidget(self.btn_anomaly)
        an_layout.addWidget(QtWidgets.QLabel("Method: L2 distance from global mean spectrum"))
        tab1_layout.addWidget(an_group)
        tab1_layout.addStretch(0)
        # ---- Integrated ROI Manager panel ----
        roi_manager_group = QtWidgets.QGroupBox("ROI Manager")
        roi_manager_layout = QtWidgets.QVBoxLayout(roi_manager_group)
        roi_manager_layout.setContentsMargins(8, 14, 8, 8)
        roi_manager_layout.setSpacing(5)
        manager_top = QtWidgets.QHBoxLayout()
        self.roi_summary_label = QtWidgets.QLabel("0 ROIs in 0 classes")
        manager_top.addWidget(self.roi_summary_label, 1)
        self.cb_show_outlines = QtWidgets.QCheckBox("Show outlines")
        self.cb_show_outlines.setChecked(True)
        self.cb_show_outlines.toggled.connect(self.on_toggle_outlines)
        manager_top.addWidget(self.cb_show_outlines)
        roi_manager_layout.addLayout(manager_top)
        self.roi_list = QtWidgets.QListWidget()
        self.roi_list.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.roi_list.itemSelectionChanged.connect(self._draw_roi_patch)
        self.roi_list.setMinimumHeight(72)
        self.roi_list.setMaximumHeight(105)
        roi_manager_layout.addWidget(self.roi_list)
        roi_btns = QtWidgets.QGridLayout()
        roi_btns.setSpacing(4)
        self.btn_roi_delete = QtWidgets.QPushButton("Delete selected")
        self.btn_roi_delete.clicked.connect(self.on_delete_rois)
        roi_btns.addWidget(self.btn_roi_delete, 0, 0)
        self.btn_roi_clear = QtWidgets.QPushButton("Clear all")
        self.btn_roi_clear.clicked.connect(self.on_clear_rois)
        roi_btns.addWidget(self.btn_roi_clear, 0, 1)
        self.btn_roi_setclass = QtWidgets.QPushButton("Set class...")
        self.btn_roi_setclass.clicked.connect(self.on_set_roi_class)
        roi_btns.addWidget(self.btn_roi_setclass, 0, 2)
        self.btn_roi_spectra = QtWidgets.QPushButton("Plot ROI spectra")
        self.btn_roi_spectra.clicked.connect(self.on_show_roi_spectra)
        roi_btns.addWidget(self.btn_roi_spectra, 1, 0)
        self.btn_class_spectra = QtWidgets.QPushButton("Plot class means")
        self.btn_class_spectra.clicked.connect(self.on_show_class_spectra)
        roi_btns.addWidget(self.btn_class_spectra, 1, 1)
        self.btn_roi_save = QtWidgets.QPushButton("Save ROIs...")
        self.btn_roi_save.clicked.connect(self.on_save_rois)
        roi_btns.addWidget(self.btn_roi_save, 1, 2)
        self.btn_roi_load = QtWidgets.QPushButton("Load ROIs...")
        self.btn_roi_load.clicked.connect(self.on_load_rois)
        roi_btns.addWidget(self.btn_roi_load, 2, 0, 1, 3)
        roi_manager_layout.addLayout(roi_btns)
        tab1_layout.addWidget(roi_manager_group, 2)

        # ROI tools, anomaly-map controls, and saved ROI management share one workspace.
        self.tabs.addTab(self._scroll_wrap(tab1), "ROI Analysis")

        tab2 = QtWidgets.QWidget()
        tab2_layout = QtWidgets.QHBoxLayout(tab2)
        tab2_layout.setContentsMargins(8, 6, 8, 6)
        tab2_layout.setSpacing(8)
        bm_group = QtWidgets.QGroupBox("Band Math")
        bm_layout = QtWidgets.QGridLayout(bm_group)
        bm_layout.setContentsMargins(8, 14, 8, 8)
        bm_layout.addWidget(QtWidgets.QLabel("Expression (B0, B1, ...):"), 0, 0, 1, 2)
        self.bandmath_edit = QtWidgets.QLineEdit()
        self.bandmath_edit.setPlaceholderText("(B80 - B40) / (B80 + B40 + 1e-6)")
        bm_layout.addWidget(self.bandmath_edit, 1, 0, 1, 2)
        self.btn_bandmath = QtWidgets.QPushButton("Apply expression")
        self.btn_bandmath.clicked.connect(self.on_apply_bandmath)
        bm_layout.addWidget(self.btn_bandmath, 2, 0, 1, 2)
        tab2_layout.addWidget(bm_group, 1)

        cl_group = QtWidgets.QGroupBox("Clustering (k-means)")
        cl_layout = QtWidgets.QGridLayout(cl_group)
        cl_layout.addWidget(QtWidgets.QLabel("Number of clusters (K):"), 0, 0)
        self.spin_k = QtWidgets.QSpinBox()
        self.spin_k.setMinimum(2)
        self.spin_k.setMaximum(20)
        self.spin_k.setValue(5)
        cl_layout.addWidget(self.spin_k, 0, 1)
        cl_layout.addWidget(QtWidgets.QLabel("Cluster on:"), 1, 0)
        self.combo_cluster_mode = QtWidgets.QComboBox()
        self.combo_cluster_mode.addItems(["Full spectra", "First 3 PCs"])
        cl_layout.addWidget(self.combo_cluster_mode, 1, 1)
        self.btn_kmeans = QtWidgets.QPushButton("Run k-means clustering")
        self.btn_kmeans.clicked.connect(self.on_run_kmeans)
        cl_layout.addWidget(self.btn_kmeans, 2, 0, 1, 2)
        if not SKLEARN_AVAILABLE:
            cl_layout.addWidget(QtWidgets.QLabel(
                "(scikit-learn not found – install it to enable k-means.)"
            ), 3, 0, 1, 2)
        tab2_layout.addWidget(cl_group, 1)
        # ---- SNR (merged into Spectral Analysis) ----
        snr_group = QtWidgets.QGroupBox("Signal-to-Noise Ratio (SNR)")
        snr_layout = QtWidgets.QVBoxLayout(snr_group)
        snr_layout.setContentsMargins(8, 14, 8, 8)
        snr_layout.setSpacing(5)
        snr_info = QtWidgets.QLabel(
            "Spatial SNR: mean/std per band. Spectral SNR: mean/std across each pixel spectrum."
        )
        snr_info.setWordWrap(True)
        snr_layout.addWidget(snr_info)
        self.cb_snr_roi = QtWidgets.QCheckBox("Spatial-domain SNR from ROI pixels only (use homogeneous ROIs)")
        snr_layout.addWidget(self.cb_snr_roi)
        self.btn_snr = QtWidgets.QPushButton("Compute SNR")
        self.btn_snr.clicked.connect(self.on_compute_snr)
        snr_layout.addWidget(self.btn_snr)
        snr_btns = QtWidgets.QHBoxLayout()
        self.btn_snr_curve = QtWidgets.QPushButton("Show SNR vs wavelength")
        self.btn_snr_curve.clicked.connect(self.on_show_snr_curve)
        snr_btns.addWidget(self.btn_snr_curve)
        self.btn_snr_map = QtWidgets.QPushButton("Show spectral SNR map")
        self.btn_snr_map.clicked.connect(self.on_show_snr_map)
        snr_btns.addWidget(self.btn_snr_map)
        self.btn_snr_csv = QtWidgets.QPushButton("Save SNR curve (CSV)...")
        self.btn_snr_csv.clicked.connect(self.on_save_snr_csv)
        snr_btns.addWidget(self.btn_snr_csv)
        snr_layout.addLayout(snr_btns)
        self.snr_label = QtWidgets.QLabel("SNR: not computed")
        self.snr_label.setWordWrap(True)
        snr_layout.addWidget(self.snr_label)
        snr_layout.addStretch(1)
        tab2_layout.addWidget(snr_group, 1)
        for _g in (bm_group, cl_group, snr_group):
            _g.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Preferred)
        self.tabs.addTab(self._scroll_wrap(tab2), "Spectral Analysis")

        tab3 = QtWidgets.QWidget()
        tab3_layout = QtWidgets.QHBoxLayout(tab3)
        tab3_layout.setContentsMargins(8, 8, 8, 8)
        tab3_layout.setSpacing(10)
        un_group = QtWidgets.QGroupBox("3. Linear Unmixing (ROI / ASD)")
        un_layout = QtWidgets.QVBoxLayout(un_group)
        self.endmember_label = QtWidgets.QLabel("Endmembers: 0")
        un_layout.addWidget(self.endmember_label)
        self.btn_clear_endmembers = QtWidgets.QPushButton("Clear endmembers")
        self.btn_clear_endmembers.clicked.connect(self.on_clear_endmembers)
        un_layout.addWidget(self.btn_clear_endmembers)
        self.btn_import_asd = QtWidgets.QPushButton("Import ASD field spectra (.asd)...")
        self.btn_import_asd.setToolTip("Field spectra become endmembers for unmixing and SAM")
        self.btn_import_asd.clicked.connect(self.import_asd_dialog)
        un_layout.addWidget(self.btn_import_asd)
        self.btn_asd_to_em = QtWidgets.QPushButton("Add ASD spectra as endmembers")
        self.btn_asd_to_em.clicked.connect(self.on_add_asd_endmembers)
        un_layout.addWidget(self.btn_asd_to_em)
        asd_row = QtWidgets.QHBoxLayout()
        self.btn_asd_plot = QtWidgets.QPushButton("Plot ASD spectra")
        self.btn_asd_plot.clicked.connect(self._plot_asd_library)
        asd_row.addWidget(self.btn_asd_plot)
        self.btn_asd_csv = QtWidgets.QPushButton("Save ASD (CSV)...")
        self.btn_asd_csv.clicked.connect(self.on_save_asd_csv)
        asd_row.addWidget(self.btn_asd_csv)
        un_layout.addLayout(asd_row)
        self.btn_unmix = QtWidgets.QPushButton("Run unmixing (dominant map)")
        self.btn_unmix.clicked.connect(self.on_run_unmixing)
        un_layout.addWidget(self.btn_unmix)
        un_layout.addWidget(self._help_box(
            "1) Enable ROI and select a region\n"
            "2) Add ROI mean as endmember\n"
            "3) Repeat for 2+ materials (or use endmembers extracted in column 2, or imported ASD spectra)\n"
            "4) Run unmixing", title="How to use"))
        un_layout.addStretch(1)

        nmf_group = QtWidgets.QGroupBox("4. Automatic Unmixing (NMF)")
        nmf_layout = QtWidgets.QGridLayout(nmf_group)
        nmf_layout.addWidget(QtWidgets.QLabel("Number:"), 0, 0)
        self.spin_nmf_n = QtWidgets.QSpinBox()
        self.spin_nmf_n.setRange(2, 100)
        self.spin_nmf_n.setValue(5)
        nmf_layout.addWidget(self.spin_nmf_n, 0, 1)
        nmf_layout.addWidget(QtWidgets.QLabel("Seed:"), 0, 2)
        self.spin_nmf_seed = QtWidgets.QSpinBox()
        self.spin_nmf_seed.setRange(0, 9999)
        self.spin_nmf_seed.setValue(42)
        self.spin_nmf_seed.setToolTip("NMF is not unique: a different seed can give a different (equally valid) result.")
        nmf_layout.addWidget(self.spin_nmf_seed, 0, 3)
        nmf_layout.addWidget(QtWidgets.QLabel("Bands:"), 1, 0)
        self.spin_nmf_b0 = QtWidgets.QSpinBox()
        self.spin_nmf_b0.setRange(0, 0)
        nmf_layout.addWidget(self.spin_nmf_b0, 1, 1)
        nmf_layout.addWidget(QtWidgets.QLabel("to:"), 1, 2)
        self.spin_nmf_b1 = QtWidgets.QSpinBox()
        self.spin_nmf_b1.setRange(0, 0)
        nmf_layout.addWidget(self.spin_nmf_b1, 1, 3)
        nmf_layout.addWidget(QtWidgets.QLabel("Fit px (0 = all):"), 2, 0, 1, 2)
        self.spin_nmf_px = QtWidgets.QSpinBox()
        self.spin_nmf_px.setRange(0, 5000000)
        self.spin_nmf_px.setSingleStep(10000)
        self.spin_nmf_px.setValue(100000)
        self.spin_nmf_px.setToolTip("Endmembers are fitted on a random subset of pixels; abundances are then computed for all pixels.")
        nmf_layout.addWidget(self.spin_nmf_px, 2, 2, 1, 2)
        self.btn_nmf = QtWidgets.QPushButton("Run NMF unmixing")
        self.btn_nmf.clicked.connect(self.on_run_nmf)
        nmf_layout.addWidget(self.btn_nmf, 3, 0, 1, 4)
        nmf_layout.addWidget(QtWidgets.QLabel("Abundance of:"), 4, 0, 1, 2)
        self.spin_nmf_show = QtWidgets.QSpinBox()
        self.spin_nmf_show.setRange(1, 1)
        nmf_layout.addWidget(self.spin_nmf_show, 4, 2)
        self.btn_nmf_show = QtWidgets.QPushButton("Show")
        self.btn_nmf_show.clicked.connect(self.on_show_abundance)
        nmf_layout.addWidget(self.btn_nmf_show, 4, 3)
        self.btn_nmf_all = QtWidgets.QPushButton("All maps")
        self.btn_nmf_all.clicked.connect(self.on_show_all_abundance)
        nmf_layout.addWidget(self.btn_nmf_all, 5, 0, 1, 2)
        self.btn_nmf_spec = QtWidgets.QPushButton("Endmember spectra")
        self.btn_nmf_spec.clicked.connect(self.on_plot_endmembers)
        nmf_layout.addWidget(self.btn_nmf_spec, 5, 2, 1, 2)
        nmf_layout.setRowStretch(6, 1)

        # ---- Placeholders: endmember count estimation and PPI-family extraction (algorithms pending) ----
        # Column 1: how many endmembers (HFC / NWHFC)
        hfc_group = QtWidgets.QGroupBox("1. Estimate Number of Endmembers")
        hfc_outer = QtWidgets.QVBoxLayout(hfc_group)
        hfc_outer.setContentsMargins(8, 14, 8, 8)
        hfc_outer.setSpacing(6)
        hfc_outer.addWidget(self._help_box(
            "HFC compares the eigenvalues of the correlation and covariance matrices of the scene; each "
            "eigenvalue where the correlation one is clearly larger indicates one signal source. NWHFC first "
            "whitens the noise. The estimate is a guide: the endmember counts in the next columns "
            "are filled in but can always be edited by hand.",
            "Chang, C.-I. and Q. Du. Estimation of Number of Spectrally Distinct Signal Sources in "
            "Hyperspectral Imagery. IEEE TGRS 42(3), 2004."))
        hfc_box = QtWidgets.QWidget()
        hfc_l = QtWidgets.QFormLayout(hfc_box)
        hfc_l.setContentsMargins(0, 0, 0, 0)
        hfc_l.setRowWrapPolicy(QtWidgets.QFormLayout.WrapLongRows)
        hfc_l.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)
        self.hfc_pfa_spin = QtWidgets.QDoubleSpinBox()
        self.hfc_pfa_spin.setDecimals(10)
        self.hfc_pfa_spin.setRange(1e-10, 0.5)
        self.hfc_pfa_spin.setValue(1e-3)
        self.hfc_pfa_spin.setSingleStep(1e-3)
        self.hfc_pfa_spin.setToolTip("Lower values are stricter and give fewer endmembers.")
        hfc_l.addRow("False-alarm probability:", self.hfc_pfa_spin)
        self.hfc_whiten_check = QtWidgets.QCheckBox("Noise whitening (NWHFC)")
        self.hfc_whiten_check.setChecked(True)
        self.hfc_whiten_check.setToolTip("Ticked: NWHFC. Unticked: plain HFC.")
        hfc_l.addRow(self.hfc_whiten_check)
        self.btn_hfc = QtWidgets.QPushButton("Estimate number of endmembers")
        self.btn_hfc.clicked.connect(self.on_run_hfc)
        hfc_l.addRow(self.btn_hfc)
        self.btn_hfc_plot = QtWidgets.QPushButton("Plot HFC eigenvalue test")
        self.btn_hfc_plot.setEnabled(False)
        self.btn_hfc_plot.clicked.connect(self.on_plot_hfc)
        hfc_l.addRow(self.btn_hfc_plot)
        self.hfc_result_label = QtWidgets.QLabel("Estimated endmembers: not computed")
        self.hfc_result_label.setWordWrap(True)
        hfc_l.addRow(self.hfc_result_label)
        hfc_outer.addWidget(hfc_box)
        hfc_outer.addStretch(1)

        # Column 2: extraction methods, one at a time
        est_group = QtWidgets.QGroupBox("2. Extract Endmembers")
        est_layout = QtWidgets.QVBoxLayout(est_group)
        est_layout.setContentsMargins(8, 14, 8, 8)
        est_layout.setSpacing(6)
        est_layout.addWidget(self._help_box(
            "N-FINDR finds the pixels spanning the largest simplex. PPI projects the pixels onto random "
            "vectors and keeps the pixels that are extreme most often. Fast Iterative PPI starts from "
            "ATGP skewers and repeats until the skewer set no longer changes.",
            "Winter, M. E. N-FINDR, SPIE 3753, 1999. Boardman, Kruse and Green. Mapping target signatures via "
            "partial unmixing of AVIRIS data. 1995. Chang, C.-I. and A. Plaza. A Fast Iterative Algorithm for "
            "Implementation of Pixel Purity Index. IEEE GRSL 3(1), 2006. https://doi.org/10.1109/LGRS.2005.856701"))
        self.extract_method_combo = QtWidgets.QComboBox()
        self.extract_method_combo.addItems(["N-FINDR", "Pixel Purity Index (PPI)", "Fast Iterative PPI (FIPPI)"])
        est_layout.addWidget(self.extract_method_combo)
        self.extract_stack = QtWidgets.QStackedWidget()
        self.extract_method_combo.currentIndexChanged.connect(self.extract_stack.setCurrentIndex)
        est_layout.addWidget(self.extract_stack)

        nf_box = QtWidgets.QWidget()
        nf_l = QtWidgets.QFormLayout(nf_box)
        nf_l.setContentsMargins(0, 0, 0, 0)
        nf_l.setRowWrapPolicy(QtWidgets.QFormLayout.WrapLongRows)
        nf_l.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)
        self.nfindr_n_spin = QtWidgets.QSpinBox()
        self.nfindr_n_spin.setRange(2, 100)
        self.nfindr_n_spin.setValue(5)
        nf_l.addRow("Number of endmembers:", self.nfindr_n_spin)
        self.nfindr_iter_spin = QtWidgets.QSpinBox()
        self.nfindr_iter_spin.setRange(1, 100)
        self.nfindr_iter_spin.setValue(3)
        nf_l.addRow("Max iterations:", self.nfindr_iter_spin)
        self.btn_nfindr = QtWidgets.QPushButton("Run N-FINDR")
        self.btn_nfindr.clicked.connect(self.on_run_nfindr)
        nf_l.addRow(self.btn_nfindr)
        self.btn_nfindr_add = QtWidgets.QPushButton("Use as endmembers for linear unmixing")
        self.btn_nfindr_add.setEnabled(False)
        self.btn_nfindr_add.clicked.connect(self.on_nfindr_to_endmembers)
        nf_l.addRow(self.btn_nfindr_add)
        self.extract_stack.addWidget(nf_box)
        self.nfindr_endmembers = None

        ppi_box = QtWidgets.QWidget()
        ppi_l = QtWidgets.QFormLayout(ppi_box)
        ppi_l.setContentsMargins(0, 0, 0, 0)
        ppi_l.setRowWrapPolicy(QtWidgets.QFormLayout.WrapLongRows)
        ppi_l.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)
        self.ppi_n_spin = QtWidgets.QSpinBox()
        self.ppi_n_spin.setRange(2, 100)
        self.ppi_n_spin.setValue(5)
        ppi_l.addRow("Number of endmembers:", self.ppi_n_spin)
        self.ppi_vectors_spin = QtWidgets.QSpinBox()
        self.ppi_vectors_spin.setRange(1, 10000000)
        self.ppi_vectors_spin.setValue(104)
        self.ppi_vectors_spin.setToolTip("Number of random unit vectors (skewers) the pixels are projected onto.")
        ppi_l.addRow("Number of skewers:", self.ppi_vectors_spin)
        self.ppi_reduction_combo = QtWidgets.QComboBox()
        self.ppi_reduction_combo.addItems(["MNF", "PCA"])
        self.ppi_reduction_combo.setToolTip("Dimensionality reduction before the projections (MNF is the default).")
        ppi_l.addRow("Preprocessing:", self.ppi_reduction_combo)
        self.ppi_seed_spin = QtWidgets.QSpinBox()
        self.ppi_seed_spin.setRange(0, 999999)
        self.ppi_seed_spin.setValue(0)
        self.ppi_seed_spin.setToolTip("PPI uses random skewers, so another seed can give slightly different endmembers.")
        ppi_l.addRow("Random seed:", self.ppi_seed_spin)
        self.btn_ppi = QtWidgets.QPushButton("Run PPI")
        self.btn_ppi.clicked.connect(self.on_run_ppi)
        ppi_l.addRow(self.btn_ppi)
        self.btn_ppi_add = QtWidgets.QPushButton("Use as endmembers for linear unmixing")
        self.btn_ppi_add.setEnabled(False)
        self.btn_ppi_add.clicked.connect(self.on_ppi_to_endmembers)
        ppi_l.addRow(self.btn_ppi_add)
        self.ppi_endmembers = None
        self.extract_stack.addWidget(ppi_box)

        fippi_box = QtWidgets.QWidget()
        fippi_l = QtWidgets.QFormLayout(fippi_box)
        fippi_l.setContentsMargins(0, 0, 0, 0)
        fippi_l.setRowWrapPolicy(QtWidgets.QFormLayout.WrapLongRows)
        fippi_l.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)
        self.fippi_n_spin = QtWidgets.QSpinBox()
        self.fippi_n_spin.setRange(2, 100)
        self.fippi_n_spin.setValue(5)
        fippi_l.addRow("Number of endmembers:", self.fippi_n_spin)
        self.fippi_reduction_combo = QtWidgets.QComboBox()
        self.fippi_reduction_combo.addItems(["MNF", "PCA"])
        self.fippi_reduction_combo.setToolTip("Dimensionality reduction before the pixel purity counting (MNF is the default).")
        fippi_l.addRow("Preprocessing:", self.fippi_reduction_combo)
        self.fippi_iter_spin = QtWidgets.QSpinBox()
        self.fippi_iter_spin.setRange(1, 1000)
        self.fippi_iter_spin.setValue(50)
        self.fippi_iter_spin.setToolTip("Upper limit; the iteration normally stops earlier when the skewer set stops changing.")
        fippi_l.addRow("Max iterations:", self.fippi_iter_spin)
        self.btn_fippi = QtWidgets.QPushButton("Run Fast Iterative PPI")
        self.btn_fippi.clicked.connect(self.on_run_fippi)
        fippi_l.addRow(self.btn_fippi)
        self.btn_fippi_add = QtWidgets.QPushButton("Use as endmembers for linear unmixing")
        self.btn_fippi_add.setEnabled(False)
        self.btn_fippi_add.clicked.connect(self.on_fippi_to_endmembers)
        fippi_l.addRow(self.btn_fippi_add)
        self.fippi_endmembers = None
        self.extract_stack.addWidget(fippi_box)
        est_layout.addStretch(1)

        # four equal-width columns, left to right = workflow order
        for _g in (hfc_group, est_group, un_group, nmf_group):
            _g.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Preferred)
            tab3_layout.addWidget(_g, 1)
        self.tabs.addTab(self._scroll_wrap(tab3), "Unmixing")

        tab4 = QtWidgets.QWidget()
        tab4_layout = QtWidgets.QHBoxLayout(tab4)
        tab4_layout.setContentsMargins(8, 6, 8, 6)
        pca_group = QtWidgets.QGroupBox("PCA / Dimensionality Reduction")
        pca_layout = QtWidgets.QVBoxLayout(pca_group)
        pca_layout.setContentsMargins(8, 14, 8, 8)
        pca_layout.setSpacing(5)
        comp_layout = QtWidgets.QHBoxLayout()
        comp_layout.addWidget(QtWidgets.QLabel("Compute first N PCs:"))
        self.spin_pca_n = QtWidgets.QSpinBox()
        self.spin_pca_n.setMinimum(3)
        self.spin_pca_n.setMaximum(20)
        self.spin_pca_n.setValue(6)
        comp_layout.addWidget(self.spin_pca_n)
        pca_layout.addLayout(comp_layout)
        self.btn_compute_pca = QtWidgets.QPushButton("Compute PCA")
        self.btn_compute_pca.clicked.connect(self.on_compute_pca)
        pca_layout.addWidget(self.btn_compute_pca)
        pca_btns = QtWidgets.QGridLayout()
        self.btn_show_pc1 = QtWidgets.QPushButton("Show PC1")
        self.btn_show_pc1.clicked.connect(lambda: self.on_show_pc(0))
        pca_btns.addWidget(self.btn_show_pc1, 0, 0)
        self.btn_show_pc2 = QtWidgets.QPushButton("Show PC2")
        self.btn_show_pc2.clicked.connect(lambda: self.on_show_pc(1))
        pca_btns.addWidget(self.btn_show_pc2, 0, 1)
        self.btn_show_pc3 = QtWidgets.QPushButton("Show PC3")
        self.btn_show_pc3.clicked.connect(lambda: self.on_show_pc(2))
        pca_btns.addWidget(self.btn_show_pc3, 0, 2)
        self.btn_show_pca_rgb = QtWidgets.QPushButton("Show PCA RGB (PC1/PC2/PC3)")
        self.btn_show_pca_rgb.clicked.connect(self.on_show_pca_rgb)
        pca_btns.addWidget(self.btn_show_pca_rgb, 1, 0, 1, 2)
        self.btn_show_scree = QtWidgets.QPushButton("Show Scree Plot")
        self.btn_show_scree.clicked.connect(self.on_show_scree)
        pca_btns.addWidget(self.btn_show_scree, 1, 2)
        pca_layout.addLayout(pca_btns)
        self.pca_label = QtWidgets.QLabel("PCA: not computed")
        pca_layout.addWidget(self.pca_label)
        tab4_layout.addWidget(pca_group)
        self.tabs.addTab(tab4, "PCA")

        # ---- Classification tab ----
        tab6 = QtWidgets.QWidget()
        tab6_layout = QtWidgets.QHBoxLayout(tab6)

        sam_group = QtWidgets.QGroupBox("Spectral Angle Mapper (SAM)")
        sam_layout = QtWidgets.QGridLayout(sam_group)
        sam_layout.addWidget(QtWidgets.QLabel("Reference spectra:"), 0, 0)
        self.combo_sam_source = QtWidgets.QComboBox()
        self.combo_sam_source.addItems(
            ["ROI classes (mean per class)", "Endmember list (Unmixing tab)"]
        )
        sam_layout.addWidget(self.combo_sam_source, 0, 1)
        sam_layout.addWidget(QtWidgets.QLabel("Max angle (deg, 0 = none):"), 1, 0)
        self.spin_sam_thr = QtWidgets.QDoubleSpinBox()
        self.spin_sam_thr.setRange(0.0, 90.0)
        self.spin_sam_thr.setDecimals(2)
        self.spin_sam_thr.setSingleStep(0.5)
        self.spin_sam_thr.setValue(0.0)
        sam_layout.addWidget(self.spin_sam_thr, 1, 1)
        self.btn_sam = QtWidgets.QPushButton("Run SAM classification")
        self.btn_sam.clicked.connect(self.on_run_sam)
        sam_layout.addWidget(self.btn_sam, 2, 0, 1, 2)
        self.btn_sam_angle = QtWidgets.QPushButton("Show best-match angle map")
        self.btn_sam_angle.clicked.connect(self.on_show_sam_angle)
        sam_layout.addWidget(self.btn_sam_angle, 3, 0, 1, 2)
        sam_layout.setRowStretch(4, 1)
        tab6_layout.addWidget(sam_group, 1)

        sup_group = QtWidgets.QGroupBox("Supervised classification (labelled ROIs)")
        sup_layout = QtWidgets.QGridLayout(sup_group)
        sup_layout.addWidget(QtWidgets.QLabel("Classifier:"), 0, 0)
        self.combo_clf = QtWidgets.QComboBox()
        if SKLEARN_AVAILABLE:
            self.combo_clf.addItems(list(CLASSIFIERS.keys()))
        sup_layout.addWidget(self.combo_clf, 0, 1)
        sup_layout.addWidget(QtWidgets.QLabel("Features:"), 1, 0)
        self.combo_features = QtWidgets.QComboBox()
        self.combo_features.addItems(["Full spectra", "PCA components (compute in PCA tab)"])
        sup_layout.addWidget(self.combo_features, 1, 1)
        sup_layout.addWidget(QtWidgets.QLabel("Validation split:"), 2, 0)
        self.combo_split = QtWidgets.QComboBox()
        self.combo_split.addItems(["Hold out whole ROIs", "Random pixels"])
        sup_layout.addWidget(self.combo_split, 2, 1)
        sup_layout.addWidget(QtWidgets.QLabel("Test fraction (%):"), 3, 0)
        self.spin_test = QtWidgets.QSpinBox()
        self.spin_test.setRange(0, 50)
        self.spin_test.setValue(30)
        sup_layout.addWidget(self.spin_test, 3, 1)
        sup_layout.addWidget(QtWidgets.QLabel("Max pixels per class:"), 4, 0)
        self.spin_max_pc = QtWidgets.QSpinBox()
        self.spin_max_pc.setRange(100, 200000)
        self.spin_max_pc.setSingleStep(500)
        self.spin_max_pc.setValue(5000)
        sup_layout.addWidget(self.spin_max_pc, 4, 1)
        self.btn_train = QtWidgets.QPushButton("Train and classify image")
        self.btn_train.clicked.connect(self.on_train_classifier)
        sup_layout.addWidget(self.btn_train, 5, 0)
        self.btn_report = QtWidgets.QPushButton("Show accuracy report")
        self.btn_report.clicked.connect(self.on_show_report)
        sup_layout.addWidget(self.btn_report, 5, 1)
        sup_layout.addWidget(QtWidgets.QLabel("Cross-validation folds:"), 6, 0)
        self.spin_cv_k = QtWidgets.QSpinBox()
        self.spin_cv_k.setRange(2, 10)
        self.spin_cv_k.setValue(5)
        sup_layout.addWidget(self.spin_cv_k, 6, 1)
        self.btn_cv = QtWidgets.QPushButton("Run k-fold cross-validation (accuracy only)")
        self.btn_cv.clicked.connect(self.on_crossval)
        sup_layout.addWidget(self.btn_cv, 7, 0, 1, 2)
        if not SKLEARN_AVAILABLE:
            sup_layout.addWidget(QtWidgets.QLabel(
                "(scikit-learn not found - install it to enable this.)"
            ), 8, 0, 1, 2)
        tab6_layout.addWidget(sup_group, 1)
        self.tabs.addTab(self._scroll_wrap(tab6), "Classification")

        # ---- 2D CNN tab ----
        tab8 = QtWidgets.QWidget()
        tab8_layout = QtWidgets.QHBoxLayout(tab8)
        cnn_group = QtWidgets.QGroupBox("2D CNN on PCA patches (trained on labelled ROIs)")
        cnn_layout = QtWidgets.QGridLayout(cnn_group)
        cnn_layout.addWidget(QtWidgets.QLabel("PCA components (whitened):"), 0, 0)
        self.spin_cnn_pca = QtWidgets.QSpinBox()
        self.spin_cnn_pca.setRange(3, 100)
        self.spin_cnn_pca.setValue(30)
        cnn_layout.addWidget(self.spin_cnn_pca, 0, 1)
        cnn_layout.addWidget(QtWidgets.QLabel("Patch size (odd):"), 1, 0)
        self.spin_cnn_patch = QtWidgets.QSpinBox()
        self.spin_cnn_patch.setRange(9, 31)
        self.spin_cnn_patch.setSingleStep(2)
        self.spin_cnn_patch.setValue(15)
        cnn_layout.addWidget(self.spin_cnn_patch, 1, 1)
        cnn_layout.addWidget(QtWidgets.QLabel("Epochs:"), 2, 0)
        self.spin_cnn_epochs = QtWidgets.QSpinBox()
        self.spin_cnn_epochs.setRange(1, 500)
        self.spin_cnn_epochs.setValue(20)
        cnn_layout.addWidget(self.spin_cnn_epochs, 2, 1)
        cnn_layout.addWidget(QtWidgets.QLabel("Batch size:"), 3, 0)
        self.spin_cnn_batch = QtWidgets.QSpinBox()
        self.spin_cnn_batch.setRange(8, 1024)
        self.spin_cnn_batch.setValue(64)
        cnn_layout.addWidget(self.spin_cnn_batch, 3, 1)
        cnn_layout.addWidget(QtWidgets.QLabel("Network:"), 4, 0)
        self.combo_cnn_net = QtWidgets.QComboBox()
        self.combo_cnn_net.addItems(["Reference (64-128-256-512)", "Compact (32-64-128, faster)"])
        cnn_layout.addWidget(self.combo_cnn_net, 4, 1)

        cnn_layout.addWidget(QtWidgets.QLabel("Validation split:"), 0, 2)
        self.combo_cnn_split = QtWidgets.QComboBox()
        self.combo_cnn_split.addItems(["Hold out whole ROIs", "Random pixels"])
        cnn_layout.addWidget(self.combo_cnn_split, 0, 3)
        cnn_layout.addWidget(QtWidgets.QLabel("Test fraction (%):"), 1, 2)
        self.spin_cnn_test = QtWidgets.QSpinBox()
        self.spin_cnn_test.setRange(0, 50)
        self.spin_cnn_test.setValue(30)
        cnn_layout.addWidget(self.spin_cnn_test, 1, 3)
        cnn_layout.addWidget(QtWidgets.QLabel("Max patches per class:"), 2, 2)
        self.spin_cnn_maxpc = QtWidgets.QSpinBox()
        self.spin_cnn_maxpc.setRange(100, 200000)
        self.spin_cnn_maxpc.setSingleStep(500)
        self.spin_cnn_maxpc.setValue(3000)
        cnn_layout.addWidget(self.spin_cnn_maxpc, 2, 3)
        cnn_layout.addWidget(QtWidgets.QLabel("Classify:"), 3, 2)
        self.combo_cnn_scope = QtWidgets.QComboBox()
        self.combo_cnn_scope.addItems(["Whole image (slow on large images)", "Labelled ROI pixels only"])
        cnn_layout.addWidget(self.combo_cnn_scope, 3, 3)
        self.btn_cnn = QtWidgets.QPushButton("Train 2D CNN and classify")
        self.btn_cnn.clicked.connect(self.on_train_cnn)
        cnn_layout.addWidget(self.btn_cnn, 4, 2)
        self.btn_cnn_report = QtWidgets.QPushButton("Show report / training log")
        self.btn_cnn_report.clicked.connect(self.on_show_report)
        cnn_layout.addWidget(self.btn_cnn_report, 4, 3)
        note = ("Needs TensorFlow (pip install tensorflow)." if not TF_AVAILABLE else
                "Training runs in the background; use 'Stop task' (status bar) to cancel.")
        cnn_note = QtWidgets.QLabel(note)
        cnn_note.setWordWrap(True)
        cnn_layout.addWidget(cnn_note, 5, 0, 1, 4)
        cnn_layout.setRowStretch(6, 1)
        tab8_layout.addWidget(cnn_group)
        self.tabs.addTab(self._scroll_wrap(tab8), "2D CNN")

        # ---- Band Management: three compact side-by-side columns ----
        band_tab = QtWidgets.QWidget()
        band_layout = QtWidgets.QHBoxLayout(band_tab)
        band_layout.setContentsMargins(8, 8, 8, 8)
        band_layout.setSpacing(10)

        # Column 0: wavelengths for any file format
        wl_group = QtWidgets.QGroupBox("0. Wavelengths (any file format)")
        wl_layout = QtWidgets.QVBoxLayout(wl_group)
        wl_layout.setContentsMargins(8, 14, 8, 8)
        wl_layout.setSpacing(5)
        self.band_wl_source_label = QtWidgets.QLabel("Load a hyperspectral cube first")
        self.band_wl_source_label.setWordWrap(True)
        wl_layout.addWidget(self.band_wl_source_label)
        info = QtWidgets.QLabel(
            "Only some formats store wavelengths (ENVI, many HDF / NetCDF / MAT files). For GeoTIFF, "
            "NumPy, MAT or any file without them, assign the wavelengths here: from a text / CSV / .hdr "
            "file, as a uniform range, or by pasting a list. The other Band Management tools then work "
            "on any cube.")
        info.setWordWrap(True)
        wl_layout.addWidget(info)
        self.btn_assign_wl = QtWidgets.QPushButton("Assign / replace wavelengths...")
        self.btn_assign_wl.clicked.connect(self.on_assign_wavelengths)
        wl_layout.addWidget(self.btn_assign_wl)
        wl_layout.addStretch(1)
        band_layout.addWidget(wl_group, 1)

        # Column 1: Remove bands
        remove_group = QtWidgets.QGroupBox("1. Remove Spectral Bands")
        remove_layout = QtWidgets.QVBoxLayout(remove_group)
        remove_layout.setContentsMargins(8, 14, 8, 8)
        remove_layout.setSpacing(5)
        remove_layout.addWidget(QtWidgets.QLabel("Current wavelength range:"))
        self.band_wavelength_range_label = QtWidgets.QLabel("Load a hyperspectral cube first")
        self.band_wavelength_range_label.setWordWrap(True)
        self.band_wavelength_range_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        remove_layout.addWidget(self.band_wavelength_range_label)
        remove_layout.addWidget(QtWidgets.QLabel("Wavelength intervals to remove (nm):"))
        self.remove_wavelength_ranges_edit = QtWidgets.QLineEdit("410 450; 620 850")
        self.remove_wavelength_ranges_edit.setPlaceholderText("410 450; 620 850")
        self.remove_wavelength_ranges_edit.setToolTip(
            "Enter inclusive wavelength intervals separated by semicolons or new lines. Example: 410 450; 620 850"
        )
        remove_layout.addWidget(self.remove_wavelength_ranges_edit)
        self.btn_remove_bands = QtWidgets.QPushButton("Remove Bands")
        self.btn_remove_bands.clicked.connect(self.on_remove_bands)
        remove_layout.addWidget(self.btn_remove_bands)
        self.btn_plot_band_removal = QtWidgets.QPushButton("Plot Original vs New Wavelengths")
        self.btn_plot_band_removal.clicked.connect(self.on_plot_band_removal)
        self.btn_plot_band_removal.setEnabled(False)
        self.btn_plot_band_removal.setToolTip("Compare the wavelength grid before and after removing intervals")
        remove_layout.addWidget(self.btn_plot_band_removal)
        self.band_removal_status_label = QtWidgets.QLabel(
            "Intervals are inclusive. Needs wavelengths (assign them in column 0 if the file has none)."
        )
        self.band_removal_status_label.setWordWrap(True)
        remove_layout.addWidget(self.band_removal_status_label)
        remove_layout.addStretch(1)
        band_layout.addWidget(remove_group, 1)

        # Column 2: Resample bands
        resample_group = QtWidgets.QGroupBox("2. Resample Spectral Bands")
        resample_layout = QtWidgets.QVBoxLayout(resample_group)
        resample_layout.setContentsMargins(8, 14, 8, 8)
        resample_layout.setSpacing(5)
        resample_layout.addWidget(QtWidgets.QLabel("Target wavelength spacing (nm):"))
        self.resample_spacing_spin = QtWidgets.QSpinBox()
        self.resample_spacing_spin.setRange(1, 10000)
        self.resample_spacing_spin.setValue(10)
        self.resample_spacing_spin.setToolTip("Positive integer spacing for the uniform wavelength grid")
        resample_layout.addWidget(self.resample_spacing_spin)
        resample_layout.addWidget(QtWidgets.QLabel("Interpolation method:"))
        self.resample_method_combo = QtWidgets.QComboBox()
        self.resample_method_combo.addItems(["nearest", "bilinear", "cubic"])
        resample_layout.addWidget(self.resample_method_combo)
        self.btn_resample_bands = QtWidgets.QPushButton("Resample Bands")
        self.btn_resample_bands.clicked.connect(self.on_resample_bands)
        resample_layout.addWidget(self.btn_resample_bands)
        self.btn_plot_resample = QtWidgets.QPushButton("Plot Original vs Resampled Grid")
        self.btn_plot_resample.clicked.connect(self.on_plot_resampled_bands)
        self.btn_plot_resample.setEnabled(False)
        resample_layout.addWidget(self.btn_plot_resample)
        self.resample_status_label = QtWidgets.QLabel(
            "Interpolate every pixel spectrum onto a uniform wavelength grid."
        )
        self.resample_status_label.setWordWrap(True)
        resample_layout.addWidget(self.resample_status_label)
        resample_layout.addStretch(1)
        band_layout.addWidget(resample_group, 1)

        # Column 3: Select informative bands
        select_group = QtWidgets.QGroupBox("3. Select Informative Bands")
        select_layout = QtWidgets.QVBoxLayout(select_group)
        select_layout.setContentsMargins(8, 14, 8, 8)
        select_layout.setSpacing(5)
        select_layout.addWidget(QtWidgets.QLabel("Endmembers to estimate:"))
        self.select_endmembers_spin = QtWidgets.QSpinBox()
        self.select_endmembers_spin.setRange(2, 50)
        self.select_endmembers_spin.setValue(9)
        select_layout.addWidget(self.select_endmembers_spin)
        select_layout.addWidget(QtWidgets.QLabel("Number of bands to retain:"))
        self.select_nbands_spin = QtWidgets.QSpinBox()
        self.select_nbands_spin.setRange(1, 5000)
        self.select_nbands_spin.setValue(10)
        select_layout.addWidget(self.select_nbands_spin)
        self.btn_select_bands = QtWidgets.QPushButton("Estimate Endmembers & Select Bands")
        self.btn_select_bands.setToolTip("Estimate representative spectra, then select informative wavelengths")
        self.btn_select_bands.clicked.connect(self.on_select_bands)
        self.btn_select_bands.setMinimumHeight(42)
        self.btn_select_bands.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        select_layout.addWidget(self.btn_select_bands)
        self.btn_plot_selected_bands = QtWidgets.QPushButton("Plot Selected Wavelengths")
        self.btn_plot_selected_bands.clicked.connect(self.on_plot_selected_bands)
        self.btn_plot_selected_bands.setEnabled(False)
        select_layout.addWidget(self.btn_plot_selected_bands)
        self.select_status_label = QtWidgets.QLabel(
            "Estimate representative pixel spectra and select bands with strong spectral contrast."
        )
        self.select_status_label.setWordWrap(True)
        select_layout.addWidget(self.select_status_label)
        select_layout.addStretch(1)
        band_layout.addWidget(select_group, 1)

        self.tabs.addTab(self._scroll_wrap(band_tab), "Band Management")

        # ---- Data Correction: radiometric calibration is functional; other panels remain placeholders ----
        correction_tab = QtWidgets.QWidget()
        correction_layout = QtWidgets.QHBoxLayout(correction_tab)
        correction_layout.setContentsMargins(8, 8, 8, 8)
        correction_layout.setSpacing(10)

        radiometric_group = QtWidgets.QGroupBox("1. Radiometric Calibration")
        radiometric_layout = QtWidgets.QVBoxLayout(radiometric_group)
        radiometric_layout.setContentsMargins(8, 14, 8, 8)
        radiometric_layout.setSpacing(6)
        radiometric_layout.addWidget(self._help_box(
            "Convert DN to TOA radiance or reflectance, or convert radiance to TOA reflectance. "
            "Gain / offset / solar-irradiance values come from the file metadata when present and can "
            "be edited or typed in below for any file format."))
        self.radiometric_badbands_label = QtWidgets.QLabel("Bad bands: load a hyperspectral cube first")
        self.radiometric_badbands_label.setWordWrap(True)
        self.radiometric_badbands_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        radiometric_layout.addWidget(self.radiometric_badbands_label)
        self.radiometric_stage_label = QtWidgets.QLabel("Current data stage: DN / unknown")
        self.radiometric_stage_label.setWordWrap(True)
        radiometric_layout.addWidget(self.radiometric_stage_label)
        radiometric_layout.addWidget(QtWidgets.QLabel("Select conversion function:"))
        self.radiometric_function_combo = QtWidgets.QComboBox()
        self.radiometric_function_combo.addItems([
            "DN → Radiance (Gain, Bias)",
            "DN → Reflectance (RGain, ROffset)",
            "Radiance → Reflectance (ESUN, solar geometry)",
        ])
        self.radiometric_function_combo.currentIndexChanged.connect(self._on_radiometric_function_changed)
        radiometric_layout.addWidget(self.radiometric_function_combo)

        self.radiometric_parameter_stack = QtWidgets.QStackedWidget()
        self.radiometric_gain_edit = QtWidgets.QLineEdit()
        self.radiometric_bias_edit = QtWidgets.QLineEdit()
        self.radiometric_rgain_edit = QtWidgets.QLineEdit()
        self.radiometric_roffset_edit = QtWidgets.QLineEdit()
        self.radiometric_esun_edit = QtWidgets.QLineEdit()
        self.radiometric_distance_edit = QtWidgets.QLineEdit()
        self.radiometric_elevation_edit = QtWidgets.QLineEdit()

        def parameter_page(fields):
            page = QtWidgets.QWidget()
            form = QtWidgets.QFormLayout(page)
            form.setContentsMargins(2, 4, 2, 4)
            form.setSpacing(5)
            for label, editor, hint in fields:
                editor.setPlaceholderText(hint)
                editor.setToolTip("Enter one value per band for vector parameters, separated by commas or spaces; scalar parameters take one value.")
                form.addRow(label, editor)
            return page

        self.radiometric_parameter_stack.addWidget(parameter_page([
            ("Gain (per band)", self.radiometric_gain_edit, "e.g. 0.01, 0.01, ..."),
            ("Bias (per band)", self.radiometric_bias_edit, "e.g. 0, 0, ..."),
        ]))
        self.radiometric_parameter_stack.addWidget(parameter_page([
            ("RGain (per band)", self.radiometric_rgain_edit, "e.g. 0.0001, 0.0001, ..."),
            ("ROffset (per band)", self.radiometric_roffset_edit, "e.g. 0, 0, ..."),
        ]))
        self.radiometric_parameter_stack.addWidget(parameter_page([
            ("ESUN (per band)", self.radiometric_esun_edit, "solar irradiance values"),
            ("Earth–Sun distance (AU)", self.radiometric_distance_edit, "e.g. 1.0"),
            ("Sun elevation (degrees)", self.radiometric_elevation_edit, "e.g. 45"),
        ]))
        radiometric_layout.addWidget(self.radiometric_parameter_stack)
        self.btn_apply_radiometric = QtWidgets.QPushButton("Apply Selected Calibration")
        self.btn_apply_radiometric.clicked.connect(self.on_apply_radiometric_function)
        self.btn_apply_radiometric.setMinimumHeight(34)
        radiometric_layout.addWidget(self.btn_apply_radiometric)
        preview_row = QtWidgets.QHBoxLayout()
        preview_row.addWidget(QtWidgets.QLabel("Band number:"))
        self.radiometric_band_spin = QtWidgets.QSpinBox()
        self.radiometric_band_spin.setRange(1, 1)
        self.radiometric_band_spin.setValue(1)
        preview_row.addWidget(self.radiometric_band_spin)
        radiometric_layout.addLayout(preview_row)
        self.btn_radiometric_preview = QtWidgets.QPushButton("Compare Input / Output Band")
        self.btn_radiometric_preview.clicked.connect(self.on_radiometric_preview)
        self.btn_radiometric_preview.setEnabled(False)
        radiometric_layout.addWidget(self.btn_radiometric_preview)
        self.radiometric_status_label = QtWidgets.QLabel("Load ENVI data with radiometric calibration metadata.")
        self.radiometric_status_label.setWordWrap(True)
        radiometric_layout.addWidget(self.radiometric_status_label)
        radiometric_layout.addStretch(1)
        correction_layout.addWidget(radiometric_group, 1)

        # ---- 2. Atmospheric Correction (one page per method) ----
        self.dark_file_array = None
        self.dark_file_name = ""
        atm_group = QtWidgets.QGroupBox("2. Atmospheric Correction")
        atm_layout = QtWidgets.QVBoxLayout(atm_group)
        atm_layout.setContentsMargins(8, 14, 8, 8)
        atm_layout.setSpacing(5)
        method_row = QtWidgets.QHBoxLayout()
        method_row.addWidget(QtWidgets.QLabel("Method:"))
        self.atm_method_combo = QtWidgets.QComboBox()
        self.atm_method_combo.addItems([
            "Empirical line", "Dark pixel subtraction", "Flat field", "IARR",
        ])                                                    # more methods: add a page below
        method_row.addWidget(self.atm_method_combo, 1)
        atm_layout.addLayout(method_row)
        self.atm_stack = QtWidgets.QStackedWidget()
        self.atm_method_combo.currentIndexChanged.connect(self.atm_stack.setCurrentIndex)
        atm_layout.addWidget(self.atm_stack, 1)

        def atm_page():
            page = QtWidgets.QWidget()
            lay = QtWidgets.QVBoxLayout(page)
            lay.setContentsMargins(0, 2, 0, 0)
            lay.setSpacing(5)
            self.atm_stack.addWidget(page)
            return lay

        def info_label(text, tip=None):
            lab = QtWidgets.QLabel(text)
            lab.setWordWrap(True)
            if tip:
                lab.setToolTip(tip)
            return lab

        help_box = self._help_box

        # -- page 0: empirical line --
        lay = atm_page()
        lay.addWidget(help_box(
            "Empirical line: reflectance = gain x measured + offset, fitted per band from reference "
            "materials of known reflectance (field spectra). The measured value can be DN, TOA radiance "
            "or TOA reflectance. 2+ references: linear regression; 1 reference: offset = 0.",
            "Souri, A. H. and M. A. Sharifi. Evaluation of Scene-Based Empirical Approaches for Atmospheric "
            "Correction of Hyperspectral Imagery. 33rd Asian Conference on Remote Sensing, 2012."))
        lay.addWidget(QtWidgets.QLabel("Reference materials (image ROI + field spectrum):"))
        self.el_list = QtWidgets.QListWidget()
        self.el_list.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.el_list.setMinimumHeight(70)
        self.el_list.setMaximumHeight(110)
        lay.addWidget(self.el_list)
        el_btns = QtWidgets.QHBoxLayout()
        self.btn_el_add = QtWidgets.QPushButton("Add reference...")
        self.btn_el_add.setToolTip("Pick a saved ROI, then the field spectrum (CSV / TXT: wavelength, reflectance).")
        self.btn_el_add.clicked.connect(self.on_el_add_reference)
        el_btns.addWidget(self.btn_el_add)
        self.btn_el_remove = QtWidgets.QPushButton("Remove")
        self.btn_el_remove.clicked.connect(self.on_el_remove_reference)
        el_btns.addWidget(self.btn_el_remove)
        self.btn_el_clear = QtWidgets.QPushButton("Clear")
        self.btn_el_clear.clicked.connect(self.on_el_clear_references)
        el_btns.addWidget(self.btn_el_clear)
        lay.addLayout(el_btns)
        units_row = QtWidgets.QHBoxLayout()
        units_row.addWidget(QtWidgets.QLabel("Field reflectance units:"))
        self.el_units_combo = QtWidgets.QComboBox()
        self.el_units_combo.addItems(["Fraction (0-1)", "Percent (0-100)"])
        units_row.addWidget(self.el_units_combo, 1)
        lay.addLayout(units_row)
        self.btn_el_apply = QtWidgets.QPushButton("Apply Empirical Line")
        self.btn_el_apply.setMinimumHeight(34)
        self.btn_el_apply.clicked.connect(self.on_el_apply)
        lay.addWidget(self.btn_el_apply)
        self.btn_el_plot_gain = QtWidgets.QPushButton("Plot gain / offset")
        self.btn_el_plot_gain.setEnabled(False)
        self.btn_el_plot_gain.clicked.connect(self.on_el_plot_gain_offset)
        lay.addWidget(self.btn_el_plot_gain)
        self.btn_el_plot_refs = QtWidgets.QPushButton("Plot corrected vs field spectra")
        self.btn_el_plot_refs.setEnabled(False)
        self.btn_el_plot_refs.clicked.connect(self.on_el_plot_references)
        lay.addWidget(self.btn_el_plot_refs)
        self.el_status_label = info_label("Draw ROIs on reference targets (ROI Analysis tab), then add their field spectra.")
        lay.addWidget(self.el_status_label)
        lay.addStretch(1)

        # -- page 1: dark pixel subtraction --
        lay = atm_page()
        lay.addWidget(help_box(
            "Dark pixel subtraction: subtracts a dark value (the atmospheric path-radiance / haze "
            "contribution) from the cube. The value can be one number for all bands, one value per "
            "band, or a per-pixel array.",
            "Souri, A. H. and M. A. Sharifi. Evaluation of Scene-Based Empirical Approaches for Atmospheric "
            "Correction of Hyperspectral Imagery. 33rd Asian Conference on Remote Sensing, 2012."))
        lay.addWidget(QtWidgets.QLabel("Dark value source:"))
        self.dark_mode_combo = QtWidgets.QComboBox()
        self.dark_mode_combo.addItems([
            "Scalar value (all bands)",
            "Dark ROI(s) selected in ROI Manager (mean per band)",
            "Darkest scene pixels (percentile per band)",
            "From file (.npy any shape, or .csv/.txt scalar / per-band)",
        ])
        self.dark_mode_combo.currentIndexChanged.connect(self._on_dark_mode_changed)
        lay.addWidget(self.dark_mode_combo)
        dark_form = QtWidgets.QFormLayout()
        dark_form.setSpacing(4)
        self.dark_scalar_spin = QtWidgets.QDoubleSpinBox()
        self.dark_scalar_spin.setDecimals(6)
        self.dark_scalar_spin.setRange(-1e9, 1e9)
        self.dark_scalar_spin.setValue(500.0)
        dark_form.addRow("Value:", self.dark_scalar_spin)
        self.dark_pct_spin = QtWidgets.QDoubleSpinBox()
        self.dark_pct_spin.setDecimals(2)
        self.dark_pct_spin.setRange(0.0, 10.0)
        self.dark_pct_spin.setSingleStep(0.1)
        self.dark_pct_spin.setValue(0.1)
        self.dark_pct_spin.setToolTip("Percentile of each band used as its dark value (0 = band minimum). Using a small percentile above 0 avoids single bad pixels.")
        dark_form.addRow("Percentile (%):", self.dark_pct_spin)
        lay.addLayout(dark_form)
        file_row = QtWidgets.QHBoxLayout()
        self.btn_dark_file = QtWidgets.QPushButton("Choose file...")
        self.btn_dark_file.clicked.connect(self.on_dark_choose_file)
        file_row.addWidget(self.btn_dark_file)
        self.dark_file_label = QtWidgets.QLabel("no file")
        self.dark_file_label.setWordWrap(True)
        file_row.addWidget(self.dark_file_label, 1)
        lay.addLayout(file_row)
        self.dark_clip_check = QtWidgets.QCheckBox("Clip negative results to 0")
        self.dark_clip_check.setToolTip("Off by default: the values are subtracted exactly as given.")
        lay.addWidget(self.dark_clip_check)
        self.btn_dark_apply = QtWidgets.QPushButton("Apply Dark Pixel Subtraction")
        self.btn_dark_apply.setMinimumHeight(34)
        self.btn_dark_apply.clicked.connect(self.on_dark_apply)
        lay.addWidget(self.btn_dark_apply)
        self.dark_status_label = info_label("Choose a dark value source and apply.")
        lay.addWidget(self.dark_status_label)
        lay.addStretch(1)
        self._on_dark_mode_changed(0)

        # -- page 2: flat field --
        lay = atm_page()
        lay.addWidget(help_box(
            "Flat field: divides every pixel spectrum by the mean spectrum of a bright, spectrally flat "
            "area of the scene. The result is relative reflectance (the flat area becomes about 1).",
            "Roberts, D. A., Y. Yamaguchi, and R. J. P. Lyon. Comparison of Various Techniques for "
            "Calibration of AIS Data. In Proceedings of the Second Airborne Imaging Spectrometer Data "
            "Analysis Workshop, 21-30. Pasadena: Jet Propulsion Laboratory, 1986."))
        lay.addWidget(QtWidgets.QLabel("Flat-field area (ROI):"))
        self.ff_roi_mode = QtWidgets.QComboBox()
        self.ff_roi_mode.addItems([
            "Rectangle [row, column, height, width]",
            "ROI(s) selected in ROI Manager",
        ])
        self.ff_roi_mode.currentIndexChanged.connect(self._on_ff_roi_mode_changed)
        lay.addWidget(self.ff_roi_mode)
        ff_form = QtWidgets.QFormLayout()
        ff_form.setSpacing(4)
        self.ff_roi_spins = []
        for lbl, val in (("Row (1 = top):", 1), ("Column (1 = left):", 1), ("Height (pixels):", 10), ("Width (pixels):", 10)):
            sp = QtWidgets.QSpinBox()
            sp.setRange(1, 1000000)
            sp.setValue(val)
            ff_form.addRow(lbl, sp)
            self.ff_roi_spins.append(sp)
        lay.addLayout(ff_form)
        lay.addWidget(QtWidgets.QLabel("Optional reference spectrum for spectral matching:"))
        ref_row = QtWidgets.QHBoxLayout()
        self.btn_ff_ref = QtWidgets.QPushButton("Choose ECOSTRESS / CSV spectrum...")
        self.btn_ff_ref.setToolTip("Text file with wavelength, reflectance columns (ECOSTRESS .spectrum.txt, USGS, CSV). "
                                   "Micrometre wavelengths are converted to nm.")
        self.btn_ff_ref.clicked.connect(self.on_ff_choose_reference)
        ref_row.addWidget(self.btn_ff_ref)
        self.btn_ff_ref_clear = QtWidgets.QPushButton("Clear")
        self.btn_ff_ref_clear.clicked.connect(self.on_ff_clear_reference)
        ref_row.addWidget(self.btn_ff_ref_clear)
        lay.addLayout(ref_row)
        self.ff_ref_label = QtWidgets.QLabel("no reference spectrum (distance scores are skipped)")
        self.ff_ref_label.setWordWrap(True)
        lay.addWidget(self.ff_ref_label)
        self.ff_ref = None          # (wavelengths_nm, values, file name)
        self._on_ff_roi_mode_changed(0)
        self.btn_ff_apply = QtWidgets.QPushButton("Apply Flat Field Correction")
        self.btn_ff_apply.setMinimumHeight(34)
        self.btn_ff_apply.clicked.connect(self.on_flat_field_apply)
        lay.addWidget(self.btn_ff_apply)
        self.ff_status_label = info_label("Set the flat-field area and apply.")
        lay.addWidget(self.ff_status_label)
        lay.addStretch(1)

        # -- page 3: IARR --
        lay = atm_page()
        lay.addWidget(help_box(
            "IARR (Internal Average Relative Reflectance): divides every pixel spectrum by the mean "
            "spectrum of the whole scene. No reference ROI or field data is needed; the result is "
            "relative reflectance.",
            "Kruse, F. A. Use of Airborne Imaging Spectrometer Data to Map Minerals Associated with "
            "Hydrothermally Altered Rocks in the Northern Grapevine Mountains, Nevada, and California. "
            "Remote Sensing of Environment 24(1), 1988, 31-51. https://doi.org/10.1016/0034-4257(88)90004-1"))
        self.btn_iarr_apply = QtWidgets.QPushButton("Apply IARR")
        self.btn_iarr_apply.setMinimumHeight(34)
        self.btn_iarr_apply.clicked.connect(self.on_iarr_apply)
        lay.addWidget(self.btn_iarr_apply)
        self.iarr_status_label = info_label("Applies to the whole active cube.")
        lay.addWidget(self.iarr_status_label)
        lay.addStretch(1)

        correction_layout.addWidget(atm_group, 1)

        # ---- 3. Spectral Correction (smile) ----
        sc_group = QtWidgets.QGroupBox("3. Spectral Correction (smile)")
        sc_layout = QtWidgets.QVBoxLayout(sc_group)
        sc_layout.setContentsMargins(8, 14, 8, 8)
        sc_layout.setSpacing(5)
        sc_layout.addWidget(self._help_box(
            "Reduce the spectral smile effect (across-track shift of the band centres). "
            "Spectral smoothing averages each pixel spectrum over a moving window along the spectral "
            "axis (windows above 9 can remove fine absorption features); the MNF method rebuilds the "
            "cube from its leading MNF components.",
            "Perkins et al., Speed and Accuracy Improvements in FLAASH Atmospheric Correction of "
            "Hyperspectral Imagery, Optical Engineering 51(11), 2012. "
            "Yokoya, Miyamura, Iwasaki, Applied Optics 49(24), 2010."))
        sc_layout.addWidget(QtWidgets.QLabel("Method:"))
        self.smile_method_combo = QtWidgets.QComboBox()
        self.smile_method_combo.addItems(["SpectralSmoothing", "MNF"])
        self.smile_method_combo.currentIndexChanged.connect(self._on_smile_method_changed)
        sc_layout.addWidget(self.smile_method_combo)
        sc_form = QtWidgets.QFormLayout()
        sc_form.setSpacing(4)
        self.smile_window_spin = QtWidgets.QSpinBox()
        self.smile_window_spin.setRange(1, 10000)
        self.smile_window_spin.setValue(5)
        self.smile_window_spin.setToolTip("Averaging window along the spectral dimension (bands). Windows above 9 "
                                          "can wash out fine atmospheric absorption features.")
        sc_form.addRow("Spectral window (bands):", self.smile_window_spin)
        self.smile_mnf_spin = QtWidgets.QSpinBox()
        self.smile_mnf_spin.setRange(1, 10000)
        self.smile_mnf_spin.setValue(10)
        sc_form.addRow("MNF components kept:", self.smile_mnf_spin)
        self.smile_band_a = QtWidgets.QSpinBox()
        self.smile_band_a.setRange(1, 10000)
        self.smile_band_a.setValue(40)
        self.smile_band_b = QtWidgets.QSpinBox()
        self.smile_band_b.setRange(1, 10000)
        self.smile_band_b.setValue(42)
        sc_form.addRow("Check band A (1-based):", self.smile_band_a)
        sc_form.addRow("Check band B (1-based):", self.smile_band_b)
        sc_layout.addLayout(sc_form)
        self.btn_smile_apply = QtWidgets.QPushButton("Apply Smile Reduction")
        self.btn_smile_apply.setMinimumHeight(34)
        self.btn_smile_apply.clicked.connect(self.on_smile_apply)
        sc_layout.addWidget(self.btn_smile_apply)
        self.smile_status_label = QtWidgets.QLabel(
            "After applying, the absolute difference of bands A and B is shown before and after.")
        self.smile_status_label.setWordWrap(True)
        self.smile_status_label.setStyleSheet("color: #888888;")
        sc_layout.addWidget(self.smile_status_label)
        sc_layout.addStretch(1)
        correction_layout.addWidget(sc_group, 1)
        self._on_smile_method_changed(0)

        # ---- 4. Noise Reduction (NGmeet) ----
        self._ng_last_result = None
        ng_group = QtWidgets.QGroupBox("4. Noise Reduction (NGmeet)")
        ng_layout = QtWidgets.QVBoxLayout(ng_group)
        ng_layout.setContentsMargins(8, 14, 8, 8)
        ng_layout.setSpacing(5)
        ng_layout.addWidget(self._help_box(
            "Non-local meets global denoising: spectral low-rank projection, non-local spatial "
            "filtering of the eigen-images, inverse projection and iterative regularisation. "
            "The cube is normalised to [0, 1] first."))
        self.ng_mode_combo = QtWidgets.QComboBox()
        self.ng_mode_combo.addItems([
            "Validate: add Gaussian noise and compare (cube unchanged)",
            "Denoise the active cube (replaces it; can be reverted)",
        ])
        self.ng_mode_combo.currentIndexChanged.connect(self._on_ng_mode_changed)
        ng_layout.addWidget(self.ng_mode_combo)
        ng_form = QtWidgets.QFormLayout()
        ng_form.setSpacing(4)
        self.ng_sigma_spin = QtWidgets.QDoubleSpinBox()
        self.ng_sigma_spin.setDecimals(4)
        self.ng_sigma_spin.setRange(0.0001, 1.0)
        self.ng_sigma_spin.setSingleStep(0.005)
        self.ng_sigma_spin.setValue(0.01)
        self.ng_sigma_spin.setToolTip(
            "Smoothing parameter of the non-local filter, in normalised [0, 1] units. "
            "Values close to the noise standard deviation work best; larger = smoother."
        )
        ng_form.addRow("Sigma (smoothing):", self.ng_sigma_spin)
        self.ng_iter_spin = QtWidgets.QSpinBox()
        self.ng_iter_spin.setRange(1, 10)
        self.ng_iter_spin.setValue(2)
        ng_form.addRow("Iterations:", self.ng_iter_spin)
        self.ng_lambda_spin = QtWidgets.QDoubleSpinBox()
        self.ng_lambda_spin.setDecimals(2)
        self.ng_lambda_spin.setRange(0.05, 1.0)
        self.ng_lambda_spin.setSingleStep(0.05)
        self.ng_lambda_spin.setValue(0.9)
        self.ng_lambda_spin.setToolTip("Regularisation weight: Y(i+1) = lambda * X(i) + (1 - lambda) * Y(i).")
        ng_form.addRow("Lambda:", self.ng_lambda_spin)
        rank_row = QtWidgets.QHBoxLayout()
        self.ng_rank_auto = QtWidgets.QCheckBox("Auto")
        self.ng_rank_auto.setChecked(True)
        self.ng_rank_spin = QtWidgets.QSpinBox()
        self.ng_rank_spin.setRange(1, 100)
        self.ng_rank_spin.setValue(10)
        self.ng_rank_spin.setEnabled(False)
        self.ng_rank_auto.toggled.connect(lambda on: self.ng_rank_spin.setEnabled(not on))
        rank_row.addWidget(self.ng_rank_auto)
        rank_row.addWidget(self.ng_rank_spin, 1)
        ng_form.addRow("Subspace rank:", rank_row)
        self.ng_noise_spin = QtWidgets.QDoubleSpinBox()
        self.ng_noise_spin.setDecimals(3)
        self.ng_noise_spin.setRange(0.001, 0.5)
        self.ng_noise_spin.setSingleStep(0.005)
        self.ng_noise_spin.setValue(0.05)
        self.ng_noise_spin.setToolTip("Standard deviation of the simulated Gaussian noise on the normalised cube (variance = std squared).")
        ng_form.addRow("Noise std (validation):", self.ng_noise_spin)
        self.ng_seed_spin = QtWidgets.QSpinBox()
        self.ng_seed_spin.setRange(0, 9999)
        self.ng_seed_spin.setValue(42)
        self.ng_seed_spin.setToolTip("Fixed seed so the simulated noise is repeatable.")
        ng_form.addRow("Noise seed:", self.ng_seed_spin)
        ng_layout.addLayout(ng_form)
        self.btn_ngmeet = QtWidgets.QPushButton("Run NGmeet denoising")
        self.btn_ngmeet.setMinimumHeight(34)
        self.btn_ngmeet.clicked.connect(self.on_run_ngmeet)
        ng_layout.addWidget(self.btn_ngmeet)
        self.btn_ng_compare = QtWidgets.QPushButton("Show last comparison")
        self.btn_ng_compare.setEnabled(False)
        self.btn_ng_compare.clicked.connect(self._show_ng_figure)
        ng_layout.addWidget(self.btn_ng_compare)
        self.ng_status_label = QtWidgets.QLabel("Load a cube first.")
        self.ng_status_label.setWordWrap(True)
        ng_layout.addWidget(self.ng_status_label)
        ng_layout.addStretch(1)
        correction_layout.addWidget(ng_group, 1)

        # Give every correction column the same width: ignore each panel's content-based size hint so
        # the equal stretch factors decide, with a minimum width that keeps the controls readable.
        for i in range(correction_layout.count()):
            panel = correction_layout.itemAt(i).widget()
            if panel is not None:
                panel.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Expanding)
                panel.setMinimumWidth(270)
                correction_layout.setStretch(i, 1)

        self.tabs.addTab(self._scroll_wrap(correction_tab), "Data Correction")

        # ---- Preprocessing: normalisation, units / common grid, spectral features, statistics ----
        self._build_preprocessing_tab()

        # Keep the order: Band Management -> Data Correction -> ROI Analysis.
        tab_bar = self.tabs.tabBar()
        band_management_index = next(
            (i for i in range(self.tabs.count()) if self.tabs.tabText(i) == "Band Management"), -1
        )
        roi_analysis_index = next(
            (i for i in range(self.tabs.count()) if self.tabs.tabText(i) == "ROI Analysis"), -1
        )
        if band_management_index >= 0 and roi_analysis_index >= 0 and band_management_index != roi_analysis_index - 1:
            tab_bar.moveTab(band_management_index, roi_analysis_index)
        data_correction_index = next(
            (i for i in range(self.tabs.count()) if self.tabs.tabText(i) == "Data Correction"), -1
        )
        roi_analysis_index = next(
            (i for i in range(self.tabs.count()) if self.tabs.tabText(i) == "ROI Analysis"), -1
        )
        if data_correction_index >= 0 and roi_analysis_index >= 0 and data_correction_index != roi_analysis_index - 1:
            tab_bar.moveTab(data_correction_index, roi_analysis_index)
        # Preprocessing comes first: Preprocessing -> Band Management -> Data Correction -> ROI Analysis
        preprocessing_index = next(
            (i for i in range(self.tabs.count()) if self.tabs.tabText(i) == "Preprocessing"), -1
        )
        band_management_index = next(
            (i for i in range(self.tabs.count()) if self.tabs.tabText(i) == "Band Management"), -1
        )
        if preprocessing_index >= 0 and band_management_index >= 0 and preprocessing_index > band_management_index:
            tab_bar.moveTab(preprocessing_index, band_management_index)
        self.tabs.setCurrentIndex(0)
        self.tabs.currentChanged.connect(self._on_tab_changed)

        self._band_removal_plot_data = None
        self._band_resample_plot_data = None
        self._selected_band_plot_data = None

        open_action = QtWidgets.QAction("&Open hyperspectral image...", self)
        open_action.setShortcut(QtGui.QKeySequence.Open)
        open_action.setToolTip("ENVI, MATLAB, IDL, HDF5/HDF4, NetCDF, Zarr, GeoTIFF, NumPy, FITS and other GDAL formats")
        open_action.triggered.connect(self.open_envi)
        menubar = self.menuBar()
        file_menu = menubar.addMenu("&File")
        file_menu.addAction(open_action)
        open_zarr_action = QtWidgets.QAction("Open &Zarr folder...", self)
        open_zarr_action.setToolTip("Zarr stores are folders; pick the .zarr folder itself")
        open_zarr_action.triggered.connect(self.open_zarr_folder)
        file_menu.addAction(open_zarr_action)
        import_asd_action = QtWidgets.QAction("Import &ASD field spectra...", self)
        import_asd_action.setToolTip("ASD FieldSpec .asd point spectra as reference spectra / endmembers")
        import_asd_action.triggered.connect(self.import_asd_dialog)
        file_menu.addAction(import_asd_action)
        file_menu.addSeparator()
        save_rois_action = QtWidgets.QAction("&Save ROIs...", self)
        save_rois_action.triggered.connect(self.on_save_rois)
        file_menu.addAction(save_rois_action)
        load_rois_action = QtWidgets.QAction("&Load ROIs...", self)
        load_rois_action.triggered.connect(self.on_load_rois)
        file_menu.addAction(load_rois_action)
        view_menu = menubar.addMenu("&View")
        theme_menu = view_menu.addMenu("&Theme")
        theme_group = QtWidgets.QActionGroup(self)
        theme_group.setExclusive(True)
        self.theme_actions = {}
        for key in ("dark", "light"):
            act = QtWidgets.QAction(THEMES[key]["label"], self, checkable=True)
            act.setChecked(key == self.theme)
            act.triggered.connect(lambda _checked=False, n=key: self.on_theme_selected(n))
            theme_group.addAction(act)
            theme_menu.addAction(act)
            self.theme_actions[key] = act
        view_menu.addSeparator()
        history_action = QtWidgets.QAction("Processing &history...", self)
        history_action.triggered.connect(self.on_show_processing_history)
        view_menu.addAction(history_action)
        revert_action = QtWidgets.QAction("&Revert to original cube", self)
        revert_action.triggered.connect(self.on_revert_to_original)
        view_menu.addAction(revert_action)
        help_menu = menubar.addMenu("&Help")
        about_action = QtWidgets.QAction(f"&About {APP_NAME}", self)
        about_action.triggered.connect(self.show_about_dialog)
        help_menu.addAction(about_action)
        formats_action = QtWidgets.QAction("Supported &file formats...", self)
        formats_action.triggered.connect(self.show_formats_dialog)
        help_menu.addAction(formats_action)

        self.image_canvas.mpl_connect("button_press_event", self.on_image_click)
        self.image_canvas.after_clear = self._on_canvas_cleared

        self._busy_buttons = [self.btn_train, self.btn_cv, self.btn_cnn, self.btn_sam, self.btn_nmf,
                              self.btn_ngmeet]
        self.btn_stop = QtWidgets.QPushButton("Stop task")
        self.btn_stop.setVisible(False)
        self.btn_stop.clicked.connect(self.on_stop_task)
        self.statusBar().addPermanentWidget(self.btn_stop)
        self.load_bar = QtWidgets.QProgressBar()
        self.load_bar.setRange(0, 100)
        self.load_bar.setFixedWidth(260)
        self.load_bar.setTextVisible(True)
        self.load_bar.setFormat("%p%")
        self.load_bar.setVisible(False)
        self.load_bar_label = QtWidgets.QLabel("")
        self.load_bar_label.setVisible(False)
        self.statusBar().addPermanentWidget(self.load_bar_label)
        self.statusBar().addPermanentWidget(self.load_bar)

    @staticmethod
    def _help_box(text, reference=None, title="About this method"):
        """Collapsed-by-default description: a small header button that expands to the full text."""
        box = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(box)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)
        btn = QtWidgets.QToolButton()
        btn.setText(f" {title}")
        btn.setCheckable(True)
        btn.setChecked(False)
        btn.setArrowType(QtCore.Qt.RightArrow)
        btn.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        btn.setAutoRaise(True)
        btn.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        body = QtWidgets.QLabel(text + (f"\n\nReference: {reference}" if reference else ""))
        body.setWordWrap(True)
        body.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        body.setVisible(False)
        lay.addWidget(btn)
        lay.addWidget(body)

        def toggle(on):
            body.setVisible(on)
            btn.setArrowType(QtCore.Qt.DownArrow if on else QtCore.Qt.RightArrow)
        btn.toggled.connect(toggle)
        box.help_button = btn
        box.help_body = body
        return box

    @staticmethod
    def _scroll_wrap(widget):
        area = QtWidgets.QScrollArea()
        area.setWidgetResizable(True)
        area.setFrameShape(QtWidgets.QFrame.NoFrame)
        area.setWidget(widget)
        return area

    def _apply_window_qss(self, name):
        t = THEMES[name]
        self.setStyleSheet(
            "QGroupBox { font-weight: bold; margin-top: 8px; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px 0 4px; }"
            "QPushButton, QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox { min-height: 24px; }"
            f"QTabWidget::pane {{ border: 1px solid {t['pane_border']}; }}"
            "QTabBar::tab { min-width: 110px; min-height: 24px; padding: 3px 8px; }"
        )
        info = getattr(self, "info_label", None)
        if info is not None:
            info.setStyleSheet(
                "QLabel {" + t["info_style"] + " border-radius:6px; padding:10px;}"
            )

    def on_theme_selected(self, name):
        if name == self.theme:
            return
        self.theme = name
        set_matplotlib_theme(name)
        apply_app_theme(QtWidgets.QApplication.instance(), name)
        self._apply_window_qss(name)
        for canvas in (self.image_canvas, self.plot_canvas):
            restyle_figure(canvas.fig)
            canvas.draw()
        for key, act in self.theme_actions.items():
            act.setChecked(key == name)
        QtCore.QSettings(SETTINGS_ORG, APP_NAME).setValue("theme", name)
        self.info_label.setText(f"Theme: {THEMES[name]['label']}.")

    def show_about_dialog(self):
        QtWidgets.QMessageBox.about(
            self,
            f"About {APP_NAME}",
            f"<b>{APP_NAME} {APP_VERSION}</b><br>"
            f"{APP_SUBTITLE.replace('&', '&amp;')}<br><br>"
            f"<b>{APP_CREDIT}</b><br><br>"
            "Features include multi-format loading (ENVI, MATLAB, IDL, HDF5/HDF4, NetCDF, Zarr, GeoTIFF, "
            "ASD field spectra, preprocessing (normalisation, SNV, units, common wavelength grid, "
            "spectral indices, band depth, band gap, statistics), "
            "NumPy, FITS and GDAL rasters), spectra plotting, RGB composites, "
            "rectangle and polygon ROIs, anomaly maps, band math, k-means, "
            "linear and NMF unmixing, PCA, Spectral Angle Mapper, random forest / SVM "
            "classification with cross-validation, a 2D CNN, and signal-to-noise analysis. "
            "Data correction: radiometric calibration, atmospheric correction (empirical line, dark pixel "
            "subtraction, flat field, IARR) and NGmeet denoising, with a processing history and "
            "revert to the original cube."
        )

    @staticmethod
    def _metadata_key_normalize(key):
        return "".join(ch.lower() for ch in str(key) if ch.isalnum())

    def _metadata_value(self, *aliases):
        """Find a metadata value using case/spacing-insensitive aliases."""
        if not isinstance(getattr(self, "metadata", None), dict):
            return None
        normalized = {self._metadata_key_normalize(k): v for k, v in self.metadata.items()}
        for alias in aliases:
            key = self._metadata_key_normalize(alias)
            if key in normalized:
                return normalized[key]
        return None

    def _metadata_numbers(self, *aliases):
        """Parse scalar or per-band numeric metadata values from ENVI-style strings/lists."""
        value = self._metadata_value(*aliases)
        if value is None:
            return None
        if isinstance(value, (list, tuple, np.ndarray)):
            raw_items = list(value)
        else:
            text = str(value).strip().strip("{}[]()")
            raw_items = text.replace(",", " ").replace(";", " ").split()
        numbers = []
        for item in raw_items:
            try:
                numbers.append(float(str(item).strip().strip("{}[](),")))
            except (TypeError, ValueError):
                continue
        return np.asarray(numbers, dtype=float) if numbers else None

    def _bad_band_mask_from_metadata(self):
        """Return True for good bands and False for bad bands, when metadata exists."""
        values = self._metadata_numbers("BadBands", "bbl", "bad bands", "bad_band_list")
        if values is None:
            return None
        values = np.asarray(values).ravel()
        if self.cube is not None and values.size != self.cube.shape[2]:
            return None
        return np.isfinite(values) & (values != 0)

    def _on_radiometric_function_changed(self, index):
        """Show only the parameter controls for the selected radiometric operation."""
        self.radiometric_parameter_stack.setCurrentIndex(max(0, min(2, int(index))))

    def _format_metadata_parameter(self, *aliases):
        values = self._metadata_numbers(*aliases)
        if values is None:
            return ""
        return ", ".join(f"{float(v):.10g}" for v in values)

    def _refresh_radiometric_parameter_fields(self):
        """Populate editable calibration controls from metadata, if available."""
        for editor, aliases in [
            (self.radiometric_gain_edit, ("Gain", "data gain values", "gain values")),
            (self.radiometric_bias_edit, ("Bias", "Offset", "data offset values", "offset values")),
            (self.radiometric_rgain_edit, ("RGain", "reflectance gain", "reflectance gain values", "r gain")),
            (self.radiometric_roffset_edit, ("ROffset", "reflectance offset", "reflectance offset values", "r offset")),
            (self.radiometric_esun_edit, ("ESUN", "solar irradiance", "mean solar irradiance", "solar irradiance values")),
            (self.radiometric_distance_edit, ("EarthSunDistance", "earth sun distance", "earth-sun distance", "earth sun distance au")),
            (self.radiometric_elevation_edit, ("SunElevation", "sun elevation", "solar elevation", "sun elevation angle")),
        ]:
            # Keep user-entered values when metadata is absent; otherwise refresh from loaded metadata.
            value = self._format_metadata_parameter(*aliases)
            if value:
                editor.setText(value)

    @staticmethod
    def _parse_radiometric_parameter(text, name, expected_count=None, scalar=False):
        cleaned = str(text).strip().strip("{}[]()")
        try:
            values = np.asarray([float(x) for x in cleaned.replace(",", " ").replace(";", " ").split()], dtype=float)
        except ValueError:
            raise ValueError(f"{name} contains a non-numeric value.")
        if values.size == 0 or not np.all(np.isfinite(values)):
            raise ValueError(f"Enter finite numeric value(s) for {name}.")
        if scalar and values.size != 1:
            raise ValueError(f"{name} must contain exactly one value.")
        if expected_count is not None and values.size != expected_count:
            raise ValueError(f"{name} requires {expected_count} values (one per active band); received {values.size}.")
        return values

    def on_apply_radiometric_function(self):
        """Run the selected conversion using editable parameter controls."""
        if self.cube is None:
            QtWidgets.QMessageBox.warning(self, "No data", "Load a hyperspectral cube first.")
            return
        index = self.radiometric_function_combo.currentIndex()
        try:
            if index == 0:
                self.on_dn_to_radiance(self.radiometric_gain_edit.text(), self.radiometric_bias_edit.text())
            elif index == 1:
                self.on_dn_to_reflectance(self.radiometric_rgain_edit.text(), self.radiometric_roffset_edit.text())
            else:
                self.on_radiance_to_reflectance(
                    self.radiometric_esun_edit.text(), self.radiometric_distance_edit.text(), self.radiometric_elevation_edit.text()
                )
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, "Invalid calibration parameters", str(exc))

    def _refresh_radiometric_metadata_status(self):
        if self.cube is None:
            self.radiometric_badbands_label.setText("Bad bands: load a hyperspectral cube first")
            self.radiometric_status_label.setText("Load ENVI data with radiometric calibration metadata.")
            return
        raw = self._metadata_numbers("BadBands", "bbl", "bad bands", "bad_band_list")
        if raw is None:
            bad_text = "Bad bands: not specified in header metadata"
        else:
            bad = np.where(np.isfinite(raw) & (raw == 0))[0] + 1
            bad_text = "Bad band numbers (1-based): " + (", ".join(map(str, bad.tolist())) if bad.size else "none")
            if raw.size != self.cube.shape[2]:
                bad_text += f" (metadata lists {raw.size} bands; active cube has {self.cube.shape[2]})"
        self.radiometric_badbands_label.setText(bad_text)
        self.radiometric_status_label.setText(
            "Calibration parameters are read from the ENVI header. Bad bands are excluded automatically when the metadata band list matches the active cube."
        )

    def _radiometric_input_and_coefficients(self, gain_aliases, offset_aliases, operation_name):
        if self.cube is None:
            QtWidgets.QMessageBox.warning(self, "No data", "Load a hyperspectral cube first.")
            return None
        gain = self._metadata_numbers(*gain_aliases)
        offset = self._metadata_numbers(*offset_aliases)
        if gain is None or offset is None:
            QtWidgets.QMessageBox.warning(
                self, "Missing calibration metadata",
                f"{operation_name} requires per-band gain and offset values in the ENVI header.\n\n"
                f"Gain aliases checked: {', '.join(gain_aliases)}\n"
                f"Offset aliases checked: {', '.join(offset_aliases)}"
            )
            return None
        cube = np.asarray(self.cube, dtype=float)
        wavelengths = np.asarray(self.wavelengths, dtype=float).copy() if self.wavelengths is not None and len(self.wavelengths) == cube.shape[2] else None
        bbl = self._metadata_numbers("BadBands", "bbl", "bad bands", "bad_band_list")
        if bbl is not None and bbl.size == cube.shape[2] and gain.size == bbl.size and offset.size == bbl.size:
            good = np.isfinite(bbl) & (bbl != 0)
            if not np.any(good):
                QtWidgets.QMessageBox.warning(self, "Invalid bad-band metadata", "The metadata marks every band as bad.")
                return None
            if not np.all(good):
                original_count = good.size
                cube = cube[:, :, good]
                gain = gain[good]
                offset = offset[good]
                if wavelengths is not None:
                    wavelengths = wavelengths[good]
                # Keep all known per-band calibration vectors aligned after removing bad bands,
                # so the next radiometric operation (for example radiance -> reflectance) works.
                for canonical, aliases in {
                    "Gain": ("Gain", "data gain values", "gain values"),
                    "Offset": ("Bias", "Offset", "data offset values", "offset values"),
                    "RGain": ("RGain", "reflectance gain", "reflectance gain values", "r gain"),
                    "ROffset": ("ROffset", "reflectance offset", "reflectance offset values", "r offset"),
                    "ESUN": ("ESUN", "solar irradiance", "mean solar irradiance", "solar irradiance values"),
                }.items():
                    vector = self._metadata_numbers(*aliases)
                    if vector is not None and vector.size == original_count:
                        self.metadata[canonical] = vector[good].tolist()
                self.metadata["BadBands"] = [1] * int(np.count_nonzero(good))
        elif gain.size != cube.shape[2] or offset.size != cube.shape[2]:
            QtWidgets.QMessageBox.warning(
                self, "Band-count mismatch",
                f"The active cube has {cube.shape[2]} bands, but the calibration metadata contains "
                f"{gain.size} gain and {offset.size} offset values. Load the original cube and run calibration before changing its bands."
            )
            return None
        if gain.size != cube.shape[2] or offset.size != cube.shape[2]:
            QtWidgets.QMessageBox.warning(self, "Band-count mismatch", "Calibration coefficients do not match the active cube after bad-band handling.")
            return None
        return cube, wavelengths, gain, offset

    def _apply_radiometric_output(self, input_cube, output_cube, wavelengths, stage, operation_label):
        self._radiometric_preview_data = (np.asarray(input_cube).copy(), np.asarray(output_cube).copy(), operation_label)
        self.cube = np.asarray(output_cube, dtype=float)
        if wavelengths is not None and len(wavelengths) == self.cube.shape[2]:
            self.wavelengths = np.asarray(wavelengths, dtype=float)
        bands = self.cube.shape[2]
        self.band_slider.setMaximum(bands - 1)
        self.band_slider.setValue(min(self.band_slider.value(), bands - 1))
        for spin in (self.spin_r, self.spin_g, self.spin_b, self.spin_nmf_b0, self.spin_nmf_b1):
            spin.setMaximum(bands - 1)
        self.radiometric_band_spin.setMaximum(max(1, bands))
        self.radiometric_band_spin.setValue(max(1, min(bands, self.radiometric_band_spin.value())))
        self.radiometric_stage = stage
        self.radiometric_stage_label.setText(f"Current data stage: {stage}")
        self.btn_radiometric_preview.setEnabled(True)
        self._invalidate_spectral_results()
        self._update_band_management_range()
        self.update_image()
        self.radiometric_status_label.setText(f"{operation_label} completed. Active cube: {self.cube.shape[0]} × {self.cube.shape[1]} × {bands} bands.")
        self._refresh_radiometric_metadata_status()
        self._record_step(operation_label, {"stage": stage}, bands_before=np.asarray(input_cube).shape[2])
        self.info_label.setText(f"Radiometric calibration complete: {operation_label}.")

    def on_dn_to_radiance(self, gain_text=None, bias_text=None):
        """Convert DN values to TOA radiance: Lλ = DN × Gain + Bias/Offset."""
        if self.radiometric_stage.lower().startswith("radiance") or self.radiometric_stage.lower().startswith("toa radiance"):
            reply = QtWidgets.QMessageBox.question(
                self, "Data already calibrated", "The active data stage is already radiance. Apply DN-to-radiance again?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No, QtWidgets.QMessageBox.No
            )
            if reply != QtWidgets.QMessageBox.Yes:
                return
        if self.cube is None:
            QtWidgets.QMessageBox.warning(self, "No data", "Load a hyperspectral cube first.")
            return
        cube_current = np.asarray(self.cube, dtype=float)
        gain_values = self._parse_radiometric_parameter(gain_text if gain_text is not None else self.radiometric_gain_edit.text(), "Gain", cube_current.shape[2])
        bias_values = self._parse_radiometric_parameter(bias_text if bias_text is not None else self.radiometric_bias_edit.text(), "Bias", cube_current.shape[2])
        # Route editable values through the common bad-band handler so coefficients and bands stay aligned.
        self.metadata["Gain"] = gain_values.tolist()
        self.metadata["Bias"] = bias_values.tolist()
        result = self._radiometric_input_and_coefficients(
            ("Gain",), ("Bias",), "DN-to-radiance conversion"
        )
        if result is None:
            return
        cube, wavelengths, gain, offset = result
        out = cube * gain.reshape(1, 1, -1) + offset.reshape(1, 1, -1)
        self._apply_radiometric_output(cube, out, wavelengths, "TOA radiance", "DN → Radiance")

    def on_dn_to_reflectance(self, rgain_text=None, roffset_text=None):
        """Convert DN values directly to TOA reflectance: rho = DN × RGain + ROffset."""
        if self.cube is None:
            QtWidgets.QMessageBox.warning(self, "No data", "Load a hyperspectral cube first.")
            return
        cube_current = np.asarray(self.cube, dtype=float)
        rgain_values = self._parse_radiometric_parameter(rgain_text if rgain_text is not None else self.radiometric_rgain_edit.text(), "RGain", cube_current.shape[2])
        roffset_values = self._parse_radiometric_parameter(roffset_text if roffset_text is not None else self.radiometric_roffset_edit.text(), "ROffset", cube_current.shape[2])
        self.metadata["RGain"] = rgain_values.tolist()
        self.metadata["ROffset"] = roffset_values.tolist()
        result = self._radiometric_input_and_coefficients(
            ("RGain",), ("ROffset",), "DN-to-reflectance conversion"
        )
        if result is None:
            return
        cube, wavelengths, gain, offset = result
        out = cube * gain.reshape(1, 1, -1) + offset.reshape(1, 1, -1)
        self._apply_radiometric_output(cube, out, wavelengths, "TOA reflectance", "DN → Reflectance")

    def on_radiance_to_reflectance(self, esun_text=None, distance_text=None, elevation_text=None):
        """Convert TOA radiance to TOA reflectance using solar irradiance and geometry metadata."""
        if self.cube is None:
            QtWidgets.QMessageBox.warning(self, "No data", "Load a hyperspectral cube first.")
            return
        cube = np.asarray(self.cube, dtype=float)
        esun = self._parse_radiometric_parameter(esun_text if esun_text is not None else self.radiometric_esun_edit.text(), "ESUN", cube.shape[2])
        distance = self._parse_radiometric_parameter(distance_text if distance_text is not None else self.radiometric_distance_edit.text(), "Earth-Sun distance", scalar=True)
        elevation = self._parse_radiometric_parameter(elevation_text if elevation_text is not None else self.radiometric_elevation_edit.text(), "Sun elevation", scalar=True)
        wavelengths = np.asarray(self.wavelengths, dtype=float).copy() if self.wavelengths is not None and len(self.wavelengths) == cube.shape[2] else None
        bbl = self._metadata_numbers("BadBands", "bbl", "bad bands", "bad_band_list")
        if bbl is not None and bbl.size == cube.shape[2] and esun.size == bbl.size:
            good = np.isfinite(bbl) & (bbl != 0)
            if not np.any(good):
                QtWidgets.QMessageBox.warning(self, "Invalid bad-band metadata", "The metadata marks every band as bad.")
                return
            if not np.all(good):
                original_count = good.size
                cube, esun = cube[:, :, good], esun[good]
                if wavelengths is not None: wavelengths = wavelengths[good]
                for canonical, aliases in {
                    "Gain": ("Gain", "data gain values", "gain values"),
                    "Offset": ("Bias", "Offset", "data offset values", "offset values"),
                    "RGain": ("RGain", "reflectance gain", "reflectance gain values", "r gain"),
                    "ROffset": ("ROffset", "reflectance offset", "reflectance offset values", "r offset"),
                }.items():
                    vector = self._metadata_numbers(*aliases)
                    if vector is not None and vector.size == original_count:
                        self.metadata[canonical] = vector[good].tolist()
                self.metadata["ESUN"] = esun.tolist()
                self.metadata["BadBands"] = [1] * int(np.count_nonzero(good))
        if esun.size != cube.shape[2]:
            QtWidgets.QMessageBox.warning(self, "Band-count mismatch", f"The active cube has {cube.shape[2]} bands but ESUN contains {esun.size} values. Run this operation before changing the band set, or update the metadata.")
            return
        d = float(distance[0])
        elev_deg = float(elevation[0])
        if not np.isfinite(d) or d <= 0 or not np.isfinite(elev_deg) or not (0 < elev_deg <= 90):
            QtWidgets.QMessageBox.warning(self, "Invalid solar geometry", "Earth-Sun distance must be positive and sun elevation must be in (0, 90] degrees.")
            return
        sin_elev = np.sin(np.deg2rad(elev_deg))
        denominator = esun * sin_elev
        if np.any(~np.isfinite(denominator)) or np.any(denominator <= 0):
            QtWidgets.QMessageBox.warning(self, "Invalid ESUN metadata", "All ESUN values must be finite and positive.")
            return
        out = (np.pi * d * d * cube) / denominator.reshape(1, 1, -1)
        self._apply_radiometric_output(cube, out, wavelengths, "TOA reflectance", "Radiance → Reflectance")

    def on_radiometric_preview(self):
        """Display the selected input and output spectral band side by side."""
        if self._radiometric_preview_data is None:
            QtWidgets.QMessageBox.information(self, "No calibration result", "Run a radiometric conversion first.")
            return
        input_cube, output_cube, operation = self._radiometric_preview_data
        band = int(self.radiometric_band_spin.value()) - 1
        if band >= input_cube.shape[2] or band >= output_cube.shape[2]:
            QtWidgets.QMessageBox.warning(self, "Band unavailable", "The selected band is not present in both preview cubes.")
            return
        input_band = input_cube[:, :, band]
        output_band = output_cube[:, :, band]
        def build(fig):
            ax1, ax2 = fig.subplots(1, 2)
            im1 = ax1.imshow(input_band, cmap="gray")
            ax1.set_title("Input Band")
            ax1.axis("off")
            fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
            im2 = ax2.imshow(output_band, cmap="gray")
            ax2.set_title("Output Band")
            ax2.axis("off")
            fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
            fig.suptitle(f"{operation} — Band {band + 1}")
        self._show_figure_dialog(f"Radiometric Calibration Preview — Band {band + 1}", build, size=(10, 5))

    def open_envi(self):
        """File > Open: choose any supported hyperspectral file."""
        settings = QtCore.QSettings(SETTINGS_ORG, APP_NAME)
        start_dir = str(settings.value("last_dir", ""))
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open hyperspectral image", start_dir, hsi_file_filter()
        )
        if not path:
            return
        settings.setValue("last_dir", str(Path(path).parent))
        self.load_cube_file(path)

    open_file = open_envi

    def open_zarr_folder(self):
        """File > Open Zarr folder: Zarr stores are directories, so they need a folder picker."""
        settings = QtCore.QSettings(SETTINGS_ORG, APP_NAME)
        start_dir = str(settings.value("last_dir", ""))
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Open Zarr store (.zarr folder)", start_dir)
        if not path:
            return
        settings.setValue("last_dir", str(Path(path).parent))
        self.load_cube_file(path)

    def _modal(self, fn):
        """Run a dialog with the normal cursor even while a wait cursor is active."""
        n = 0
        while QtWidgets.QApplication.overrideCursor() is not None and n < 20:
            QtWidgets.QApplication.restoreOverrideCursor()
            n += 1
        try:
            return fn()
        finally:
            for _ in range(n):
                QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)

    def _choose_dataset_dialog(self, title, label, items, default):
        def ask():
            item, ok = QtWidgets.QInputDialog.getItem(self, title, label, items, default, False)
            return items.index(item) if ok and item in items else None
        return self._modal(ask)

    def _ask_band_axis_dialog(self, shape, guess):
        layouts = {
            0: "(bands, rows, cols)  - band-first / BSQ",
            1: "(rows, bands, cols)  - BIL",
            2: "(rows, cols, bands)  - band-last / BIP",
        }
        items = [f"Axis {a} (length {shape[a]}) is spectral   ->   {layouts[a]}" for a in range(3)]
        label = (f"The array has shape {shape[0]} x {shape[1]} x {shape[2]} and the file does not say "
                 "which axis holds the spectral bands.\nWhich axis is the spectral one?")

        def ask():
            item, ok = QtWidgets.QInputDialog.getItem(
                self, "Spectral axis", label, items, int(guess), False)
            return items.index(item) if ok and item in items else None
        return self._modal(ask)

    # ------------------------------------------------------------------
    # ASD field spectra
    # ------------------------------------------------------------------
    def _cube_has_wavelengths(self):
        return (self.cube is not None and self.wavelengths is not None
                and len(self.wavelengths) == self.cube.shape[2])

    def import_asd_dialog(self):
        settings = QtCore.QSettings(SETTINGS_ORG, APP_NAME)
        start_dir = str(settings.value("last_dir", ""))
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Import ASD field spectra", start_dir, "ASD spectra (*.asd *.ASD);;All files (*)"
        )
        if not paths:
            return
        settings.setValue("last_dir", str(Path(paths[0]).parent))
        self.import_asd_files(paths)

    def import_asd_files(self, paths):
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Import ASD spectra")
        lay = QtWidgets.QVBoxLayout(dlg)
        lay.addWidget(QtWidgets.QLabel(f"{len(paths)} ASD file(s) selected."))
        cb_splice = QtWidgets.QCheckBox("Apply splice (detector jump) correction at ~1000 nm and ~1800 nm")
        cb_splice.setChecked(True)
        lay.addWidget(cb_splice)
        can_em = self._cube_has_wavelengths()
        cb_em = QtWidgets.QCheckBox("Add to the endmember list (resampled to the loaded cube's wavelengths)")
        cb_em.setChecked(can_em)
        cb_em.setEnabled(can_em)
        lay.addWidget(cb_em)
        if not can_em:
            note = QtWidgets.QLabel("Load a cube with wavelength metadata to use these as endmembers. "
                                    "You can add them later from the Unmixing tab.")
            note.setWordWrap(True)
            lay.addWidget(note)
        cb_replace = QtWidgets.QCheckBox("Replace previously imported ASD spectra")
        cb_replace.setChecked(True)
        lay.addWidget(cb_replace)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        lay.addWidget(buttons)
        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return

        spectra, errors = [], []
        for p in paths:
            try:
                spectra.append(read_asd(p, splice_correction=cb_splice.isChecked()))
            except HSILoadError as e:
                errors.append(str(e))
            except Exception as e:
                errors.append(f"{Path(p).name}: {e}")
        if errors:
            QtWidgets.QMessageBox.warning(
                self, "Some ASD files could not be read",
                "\n".join(errors[:15]) + (f"\n... and {len(errors) - 15} more" if len(errors) > 15 else "")
            )
        if not spectra:
            return
        if cb_replace.isChecked():
            self.asd_library = []
        self.asd_library.extend(spectra)
        self._plot_asd_library()
        msg = f"Imported {len(spectra)} ASD spectra ({len(self.asd_library)} in the library)."
        if cb_em.isChecked():
            names = self._add_asd_endmembers(spectra)
            if names:
                msg += " Added as endmembers: " + ", ".join(names) + "."
                msg += self._asd_scale_note()
        self.info_label.setText(msg)

    def _add_asd_endmembers(self, spectra):
        """Resample ASD spectra onto the cube's wavelengths and append them to the endmember list."""
        if not self._cube_has_wavelengths():
            return []
        cube_wl = np.asarray(self.wavelengths, dtype=float)
        names = []
        for spec in spectra:
            order = np.argsort(spec["wavelengths"])
            em = np.interp(cube_wl, spec["wavelengths"][order], spec["values"][order],
                           left=np.nan, right=np.nan)
            if np.all(np.isnan(em)):
                continue                        # no wavelength overlap with the cube
            self.endmembers.append(em)
            names.append(f"E{len(self.endmembers)} = {spec['name']}")
        self.endmember_label.setText(f"Endmembers: {len(self.endmembers)}")
        return names

    def _asd_scale_note(self):
        """Warn when the cube is clearly not on the same 0-1 reflectance scale as the field spectra."""
        try:
            med = float(np.nanmedian(self.cube[::max(1, self.cube.shape[0] // 50),
                                               ::max(1, self.cube.shape[1] // 50), :]))
        except Exception:
            return ""
        if np.isfinite(med) and med > 2.0:
            return (" Note: the cube does not look like 0-1 reflectance (median value "
                    f"{med:.3g}). SAM is scale-invariant, but linear unmixing needs both on the same scale.")
        return ""

    def on_add_asd_endmembers(self):
        if not self.asd_library:
            QtWidgets.QMessageBox.information(self, "No ASD spectra", "Import ASD field spectra first.")
            return
        if not self._cube_has_wavelengths():
            QtWidgets.QMessageBox.warning(
                self, "Wavelengths needed",
                "Load a hyperspectral cube with wavelength metadata first, so the field spectra "
                "can be resampled onto its bands."
            )
            return
        names = self._add_asd_endmembers(self.asd_library)
        if not names:
            QtWidgets.QMessageBox.information(
                self, "No overlap", "None of the ASD spectra overlap the cube's wavelength range.")
            return
        self.info_label.setText("Added as endmembers: " + ", ".join(names) + "." + self._asd_scale_note())

    def _plot_asd_library(self):
        if not self.asd_library:
            QtWidgets.QMessageBox.information(self, "No ASD spectra", "Import ASD field spectra first.")
            return
        lib = self.asd_library
        x = lib[0]["wavelengths"]
        colors = class_colors(max(1, len(lib)))
        series = []
        for i, spec in enumerate(lib):
            same_grid = len(spec["wavelengths"]) == len(x) and np.allclose(spec["wavelengths"], x)
            y = spec["values"] if same_grid else np.interp(
                x, spec["wavelengths"], spec["values"], left=np.nan, right=np.nan)
            series.append((spec["name"], y, colors[i]))
        kinds = sorted({spec["kind"] for spec in lib})
        ylabel = kinds[0].capitalize() if len(kinds) == 1 else "Value"
        self.plot_canvas.show_multiline(x, series, title=f"ASD field spectra ({len(lib)})",
                                        xlabel="Wavelength (nm)", ylabel=ylabel)
        if len(series) > 12:                    # a legend with dozens of entries hides the plot
            leg = self.plot_canvas.ax.get_legend()
            if leg is not None:
                leg.remove()
                self.plot_canvas.draw()

    def on_save_asd_csv(self):
        if not self.asd_library:
            QtWidgets.QMessageBox.information(self, "No ASD spectra", "Import ASD field spectra first.")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save ASD spectra", "asd_spectra.csv", "CSV files (*.csv)")
        if not path:
            return
        x = self.asd_library[0]["wavelengths"]
        cols = [x]
        for spec in self.asd_library:
            cols.append(np.interp(x, spec["wavelengths"], spec["values"], left=np.nan, right=np.nan))
        header = "wavelength_nm," + ",".join(spec["name"].replace(",", "_") for spec in self.asd_library)
        try:
            np.savetxt(path, np.column_stack(cols), delimiter=",", header=header, comments="", fmt="%.6g")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to save CSV:\n{e}")
            return
        self.info_label.setText(f"Saved {len(self.asd_library)} ASD spectra to {path}")

    # ==================================================================
    # Preprocessing tab
    # ==================================================================
    @staticmethod
    def _dspin(lo, hi, value, decimals=2, step=1.0, tooltip=None):
        sp = QtWidgets.QDoubleSpinBox()
        sp.setRange(lo, hi)
        sp.setDecimals(decimals)
        sp.setSingleStep(step)
        sp.setValue(value)
        if tooltip:
            sp.setToolTip(tooltip)
        return sp

    @staticmethod
    def _prep_group(title):
        group = QtWidgets.QGroupBox(title)
        lay = QtWidgets.QVBoxLayout(group)
        lay.setContentsMargins(8, 14, 8, 8)
        lay.setSpacing(5)
        return group, lay

    @staticmethod
    def _status_label(text):
        lab = QtWidgets.QLabel(text)
        lab.setWordWrap(True)
        lab.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        return lab

    def _build_preprocessing_tab(self):
        """Preprocessing tab: five sub-pages, each laid out side by side to fit the short tab area."""
        prep_tab = QtWidgets.QWidget()
        outer = QtWidgets.QVBoxLayout(prep_tab)
        outer.setContentsMargins(6, 6, 6, 6)
        outer.setSpacing(6)
        self.prep_pages = QtWidgets.QTabWidget()
        outer.addWidget(self.prep_pages, 1)

        def compact(combo, chars=18):
            combo.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon)
            combo.setMinimumContentsLength(chars)
            return combo

        def page(title):
            w = QtWidgets.QWidget()
            row = QtWidgets.QHBoxLayout(w)
            row.setContentsMargins(6, 6, 6, 6)
            row.setSpacing(10)
            self.prep_pages.addTab(self._scroll_wrap(w), title)
            return row

        def column(row, title=None, stretch=1):
            if title:
                box, lay = self._prep_group(title)
            else:
                box = QtWidgets.QWidget()
                lay = QtWidgets.QVBoxLayout(box)
                lay.setContentsMargins(0, 0, 0, 0)
                lay.setSpacing(5)
            row.addWidget(box, stretch)
            return lay

        # ---------- Normalisation ----------
        row = page("Normalisation")
        c1 = column(row, "Method")
        self.prep_norm_method = compact(QtWidgets.QComboBox(), 22)
        for key, label in NORMALIZATION_METHODS:
            self.prep_norm_method.addItem(label, key)
        self.prep_norm_method.setToolTip(
            "Min-max: linear rescale to a range.\n"
            "SNV: centre and scale each pixel spectrum (removes scatter / brightness offsets).\n"
            "Z-score: each band to mean 0 and std 1 over the image (before PCA / classifiers).\n"
            "L2: each spectrum to unit length (removes illumination intensity).")
        c1.addWidget(self.prep_norm_method)
        c1.addStretch(1)
        c2 = column(row, "Min-max options")
        rng_row = QtWidgets.QHBoxLayout()
        rng_row.addWidget(QtWidgets.QLabel("Range"))
        self.prep_minmax_lo = self._dspin(-1e9, 1e9, 0.0, 4, 0.1)
        self.prep_minmax_hi = self._dspin(-1e9, 1e9, 1.0, 4, 0.1)
        rng_row.addWidget(self.prep_minmax_lo, 1)
        rng_row.addWidget(QtWidgets.QLabel("to"))
        rng_row.addWidget(self.prep_minmax_hi, 1)
        c2.addLayout(rng_row)
        scope_row = QtWidgets.QHBoxLayout()
        scope_row.addWidget(QtWidgets.QLabel("Scope"))
        self.prep_minmax_scope = compact(QtWidgets.QComboBox(), 12)
        for key, label in MINMAX_SCOPES:
            self.prep_minmax_scope.addItem(label, key)
        scope_row.addWidget(self.prep_minmax_scope, 1)
        c2.addLayout(scope_row)
        self.prep_minmax_robust = QtWidgets.QCheckBox("Robust (2-98 % percentiles, clipped)")
        self.prep_minmax_robust.setToolTip("Ignore extreme outliers when finding the minimum and maximum")
        c2.addWidget(self.prep_minmax_robust)
        c2.addStretch(1)
        c3 = column(row, "Run")
        btn = QtWidgets.QPushButton("Apply Normalisation")
        btn.clicked.connect(self.on_prep_normalize)
        c3.addWidget(btn)
        self.btn_prep_compare = QtWidgets.QPushButton("Plot Before / After Spectra")
        self.btn_prep_compare.setEnabled(False)
        self.btn_prep_compare.setToolTip("Sample spectra (and the current ROI mean) before and after the last step")
        self.btn_prep_compare.clicked.connect(self.on_prep_plot_compare)
        c3.addWidget(self.btn_prep_compare)
        self.prep_norm_status = self._status_label("Replaces the active cube; use Undo to go back.")
        c3.addWidget(self.prep_norm_status)
        c3.addStretch(1)
        self.prep_norm_method.currentIndexChanged.connect(self._prep_norm_method_changed)

        # ---------- Units ----------
        row = page("Units")
        c1 = column(row, "Data values", 3)
        self.prep_value_info = self._status_label("Value range: load a cube first")
        c1.addWidget(self.prep_value_info)
        self.prep_unit_combo = compact(QtWidgets.QComboBox(), 24)
        for label, factor, offset in VALUE_UNIT_CONVERSIONS:
            self.prep_unit_combo.addItem(label, (factor, offset))
        c1.addWidget(self.prep_unit_combo)
        fo_row = QtWidgets.QHBoxLayout()
        fo_row.addWidget(QtWidgets.QLabel("Factor"))
        self.prep_unit_factor = self._dspin(-1e12, 1e12, 1.0, 6, 0.1)
        fo_row.addWidget(self.prep_unit_factor, 1)
        fo_row.addWidget(QtWidgets.QLabel("Offset"))
        self.prep_unit_offset = self._dspin(-1e12, 1e12, 0.0, 6, 0.1)
        fo_row.addWidget(self.prep_unit_offset, 1)
        c1.addLayout(fo_row)
        btn = QtWidgets.QPushButton("Convert Data Values")
        btn.clicked.connect(self.on_prep_convert_values)
        c1.addWidget(btn)
        c1.addStretch(1)
        c2 = column(row, "Wavelength units", 2)
        self.prep_wl_info = self._status_label("Wavelengths: load a cube first")
        c2.addWidget(self.prep_wl_info)
        wl_row = QtWidgets.QHBoxLayout()
        wl_row.addWidget(QtWidgets.QLabel("Values are actually in"))
        self.prep_wl_unit_combo = compact(QtWidgets.QComboBox(), 14)
        for key, label in WAVELENGTH_UNITS:
            self.prep_wl_unit_combo.addItem(label, key)
        self.prep_wl_unit_combo.setCurrentIndex(1)
        self.prep_wl_unit_combo.setToolTip(
            "Fix wrong wavelength metadata. The values are converted to nanometres;\n"
            "wavenumbers are converted and the bands reordered to increasing wavelength.")
        wl_row.addWidget(self.prep_wl_unit_combo, 1)
        c2.addLayout(wl_row)
        btn = QtWidgets.QPushButton("Convert Wavelengths to nm")
        btn.clicked.connect(self.on_prep_convert_wavelengths)
        c2.addWidget(btn)
        self.prep_units_status = self._status_label("")
        c2.addWidget(self.prep_units_status)
        c2.addStretch(1)
        self.prep_unit_combo.currentIndexChanged.connect(self._prep_unit_combo_changed)
        self._prep_unit_combo_changed()

        # ---------- Common wavelength grid ----------
        row = page("Common Grid")
        c1 = column(row, "Target grid")
        self.prep_grid_source = compact(QtWidgets.QComboBox(), 18)
        self.prep_grid_source.addItems(["Uniform: start / end / step", "From file (.hdr, .txt, .csv)"])
        c1.addWidget(self.prep_grid_source)
        self.prep_grid_uniform = QtWidgets.QWidget()
        g = QtWidgets.QGridLayout(self.prep_grid_uniform)
        g.setContentsMargins(0, 0, 0, 0)
        self.prep_grid_start = self._dspin(0, 1e6, 400.0, 2, 1.0)
        self.prep_grid_stop = self._dspin(0, 1e6, 2500.0, 2, 1.0)
        self.prep_grid_step = self._dspin(0.001, 1e5, 10.0, 3, 1.0)
        for col_i, (text, sp) in enumerate((("Start", self.prep_grid_start), ("End", self.prep_grid_stop),
                                            ("Step", self.prep_grid_step))):
            g.addWidget(QtWidgets.QLabel(f"{text} (nm)"), 0, col_i)
            g.addWidget(sp, 1, col_i)
        c1.addWidget(self.prep_grid_uniform)
        self.btn_prep_grid_fill = QtWidgets.QPushButton("Use Cube Range")
        self.btn_prep_grid_fill.clicked.connect(self._prep_fill_grid_from_cube)
        c1.addWidget(self.btn_prep_grid_fill)
        self.btn_prep_grid_file = QtWidgets.QPushButton("Load Grid File...")
        self.btn_prep_grid_file.setToolTip(
            "ENVI .hdr (wavelength + fwhm), or a text / CSV file with one band per line:\n"
            "wavelength [, fwhm]  or  band, wavelength [, fwhm]. Use it to match another sensor.")
        self.btn_prep_grid_file.clicked.connect(self.on_prep_load_grid_file)
        c1.addWidget(self.btn_prep_grid_file)
        self.prep_grid_file_label = self._status_label("No grid file loaded")
        c1.addWidget(self.prep_grid_file_label)
        c1.addStretch(1)
        c2 = column(row, "Resampling")
        self.prep_grid_method = compact(QtWidgets.QComboBox(), 18)
        for key, label in GRID_METHODS:
            self.prep_grid_method.addItem(label, key)
        self.prep_grid_method.setToolTip(
            "Linear: interpolate between neighbouring bands (same or finer resolution).\n"
            "Gaussian SRF: average source bands with a Gaussian response of the given FWHM -\n"
            "the right choice when matching a coarser sensor.")
        c2.addWidget(self.prep_grid_method)
        fwhm_row = QtWidgets.QHBoxLayout()
        fwhm_row.addWidget(QtWidgets.QLabel("FWHM (nm, 0 = auto)"))
        self.prep_grid_fwhm = self._dspin(0, 1e5, 0.0, 3, 1.0,
                                          "0 uses the FWHM from the grid file, otherwise the grid spacing")
        fwhm_row.addWidget(self.prep_grid_fwhm, 1)
        c2.addLayout(fwhm_row)
        btn = QtWidgets.QPushButton("Resample to Common Grid")
        btn.clicked.connect(self.on_prep_resample_grid)
        c2.addWidget(btn)
        c2.addStretch(1)
        c3 = column(row, "Result")
        self.prep_grid_status = self._status_label("Puts cubes from different sensors on the same wavelengths.")
        c3.addWidget(self.prep_grid_status)
        c3.addStretch(1)
        self.prep_grid_source.currentIndexChanged.connect(self._prep_grid_source_changed)
        self._prep_grid_source_changed()

        # ---------- Indices & spectral features ----------
        row = page("Indices && Features")
        c1 = column(row, "Feature")
        self.prep_feat_preset = compact(QtWidgets.QComboBox(), 20)
        for label, kind, waves in SPECTRAL_FEATURE_PRESETS:
            self.prep_feat_preset.addItem(label, (kind, waves))
        self.prep_feat_preset.setToolTip("Presets fill in the feature type and wavelengths")
        c1.addWidget(QtWidgets.QLabel("Preset"))
        c1.addWidget(self.prep_feat_preset)
        self.prep_feat_kind = compact(QtWidgets.QComboBox(), 20)
        for key, label in SPECTRAL_FEATURE_KINDS:
            self.prep_feat_kind.addItem(label, key)
        c1.addWidget(QtWidgets.QLabel("Type"))
        c1.addWidget(self.prep_feat_kind)
        c1.addStretch(1)
        c2 = column(row, "Wavelengths")
        wl_form = QtWidgets.QFormLayout()
        wl_form.setContentsMargins(0, 0, 0, 0)
        self.prep_feat_lab1 = QtWidgets.QLabel("L1 (nm)")
        self.prep_feat_lab2 = QtWidgets.QLabel("L2 (nm)")
        self.prep_feat_lab3 = QtWidgets.QLabel("L3 (nm)")
        self.prep_feat_w1 = self._dspin(0, 1e6, 860.0, 2, 1.0)
        self.prep_feat_w2 = self._dspin(0, 1e6, 650.0, 2, 1.0)
        self.prep_feat_w3 = self._dspin(0, 1e6, 760.0, 2, 1.0)
        wl_form.addRow(self.prep_feat_lab1, self.prep_feat_w1)
        wl_form.addRow(self.prep_feat_lab2, self.prep_feat_w2)
        wl_form.addRow(self.prep_feat_lab3, self.prep_feat_w3)
        c2.addLayout(wl_form)
        self.prep_feat_deepest = QtWidgets.QCheckBox("Deepest point between shoulders")
        self.prep_feat_deepest.setToolTip("Otherwise the depth is measured at the centre wavelength")
        c2.addWidget(self.prep_feat_deepest)
        c2.addStretch(1)
        c3 = column(row, "Output")
        self.prep_feat_output = compact(QtWidgets.QComboBox(), 18)
        c3.addWidget(self.prep_feat_output)
        btn = QtWidgets.QPushButton("Compute Feature Image")
        btn.clicked.connect(self.on_prep_compute_feature)
        c3.addWidget(btn)
        btn_row = QtWidgets.QHBoxLayout()
        self.btn_prep_map_view = QtWidgets.QPushButton("Colour Bar && Histogram")
        self.btn_prep_map_view.clicked.connect(self.on_prep_show_map_dialog)
        btn_row.addWidget(self.btn_prep_map_view)
        self.btn_prep_map_save = QtWidgets.QPushButton("Save Image...")
        self.btn_prep_map_save.clicked.connect(self.on_prep_save_map)
        btn_row.addWidget(self.btn_prep_map_save)
        c3.addLayout(btn_row)
        self.prep_feat_status = self._status_label("Wavelengths snap to the nearest band.")
        c3.addWidget(self.prep_feat_status)
        c3.addStretch(1)
        self.prep_feat_preset.currentIndexChanged.connect(self._prep_preset_changed)
        self.prep_feat_kind.currentIndexChanged.connect(self._prep_feat_kind_changed)
        self._prep_feat_kind_changed()

        # ---------- Statistics ----------
        row = page("Statistics")
        c1 = column(row, "Per-band statistics")
        self.prep_stats_region = compact(QtWidgets.QComboBox(), 18)
        self.prep_stats_region.addItems(["Whole image", "Current ROI selection", "Each saved ROI",
                                         "Each class (saved ROIs)"])
        c1.addWidget(self.prep_stats_region)
        btn = QtWidgets.QPushButton("Plot Mean, Std && Range")
        btn.setToolTip("Mean +/- std and min-max envelope of the spectra in the region")
        btn.clicked.connect(self.on_prep_band_stats)
        c1.addWidget(btn)
        btn = QtWidgets.QPushButton("Save Statistics (CSV)...")
        btn.clicked.connect(self.on_prep_save_stats)
        c1.addWidget(btn)
        c1.addStretch(1)
        c2 = column(row, "Per-pixel statistic image")
        self.prep_pixel_stat = compact(QtWidgets.QComboBox(), 18)
        for key, label in PIXEL_STATISTICS:
            self.prep_pixel_stat.addItem(label, key)
        c2.addWidget(self.prep_pixel_stat)
        btn = QtWidgets.QPushButton("Show Statistic Image")
        btn.clicked.connect(self.on_prep_pixel_stat)
        c2.addWidget(btn)
        c2.addStretch(1)
        c3 = column(row, "Result", 2)
        self.prep_stats_status = self._status_label("Statistics are computed over valid (non-NaN) values.")
        c3.addWidget(self.prep_stats_status)
        c3.addStretch(1)

        # ---------- history / undo / export (always visible) ----------
        bottom = QtWidgets.QHBoxLayout()
        self.prep_history_label = QtWidgets.QLabel("Applied steps: none")
        self.prep_history_label.setWordWrap(True)
        bottom.addWidget(self.prep_history_label, 1)
        self.prep_keep_undo = QtWidgets.QCheckBox("Keep undo copy")
        self.prep_keep_undo.setChecked(True)
        self.prep_keep_undo.setToolTip("Keeps the previous cube in memory so the last step can be undone")
        bottom.addWidget(self.prep_keep_undo)
        self.btn_prep_undo = QtWidgets.QPushButton("Undo Last Step")
        self.btn_prep_undo.setEnabled(False)
        self.btn_prep_undo.clicked.connect(self.on_prep_undo)
        bottom.addWidget(self.btn_prep_undo)
        btn = QtWidgets.QPushButton("Export Active Cube...")
        btn.setToolTip("Save the preprocessed cube as ENVI (.hdr + .img) or NumPy (.npz)")
        btn.clicked.connect(self.on_prep_export_cube)
        bottom.addWidget(btn)
        outer.addLayout(bottom)

        self._prep_norm_method_changed()
        self.tabs.addTab(prep_tab, "Preprocessing")

    # ---------- small helpers ----------
    def _on_tab_changed(self, index):
        if self.tabs.tabText(index) == "Preprocessing":
            self._prep_refresh_info()

    def _prep_need_cube(self, need_wl=False):
        if self.cube is None:
            QtWidgets.QMessageBox.warning(self, "No data", "Load a hyperspectral cube first.")
            return False
        if need_wl and not self._prep_has_wl():
            QtWidgets.QMessageBox.warning(self, "Wavelengths needed",
                                          "This operation needs wavelength metadata for the bands.")
            return False
        return True

    def _prep_has_wl(self):
        return (self.cube is not None and self.wavelengths is not None
                and len(self.wavelengths) == self.cube.shape[2]
                and np.any(np.isfinite(self.wavelengths)))

    def _prep_refresh_info(self):
        if not hasattr(self, "prep_value_info"):
            return
        if self.cube is None:
            self.prep_value_info.setText("Value range: load a cube first")
            self.prep_wl_info.setText("Wavelengths: load a cube first")
            return
        rows, cols, bands = self.cube.shape
        step = max(1, int(np.sqrt(rows * cols / 200000.0)))
        with _quiet():
            sub = np.asarray(self.cube[::step, ::step, :], dtype=np.float64)
            p2, p50, p98 = np.nanpercentile(sub, [2, 50, 98]) if np.isfinite(sub).any() else (np.nan,) * 3
        hint = ""
        if np.isfinite(p98):
            if 100 < p98 <= 20000 and p2 >= -1000:
                hint = "  Looks like scaled reflectance (x10000) or DN."
            elif 1.5 < p98 <= 100 and p2 >= -5:
                hint = "  Could be reflectance in %."
            elif -0.2 <= p2 and p98 <= 1.5:
                hint = "  Looks like 0-1 reflectance."
        self.prep_value_info.setText(f"Value range (2-98 %): {p2:.4g} to {p98:.4g}, median {p50:.4g}.{hint}")
        if self._prep_has_wl():
            wl = np.asarray(self.wavelengths, dtype=float)
            f = wl[np.isfinite(wl)]
            spacing = np.median(np.abs(np.diff(np.sort(f)))) if f.size > 1 else float("nan")
            self.prep_wl_info.setText(f"Wavelengths: {f.min():.4g} to {f.max():.4g} nm, {bands} bands, "
                                      f"median spacing {spacing:.3g} nm.")
        else:
            self.prep_wl_info.setText(f"Wavelengths: none ({bands} bands; band numbers are used).")
        self._prep_update_history_label()

    def _prep_update_history_label(self):
        if self.prep_history:
            self.prep_history_label.setText("Applied steps: " + "  ->  ".join(self.prep_history))
        else:
            self.prep_history_label.setText("Applied steps: none")
        self.btn_prep_undo.setEnabled(self._prep_undo is not None)

    def _prep_sample_pixels(self):
        """Up to 5 valid pixels (and the pending ROI mask) used for before / after plots."""
        rows, cols, bands = self.cube.shape
        rng = np.random.default_rng(7)
        flat_idx = rng.choice(rows * cols, size=min(rows * cols, 4000), replace=False)
        spectra = np.asarray(self.cube.reshape(-1, bands)[flat_idx], dtype=float)
        valid = np.isfinite(spectra).sum(axis=1) > bands // 2
        picked = flat_idx[valid][:5]
        roi_mask = self.pending_sel["mask"] if self.pending_sel is not None else None
        return picked, roi_mask

    def _prep_spectra_at(self, picked, roi_mask):
        bands = self.cube.shape[2]
        x, xlabel = self._x_axis(np.ones(bands, dtype=bool))
        series = [(f"pixel ({i // self.cube.shape[1]}, {i % self.cube.shape[1]})",
                   np.asarray(self.cube.reshape(-1, bands)[i], dtype=float)) for i in picked]
        if roi_mask is not None and roi_mask.shape == self.cube.shape[:2] and roi_mask.any():
            with _quiet():
                series.insert(0, ("ROI mean", np.nanmean(np.asarray(self.cube[roi_mask], dtype=float), axis=0)))
        return np.asarray(x, dtype=float), xlabel, series

    def _prep_apply(self, new_cube, new_wl, label, status_label=None):
        """Replace the active cube with a preprocessed one, keeping undo + before/after data."""
        picked, roi_mask = self._prep_sample_pixels()
        before = self._prep_spectra_at(picked, roi_mask)
        bands_before = self.cube.shape[2]
        undo = (self.cube, None if self.wavelengths is None else np.asarray(self.wavelengths).copy(),
                list(self.prep_history), list(self.processing_history),
                self.radiometric_stage) if self.prep_keep_undo.isChecked() else None
        self._apply_cube_bands(new_cube, new_wl)
        self._prep_undo = undo
        self.prep_history.append(label)
        self._record_step(label, {"tab": "Preprocessing"}, bands_before)
        self.radiometric_band_spin.setMaximum(max(1, self.cube.shape[2]))
        self._prep_compare = (before, self._prep_spectra_at(picked, roi_mask), label)
        self.btn_prep_compare.setEnabled(True)
        self._prep_refresh_info()
        r, c, b = self.cube.shape
        msg = f"{label} applied. Active cube: {r} x {c} x {b}."
        if status_label is not None:
            status_label.setText(msg)
        self.info_label.setText(msg + " Analysis results that depended on the old values were cleared.")

    def _prep_run(self, fn, *args, **kwargs):
        """Run a backend function with a wait cursor; returns None after reporting an error."""
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            return fn(*args, **kwargs)
        except MemoryError:
            QtWidgets.QMessageBox.critical(self, "Not enough memory",
                                           "Not enough memory for this operation on the full cube.")
        except (ValueError, HSILoadError) as e:
            QtWidgets.QMessageBox.warning(self, "Cannot run", str(e))
        except Exception as e:
            traceback.print_exc()
            QtWidgets.QMessageBox.critical(self, "Error", f"Operation failed:\n{e}")
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        return None

    # ---------- 1. normalisation ----------
    def _prep_norm_method_changed(self, *_):
        is_minmax = self.prep_norm_method.currentData() == "minmax"
        for w in (self.prep_minmax_lo, self.prep_minmax_hi, self.prep_minmax_scope, self.prep_minmax_robust):
            w.setEnabled(is_minmax)

    def on_prep_normalize(self):
        if not self._prep_need_cube():
            return
        method = self.prep_norm_method.currentData()
        lo, hi = self.prep_minmax_lo.value(), self.prep_minmax_hi.value()
        if method == "minmax" and hi <= lo:
            QtWidgets.QMessageBox.warning(self, "Invalid range", "The upper value must be larger than the lower value.")
            return
        scope = self.prep_minmax_scope.currentData()
        robust = self.prep_minmax_robust.isChecked()
        out = self._prep_run(apply_normalization, self.cube, method, lo, hi, scope, robust)
        if out is None:
            return
        if method == "minmax":
            label = f"Min-max [{lo:g}, {hi:g}] {self.prep_minmax_scope.currentText().lower()}" + (" robust" if robust else "")
        else:
            label = {"snv": "SNV", "zscore": "Z-score (per band)", "l2": "L2 norm"}[method]
        self._prep_apply(out, self.wavelengths, label, self.prep_norm_status)

    def on_prep_plot_compare(self):
        if self._prep_compare is None:
            QtWidgets.QMessageBox.information(self, "Nothing to compare", "Apply a preprocessing step first.")
            return
        (xb, xlab_b, sb), (xa, xlab_a, sa), label = self._prep_compare

        def build(fig):
            ax1 = fig.add_subplot(211)
            ax2 = fig.add_subplot(212)
            for name, y in sb:
                ax1.plot(xb, y, label=name, lw=2.2 if name == "ROI mean" else 1.0)
            for name, y in sa:
                ax2.plot(xa, y, label=name, lw=2.2 if name == "ROI mean" else 1.0)
            ax1.set_title("Before")
            ax2.set_title(f"After: {label}")
            ax1.set_xlabel(xlab_b)
            ax2.set_xlabel(xlab_a)
            for ax in (ax1, ax2):
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=7)
            fig.tight_layout()
        self._show_figure_dialog(f"Before / after - {label}", build, size=(9, 8))

    def on_prep_undo(self):
        if self._prep_undo is None:
            return
        cube, wl, history, proc_history, stage = self._prep_undo
        self._prep_undo = None
        self._apply_cube_bands(cube, wl)
        self.prep_history = history
        self.processing_history = proc_history          # keep the shared history in step with the cube
        self._refresh_history_bar()
        self.radiometric_stage = stage
        self.radiometric_stage_label.setText(f"Current data stage: {stage}")
        self.radiometric_band_spin.setMaximum(max(1, self.cube.shape[2]))
        self._prep_compare = None
        self.btn_prep_compare.setEnabled(False)
        self._prep_refresh_info()
        self.info_label.setText("Last preprocessing step undone.")

    # ---------- 2. units ----------
    def _prep_unit_combo_changed(self, *_):
        factor, offset = self.prep_unit_combo.currentData()
        custom = factor is None
        self.prep_unit_factor.setEnabled(custom)
        self.prep_unit_offset.setEnabled(custom)
        if not custom:
            self.prep_unit_factor.setValue(factor)
            self.prep_unit_offset.setValue(offset)

    def on_prep_convert_values(self):
        if not self._prep_need_cube():
            return
        factor, offset = self.prep_unit_factor.value(), self.prep_unit_offset.value()
        if factor == 0:
            QtWidgets.QMessageBox.warning(self, "Invalid factor", "The factor must not be zero.")
            return
        out = self._prep_run(scale_values, self.cube, factor, offset)
        if out is None:
            return
        name = self.prep_unit_combo.currentText().split(":")[0].strip()
        label = (f"Values x {factor:g}" + (f" + {offset:g}" if offset else "")
                 if self.prep_unit_combo.currentData()[0] is None else name)
        self._prep_apply(out, self.wavelengths, label, self.prep_units_status)

    def on_prep_convert_wavelengths(self):
        if not self._prep_need_cube(need_wl=True):
            return
        unit = self.prep_wl_unit_combo.currentData()
        if unit == "nm":
            QtWidgets.QMessageBox.information(self, "Nothing to do", "The wavelengths are already in nanometres.")
            return
        new_wl = wavelengths_to_nm(self.wavelengths, unit)
        f = new_wl[np.isfinite(new_wl)]
        if f.size < 2:
            QtWidgets.QMessageBox.warning(self, "Invalid result", "The conversion produced no valid wavelengths.")
            return
        if f.min() < 100 or f.max() > 1e6:
            reply = QtWidgets.QMessageBox.question(
                self, "Unusual result",
                f"The converted wavelengths would run from {f.min():.4g} to {f.max():.4g} nm, which is "
                "unusual for imaging spectroscopy. Apply anyway?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No, QtWidgets.QMessageBox.No)
            if reply != QtWidgets.QMessageBox.Yes:
                return
        cube, new_wl = sort_bands(self.cube, new_wl)
        self._prep_apply(cube, new_wl, f"Wavelengths {unit} -> nm", self.prep_units_status)

    # ---------- 3. common wavelength grid ----------
    def _prep_grid_source_changed(self, *_):
        uniform = self.prep_grid_source.currentIndex() == 0
        self.prep_grid_uniform.setVisible(uniform)
        self.btn_prep_grid_fill.setVisible(uniform)
        self.btn_prep_grid_file.setVisible(not uniform)
        self.prep_grid_file_label.setVisible(not uniform)

    def _prep_fill_grid_from_cube(self):
        if not self._prep_need_cube(need_wl=True):
            return
        f = np.asarray(self.wavelengths, dtype=float)
        f = f[np.isfinite(f)]
        self.prep_grid_start.setValue(float(np.ceil(f.min())))
        self.prep_grid_stop.setValue(float(np.floor(f.max())))

    def on_prep_load_grid_file(self):
        settings = QtCore.QSettings(SETTINGS_ORG, APP_NAME)
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load target wavelength grid", str(settings.value("last_dir", "")),
            "Wavelength lists (*.hdr *.txt *.csv *.dat *.wvl);;All files (*)")
        if not path:
            return
        try:
            wl, fwhm = read_wavelength_grid(path)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Cannot read grid", str(e))
            return
        self._prep_grid_file = (wl, fwhm, Path(path).name)
        self.prep_grid_file_label.setText(
            f"{Path(path).name}: {wl.size} bands, {wl.min():.4g} to {wl.max():.4g} nm"
            + (", with FWHM" if fwhm is not None else ", no FWHM"))

    def on_prep_resample_grid(self):
        if not self._prep_need_cube(need_wl=True):
            return
        fwhm = None
        if self.prep_grid_source.currentIndex() == 0:
            try:
                target = uniform_grid(self.prep_grid_start.value(), self.prep_grid_stop.value(),
                                      self.prep_grid_step.value())
            except ValueError as e:
                QtWidgets.QMessageBox.warning(self, "Invalid grid", str(e))
                return
            grid_name = (f"{self.prep_grid_start.value():g}-{self.prep_grid_stop.value():g} nm "
                         f"step {self.prep_grid_step.value():g}")
        else:
            if self._prep_grid_file is None:
                QtWidgets.QMessageBox.information(self, "No grid file", "Load a grid file first.")
                return
            target, fwhm, grid_name = self._prep_grid_file
        if self.prep_grid_fwhm.value() > 0:
            fwhm = self.prep_grid_fwhm.value()
        wl = np.asarray(self.wavelengths, dtype=float)
        f = wl[np.isfinite(wl)]
        inside = (target >= f.min() - 1e-9) & (target <= f.max() + 1e-9)
        if inside.sum() < 2:
            QtWidgets.QMessageBox.warning(
                self, "No overlap", f"The target grid does not overlap the cube's wavelengths "
                                    f"({f.min():.4g} to {f.max():.4g} nm).")
            return
        dropped = int((~inside).sum())
        target = target[inside]
        if fwhm is not None and np.size(fwhm) == inside.size:
            fwhm = np.asarray(fwhm)[inside]
        method = self.prep_grid_method.currentData()
        out = self._prep_run(resample_spectral, self.cube, wl, target, method, fwhm)
        if out is None:
            return
        old_n = self.cube.shape[2]
        self._prep_apply(out, target, f"Grid {grid_name} ({method})", None)
        note = f" {dropped} target bands outside the cube's range were dropped." if dropped else ""
        self.prep_grid_status.setText(
            f"Resampled {old_n} -> {target.size} bands with {self.prep_grid_method.currentText().lower()}.{note}")

    # ---------- 4. spectral features ----------
    def _prep_preset_changed(self, *_):
        kind, waves = self.prep_feat_preset.currentData()
        if kind is None:
            return
        idx = self.prep_feat_kind.findData(kind)
        self.prep_feat_kind.blockSignals(True)
        self.prep_feat_kind.setCurrentIndex(idx)
        self.prep_feat_kind.blockSignals(False)
        self._prep_feat_kind_changed(keep_preset=True)
        for spin, v in zip((self.prep_feat_w1, self.prep_feat_w2, self.prep_feat_w3), waves):
            spin.setValue(v)
        if kind == "gap" and "position" in self.prep_feat_preset.currentText().lower():
            self.prep_feat_output.setCurrentIndex(1)      # red-edge position: report the wavelength

    def _prep_feat_kind_changed(self, *_, keep_preset=False):
        kind = self.prep_feat_kind.currentData()
        unit = "nm" if self._prep_has_wl() or self.cube is None else "band #"
        labels = {"nd": ("L1", "L2", None), "ratio": ("L1", "L2", None), "diff": ("L1", "L2", None),
                  "depth": ("Left shoulder", "Centre", "Right shoulder"),
                  "gap": ("Window start", "Window end", None)}[kind]
        for lab, spin, text in zip((self.prep_feat_lab1, self.prep_feat_lab2, self.prep_feat_lab3),
                                   (self.prep_feat_w1, self.prep_feat_w2, self.prep_feat_w3), labels):
            lab.setVisible(text is not None)
            spin.setVisible(text is not None)
            if text is not None:
                lab.setText(f"{text} ({unit})")
        self.prep_feat_deepest.setVisible(kind == "depth")
        self.prep_feat_output.clear()
        if kind == "depth":
            self.prep_feat_output.addItems(["Band depth (1 - R / continuum)", "Wavelength of deepest point (nm)"])
        elif kind == "gap":
            self.prep_feat_output.addItems(["Band gap energy (eV)", "Edge / inflection wavelength (nm)"])
        else:
            self.prep_feat_output.addItem("Index value")
        self.prep_feat_output.setEnabled(kind in ("depth", "gap"))
        if not keep_preset:
            self.prep_feat_preset.blockSignals(True)
            self.prep_feat_preset.setCurrentIndex(0)
            self.prep_feat_preset.blockSignals(False)

    def _prep_band_text(self, idx):
        if self._prep_has_wl():
            return f"band {idx} ({self.wavelengths[idx]:.1f} nm)"
        return f"band {idx}"

    def on_prep_compute_feature(self):
        if not self._prep_need_cube():
            return
        kind = self.prep_feat_kind.currentData()
        has_wl = self._prep_has_wl()
        nb = self.cube.shape[2]
        wl = np.asarray(self.wavelengths, dtype=float) if has_wl else np.arange(nb, dtype=float)
        v1, v2, v3 = self.prep_feat_w1.value(), self.prep_feat_w2.value(), self.prep_feat_w3.value()
        name = (self.prep_feat_preset.currentText().split(" (")[0]
                if self.prep_feat_preset.currentIndex() > 0 else self.prep_feat_kind.currentText().split("  ")[0])
        out_choice = self.prep_feat_output.currentIndex()
        if not has_wl and max(v1, v2, (v3 if kind == "depth" else 0)) >= nb:
            QtWidgets.QMessageBox.warning(
                self, "No wavelengths",
                f"This cube has no wavelength metadata, so the inputs are band numbers (0 to {nb - 1}).")
            return

        def compute():
            if kind in ("nd", "ratio", "diff"):
                i1, i2 = nearest_band(wl, v1), nearest_band(wl, v2)
                if i1 == i2:
                    raise ValueError(f"Both inputs snap to the same {self._prep_band_text(i1)}.")
                a = self.cube[:, :, i1]
                b = self.cube[:, :, i2]
                if kind == "nd":
                    m = normalized_difference(a, b)
                elif kind == "ratio":
                    m = band_ratio(a, b)
                else:
                    m = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
                return m, f"L1 = {self._prep_band_text(i1)}, L2 = {self._prep_band_text(i2)}", "", \
                    ("RdYlGn" if kind == "nd" else "viridis")
            if kind == "depth":
                depth, pos, (wL, wC, wR) = band_depth(self.cube, wl, v1, v2, v3,
                                                      deepest=self.prep_feat_deepest.isChecked())
                where = "deepest point" if self.prep_feat_deepest.isChecked() else f"{wC:.1f}"
                desc = f"shoulders {wL:.1f} / {wR:.1f}, depth at {where}"
                if out_choice == 1:
                    return pos, desc, "nm", "viridis"
                return depth, desc, "", "magma"
            eg, pos, (wa, wb) = absorption_edge(self.cube, wl, v1, v2)
            desc = f"edge searched between {wa:.1f} and {wb:.1f}"
            if out_choice == 1:
                return pos, desc, "nm", "viridis"
            return eg, desc, "eV", "plasma"

        res = self._prep_run(compute)
        if res is None:
            return
        m, desc, units, cmap = res
        title = name + (f" [{self.prep_feat_output.currentText()}]" if kind in ("depth", "gap") else "")
        self._prep_show_map(m, title, units, cmap)
        self.prep_feat_status.setText(f"{title}: {desc}. " + self._prep_map_summary(m, units))

    def _prep_map_summary(self, m, units=""):
        f = m[np.isfinite(m)]
        if f.size == 0:
            return "No valid values (check the wavelengths)."
        u = f" {units}" if units else ""
        return (f"min {f.min():.4g}, mean {f.mean():.4g}, max {f.max():.4g}{u}; "
                f"{m.size - f.size} NaN pixels.")

    def _prep_show_map(self, m, title, units="", cmap="viridis"):
        self._prep_map = (np.asarray(m, dtype=np.float64), title, units, cmap)
        self.image_canvas.show_image(self.robust_normalize(m), cmap=cmap)
        self.band_label.setText(f"{title} (display stretched 2-98 %; move the band slider to return)")
        self._draw_roi_patch()
        self.info_label.setText(f"{title}: " + self._prep_map_summary(m, units))

    def on_prep_show_map_dialog(self):
        if self._prep_map is None:
            QtWidgets.QMessageBox.information(self, "No image", "Compute a feature or statistic image first.")
            return
        m, title, units, cmap = self._prep_map
        f = m[np.isfinite(m)]
        if f.size == 0:
            QtWidgets.QMessageBox.information(self, "No valid values", "The image contains only NaN values.")
            return
        vmin, vmax = np.percentile(f, [2, 98])
        if vmax <= vmin:
            vmin, vmax = f.min(), f.max() if f.max() > f.min() else f.min() + 1

        def build(fig):
            gs = fig.add_gridspec(1, 2, width_ratios=[3, 2])
            ax = fig.add_subplot(gs[0])
            im = ax.imshow(m, cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_title(title)
            ax.set_axis_off()
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=units or "value")
            axh = fig.add_subplot(gs[1])
            axh.hist(f, bins=100, color="#4c72b0")
            axh.set_title("Histogram")
            axh.set_xlabel(units or "value")
            axh.set_ylabel("pixels")
            axh.grid(True, alpha=0.3)
            fig.tight_layout()
        self._show_figure_dialog(title, build, size=(11, 5.5))

    def on_prep_save_map(self):
        if self._prep_map is None:
            QtWidgets.QMessageBox.information(self, "No image", "Compute a feature or statistic image first.")
            return
        m, title, units, _ = self._prep_map
        default = re.sub(r"[^A-Za-z0-9_-]+", "_", title).strip("_")[:60] or "feature"
        path, filt = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save image", f"{default}.npy", "NumPy array (*.npy);;TIFF float32 (*.tif);;CSV (*.csv)")
        if not path:
            return
        try:
            low = path.lower()
            if low.endswith(".tif") or low.endswith(".tiff") or "TIFF" in filt:
                tf = _try_import("tifffile")
                if tf is None:
                    raise RuntimeError("Saving TIFF needs tifffile:  pip install tifffile")
                if not (low.endswith(".tif") or low.endswith(".tiff")):
                    path += ".tif"
                tf.imwrite(path, m.astype(np.float32))
            elif low.endswith(".csv") or "CSV" in filt:
                if not low.endswith(".csv"):
                    path += ".csv"
                np.savetxt(path, m, delimiter=",", fmt="%.6g")
            else:
                if not low.endswith(".npy"):
                    path += ".npy"
                np.save(path, m)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to save image:\n{e}")
            return
        self.info_label.setText(f"Saved '{title}' to {path}")

    # ---------- 5. statistics ----------
    def _prep_regions(self):
        """List of (name, mask or None for the whole image), or None after telling the user why not."""
        choice = self.prep_stats_region.currentIndex()
        shape = self.cube.shape[:2]
        if choice == 0:
            return [("Whole image", None)]
        if choice == 1:
            if self.pending_sel is None or self.pending_sel["mask"].shape != shape:
                QtWidgets.QMessageBox.information(self, "No selection",
                                                  "Draw a ROI on the image first (ROI Analysis tab).")
                return None
            return [("Current selection", self.pending_sel["mask"])]
        rois = [r for r in self.rois if r["mask"].shape == shape]
        if not rois:
            QtWidgets.QMessageBox.information(self, "No saved ROIs", "Save at least one ROI first (ROI Analysis tab).")
            return None
        if choice == 2:
            return [(f"{r['name']} ({r['class']})", r["mask"]) for r in rois]
        classes = {}
        for r in rois:
            classes[r["class"]] = classes.get(r["class"], np.zeros(shape, dtype=bool)) | r["mask"]
        return list(classes.items())

    def _prep_compute_region_stats(self):
        regions = self._prep_regions()
        if regions is None:
            return None
        bands = self.cube.shape[2]

        def compute():
            results = []
            for name, mask in regions:
                S = self.cube.reshape(-1, bands) if mask is None else self.cube[mask]
                valid = np.isfinite(S).any(axis=1)
                if not valid.all():
                    S = S[valid]
                if S.shape[0] == 0:
                    continue
                results.append((name, int(S.shape[0]), band_statistics(S)))
            return results

        results = self._prep_run(compute)
        if not results:
            if results is not None:
                QtWidgets.QMessageBox.information(self, "No valid pixels", "The region contains no valid pixels.")
            return None
        return results

    def on_prep_band_stats(self):
        if not self._prep_need_cube():
            return
        results = self._prep_compute_region_stats()
        if results is None:
            return
        x, xlabel = self._x_axis(np.ones(self.cube.shape[2], dtype=bool))
        x = np.asarray(x, dtype=float)
        colors = class_colors(max(1, len(results)))
        shown = results[:10]

        def build(fig):
            ax = fig.add_subplot(111)
            for (name, n, st), col in zip(shown, colors):
                ax.plot(x, st["mean"], color=col, lw=1.8, label=f"{name} mean (n={n})")
                ax.fill_between(x, st["mean"] - st["std"], st["mean"] + st["std"], color=col, alpha=0.25,
                                label="+/- 1 std" if len(shown) == 1 else None)
                if len(shown) == 1:
                    ax.fill_between(x, st["min"], st["max"], color=col, alpha=0.08, label="min - max range")
                    ax.plot(x, st["median"], color=col, lw=1.0, ls="--", label="median")
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Value")
            ax.set_title("Per-band statistics" + (" (first 10 regions)" if len(results) > 10 else ""))
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=7)
            fig.tight_layout()
        self._show_figure_dialog("Per-band statistics", build, size=(10, 6))
        lines = []
        for name, n, st in results[:6]:
            with _quiet():
                lines.append(f"{name}: {n} px, mean {np.nanmean(st['mean']):.4g}, "
                             f"mean std {np.nanmean(st['std']):.4g}, "
                             f"range {np.nanmin(st['min']):.4g} to {np.nanmax(st['max']):.4g}")
        self.prep_stats_status.setText("\n".join(lines) + ("\n..." if len(results) > 6 else ""))

    def on_prep_save_stats(self):
        if not self._prep_need_cube():
            return
        results = self._prep_compute_region_stats()
        if results is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save statistics", "band_statistics.csv",
                                                        "CSV files (*.csv)")
        if not path:
            return
        bands = self.cube.shape[2]
        x, xlabel = self._x_axis(np.ones(bands, dtype=bool))
        cols = [np.arange(bands, dtype=float), np.asarray(x, dtype=float)]
        header = ["band", "wavelength_nm" if self._prep_has_wl() else "band_index"]
        for name, n, st in results:
            safe = name.replace(",", "_")
            for key in ("mean", "std", "min", "max", "range", "median", "n"):
                cols.append(np.asarray(st[key], dtype=float))
                header.append(f"{safe} {key}")
        try:
            np.savetxt(path, np.column_stack(cols), delimiter=",", header=",".join(header),
                       comments="", fmt="%.6g")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to save CSV:\n{e}")
            return
        note = " (medians from a 500 000-pixel sample)" if any(st["median_sampled"] for _, _, st in results) else ""
        self.prep_stats_status.setText(f"Saved statistics of {len(results)} region(s) to {path}{note}.")

    def on_prep_pixel_stat(self):
        if not self._prep_need_cube():
            return
        key = self.prep_pixel_stat.currentData()
        m = self._prep_run(pixel_statistic_map, self.cube, key)
        if m is None:
            return
        title = f"Per-pixel {self.prep_pixel_stat.currentText().lower()} over bands"
        self._prep_show_map(m, title, "", "viridis")
        self.prep_stats_status.setText(f"{title}: " + self._prep_map_summary(m))

    # ---------- export ----------
    def on_prep_export_cube(self):
        if not self._prep_need_cube():
            return
        path, filt = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export active cube", "preprocessed.hdr", "ENVI (*.hdr);;NumPy (*.npz)")
        if not path:
            return
        cube32 = np.asarray(self.cube, dtype=np.float32)
        history = self._history_text()
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            if path.lower().endswith(".npz") or "NumPy" in filt:
                if not path.lower().endswith(".npz"):
                    path += ".npz"
                np.savez(path, cube=cube32,
                         wavelengths=(np.asarray(self.wavelengths, dtype=float) if self._prep_has_wl()
                                      else np.array([])),
                         preprocessing=np.array(history))
            else:
                if not path.lower().endswith(".hdr"):
                    path += ".hdr"
                meta = {"description": f"{APP_NAME} {APP_VERSION} processed cube. Data stage: "
                                       f"{self.radiometric_stage}. Steps: {history}",
                        "data ignore value": "nan"}
                if self._prep_has_wl():
                    meta["wavelength"] = [f"{w:.6g}" for w in self.wavelengths]
                    meta["wavelength units"] = "Nanometers"
                spectral.envi.save_image(path, cube32, dtype=np.float32, interleave="bsq",
                                         ext=".img", force=True, metadata=meta)
        except Exception as e:
            traceback.print_exc()
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to export the cube:\n{e}")
            return
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        self.info_label.setText(f"Exported the active cube to {path}")

    def dragEnterEvent(self, event):
        md = event.mimeData()
        if md.hasUrls() and any(u.isLocalFile() for u in md.urls()):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        paths = [u.toLocalFile() for u in event.mimeData().urls() if u.isLocalFile()]
        if not paths:
            return
        event.acceptProposedAction()
        asd_files = [p for p in paths if p.lower().endswith(".asd")]
        if asd_files and len(asd_files) == len(paths):
            self.import_asd_files(asd_files)
            return
        headers = [p for p in paths if p.lower().endswith(".hdr")]
        self.load_cube_file(headers[0] if headers else paths[0])

    def show_formats_dialog(self):
        rows = "".join(
            f"<tr><td>{'&#10004;' if ok else '&#10008;'}</td><td><b>{name}</b></td><td>{what}</td></tr>"
            for name, what, ok in reader_status()
        )
        QtWidgets.QMessageBox.information(
            self, "Supported file formats",
            "<p>Formats are read by optional Python packages. "
            "&#10004; = installed, &#10008; = not installed.</p>"
            f"<table cellspacing='6'>{rows}</table>"
            "<p>Install missing readers with pip, e.g.<br>"
            "<code>pip install h5py scipy rasterio tifffile astropy netCDF4 zarr</code></p>"
            "<p>Zarr stores are folders: use File &gt; Open Zarr folder, drag the folder onto "
            "the window, or pick the zarr.json / .zgroup file inside it.</p>"
            "<p>ASD FieldSpec .asd spectra need no extra package: File &gt; Import ASD field spectra "
            "adds them as reference spectra for SAM and unmixing.</p>"
            "<p>For ENVI files you can open either the .hdr header or the data file. "
            "When a file holds several 3-D arrays you are asked which one to use; when the spectral "
            "axis cannot be identified from wavelengths you are asked which axis holds the bands.</p>"
        )

    def _set_load_progress(self, pct, msg=""):
        self.load_bar.setValue(int(max(0, min(100, pct))))
        if msg:
            self.load_bar_label.setText(msg)
            self.info_label.setText(f"Loading: {msg} ({int(pct)}%)")
        QtWidgets.QApplication.processEvents(QtCore.QEventLoop.ExcludeUserInputEvents)

    def load_cube_file(self, path):
        """Load a file with a progress bar in the status bar, then reset the analysis state."""
        global _LOAD_PROGRESS_CB
        if Path(path).suffix.lower() == ".asd":
            return self._load_cube_file_impl(path)
        self.load_bar.setValue(0)
        self.load_bar_label.setText(f"Loading {Path(path).name}")
        self.load_bar.setVisible(True)
        self.load_bar_label.setVisible(True)
        _LOAD_PROGRESS_CB = self._set_load_progress
        ok = False
        try:
            self._set_load_progress(1, f"Loading {Path(path).name}")
            self._load_cube_file_impl(path)
            ok = self.cube is not None
        finally:
            _LOAD_PROGRESS_CB = None
            if ok:
                self._set_load_progress(100, "Loaded")
                QtCore.QTimer.singleShot(1200, self._hide_load_bar)
            else:
                self._hide_load_bar()

    def _hide_load_bar(self):
        self.load_bar.setVisible(False)
        self.load_bar_label.setVisible(False)

    def _load_cube_file_impl(self, path):
        """Load any supported hyperspectral file and reset the analysis state."""
        if Path(path).suffix.lower() == ".asd":      # point spectra: import, keep the cube
            self.import_asd_files([path])
            return
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            self.info_label.setText(f"Loading {Path(path).name} ...")
            QtWidgets.QApplication.processEvents()
            result = load_hyperspectral(
                path, choose=self._choose_dataset_dialog, ask_axis=self._ask_band_axis_dialog
            )
        except HSILoadCancelled:
            self.info_label.setText("Loading cancelled.")
            return
        except MemoryError:
            QtWidgets.QMessageBox.critical(
                self, "Not enough memory",
                "The cube is too large to load into memory. Try a spatial subset of the file."
            )
            return
        except Exception as e:
            traceback.print_exc()
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to open file:\n{e}")
            return
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

        self._set_load_progress(80, "Setting up viewer")
        cube_arr = result["cube"]
        rows, cols, n_bands_file = cube_arr.shape
        wl_arr = result.get("wavelengths")

        band_all_nan = np.all(np.isnan(cube_arr), axis=(0, 1))
        if np.any(band_all_nan):
            bad_idxs = np.where(band_all_nan)[0]
            good_idxs = np.where(~band_all_nan)[0]
            if good_idxs.size == 0:
                QtWidgets.QMessageBox.critical(self, "Error", "All bands are all-NaN.")
                return

            cube_arr = cube_arr[:, :, good_idxs]
            QtWidgets.QMessageBox.information(
                self, "Dropped NaN-only bands",
                f"Some bands were entirely NaN and excluded:\n{bad_idxs.tolist()}"
            )

        if wl_arr is not None:
            wl_arr = np.asarray(wl_arr, dtype=float)
            wl_arr = wl_arr[~band_all_nan] if wl_arr.shape[0] == n_bands_file else None

        self.cube = cube_arr
        bands = self.cube.shape[2]
        self.wavelengths = wl_arr
        self.metadata = dict(result.get("metadata") or {})
        self._active_original_band_indices = np.arange(bands, dtype=int)
        self.radiometric_stage = "DN / unknown"
        self.processing_history = []
        self.original_cube = None            # set below, once wavelengths are parsed
        self._radiometric_preview_data = None
        self.prep_history = []
        self._prep_undo = None
        self._prep_compare = None
        self._prep_map = None

        self.band_slider.setEnabled(True)
        self.band_slider.setMinimum(0)
        self.band_slider.setMaximum(bands - 1)
        self.band_slider.setValue(bands // 2)

        self.spin_r.setMaximum(bands - 1)
        self.spin_g.setMaximum(bands - 1)
        self.spin_b.setMaximum(bands - 1)

        self.spin_nmf_b0.setMaximum(bands - 1)
        self.spin_nmf_b1.setMaximum(bands - 1)
        self.spin_nmf_b0.setValue(0)
        self.spin_nmf_b1.setValue(bands - 1)

        self.spin_r.setValue(int(bands * 0.7))
        self.spin_g.setValue(int(bands * 0.5))
        self.spin_b.setValue(int(bands * 0.3))

        self.pending_sel = None
        self._remove_roi_patch()
        self.roi_label.setText("ROI: none")
        self.last_roi_mean_full = None
        self.rois = []
        self.roi_counter = 0
        self._refresh_roi_list()
        self._refresh_class_combo()
        self.sam_angle_map = None
        self.class_names_result = None
        self.last_report = None

        self.anomaly_map = None
        self.class_labels = None
        self.class_rgb_map = None
        self.endmembers = []
        self.endmember_label.setText("Endmembers: 0")

        self.nmf_abundance = None
        self.nmf_endmembers = None
        self.nmf_band_range = None
        self.snr_band = None
        self.snr_pixel = None
        self.snr_label.setText("SNR: not computed")

        self.pca_scores = None
        self.pca_explained_ratio = None
        self.pca_valid_mask = None
        self.pca_n_components = 0
        self.pca_label.setText("PCA: not computed")

        self._update_band_management_range()
        self._band_removal_plot_data = None
        self.btn_plot_band_removal.setEnabled(False)
        self._band_resample_plot_data = None
        self.btn_plot_resample.setEnabled(False)
        self._selected_band_plot_data = None
        self.btn_plot_selected_bands.setEnabled(False)
        self.band_removal_status_label.setText("Wavelength ranges are inclusive. Cubes without wavelengths: assign them in column 0.")
        self.resample_status_label.setText("Resamples each pixel spectrum onto a uniform wavelength grid.")
        self.select_status_label.setText("Endmembers are estimated from representative pixel spectra; informative bands are selected from their spectral contrast.")
        self.select_nbands_spin.setMaximum(bands)
        self.update_image()
        self._prep_refresh_info()

        name = Path(path).name
        self.setWindowTitle(f"{APP_NAME} v{APP_VERSION} — {name} | {APP_CREDIT}")
        ds = f", dataset '{result['dataset']}'" if result.get("dataset") else ""
        if self.wavelengths is not None and len(self.wavelengths):
            wl_text = (f"Wavelengths {np.nanmin(self.wavelengths):.1f}–"
                       f"{np.nanmax(self.wavelengths):.1f} nm.")
        else:
            wl_text = "No wavelength metadata found (band numbers are used instead)."
        dtype_note = " Stored as float32 to save memory." if self.cube.dtype == np.float32 else ""
        self.info_label.setText(
            f"Loaded {result.get('format', '')} file '{name}'{ds}: {rows} x {cols} x {bands}. "
            f"{wl_text}{dtype_note} "
            "Use spectra, ROI, anomaly, band math, clustering, unmixing, or PCA."
        )
        self.radiometric_stage_label.setText(f"Current data stage: {self.radiometric_stage}")
        self.radiometric_band_spin.setMaximum(max(1, bands))
        self.radiometric_band_spin.setValue(max(1, min(bands, bands // 2 + 1)))
        self.btn_radiometric_preview.setEnabled(False)
        self._refresh_radiometric_parameter_fields()
        self._refresh_radiometric_metadata_status()
        # Keep the loaded cube so every later operation can be reverted.
        self.original_cube = self.cube
        self.original_wavelengths = None if self.wavelengths is None else self.wavelengths.copy()
        self.original_metadata = copy.deepcopy(self.metadata)
        self.ng_status_label.setText("Normalises the cube to [0, 1], then denoises with NGmeet.")
        self.btn_ng_compare.setEnabled(False)
        self._ng_last_result = None
        self.el_refs = []                    # reference ROIs belong to the previous image
        self._reset_el_result()
        self._refresh_el_list()

    # ------------------------------------------------------------------
    # Processing history / single commit point for operations that change the cube
    # ------------------------------------------------------------------
    def _record_step(self, step, params=None, bands_before=None):
        """Append one operation to the processing history (kept with exports / library entries)."""
        self.processing_history.append({
            "step": step,
            "params": params or {},
            "bands_before": None if bands_before is None else int(bands_before),
            "bands_after": int(self.cube.shape[2]) if self.cube is not None else None,
            "stage": self.radiometric_stage,
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        })
        self._refresh_history_bar()

    def _refresh_history_bar(self):
        lbl = getattr(self, "history_bar_label", None)
        if lbl is None:
            return
        n = len(self.processing_history)
        if n == 0:
            lbl.setText("History: no operations yet (cube as loaded)")
        else:
            lbl.setText(f"History: {n} operation(s); last: {self.processing_history[-1]['step']}")
        btn = getattr(self, "btn_revert_all", None)
        if btn is not None:
            btn.setEnabled(n > 0)

    def on_save_processing_history(self):
        if not self.processing_history:
            QtWidgets.QMessageBox.information(self, "Save history", "No processing steps have been applied yet.")
            return
        path, filt = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save processing history", "processing_history.json", "JSON (*.json);;Text (*.txt)")
        if not path:
            return
        try:
            if path.lower().endswith(".txt") or "Text" in filt:
                if not path.lower().endswith(".txt"):
                    path += ".txt"
                lines = [f"{APP_NAME} {APP_VERSION} processing history", f"Data stage: {self.radiometric_stage}", ""]
                for i, h in enumerate(self.processing_history, 1):
                    params = ", ".join(f"{k}={v}" for k, v in h["params"].items())
                    lines.append(f"{i}. {h['time']}  {h['step']}" + (f"  ({params})" if params else ""))
                with open(path, "w", encoding="utf-8") as f:
                    f.write("\n".join(lines) + "\n")
            else:
                if not path.lower().endswith(".json"):
                    path += ".json"
                payload = {"app": f"{APP_NAME} {APP_VERSION}", "stage": self.radiometric_stage,
                           "cube_shape": list(self.cube.shape) if self.cube is not None else None,
                           "steps": self.processing_history}
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(payload, f, indent=2, default=str)
        except Exception as e:
            traceback.print_exc()
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to save the history:\n{e}")
            return
        self.info_label.setText(f"Saved processing history to {path}")

    def _commit_cube(self, new_cube, new_wavelengths, step, params=None, stage=None):
        """Replace the active cube after an operation, update the UI and record the step."""
        bands_before = self.cube.shape[2] if self.cube is not None else None
        if stage:
            self.radiometric_stage = stage
            self.radiometric_stage_label.setText(f"Current data stage: {stage}")
        self._apply_cube_bands(new_cube, new_wavelengths)
        bands = self.cube.shape[2]
        self.radiometric_band_spin.setMaximum(max(1, bands))
        self.radiometric_band_spin.setValue(max(1, min(bands, self.radiometric_band_spin.value())))
        self._record_step(step, params, bands_before)

    def _history_text(self):
        """One-line summary of every step applied to the active cube (for exports)."""
        if not self.processing_history:
            return "none"
        return "; ".join(h["step"] for h in self.processing_history)

    def on_show_processing_history(self):
        if not self.processing_history:
            QtWidgets.QMessageBox.information(self, "Processing history", "No processing steps have been applied yet.")
            return
        lines = [f"Active data stage: {self.radiometric_stage}", ""]
        for i, h in enumerate(self.processing_history, 1):
            params = ", ".join(f"{k}={v}" for k, v in h["params"].items())
            bands = ""
            if h["bands_before"] is not None and h["bands_before"] != h["bands_after"]:
                bands = f"  [{h['bands_before']} -> {h['bands_after']} bands]"
            lines.append(f"{i}. {h['time']}  {h['step']}{bands}")
            if params:
                lines.append(f"     {params}")
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Processing history")
        lay = QtWidgets.QVBoxLayout(dlg)
        box = QtWidgets.QPlainTextEdit()
        box.setReadOnly(True)
        box.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap)
        box.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont))
        box.setPlainText("\n".join(lines))
        lay.addWidget(box)
        btn = QtWidgets.QPushButton("Close")
        btn.clicked.connect(dlg.accept)
        lay.addWidget(btn)
        dlg.resize(640, 420)
        dlg.exec_()

    def on_revert_to_original(self):
        if self.original_cube is None:
            QtWidgets.QMessageBox.information(self, "Nothing to revert", "Load a cube first.")
            return
        if not self.processing_history:
            QtWidgets.QMessageBox.information(self, "Nothing to revert", "The cube has not been modified.")
            return
        if self._busy:
            QtWidgets.QMessageBox.information(self, "Task running", "Wait for the running task to finish or stop it first.")
            return
        ans = QtWidgets.QMessageBox.question(
            self, "Revert to original",
            f"Discard {len(self.processing_history)} processing step(s) and restore the cube as loaded?\n"
            "Analysis results (PCA, classification, ...) are cleared; ROIs are kept.",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No, QtWidgets.QMessageBox.No)
        if ans != QtWidgets.QMessageBox.Yes:
            return
        self.metadata = copy.deepcopy(self.original_metadata)
        self.radiometric_stage = "DN / unknown"
        self.radiometric_stage_label.setText(f"Current data stage: {self.radiometric_stage}")
        self._radiometric_preview_data = None
        self.btn_radiometric_preview.setEnabled(False)
        self._apply_cube_bands(self.original_cube, self.original_wavelengths)
        bands = self.cube.shape[2]
        self.radiometric_band_spin.setMaximum(max(1, bands))
        self._refresh_radiometric_parameter_fields()
        self._refresh_radiometric_metadata_status()
        self.processing_history = []
        self._ng_last_result = None
        self._refresh_history_bar()
        self.btn_ng_compare.setEnabled(False)
        self._reset_el_result()
        self.prep_history = []
        self._prep_undo = None
        self._prep_compare = None
        self.btn_prep_compare.setEnabled(False)
        self._prep_refresh_info()
        self._prep_update_history_label()
        self.info_label.setText(f"Reverted to the original cube ({self.cube.shape[0]} x {self.cube.shape[1]} x {bands}).")

    def _wavelengths_ok(self):
        return (self.cube is not None and self.wavelengths is not None
                and len(self.wavelengths) == self.cube.shape[2])

    def _require_wavelengths(self, feature):
        """True when the cube has usable wavelengths; otherwise offer to assign them right now."""
        if self._wavelengths_ok():
            return True
        reply = QtWidgets.QMessageBox.question(
            self, "Wavelengths needed",
            f"{feature} needs a wavelength for every band, and this file did not provide them.\n\n"
            "Assign wavelengths now (from a file, a uniform range or a pasted list)?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No, QtWidgets.QMessageBox.Yes)
        if reply != QtWidgets.QMessageBox.Yes:
            return False
        self.on_assign_wavelengths()
        return self._wavelengths_ok()

    def on_assign_wavelengths(self):
        """Dialog: give the cube a wavelength per band, whatever format it was loaded from."""
        if self.cube is None:
            QtWidgets.QMessageBox.warning(self, "No data", "Load a hyperspectral cube first.")
            return
        nb = self.cube.shape[2]
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Assign wavelengths")
        lay = QtWidgets.QVBoxLayout(dlg)
        lay.addWidget(QtWidgets.QLabel(f"The cube has {nb} bands. Provide one wavelength per band."))
        src = QtWidgets.QComboBox()
        src.addItems(["Uniform range (first band to last band)", "From file (.hdr, .txt, .csv, .wvl)",
                      "Paste a list"])
        lay.addWidget(src)
        form = QtWidgets.QFormLayout()
        start = self._dspin(0, 1e7, 400.0, 3, 1.0)
        stop = self._dspin(0, 1e7, 1000.0, 3, 1.0)
        form.addRow("First band:", start)
        form.addRow("Last band:", stop)
        uni_w = QtWidgets.QWidget(); uni_w.setLayout(form)
        lay.addWidget(uni_w)
        file_w = QtWidgets.QWidget()
        fl = QtWidgets.QHBoxLayout(file_w); fl.setContentsMargins(0, 0, 0, 0)
        btn_file = QtWidgets.QPushButton("Choose file...")
        file_lbl = QtWidgets.QLabel("no file")
        file_lbl.setWordWrap(True)
        fl.addWidget(btn_file); fl.addWidget(file_lbl, 1)
        lay.addWidget(file_w)
        paste = QtWidgets.QPlainTextEdit()
        paste.setPlaceholderText("One wavelength per band, separated by commas, spaces or new lines")
        lay.addWidget(paste)
        unit_row = QtWidgets.QHBoxLayout()
        unit_row.addWidget(QtWidgets.QLabel("Values are in:"))
        unit = QtWidgets.QComboBox()
        for key, label in WAVELENGTH_UNITS:
            unit.addItem(label, key)
        unit_row.addWidget(unit, 1)
        lay.addLayout(unit_row)
        loaded = {"wl": None}

        def pick_file():
            path, _ = QtWidgets.QFileDialog.getOpenFileName(
                dlg, "Wavelength list", "", "Wavelength lists (*.hdr *.txt *.csv *.dat *.wvl);;All files (*)")
            if not path:
                return
            try:
                wl, _f = read_wavelength_grid(path)
            except Exception as e:
                QtWidgets.QMessageBox.warning(dlg, "Cannot read file", str(e))
                return
            loaded["wl"] = np.asarray(wl, dtype=float)
            file_lbl.setText(f"{Path(path).name}: {wl.size} values")
        btn_file.clicked.connect(pick_file)

        def mode_changed(i):
            uni_w.setVisible(i == 0); file_w.setVisible(i == 1); paste.setVisible(i == 2)
        src.currentIndexChanged.connect(mode_changed)
        mode_changed(0)
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        bb.accepted.connect(dlg.accept); bb.rejected.connect(dlg.reject)
        lay.addWidget(bb)
        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return
        try:
            i = src.currentIndex()
            if i == 0:
                wl = np.linspace(start.value(), stop.value(), nb)
                how = f"uniform {start.value():g}-{stop.value():g}"
            elif i == 1:
                if loaded["wl"] is None:
                    raise ValueError("Choose a wavelength file first.")
                wl = loaded["wl"]
                how = "file"
            else:
                wl = np.asarray([float(t) for t in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?",
                                                              paste.toPlainText())], dtype=float)
                how = "pasted list"
            if wl.size != nb:
                raise ValueError(f"{wl.size} wavelengths were given but the cube has {nb} bands.")
            wl = wavelengths_to_nm(wl, unit.currentData())
            if not np.all(np.isfinite(wl)):
                raise ValueError("The wavelengths contain invalid values.")
        except ValueError as e:
            QtWidgets.QMessageBox.warning(self, "Assign wavelengths", str(e))
            return
        cube = self.cube
        if np.any(np.diff(wl) < 0):                      # e.g. wavenumbers: restore increasing order
            cube, wl = sort_bands(self.cube, wl)
        bands_before = self.cube.shape[2]
        if cube is not self.cube:
            self._apply_cube_bands(cube, wl)
        else:
            self.wavelengths = np.asarray(wl, dtype=float)
        self._record_step("Assign wavelengths", {"source": how, "unit": str(unit.currentData()),
                                                 "range_nm": f"{wl.min():.4g}-{wl.max():.4g}"}, bands_before)
        self._update_band_management_range()
        self._prep_refresh_info()
        self.update_image()
        self.info_label.setText(f"Wavelengths assigned ({how}): {wl.min():.1f}-{wl.max():.1f} nm.")

    def _update_band_management_range(self):
        """Refresh the wavelength range shown in Band Management."""
        if self.cube is None:
            self.band_wl_source_label.setText("Load a hyperspectral cube first")
        elif self._wavelengths_ok():
            self.band_wl_source_label.setText("Wavelengths available for this cube.")
        else:
            self.band_wl_source_label.setText("This file has no wavelengths - assign them below.")
        if self.cube is None:
            self.band_wavelength_range_label.setText("Load a hyperspectral cube first")
            return
        if self.wavelengths is None or len(self.wavelengths) != self.cube.shape[2]:
            self.band_wavelength_range_label.setText("No wavelengths yet - use '0. Wavelengths' to assign them")
            return
        finite = np.asarray(self.wavelengths, dtype=float)
        finite = finite[np.isfinite(finite)]
        if finite.size:
            self.band_wavelength_range_label.setText(
                f"[{np.min(finite):g}, {np.max(finite):g}] nm  |  {len(self.wavelengths)} bands"
            )
        else:
            self.band_wavelength_range_label.setText("No valid wavelength values")

    def on_remove_bands(self):
        """Remove all bands whose wavelengths fall within any entered interval."""
        if self.cube is None:
            QtWidgets.QMessageBox.warning(self, "No data", "Load a hyperspectral cube first.")
            return
        if not self._require_wavelengths("Band removal by wavelength"):
            return

        raw = self.remove_wavelength_ranges_edit.text().strip()
        if not raw:
            QtWidgets.QMessageBox.information(self, "No ranges", "Enter at least one wavelength range.")
            return
        import re
        ranges = []
        try:
            for part in re.split(r"[;\n]+", raw):
                part = part.strip().strip("[]")
                if not part:
                    continue
                values = [float(v) for v in re.split(r"[\s,]+", part) if v]
                if len(values) != 2 or not np.all(np.isfinite(values)):
                    raise ValueError(f"Invalid interval: {part!r}. Use two numbers per interval.")
                lo, hi = values
                if lo > hi:
                    raise ValueError(f"Range start ({lo:g}) must not exceed range end ({hi:g}).")
                ranges.append((lo, hi))
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, "Invalid wavelength ranges", str(exc))
            return
        if not ranges:
            QtWidgets.QMessageBox.information(self, "No ranges", "Enter at least one valid wavelength range.")
            return

        original_wavelengths = np.asarray(self.wavelengths, dtype=float).copy()
        remove_mask = np.zeros(original_wavelengths.shape, dtype=bool)
        for lo, hi in ranges:
            remove_mask |= np.isfinite(original_wavelengths) & (original_wavelengths >= lo) & (original_wavelengths <= hi)
        keep_mask = ~remove_mask
        removed_count = int(remove_mask.sum())
        if removed_count == 0:
            QtWidgets.QMessageBox.information(
                self, "No matching bands", "No bands fall within the specified wavelength ranges."
            )
            return
        if not np.any(keep_mask):
            QtWidgets.QMessageBox.warning(
                self, "Cannot remove all bands", "These wavelength ranges would remove every spectral band."
            )
            return
        answer = QtWidgets.QMessageBox.question(
            self, "Confirm band removal",
            f"Remove {removed_count} of {len(original_wavelengths)} bands?\n"
            "This will update the active cube for subsequent analysis.",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.Cancel,
            QtWidgets.QMessageBox.Cancel,
        )
        if answer != QtWidgets.QMessageBox.Yes:
            return

        new_wavelengths = original_wavelengths[keep_mask]
        self._band_removal_plot_data = (original_wavelengths, new_wavelengths.copy())
        bands_before_removal = int(self.cube.shape[2])
        self.cube = self.cube[:, :, keep_mask].copy()
        self.wavelengths = new_wavelengths
        bands = self.cube.shape[2]

        # Keep viewer controls consistent with the updated spectral dimension.
        self.band_slider.blockSignals(True)
        self.band_slider.setMaximum(bands - 1)
        self.band_slider.setValue(min(self.band_slider.value(), bands - 1))
        self.band_slider.blockSignals(False)
        for spin in (self.spin_r, self.spin_g, self.spin_b, self.spin_nmf_b0, self.spin_nmf_b1):
            spin.setMaximum(bands - 1)
        self.spin_r.setValue(min(self.spin_r.value(), bands - 1))
        self.spin_g.setValue(min(self.spin_g.value(), bands - 1))
        self.spin_b.setValue(min(self.spin_b.value(), bands - 1))
        self.spin_nmf_b0.setValue(0)
        self.spin_nmf_b1.setValue(bands - 1)

        # Spectral results computed against the old band dimension are no longer valid.
        self.last_roi_mean_full = None
        self.endmembers = []
        self.endmember_label.setText("Endmembers: 0")
        self.pca_scores = None
        self.pca_explained_ratio = None
        self.pca_valid_mask = None
        self.pca_n_components = 0
        self.pca_label.setText("PCA: not computed")
        self.nmf_abundance = None
        self.nmf_endmembers = None
        self.nmf_band_range = None
        self.snr_band = None
        self.snr_pixel = None
        self.snr_label.setText("SNR: not computed")
        self._update_band_management_range()
        self.btn_plot_band_removal.setEnabled(True)
        self._record_step("Remove bands", {"ranges_nm": ranges, "removed": removed_count},
                          bands_before=bands_before_removal)
        self.update_image()
        self.band_removal_status_label.setText(
            f"Removed {removed_count} bands; {bands} bands remain. "
            f"Active cube dimensions: {self.cube.shape[0]} × {self.cube.shape[1]} × {bands}."
        )
        self.info_label.setText(f"Band removal complete: removed {removed_count} bands; {bands} remain.")

    def on_plot_band_removal(self):
        """Plot original and retained wavelengths against their band indices."""
        if self._band_removal_plot_data is None:
            QtWidgets.QMessageBox.information(self, "No band-removal result", "Remove bands first.")
            return
        original, new = self._band_removal_plot_data
        def build(fig):
            ax = fig.add_subplot(111)
            ax.plot(np.arange(1, len(original) + 1), original, "o", label="Original Values")
            ax.plot(np.arange(1, len(new) + 1), new, "or", label="New Values")
            ax.set_xlabel("Band Number")
            ax.set_ylabel("Wavelength (nm)")
            ax.legend(loc="lower right")
            ax.grid(True, alpha=0.25)
            fig.tight_layout()
        self._show_figure_dialog("Original vs New Wavelength Values", build, size=(9, 5))

    def _invalidate_spectral_results(self):
        """Clear analysis products whose spectral dimension no longer matches the active cube."""
        self.last_roi_mean_full = None
        self.endmembers = []
        self.endmember_label.setText("Endmembers: 0")
        self.pca_scores = None
        self.pca_explained_ratio = None
        self.pca_valid_mask = None
        self.pca_n_components = 0
        self.pca_label.setText("PCA: not computed")
        self.nmf_abundance = None
        self.nmf_endmembers = None
        self.nmf_band_range = None
        self.snr_band = None
        self.snr_pixel = None
        self.snr_label.setText("SNR: not computed")
        self.sam_angle_map = None
        self.anomaly_map = None
        self.class_labels = None
        self.class_rgb_map = None
        self.class_names_result = None
        self.last_report = None

    def _apply_cube_bands(self, new_cube, new_wavelengths):
        """Replace active cube/wavelengths and keep the main viewer controls in sync."""
        self.cube = np.asarray(new_cube)
        self.wavelengths = None if new_wavelengths is None else np.asarray(new_wavelengths, dtype=float)
        bands = self.cube.shape[2]
        self.band_slider.blockSignals(True)
        self.band_slider.setMinimum(0)
        self.band_slider.setMaximum(bands - 1)
        self.band_slider.setValue(min(self.band_slider.value(), bands - 1))
        self.band_slider.setEnabled(True)
        self.band_slider.blockSignals(False)
        for spin in (self.spin_r, self.spin_g, self.spin_b, self.spin_nmf_b0, self.spin_nmf_b1):
            spin.setMaximum(bands - 1)
        self.spin_r.setValue(min(self.spin_r.value(), bands - 1))
        self.spin_g.setValue(min(self.spin_g.value(), bands - 1))
        self.spin_b.setValue(min(self.spin_b.value(), bands - 1))
        self.spin_nmf_b0.setValue(0)
        self.spin_nmf_b1.setValue(bands - 1)
        self.select_nbands_spin.setMaximum(max(1, bands))
        self.select_nbands_spin.setValue(min(self.select_nbands_spin.value(), bands))
        self._invalidate_spectral_results()
        self._update_band_management_range()
        self.radiometric_band_spin.setMaximum(max(1, bands))
        self.update_image()
        self._prep_refresh_info()

    def on_resample_bands(self):
        """Resample spectra onto a uniform wavelength grid using the selected interpolator."""
        if self.cube is None:
            QtWidgets.QMessageBox.warning(self, "No data", "Load a hyperspectral cube first.")
            return
        if not self._require_wavelengths("Resampling"):
            return
        wl = np.asarray(self.wavelengths, dtype=float)
        if not np.all(np.isfinite(wl)):
            QtWidgets.QMessageBox.warning(self, "Invalid wavelengths", "Wavelengths must all be finite to resample.")
            return
        # Sort wavelengths before interpolation; apply the same order to data.
        order = np.argsort(wl)
        wl_sorted = wl[order]
        cube_sorted = self.cube[:, :, order]
        unique_wl, unique_idx = np.unique(wl_sorted, return_index=True)
        if len(unique_wl) < 2:
            QtWidgets.QMessageBox.warning(self, "Insufficient wavelengths", "At least two unique wavelengths are required.")
            return
        cube_sorted = cube_sorted[:, :, unique_idx]
        spacing = int(self.resample_spacing_spin.value())
        target_wl = np.arange(unique_wl[0], unique_wl[-1] + spacing * 1e-9, spacing, dtype=float)
        if target_wl.size < 2:
            QtWidgets.QMessageBox.warning(self, "Target spacing too large", "Choose a smaller wavelength spacing to create at least two bands.")
            return
        # Ensure the final grid never extends beyond the source wavelength range.
        target_wl = target_wl[target_wl <= unique_wl[-1] + 1e-9]
        if target_wl.size < 2:
            QtWidgets.QMessageBox.warning(self, "Target spacing too large", "Choose a smaller wavelength spacing to create at least two bands.")
            return
        method = self.resample_method_combo.currentText()
        flat = np.asarray(cube_sorted, dtype=np.float32).reshape(-1, cube_sorted.shape[2])
        out = np.empty((flat.shape[0], target_wl.size), dtype=np.float32)
        if method == "cubic":
            try:
                from scipy.interpolate import interp1d
                # Chunk both source data and interpolation to limit peak memory.
                for start in range(0, flat.shape[0], 20000):
                    block_interp = interp1d(
                        unique_wl, flat[start:start+20000], axis=1, kind="cubic",
                        bounds_error=False, fill_value="extrapolate", assume_sorted=True
                    )
                    out[start:start+20000] = block_interp(target_wl)
            except Exception as exc:
                QtWidgets.QMessageBox.warning(self, "Cubic interpolation unavailable", f"Cubic interpolation failed (requires SciPy and at least 4 unique wavelengths):\n{exc}")
                return
        elif method == "nearest":
            right = np.searchsorted(unique_wl, target_wl, side="left").clip(0, len(unique_wl)-1)
            left = np.maximum(right - 1, 0)
            choose_left = np.abs(target_wl - unique_wl[left]) <= np.abs(unique_wl[right] - target_wl)
            nearest = np.where(choose_left, left, right)
            out[:] = flat[:, nearest]
        else:  # bilinear in a 1-D spectrum reduces to linear interpolation along wavelength.
            for start in range(0, flat.shape[0], 20000):
                block = flat[start:start+20000]
                for j, row in enumerate(block):
                    finite = np.isfinite(row)
                    if finite.sum() < 2:
                        out[start+j] = np.nan
                    else:
                        out[start+j] = np.interp(target_wl, unique_wl[finite], row[finite])
        new_cube = out.reshape(self.cube.shape[0], self.cube.shape[1], target_wl.size)
        old_wl = wl.copy()
        self._band_resample_plot_data = (old_wl, target_wl.copy())
        bands_before_resample = int(self.cube.shape[2])
        self._apply_cube_bands(new_cube, target_wl)
        self._record_step("Resample bands", {"spacing_nm": spacing, "method": method},
                          bands_before=bands_before_resample)
        self.btn_plot_resample.setEnabled(True)
        self.resample_status_label.setText(
            f"Resampling complete using {method}. {len(old_wl)} original bands → {len(target_wl)} uniform bands; "
            f"spacing {spacing} nm. Active cube: {new_cube.shape[0]} × {new_cube.shape[1]} × {new_cube.shape[2]}."
        )
        self.info_label.setText("Spectral resampling complete.")

    def on_plot_resampled_bands(self):
        if self._band_resample_plot_data is None:
            QtWidgets.QMessageBox.information(self, "No resampling result", "Resample bands first.")
            return
        original, resampled = self._band_resample_plot_data
        def build(fig):
            ax = fig.add_subplot(111)
            ax.plot(np.arange(1, len(original)+1), original, "o", label="Original wavelengths")
            ax.plot(np.arange(1, len(resampled)+1), resampled, "or", label="Resampled wavelengths")
            ax.set_xlabel("Band Number")
            ax.set_ylabel("Wavelength (nm)")
            ax.set_title("Original vs Resampled Wavelength Grid")
            ax.legend(loc="best")
            ax.grid(True, alpha=0.25)
            fig.tight_layout()
        self._show_figure_dialog("Resampled Wavelength Grid", build, size=(9, 5))

    def on_select_bands(self):
        """Estimate representative endmembers with a fast PPI-style projection method and select informative bands."""
        if self.cube is None:
            QtWidgets.QMessageBox.warning(self, "No data", "Load a hyperspectral cube first.")
            return
        if not self._require_wavelengths("Band selection"):
            return
        n_end = int(self.select_endmembers_spin.value())
        n_select = int(self.select_nbands_spin.value())
        rows, cols, bands = self.cube.shape
        if n_select > bands:
            QtWidgets.QMessageBox.warning(self, "Invalid band count", f"Choose no more than {bands} bands.")
            return
        if bands < 2:
            QtWidgets.QMessageBox.warning(self, "Insufficient bands", "At least two bands are required.")
            return
        # Sample pixels to keep endmember estimation responsive on large cubes.
        flat = np.asarray(self.cube, dtype=np.float32).reshape(-1, bands)
        valid = np.all(np.isfinite(flat), axis=1)
        X = flat[valid]
        if X.shape[0] < max(10, n_end):
            QtWidgets.QMessageBox.warning(self, "Insufficient valid pixels", "Not enough finite pixel spectra to estimate endmembers.")
            return
        rng = np.random.default_rng(42)
        if len(X) > 50000:
            X = X[rng.choice(len(X), 50000, replace=False)]
        # Center and scale for random-projection purity scoring (PPI-style fast approximation).
        mean = X.mean(axis=0, keepdims=True)
        scale = X.std(axis=0, keepdims=True)
        scale[scale < 1e-8] = 1.0
        Z = (X - mean) / scale
        score = np.zeros(len(Z), dtype=np.float64)
        n_proj = 256
        for _ in range(n_proj):
            direction = rng.normal(size=bands)
            direction /= max(np.linalg.norm(direction), 1e-12)
            projection = Z @ direction
            score[int(np.argmin(projection))] += 1
            score[int(np.argmax(projection))] += 1
        # Candidate extreme spectra; greedily enforce distinctness in standardized spectral space.
        candidates = np.argsort(score)[::-1]
        chosen = []
        min_dist = 0.5 * np.sqrt(bands)
        for idx in candidates:
            if all(np.linalg.norm(Z[idx] - Z[j]) >= min_dist for j in chosen):
                chosen.append(int(idx))
            if len(chosen) >= n_end:
                break
        if len(chosen) < n_end:
            for idx in candidates:
                if int(idx) not in chosen:
                    chosen.append(int(idx))
                if len(chosen) >= n_end:
                    break
        endmembers = X[chosen]
        # Informative-band score: endmember contrast plus global variability.
        em_std = np.std(endmembers, axis=0)
        global_std = np.std(X, axis=0)
        score_band = em_std * 0.7 + global_std * 0.3
        # Select high-scoring bands, then present in wavelength order.
        selected_idx = np.argsort(score_band)[-n_select:]
        selected_idx = np.sort(selected_idx)
        original_wl = np.asarray(self.wavelengths, dtype=float).copy()
        selected_wl = original_wl[selected_idx]
        self._selected_band_plot_data = (original_wl, selected_wl.copy(), selected_idx.copy())
        new_cube = self.cube[:, :, selected_idx].copy()
        self._apply_cube_bands(new_cube, selected_wl)
        self._record_step("Select informative bands", {"endmembers": n_end, "n_bands": n_select},
                          bands_before=bands)
        self.btn_plot_selected_bands.setEnabled(True)
        self.select_status_label.setText(
            f"Selected {n_select} of {bands} bands using {n_end} representative endmembers and a fast PPI-style purity estimate. "
            f"Active cube: {new_cube.shape[0]} × {new_cube.shape[1]} × {new_cube.shape[2]}."
        )
        self.info_label.setText("Informative-band selection complete. Note: this is a Python PPI-style approximation, not MATLAB's exact FIPPI implementation.")

    def on_plot_selected_bands(self):
        if self._selected_band_plot_data is None:
            QtWidgets.QMessageBox.information(self, "No selection result", "Select bands first.")
            return
        original, selected, indices = self._selected_band_plot_data
        def build(fig):
            ax = fig.add_subplot(111)
            ax.plot(np.arange(1, len(original)+1), original, "o", label="Original wavelengths", markersize=4)
            ax.plot(indices+1, selected, "or", label="Selected wavelengths", markersize=6)
            ax.set_xlabel("Original Band Number")
            ax.set_ylabel("Wavelength (nm)")
            ax.set_title("Selected Informative Bands")
            ax.legend(loc="best")
            ax.grid(True, alpha=0.25)
            fig.tight_layout()
        self._show_figure_dialog("Selected Wavelengths", build, size=(9, 5))

    def normalize_channel(self, ch):
        ch = np.asarray(ch, dtype=float)
        if np.all(np.isnan(ch)):
            return np.zeros_like(ch)

        vmin = np.nanmin(ch)
        vmax = np.nanmax(ch)
        if vmax > vmin:
            out = (ch - vmin) / (vmax - vmin)
        else:
            out = np.zeros_like(ch)
        return np.clip(out, 0.0, 1.0)

    def robust_normalize(self, arr):
        arr = np.asarray(arr, dtype=float)
        if np.all(np.isnan(arr)):
            return np.zeros_like(arr)

        p2 = np.nanpercentile(arr, 2.0)
        p98 = np.nanpercentile(arr, 98.0)
        if p98 > p2:
            out = (arr - p2) / (p98 - p2)
        else:
            mn = np.nanmin(arr)
            mx = np.nanmax(arr)
            if mx > mn:
                out = (arr - mn) / (mx - mn)
            else:
                out = np.zeros_like(arr)
        return np.clip(out, 0.0, 1.0)

    def _remove_roi_patch(self):
        for art in self.roi_patches:
            try:
                art.remove()
            except Exception:
                pass
        self.roi_patches = []

    def _class_list(self):
        """Class names in order of first appearance among the saved ROIs."""
        names = []
        for roi in self.rois:
            if roi["class"] not in names:
                names.append(roi["class"])
        return names

    def _selected_roi_rows(self):
        lst = getattr(self, "roi_list", None)
        if lst is None:
            return set()
        return {lst.row(it) for it in lst.selectedItems()}

    @staticmethod
    def _outline_artist(kind, rect, vertices, color, lw, ls="-"):
        if kind == "rect":
            ymin, ymax, xmin, xmax = rect
            return patches.Rectangle(
                (xmin - 0.5, ymin - 0.5), (xmax - xmin + 1), (ymax - ymin + 1),
                fill=False, edgecolor=color, linewidth=lw, linestyle=ls,
            )
        return patches.Polygon(
            np.asarray(vertices, dtype=float), closed=True,
            fill=False, edgecolor=color, linewidth=lw, linestyle=ls,
        )

    def _draw_roi_patch(self):
        """Redraw saved ROI outlines (coloured by class) and the pending selection."""
        self._remove_roi_patch()
        ax = self.image_canvas.ax

        if self.show_roi_outlines and self.rois:
            names = self._class_list()
            palette = class_colors(max(1, len(names)))
            selected = self._selected_roi_rows()
            for i, roi in enumerate(self.rois):
                color = palette[names.index(roi["class"])]
                art = self._outline_artist(
                    roi["kind"], roi.get("rect"), roi.get("vertices"),
                    color, 3.0 if i in selected else 1.4,
                )
                ax.add_patch(art)
                self.roi_patches.append(art)
                ymin, _, xmin, _ = roi["bbox"]
                txt = ax.text(
                    xmin - 0.5, ymin - 0.5, str(i + 1), color="white", fontsize=7,
                    va="bottom", ha="left",
                    bbox=dict(facecolor=color, edgecolor="none", pad=1.0, alpha=0.9),
                )
                self.roi_patches.append(txt)

        if self.pending_sel is not None:
            sel = self.pending_sel
            art = self._outline_artist(
                sel["kind"], sel.get("rect"), sel.get("vertices"),
                "yellow", 1.5, ls="--",
            )
            ax.add_patch(art)
            self.roi_patches.append(art)

        self.image_canvas.draw()

    def update_image(self):
        if self.cube is None:
            return

        rows, cols, bands = self.cube.shape

        if self.display_mode == "single":
            b = int(self.band_slider.value())
            if b < 0 or b >= bands:
                return

            band_data = self.cube[:, :, b]
            band_norm = self.normalize_channel(band_data)
            self.image_canvas.show_image(band_norm)

            if self.wavelengths is not None and len(self.wavelengths) > b:
                self.band_label.setText(f"Band: {b} (λ ≈ {self.wavelengths[b]:.1f} nm)")
            else:
                self.band_label.setText(f"Band: {b}")

        else:
            r_idx = max(0, min(int(self.spin_r.value()), bands - 1))
            g_idx = max(0, min(int(self.spin_g.value()), bands - 1))
            b_idx = max(0, min(int(self.spin_b.value()), bands - 1))

            rgb = np.dstack([
                self.normalize_channel(self.cube[:, :, r_idx]),
                self.normalize_channel(self.cube[:, :, g_idx]),
                self.normalize_channel(self.cube[:, :, b_idx]),
            ])

            self.image_canvas.show_image(rgb)

            label_text = f"RGB bands: R={r_idx}, G={g_idx}, B={b_idx}"
            if self.wavelengths is not None and len(self.wavelengths) == bands:
                label_text += (
                    f" (λR≈{self.wavelengths[r_idx]:.1f}, "
                    f"λG≈{self.wavelengths[g_idx]:.1f}, "
                    f"λB≈{self.wavelengths[b_idx]:.1f} nm)"
                )
            self.band_label.setText(label_text)

        self._draw_roi_patch()

    def on_band_changed(self, value):
        if self.display_mode == "single":
            self.update_image()

    def on_mode_changed(self):
        if self.radio_single.isChecked():
            self.display_mode = "single"
            self.band_slider.setEnabled(True)
        else:
            self.display_mode = "rgb"
            self.band_slider.setEnabled(False)
        self.update_image()

    def on_rgb_band_changed(self, value):
        if self.display_mode == "rgb":
            self.update_image()

    # ------------------------------------------------------------------
    # ROI selection (rectangle / polygon)
    # ------------------------------------------------------------------
    def _on_canvas_cleared(self):
        # ax.clear() detaches selector artists, so attach a fresh selector
        if self.roi_active:
            self._make_selector()

    def _make_selector(self):
        if self.roi_selector is not None:
            try:
                self.roi_selector.set_visible(False)
                self.roi_selector.disconnect_events()
            except Exception:
                pass
            self.roi_selector = None

        if not self.roi_active or self.cube is None:
            return

        ax = self.image_canvas.ax
        if self.combo_roi_shape.currentIndex() == 0:
            self.roi_selector = RectangleSelector(
                ax,
                self.on_roi_selected,
                useblit=True,
                button=[1],
                minspanx=2,
                minspany=2,
                interactive=False,
                spancoords="pixels",
            )
        else:
            self.roi_selector = PolygonSelector(ax, self.on_polygon_selected, useblit=False)

    def _set_roi_hint(self):
        if self.combo_roi_shape.currentIndex() == 0:
            self.info_label.setText("ROI mode: drag a rectangle to compute its mean spectrum.")
        else:
            self.info_label.setText(
                "ROI mode: click to place polygon vertices, then click the first vertex "
                "to close the shape. Hold Shift to move the whole polygon, Esc to restart."
            )

    def on_roi_shape_changed(self, index):
        if self.roi_active:
            self._make_selector()
            self._set_roi_hint()

    def on_roi_toggled(self, checked):
        self.roi_active = checked

        if checked and self.cube is None:
            QtWidgets.QMessageBox.warning(self, "No data", "Load a hyperspectral cube first.")
            self.cb_roi.setChecked(False)
            self.roi_active = False
            return

        if checked:
            self._make_selector()
            self._set_roi_hint()
        else:
            self._make_selector()  # tears the selector down (roi_active is False)
            self.pending_sel = None
            self.roi_label.setText("ROI: none")
            self._draw_roi_patch()
            self.info_label.setText("ROI mode disabled. Click a pixel to see its spectrum.")

    def _x_axis(self, band_mask):
        """x values and axis label for the bands selected by band_mask."""
        bands = self.cube.shape[2]
        if self.wavelengths is not None and len(self.wavelengths) == bands:
            return self.wavelengths[band_mask], "Wavelength (nm)"
        return np.arange(bands)[band_mask], "Band"

    def on_roi_selected(self, eclick, erelease):
        """Rectangle selector callback."""
        if self.cube is None:
            return

        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        if x1 is None or y1 is None or x2 is None or y2 is None:
            return

        rows, cols, bands = self.cube.shape

        xmin = max(0, min(cols - 1, int(round(min(x1, x2)))))
        xmax = max(0, min(cols - 1, int(round(max(x1, x2)))))
        ymin = max(0, min(rows - 1, int(round(min(y1, y2)))))
        ymax = max(0, min(rows - 1, int(round(max(y1, y2)))))

        if xmax <= xmin or ymax <= ymin:
            self.pending_sel = None
            self.roi_label.setText("ROI: invalid / too small")
            self._draw_roi_patch()
            return

        self.pending_sel = {
            "kind": "rect",
            "rect": (ymin, ymax, xmin, xmax),
            "mask": rect_mask(ymin, ymax, xmin, xmax, rows, cols),
        }
        self._after_selection(f"rows {ymin}-{ymax}, cols {xmin}-{xmax}")

    def on_polygon_selector_reset(self):
        self._make_selector()

    def on_polygon_selected(self, verts):
        """Polygon selector callback (called when the polygon is closed)."""
        if self.cube is None:
            return
        rows, cols, bands = self.cube.shape
        verts = [(float(x), float(y)) for x, y in verts]
        mask = polygon_mask(verts, rows, cols)
        if mask.sum() < 2:
            self.pending_sel = None
            self.roi_label.setText("ROI: polygon too small / outside image")
            self._draw_roi_patch()
        else:
            self.pending_sel = {"kind": "polygon", "vertices": verts, "mask": mask}
            self._after_selection(f"polygon with {len(verts)} vertices")
        # start with a fresh selector so the next click begins a new polygon
        QtCore.QTimer.singleShot(0, self.on_polygon_selector_reset)

    def _after_selection(self, description):
        """Shared by rectangle/polygon: mean spectrum + status text for the pending selection."""
        mask = self.pending_sel["mask"]
        spectra = self.cube[mask].astype(float)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            self.last_roi_mean_full = np.nanmean(spectra, axis=0)

        valid_px = ~np.all(np.isnan(spectra), axis=1)
        if not np.any(valid_px):
            self.roi_label.setText(f"ROI: {description} (no valid pixels)")
            QtWidgets.QMessageBox.information(
                self, "ROI contains only NaNs",
                "Selected ROI contains only NaN/nodata pixels."
            )
            self._draw_roi_patch()
            return

        valid_spectra = spectra[valid_px]
        valid_band_mask = ~np.all(np.isnan(valid_spectra), axis=0)
        if not np.any(valid_band_mask):
            self.roi_label.setText(f"ROI: {description} (no valid bands)")
            QtWidgets.QMessageBox.information(
                self, "ROI bands all NaN",
                "Within this ROI, all bands are NaN."
            )
            self._draw_roi_patch()
            return

        valid_spectra = valid_spectra[:, valid_band_mask]
        mean_spec = np.nanmean(valid_spectra, axis=0)

        x, xlabel = self._x_axis(valid_band_mask)
        self.plot_canvas.show_spectrum(x, mean_spec, title="ROI mean spectrum", xlabel=xlabel)

        self.roi_label.setText(
            f"ROI: {description} "
            f"(valid pixels: {valid_spectra.shape[0]}, valid bands: {np.sum(valid_band_mask)})"
        )

        self._draw_roi_patch()
        self.info_label.setText(
            "ROI mean spectrum plotted. Choose a class and press 'Add selection to ROI list' "
            "to keep it for classification, or add it as an endmember."
        )

    # ------------------------------------------------------------------
    # ROI list management
    # ------------------------------------------------------------------
    @staticmethod
    def _bbox_of(mask):
        ys, xs = np.nonzero(mask)
        return (int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max()))

    def _refresh_roi_list(self):
        self.roi_list.blockSignals(True)
        self.roi_list.clear()
        for i, roi in enumerate(self.rois):
            self.roi_list.addItem(
                f"{i + 1}. {roi['name']}  [{roi['class']}]  {roi['kind']}, "
                f"{int(roi['mask'].sum())} px"
            )
        self.roi_list.blockSignals(False)
        self.roi_summary_label.setText(
            f"{len(self.rois)} ROIs in {len(self._class_list())} classes"
        )

    def _refresh_class_combo(self, select=None):
        cur = select if select is not None else self.combo_class.currentText()
        self.combo_class.blockSignals(True)
        self.combo_class.clear()
        self.combo_class.addItems(self._class_list())
        self.combo_class.setEditText(cur or "Class 1")
        self.combo_class.blockSignals(False)

    def on_add_roi(self):
        if self.cube is None:
            QtWidgets.QMessageBox.warning(self, "No data", "Load a hyperspectral cube first.")
            return
        if self.pending_sel is None:
            QtWidgets.QMessageBox.information(
                self, "No selection",
                "Enable ROI selection and draw a rectangle or polygon first."
            )
            return

        cls = self.combo_class.currentText().strip() or "Class 1"
        sel = self.pending_sel
        self.roi_counter += 1
        roi = {
            "name": f"ROI {self.roi_counter}",
            "class": cls,
            "kind": sel["kind"],
            "mask": sel["mask"],
            "bbox": self._bbox_of(sel["mask"]),
        }
        if sel["kind"] == "rect":
            roi["rect"] = tuple(sel["rect"])
        else:
            roi["vertices"] = list(sel["vertices"])
        self.rois.append(roi)

        self.pending_sel = None
        self.roi_label.setText(f"ROI: added '{roi['name']}' to class '{cls}'")
        self._refresh_roi_list()
        self._refresh_class_combo(select=cls)
        self._draw_roi_patch()
        self.info_label.setText(
            f"Added {roi['name']} ({int(roi['mask'].sum())} px) to class '{cls}'. "
            f"{len(self.rois)} ROIs, {len(self._class_list())} classes in total."
        )

    def on_delete_rois(self):
        rows = sorted(self._selected_roi_rows(), reverse=True)
        if not rows:
            QtWidgets.QMessageBox.information(self, "Nothing selected", "Select ROIs in the list first.")
            return
        for r in rows:
            del self.rois[r]
        self._refresh_roi_list()
        self._refresh_class_combo()
        self._draw_roi_patch()
        self.info_label.setText(f"Deleted {len(rows)} ROI(s).")

    def on_clear_rois(self):
        if not self.rois:
            return
        ans = QtWidgets.QMessageBox.question(
            self, "Clear ROIs", f"Remove all {len(self.rois)} ROIs?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No, QtWidgets.QMessageBox.No,
        )
        if ans != QtWidgets.QMessageBox.Yes:
            return
        self.rois = []
        self._refresh_roi_list()
        self._refresh_class_combo()
        self._draw_roi_patch()
        self.info_label.setText("All ROIs cleared.")

    def on_set_roi_class(self):
        rows = sorted(self._selected_roi_rows())
        if not rows:
            QtWidgets.QMessageBox.information(self, "Nothing selected", "Select ROIs in the list first.")
            return
        name, ok = QtWidgets.QInputDialog.getText(
            self, "Set class", f"Class name for {len(rows)} selected ROI(s):",
            text=self.rois[rows[0]]["class"],
        )
        name = name.strip()
        if not ok or not name:
            return
        for r in rows:
            self.rois[r]["class"] = name
        self._refresh_roi_list()
        self._refresh_class_combo()
        self._draw_roi_patch()

    def _roi_mean_spectrum(self, mask):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            return np.nanmean(self.cube[mask].astype(float), axis=0)

    def on_show_roi_spectra(self):
        if self.cube is None or not self.rois:
            QtWidgets.QMessageBox.information(self, "No ROIs", "Add ROIs to the list first.")
            return
        rows = sorted(self._selected_roi_rows()) or list(range(len(self.rois)))
        names = self._class_list()
        palette = class_colors(max(1, len(names)))
        series = []
        for r in rows:
            roi = self.rois[r]
            series.append((
                f"{r + 1}. {roi['name']} [{roi['class']}]",
                self._roi_mean_spectrum(roi["mask"]),
                palette[names.index(roi["class"])],
            ))
        x, xlabel = self._x_axis(np.ones(self.cube.shape[2], dtype=bool))
        self.plot_canvas.show_multiline(x, series, title="ROI mean spectra", xlabel=xlabel)

    def on_show_class_spectra(self):
        if self.cube is None or not self.rois:
            QtWidgets.QMessageBox.information(self, "No ROIs", "Add ROIs to the list first.")
            return
        names = self._class_list()
        palette = class_colors(max(1, len(names)))
        series = []
        for k, name in enumerate(names):
            union = np.zeros(self.cube.shape[:2], dtype=bool)
            for roi in self.rois:
                if roi["class"] == name:
                    union |= roi["mask"]
            series.append((f"{name} ({int(union.sum())} px)", self._roi_mean_spectrum(union), palette[k]))
        x, xlabel = self._x_axis(np.ones(self.cube.shape[2], dtype=bool))
        self.plot_canvas.show_multiline(x, series, title="Class mean spectra", xlabel=xlabel)

    def on_toggle_outlines(self, checked):
        self.show_roi_outlines = checked
        self._draw_roi_patch()

    def on_save_rois(self):
        if not self.rois:
            QtWidgets.QMessageBox.information(self, "No ROIs", "There are no ROIs to save.")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save ROIs", "rois.json", "ROI files (*.json)"
        )
        if not path:
            return
        rows, cols, _ = self.cube.shape
        data = {"format": "DS3-ROIs", "version": 1, "image_shape": [rows, cols], "rois": []}
        for roi in self.rois:
            item = {"name": roi["name"], "class": roi["class"], "kind": roi["kind"]}
            if roi["kind"] == "rect":
                item["rect"] = [int(v) for v in roi["rect"]]
            else:
                item["vertices"] = [[float(x), float(y)] for x, y in roi["vertices"]]
            data["rois"].append(item)
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to save ROIs:\n{e}")
            return
        self.info_label.setText(f"Saved {len(self.rois)} ROIs to {path}")

    def on_load_rois(self):
        if self.cube is None:
            QtWidgets.QMessageBox.warning(self, "No data", "Load a hyperspectral cube first.")
            return
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load ROIs", "", "ROI files (*.json);;All files (*)"
        )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            items = data["rois"]
            shape = data.get("image_shape")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Not a valid ROI file:\n{e}")
            return

        rows, cols, _ = self.cube.shape
        if shape and list(shape) != [rows, cols]:
            ans = QtWidgets.QMessageBox.question(
                self, "Image size differs",
                f"These ROIs were saved for a {shape[0]} x {shape[1]} image but the current "
                f"image is {rows} x {cols}. Load anyway? (Shapes outside the image are clipped.)",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No, QtWidgets.QMessageBox.No,
            )
            if ans != QtWidgets.QMessageBox.Yes:
                return

        loaded, skipped = [], 0
        for item in items:
            try:
                if item["kind"] == "rect":
                    ymin, ymax, xmin, xmax = [int(v) for v in item["rect"]]
                    ymin, ymax = max(0, ymin), min(rows - 1, ymax)
                    xmin, xmax = max(0, xmin), min(cols - 1, xmax)
                    if ymax < ymin or xmax < xmin:
                        skipped += 1
                        continue
                    mask = rect_mask(ymin, ymax, xmin, xmax, rows, cols)
                    roi = {"rect": (ymin, ymax, xmin, xmax)}
                else:
                    verts = [(float(x), float(y)) for x, y in item["vertices"]]
                    mask = polygon_mask(verts, rows, cols)
                    roi = {"vertices": verts}
                if not mask.any():
                    skipped += 1
                    continue
                roi.update({
                    "name": str(item.get("name", "ROI")),
                    "class": str(item.get("class", "Class 1")),
                    "kind": item["kind"],
                    "mask": mask,
                    "bbox": self._bbox_of(mask),
                })
                loaded.append(roi)
            except Exception:
                skipped += 1

        if not loaded:
            QtWidgets.QMessageBox.information(self, "Nothing loaded", "No usable ROIs found in the file.")
            return

        if self.rois:
            ans = QtWidgets.QMessageBox.question(
                self, "Existing ROIs",
                "Replace the current ROIs? (Yes = replace, No = append to the list)",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No | QtWidgets.QMessageBox.Cancel,
                QtWidgets.QMessageBox.Yes,
            )
            if ans == QtWidgets.QMessageBox.Cancel:
                return
            if ans == QtWidgets.QMessageBox.Yes:
                self.rois = []
        self.rois.extend(loaded)
        self.roi_counter = max(self.roi_counter, len(self.rois))
        self._refresh_roi_list()
        self._refresh_class_combo()
        self._draw_roi_patch()
        msg = f"Loaded {len(loaded)} ROIs from {path}."
        if skipped:
            msg += f" Skipped {skipped} that fell outside the image or were invalid."
        self.info_label.setText(msg)

    def on_add_endmember_from_roi(self):
        if self.cube is None:
            QtWidgets.QMessageBox.warning(self, "No data", "Load a hyperspectral cube first.")
            return

        if self.last_roi_mean_full is None:
            QtWidgets.QMessageBox.information(
                self, "No ROI",
                "Select an ROI first so we can use its mean spectrum as an endmember."
            )
            return

        endm = np.asarray(self.last_roi_mean_full, dtype=float).ravel()
        if endm.size != self.cube.shape[2]:
            QtWidgets.QMessageBox.critical(
                self, "Internal error",
                "ROI mean length does not match number of bands."
            )
            return

        if np.all(np.isnan(endm)):
            QtWidgets.QMessageBox.information(
                self, "Invalid endmember",
                "The current ROI mean spectrum is all NaN."
            )
            return

        self.endmembers.append(endm)
        self.endmember_label.setText(f"Endmembers: {len(self.endmembers)}")
        self.info_label.setText(
            f"Added ROI mean as endmember #{len(self.endmembers)}."
        )

    def on_clear_endmembers(self):
        self.endmembers = []
        self.endmember_label.setText("Endmembers: 0")
        self.info_label.setText("Endmembers cleared.")

    def on_image_click(self, event):
        if self.roi_active:
            return

        if self.cube is None:
            return
        if event.inaxes != self.image_canvas.ax:
            return
        if event.xdata is None or event.ydata is None:
            return

        col = int(round(event.xdata))
        row = int(round(event.ydata))

        rows, cols, bands = self.cube.shape
        if row < 0 or row >= rows or col < 0 or col >= cols:
            return

        pix = np.asarray(self.cube[row, col, :], dtype=float).ravel()
        valid_band_mask = ~np.isnan(pix)
        if not np.any(valid_band_mask):
            self.info_label.setText(f"Pixel ({row}, {col}) spectrum is all NaN.")
            return

        spec_valid = pix[valid_band_mask]
        x, xlabel = self._x_axis(valid_band_mask)

        self.plot_canvas.show_spectrum(
            x, spec_valid, title=f"Pixel spectrum (row={row}, col={col})", xlabel=xlabel
        )
        self.info_label.setText(f"Spectrum at (row={row}, col={col})")

    def compute_anomaly_map(self):
        if self.cube is None:
            raise RuntimeError("No cube loaded")

        rows, cols, bands = self.cube.shape
        data = self.cube.reshape(-1, bands).astype(float)

        valid_px = ~np.all(np.isnan(data), axis=1)
        if not np.any(valid_px):
            raise RuntimeError("All pixels are all-NaN; anomaly map is undefined.")

        data_valid_px = data[valid_px]
        valid_band_mask = ~np.all(np.isnan(data_valid_px), axis=0)
        if not np.any(valid_band_mask):
            raise RuntimeError("All bands are NaN across valid pixels.")

        data_valid = data_valid_px[:, valid_band_mask]

        with np.errstate(invalid="ignore"):
            band_means = np.nanmean(data_valid, axis=0, keepdims=True)
        nan_mask = np.isnan(data_valid)
        if np.any(nan_mask):
            data_valid[nan_mask] = band_means.repeat(data_valid.shape[0], axis=0)[nan_mask]

        mean_spec = np.mean(data_valid, axis=0, keepdims=True)
        diff = data_valid - mean_spec
        dist_valid = np.linalg.norm(diff, axis=1)

        anomaly = np.zeros(data.shape[0], dtype=float)
        anomaly[valid_px] = dist_valid
        return anomaly.reshape(rows, cols)

    def on_compute_anomaly(self):
        if self.cube is None:
            QtWidgets.QMessageBox.warning(self, "No data", "Load a hyperspectral cube first.")
            return

        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            self.info_label.setText("Computing anomaly map...")
            anomaly = self.compute_anomaly_map()
            self.anomaly_map = anomaly

            global_min = float(np.nanmin(anomaly))
            global_max = float(np.nanmax(anomaly))
            p2 = float(np.nanpercentile(anomaly, 2.0))
            p98 = float(np.nanpercentile(anomaly, 98.0))
            vmin, vmax = (p2, p98) if p98 > p2 else (global_min, global_max)

            if vmax > vmin:
                an_norm = np.clip((anomaly - vmin) / (vmax - vmin), 0.0, 1.0)
            else:
                an_norm = np.zeros_like(anomaly)

            self.image_canvas.show_image(an_norm)
            self.band_label.setText("Anomaly map (L2 distance to global mean)")
            self._draw_roi_patch()

            self.info_label.setText(
                f"Anomaly map computed. Raw range: "
                f"[{global_min:.4g}, {global_max:.4g}], stretch: [{vmin:.4g}, {vmax:.4g}]"
            )
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to compute anomaly map:\n{e}")
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

    def on_apply_bandmath(self):
        if self.cube is None:
            QtWidgets.QMessageBox.warning(self, "No data", "Load a hyperspectral cube first.")
            return

        expr = self.bandmath_edit.text().strip()
        if not expr:
            QtWidgets.QMessageBox.information(self, "Empty expression", "Enter a band math expression.")
            return

        rows, cols, bands = self.cube.shape
        allowed = {"np": np}
        for b in range(bands):
            allowed[f"B{b}"] = self.cube[:, :, b].astype(float)

        try:
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                result = eval(expr, {"__builtins__": {}}, allowed)

            arr = np.asarray(result, dtype=float)
            if arr.shape != (rows, cols):
                if arr.size == 1:
                    arr = np.full((rows, cols), float(arr))
                else:
                    raise ValueError(
                        f"Expression result shape {arr.shape} does not match ({rows}, {cols})."
                    )

            img = self.robust_normalize(arr)
            self.image_canvas.show_image(img)
            self.band_label.setText(f"Band Math: {expr}")
            self._draw_roi_patch()
            self.info_label.setText("Band math applied.")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to evaluate band math expression:\n{e}")

    def on_run_kmeans(self):
        if self.cube is None:
            QtWidgets.QMessageBox.warning(self, "No data", "Load a hyperspectral cube first.")
            return
        if not SKLEARN_AVAILABLE:
            QtWidgets.QMessageBox.warning(self, "Missing dependency", "Install scikit-learn for k-means.")
            return

        K = int(self.spin_k.value())
        mode = self.combo_cluster_mode.currentText()

        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            if mode == "Full spectra":
                rows, cols, bands = self.cube.shape
                data = self.cube.reshape(-1, bands).astype(float)

                valid_px = ~np.all(np.isnan(data), axis=1)
                if not np.any(valid_px):
                    raise RuntimeError("All pixels are all-NaN.")

                data_valid_px = data[valid_px]
                valid_band_mask = ~np.all(np.isnan(data_valid_px), axis=0)
                if not np.any(valid_band_mask):
                    raise RuntimeError("All bands are NaN across valid pixels.")

                feats = data_valid_px[:, valid_band_mask]

                with np.errstate(invalid="ignore"):
                    band_means = np.nanmean(feats, axis=0, keepdims=True)
                nan_mask = np.isnan(feats)
                if np.any(nan_mask):
                    feats[nan_mask] = band_means.repeat(feats.shape[0], axis=0)[nan_mask]

                source_text = "full spectra"

            else:
                if self.pca_scores is None or self.pca_n_components < 3:
                    raise RuntimeError("Compute PCA first; need at least 3 PCs for PCA clustering.")

                rows, cols, _ = self.pca_scores.shape
                pca_flat = self.pca_scores[:, :, :3].reshape(-1, 3).astype(float)

                valid_px = self.pca_valid_mask
                if valid_px is None or not np.any(valid_px):
                    raise RuntimeError("No valid PCA pixels available.")

                feats = pca_flat[valid_px, :]
                source_text = "first 3 PCs"

            self.info_label.setText(f"Running k-means with K={K} on {source_text}...")
            kmeans = MiniBatchKMeans(n_clusters=K, random_state=0, batch_size=4096)
            labels_valid = kmeans.fit_predict(feats)

            labels = -1 * np.ones(rows * cols, dtype=int)
            labels[valid_px] = labels_valid
            labels_img = labels.reshape(rows, cols)
            self.class_labels = labels_img

            cmap = matplotlib.colormaps.get_cmap("tab10")
            rgb_map = np.zeros((rows, cols, 3), dtype=float)
            for k in range(K):
                mask = labels_img == k
                color = cmap(k % cmap.N)
                rgb_map[mask] = np.array(color)[:3]

            self.class_rgb_map = rgb_map
            self.image_canvas.show_image(rgb_map)
            self.band_label.setText(f"k-means clustering (K={K}, {source_text})")
            self._draw_roi_patch()

            counts = [int(np.sum(labels_img == k)) for k in range(K)]
            self.info_label.setText(
                f"k-means on {source_text} done. "
                + ", ".join([f"Class {k}: {c} px" for k, c in enumerate(counts)])
            )

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to run k-means:\n{e}")
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

    def on_run_unmixing(self):
        if self.cube is None:
            QtWidgets.QMessageBox.warning(self, "No data", "Load a hyperspectral cube first.")
            return

        M = len(self.endmembers)
        if M < 2:
            QtWidgets.QMessageBox.information(
                self, "Not enough endmembers",
                "Define at least 2 endmembers first."
            )
            return

        rows, cols, bands = self.cube.shape

        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            E_raw = np.stack(self.endmembers, axis=1)
            X_raw = self.cube.reshape(-1, bands).astype(float).T

            valid_band_mask = ~np.any(np.isnan(E_raw), axis=1)
            if not np.any(valid_band_mask):
                raise RuntimeError("All bands are NaN in at least one endmember.")

            E = E_raw[valid_band_mask, :]
            X = X_raw[valid_band_mask, :]

            with np.errstate(invalid="ignore"):
                band_means = np.nanmean(X, axis=1, keepdims=True)
            nan_mask = np.isnan(X)
            if np.any(nan_mask):
                X[nan_mask] = np.broadcast_to(band_means, X.shape)[nan_mask]

            E_pinv = np.linalg.pinv(E)
            A = E_pinv @ X

            dominant = np.argmax(A, axis=0).reshape(rows, cols)

            cmap = matplotlib.colormaps.get_cmap("tab10")
            rgb_map = np.zeros((rows, cols, 3), dtype=float)
            for m in range(M):
                mask = dominant == m
                color = cmap(m % cmap.N)
                rgb_map[mask] = np.array(color)[:3]

            self.image_canvas.show_image(rgb_map)
            self.band_label.setText(f"Unmixing: dominant endmember map (M={M})")
            self._draw_roi_patch()

            counts = [int(np.sum(dominant == m)) for m in range(M)]
            self.info_label.setText(
                "Unmixing done. " + ", ".join([f"E{m}: {c} px" for m, c in enumerate(counts)])
            )

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to run unmixing:\n{e}")
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

    def compute_pca(self, n_components):
        if self.cube is None:
            raise RuntimeError("No cube loaded")

        rows, cols, bands = self.cube.shape
        data = self.cube.reshape(-1, bands).astype(float)

        valid_px = ~np.all(np.isnan(data), axis=1)
        if not np.any(valid_px):
            raise RuntimeError("All pixels are all-NaN; PCA undefined.")

        data_valid_px = data[valid_px]
        valid_band_mask = ~np.all(np.isnan(data_valid_px), axis=0)
        if not np.any(valid_band_mask):
            raise RuntimeError("All bands are NaN across valid pixels; PCA undefined.")

        X = data_valid_px[:, valid_band_mask]

        with np.errstate(invalid="ignore"):
            band_means = np.nanmean(X, axis=0, keepdims=True)
        nan_mask = np.isnan(X)
        if np.any(nan_mask):
            X[nan_mask] = band_means.repeat(X.shape[0], axis=0)[nan_mask]

        mean_vec = np.mean(X, axis=0, keepdims=True)
        Xc = X - mean_vec

        max_comp = min(Xc.shape[0], Xc.shape[1], n_components)
        if max_comp < 1:
            raise RuntimeError("Not enough valid data for PCA.")

        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        scores_valid = Xc @ Vt[:max_comp].T

        denom = np.sum(S**2)
        if denom > 0:
            explained_ratio = (S[:max_comp]**2) / denom
        else:
            explained_ratio = np.zeros(max_comp, dtype=float)

        scores_full = np.zeros((data.shape[0], max_comp), dtype=float)
        scores_full[valid_px, :] = scores_valid
        scores_img = scores_full.reshape(rows, cols, max_comp)

        self.pca_scores = scores_img
        self.pca_explained_ratio = explained_ratio
        self.pca_valid_mask = valid_px
        self.pca_n_components = max_comp

    def on_compute_pca(self):
        if self.cube is None:
            QtWidgets.QMessageBox.warning(self, "No data", "Load a hyperspectral cube first.")
            return

        n_components = int(self.spin_pca_n.value())

        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            self.info_label.setText(f"Computing first {n_components} principal components...")
            self.compute_pca(n_components)
            self.pca_label.setText(f"PCA: computed {self.pca_n_components} components")

            if self.pca_explained_ratio is not None and self.pca_explained_ratio.size >= 3:
                var3 = 100.0 * np.sum(self.pca_explained_ratio[:3])
                self.info_label.setText(
                    f"PCA computed. First 3 PCs explain about {var3:.2f}% of the variance."
                )
            else:
                self.info_label.setText("PCA computed.")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to compute PCA:\n{e}")
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

    def on_show_pc(self, pc_index):
        if self.pca_scores is None:
            QtWidgets.QMessageBox.information(self, "PCA not ready", "Compute PCA first.")
            return
        if pc_index >= self.pca_n_components:
            QtWidgets.QMessageBox.information(
                self, "Not available",
                f"PC{pc_index+1} is not available. Only {self.pca_n_components} PCs were computed."
            )
            return

        pc_img = self.pca_scores[:, :, pc_index]
        pc_norm = self.robust_normalize(pc_img)
        self.image_canvas.show_image(pc_norm)
        self.band_label.setText(f"PC{pc_index+1}")

        if self.pca_explained_ratio is not None and pc_index < len(self.pca_explained_ratio):
            ev = 100.0 * self.pca_explained_ratio[pc_index]
            self.info_label.setText(f"Showing PC{pc_index+1} ({ev:.2f}% explained variance).")
        else:
            self.info_label.setText(f"Showing PC{pc_index+1}.")

        self._draw_roi_patch()

    def on_show_pca_rgb(self):
        if self.pca_scores is None:
            QtWidgets.QMessageBox.information(self, "PCA not ready", "Compute PCA first.")
            return
        if self.pca_n_components < 3:
            QtWidgets.QMessageBox.information(
                self, "Not enough PCs",
                "Need at least 3 principal components to show PCA RGB."
            )
            return

        rgb = np.dstack([
            self.robust_normalize(self.pca_scores[:, :, 0]),
            self.robust_normalize(self.pca_scores[:, :, 1]),
            self.robust_normalize(self.pca_scores[:, :, 2]),
        ])

        self.image_canvas.show_image(rgb)
        self.band_label.setText("PCA RGB (PC1 / PC2 / PC3)")

        var3 = 100.0 * np.sum(self.pca_explained_ratio[:3])
        self.info_label.setText(
            f"Showing PCA RGB. First 3 PCs explain about {var3:.2f}% of the variance."
        )
        self._draw_roi_patch()

    def on_show_scree(self):
        if self.pca_explained_ratio is None:
            QtWidgets.QMessageBox.information(self, "PCA not ready", "Compute PCA first.")
            return

        x = np.arange(1, len(self.pca_explained_ratio) + 1)
        y = 100.0 * self.pca_explained_ratio

        self.plot_canvas.show_barplot(
            x, y,
            title="PCA Scree Plot",
            xlabel="Principal Component",
            ylabel="Explained Variance (%)"
        )
        self.info_label.setText(
            "Showing scree plot: each bar is the percent variance explained by that principal component."
        )

    # ------------------------------------------------------------------
    # Classification (SAM + supervised)
    # ------------------------------------------------------------------
    def _show_classification(self, labels_img, names, title):
        """Colour a label image (-1 = unclassified) and show it with a legend."""
        all_names = self._class_list()
        palette = class_colors(max(1, len(all_names)))
        fallback = class_colors(max(1, len(names)))
        colors = [
            palette[all_names.index(n)] if n in all_names else fallback[k]
            for k, n in enumerate(names)
        ]

        rows, cols = labels_img.shape
        rgb = np.zeros((rows, cols, 3), dtype=float)
        for k in range(len(names)):
            rgb[labels_img == k] = colors[k]

        counts = [int(np.sum(labels_img == k)) for k in range(len(names))]
        n_unc = int(np.sum(labels_img < 0))
        legend = [(f"{names[k]} ({counts[k]})", colors[k]) for k in range(len(names))]
        if n_unc:
            legend.append((f"Unclassified ({n_unc})", (0.0, 0.0, 0.0)))

        self.image_canvas.show_image(rgb, legend=legend)
        self.band_label.setText(f"{title} classification ({len(names)} classes)")
        self._draw_roi_patch()
        self.info_label.setText(
            f"{title} done. " + ", ".join(f"{names[k]}: {counts[k]} px" for k in range(len(names)))
            + (f", unclassified: {n_unc} px" if n_unc else "")
        )

    def on_run_sam(self):
        if self.cube is None:
            QtWidgets.QMessageBox.warning(self, "No data", "Load a hyperspectral cube first.")
            return

        rows, cols, bands = self.cube.shape
        thr_deg = float(self.spin_sam_thr.value())
        use_rois = self.combo_sam_source.currentIndex() == 0

        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            self.info_label.setText("Running Spectral Angle Mapper...")
            X, valid_px = prepare_features(self.cube.reshape(-1, bands))

            if use_rois:
                names = self._class_list()
                if not names:
                    raise RuntimeError("Add labelled ROIs first in the ROI Analysis tab.")
                refs = []
                for name in names:
                    union = np.zeros((rows, cols), dtype=bool)
                    for roi in self.rois:
                        if roi["class"] == name:
                            union |= roi["mask"]
                    sel = union.ravel() & valid_px
                    if not sel.any():
                        raise RuntimeError(f"Class '{name}' has no valid pixels.")
                    refs.append(X[sel].mean(axis=0))
                E = np.vstack(refs)
            else:
                if not self.endmembers:
                    raise RuntimeError("The endmember list is empty (add endmembers from ROIs first).")
                names = [f"E{i + 1}" for i in range(len(self.endmembers))]
                E = np.stack(self.endmembers, axis=0).astype(np.float32)
                band_means = X[valid_px].mean(axis=0)
                E = np.where(np.isnan(E), band_means, E)

            if E.shape[0] < 2 and thr_deg <= 0:
                raise RuntimeError(
                    "With a single reference spectrum set a maximum angle above 0, "
                    "otherwise every pixel would be assigned to it."
                )

            best_idx, best_ang = spectral_angle_map(X[valid_px], E)

            labels = -np.ones(rows * cols, dtype=np.int32)
            angle_full = np.full(rows * cols, np.nan, dtype=np.float32)
            labels[valid_px] = best_idx
            angle_full[valid_px] = best_ang

            if thr_deg > 0:
                with np.errstate(invalid="ignore"):
                    labels[angle_full > np.deg2rad(thr_deg)] = -1

            labels_img = labels.reshape(rows, cols)
            self.class_labels = labels_img
            self.sam_angle_map = np.rad2deg(angle_full).reshape(rows, cols)
            self.class_names_result = names

            self._show_classification(labels_img, names, "SAM")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to run SAM:\n{e}")
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

    def on_show_sam_angle(self):
        if self.sam_angle_map is None:
            QtWidgets.QMessageBox.information(self, "SAM not run", "Run SAM classification first.")
            return
        ang = self.sam_angle_map
        img = self.robust_normalize(ang)
        self.image_canvas.show_image(img)
        self.band_label.setText("SAM best-match angle (dark = small angle = close match)")
        self._draw_roi_patch()
        self.info_label.setText(
            f"SAM best-match angle map. Range: {np.nanmin(ang):.2f} to {np.nanmax(ang):.2f} degrees. "
            "Use it to choose a sensible maximum-angle threshold."
        )

    # ------------------------------------------------------------------
    # Background jobs (supervised, cross-validation, CNN, NMF)
    # ------------------------------------------------------------------
    def _set_busy(self, busy):
        if busy and not self._busy:
            QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        elif not busy and self._busy:
            QtWidgets.QApplication.restoreOverrideCursor()
        self._busy = busy
        for b in self._busy_buttons:
            b.setEnabled(not busy)
        self.btn_stop.setVisible(busy)
        self.btn_stop.setEnabled(busy)

    def _start_job(self, job, on_done, label):
        """Run job(progress, cancel) on a worker thread; on_done(result) runs on the GUI thread."""
        if self._busy:
            return
        self._cancel_event = threading.Event()
        cancel = self._cancel_event
        worker = Worker(lambda progress: job(progress, cancel))
        worker.setAutoDelete(False)
        worker.signals.progress.connect(self._on_worker_progress)
        worker.signals.finished.connect(self._on_job_finished)
        worker.signals.failed.connect(self._on_worker_failed)
        self._worker = worker
        self._worker_cube = self.cube
        self._on_done = on_done
        self._set_busy(True)
        self.info_label.setText(label)
        QtCore.QThreadPool.globalInstance().start(worker)

    def on_stop_task(self):
        if self._cancel_event is not None:
            self._cancel_event.set()
            self.btn_stop.setEnabled(False)
            self.info_label.setText("Stopping... (the current step has to finish first)")

    def _on_worker_progress(self, msg):
        self.info_label.setText(msg)

    def _on_job_finished(self, res):
        self._set_busy(False)
        if self._worker_cube is not self.cube:
            self.info_label.setText("A different image was loaded meanwhile; the result was discarded.")
            return
        cb, self._on_done = self._on_done, None
        if cb is not None:
            cb(res)

    def _on_worker_failed(self, msg):
        self._set_busy(False)
        if msg == "__cancelled__":
            self.info_label.setText("Task cancelled.")
            return
        QtWidgets.QMessageBox.critical(self, "Error", f"Task failed:\n{msg}")

    def _supervised_precheck(self, need_tf=False):
        """Common checks. Returns the class list, or None after telling the user why not."""
        if self._busy:
            return None
        if self.cube is None:
            QtWidgets.QMessageBox.warning(self, "No data", "Load a hyperspectral cube first.")
            return None
        if not SKLEARN_AVAILABLE:
            QtWidgets.QMessageBox.warning(self, "Missing dependency", "Install scikit-learn first.")
            return None
        if need_tf and not TF_AVAILABLE:
            QtWidgets.QMessageBox.warning(
                self, "TensorFlow not found",
                "The 2D CNN needs TensorFlow.\nInstall it with:  pip install tensorflow"
            )
            return None
        names = self._class_list()
        if len(names) < 2:
            QtWidgets.QMessageBox.information(
                self, "Not enough classes",
                "Add ROIs for at least 2 different classes first."
            )
            return None
        return names

    def _supervised_args(self):
        """(feature_source, pca_scores, pca_valid) from the Classification tab, or None."""
        use_pca = self.combo_features.currentIndex() == 1
        if use_pca and self.pca_scores is None:
            QtWidgets.QMessageBox.information(self, "PCA not ready", "Compute PCA first (PCA tab).")
            return None
        return ("pca" if use_pca else "spectra"), self.pca_scores, self.pca_valid_mask

    def on_train_classifier(self):
        class_names = self._supervised_precheck()
        if class_names is None:
            return
        feat = self._supervised_args()
        if feat is None:
            return
        source, pca_scores, pca_valid = feat

        clf_name = self.combo_clf.currentText()
        test_frac = self.spin_test.value() / 100.0
        split_mode = "rois" if self.combo_split.currentIndex() == 0 else "pixels"
        max_pc = int(self.spin_max_pc.value())
        cube, rois = self.cube, list(self.rois)

        def job(progress, cancel):
            return run_supervised(
                cube, rois, class_names, source, pca_scores, pca_valid,
                clf_name, test_frac, split_mode, max_pc, progress=progress, cancel=cancel,
            )

        self._start_job(job, self._finish_supervised, f"Starting {clf_name}...")

    def on_crossval(self):
        class_names = self._supervised_precheck()
        if class_names is None:
            return
        feat = self._supervised_args()
        if feat is None:
            return
        source, pca_scores, pca_valid = feat

        clf_name = self.combo_clf.currentText()
        k = int(self.spin_cv_k.value())
        split_mode = "rois" if self.combo_split.currentIndex() == 0 else "pixels"
        max_pc = int(self.spin_max_pc.value())
        cube, rois = self.cube, list(self.rois)

        def job(progress, cancel):
            return run_crossval(
                cube, rois, class_names, source, pca_scores, pca_valid,
                clf_name, k, split_mode, max_pc, progress=progress, cancel=cancel,
            )

        self._start_job(job, self._finish_report_only, f"Starting {k}-fold cross-validation...")

    def on_train_cnn(self):
        class_names = self._supervised_precheck(need_tf=True)
        if class_names is None:
            return
        n_pca = int(self.spin_cnn_pca.value())
        patch = int(self.spin_cnn_patch.value()) | 1
        epochs = int(self.spin_cnn_epochs.value())
        batch = int(self.spin_cnn_batch.value())
        net = "reference" if self.combo_cnn_net.currentIndex() == 0 else "compact"
        test_frac = self.spin_cnn_test.value() / 100.0
        split_mode = "rois" if self.combo_cnn_split.currentIndex() == 0 else "pixels"
        max_pc = int(self.spin_cnn_maxpc.value())
        scope = "image" if self.combo_cnn_scope.currentIndex() == 0 else "rois"
        cube, rois = self.cube, list(self.rois)

        def job(progress, cancel):
            return run_cnn(
                cube, rois, class_names, n_pca, patch, epochs, batch, net,
                test_frac, split_mode, max_pc, scope, progress=progress, cancel=cancel,
            )

        self._start_job(job, self._finish_supervised, "Starting 2D CNN (the first step loads TensorFlow)...")

    def _finish_supervised(self, res):
        self.class_labels = res["labels"]
        self.class_names_result = res["names"]
        self.last_report = res["report"]
        self._show_classification(res["labels"], res["names"], res["title"])
        self.on_show_report()

    def _finish_report_only(self, res):
        self.last_report = res["report"]
        self.info_label.setText("Cross-validation finished. See the report window.")
        self.on_show_report()

    def on_show_report(self):
        if not self.last_report:
            QtWidgets.QMessageBox.information(
                self, "No report", "Train a supervised classifier first."
            )
            return
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Classification report")
        lay = QtWidgets.QVBoxLayout(dlg)
        box = QtWidgets.QPlainTextEdit()
        box.setReadOnly(True)
        box.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap)
        box.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont))
        box.setPlainText(self.last_report)
        lay.addWidget(box)
        btn = QtWidgets.QPushButton("Close")
        btn.clicked.connect(dlg.accept)
        lay.addWidget(btn)
        dlg.resize(760, 520)
        dlg.exec_()


    # ------------------------------------------------------------------
    # Helper: pop-up matplotlib figure with the standard toolbar (zoom / save PNG)
    # ------------------------------------------------------------------
    def _show_figure_dialog(self, title, builder, size=(9, 7)):
        from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle(title)
        dlg.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        fig = Figure(figsize=size)
        canvas = FigureCanvas(fig)
        lay = QtWidgets.QVBoxLayout(dlg)
        lay.addWidget(NavigationToolbar2QT(canvas, dlg))
        lay.addWidget(canvas, 1)
        builder(fig)
        restyle_figure(fig)
        canvas.draw()
        dlg.resize(int(size[0] * 85), int(size[1] * 85))
        self._figure_dialogs.append(dlg)
        dlg.destroyed.connect(lambda *_: self._figure_dialogs.remove(dlg)
                              if dlg in self._figure_dialogs else None)
        dlg.show()

    # ------------------------------------------------------------------
    # Atmospheric correction: empirical line
    # ------------------------------------------------------------------
    def _reset_el_result(self):
        self.el_result = None
        self.btn_el_plot_gain.setEnabled(False)
        self.btn_el_plot_refs.setEnabled(False)

    def _refresh_el_list(self):
        self.el_list.clear()
        for i, ref in enumerate(self.el_refs):
            wl = ref["wl"]
            self.el_list.addItem(
                f"{i + 1}. {ref['roi_name']} [{ref['roi_class']}] ({ref['n_px']} px)  <->  {ref['name']} "
                f"({wl.min():g}-{wl.max():g}, {len(wl)} pts)"
            )

    def on_el_add_reference(self):
        if self.cube is None:
            QtWidgets.QMessageBox.warning(self, "No data", "Load a hyperspectral cube first.")
            return
        if not self.rois:
            QtWidgets.QMessageBox.information(
                self, "No ROIs",
                "Draw an ROI on each reference target and add it to the ROI list first "
                "(ROI Analysis tab)."
            )
            return
        items = [f"{i + 1}. {r['name']} [{r['class']}] - {int(r['mask'].sum())} px" for i, r in enumerate(self.rois)]
        choice, ok = QtWidgets.QInputDialog.getItem(
            self, "Reference ROI", "Image ROI covering the reference material:", items, 0, False)
        if not ok:
            return
        roi = self.rois[items.index(choice)]
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Field reflectance spectrum (wavelength, reflectance)", "",
            "Spectra (*.csv *.txt *.dat *.tsv);;All files (*)")
        if not path:
            return
        try:
            wl, values = load_field_spectrum(path)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Could not read the field spectrum:\n{e}")
            return
        self.el_refs.append({
            "roi_name": roi["name"], "roi_class": roi["class"], "mask": roi["mask"].copy(),
            "n_px": int(roi["mask"].sum()), "name": Path(path).name, "wl": wl, "values": values,
        })
        self._reset_el_result()
        self._refresh_el_list()
        self.el_status_label.setText(f"{len(self.el_refs)} reference(s) defined.")

    def on_el_remove_reference(self):
        rows = sorted({self.el_list.row(it) for it in self.el_list.selectedItems()}, reverse=True)
        if not rows:
            QtWidgets.QMessageBox.information(self, "Nothing selected", "Select references in the list first.")
            return
        for r in rows:
            del self.el_refs[r]
        self._reset_el_result()
        self._refresh_el_list()
        self.el_status_label.setText(f"{len(self.el_refs)} reference(s) defined.")

    def on_el_clear_references(self):
        self.el_refs = []
        self._reset_el_result()
        self._refresh_el_list()
        self.el_status_label.setText("References cleared.")

    def on_el_apply(self):
        """Estimate per-band gain and offset from the references and apply them to the whole cube."""
        if self.cube is None:
            QtWidgets.QMessageBox.warning(self, "No data", "Load a hyperspectral cube first.")
            return
        if self._busy:
            return
        if not self.el_refs:
            QtWidgets.QMessageBox.information(
                self, "No references", "Add at least one reference (ROI + field spectrum) first.")
            return
        bands = self.cube.shape[2]
        if self.wavelengths is None or len(self.wavelengths) != bands:
            QtWidgets.QMessageBox.warning(
                self, "Wavelength metadata unavailable",
                "Empirical line resamples the field spectra to the image wavelengths, so the cube needs wavelength metadata.")
            return
        if self.radiometric_stage.lower().startswith("surface reflectance"):
            ans = QtWidgets.QMessageBox.question(
                self, "Already surface reflectance",
                "The active data stage is already surface reflectance. Apply empirical line again?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No, QtWidgets.QMessageBox.No)
            if ans != QtWidgets.QMessageBox.Yes:
                return
        rows, cols = self.cube.shape[:2]
        if any(ref["mask"].shape != (rows, cols) for ref in self.el_refs):
            QtWidgets.QMessageBox.warning(self, "ROI size mismatch", "A reference ROI does not match the image size. Clear and re-add the references.")
            return

        scale = 0.01 if self.el_units_combo.currentIndex() == 1 else 1.0
        cube_wl_nm = auto_wavelengths_to_nm(self.wavelengths)
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            measured = np.vstack([self._roi_mean_spectrum(ref["mask"]) for ref in self.el_refs])
            field = np.vstack([
                resample_field_spectrum(auto_wavelengths_to_nm(ref["wl"]), ref["values"] * scale, cube_wl_nm)
                for ref in self.el_refs
            ])
            gain, offset, n_used, through_origin = empirical_line(measured, field)
            out = apply_gain_offset(self.cube, gain, offset)
        except ValueError as e:
            QtWidgets.QApplication.restoreOverrideCursor()
            QtWidgets.QMessageBox.warning(self, "Empirical line", str(e))
            return
        except Exception as e:
            QtWidgets.QApplication.restoreOverrideCursor()
            QtWidgets.QMessageBox.critical(self, "Error", f"Empirical line failed:\n{e}")
            return
        QtWidgets.QApplication.restoreOverrideCursor()

        corrected_refs = measured * gain + offset
        ok = np.isfinite(field)
        rmse = [float(np.sqrt(np.mean((corrected_refs[i][ok[i]] - field[i][ok[i]]) ** 2))) for i in range(len(self.el_refs))]
        names = [f"{r['roi_name']} / {r['name']}" for r in self.el_refs]
        result = {"wl": self.wavelengths.copy(), "gain": gain, "offset": offset, "field": field,
                  "corrected": corrected_refs, "names": names, "rmse": rmse, "n_used": n_used}
        params = {"references": names, "n_references": len(names),
                  "field_units": "percent" if scale != 1.0 else "fraction",
                  "input_stage": self.radiometric_stage}
        self._commit_cube(out, self.wavelengths, "Empirical line", params,
                          stage="Surface reflectance (empirical line)")
        self.el_result = result
        self.btn_el_plot_gain.setEnabled(True)
        self.btn_el_plot_refs.setEnabled(True)

        n_origin = int(through_origin.sum())
        msg = (f"Empirical line applied with {len(names)} reference(s) to {bands} bands. "
               f"Mean RMSE at the references: {np.mean(rmse):.4f} (reflectance units).")
        if n_origin:
            msg += (f" {n_origin} band(s) had only one usable reference (or no contrast between references) "
                    "and used offset = 0.")
        if len(names) == 1:
            msg += " With a single reference the fit is exact by construction, so RMSE says nothing about accuracy."
        self.el_status_label.setText(msg)
        self.info_label.setText("Empirical line correction applied. Active stage: surface reflectance.")
        self.on_el_plot_references()

    def on_el_plot_gain_offset(self):
        if self.el_result is None:
            QtWidgets.QMessageBox.information(self, "No result", "Apply empirical line first.")
            return
        r = self.el_result

        def build(fig):
            ax1, ax2 = fig.subplots(2, 1, sharex=True)
            ax1.plot(r["wl"], r["gain"], color="tab:blue")
            ax1.set_ylabel("Gain")
            ax1.set_title("Empirical line gain and offset per band")
            ax1.grid(True, alpha=0.3)
            ax2.plot(r["wl"], r["offset"], color="tab:orange")
            ax2.set_ylabel("Offset")
            ax2.set_xlabel("Wavelength (nm)")
            ax2.grid(True, alpha=0.3)
            fig.tight_layout()

        self._show_figure_dialog("Empirical line: gain / offset", build, size=(9, 6))

    def on_el_plot_references(self):
        if self.el_result is None:
            QtWidgets.QMessageBox.information(self, "No result", "Apply empirical line first.")
            return
        r = self.el_result
        colors = class_colors(max(1, len(r["names"])))

        def build(fig):
            ax = fig.add_subplot(111)
            for i, name in enumerate(r["names"]):
                ax.plot(r["wl"], r["field"][i], "-", color=colors[i], label=f"{name} - field")
                ax.plot(r["wl"], r["corrected"][i], "--", color=colors[i],
                        label=f"{name} - corrected image (RMSE {r['rmse'][i]:.4f})")
            ax.set_xlabel("Wavelength (nm)")
            ax.set_ylabel("Surface reflectance")
            ax.set_title("Reference spectra: field (solid) vs corrected image ROI mean (dashed)")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=7)
            fig.tight_layout()

        self._show_figure_dialog("Empirical line: reference spectra", build, size=(9, 6))

    # ------------------------------------------------------------------
    # Atmospheric correction: dark pixel subtraction, flat field, IARR
    # ------------------------------------------------------------------
    def _selected_roi_union(self, what):
        """Union mask of the ROIs selected in the ROI Manager list, or None after telling the user."""
        rows = sorted(self._selected_roi_rows())
        if not rows:
            QtWidgets.QMessageBox.information(
                self, "No ROI selected",
                f"Select the {what} ROI(s) in the ROI Manager list (ROI Analysis tab) first.")
            return None
        mask = np.zeros(self.cube.shape[:2], dtype=bool)
        for r in rows:
            mask |= self.rois[r]["mask"]
        return mask

    def _atm_precheck(self, relative_output=False):
        if self._busy:
            return False
        if self.cube is None:
            QtWidgets.QMessageBox.warning(self, "No data", "Load a hyperspectral cube first.")
            return False
        stage = self.radiometric_stage.lower()
        if relative_output and stage.startswith(("surface reflectance", "relative reflectance")):
            ans = QtWidgets.QMessageBox.question(
                self, "Already reflectance",
                f"The active data stage is '{self.radiometric_stage}'. Apply this correction anyway?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No, QtWidgets.QMessageBox.No)
            if ans != QtWidgets.QMessageBox.Yes:
                return False
        return True

    def _show_atm_comparison(self, title, ref_label, ref_spectrum, before_mean, after_mean,
                             rgb_pair=None):
        """Spectra figure; with rgb_pair=(before_rgb, after_rgb, label) the two RGB images go on top."""
        wl = self.wavelengths
        bands = len(ref_spectrum)
        x = wl if wl is not None and len(wl) == bands else np.arange(bands)
        xlabel = "Wavelength (nm)" if x is wl else "Band"

        def build(fig):
            if rgb_pair is not None:
                gs = fig.add_gridspec(3, 2, height_ratios=[2.4, 1, 1])
                axa, axb = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
                axa.imshow(rgb_pair[0]); axa.set_title("Original (RGB)", fontsize=10)
                axb.imshow(rgb_pair[1]); axb.set_title(f"After {rgb_pair[2]} (RGB)", fontsize=10)
                for a in (axa, axb):
                    a.set_xticks([]); a.set_yticks([])
                ax1 = fig.add_subplot(gs[1, :])
                ax2 = fig.add_subplot(gs[2, :], sharex=ax1)
            else:
                ax1, ax2 = fig.subplots(2, 1, sharex=True)
            ax1.plot(x, ref_spectrum, color="tab:red")
            ax1.set_title(f"{title}: {ref_label}")
            ax1.grid(True, alpha=0.3)
            ax2.plot(x, before_mean, label="Scene mean before")
            ax2.plot(x, after_mean, label="Scene mean after")
            ax2.set_xlabel(xlabel)
            ax2.grid(True, alpha=0.3)
            ax2.legend(fontsize=8)
            fig.tight_layout()

        self._show_figure_dialog(title, build, size=(9, 9.5) if rgb_pair is not None else (9, 6.5))

    def _on_dark_mode_changed(self, index):
        self.dark_scalar_spin.setEnabled(index == 0)
        self.dark_pct_spin.setEnabled(index == 2)
        self.btn_dark_file.setEnabled(index == 3)

    def on_dark_choose_file(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Dark pixel values", "", "Dark values (*.npy *.csv *.txt);;All files (*)")
        if not path:
            return
        try:
            arr = load_dark_array(path)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Could not read the file:\n{e}")
            return
        self.dark_file_array = arr
        self.dark_file_name = Path(path).name
        shape = "scalar" if arr.size == 1 else " x ".join(str(n) for n in arr.shape)
        self.dark_file_label.setText(f"{self.dark_file_name} ({shape})")

    def on_dark_apply(self):
        if not self._atm_precheck():
            return
        mode = self.dark_mode_combo.currentIndex()
        bands = self.cube.shape[2]
        try:
            if mode == 0:
                dark = float(self.dark_scalar_spin.value())
                desc, params = f"scalar {dark:g}", {"source": "scalar", "value": dark}
            elif mode == 1:
                mask = self._selected_roi_union("dark")
                if mask is None:
                    return
                dark = self._roi_mean_spectrum(mask)
                desc = f"mean of {int(mask.sum())} dark ROI pixels per band"
                params = {"source": "ROI mean", "pixels": int(mask.sum())}
            elif mode == 2:
                pct = float(self.dark_pct_spin.value())
                flat = self.cube.reshape(-1, bands)
                if flat.shape[0] > 500000:                              # subsample large scenes
                    sel = np.random.default_rng(0).choice(flat.shape[0], 500000, replace=False)
                    flat = flat[np.sort(sel)]
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    dark = np.nanpercentile(flat, pct, axis=0)
                desc = f"darkest pixels ({pct:g}th percentile per band)"
                params = {"source": "percentile", "percentile": pct}
            else:
                if self.dark_file_array is None:
                    QtWidgets.QMessageBox.information(self, "No file", "Choose a dark-value file first.")
                    return
                dark = self.dark_file_array
                desc = f"file {self.dark_file_name}"
                params = {"source": "file", "file": self.dark_file_name}
            if np.any(~np.isfinite(np.asarray(dark, dtype=float))):
                raise ValueError("The dark values contain NaN / infinite entries (for example a band that is "
                                 "empty inside the ROI).")
            QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
            try:
                resolved = resolve_dark_pixels(dark, self.cube.shape)
                out = subtract_dark_pixels(self.cube, dark, clip_negative=self.dark_clip_check.isChecked())
            finally:
                QtWidgets.QApplication.restoreOverrideCursor()
        except ValueError as e:
            QtWidgets.QMessageBox.warning(self, "Dark pixel subtraction", str(e))
            return
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Dark pixel subtraction failed:\n{e}")
            return
        before = scene_mean_spectrum(self.cube)
        orig_cube = self.cube
        rgb_before, rgb_after, _ = rgb_pair_shared_stretch(orig_cube, out, self.wavelengths)
        if resolved.shape[2] == 1:                                   # scalar or per-pixel: same for every band
            ref_spec = np.full(bands, float(np.nanmean(resolved)))
        else:
            ref_spec = np.nanmean(resolved, axis=(0, 1))
        n_neg = float(np.mean(out < 0)) * 100.0
        params["clip_negative"] = bool(self.dark_clip_check.isChecked())
        self._commit_cube(out, self.wavelengths, "Dark pixel subtraction", params)
        msg = f"Subtracted {desc}."
        if n_neg > 0:
            msg += f" {n_neg:.2f}% of the values are now negative (tick 'Clip negative results' to set them to 0)."
        self.dark_status_label.setText(msg)
        self.info_label.setText("Dark pixel subtraction applied.")
        self._show_atm_comparison("Dark pixel subtraction", "dark value subtracted per band",
                                  ref_spec, before, scene_mean_spectrum(self.cube),
                                  rgb_pair=(rgb_before, rgb_after, "dark pixel subtraction"))

    def on_run_hfc(self):
        if self._busy:
            return
        if self.cube is None:
            QtWidgets.QMessageBox.warning(self, "No data", "Load a hyperspectral cube first.")
            return
        pfa = float(self.hfc_pfa_spin.value())
        whiten = self.hfc_whiten_check.isChecked()
        cube = self.cube

        def job(progress, cancel):
            progress("Estimating the number of endmembers...")
            return count_endmembers_hfc(cube, pfa, whiten)

        def done(res):
            n, det = res
            self._hfc_details = det
            name = "NWHFC" if whiten else "HFC"
            self.hfc_result_label.setText(f"Estimated endmembers ({name}): {n}  (false-alarm probability {pfa:g})")
            self.btn_hfc_plot.setEnabled(True)
            if n >= 2:
                for sp in (self.spin_nmf_n, self.nfindr_n_spin, self.ppi_n_spin, self.fippi_n_spin):
                    sp.setValue(max(sp.minimum(), min(sp.maximum(), n)))
            self.info_label.setText(f"{name}: {n} endmembers estimated; the endmember-count boxes were set to {n}.")
        self._start_job(job, done, "Estimating the number of endmembers...")

    def on_plot_hfc(self):
        det = getattr(self, "_hfc_details", None)
        if det is None:
            return

        def build(fig):
            ax = fig.add_subplot(111)
            k = np.arange(1, det["lam_corr"].size + 1)
            m = min(det["lam_corr"].size, max(15, int(np.count_nonzero(det["detected"])) * 3))
            ax.semilogy(k[:m], det["lam_corr"][:m], "o-", label="Correlation eigenvalues")
            ax.semilogy(k[:m], det["lam_cov"][:m], "s-", label="Covariance eigenvalues")
            ax.set_xlabel("Eigenvalue index")
            ax.set_ylabel("Eigenvalue")
            ax.set_title(f"{'NWHFC' if det['noise_whiten'] else 'HFC'}: {int(det['detected'].sum())} signal sources "
                         f"(Pfa {det['pfa']:g}, {det['pixels']} pixels)", fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.legend()
            fig.tight_layout()
        self._show_figure_dialog("HFC eigenvalue test", build, size=(8, 5))

    def on_run_nfindr(self):
        if self._busy:
            return
        if self.cube is None:
            QtWidgets.QMessageBox.warning(self, "No data", "Load a hyperspectral cube first.")
            return
        n = int(self.nfindr_n_spin.value())
        it = int(self.nfindr_iter_spin.value())
        cube = self.cube

        def job(progress, cancel):
            return nfindr_endmembers(cube, n, max_iter=it, progress=progress, cancel=cancel)

        def done(res):
            em, _idx = res
            self.nfindr_endmembers = em
            self.btn_nfindr_add.setEnabled(True)
            bands = em.shape[1]
            x = (np.asarray(self.wavelengths, dtype=float)
                 if self.wavelengths is not None and len(self.wavelengths) == bands else np.arange(1, bands + 1))
            xlabel = "Wavelength (nm)" if self.wavelengths is not None and len(self.wavelengths) == bands else "Band Number"
            colors = class_colors(em.shape[0])
            self.plot_canvas.show_multiline(
                x, [(f"Endmember {i + 1}", em[i], colors[i]) for i in range(em.shape[0])],
                title=f"Number of Endmembers: {em.shape[0]}", xlabel=xlabel, ylabel="Data Values")
            self.info_label.setText(f"N-FINDR found {em.shape[0]} endmembers.")
        self._start_job(job, done, f"Running N-FINDR with {n} endmembers...")

    def on_run_ppi(self):
        if self._busy:
            return
        if self.cube is None:
            QtWidgets.QMessageBox.warning(self, "No data", "Load a hyperspectral cube first.")
            return
        n = int(self.ppi_n_spin.value())
        nv = int(self.ppi_vectors_spin.value())
        red = self.ppi_reduction_combo.currentText()
        seed = int(self.ppi_seed_spin.value())
        cube = self.cube

        def job(progress, cancel):
            return ppi_endmembers(cube, n, nv, red, seed=seed, progress=progress, cancel=cancel)

        def done(res):
            em, _sel, counts = res
            self.ppi_endmembers = em
            self.btn_ppi_add.setEnabled(True)
            colors = class_colors(em.shape[0])
            self.plot_canvas.show_multiline(
                np.arange(1, em.shape[1] + 1),
                [(f"Endmember {i + 1}", em[i], colors[i]) for i in range(em.shape[0])],
                title=f"Endmembers Spectra - Number of Endmembers = {em.shape[0]}",
                xlabel="Band Number", ylabel="Pixel Values")
            self.info_label.setText(
                f"PPI ({red}, {nv} skewers) found {em.shape[0]} endmembers; "
                f"{int(np.count_nonzero(counts))} pixels were an extreme at least once.")
        self._start_job(job, done, f"Running PPI with {n} endmembers ({red}, {nv} skewers)...")

    def on_ppi_to_endmembers(self):
        if self.ppi_endmembers is None:
            return
        for e in self.ppi_endmembers:
            self.endmembers.append(np.asarray(e, dtype=float))
        self.endmember_label.setText(f"Endmembers: {len(self.endmembers)}")
        self.info_label.setText(f"Added {len(self.ppi_endmembers)} PPI endmembers to the linear unmixing list.")

    def on_run_fippi(self):
        if self._busy:
            return
        if self.cube is None:
            QtWidgets.QMessageBox.warning(self, "No data", "Load a hyperspectral cube first.")
            return
        n = int(self.fippi_n_spin.value())
        red = self.fippi_reduction_combo.currentText()
        it = int(self.fippi_iter_spin.value())
        cube = self.cube

        def job(progress, cancel):
            return fippi_endmembers(cube, n, red, it, progress=progress, cancel=cancel)

        def done(res):
            em, _sel, iters = res
            self.fippi_endmembers = em
            self.btn_fippi_add.setEnabled(True)
            bands = em.shape[1]
            x = np.arange(1, bands + 1)
            colors = class_colors(em.shape[0])
            self.plot_canvas.show_multiline(
                x, [(f"Endmember {i + 1}", em[i], colors[i]) for i in range(em.shape[0])],
                title=f"Endmembers Spectra - Number of Endmembers = {em.shape[0]}",
                xlabel="Band Number", ylabel="Pixel Values")
            self.info_label.setText(f"FIPPI ({red}) found {em.shape[0]} endmembers in {iters} iteration(s).")
        self._start_job(job, done, f"Running FIPPI with {n} endmembers ({red})...")

    def on_fippi_to_endmembers(self):
        if self.fippi_endmembers is None:
            return
        for e in self.fippi_endmembers:
            self.endmembers.append(np.asarray(e, dtype=float))
        self.endmember_label.setText(f"Endmembers: {len(self.endmembers)}")
        self.info_label.setText(f"Added {len(self.fippi_endmembers)} FIPPI endmembers to the linear unmixing list.")

    def on_nfindr_to_endmembers(self):
        if self.nfindr_endmembers is None:
            return
        for e in self.nfindr_endmembers:
            self.endmembers.append(np.asarray(e, dtype=float))
        self.endmember_label.setText(f"Endmembers: {len(self.endmembers)}")
        self.info_label.setText(f"Added {len(self.nfindr_endmembers)} N-FINDR endmembers to the linear unmixing list.")

    def _on_smile_method_changed(self, index):
        self.smile_window_spin.setEnabled(index == 0)
        self.smile_mnf_spin.setEnabled(index == 1)

    def on_smile_apply(self):
        if not self._atm_precheck():
            return
        bands = self.cube.shape[2]
        method = self.smile_method_combo.currentText()
        ba, bb = self.smile_band_a.value(), self.smile_band_b.value()
        if not (1 <= ba <= bands and 1 <= bb <= bands):
            QtWidgets.QMessageBox.warning(
                self, "Smile reduction", f"The check bands must be between 1 and {bands} (cube has {bands} bands).")
            return
        window = self.smile_window_spin.value()
        if method == "SpectralSmoothing":
            if window > bands:
                QtWidgets.QMessageBox.warning(
                    self, "Smile reduction", f"The window ({window}) cannot exceed the number of bands ({bands}).")
                return
            if window > 9:
                ans = QtWidgets.QMessageBox.question(
                    self, "Large window",
                    "A window larger than 9 can remove fine atmospheric absorption features. Continue?",
                    QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No, QtWidgets.QMessageBox.No)
                if ans != QtWidgets.QMessageBox.Yes:
                    return
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            if method == "SpectralSmoothing":
                out = spectral_moving_mean(self.cube, window)
                params = {"method": method, "window": window}
                desc = f"spectral smoothing, window {window}"
            else:
                k = min(self.smile_mnf_spin.value(), bands)
                out = mnf_smile_reduction(self.cube, k)
                params = {"method": method, "components": k}
                desc = f"MNF, {k} components kept"
        except Exception as e:
            QtWidgets.QApplication.restoreOverrideCursor()
            QtWidgets.QMessageBox.critical(self, "Smile reduction failed", str(e))
            return
        QtWidgets.QApplication.restoreOverrideCursor()
        before_diff = np.abs(self.cube[:, :, ba - 1].astype(np.float64) - self.cube[:, :, bb - 1].astype(np.float64))
        after_diff = np.abs(out[:, :, ba - 1].astype(np.float64) - out[:, :, bb - 1].astype(np.float64))
        self._commit_cube(out, self.wavelengths, "Smile reduction", params)
        self.smile_status_label.setText(f"Smile reduction applied ({desc}). Output type: {out.dtype}.")
        self.info_label.setText("Smile reduction applied.")

        def build(fig):
            axes = fig.subplots(1, 2)
            vmax = float(np.nanmax(np.concatenate([before_diff.ravel(), after_diff.ravel()])))
            for ax, img, ttl in ((axes[0], before_diff, f"Difference of Bands {ba} and {bb} Before Smile Correction"),
                                 (axes[1], after_diff, f"Difference of Bands {ba} and {bb} After Smile Correction")):
                im = ax.imshow(img, vmin=0, vmax=vmax)
                ax.set_title(ttl, fontsize=9)
                ax.axis("off")
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            fig.tight_layout()
        self._show_figure_dialog("Smile correction", build, size=(11, 5))

    def _on_ff_roi_mode_changed(self, index):
        for sp in self.ff_roi_spins:
            sp.setEnabled(index == 0)

    def on_ff_choose_reference(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Reference spectrum (e.g. ECOSTRESS)", "", "Spectra (*.txt *.csv *.asc *.dat);;All files (*)")
        if not path:
            return
        try:
            wl, val = load_field_spectrum(path)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Could not read the spectrum:\n{e}")
            return
        wl_nm = auto_wavelengths_to_nm(wl)
        self.ff_ref = (wl_nm, val, Path(path).name)
        self.ff_ref_label.setText(f"{Path(path).name}: {wl_nm.size} points, {wl_nm.min():.0f}-{wl_nm.max():.0f} nm")

    def on_ff_clear_reference(self):
        self.ff_ref = None
        self.ff_ref_label.setText("no reference spectrum (distance scores are skipped)")

    def _ff_roi_mask(self):
        """Boolean mask of the flat-field area, or None after telling the user why not."""
        if self.ff_roi_mode.currentIndex() == 1:
            return self._selected_roi_union("flat-field")
        r, c, h, w = (sp.value() for sp in self.ff_roi_spins)
        rows, cols = self.cube.shape[:2]
        mask = np.zeros((rows, cols), dtype=bool)
        mask[r - 1:r - 1 + h, c - 1:c - 1 + w] = True
        if not mask.any():
            QtWidgets.QMessageBox.warning(
                self, "Flat field",
                f"The rectangle (row {r}, column {c}, {h} x {w}) lies outside the {rows} x {cols} image.")
            return None
        return mask

    def _show_match_scores(self, score_before, score_after, ref_name):
        def build(fig):
            ax = fig.add_subplot(111)
            both = np.hstack([score_before, score_after])
            im = ax.imshow(both, cmap="viridis")
            fig.colorbar(im, ax=ax, label="Spectral angle (rad): low = stronger match")
            ax.set_title("Original Distance Score | Distance Score after Flat Field Correction", fontsize=10)
            ax.axvline(score_before.shape[1] - 0.5, color="white", lw=0.8)
            ax.set_xticks([]); ax.set_yticks([])
            fig.tight_layout()
        self._show_figure_dialog(f"Spectral match to {ref_name}", build, size=(10, 5))

    def on_flat_field_apply(self):
        if not self._atm_precheck(relative_output=True):
            return
        mask = self._ff_roi_mask()
        if mask is None:
            return
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            flat_spec = self._roi_mean_spectrum(mask)
            before = scene_mean_spectrum(self.cube)
            out, ref = flat_field_correct(self.cube, flat_spec)
            orig_cube = self.cube
            rgb_before, _ = rgb_estimate(orig_cube, self.wavelengths)       # contrast-stretched RGB
            rgb_after, _ = rgb_estimate(out, self.wavelengths)
        except ValueError as e:
            QtWidgets.QApplication.restoreOverrideCursor()
            QtWidgets.QMessageBox.warning(self, "Flat field", str(e))
            return
        except Exception as e:
            QtWidgets.QApplication.restoreOverrideCursor()
            QtWidgets.QMessageBox.critical(self, "Error", f"Flat field correction failed:\n{e}")
            return
        QtWidgets.QApplication.restoreOverrideCursor()
        roi_desc = ("rectangle [%d %d %d %d]" % tuple(sp.value() for sp in self.ff_roi_spins)
                    if self.ff_roi_mode.currentIndex() == 0 else "ROI Manager selection")
        self._commit_cube(out, self.wavelengths, "Flat field correction",
                          {"flat_field_pixels": int(mask.sum()), "roi": roi_desc},
                          stage="Relative reflectance (flat field)")
        self.ff_status_label.setText(
            f"Divided by the mean spectrum of {int(mask.sum())} flat-field pixels ({roi_desc}). "
            "Stage: relative reflectance.")
        self.info_label.setText("Flat field correction applied.")
        self._show_atm_comparison("Flat field correction", "mean spectrum of the flat-field area",
                                  ref, before, scene_mean_spectrum(self.cube),
                                  rgb_pair=(rgb_before, rgb_after, "flat field correction"))
        if self.ff_ref is not None:
            self._ff_spectral_match(orig_cube, out)

    def _ff_spectral_match(self, cube_before, cube_after):
        """Spectral-angle distance of every pixel to the reference spectrum, before and after."""
        if self.wavelengths is None:
            QtWidgets.QMessageBox.information(
                self, "Spectral match", "The cube has no wavelengths, so the reference spectrum cannot be matched.")
            return
        ref_wl, ref_val, name = self.ff_ref
        wl = np.asarray(self.wavelengths, dtype=float)
        ref = resample_field_spectrum(ref_wl, ref_val, wl)
        ok = np.isfinite(ref)
        if ok.sum() < 3:
            QtWidgets.QMessageBox.warning(
                self, "Spectral match",
                f"The reference spectrum ({ref_wl.min():.0f}-{ref_wl.max():.0f} nm) overlaps fewer than 3 cube bands "
                f"({wl.min():.0f}-{wl.max():.0f} nm).")
            return
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            rows, cols = cube_before.shape[:2]
            scores = []
            for cube in (cube_before, cube_after):
                X = np.asarray(cube[:, :, ok], dtype=np.float32).reshape(-1, int(ok.sum()))
                _, ang = spectral_angle_map(X, ref[ok][None, :])
                scores.append(ang.reshape(rows, cols))
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        msg = f"Spectral match to {name} used {int(ok.sum())} of {wl.size} bands."
        self.ff_status_label.setText(self.ff_status_label.text() + " " + msg)
        self._show_match_scores(scores[0], scores[1], name)

    def on_iarr_apply(self):
        if not self._atm_precheck(relative_output=True):
            return
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            before = scene_mean_spectrum(self.cube)
            out, ref = iarr_correct(self.cube)
        except ValueError as e:
            QtWidgets.QApplication.restoreOverrideCursor()
            QtWidgets.QMessageBox.warning(self, "IARR", str(e))
            return
        except Exception as e:
            QtWidgets.QApplication.restoreOverrideCursor()
            QtWidgets.QMessageBox.critical(self, "Error", f"IARR failed:\n{e}")
            return
        QtWidgets.QApplication.restoreOverrideCursor()
        self._commit_cube(out, self.wavelengths, "IARR", {}, stage="Relative reflectance (IARR)")
        self.iarr_status_label.setText("Divided by the scene mean spectrum. Stage: relative reflectance.")
        self.info_label.setText("IARR correction applied.")
        self._show_atm_comparison("IARR", "scene mean spectrum (divisor)",
                                  ref, before, scene_mean_spectrum(self.cube))

    # ------------------------------------------------------------------
    # NGmeet noise reduction
    # ------------------------------------------------------------------
    def _on_ng_mode_changed(self, index):
        validate = index == 0
        self.ng_noise_spin.setEnabled(validate)
        self.ng_seed_spin.setEnabled(validate)

    def on_run_ngmeet(self):
        if self._busy:
            return
        if self.cube is None:
            QtWidgets.QMessageBox.warning(self, "No data", "Load a hyperspectral cube first.")
            return
        if importlib.util.find_spec("skimage") is None:
            QtWidgets.QMessageBox.warning(
                self, "Missing dependency",
                "NGmeet needs scikit-image (non-local means and SSIM).\nInstall it with:  pip install scikit-image")
            return
        validate = self.ng_mode_combo.currentIndex() == 0
        sigma = float(self.ng_sigma_spin.value())
        n_iter = int(self.ng_iter_spin.value())
        lam = float(self.ng_lambda_spin.value())
        rank = None if self.ng_rank_auto.isChecked() else int(self.ng_rank_spin.value())
        noise_std = float(self.ng_noise_spin.value())
        seed = int(self.ng_seed_spin.value())
        cube = self.cube
        self._ng_validate = validate

        def job(progress, cancel):
            return run_ngmeet_job(cube, validate, sigma, n_iter, lam, rank, noise_std, seed,
                                  progress=progress, cancel=cancel)

        self.ng_status_label.setText("Running NGmeet...")
        self._start_job(job, self._finish_ngmeet, "Starting NGmeet denoising...")

    def _finish_ngmeet(self, res):
        rows, cols, bands = res["clean"].shape
        wl = self.wavelengths
        p = res["params"]
        param_text = (f"Sigma={p['sigma']:g}, iterations={p['iterations']}, "
                      f"lambda={p['lambda']:g}, rank={p['rank']}")
        if res["validate"]:
            m = res["metrics"]
            rgb_clean, idx = rgb_estimate(res["clean"], wl)
            rgb_noisy, _ = rgb_estimate(res["noisy"], wl)
            rgb_den, _ = rgb_estimate(res["denoised"], wl)
            images = [
                ("Original", rgb_clean),
                (f"Noisy (std {p['noise_std']:g})\nPSNR {m['psnr_noisy']:.2f} dB, SSIM {m['ssim_noisy']:.3f}", rgb_noisy),
                (f"NGmeet denoised\nPSNR {m['psnr_denoised']:.2f} dB, SSIM {m['ssim_denoised']:.3f}", rgb_den),
            ]
            self._ng_last_result = ("NGmeet validation", images, param_text, idx)
            self.ng_status_label.setText(
                f"Validation done ({param_text}). PSNR {m['psnr_noisy']:.2f} -> {m['psnr_denoised']:.2f} dB, "
                f"SSIM {m['ssim_noisy']:.3f} -> {m['ssim_denoised']:.3f}. The active cube was not changed."
            )
            self.info_label.setText("NGmeet validation finished (display only).")
        else:
            span = res["hi"] - res["lo"]
            out = res["denoised"].astype(np.float32) * span + res["lo"]
            flat = out.reshape(-1, bands)
            flat[~res["valid_px"]] = np.nan                      # fully invalid pixels stay NaN
            rgb_before, idx = rgb_estimate(self.cube, wl)
            rgb_after, _ = rgb_estimate(out, wl)
            images = [("Before (active cube)", rgb_before), ("After NGmeet denoising", rgb_after)]
            self._ng_last_result = ("NGmeet denoising", images, param_text, idx)
            self._commit_cube(out, wl, "NGmeet denoise", dict(p))
            self.ng_status_label.setText(
                f"Active cube denoised ({param_text}). Use View > Revert to original cube to undo."
            )
            self.info_label.setText("NGmeet denoising applied to the active cube.")
        self.btn_ng_compare.setEnabled(True)
        self._show_ng_figure()

    def _show_ng_figure(self):
        if self._ng_last_result is None:
            QtWidgets.QMessageBox.information(self, "No result", "Run NGmeet first.")
            return
        title, images, param_text, idx = self._ng_last_result
        if self.wavelengths is not None and len(self.wavelengths) > max(idx):
            rgb_text = "RGB bands: " + ", ".join(f"{self.wavelengths[i]:.0f} nm" for i in idx)
        else:
            rgb_text = f"RGB band indices: {idx}"

        def build(fig):
            axes = fig.subplots(1, len(images))
            for ax, (t, img) in zip(np.atleast_1d(axes), images):
                ax.imshow(img)
                ax.set_title(t, fontsize=10)
                ax.axis("off")
            fig.suptitle(f"{title} ({param_text}; {rgb_text}; 2-98% contrast stretch)", fontsize=10)
            fig.tight_layout()

        self._show_figure_dialog(title, build, size=(4.6 * len(images) + 1, 5.2))

    # ------------------------------------------------------------------
    # NMF unmixing
    # ------------------------------------------------------------------
    def on_run_nmf(self):
        if self._busy:
            return
        if self.cube is None:
            QtWidgets.QMessageBox.warning(self, "No data", "Load a hyperspectral cube first.")
            return
        if not SKLEARN_AVAILABLE:
            QtWidgets.QMessageBox.warning(self, "Missing dependency", "Install scikit-learn first.")
            return
        n = int(self.spin_nmf_n.value())
        b0, b1 = int(self.spin_nmf_b0.value()), int(self.spin_nmf_b1.value())
        if b1 - b0 + 1 < max(n, 2):
            QtWidgets.QMessageBox.information(
                self, "Band range",
                f"Choose a band range with at least {max(n, 2)} bands (currently {max(0, b1 - b0 + 1)})."
            )
            return
        fit_px = int(self.spin_nmf_px.value())
        seed = int(self.spin_nmf_seed.value())
        cube = self.cube

        def job(progress, cancel):
            return run_nmf_unmix(cube, n, b0, b1, fit_px, seed=seed,
                                 progress=progress, cancel=cancel)

        self._start_job(job, self._finish_nmf, f"Running NMF with {n} endmembers...")

    def _finish_nmf(self, res):
        self.nmf_abundance = res["abundance"]
        self.nmf_endmembers = res["endmembers"]
        self.nmf_band_range = res["band_range"]
        k = self.nmf_abundance.shape[2]
        self.spin_nmf_show.setMaximum(k)
        self.spin_nmf_show.setValue(1)
        names = [f"Endmember {i + 1}" for i in range(k)]
        self.class_labels = res["classified"]
        self.class_names_result = names
        self._show_classification(res["classified"], names, "NMF unmixing")
        self.on_plot_endmembers()
        if res["n_negative"]:
            self.info_label.setText(
                self.info_label.text()
                + f"  (Note: {res['n_negative']} negative values were clipped to 0 for NMF.)"
            )

    def on_show_abundance(self):
        if self.nmf_abundance is None:
            QtWidgets.QMessageBox.information(self, "NMF not run", "Run NMF unmixing first.")
            return
        i = int(self.spin_nmf_show.value()) - 1
        if i < 0 or i >= self.nmf_abundance.shape[2]:
            return
        ab = self.nmf_abundance[:, :, i]
        self.image_canvas.show_image(self.robust_normalize(ab), cmap="viridis")
        self.band_label.setText(f"NMF abundance of endmember {i + 1}")
        self._draw_roi_patch()
        self.info_label.setText(
            f"Abundance of endmember {i + 1}: min {np.nanmin(ab):.4g}, max {np.nanmax(ab):.4g} "
            "(display stretched between the 2nd and 98th percentiles)."
        )

    def on_show_all_abundance(self):
        if self.nmf_abundance is None:
            QtWidgets.QMessageBox.information(self, "NMF not run", "Run NMF unmixing first.")
            return
        ab = self.nmf_abundance
        k = ab.shape[2]
        ncols = 2 if k <= 4 else 3
        nrows = int(math.ceil(k / ncols))

        def build(fig):
            for i in range(nrows * ncols):
                ax = fig.add_subplot(nrows, ncols, i + 1)
                if i < k:
                    ax.imshow(ab[:, :, i], cmap="viridis")
                    ax.set_title(f"Abundance of Endmember {i + 1}", fontsize=9)
                ax.axis("off")
            fig.tight_layout()

        self._show_figure_dialog("NMF abundance maps", build, size=(4.2 * ncols, 3.8 * nrows))

    def on_plot_endmembers(self):
        if self.nmf_endmembers is None:
            QtWidgets.QMessageBox.information(self, "NMF not run", "Run NMF unmixing first.")
            return
        b0, b1 = self.nmf_band_range
        H = self.nmf_endmembers
        if self.wavelengths is not None and len(self.wavelengths) == self.cube.shape[2]:
            x, xlabel = self.wavelengths[b0:b1 + 1], "Wavelength (nm)"
        else:
            x, xlabel = np.arange(b0, b1 + 1), "Band"
        colors = class_colors(H.shape[0])
        series = [(f"Endmember {i + 1}", H[i], colors[i]) for i in range(H.shape[0])]
        self.plot_canvas.show_multiline(x, series, title="Extracted endmembers (NMF)",
                                        xlabel=xlabel, ylabel="Reflectance")

    # ------------------------------------------------------------------
    # Signal-to-noise ratio
    # ------------------------------------------------------------------
    def on_compute_snr(self):
        if self.cube is None:
            QtWidgets.QMessageBox.warning(self, "No data", "Load a hyperspectral cube first.")
            return
        mask = None
        if self.cb_snr_roi.isChecked():
            if not self.rois:
                QtWidgets.QMessageBox.information(
                    self, "No ROIs", "Add ROIs first, or untick the ROI-only option."
                )
                return
            mask = np.zeros(self.cube.shape[:2], dtype=bool)
            for roi in self.rois:
                mask |= roi["mask"]

        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            self.snr_band, self.snr_pixel = compute_snr(self.cube, mask)
            src = "ROI pixels" if mask is not None else "whole image"
            self.snr_label.setText(
                f"Average spatial-domain SNR ({src}): {np.nanmean(self.snr_band):.2f}\n"
                f"Average spectral-domain SNR: {np.nanmean(self.snr_pixel):.2f}"
            )
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to compute SNR:\n{e}")
            return
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        self.on_show_snr_curve()

    def on_show_snr_curve(self):
        if self.snr_band is None:
            QtWidgets.QMessageBox.information(self, "SNR not computed", "Press 'Compute SNR' first.")
            return
        x, xlabel = self._x_axis(np.ones(self.cube.shape[2], dtype=bool))
        self.plot_canvas.show_lineplot(
            x, self.snr_band, title="Spatial-domain SNR per band", xlabel=xlabel, ylabel="SNR"
        )
        self.info_label.setText(
            f"Spatial-domain SNR: mean {np.nanmean(self.snr_band):.2f}, "
            f"min {np.nanmin(self.snr_band):.2f}, max {np.nanmax(self.snr_band):.2f}."
        )

    def on_show_snr_map(self):
        if self.snr_pixel is None:
            QtWidgets.QMessageBox.information(self, "SNR not computed", "Press 'Compute SNR' first.")
            return
        m = self.snr_pixel
        self.image_canvas.show_image(self.robust_normalize(m), cmap="viridis")
        self.band_label.setText("Spectral-domain SNR map (mean / std across bands)")
        self._draw_roi_patch()
        self.info_label.setText(
            f"Spectral-domain SNR map: mean {np.nanmean(m):.2f}, min {np.nanmin(m):.2f}, "
            f"max {np.nanmax(m):.2f} (display stretched between the 2nd and 98th percentiles)."
        )

    def on_save_snr_csv(self):
        if self.snr_band is None:
            QtWidgets.QMessageBox.information(self, "SNR not computed", "Press 'Compute SNR' first.")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save SNR curve", "snr_per_band.csv", "CSV files (*.csv)"
        )
        if not path:
            return
        bands = len(self.snr_band)
        idx = np.arange(bands)
        try:
            if self.wavelengths is not None and len(self.wavelengths) == bands:
                data = np.column_stack([idx, self.wavelengths, self.snr_band])
                header = "band,wavelength,snr"
            else:
                data = np.column_stack([idx, self.snr_band])
                header = "band,snr"
            np.savetxt(path, data, delimiter=",", header=header, comments="", fmt="%.6g")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to save CSV:\n{e}")
            return
        self.info_label.setText(f"Saved SNR curve to {path}")



def main():
    try:
        QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    except Exception:
        pass
    try:
        QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)
    except Exception:
        pass
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setOrganizationName(SETTINGS_ORG)
    global _ORIG_STYLE
    _ORIG_STYLE = app.style().objectName()       # native style, needed for the light theme
    theme = str(QtCore.QSettings(SETTINGS_ORG, APP_NAME).value("theme", "dark"))
    if theme not in THEMES:
        theme = "dark"
    set_matplotlib_theme(theme)
    apply_app_theme(app, theme)
    viewer = HyperspectralViewer()
    viewer.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
