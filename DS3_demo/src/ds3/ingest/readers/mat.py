"""MATLAB .mat hyperspectral cubes (e.g. PaviaU.mat, Indian_pines_corrected.mat, Salinas_corrected.mat).

A .mat cube is a bare array: it carries NO wavelengths. The wavelengths therefore come from, in this order:
  1. wavelengths=[...] / wavelength_range=(lo_nm, hi_nm)  given by the user (always preferred)
  2. a 1-D variable in the same file called wavelength / wvl / lambda with one value per band
  3. a small table of well-known public scenes (matched on variable name + band count). These are NOMINAL, evenly
     spaced centres taken from the sensor's published range, NOT read from the file; the reader writes that into
     the cube notes so it is visible in the app.
If none applies a MissingWavelengthsError explains what to pass.

Reads MATLAB v5 (scipy) and v7.3 / HDF5 (h5py). The array is loaded into memory (fine for the usual benchmark scenes).
Axis order is taken as (rows, cols, bands); pass band_axis= if yours differs.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
from ..contract import Cube, to_physical
from ..metadata import (standardize_spectral_axis, infer_unit, MissingWavelengthsError, InvalidHeaderError,
                        UnsupportedFormatError)

# (variable-name fragment, bands) -> (lo_nm, hi_nm, sensor, note)
KNOWN_SCENES = [
    ("pavia", 103, 430.0, 860.0, "ROSIS-03", "ROSIS-03 nominal range 430-860 nm (Pavia University, 103 bands used)"),
    ("pavia", 102, 430.0, 860.0, "ROSIS-03", "ROSIS-03 nominal range 430-860 nm (Pavia Centre, 102 bands used)"),
    ("indian", 200, 400.0, 2500.0, "AVIRIS", "AVIRIS nominal range 400-2500 nm (Indian Pines, 200 bands after water-band removal)"),
    ("indian", 220, 400.0, 2500.0, "AVIRIS", "AVIRIS nominal range 400-2500 nm (Indian Pines, 220 bands)"),
    ("salinas", 204, 400.0, 2500.0, "AVIRIS", "AVIRIS nominal range 400-2500 nm (Salinas, 204 bands after water-band removal)"),
    ("salinas", 224, 400.0, 2500.0, "AVIRIS", "AVIRIS nominal range 400-2500 nm (Salinas, 224 bands)"),
]
SKIP_VARS = ("__header__", "__version__", "__globals__")


def _load_all(path):
    """{name: ndarray} of every numeric array in the file."""
    out = {}
    try:
        from scipy.io import loadmat
        d = loadmat(str(path), squeeze_me=False)
        for k, v in d.items():
            if k in SKIP_VARS or k.startswith("__"):
                continue
            if isinstance(v, np.ndarray) and v.dtype.kind in "iufb":
                out[k] = v
        return out
    except NotImplementedError:
        pass                                    # v7.3 -> HDF5
    except ImportError as e:
        raise UnsupportedFormatError("reading .mat needs scipy: pip install scipy") from e
    except Exception as e:
        if "Please use HDF reader" not in str(e):
            raise InvalidHeaderError(f"{Path(path).name}: not a readable MATLAB file ({e})") from e
    import h5py
    with h5py.File(path, "r") as f:
        for k, o in f.items():
            if isinstance(o, h5py.Dataset) and o.dtype.kind in "iufb":
                out[k] = np.asarray(o).transpose()      # MATLAB stores column-major -> reverse axes
    return out


def inspect_mat(path):
    """[(name, shape, dtype)] for every numeric array - used for classification and error messages."""
    return [(k, v.shape, str(v.dtype)) for k, v in _load_all(path).items()]


def _find_cube(arrs, variable):
    if variable:
        if variable not in arrs:
            raise InvalidHeaderError(f"variable '{variable}' not in file; found {[(k, v.shape) for k, v in arrs.items()]}")
        if arrs[variable].ndim != 3:
            raise InvalidHeaderError(f"variable '{variable}' is {arrs[variable].ndim}-D, a cube must be 3-D")
        return variable
    c = [(k, v) for k, v in arrs.items() if v.ndim == 3]
    if not c:
        raise UnsupportedFormatError(
            f"no 3-D array in file (found {[(k, v.shape) for k, v in arrs.items()]}). A 2-D array is usually a "
            "ground-truth / label map: import the image first, then use it as 'labels' to build class-mean spectra.")
    return max(c, key=lambda t: t[1].size)[0]


def read_mat(path, unit=None, wavelengths=None, wavelength_range=None, variable=None, band_axis=None,
             wavelength_units=None, reflectance_divisor=None, **_):
    arrs = _load_all(path)
    name = _find_cube(arrs, variable)
    a = arrs[name]
    notes = [f"MATLAB variable '{name}' shape {a.shape}"]
    ax = band_axis if band_axis is not None else 2
    if ax != 2:
        a = np.moveaxis(a, ax, 2)
    rows, cols, nb = a.shape
    sensor = None
    units = wavelength_units
    if wavelengths is not None:
        wl = np.asarray(wavelengths, float)
        notes.append("wavelengths supplied by the user")
    elif wavelength_range is not None:
        lo, hi = map(float, wavelength_range)
        wl = np.linspace(lo, hi, nb)
        notes.append(f"wavelengths ASSUMED evenly spaced {lo:g}-{hi:g} nm (user-supplied range), not read from the file")
    else:
        wl = None
        for k, v in arrs.items():
            if v.ndim <= 2 and v.size == nb and any(t in k.lower() for t in ("wavelength", "wvl", "lambda")):
                wl = np.ravel(v).astype(float); notes.append(f"wavelengths read from variable '{k}'"); break
        if wl is None:
            for frag, n, lo, hi, sens, note in KNOWN_SCENES:
                if frag in name.lower() and n == nb:
                    wl = np.linspace(lo, hi, nb); sensor = sens
                    notes.append(f"wavelengths ASSUMED from known scene: {note}; evenly spaced, not read from the file. "
                                 "Pass wavelengths= / wavelength_range= to override.")
                    break
        if wl is None:
            raise MissingWavelengthsError(
                f"{Path(path).name}: a .mat cube has no wavelengths (variable '{name}', {nb} bands). "
                f"Give the wavelength range, e.g. wavelength_range=(430, 860), or a list of {nb} centres.")
    if len(wl) != nb:
        raise InvalidHeaderError(f"{len(wl)} wavelengths but the cube has {nb} bands")
    kind = a.dtype.kind
    if unit is None:
        unit, _ = infer_unit(a[:32, :32].astype(np.float64), "f" if reflectance_divisor else kind)
        if reflectance_divisor:
            unit = "reflectance"
        notes.append(f"unit inferred as '{unit}'; pass unit= to override")
    div = float(reflectance_divisor) if reflectance_divisor else 1.0
    spec = standardize_spectral_axis(wl, units, None, None)
    data = a

    def raw_loader(r0, r1, c0, c1):
        return to_physical(data[r0:r1, c0:c1, :], None, 1.0 / div, 0.0)

    return Cube.make(raw_loader, (rows, cols, nb), spec, unit, sensor=sensor, source_path=str(path),
                     source_format="mat", metadata=dict(variable=name, notes=notes))
