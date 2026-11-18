"""Fallback readers that make the Image module open (almost) any hyperspectral file.

read_any  - every array format the Data Aggregation loader knows (hvva.core.loaders): multi-page / ImageJ /
            planar TIFF stacks, NumPy .npy / .npz, NetCDF-3 / -4, HDF5 with odd layouts, MATLAB v5 / v7.3,
            EnMAP / DESIS product TIFFs with their METADATA.XML, magic-byte sniffing for unknown extensions.
            The whole cube is read into memory (fine for scenes up to a few GB).
read_gdal - any raster GDAL (rasterio) can open: ERDAS Imagine .img without an ENVI header, PCIDSK .pix,
            GDAL .vrt, JPEG 2000, NITF, PDS .lbl, FITS, HDF4 (Hyperion .L1R) ... Windowed (lazy) reads.

Both accept wavelengths=[...] or wavelength_range=(lo, hi) for files that carry no wavelengths, and raise
MissingWavelengthsError (the Image page then asks for the sensor range) when none are found."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..contract import Cube, to_physical
from ..metadata import InvalidHeaderError, MissingWavelengthsError, infer_unit, standardize_spectral_axis

UNITS = ("reflectance", "radiance", "dn", "transmittance", "emissivity", "absorbance")

# Layers of sensor products that are NOT the spectral image (EnMAP / DESIS / PRISMA folders hold many TIFFs)
import re as _re
PRODUCT_LAYERS = [
    (r"ACOUT_SICOR_CWV|_CWV\b|WATER_?VAPOU?R", "columnar water-vapour map of the atmospheric correction (1 band)"),
    (r"ACOUT_SICOR_AOT|_AOT\b|AEROSOL", "aerosol optical thickness map of the atmospheric correction (1 band)"),
    (r"QL_|QUICKLOOK", "quicklook picture (RGB / VNIR / SWIR preview)"),
    (r"QUALITY|_QA\b", "quality layer (cloud / haze / snow / cirrus / class flags)"),
    (r"PIXELMASK|DEADPIXEL", "pixel mask"),
]
PRODUCT_PREFIX = _re.compile(r"^(ENMAP|DESIS|PRS_)|ACOUT_", _re.I)


def product_layer_hint(path):
    """A readable message when `path` is an ancillary layer of a sensor product (EnMAP, DESIS, PRISMA), else None.
    Only file names of those products are checked, so a user's own file called 'quality.tif' is not affected."""
    name = Path(path).name
    up = name.upper()
    if not PRODUCT_PREFIX.search(up) or "SPECTRAL_IMAGE" in up:
        return None
    for pat, what in PRODUCT_LAYERS:
        if _re.search(pat, up):
            sensor = "EnMAP" if up.startswith("ENMAP") else ("DESIS" if up.startswith("DESIS") else
                                                             "PRISMA" if up.startswith("PRS_") else "product")
            tip = ("open the file ending in -SPECTRAL_IMAGE.TIF from the same folder (keep the -METADATA.XML next to it), "
                   "or the whole product .zip" if sensor == "EnMAP" else "open the product's spectral image file instead")
            return f"{name} is the {sensor} {what}, not spectral data - {tip}."
    return None


def _user_wavelengths(nb, wavelengths, wavelength_range, notes):
    if wavelengths is not None:
        wl = np.asarray(wavelengths, float)
        if len(wl) != nb:
            raise MissingWavelengthsError(f"{len(wl)} wavelengths given but the image has {nb} bands")
        notes.append("wavelengths supplied by the user")
        return wl
    if wavelength_range is not None:
        lo, hi = map(float, wavelength_range)
        notes.append(f"wavelengths ASSUMED evenly spaced {lo:g}-{hi:g} nm (user-supplied range), not read from the file")
        return np.linspace(lo, hi, nb)
    return None


def _missing(path, nb):
    return MissingWavelengthsError(
        f"{Path(path).name}: the image has {nb} bands but no wavelengths in the file. "
        "Give the sensor's wavelength range (for example 430-860) to open it.")


def _finish_unit(unit, sample, kind, notes):
    if unit in UNITS:
        return unit
    u, _ = infer_unit(sample, kind)
    notes.append(f"unit inferred as '{u}'; pass unit= to override")
    return u


# ----------------------------------------------------------------------------- everything hvva can read
def read_any(path, unit=None, wavelengths=None, wavelength_range=None, reflectance_divisor=None, band_axis=None,
             variable=None, **_):
    try:
        from hvva.core import quality as HQ
        from hvva.core.loaders import _WL_NAMES, open_source
        from hvva.core.scene import _is_envi, _is_native_npz, load_scene
    except Exception as e:                                       # noqa: BLE001
        raise InvalidHeaderError(f"{Path(path).name}: the general loader (hvva) is not available: {e}") from None
    kind = "f"
    try:
        real, _note = HQ.resolve_input(str(path))
        if _is_native_npz(real) or _is_envi(real):
            sc = load_scene(real)
        else:
            src = open_source(real)
            g = src.guess()
            if variable:
                g["cube_key"] = variable
            shape = src.arrays[g["cube_key"]][0]
            kind = np.dtype(src.arrays[g["cube_key"]][1]).kind
            if band_axis is not None:
                g["band_axis"] = int(band_axis)
            elif src.band_axis_hint is None and src.wavelengths is None:
                # a wavelength array in the file tells which axis is spectral (more reliable than "the shortest axis")
                for k, (s1, _dt) in src.arrays.items():
                    n = max(s1) if len(s1) in (1, 2) and (len(s1) == 1 or 1 in s1) else None
                    if n and _WL_NAMES.search(k) and n in shape and n != shape[g["band_axis"]]:
                        g["band_axis"], g["wavelengths_key"] = list(shape).index(n), k
                        break
            g.pop("labels_key", None)
            sc = src.build(**g)
    except MissingWavelengthsError:
        raise
    except Exception as e:                                       # noqa: BLE001
        raise InvalidHeaderError(f"{Path(path).name}: {e}") from None
    a = np.asarray(sc.cube, np.float32)
    if a.ndim != 3 or a.shape[2] < 2:
        raise InvalidHeaderError(product_layer_hint(path) or f"{Path(path).name}: a single-band image {a.shape[:2]} - "
                                 "not a hyperspectral cube (a cube needs many bands)")
    rows, cols, nb = a.shape
    notes = [n for n in (sc.unit_note,) if n]
    wl = _user_wavelengths(nb, wavelengths, wavelength_range, notes)
    if wl is None:
        if sc.wavelengths is None or len(sc.wavelengths) != nb:
            raise _missing(path, nb)
        wl = np.asarray(sc.wavelengths, float)
        notes.append("wavelengths read from the file")
    if reflectance_divisor:
        a = a / np.float32(reflectance_divisor)
        unit = unit or "reflectance"
    bad = None if sc.bad_file is None or len(sc.bad_file) != nb else np.asarray(sc.bad_file, bool)
    spec = standardize_spectral_axis(wl, "nm", None, bad)
    r0, c0 = max(0, rows // 2 - 16), max(0, cols // 2 - 16)
    unit = unit or (sc.unit if sc.unit in UNITS else None)
    unit = _finish_unit(unit, a[r0:r0 + 32, c0:c0 + 32], kind if not reflectance_divisor else "f", notes)
    crs = tr = None
    if sc.geo is not None:
        tr = tuple(float(v) for v in sc.geo.transform[:6])
        crs = sc.geo.crs if sc.geo.crs and "unknown" not in str(sc.geo.crs).lower() else None

    def raw_loader(r0, r1, c0, c1):
        return a[r0:r1, c0:c1, :]

    ext = Path(path).suffix.lower().lstrip(".") or "file"
    return Cube.make(raw_loader, (rows, cols, nb), spec, unit, crs=crs, transform=tr,
                     source_path=str(path), source_format=f"{ext} (general loader)",
                     metadata=dict(notes=notes))


# ----------------------------------------------------------------------------- anything GDAL can open
def read_gdal(path, unit=None, wavelengths=None, wavelength_range=None, reflectance_divisor=None, **_):
    try:
        import rasterio
        from rasterio.windows import Window
    except ImportError:
        raise InvalidHeaderError(f"{Path(path).name}: this format needs rasterio (GDAL): pip install rasterio") from None
    src = str(path)
    with rasterio.open(src) as ds:
        if ds.count == 0 and ds.subdatasets:                     # HDF4 / NetCDF containers: the subdataset with most bands
            best, nmax = None, 0
            for sd in ds.subdatasets:
                try:
                    with rasterio.open(sd) as s:
                        if s.count > nmax:
                            best, nmax = sd, s.count
                except Exception:                                # noqa: BLE001
                    continue
            if best is None:
                raise InvalidHeaderError(f"{Path(path).name}: no readable raster inside")
            src = best
    from .geotiff import _find_wavelengths
    with rasterio.open(src) as ds:
        nb, nl, ns = ds.count, ds.height, ds.width
        if nb < 2:
            raise InvalidHeaderError(product_layer_hint(path) or f"{Path(path).name}: only {nb} band - not a hyperspectral cube")
        notes = []
        wl = _user_wavelengths(nb, wavelengths, wavelength_range, notes)
        units = fwhm = bad = None
        if wl is None:
            try:
                wl, units, fwhm, bad, how = _find_wavelengths(ds, path, None)
                notes.append(f"wavelengths read from {how}")
            except MissingWavelengthsError:
                raise _missing(path, nb) from None
        spec = standardize_spectral_axis(wl, units, fwhm, bad)
        scale = float(ds.scales[0]) if ds.scales and ds.scales[0] not in (None, 0) else 1.0
        offset = float(ds.offsets[0]) if ds.offsets and ds.offsets[0] is not None else 0.0
        if reflectance_divisor:
            scale /= float(reflectance_divisor)
            unit = unit or "reflectance"
        nodata = ds.nodata
        kind = np.dtype(ds.dtypes[0]).kind
        r0, c0 = max(0, nl // 2 - 16), max(0, ns // 2 - 16)
        samp = to_physical(np.moveaxis(ds.read(window=Window(c0, r0, min(32, ns - c0), min(32, nl - r0))), 0, -1),
                           nodata, scale, offset)
        unit = _finish_unit(unit, samp, "f" if scale != 1.0 else kind, notes)
        tr = tuple(ds.transform)[:6] if ds.crs is not None or not ds.transform.is_identity else None
        crs = ds.crs.to_string() if ds.crs else None
        driver = ds.driver

    def raw_loader(r0, r1, c0, c1):
        with rasterio.open(src) as d:
            arr = d.read(window=Window(c0, r0, c1 - c0, r1 - r0))
        return to_physical(np.moveaxis(arr, 0, -1), nodata, scale, offset)

    return Cube.make(raw_loader, (nl, ns, nb), spec, unit, nodata=nodata, crs=crs, transform=tr,
                     source_path=str(path), source_format=f"gdal:{driver}", metadata=dict(notes=notes))
