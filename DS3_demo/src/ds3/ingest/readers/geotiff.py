"""GeoTIFF reader (rasterio). Spectral metadata is NOT standardised in TIFF, so
wavelengths are searched in a fixed priority order (see read_geotiff)."""
from __future__ import annotations
import re
from pathlib import Path
import numpy as np
import rasterio
from rasterio.windows import Window
from ..contract import Cube, to_physical
from ..metadata import (standardize_spectral_axis, parse_number_list, parse_wavelength_strings,
                        infer_unit, MissingWavelengthsError)
from .envi import parse_envi_header
from .wavelength_files import read_wavelengths_file, find_sibling_xml

WL_KEYS = ("wavelength", "WAVELENGTH", "center_wavelength", "CENTER_WAVELENGTH", "wavelength_nm", "WAVELENGTH_NM")


def _find_wavelengths(ds, path, user_wl):
    tried = []
    n = ds.count
    if user_wl is not None:
        wl = np.asarray(user_wl, float)
        if len(wl) != n:
            raise MissingWavelengthsError(f"wavelengths= has {len(wl)} values but file has {n} bands")
        return wl, None, None, None, "user argument"
    # 1. ENVI-style domain tags
    env = ds.tags(ns="ENVI")
    if "wavelength" in env:
        wl = parse_number_list(env["wavelength"])
        if len(wl) == n:
            fw = parse_number_list(env["fwhm"]) if "fwhm" in env else None
            bbl = parse_number_list(env["bbl"]) if "bbl" in env else None
            bad = (np.array(bbl) == 0) if bbl and len(bbl) == n else None
            return wl, env.get("wavelength_units") or env.get("wavelength units"), fw, bad, "ENVI tag domain"
    tried.append("ENVI tag domain")
    # 2. dataset-level tags
    tags = ds.tags()
    for k in WL_KEYS:
        if k in tags:
            wl = parse_number_list(tags[k])
            if len(wl) == n:
                return wl, tags.get("wavelength_units"), None, None, f"dataset tag '{k}'"
    tried.append("dataset tags")
    # 3. per-band tags
    per = []
    for i in range(1, n + 1):
        bt = ds.tags(i)
        v = next((bt[k] for k in WL_KEYS if k in bt), None)
        per.append(v)
    if all(v is not None for v in per):
        try:
            return [float(parse_number_list(v)[0]) for v in per], tags.get("wavelength_units"), None, None, "per-band tags"
        except Exception:
            pass
    tried.append("per-band tags")
    # 4. band descriptions
    wl, ok = parse_wavelength_strings(list(ds.descriptions))
    if ok:
        return wl, "nm", None, None, "band descriptions"
    tried.append("band descriptions")
    # 5. sidecar .hdr
    for h in (Path(str(path) + ".hdr"), Path(path).with_suffix(".hdr")):
        if h.is_file():
            hd = parse_envi_header(h)
            if "wavelength" in hd:
                wl = parse_number_list(hd["wavelength"])
                if len(wl) == n:
                    return wl, hd.get("wavelength units"), None, None, "sidecar .hdr"
    tried.append("sidecar .hdr")
    # 6. sibling METADATA.XML (EnMAP keeps band centres there, not in the TIFF)
    xml = find_sibling_xml(path)
    if xml is not None:
        try:
            wl, fw, _ = read_wavelengths_file(xml)
            if len(wl) == n:
                return wl, "nm", fw, None, f"sibling {xml.name}"
            tried.append(f"{xml.name} ({len(wl)} bands there, {n} in the TIFF)")
        except Exception as e:
            tried.append(f"{xml.name} ({type(e).__name__})")
    else:
        tried.append("sibling METADATA.XML")
    raise MissingWavelengthsError(
        f"{Path(path).name}: no wavelengths found (tried: {', '.join(tried)}). "
        "Pass wavelengths=[...] to open_cube().")


def read_geotiff(path, unit=None, wavelengths=None, reflectance_divisor=None, **_):
    ds = rasterio.open(path)
    nb, nl, ns = ds.count, ds.height, ds.width
    wl, units, fwhm, bad, how = _find_wavelengths(ds, path, wavelengths)
    spec = standardize_spectral_axis(wl, units, fwhm, bad)
    spec["notes"].insert(0, f"wavelengths read from {how}")
    scale = float(ds.scales[0]) if ds.scales and ds.scales[0] not in (None, 0) else 1.0
    offset = float(ds.offsets[0]) if ds.offsets and ds.offsets[0] is not None else 0.0
    if reflectance_divisor:
        scale = scale / reflectance_divisor
    nodata = ds.nodata
    kind = np.dtype(ds.dtypes[0]).kind
    notes = []
    sensor = None
    if (re.search(r"enmap.*l2a", Path(path).name, re.I) and unit in (None, "reflectance") and not reflectance_divisor
            and kind in "iu" and scale == 1.0):
        # EnMAP L2A surface reflectance is stored as int16 = reflectance x 10000 (ASSUMED from product docs; override if wrong)
        reflectance_divisor, unit, sensor = 10000.0, "reflectance", "EnMAP"
        scale = scale / reflectance_divisor
        notes.append("EnMAP L2A file name detected: assumed int16 reflectance x 10000 (override with reflectance_divisor= / unit=)")
    if unit is None:
        tag_unit = (ds.tags().get("unit") or ds.tags().get("units") or "").lower()
        if tag_unit in ("reflectance", "radiance", "dn"):
            unit = tag_unit
        elif reflectance_divisor or scale != 1.0 and kind in "iu":
            unit = "reflectance"
        else:
            r0, c0 = max(0, nl // 2 - 16), max(0, ns // 2 - 16)
            samp = ds.read(window=Window(c0, r0, min(32, ns - c0), min(32, nl - r0)))
            samp = to_physical(np.moveaxis(samp, 0, -1), nodata, scale, offset)
            unit, _ = infer_unit(samp, "f" if scale != 1.0 else kind)
            notes.append(f"unit inferred as '{unit}'; pass unit= (and reflectance_divisor=) to override")
    path = str(path)
    ds.close()

    def raw_loader(r0, r1, c0, c1):
        with rasterio.open(path) as d:
            arr = d.read(window=Window(c0, r0, c1 - c0, r1 - r0))      # (b, r, c)
        return to_physical(np.moveaxis(arr, 0, -1), nodata, scale, offset)

    with rasterio.open(path) as d:
        tr = tuple(d.transform)[:6] if d.crs is not None or not d.transform.is_identity else None
        crs = d.crs.to_string() if d.crs else None
    return Cube.make(raw_loader, (nl, ns, nb), spec, unit, nodata=nodata, crs=crs, transform=tr,
                     sensor=sensor, source_path=path, source_format="geotiff",
                     metadata=dict(notes=notes))
