"""ENVI reader (.hdr + flat binary, BSQ / BIL / BIP), lazy via numpy memmap."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from ..contract import Cube, to_physical
from ..metadata import (standardize_spectral_axis, parse_number_list, infer_unit,
                        InvalidHeaderError, IngestError)

DTYPES = {1: "u1", 2: "i2", 3: "i4", 4: "f4", 5: "f8", 12: "u2", 13: "u4", 14: "i8", 15: "u8"}
BIN_EXT = ["", ".img", ".dat", ".bin", ".raw", ".bsq", ".bil", ".bip", ".sli", ".hsi"]


def parse_envi_header(path):
    lines = Path(path).read_text(errors="replace").splitlines()
    if not lines or not lines[0].strip().upper().startswith("ENVI"):
        raise InvalidHeaderError(f"{path}: not an ENVI header (first line must be 'ENVI')")
    hdr, i = {}, 1
    while i < len(lines):
        line = lines[i]; i += 1
        if "=" not in line:
            continue
        key, val = line.split("=", 1)
        key, val = key.strip().lower(), val.strip()
        if val.startswith("{") and "}" not in val:
            while i < len(lines) and "}" not in val:
                val += " " + lines[i].strip(); i += 1
        hdr[key] = val
    return hdr


def _find_files(path):
    p = Path(path)
    if p.suffix.lower() == ".hdr":
        hdr_path = p
        stem = p.with_suffix("")
        cands = [stem] + [stem.with_suffix(e) if e else stem for e in BIN_EXT] + [Path(str(stem) + e) for e in BIN_EXT]
        data = next((c for c in cands if c.is_file() and c.suffix.lower() != ".hdr"), None)
    else:
        data = p
        hdr_path = next((h for h in (Path(str(p) + ".hdr"), p.with_suffix(".hdr")) if h.is_file()), None)
    if hdr_path is None or not hdr_path.is_file():
        raise InvalidHeaderError(f"no .hdr found for {path}")
    if data is None or not data.is_file():
        raise InvalidHeaderError(f"no binary data file found next to {hdr_path}")
    return hdr_path, data


def _get_int(hdr, key):
    try:
        return int(float(hdr[key]))
    except Exception:
        raise InvalidHeaderError(f"ENVI header missing/invalid '{key}'")


def read_envi(path, unit=None, reflectance_divisor=None, **_):
    hdr_path, data_path = _find_files(path)
    hdr = parse_envi_header(hdr_path)
    ns, nl, nb = _get_int(hdr, "samples"), _get_int(hdr, "lines"), _get_int(hdr, "bands")
    code = _get_int(hdr, "data type")
    if code not in DTYPES:
        raise InvalidHeaderError(f"unsupported ENVI data type {code}")
    interleave = hdr.get("interleave", "bsq").strip().lower()
    if interleave not in ("bsq", "bil", "bip"):
        raise InvalidHeaderError(f"unsupported interleave {interleave!r}")
    bo = "<" if int(float(hdr.get("byte order", 0))) == 0 else ">"
    dt = np.dtype(bo + DTYPES[code])
    offset = int(float(hdr.get("header offset", 0)))
    shape_file = {"bsq": (nb, nl, ns), "bil": (nl, nb, ns), "bip": (nl, ns, nb)}[interleave]
    need = offset + int(np.prod(shape_file)) * dt.itemsize
    if data_path.stat().st_size < need:
        raise InvalidHeaderError(f"{data_path.name} is {data_path.stat().st_size} bytes but header implies {need}")
    mm = np.memmap(data_path, dtype=dt, mode="r", offset=offset, shape=shape_file)

    # --- spectral metadata
    if "wavelength" not in hdr:
        from ..metadata import MissingWavelengthsError
        raise MissingWavelengthsError(f"{hdr_path.name}: no 'wavelength' field in ENVI header")
    wl = parse_number_list(hdr["wavelength"])
    if len(wl) != nb:
        raise InvalidHeaderError(f"{hdr_path.name}: {len(wl)} wavelengths but {nb} bands")
    fwhm = parse_number_list(hdr["fwhm"]) if "fwhm" in hdr else None
    bad = None
    if "bbl" in hdr:
        bbl = parse_number_list(hdr["bbl"])
        if len(bbl) == nb:
            bad = np.array(bbl) == 0          # ENVI bbl: 1 = good, 0 = bad
    spec = standardize_spectral_axis(wl, hdr.get("wavelength units"), fwhm, bad)

    # --- scaling / nodata
    nodata = float(hdr["data ignore value"]) if "data ignore value" in hdr else None
    div = reflectance_divisor or (float(hdr["reflectance scale factor"]) if "reflectance scale factor" in hdr else None)
    scale = 1.0 / div if div else 1.0
    notes = []
    if unit is None:
        if div:
            unit = "reflectance"
        else:
            # read a small centre window for inference
            r0, c0 = max(0, nl // 2 - 16), max(0, ns // 2 - 16)
            samp = _window(mm, interleave, r0, min(nl, r0 + 32), c0, min(ns, c0 + 32))
            samp = to_physical(samp, nodata, 1.0)
            unit, inferred = infer_unit(samp, dt.kind)
            notes.append(f"unit inferred as '{unit}' (no 'reflectance scale factor' in header); pass unit= to override")
    desc = hdr.get("description", "").lower()
    level = "radiance" if "radiance" in desc else ("reflectance" if unit == "reflectance" else None)

    def raw_loader(r0, r1, c0, c1):
        return to_physical(_window(mm, interleave, r0, r1, c0, c1), nodata, scale)

    transform = _transform(hdr)
    crs = hdr.get("coordinate system string", "").strip("{} ") or None
    cube = Cube.make(raw_loader, (nl, ns, nb), spec, unit, nodata=nodata, crs=crs, transform=transform,
                     sensor=hdr.get("sensor type"), processing_level=level,
                     source_path=str(path), source_format="envi",
                     metadata=dict(interleave=interleave, dtype=str(dt), notes=notes,
                                   envi_map_info=hdr.get("map info")))
    return cube


def _window(mm, interleave, r0, r1, c0, c1):
    if interleave == "bsq":
        return np.transpose(mm[:, r0:r1, c0:c1], (1, 2, 0))
    if interleave == "bil":
        return np.transpose(mm[r0:r1, :, c0:c1], (0, 2, 1))
    return mm[r0:r1, c0:c1, :]


def _transform(hdr):
    mi = hdr.get("map info")
    if not mi:
        return None
    try:
        p = [x.strip() for x in mi.strip("{} ").split(",")]
        ref_x, ref_y = float(p[1]), float(p[2])
        x, y, dx, dy = float(p[3]), float(p[4]), float(p[5]), float(p[6])
        return (dx, 0.0, x - (ref_x - 1) * dx, 0.0, -dy, y + (ref_y - 1) * dy)
    except Exception:
        return None
