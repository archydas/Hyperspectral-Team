"""Wavelengths from a side file: EnMAP METADATA.XML, or a plain .txt / .csv list.

WHY: many products keep the band centres OUTSIDE the image file (EnMAP: XML next to the GeoTIFF).
`read_wavelengths_file` returns (wavelengths_nm, fwhm_nm_or_None, notes).

STATUS: the XML parser is tolerant (it looks for any element whose tag contains 'wavelength' + 'center'
and, optionally, 'fwhm', in document order, or ordered by a bandID number attribute). It has been tested on a
SYNTHETIC XML written from the published EnMAP layout, NOT on a real EnMAP file yet.
"""
from __future__ import annotations
import re
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from ..metadata import MissingWavelengthsError, InvalidHeaderError, to_nm


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].lower()


def _num(text):
    try:
        return float(str(text).strip().split()[0])
    except (ValueError, IndexError):
        return None


def parse_wavelength_xml(path):
    """(wavelengths, fwhm|None, units|None) from an EnMAP-style metadata XML."""
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as e:
        raise InvalidHeaderError(f"{Path(path).name}: not valid XML ({e})") from e
    found = []          # (band_number or None, wavelength, fwhm)
    for el in root.iter():
        kids = list(el)
        if not kids:
            continue
        wl = fw = None
        for k in kids:
            t = _local(k.tag)
            if "wavelength" in t and "center" in t and wl is None:
                wl = _num(k.text)
            elif t in ("fwhmofband", "fwhm", "bandwidth", "fwhm_nm") and fw is None:
                fw = _num(k.text)
        if wl is not None:
            num = el.attrib.get("number") or el.attrib.get("id") or el.attrib.get("band")
            found.append((int(num) if num and str(num).isdigit() else None, wl, fw))
    if not found:
        raise MissingWavelengthsError(
            f"{Path(path).name}: no element with 'wavelength' + 'center' in its name was found. "
            "Open the XML, find the band-centre tag, and send it so the adapter can be extended.")
    if all(n is not None for n, _, _ in found):
        found.sort(key=lambda t: t[0])
    wl = np.array([w for _, w, _ in found], float)
    fw = np.array([f for _, _, f in found], float) if all(f is not None for _, _, f in found) else None
    return wl, fw, None


def read_wavelengths_file(path, units=None):
    """.xml -> parse_wavelength_xml; .txt/.csv -> numbers (col 1 = wavelength, optional col 2 = FWHM).
    Returns (wavelengths_nm, fwhm_nm|None, notes). Micrometres are converted to nm."""
    p = Path(path)
    ext = p.suffix.lower()
    if ext == ".xml":
        wl, fw, u = parse_wavelength_xml(p)
        wl, f, notes = to_nm(wl, units or u)
        return wl, (None if fw is None else fw * f), [f"wavelengths read from {p.name}"] + notes
    rows = []
    for line in p.read_text(errors="replace").splitlines():
        parts = re.split(r"[,\s;]+", line.strip())
        try:
            rows.append([float(x) for x in parts[:2] if x != ""])
        except ValueError:
            continue
    rows = [r for r in rows if r]
    if len(rows) < 2:
        raise MissingWavelengthsError(f"{p.name}: no numeric wavelength rows found")
    width = min(len(r) for r in rows)
    a = np.array([r[:width] for r in rows])
    wl, f, notes = to_nm(a[:, 0], units)
    fw = a[:, 1] * f if width > 1 else None
    return wl, fw, [f"wavelengths read from {p.name}"] + notes


def find_sibling_xml(tif_path):
    """EnMAP: '<prefix>-SPECTRAL_IMAGE.TIF' <-> '<prefix>-METADATA.XML' in the same folder.
    Falls back to the only '*METADATA.XML' in that folder. Returns Path or None."""
    p = Path(tif_path)
    for old in ("SPECTRAL_IMAGE", "spectral_image"):
        if old in p.name:
            cand = p.with_name(p.name.replace(old, "METADATA")).with_suffix(".XML")
            for c in (cand, cand.with_suffix(".xml")):
                if c.is_file():
                    return c
    others = [f for f in p.parent.iterdir() if f.is_file() and f.name.lower().endswith("metadata.xml")]
    return others[0] if len(others) == 1 else None
