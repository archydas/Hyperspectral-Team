"""Metadata normalisation: wavelengths, units, bad-band flags."""
from __future__ import annotations
import re
import numpy as np

# Default water-absorption windows (nm). These are FLAGGED, never deleted.
WATER_WINDOWS_NM = [(1350.0, 1450.0), (1800.0, 1950.0)]


class IngestError(Exception):
    """Base class for all ds3.ingest errors."""

class MissingWavelengthsError(IngestError):
    pass

class InvalidHeaderError(IngestError):
    pass

class UnsupportedFormatError(IngestError):
    pass


def to_nm(wl, units=None):
    """Return (wavelengths_nm, factor_applied, notes). Unit hint is respected;
    without a hint, values < 30 are assumed to be micrometres (and noted)."""
    wl = np.asarray(wl, dtype=float)
    u = (units or "").strip().lower()
    notes = []
    factor = 1.0
    if u.startswith("micro") or u in ("um", "µm", "micron", "microns"):
        factor = 1000.0
        notes.append("wavelengths converted from micrometres to nm (header units)")
    elif u.startswith("nano") or u == "nm":
        pass
    elif u:
        raise InvalidHeaderError(f"Unsupported wavelength units: {units!r}")
    elif np.nanmax(wl) < 30:
        factor = 1000.0
        notes.append("wavelength units missing; max < 30 so assumed micrometres and converted to nm")
    return wl * factor, factor, notes


def standardize_spectral_axis(wl_raw, units=None, fwhm=None, bad=None):
    """Convert to nm, sort ascending, align fwhm / bad-band flags.

    bad : bool array, True = bad band (or None).
    Returns dict(wavelengths_nm, order, fwhm_nm, bad_band_mask, notes).
    `order` is the index array that maps file band order -> sorted order
    (None if already ascending)."""
    wl, factor, notes = to_nm(wl_raw, units)
    if not np.all(np.isfinite(wl)):
        raise InvalidHeaderError("wavelengths contain NaN/inf")
    order = np.argsort(wl, kind="stable")
    already = np.array_equal(order, np.arange(len(wl)))
    if not already:
        if np.array_equal(order, np.arange(len(wl))[::-1]):
            notes.append("wavelengths were descending; bands reordered to ascending")
        else:
            notes.append("wavelengths were not monotonic (e.g. merged VNIR+SWIR); bands sorted ascending")
    wl_s = wl[order]
    if np.any(np.diff(wl_s) <= 0):
        raise InvalidHeaderError("duplicate wavelengths after sorting; cannot build a unique spectral axis")
    fwhm_s = None
    if fwhm is not None:
        f = np.asarray(fwhm, dtype=float) * factor
        if f.shape == wl.shape:
            fwhm_s = f[order].astype(np.float32)
        else:
            notes.append("fwhm length did not match wavelengths; fwhm dropped")
    bad_s = np.zeros(len(wl), dtype=bool) if bad is None else np.asarray(bad, dtype=bool)[order]
    return dict(wavelengths_nm=wl_s, order=None if already else order,
                fwhm_nm=fwhm_s, bad_band_mask=bad_s, notes=notes)


_NUM = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")
_UNIT = re.compile(r"(nm|nanometers?|µm|um|micrometers?|microns?)", re.I)


def parse_number_list(text):
    """'{450, 460.5, 470}' / '450,460' / '450 460' -> list[float]"""
    t = str(text).strip().strip("{}[]()")
    return [float(x) for x in _NUM.findall(t)]


def parse_wavelength_strings(strings):
    """Parse per-band descriptions such as '450 nm', '0.45 um', 'Band 3: 470.2nm'.
    Strings with no unit are accepted only if ALL bands parse, are >= 200 and
    strictly monotonic (otherwise 'Band_1' style names would be misread).
    Returns (values_nm, True) or (None, False)."""
    vals, with_unit = [], []
    for s in strings:
        if s is None:
            return None, False
        s = str(s)
        nums = _NUM.findall(s)
        if not nums:
            return None, False
        m = _UNIT.search(s)
        v = float(nums[-1])
        if m:
            u = m.group(1).lower()
            if u.startswith(("micro", "µ", "um")):
                v *= 1000.0
            with_unit.append(True)
        else:
            with_unit.append(False)
        vals.append(v)
    vals = np.array(vals)
    if all(with_unit):
        return vals, True
    if not any(with_unit) and np.all(vals >= 200) and np.all(np.diff(vals) > 0):
        return vals, True
    return None, False


def infer_unit(sample, kind):
    """Best-effort unit guess. Returns (unit, inferred_flag)."""
    if kind in ("i", "u"):
        return "dn", True
    v = np.asarray(sample, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return "unknown", True
    if np.percentile(v, 99) <= 2.0 and v.min() >= -0.5:
        return "reflectance", True
    return "unknown", True


def water_absorption_mask(wavelengths_nm, windows=None):
    wl = np.asarray(wavelengths_nm)
    m = np.zeros(wl.shape, dtype=bool)
    for lo, hi in (windows if windows is not None else WATER_WINDOWS_NM):
        m |= (wl >= lo) & (wl <= hi)
    return m
