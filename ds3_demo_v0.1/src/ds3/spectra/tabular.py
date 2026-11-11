"""Tolerant reader for plain-text spectral libraries (.txt / .csv / .tsv / .dat) whose layout is not known in advance.

WHY: public and in-house libraries ship as text tables in many layouts. Instead of one reader per file, this one
inspects the file and recognises these layouts (the choice is written into every spectrum's metadata['layout']):

  rows_by_spectrum   header = wavelengths ("350", "X350", "350nm", "0.35"), one row per spectrum; any text
                     columns before them (name, class, sample id, ...) are kept as metadata
  wavelength_column  first column = wavelength, every other column = one spectrum (header = names)
  long               columns material/name + wavelength + value, many rows per material
  headerless         all numbers: first column = wavelength, others = spectra

Delimiters (comma, tab, semicolon, pipe, whitespace), '#' comment lines, quotes, BOM, percent reflectance and
micrometres are handled. Nodata sentinels (<= -1e30, -9999, 'NA', 'NaN') become NaN.

STATUS: tested on synthetic files written in each layout above, NOT on your real files yet. If a file is refused the
message says what was seen; run `python scripts/inspect_file.py <file>` and send the output.
"""
from __future__ import annotations
import csv, io, re
from pathlib import Path
import numpy as np
from ..ingest.contract import Spectrum
from ..ingest.metadata import InvalidHeaderError, MissingWavelengthsError, to_nm
from .categories import guess_category

NAME_KEYS = ("material", "name", "sample_name", "samplename", "sample", "spectrum", "spectrum_name", "id", "label",
             "mineral", "title", "class", "specimen")
CATEGORY_KEYS = ("category", "type", "group", "class", "material_class", "materialtype", "genericclass", "classification")
VALUE_KEYS = ("reflectance", "value", "values", "radiance", "intensity", "refl", "r", "signal", "dn")
WL_KEYS = ("wavelength", "wavelengths", "wl", "wvl", "lambda", "wave")
_NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
_WL_CELL = re.compile(rf"^(?:[A-Za-z_]{{0,6}}[_\s:=]?)?({_NUM})\s*(nm|um|µm|micron|microns|micrometers?)?$", re.I)
_SENTINEL = re.compile(r"^(na|n/a|nan|null|none|-|--|)$", re.I)


def _split_lines(text):
    out = []
    for ln in text.splitlines():
        s = ln.strip().lstrip("﻿")
        if not s or s[0] in "#%" or s.startswith("//"):
            continue
        out.append(s)
    return out


def _delimiter(lines):
    sample = lines[:30]
    best, score = None, 0
    for d in (",", "\t", ";", "|"):
        counts = [ln.count(d) for ln in sample]
        if min(counts) > 0 and len(set(counts)) <= max(2, len(sample) // 4):
            sc = min(counts)
            if sc > score:
                best, score = d, sc
    return best


def _rows(lines, delim):
    if delim:
        return [[c.strip().strip('"').strip("'") for c in r] for r in csv.reader(io.StringIO("\n".join(lines)), delimiter=delim)]
    return [ln.split() for ln in lines]


def _f(x):
    try:
        v = float(str(x).strip())
    except (TypeError, ValueError):
        return np.nan
    return np.nan if (not np.isfinite(v) or v <= -1e30 or v == -9999 or v == -9999.0) else v


def _wl_cell(c):
    m = _WL_CELL.match(str(c).strip())
    return (float(m.group(1)), (m.group(2) or "").lower()) if m else None


def _unit_of(suffix):
    return "micrometers" if suffix.startswith(("um", "µm", "micro")) else ("nm" if suffix == "nm" else None)


def _pick(headers, keys):
    low = [h.strip().lower().replace(" ", "_") for h in headers]
    for k in keys:
        if k in low:
            return low.index(k)
    return None


def _spectrum(name, wl, vals, unit, notes, src, layout, meta, category):
    wl = np.asarray(wl, float); vals = np.asarray(vals, float)
    ok = np.isfinite(wl)
    wl, vals = wl[ok], vals[ok]
    o = np.argsort(wl, kind="stable")
    wl, vals = wl[o], vals[o]
    keep = np.concatenate([[True], np.diff(wl) > 0])          # drop repeated wavelengths
    wl, vals = wl[keep], vals[keep]
    if len(wl) < 3 or not np.isfinite(vals).sum() >= 3:
        return None
    notes = list(notes)
    if unit == "reflectance" and np.nanmax(vals) > 1.5:
        vals = vals / 100.0
        notes.append("values looked like percent; divided by 100")
    m = {k: v for k, v in meta.items() if v not in (None, "")}
    m.update(source_file=src, notes=notes, layout=layout)
    return Spectrum(material=str(name) or "unnamed", wavelengths_nm=wl, values=vals, unit=unit,
                    purity="ground_measured", category=category or guess_category(str(name)), metadata=m)


def describe_layout(path, max_lines=400):
    """Return a short, human-readable diagnosis of a text table (used by scripts/inspect_file.py and error messages)."""
    text = Path(path).read_text(errors="replace")
    lines = _split_lines(text[:600_000])[:max_lines]
    if not lines:
        return "file is empty"
    d = _delimiter(lines)
    rows = _rows(lines, d)
    hdr = rows[0]
    nwl = sum(1 for c in hdr if _wl_cell(c))
    return (f"{len(lines)}+ non-empty lines; delimiter={'whitespace' if d is None else repr(d)}; first row has {len(hdr)} cells, "
            f"{nwl} look like wavelengths; first cells: {hdr[:6]}; second row cells: {len(rows[1]) if len(rows) > 1 else 0}, "
            f"first: {rows[1][:6] if len(rows) > 1 else []}")


def read_tabular(path, unit="reflectance", wavelength_units=None, **_):
    p = Path(path)
    text = p.read_text(encoding="utf-8-sig", errors="replace")
    lines = _split_lines(text)
    if len(lines) < 3:
        raise InvalidHeaderError(f"{p.name}: fewer than 3 data lines")
    d = _delimiter(lines)
    rows = _rows(lines, d)
    hdr = rows[0]
    ncol = len(hdr)
    src = p.name
    unit = unit or "reflectance"

    # ---- layout 1: header holds wavelengths ------------------------------------------------------------------
    cells = [_wl_cell(c) for c in hdr]
    wl_idx = [i for i, c in enumerate(cells) if c is not None]
    band_numbers, band_msg = False, None
    if len(wl_idx) >= 8 and len(wl_idx) >= 0.6 * ncol:
        first = wl_idx[0]
        # trailing non-wavelength columns are metadata too
        idx = [i for i in wl_idx if i >= first]
        wl_raw = np.array([cells[i][0] for i in idx])
        band_numbers = np.allclose(wl_raw, np.arange(1, len(wl_raw) + 1)) or (wl_raw.max() <= len(wl_raw) + 1 and np.all(np.diff(wl_raw) == 1))
        if band_numbers:           # headers like soil1, soil2, ... are sample names, not wavelengths: try the other layouts first
            band_msg = (f"{p.name}: the column headers look like band numbers 1..{len(wl_raw)}, not wavelengths. "
                        "Provide a wavelength list/range.")
            idx = []
    if len(wl_idx) >= 8 and len(wl_idx) >= 0.6 * ncol and not band_numbers:
        sfx = next((c[1] for c in cells if c and c[1]), "")
        wl, _, notes = to_nm(wl_raw, wavelength_units or _unit_of(sfx))
        meta_cols = [i for i in range(ncol) if i not in set(idx)]
        mh = [hdr[i] for i in meta_cols]
        name_i = _pick(mh, NAME_KEYS)
        cat_i = _pick(mh, CATEGORY_KEYS)
        out = []
        for r in rows[1:]:
            if len(r) < first + 3:
                continue
            vals = [_f(r[i]) if i < len(r) else np.nan for i in idx]
            meta = {mh[k].strip(): r[i] for k, i in enumerate(meta_cols) if i < len(r) and r[i] not in ("",)}
            name = (r[meta_cols[name_i]] if name_i is not None and meta_cols[name_i] < len(r) else
                    (r[meta_cols[0]] if meta_cols and meta_cols[0] < len(r) else f"row{len(out) + 1}"))
            cat = r[meta_cols[cat_i]] if cat_i is not None and meta_cols[cat_i] < len(r) else None
            s = _spectrum(name or f"row{len(out) + 1}", wl, vals, unit, notes, src, "rows_by_spectrum", meta, cat or None)
            if s is not None:
                out.append(s)
        if out:
            return _dedupe(out)

    # ---- layout 2: long (material, wavelength, value) ---------------------------------------------------------
    ih = [c.strip().lower().replace(" ", "_") for c in hdr]
    wi = next((i for i, h in enumerate(ih) if any(h == k or h.startswith(k + "_") or h.startswith(k + "(") for k in WL_KEYS)), None)
    vi = next((i for i, h in enumerate(ih) if i != wi and h in VALUE_KEYS), None)
    ni = next((i for i, h in enumerate(ih) if i not in (wi, vi) and h in NAME_KEYS), None)
    if wi is not None and vi is not None and ni is not None:
        groups = {}
        for r in rows[1:]:
            if len(r) <= max(wi, vi, ni):
                continue
            groups.setdefault(r[ni], ([], []))
            groups[r[ni]][0].append(_f(r[wi])); groups[r[ni]][1].append(_f(r[vi]))
        sfx = _wl_cell(hdr[wi].split("(")[-1].strip(") ")) if "(" in hdr[wi] else None
        out = []
        for name, (w, v) in groups.items():
            wl, _, notes = to_nm(np.asarray(w, float), wavelength_units or (_unit_of(sfx[1]) if sfx else None))
            s = _spectrum(name, wl, v, unit, notes, src, "long", {}, None)
            if s is not None:
                out.append(s)
        if out:
            return _dedupe(out)

    # ---- layout 3: first column wavelength, others spectra (with or without a header) -------------------------------------
    body = [r for r in rows if r and not np.isnan(_f(r[0]))]
    has_header = np.isnan(_f(hdr[0]))
    if len(body) >= 8 and ncol >= 2:
        w = np.array([_f(r[0]) for r in body])
        mono = np.all(np.diff(w) > 0) or np.all(np.diff(w) < 0)
        if mono:
            names = [h.strip() or f"col{j}" for j, h in enumerate(hdr)] if has_header else [f"spectrum{j}" for j in range(ncol)]
            sfx = _wl_cell(hdr[0].split("(")[-1].strip(") ")) if has_header and "(" in hdr[0] else None
            wl, _, notes = to_nm(w, wavelength_units or (_unit_of(sfx[1]) if sfx else None))
            out = []
            for j in range(1, ncol):
                s = _spectrum(names[j], wl, [_f(r[j]) if j < len(r) else np.nan for r in body], unit, notes, src,
                              "wavelength_column" if has_header else "headerless", {}, None)
                if s is not None:
                    out.append(s)
            if out:
                return _dedupe(out)

    if band_msg:
        raise MissingWavelengthsError(band_msg)
    raise InvalidHeaderError(f"{p.name}: could not recognise the table layout ({describe_layout(p)}). "
                             "Expected wavelengths as column headers, or a wavelength column, or material/wavelength/value columns.")


def _dedupe(spectra):
    seen = {}
    for s in spectra:
        n = seen.get(s.material, 0)
        seen[s.material] = n + 1
        if n:
            s.material = f"{s.material} ({n + 1})"
    return spectra
