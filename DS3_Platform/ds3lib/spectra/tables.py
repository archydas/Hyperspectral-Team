"""Spectra stored as Excel (.xlsx), JSON or Parquet. Excel and Parquet are turned into a text table and read with the
tolerant table reader (spectra/tabular.py); JSON is read directly (also DS3 aggregation exports)."""
from __future__ import annotations
import csv, io, json, tempfile
from pathlib import Path
import numpy as np
from ..ingest.contract import Spectrum, UNITS, PURITY
from ..ingest.metadata import IngestError, UnsupportedFormatError, to_nm


def _need(mod, hint):
    try:
        return __import__(mod)
    except ImportError as e:
        raise UnsupportedFormatError(f"reading this format needs the optional package '{mod}': pip install {hint}") from e


def _via_csv(rows, name, **kw):
    from .tabular import read_tabular
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / f"{name}.csv"
        with open(p, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerows(rows)
        return read_tabular(p, **kw)


def read_excel(path, **kw):
    """Every sheet that holds a spectra table is read; sheet names are noted in the metadata."""
    openpyxl = _need("openpyxl", "openpyxl")
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    out, errors = [], []
    for ws in wb.worksheets:
        rows = [["" if c is None else c for c in r] for r in ws.iter_rows(values_only=True)]
        rows = [r for r in rows if any(str(c).strip() for c in r)]
        if len(rows) < 3:
            continue
        try:
            for s in _via_csv(rows, ws.title, **kw):
                s.metadata = {**(s.metadata or {}), "sheet": ws.title}
                out.append(s)
        except (IngestError, ValueError) as e:
            errors.append(f"{ws.title}: {e}")
    wb.close()
    if not out:
        raise IngestError(f"{Path(path).name}: no sheet with a readable spectra table" + (f" ({'; '.join(errors)})" if errors else ""))
    return out


def read_parquet(path, **kw):
    pa = _need("pyarrow.parquet", "pyarrow")
    tbl = pa.read_table(path)
    cols = tbl.column_names
    rows = [cols] + [list(r.values()) for r in tbl.to_pylist()]
    return _via_csv(rows, Path(path).stem, **kw)


def _mk(name, wl, vals, unit, src, meta, purity="ground_measured"):
    wl = np.asarray(wl, float)
    v = np.asarray([np.nan if x is None else x for x in vals], float)
    if wl.size != v.size or wl.size < 2:
        raise IngestError(f"{src}: '{name}' has {v.size} values for {wl.size} wavelengths")
    if np.nanmax(wl) < 30:                                  # micrometres
        wl = wl * 1000.0
    order = np.argsort(wl)
    unit = unit if unit in UNITS else ("reflectance" if not unit else "unknown")
    purity = purity if purity in PURITY else "ground_measured"
    return Spectrum(material=str(name), wavelengths_nm=wl[order], values=v[order], unit=unit, purity=purity,
                    metadata={"source_file": src, **meta})


def read_json(path, unit="reflectance", **_):
    """Accepted shapes:
       {"wavelengths": [...], "spectra": [{"name":..., "values":[...]}]}    (also key "classes" with "mean": DS3 export)
       [{"name":..., "wavelengths":[...], "values":[...]}, ...]"""
    d = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    src, out = Path(path).name, []
    unit = (d.get("unit") if isinstance(d, dict) else None) or unit
    if isinstance(d, dict):
        wl = d.get("wavelengths") or d.get("wavelength_nm")
        items = d.get("spectra") or d.get("classes") or []
        for it in items:
            vals = it.get("values") or it.get("mean")
            meta = {k: it[k] for k in ("n_pixels",) if k in it}
            if "std" in it:
                meta["std"] = it["std"]
            out.append(_mk(it.get("name", f"spectrum_{len(out) + 1}"), it.get("wavelengths") or wl, vals, unit, src, meta,
                           purity=it.get("purity") or ("image_pixel" if "mean" in it else "ground_measured")))
    elif isinstance(d, list):
        for i, it in enumerate(d, 1):
            out.append(_mk(it.get("name", f"spectrum_{i}"), it["wavelengths"], it["values"], it.get("unit") or unit, src, {}, purity=it.get("purity") or "ground_measured"))
    if not out:
        raise IngestError(f"{src}: no spectra found in this JSON file")
    return out
