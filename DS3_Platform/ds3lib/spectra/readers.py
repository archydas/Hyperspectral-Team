"""Readers for point spectra (field instruments and public libraries).

Every reader returns list[Spectrum] with wavelengths in nm (ascending), values as float64,
reflectance as FRACTION 0-1 (percent is converted and noted in metadata).
Supported: csv, svc (.sig), asd/sed (via specdal, optional), envi_sli (.sli+.hdr),
usgs (splib07 ASCII, needs wavelengths file), ecostress (.spectrum.txt).
"""
from __future__ import annotations
import csv, re
from pathlib import Path
import numpy as np
from ..ingest.contract import Spectrum
from ..ingest.metadata import (IngestError, InvalidHeaderError, UnsupportedFormatError,
                               MissingWavelengthsError, to_nm, parse_number_list)
from ..ingest.readers.envi import parse_envi_header

SPECTRA_FORMATS = {
    "csv": "CSV: wavelength column + spectra columns, or one spectrum per row",
    "svc": "Spectra Vista (SVC) .sig field spectroradiometer",
    "asd": "ASD FieldSpec binary (.asd) - needs `pip install specdal`",
    "sed": "Spectral Evolution .sed - needs `pip install specdal`",
    "envi_sli": "ENVI spectral library (.sli + .hdr)",
    "usgs": "USGS splib07 ASCII (needs wavelength file via wavelengths_file=)",
    "ecostress": "ECOSTRESS / JPL spectral library .spectrum.txt",
    "excel": "Excel workbook (.xlsx): every sheet with a spectra table (needs `pip install openpyxl`)",
    "json": "JSON: wavelengths + list of spectra, or a DS3 aggregation export",
    "parquet": "Parquet table (wide or long; needs `pip install pyarrow`)",
    "tabular": "any text table (.txt/.csv/.tsv/.dat): layout detected automatically (see spectra/tabular.py)",
}


def detect_spectra_format(path):
    p = Path(path)
    name, ext = p.name.lower(), p.suffix.lower()
    if ext == ".sig": return "svc"
    if ext == ".asd": return "asd"
    if ext == ".sed": return "sed"
    if ext == ".sli": return "envi_sli"
    if ext == ".mat":
        raise UnsupportedFormatError(f"{p.name}: MATLAB .mat files are images (cubes), not spectra - they are imported as images")
    if name.endswith(".spectrum.txt"): return "ecostress"
    if ext == ".hdr":
        try:
            h = parse_envi_header(p)
            if "spectral library" in h.get("file type", "").lower(): return "envi_sli"
        except IngestError:
            pass
        return None
    if ext in (".xlsx", ".xlsm"): return "excel"
    if ext == ".json": return "json"
    if ext == ".parquet": return "parquet"
    if ext == ".csv": return "csv"
    if ext in (".txt", ".tsv", ".dat", ".asc"):
        head = p.read_text(errors="replace")[:4000]
        if re.search(r"^\s*Name:\s*.+", head, re.M) and re.search(r"X Units|Y Units", head): return "ecostress"
        # USGS splib07 value files are ONE number per line (title line first) and need a wavelength list beside them
        lines = [l.split() for l in head.splitlines()[1:40] if l.strip()]
        if lines and all(len(t) == 1 for t in lines) and sorted(p.parent.glob("*avelength*")):
            return "usgs"
        return "tabular"
    return None


def _scale_refl(vals, unit, notes):
    """percent (max > 1.5) -> fraction. Only applied to reflectance."""
    if unit == "reflectance" and np.nanmax(vals) > 1.5:
        notes.append("values looked like percent; divided by 100")
        return vals / 100.0
    return vals


def _finish(wl, vals, **kw):
    wl = np.asarray(wl, float); vals = np.asarray(vals, float)
    if len(wl) != len(vals): raise InvalidHeaderError("wavelength / value length mismatch")
    o = np.argsort(wl)
    return wl[o], vals[o]


def _mk(material, wl, vals, unit, notes, source, **meta):
    wl, vals = _finish(wl, vals)
    vals = _scale_refl(vals, unit, notes)
    return Spectrum(material=material, wavelengths_nm=wl, values=vals, unit=unit,
                    purity="ground_measured", metadata=dict(source_file=source, notes=notes, **meta))


# ---------------- CSV ----------------
def read_csv(path, unit="reflectance", wavelength_units=None, **_):
    rows = list(csv.reader(Path(path).read_text(errors="replace").splitlines()))
    rows = [r for r in rows if r and any(c.strip() for c in r)]
    if len(rows) < 2: raise InvalidHeaderError(f"{path}: CSV too short")
    hdr = rows[0]
    def isnum(s):
        try: float(s); return True
        except ValueError: return False
    out, src = [], Path(path).name
    hdr_numeric = [isnum(c) for c in hdr[1:]]
    if hdr_numeric and all(hdr_numeric):        # row-wise: header = wavelengths
        wl, f, notes = to_nm([float(c) for c in hdr[1:]], wavelength_units)
        for r in rows[1:]:
            v = np.array([float(x) if isnum(x) else np.nan for x in r[1:1 + len(wl)]])
            out.append(_mk(r[0], wl, v, unit, list(notes), src))
    else:                                         # wide: col0 = wavelength, others = spectra
        body = rows[1:]
        wl_raw = [float(r[0]) for r in body if isnum(r[0])]
        wl, f, notes = to_nm(wl_raw, wavelength_units)
        for j, name in enumerate(hdr[1:], start=1):
            v = np.array([float(r[j]) if j < len(r) and isnum(r[j]) else np.nan for r in body if isnum(r[0])])
            out.append(_mk(name.strip() or f"col{j}", wl, v, unit, list(notes), src))
    return out


# ---------------- SVC .sig ----------------
def read_svc(path, **_):
    lines = Path(path).read_text(errors="replace").splitlines()
    meta, i = {}, 0
    while i < len(lines) and not lines[i].lower().startswith("data="):
        if "=" in lines[i]:
            k, v = lines[i].split("=", 1); meta[k.strip().lower()] = v.strip()
        i += 1
    if i >= len(lines): raise InvalidHeaderError(f"{path}: no 'data=' marker; not an SVC .sig file")
    data = np.array([[float(x) for x in l.split()[:4]] for l in lines[i + 1:] if len(l.split()) >= 4])
    if data.size == 0: raise InvalidHeaderError(f"{path}: no data rows")
    wl = data[:, 0]; refl = data[:, 3]
    notes = ["SVC: using 4th column (reflectance %) of [wavelength, ref, target, reflectance]"]
    name = meta.get("name", Path(path).stem)
    return [_mk(name, wl, refl, "reflectance", notes, Path(path).name,
                instrument=meta.get("instrument"), time=meta.get("time"))]


# ---------------- ASD / SED (optional specdal) ----------------
def read_asd_sed(path, **_):
    try:
        from specdal.containers.spectrum import Spectrum as SD
    except ImportError as e:
        raise IngestError("ASD/SED needs specdal: pip install specdal") from e
    s = SD(filepath=str(path))
    meas = s.measurement
    return [_mk(Path(path).stem, meas.index.values.astype(float), meas.values.astype(float),
                "reflectance", ["read via specdal"], Path(path).name)]


# ---------------- ENVI .sli ----------------
def read_envi_sli(path, unit="reflectance", **_):
    from ..ingest.readers.envi import DTYPES
    p = Path(path)
    hp = p if p.suffix.lower() == ".hdr" else next((h for h in (Path(str(p) + ".hdr"), p.with_suffix(".hdr")) if h.is_file()), None)
    if hp is None: raise InvalidHeaderError(f"no .hdr for {path}")
    h = parse_envi_header(hp)
    dp = p if p.suffix.lower() != ".hdr" else next((c for c in (p.with_suffix(".sli"), p.with_suffix("")) if c.is_file()), None)
    if dp is None: raise InvalidHeaderError(f"no .sli data next to {hp}")
    ns, nl = int(float(h["samples"])), int(float(h["lines"]))
    dt = np.dtype(("<" if int(float(h.get("byte order", 0))) == 0 else ">") + DTYPES[int(float(h["data type"]))])
    arr = np.fromfile(dp, dtype=dt, count=ns * nl).reshape(nl, ns).astype(float)
    if "wavelength" not in h: raise MissingWavelengthsError(f"{hp.name}: no wavelength field")
    wl, f, notes = to_nm(parse_number_list(h["wavelength"]), h.get("wavelength units"))
    names = [n.strip() for n in h.get("spectra names", "").strip("{} ").split(",") if n.strip()]
    if len(names) != nl: names = [f"{p.stem}_{i}" for i in range(nl)]
    scale = float(h.get("reflectance scale factor", 1) or 1)
    ign = h.get("data ignore value")
    out = []
    for n, row in zip(names, arr):
        if ign is not None: row = np.where(row == float(ign), np.nan, row)
        row = np.where(np.isfinite(row) & (row > -1e30), row, np.nan) / (scale if scale > 1 else 1)
        out.append(_mk(n, wl, row, unit, list(notes), p.name))
    return out


# ---------------- USGS splib07 ASCII ----------------
USGS_DELETED = -1.23e34

def read_usgs(path, wavelengths_file=None, unit="reflectance", wavelength_units="micrometers", **_):
    if wavelengths_file is None:
        raise MissingWavelengthsError("USGS splib07 ASCII has no wavelengths inside; pass wavelengths_file= "
                                      "(the splib07a_Wavelengths_*.txt file from the same download)")
    wl_lines = Path(wavelengths_file).read_text(errors="replace").splitlines()[1:]
    wl = np.array([float(l) for l in wl_lines if l.strip()])
    lines = Path(path).read_text(errors="replace").splitlines()
    title = lines[0].strip()
    v = np.array([float(l) for l in lines[1:] if l.strip()])
    if len(v) != len(wl): raise InvalidHeaderError(f"{Path(path).name}: {len(v)} values vs {len(wl)} wavelengths")
    v[v <= USGS_DELETED * 0.5] = np.nan
    wl_nm, f, notes = to_nm(wl, wavelength_units)
    notes.append("USGS deleted points (-1.23e34) -> NaN")
    mat = re.sub(r"\s+(splib\d+\w*|Record=\d+.*)$", "", title).strip() or Path(path).stem
    return [_mk(mat, wl_nm, v, unit, notes, Path(path).name, title=title)]


# ---------------- ECOSTRESS ----------------
def read_ecostress(path, **_):
    lines = Path(path).read_text(errors="replace").splitlines()
    meta, data = {}, []
    for l in lines:
        m = re.match(r"^\s*([A-Za-z][^:]*):\s*(.*)$", l)
        parts = l.split()
        if len(parts) == 2 and not m:
            try: data.append((float(parts[0]), float(parts[1]))); continue
            except ValueError: pass
        if m: meta[m.group(1).strip().lower()] = m.group(2).strip()
    if not data: raise InvalidHeaderError(f"{path}: no numeric data rows")
    a = np.array(data)
    xu = meta.get("x units", "micrometers").lower()
    yu = meta.get("y units", "reflectance (percent)").lower()
    wl, f, notes = to_nm(a[:, 0], "micrometers" if "micro" in xu else ("nm" if "nano" in xu else None))
    unit = "reflectance" if "reflect" in yu else "unknown"
    name = meta.get("name") or meta.get("material") or Path(path).stem
    s = _mk(name, wl, a[:, 1], unit, notes, Path(path).name,
            **{k: meta[k] for k in ("type", "class", "subclass", "genus", "species", "sample no.") if k in meta})
    s.category = meta.get("class") or meta.get("type")
    return [s]


def read_tabular_any(path, **kw):
    from .tabular import read_tabular
    return read_tabular(path, **kw)


_READERS = dict(csv=read_csv, svc=read_svc, asd=read_asd_sed, sed=read_asd_sed,
                envi_sli=read_envi_sli, usgs=read_usgs, ecostress=read_ecostress, tabular=read_tabular_any,
                excel=lambda p, **kw: _tables().read_excel(p, **kw), json=lambda p, **kw: _tables().read_json(p, **kw),
                parquet=lambda p, **kw: _tables().read_parquet(p, **kw))


def _tables():
    from . import tables
    return tables


def read_spectra(path, fmt=None, **kw):
    fmt = fmt or detect_spectra_format(path)
    if fmt is None: raise UnsupportedFormatError(f"cannot determine spectra format of {path}")
    if fmt == "csv":                       # the simple CSV reader first; the tolerant table reader if it does not fit
        try:
            return _READERS["csv"](path, **kw)
        except (IngestError, ValueError, IndexError):
            return read_tabular_any(path, **{k: v for k, v in kw.items() if k in ("unit", "wavelength_units")})
    return _READERS[fmt](path, **kw)
