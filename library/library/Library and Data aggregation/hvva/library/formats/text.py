"""Tabular and text readers: CSV / TSV / TXT / DAT (any layout), Excel, generic Parquet, JSON.

One table engine handles every layout:

  columns  wavelength column + one column per spectrum (wide CSV, HyperVista export with *_mean/*_std, ECOSTRESS
           "key: value" header + two columns, plain two-column text, SVC/SED-like files with a reflectance column)
  rows     one spectrum per row, wavelengths in the header row, optional metadata columns
           (name, category/class/type, acquisition/platform, sample, description ...)
  long     tidy format: name, wavelength, value (+ optional category / acquisition) per line
"""
from __future__ import annotations

import csv
import io
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from .base import (ReadOptions, Reader, SpectrumRecord, register, safe_float, stem_name, unit_from_text)

TEXT_EXT = (".csv", ".tsv", ".txt", ".dat", ".asc", ".prn", ".tab", ".spc.txt")
_WL_COL = re.compile(r"^\s*(wave\s*length|wavelength|wl|wvl|lambda|λ|nm|um|µm|micron|band\s*center|"
                     r"x\b|wavenumber|wave\s*number|cm-?1)", re.I)
_NAME_COL = re.compile(r"^(name|material|material\s*name|sample|sample\s*name|sample\s*id|id|label|spectrum|"
                       r"spectrum\s*name|target|object|class\s*name|endmember)$", re.I)
_CAT_COL = re.compile(r"^(category|class|type|group|material\s*type|material\s*class|cover|land\s*cover|"
                      r"category\s*path|taxonomy)$", re.I)
_ACQ_COL = re.compile(r"^(acquisition|acquisition\s*type|platform|measurement|measurement\s*type|"
                      r"setting|source\s*type|level|sensor\s*platform)$", re.I)
_VAL_COL = re.compile(r"^(value|values|reflectance|refl|r|y|intensity|signal|transmittance|emissivity|"
                      r"absorbance|radiance)$", re.I)
_REFL_COL = re.compile(r"reflect|refl\b|tgt\.?\s*/\s*ref|%|^r$|^y$|emissiv|transmit|absorb", re.I)
_RAW_COL = re.compile(r"\brad\b|radiance|irrad|counts?\b|\bdn\b|digital|raw|\(ref|ref\.\)|reference|"
                      r"white|dark|chan|channel", re.I)
_META_KV = re.compile(r"^\s*[#%!;]*\s*([A-Za-z][\w .()/\-\[\]°µ]*?)\s*[:=]\s*(.*?)\s*$")


# ================================================================== table engine
def _decode(raw: bytes) -> str:
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff") or (raw[:200].count(b"\x00") > 20):
        try:
            return raw.decode("utf-16")
        except UnicodeDecodeError:
            pass
    for enc in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1", errors="replace")


def _split(line: str, delim: Optional[str]) -> List[str]:
    if delim is None:
        return line.split()
    if delim == ",":
        try:
            return next(csv.reader([line]))
        except StopIteration:
            return []
    return [c.strip().strip('"') for c in line.split(delim)]


def _num(cell, decimal_comma=False) -> float:
    if isinstance(cell, (int, float, np.floating, np.integer)) and not isinstance(cell, bool):
        return float(cell)
    s = str(cell).strip().strip('"').replace("−", "-")
    if not s:
        return np.nan
    if decimal_comma:
        s = s.replace(".", "").replace(",", ".") if s.count(",") == 1 and s.count(".") >= 1 else s.replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return np.nan


def _is_num(cell, dc=False) -> bool:
    return np.isfinite(_num(cell, dc))


def find_block(lines: List[str]) -> Tuple[Optional[str], bool, int, int]:
    """(delimiter, decimal_comma, start, end) of the longest numeric table in *lines*."""
    best = (None, False, 0, 0, 0.0)
    for delim in ("\t", ",", ";", "|", None):
        for dc in ((False, True) if delim in ("\t", ";", "|", None) else (False,)):
            run_start, run_len, run_n = 0, 0, None
            for i, ln in enumerate(lines + [""]):
                cells = _split(ln, delim) if ln.strip() else []
                cells = [c for c in cells] if delim is not None else cells
                ok = len(cells) >= 2 and sum(_is_num(c, dc) for c in cells) >= max(1, len(cells) * 0.5)
                if ok and (run_n is None or len(cells) == run_n or (delim is not None and abs(len(cells) - run_n) <= 1)):
                    if run_len == 0:
                        run_start = i
                    run_len += 1
                    run_n = run_n or len(cells)
                else:
                    if run_len:
                        score = run_len * (run_n or 0)
                        if score > best[4]:
                            best = (delim, dc, run_start, run_start + run_len, score)
                    run_len, run_n = 0, None
                    if ok:
                        run_start, run_len, run_n = i, 1, len(cells)
    return best[0], best[1], best[2], best[3]


def parse_text_table(text: str):
    """-> (meta dict, header list|None, rows list[list[str]], decimal_comma)."""
    lines = text.splitlines()
    delim, dc, a, b = find_block(lines)
    if b - a < 1:
        return {}, None, [], False
    header = None
    meta: Dict[str, str] = {}
    hdr_i = a - 1
    while hdr_i >= 0 and not lines[hdr_i].strip():
        hdr_i -= 1
    if hdr_i >= 0:
        cand = _split(lines[hdr_i], delim)
        ncols = len(_split(lines[a], delim))
        if len(cand) >= 2 and abs(len(cand) - ncols) <= 1 and not lines[hdr_i].lstrip().startswith(("#", "%", "!")):
            header = [c.strip() for c in cand]
            if len(header) == ncols - 1:
                header = ["wavelength"] + header if not _WL_COL.match(header[0]) else header + [""]
        else:
            hdr_i += 1
    for ln in lines[:max(hdr_i, 0)]:
        m = _META_KV.match(ln)
        if m and m.group(2):
            meta.setdefault(m.group(1).strip().lower(), m.group(2).strip())
    rows = [_split(ln, delim) for ln in lines[a:b]]
    # rows layout: the header row (name, category, 400, 410, ...) is mostly numeric too, so the block finder
    # swallows it - take it back out when its numbers are wavelengths
    if header is None and len(rows) >= 2 and _looks_like_band_header(rows[0], rows[1:], dc):
        header, rows = [c.strip() for c in rows[0]], rows[1:]
    return meta, header, rows, dc


_HDR_WORDS = re.compile(r"^(name|material|sample|id|label|class|category|type|group|acquisition|platform|"
                        r"spectrum|description|site|date|location)", re.I)


def _looks_like_band_header(first, rest, dc=False) -> bool:
    nums = np.array([_num(c, dc) for c in first], float)
    fin = np.isfinite(nums)
    if fin.sum() < 5 or not _monotonic(nums[fin], 5):
        return False
    text = [c for c, f in zip(first, fin) if not f and str(c).strip()]
    if any(_HDR_WORDS.match(str(c).strip()) for c in text):
        return True
    v = nums[fin]
    if v.max() > 1.5 and np.allclose(np.diff(v), np.median(np.diff(v)), rtol=0.5):
        nxt = np.array([_num(c, dc) for c in rest[0]], float)[fin]
        return not _monotonic(nxt, 5) or abs(np.nanmax(nxt) - v.max()) > 0.25 * v.max()
    return False


def _monotonic(v: np.ndarray, min_n=3) -> bool:
    v = v[np.isfinite(v)]
    if v.size < min_n:
        return False
    d = np.diff(v)
    return bool(np.all(d > 0) or np.all(d < 0))


def _col_names(header, n):
    if header is None:
        return [""] * n
    h = list(header) + [""] * (n - len(header))
    return [str(c).strip() for c in h[:n]]


def _category_from_meta(meta: Dict[str, str]) -> Optional[str]:
    for k in ("category", "category path", "class path", "material category"):
        if meta.get(k):
            return meta[k]
    if meta.get("type") or meta.get("class"):
        lv = [meta.get("type", ""), meta.get("class", "")]
        if meta.get("genus"):
            lv += [meta.get("genus", ""), meta.get("species", "")]
        else:
            lv.append(meta.get("subclass", ""))
        lv = [x.strip().title() for x in lv if x and x.strip().lower() not in ("none", "n/a", "na", "-")]
        return "/".join(lv) or None
    return None


def _acq_from_meta(meta: Dict[str, str]) -> Optional[str]:
    for k in ("acquisition", "acquisition type", "platform", "measurement type", "setting"):
        if meta.get(k):
            return meta[k]
    blob = " ".join(meta.get(k, "") for k in ("instrument", "sensor", "spectrometer", "measurement", "description",
                                             "owner", "origin", "comment")).strip()
    return blob or None


def _scale_hint(*texts) -> str:
    t = " ".join(str(x) for x in texts if x).lower()
    if "percent" in t or "%" in t:
        return "percent"
    if "[1.0]" in t or "fraction" in t or "0-1" in t:
        return "unit"
    return "auto"


def _quantity_hint(*texts) -> Optional[str]:
    t = " ".join(str(x) for x in texts if x).lower()
    for q in ("reflectance", "transmittance", "emissivity", "absorbance", "radiance", "irradiance", "counts"):
        if q in t:
            return q
    if "refl" in t:
        return "reflectance"
    return None


def records_from_table(header, rows, meta, path: Path, opts: ReadOptions, *, decimal_comma=False, reader="table",
                       name_prefix="", group=None) -> List[SpectrumRecord]:
    """The layout logic shared by CSV/TXT, Excel, Parquet."""
    if not rows:
        return []
    n = max(len(r) for r in rows)
    rows = [list(r) + [""] * (n - len(r)) for r in rows]
    cols = _col_names(header, n)
    F = np.array([[_num(c, decimal_comma) for c in r] for r in rows], float)
    numeric_col = np.mean(np.isfinite(F), axis=0) >= 0.6
    hnum = np.array([_num(c, decimal_comma) for c in cols], float) if header is not None else np.full(n, np.nan)
    group = group or str(path)
    stem = name_prefix + stem_name(path)
    meta_cat, meta_acq = _category_from_meta(meta), _acq_from_meta(meta)
    if path.name.lower().endswith(".spectrum.txt") and not meta_acq:
        meta_acq = "lab"                     # ECOSTRESS / ASTER library: laboratory measurements
    file_unit = unit_from_text(meta.get("x units"), meta.get("wavelength units"), meta.get("wavelength unit"),
                               meta.get("units"))
    file_scale = _scale_hint(meta.get("y units"), meta.get("reflectance units"), meta.get("units"))
    file_qty = _quantity_hint(meta.get("y units"), meta.get("measurement"), meta.get("measurement type"))

    def pick(regex, explicit=None):
        if explicit:
            for i, c in enumerate(cols):
                if c.lower() == explicit.lower():
                    return i
        for i, c in enumerate(cols):
            if regex.match(c):
                return i
        return None

    orient = opts.orientation
    wl_i = pick(_WL_COL, opts.wavelength_column)
    name_i = pick(_NAME_COL, opts.name_column)
    band_cols = [i for i in range(n) if np.isfinite(hnum[i])]
    if orient == "auto":
        if header is not None and name_i is not None and wl_i is not None and \
                any(_VAL_COL.match(c) for c in cols) and len({rows[k][name_i] for k in range(len(rows))}) < len(rows) / 2:
            orient = "long"
        elif header is not None and len(band_cols) >= 5 and _monotonic(hnum[band_cols], 5):
            orient = "rows"
        else:
            orient = "columns"
    out: List[SpectrumRecord] = []

    if orient == "long":
        wl_i = wl_i if wl_i is not None else 1
        name_i = name_i if name_i is not None else 0
        val_i = next((i for i, c in enumerate(cols) if _VAL_COL.match(c)), None)
        if val_i is None:
            val_i = next(i for i in range(n) if numeric_col[i] and i != wl_i)
        cat_i = pick(_CAT_COL, opts.category_column)
        acq_i = pick(_ACQ_COL)
        groups: Dict[str, list] = {}
        for k, r in enumerate(rows):
            groups.setdefault(str(r[name_i]).strip() or f"{stem} {k}", []).append(k)
        unit = unit_from_text(cols[wl_i]) if unit_from_text(cols[wl_i]) != "auto" else file_unit
        for nm, ks in groups.items():
            out.append(SpectrumRecord(
                nm, F[ks, wl_i], F[ks, val_i], category=rows[ks[0]][cat_i] if cat_i is not None else meta_cat,
                acquisition=rows[ks[0]][acq_i] if acq_i is not None else meta_acq, wl_unit=unit,
                scale=_scale_hint(cols[val_i]) if _scale_hint(cols[val_i]) != "auto" else file_scale,
                quantity=_quantity_hint(cols[val_i]) or file_qty,
                material_metadata={"source_file": path.name},
                spectrum_metadata={"uid": f"{reader}:{path.name}:{nm}", "source_file": path.name}, group=group))
        return out

    if orient == "rows":
        if header is None or len(band_cols) < 2:
            raise ValueError("rows layout needs wavelengths in the header row")
        wl = hnum[band_cols]
        unit = unit_from_text(*[c for i, c in enumerate(cols) if i not in band_cols]) if file_unit == "auto" else file_unit
        meta_cols = [i for i in range(n) if i not in band_cols]
        name_i = name_i if name_i is not None else next((i for i in meta_cols if not numeric_col[i]), None)
        cat_i = pick(_CAT_COL, opts.category_column)
        acq_i = pick(_ACQ_COL)
        for k, r in enumerate(rows):
            vals = F[k, band_cols]
            if not np.isfinite(vals).any():
                continue
            nm = str(r[name_i]).strip() if name_i is not None and str(r[name_i]).strip() else f"{stem} {k + 1}"
            extra = {cols[i]: r[i] for i in meta_cols if i not in (name_i, cat_i, acq_i) and cols[i] and str(r[i]).strip()}
            out.append(SpectrumRecord(
                nm, wl, vals, category=(r[cat_i] if cat_i is not None and str(r[cat_i]).strip() else meta_cat),
                acquisition=(r[acq_i] if acq_i is not None and str(r[acq_i]).strip() else
                             (" ".join(str(v) for v in extra.values()) or meta_acq)),
                wl_unit=unit, scale=file_scale, quantity=file_qty,
                material_metadata={"source_file": path.name, **{k2: str(v)[:200] for k2, v in extra.items()}},
                spectrum_metadata={"uid": f"{reader}:{path.name}:row{k}", "source_file": path.name}, group=group))
        return out

    # ---- columns layout
    if wl_i is None:
        cand = [i for i in range(n) if numeric_col[i] and _monotonic(F[:, i])]
        if not cand:
            raise ValueError("no wavelength column found (monotonic numeric column) - set the wavelength column")
        wl_i = cand[0]
    wl = F[:, wl_i]
    ok_rows = np.isfinite(wl)
    unit = unit_from_text(cols[wl_i]) if unit_from_text(cols[wl_i]) != "auto" else file_unit
    spec_cols = [i for i in range(n) if i != wl_i and numeric_col[i]]
    # radiance + reflectance columns side by side (spectrometer exports): keep the reflectance-type ones
    if header is not None and len(spec_cols) > 1:
        refl = [i for i in spec_cols if _REFL_COL.search(cols[i]) and not _RAW_COL.search(cols[i].replace("Tgt./Ref.", ""))]
        raw = [i for i in spec_cols if _RAW_COL.search(cols[i])]
        if refl and raw:
            spec_cols = refl
    # *_mean / *_std pairs (HyperVista CSV export)
    lowered = {cols[i].lower(): i for i in spec_cols}
    std_of, skip = {}, set()
    for i in spec_cols:
        c = cols[i].lower()
        for suf in ("_mean", " mean", "-mean"):
            if c.endswith(suf):
                for ssuf in ("_std", " std", "-std", "_sd", " sd"):
                    j = lowered.get(c[:-len(suf)] + ssuf)
                    if j is not None:
                        std_of[i] = j
                        skip.add(j)
    # text-led rows (e.g. HyperVista "n_pixels" row) -> metadata
    rowmeta = {}
    for k, r in enumerate(rows):
        if not ok_rows[k] and str(r[wl_i]).strip():
            rowmeta[str(r[wl_i]).strip().lower()] = r
    single = len([i for i in spec_cols if i not in skip]) == 1
    for i in spec_cols:
        if i in skip:
            continue
        name = cols[i] if header is not None and cols[i] else ""
        for suf in ("_mean", " mean", "-mean"):
            if name.lower().endswith(suf):
                name = name[:-len(suf)]
        if single and (not name or _VAL_COL.match(name) or _REFL_COL.search(name)):
            name = meta.get("name") or ""
            if name and meta.get("sample no."):
                name = f"{name} {meta['sample no.']}"
        if not name:
            name = stem if single else f"{stem} {i}"
        std = F[ok_rows, std_of[i]] if i in std_of else None
        mm = {"source_file": path.name}
        if single:
            mm.update({k: v[:300] for k, v in meta.items() if k not in ("x units", "y units", "first column",
                                                                    "second column", "first x value", "last x value",
                                                                    "number of x values")})
        if "n_pixels" in rowmeta:
            mm["n_pixels"] = rowmeta["n_pixels"][i]
        out.append(SpectrumRecord(
            name, wl[ok_rows], F[ok_rows, i], std=std, category=meta_cat, acquisition=meta_acq, wl_unit=unit,
            scale=(_scale_hint(cols[i]) if _scale_hint(cols[i]) != "auto" else file_scale),
            quantity=_quantity_hint(cols[i]) or file_qty, material_metadata=mm,
            spectrum_metadata={"uid": f"{reader}:{path.name}:{cols[i] or i}", "source_file": path.name}, group=group))
    return out


# ================================================================== readers
@register
class TableReader(Reader):
    name = "table"
    label = "Text table (CSV / TSV / TXT)"
    extensions = TEXT_EXT

    def sniff(self, path: Path) -> float:
        if not path.is_file() or path.suffix.lower() not in TEXT_EXT:
            return 0.0
        head = _decode(path.read_bytes()[:200000])
        delim, dc, a, b = find_block(head.splitlines())
        if b - a >= 3:
            return 0.6
        return 0.0

    def describe(self, path: Path) -> str:
        meta, header, rows, dc = parse_text_table(_decode(path.read_bytes()[:400000]))
        if not rows:
            return self.label
        ncol = max(len(r) for r in rows)
        return f"{self.label}: {len(rows)}+ rows × {ncol} columns"

    def read(self, path: Path, opts: ReadOptions):
        meta, header, rows, dc = parse_text_table(_decode(path.read_bytes()))
        return records_from_table(header, rows, meta, path, opts, decimal_comma=dc, reader=self.name)


@register
class ExcelReader(Reader):
    name = "excel"
    label = "Excel workbook"
    extensions = (".xlsx", ".xlsm")

    def sniff(self, path: Path) -> float:
        return 0.9 if path.is_file() and path.suffix.lower() in self.extensions else 0.0

    def read(self, path: Path, opts: ReadOptions):
        try:
            import openpyxl
        except ImportError:
            raise RuntimeError("reading Excel files needs openpyxl - pip install openpyxl") from None
        wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
        out = []
        sheets = [opts.sheet] if opts.sheet else wb.sheetnames
        for sn in sheets:
            ws = wb[sn]
            grid = [["" if v is None else v for v in row] for row in ws.iter_rows(values_only=True)]
            grid = [r for r in grid if any(str(c).strip() for c in r)]
            if not grid:
                continue
            lines = ["\t".join(str(c) for c in r) for r in grid]
            delim, dc, a, b = find_block(lines)
            if b - a < 2:
                continue
            ncols = max(len(grid[k]) for k in range(a, b))
            header = None
            meta = {}
            if a >= 1 and sum(1 for c in grid[a - 1] if str(c).strip()) >= 2:
                header = [str(c) for c in grid[a - 1]][:ncols]
                top = a - 1
            else:
                top = a
            for r in grid[:top]:
                cells = [str(c).strip() for c in r if str(c).strip()]
                if len(cells) == 2:
                    meta.setdefault(cells[0].rstrip(":=").lower(), cells[1])
            rows = [list(r)[:ncols] for r in grid[a:b]]
            prefix = f"{sn} " if len(sheets) > 1 else ""
            out += records_from_table(header, rows, meta, path, opts, reader=self.name, name_prefix=prefix,
                                      group=f"{path}:{sn}")
        wb.close()
        return out


@register
class JsonReader(Reader):
    name = "json"
    label = "JSON spectra"
    extensions = (".json", ".geojson")

    def sniff(self, path: Path) -> float:
        if not path.is_file() or path.suffix.lower() not in self.extensions:
            return 0.0
        head = path.read_bytes()[:4000].decode("utf-8", "replace").lower()
        return 0.8 if any(k in head for k in ("wavelength", "spectr", "reflect", "hvva-signatures")) else 0.3

    def read(self, path: Path, opts: ReadOptions):
        doc = json.loads(path.read_text(encoding="utf-8-sig"))
        out: List[SpectrumRecord] = []
        g = str(path)

        def rec(name, wl, vals, std=None, extra=None, key=None):
            extra = extra or {}
            unit = unit_from_text(extra.get("wavelength_units"), extra.get("wavelength_unit"), extra.get("x_units"),
                                  doc.get("wavelength_units") if isinstance(doc, dict) else None)
            out.append(SpectrumRecord(
                str(name), np.array([safe_float(v) for v in wl]), np.array([safe_float(v) for v in vals]),
                std=None if std is None else np.array([safe_float(v) for v in std]),
                category=extra.get("category") or extra.get("class") or extra.get("type"),
                acquisition=extra.get("acquisition") or extra.get("acquisition_type") or extra.get("platform")
                or extra.get("instrument"), wl_unit=unit,
                scale=_scale_hint(extra.get("units"), extra.get("y_units")),
                material_metadata={k: v for k, v in extra.items() if isinstance(v, (str, int, float)) and k not in (
                    "category", "acquisition")} | {"source_file": path.name},
                spectrum_metadata={"uid": f"json:{path.name}:{key or name}", "source_file": path.name}, group=g))

        def wl_of(d):
            for k in ("wavelengths", "wavelength", "wl", "x", "wavelengths_nm", "bands"):
                if isinstance(d.get(k), list):
                    return d[k]
            return None

        def vals_of(d):
            for k in ("values", "reflectance", "spectrum", "y", "data", "mean", "value"):
                if isinstance(d.get(k), list):
                    return k, d[k]
            return None, None

        def walk(node, inherited_wl=None, path_key=""):
            if isinstance(node, dict):
                wl = wl_of(node) or inherited_wl
                vk, vals = vals_of(node)
                if vals is not None and wl is not None and len(vals) and not isinstance(vals[0], (list, dict)):
                    rec(node.get("name") or node.get("label") or node.get("id") or path_key or stem_name(path),
                        wl, vals, node.get("std"), node, key=path_key)
                    return
                if node.get("format") == "hvva-signatures-1" and isinstance(node.get("classes"), dict):
                    for nm, c in node["classes"].items():
                        rec(nm, node["wavelengths"], c.get("mean", []), c.get("std"),
                            {"n_pixels": c.get("n_pixels"), "source": c.get("source")}, key=nm)
                    return
                spectra = node.get("spectra")
                if isinstance(spectra, dict) and wl is not None:
                    for nm, v in spectra.items():
                        if isinstance(v, list):
                            rec(nm, wl, v, key=nm)
                        elif isinstance(v, dict):
                            walk(dict(v, name=v.get("name", nm)), wl, nm)
                    return
                for k, v in node.items():
                    if isinstance(v, (dict, list)) and k not in ("wavelengths", "wavelength"):
                        walk(v, wl, f"{path_key}/{k}".strip("/"))
            elif isinstance(node, list):
                for i, v in enumerate(node):
                    if isinstance(v, (dict, list)):
                        walk(v, inherited_wl, f"{path_key}[{i}]")
        walk(doc)
        return out


@register
class ParquetTableReader(Reader):
    name = "parquet"
    label = "Parquet table"
    extensions = (".parquet", ".pq")

    def sniff(self, path: Path) -> float:
        if not path.is_file() or path.suffix.lower() not in self.extensions or path.name.lower() == "wavelengths.parquet":
            return 0.0
        try:
            import pyarrow.parquet as pq
            pq.ParquetFile(path)
        except Exception:                                                   # noqa: BLE001
            return 0.0
        return 0.4                                    # OpenSpecLib files are claimed with higher confidence

    def read(self, path: Path, opts: ReadOptions):
        try:
            import pyarrow.parquet as pq
            import pyarrow as pa
        except ImportError:
            raise RuntimeError("reading Parquet files needs pyarrow - pip install pyarrow") from None
        t = pq.read_table(path)
        names = t.column_names
        list_cols = [c for c in names if pa.types.is_list(t.schema.field(c).type) or
                     pa.types.is_large_list(t.schema.field(c).type)]
        if list_cols:
            vc = next((c for c in list_cols if re.search(r"value|spectrum|reflect|data|^y$", c, re.I)), list_cols[0])
            wc = next((c for c in list_cols if c != vc and re.search(r"wave|wl|^x$|band", c, re.I)), None)
            if wc is None:
                raise ValueError("Parquet file has spectra in a list column but no wavelength list column")
            rows = t.to_pylist()
            nc = next((c for c in names if _NAME_COL.match(c.split(".")[-1])), None)
            cc = next((c for c in names if _CAT_COL.match(c.split(".")[-1])), None)
            ac = next((c for c in names if _ACQ_COL.match(c.split(".")[-1])), None)
            out = []
            for k, r in enumerate(rows):
                out.append(SpectrumRecord(
                    str(r.get(nc) or f"{stem_name(path)} {k + 1}"), np.array(r[wc] or [], float),
                    np.array([np.nan if v is None else v for v in (r[vc] or [])], float),
                    category=r.get(cc) if cc else None, acquisition=r.get(ac) if ac else None,
                    wl_unit=unit_from_text(wc), material_metadata={"source_file": path.name},
                    spectrum_metadata={"uid": f"parquet:{path.name}:{k}", "source_file": path.name}, group=str(path)))
            return out
        header = [str(c) for c in names]
        rows = [[r.get(c) for c in names] for r in t.to_pylist()]
        rows = [["" if v is None else v for v in r] for r in rows]
        return records_from_table(header, rows, {}, path, opts, reader=self.name)
