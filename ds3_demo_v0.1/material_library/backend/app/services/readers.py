"""Turn a spectra file into plain records:  dict(name, wavelengths [nm], values, unit, category, meta).

* With the optional `ds3` package installed (pip install -e <path>/ds3-platform) every format it supports is read:
  CSV (2 layouts), SVC .sig, ENVI .sli, USGS splib07, ECOSTRESS, ASD/SED.
* Without it, CSV files still work through the small reader below.
"""
from __future__ import annotations
import csv
import math
from pathlib import Path

DS3_ERROR = "not installed"
try:                                    # optional dependency
    from ds3.spectra import read_spectra, detect_spectra_format
    from ds3.spectra.categories import guess_category
    HAVE_DS3 = True
except Exception as _e:                 # not installed (or a compiled dependency is missing)
    HAVE_DS3 = False
    DS3_ERROR = f"{type(_e).__name__}: {_e}"
    print(f"[ds3] the DS3 core could not be loaded: {DS3_ERROR}")

    def guess_category(name):
        return None


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _finite(values):
    return [None if (v is None or not math.isfinite(v)) else float(v) for v in values]


def _read_csv_basic(path, unit):
    with open(path, newline="", encoding="utf-8-sig", errors="replace") as f:
        rows = [r for r in csv.reader(f) if r and any(c.strip() for c in r)]
    if len(rows) < 3:
        raise ValueError(f"{Path(path).name}: CSV needs a header row and at least two data rows")
    head = rows[0]
    out = []
    if len(head) > 2 and all(_num(c) is not None for c in head[1:]):          # row-wise: header = wavelengths
        wl = [float(c) for c in head[1:]]
        for r in rows[1:]:
            out.append((r[0].strip() or f"row{len(out)+1}", wl, [_num(x) for x in r[1:1 + len(wl)]]))
    else:                                                                       # wide: col0 = wavelength
        body = [r for r in rows[1:] if _num(r[0]) is not None]
        wl = [float(r[0]) for r in body]
        for j, name in enumerate(head[1:], start=1):
            out.append((name.strip() or f"col{j}", wl, [_num(r[j]) if j < len(r) else None for r in body]))
    recs = []
    for name, wl, vals in out:
        w = list(wl)
        if w and max(w) < 30:                                                   # micrometres -> nm
            w = [x * 1000 for x in w]
        if w != sorted(w):
            order = sorted(range(len(w)), key=w.__getitem__)
            w = [w[i] for i in order]; vals = [vals[i] for i in order]
        recs.append(dict(name=name, wavelengths=w, values=_finite([v if v is not None else float("nan") for v in vals]),
                         unit=unit or "reflectance", category=None, meta={}))
    return recs


def load_spectra(path, unit=None):
    """Read one file -> list of records. Raises ValueError with a readable message if the file cannot be read."""
    p = Path(path)
    try:
        if HAVE_DS3:
            fmt = detect_spectra_format(p)
            if fmt:
                kw = {"unit": unit} if unit else {}
                if fmt == "usgs":
                    sib = sorted(p.parent.glob("*avelength*.txt"))
                    if sib:
                        kw["wavelengths_file"] = sib[0]
                recs = []
                for s in read_spectra(p, **kw):
                    meta = {k: v for k, v in (s.metadata or {}).items() if isinstance(v, (str, int, float, bool, type(None)))}
                    recs.append(dict(name=s.material, wavelengths=[float(x) for x in s.wavelengths_nm],
                                     values=_finite(s.values), unit=s.unit, category=s.category, meta=meta))
                return recs
        if p.suffix.lower() == ".csv":
            return _read_csv_basic(p, unit)
    except ValueError:
        raise
    except Exception as e:                                  # reader-specific errors -> one readable message
        raise ValueError(f"{p.name}: {type(e).__name__}: {e}") from e
    hint = "" if HAVE_DS3 else f" (the DS3 core is not available: {DS3_ERROR}; install it with pip install -e <path>/ds3-platform)"
    raise ValueError(f"{p.name}: unsupported file type{hint}")


def is_candidate(path: Path) -> bool:
    n = path.name.lower()
    if n.startswith(".") or "wavelength" in n and n.endswith(".txt"):
        return False
    return n.endswith((".csv", ".sig", ".sli", ".spectrum.txt", ".asd", ".sed", ".xlsx", ".json", ".parquet")) or (HAVE_DS3 and n.endswith((".txt", ".tsv", ".dat")))
