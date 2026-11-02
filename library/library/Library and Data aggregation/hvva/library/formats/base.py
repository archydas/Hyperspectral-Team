"""Records, options, the reader registry and the normalisation every source goes through."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np

from .. import repository as repo

BAD = 1.0e30                       # |value| above this = deleted channel (USGS -1.23e34, ENVI -9999 handled separately)


# ================================================================== data structures
@dataclass
class SpectrumRecord:
    """One spectrum as a reader found it. Units are hints; normalize() turns them into nm and 0-1."""
    name: str
    wavelengths: np.ndarray
    values: np.ndarray
    std: Optional[np.ndarray] = None
    category: Optional[str] = None
    acquisition: Optional[str] = None
    wl_unit: str = "auto"                     # nm | um | cm-1 | auto
    scale: str = "auto"                       # unit | percent | x10000 | none (not reflectance) | auto
    quantity: Optional[str] = None            # reflectance | radiance | transmittance | counts | ...
    material_metadata: Dict = field(default_factory=dict)
    spectrum_metadata: Dict = field(default_factory=dict)   # must hold a source-unique "uid"
    group: str = ""                           # unit/scale decisions are made per group (= file)


@dataclass
class ReadOptions:
    """What the user can override; 'auto' everywhere means: work it out from the file."""
    wavelength_unit: str = "auto"             # auto | nm | um | cm-1
    value_scale: str = "auto"                 # auto | unit | percent | x10000 | none
    orientation: str = "auto"                 # tables: auto | columns | rows | long
    spectral_range: str = "auto"              # auto | all | vswir   (auto = vswir for reference collections)
    category: Optional[str] = None            # default category path for records without one
    category_from_folders: bool = True        # folder / zip imports: subfolder names -> category levels
    acquisition: Optional[str] = None         # default when nothing can be inferred
    force_acquisition: bool = False           # True: use `acquisition` for every record
    image_mode: str = "auto"                  # images: auto (classes, else mean) | classes | mean | sample | all
    image_samples: int = 100                  # pixels for image_mode=sample (all: cap)
    wl_range: Optional[Tuple[float, float]] = None    # files without wavelengths: spread bands over this (nm)
    wavelength_column: Optional[str] = None   # tables: override the detected columns
    name_column: Optional[str] = None
    category_column: Optional[str] = None
    sheet: Optional[str] = None               # excel: only this sheet
    limit: Optional[int] = None               # at most N spectra per source
    chapters: Optional[str] = None            # USGS: chapter letters
    categories: Optional[str] = None          # OpenSpecLib: source categories
    reader: Optional[str] = None              # force a reader by name
    warnings: List[str] = field(default_factory=list, repr=False)   # filled by readers (skipped files ...)

    @classmethod
    def from_dict(cls, d: Optional[dict]):
        d = dict(d or {})
        if isinstance(d.get("wl_range"), list):
            d["wl_range"] = tuple(d["wl_range"])
        d.pop("warnings", None)
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__ and v is not None})


@dataclass
class ReadInfo:
    reader: str
    label: str
    n_found: int = 0
    n_kept: int = 0
    wl_unit: str = ""
    scale: str = ""
    wl_min: Optional[float] = None
    wl_max: Optional[float] = None
    categories: List[str] = field(default_factory=list)
    acquisitions: Dict[str, int] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def summary(self) -> str:
        bits = [f"{self.label}: {self.n_kept} spectra"]
        if self.n_found != self.n_kept:
            bits[0] += f" (of {self.n_found} found)"
        if self.wl_min is not None:
            bits.append(f"{self.wl_min:g}–{self.wl_max:g} nm (read as {self.wl_unit})")
        if self.scale:
            bits.append(f"values: {self.scale}")
        if self.categories:
            bits.append(f"{len(self.categories)} categories")
        return " · ".join(bits)


class Reader:
    """Base class. Subclasses set name / label / extensions and implement sniff() and read()."""
    name = "reader"
    label = "Reader"
    extensions: Tuple[str, ...] = ()
    is_container = False               # folder / zip / multi-file collection
    reference_collection = False       # big public libraries: VSWIR only unless asked otherwise

    def sniff(self, path: Path) -> float:
        """0 = cannot read, 1 = certainly this format."""
        return 0.0

    def read(self, path: Path, opts: ReadOptions) -> Iterable[SpectrumRecord]:
        raise NotImplementedError

    def describe(self, path: Path) -> str:
        return self.label

    def wants_vswir(self, path: Path) -> bool:
        """spectral_range=auto: trim to 350-2500 nm? (yes for big reference collections with thermal spectra)"""
        return self.reference_collection


REGISTRY: List[Reader] = []


def register(cls):
    REGISTRY.append(cls())
    return cls


def readers_info():
    return [{"name": r.name, "label": r.label, "extensions": list(r.extensions)} for r in REGISTRY]


def _all_ext():
    out = set()
    for r in REGISTRY:
        out.update(r.extensions)
    return sorted(out)


class _LazyExt(list):
    def __iter__(self):
        return iter(_all_ext())

    def __contains__(self, x):
        return x in _all_ext()

    def __len__(self):
        return len(_all_ext())


SUPPORTED_EXTENSIONS = _LazyExt()


def file_dialog_filter() -> str:
    exts = " ".join(f"*{e}" for e in _all_ext())
    per = ";;".join(f"{r.label} ({' '.join('*' + e for e in r.extensions)})" for r in REGISTRY if r.extensions)
    return f"All supported spectral data ({exts});;{per};;All files (*)"


def detect(path, exclude_containers=False) -> Tuple[Optional[Reader], float, str]:
    """(reader, confidence, description) of the best reader for *path*, or (None, 0, reason)."""
    p = Path(path)
    if not p.exists():
        return None, 0.0, "does not exist"
    best, score = None, 0.0
    for r in REGISTRY:
        if exclude_containers and r.is_container:
            continue
        try:
            s = r.sniff(p)
        except Exception:                                                   # noqa: BLE001
            s = 0.0
        if s > score:
            best, score = r, s
    if best is None or score <= 0:
        return None, 0.0, "not a recognised spectral data file"
    try:
        desc = best.describe(p)
    except Exception:                                                       # noqa: BLE001
        desc = best.label
    return best, score, desc


def get_reader(name) -> Reader:
    for r in REGISTRY:
        if r.name == name:
            return r
    raise ValueError(f"unknown reader {name!r} - available: {', '.join(r.name for r in REGISTRY)}")


def read_path(path, opts: Optional[ReadOptions] = None,
              progress: Optional[Callable[[int, int, str], None]] = None) -> Tuple[List[SpectrumRecord], ReadInfo]:
    """Detect, read and normalise one file / folder / zip."""
    opts = opts or ReadOptions()
    p = Path(path)
    if opts.reader:
        reader = get_reader(opts.reader)
    else:
        reader, score, why = detect(p)
        if reader is None:
            raise ValueError(f"{p.name}: {why}")
    raw = []
    for i, rec in enumerate(reader.read(p, opts), 1):
        raw.append(rec)
        if progress and i % 50 == 0:
            progress(i, 0, f"reading {rec.name}")
    info = ReadInfo(reader=reader.name, label=reader.label, n_found=len(raw))
    info.warnings.extend(opts.warnings)
    vswir = opts.spectral_range == "vswir" or (opts.spectral_range == "auto" and reader.wants_vswir(p))
    recs = normalize(raw, opts, info, vswir=vswir)
    if opts.limit:
        recs = recs[:opts.limit]
    info.n_kept = len(recs)
    if recs:
        info.wl_min = float(min(np.nanmin(r.wavelengths) for r in recs))
        info.wl_max = float(max(np.nanmax(r.wavelengths) for r in recs))
    info.categories = sorted({r.category for r in recs if r.category})
    for r in recs:
        info.acquisitions[r.acquisition or "untagged"] = info.acquisitions.get(r.acquisition or "untagged", 0) + 1
    if not recs:
        info.warnings.append("no spectra could be read from this source")
    return recs, info


# ================================================================== inference helpers
_UNIT_PAT = [("cm-1", re.compile(r"cm\s*\^?\s*-\s*1|cm⁻¹|1/cm|wave\s*number|wavenumber", re.I)),
             ("um", re.compile(r"µm|μm|\bum\b|micro|micron", re.I)),
             ("nm", re.compile(r"\bnm\b|nanomet", re.I))]


def unit_from_text(*texts) -> str:
    t = " ".join(str(x) for x in texts if x)
    for unit, pat in _UNIT_PAT:
        if pat.search(t):
            return unit
    return "auto"


def guess_unit(x: np.ndarray) -> str:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return "nm"
    mx = float(np.nanmax(x))
    if mx < 30:                                   # 0.35-25 µm
        return "um"
    return "nm"


def to_nm(x: np.ndarray, unit: str) -> np.ndarray:
    if unit == "um":
        return x * 1000.0
    if unit == "cm-1":
        with np.errstate(divide="ignore"):
            return np.where(x > 0, 1.0e7 / x, np.nan)
    return x


_ACQ_WORDS = [
    ("satellite", re.compile(r"satellite|spaceborne|space-borne|hyperion|prisma|enmap|desis|emit\b|hisui|sentinel|"
                             r"landsat|worldview|aster\b|modis|tanager|wyvern|planet|superdove|gaofen|orbital", re.I)),
    ("airborne", re.compile(r"airborne|aircraft|aerial|uav|drone|uas\b|aviris|hymap|casi|aisa|hyspex|apex\b|"
                            r"specim|headwall|nano-?hyperspec|micro-?hyperspec|hytes|g-?liht|neon aop", re.I)),
    ("field", re.compile(r"\bfield\b|in[- ]?situ|outdoor|ground[- ]?truth|canopy|plot\b|spectroradiometer field", re.I)),
    ("lab", re.compile(r"\blab\b|laboratory|contact probe|integrating sphere|beckman|nicolet|perkin|lambda ?9|"
                       r"cary|ftir|bench|leaf clip|plant probe", re.I)),
]
_ACQ_EXACT = {"lab": "lab", "laboratory": "lab", "field": "field", "in-situ": "field", "insitu": "field",
              "in situ": "field", "ground": "field", "airborne": "airborne", "aircraft": "airborne", "uav": "airborne",
              "drone": "airborne", "satellite": "satellite", "spaceborne": "satellite", "space": "satellite"}


def infer_acquisition(*texts) -> Optional[str]:
    """lab / field / airborne / satellite from free text (column value, instrument, description...)."""
    for t in texts:
        if t is None:
            continue
        s = str(t).strip().lower()
        if s in _ACQ_EXACT:
            return _ACQ_EXACT[s]
    blob = " ".join(str(t) for t in texts if t)
    for tag, pat in _ACQ_WORDS:
        if pat.search(blob):
            return tag
    return None


# ================================================================== normalisation
def _clean_one(r: SpectrumRecord, unit: str):
    x = np.asarray(r.wavelengths, float).ravel()
    y = np.asarray(r.values, float).ravel()
    s = None if r.std is None else np.asarray(r.std, float).ravel()
    n = min(x.size, y.size)
    x, y = x[:n], y[:n]
    if s is not None:
        s = s[:n] if s.size >= n else None
    x = to_nm(x, unit)
    bad = ~np.isfinite(x)
    y = np.where(np.isfinite(y) & (np.abs(y) < BAD), y, np.nan)
    if s is not None:
        s = np.where(np.isfinite(s) & (np.abs(s) < BAD), s, np.nan)
    keep = ~bad
    x, y = x[keep], y[keep]
    s = s[keep] if s is not None else None
    order = np.argsort(x, kind="stable")
    x, y = x[order], y[order]
    s = s[order] if s is not None else None
    if x.size:
        uniq = np.concatenate([[True], np.diff(x) > 0])
        x, y = x[uniq], y[uniq]
        s = s[uniq] if s is not None else None
    return x, y, s


def _decide_scale(recs: List[SpectrumRecord], forced: str) -> str:
    if forced != "auto":
        return forced
    known = {r.scale for r in recs if r.scale != "auto"}
    if len(known) == 1 and all(r.scale != "auto" for r in recs):
        return known.pop()
    qty = {(r.quantity or "reflectance").lower() for r in recs}
    if qty and not qty & {"reflectance", "transmittance", "absorptance", "emissivity"}:
        return "none"                                         # radiance / counts: leave as measured
    p99 = []
    for r in recs[:400]:
        v = np.asarray(r.values, float)
        v = v[np.isfinite(v) & (np.abs(v) < BAD)]
        if v.size:
            p99.append(np.percentile(v, 99))
    if not p99:
        return "unit"
    med = float(np.median(p99))
    if med <= 1.5:
        return "unit"
    if med <= 150:
        return "percent"
    if med <= 15000:
        return "x10000"
    return "none"


_SCALE_DIV = {"unit": 1.0, "percent": 100.0, "x10000": 10000.0, "none": 1.0}


def normalize(raw: List[SpectrumRecord], opts: ReadOptions, info: Optional[ReadInfo] = None,
              vswir: bool = False) -> List[SpectrumRecord]:
    """Units -> nm, scale -> 0-1 (reflectance), bad values -> NaN, sorted grid, category / acquisition filled."""
    groups: Dict[str, List[SpectrumRecord]] = {}
    for r in raw:
        groups.setdefault(r.group, []).append(r)
    out, units, scales = [], set(), set()
    default_cat = repo.normalize_category(opts.category) if opts.category else None
    default_acq = repo.normalize_acquisition(opts.acquisition) if opts.acquisition else None
    for g, recs in groups.items():
        # unit per group: explicit option > reader hint > heuristic over the whole group
        if opts.wavelength_unit != "auto":
            unit = opts.wavelength_unit
        else:
            hints = {r.wl_unit for r in recs if r.wl_unit != "auto"}
            if len(hints) == 1:
                unit = hints.pop()
            else:
                allx = np.concatenate([np.asarray(r.wavelengths, float).ravel() for r in recs[:200]])
                unit = guess_unit(allx)
        units.add(unit)
        cleaned = []
        for r in recs:
            ru = r.wl_unit if (opts.wavelength_unit == "auto" and r.wl_unit != "auto") else unit
            x, y, s = _clean_one(r, ru)
            cleaned.append((r, x, y, s))
        scale = _decide_scale([SpectrumRecord(r.name, x, y, scale=r.scale, quantity=r.quantity)
                               for r, x, y, s in cleaned], opts.value_scale)
        scales.add(scale)
        for r, x, y, s in cleaned:
            sc = r.scale if (opts.value_scale == "auto" and r.scale != "auto") else scale
            div = _SCALE_DIV.get(sc, 1.0)
            if div != 1.0:
                y = y / div
                s = s / div if s is not None else None
            if vswir:
                if x.size and x.min() > 1000:            # thermal / MIR-only measurement
                    continue
                sel = x <= 2500.5
                x, y = x[sel], y[sel]
                s = s[sel] if s is not None else None
            if x.size < 2 or not np.isfinite(y).any():
                continue
            r.wavelengths, r.values, r.std = x, y, s
            r.wl_unit, r.scale = "nm", sc
            if r.category:
                r.category = repo.normalize_category(r.category)
            if not r.category and default_cat:
                r.category = default_cat
            if opts.force_acquisition and default_acq:
                r.acquisition = default_acq
            else:
                r.acquisition = repo.normalize_acquisition(r.acquisition) if r.acquisition in repo.ACQUISITION_TYPES \
                    else (infer_acquisition(r.acquisition) if r.acquisition else None)
                if not r.acquisition:
                    r.acquisition = default_acq
            r.name = re.sub(r"\s+", " ", str(r.name or "spectrum")).strip()[:500] or "spectrum"
            if sc not in ("unit", "none", "auto"):
                r.spectrum_metadata.setdefault("rescaled_from", sc)
            if r.quantity:
                r.spectrum_metadata.setdefault("quantity", r.quantity)
            out.append(r)
    if info is not None:
        info.wl_unit = "/".join(sorted(units))
        info.scale = "/".join(sorted({"0-1 reflectance" if s == "unit" else
                                      "percent → 0-1" if s == "percent" else
                                      "×10000 → 0-1" if s == "x10000" else "as measured" for s in scales}))
    return out


# ================================================================== small shared helpers
def rel_category(root: Path, file: Path) -> Optional[str]:
    """Subfolder path of *file* under *root* as a category path (Vegetation/Tree for root/Vegetation/Tree/x.asd)."""
    try:
        parts = file.relative_to(root).parts[:-1]
    except ValueError:
        return None
    parts = [re.sub(r"[_]+", " ", p).strip() for p in parts if p and not p.startswith(".")]
    return "/".join(parts) or None


def stem_name(path) -> str:
    return re.sub(r"[_]+", " ", Path(path).stem).strip()


def safe_float(v) -> float:
    if v is None:
        return np.nan
    if isinstance(v, (int, float, np.floating, np.integer)):
        return float(v)
    s = str(v).strip().replace("−", "-")
    if not s:
        return np.nan
    try:
        return float(s)
    except ValueError:
        try:
            return float(s.replace(",", "."))
        except ValueError:
            return np.nan


def file_kind_hint(path: Path) -> str:
    return os.path.splitext(str(path))[1].lower()
