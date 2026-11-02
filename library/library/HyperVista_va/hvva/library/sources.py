"""Local data folder -> library database, automatically - for every format hvva.library.formats can read.

Point it at a folder (e.g. D:\\speclib). scan() lists every importable source with a proposed library name;
status() says which are already in the database; import_many() / import_new() load them through the same
universal pipeline as everything else (hvva.library.ingest), creating libraries as needed.

    python -m hvva.library.sources D:/speclib              # list what is there and what is new
    python -m hvva.library.sources D:/speclib --import     # import everything new
    python -m hvva.library.sources D:/speclib --import --only leaves.csv --acquisition field

What becomes one source:
  * every file a reader recognises (CSV/TXT/Excel/JSON/Parquet/ENVI/.mat/.h5/images/.asd/.sig/.sed/...);
    ENVI header + data pairs and HyperVista's *_std.sli companions count once
  * a USGS splib07 ASCII release (zip or extracted folder) and any other zip of spectral files
  * a folder of single-spectrum instrument files (ASD / SVC / SE, ECOSTRESS *.spectrum.txt): the whole folder,
    with its subfolder names as category levels
Other folders are searched (3 levels deep by default).

"Already imported" = a data_files row with the file's SHA-256 (or the folder's path) exists, or the target
library already holds the dataset this source would create.  Re-importing is harmless anyway (spectra already in
the dataset are skipped).
"""
from __future__ import annotations

import argparse
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

from sqlalchemy.orm import Session

from . import repository as repo
from .formats import ReadOptions, detect
from .formats.base import REGISTRY
from .formats.refcollections import _walk_files
from .ingest import default_dataset_name, file_sha256, import_path
from .models import DataFile, Dataset, Library

_SKIP_DIRS = {"__pycache__", ".git", "node_modules", "$recycle.bin", "system volume information", "__macosx"}
_PER_FILE = {"asd", "sig", "sed"}              # one spectrum per file -> a folder of them is one source
_OSL_NAMES = {"ecostress": "ECOSTRESS", "usgs_splib07": "USGS splib07 (OSL)", "ecosis": "EcoSIS",
              "relab": "RELAB", "asu": "ASU TES", "bishop": "Bishop", "ossl": "OSSL"}


def kind_label(kind: str) -> str:
    for r in REGISTRY:
        if r.name == kind:
            return r.label
    return {"csv": "Wide CSV"}.get(kind, kind)


@dataclass
class Source:
    path: str
    kind: str                         # reader name (table, excel, envi_sli, image, asd, usgs, openspeclib, folder ...)
    library: str                      # proposed target library name
    dataset: str                      # dataset name the import will create
    size: int = 0
    note: str = ""
    status: str = "new"               # new | imported | error
    imported_into: List[str] = field(default_factory=list)
    checksum: Optional[str] = None

    @property
    def name(self):
        return os.path.basename(self.path.rstrip("/\\")) or self.path

    def to_dict(self):
        d = asdict(self)
        d["name"] = self.name
        d["kind_label"] = kind_label(self.kind)
        return d


# ================================================================== detection
def _pretty(stem: str) -> str:
    return re.sub(r"[_\s]+", " ", stem).strip().title()


def _usgs_library_name(stem: str) -> str:
    m = re.match(r"ASCIIdata_splib07([ab])?(?:_(\w+))?", stem, re.I)
    if not m:
        return "USGS splib07"
    variant = (m.group(2) or "").lower()
    if variant in ("", "cvasd"):
        return "USGS splib07"
    nice = {"cvavirisc2014": "AVIRIS 2014", "cvhymap2014": "HyMap 2014", "rsaster": "ASTER", "rslandsat8": "Landsat 8",
            "rssentinel2": "Sentinel-2", "rsworldview3": "WorldView-3"}.get(variant, variant)
    return f"USGS splib07 ({nice})"


def _library_for(p: Path, kind: str) -> str:
    if kind == "openspeclib":
        return _OSL_NAMES.get(p.stem.lower(), _pretty(p.stem))
    if kind == "usgs":
        return _usgs_library_name(p.stem if p.is_file() else p.name)
    ds = default_dataset_name(p, kind)
    if ds.startswith("ECOSTRESS"):
        return "ECOSTRESS"
    return _pretty(p.stem if p.is_file() else p.name)


def _make(p: Path, reader, desc: str) -> Source:
    size = p.stat().st_size if p.is_file() else 0
    note = "" if desc == reader.label else desc.replace(reader.label + ": ", "")
    st = "error" if "missing" in note else "new"
    return Source(str(p), reader.name, _library_for(p, reader.name), default_dataset_name(p, reader.name), size,
                  note=note, status=st)


def _instrument_folder(d: Path) -> Optional[int]:
    """Number of single-spectrum files if *d* is a folder of instrument files (ASD/SVC/SE/ECOSTRESS), else None."""
    files = _walk_files(d, max_files=5000)
    n_inst, n_other = 0, 0
    for f in files[:3000]:
        if f.name.lower().endswith(".spectrum.txt"):
            n_inst += 1
            continue
        if f.name.lower().endswith(".ancillary.txt"):
            continue
        r, s, _ = detect(f, exclude_containers=True)
        if r is None:
            continue
        if r.name in _PER_FILE:
            n_inst += 1
        else:
            n_other += 1
    return n_inst if n_inst >= 2 and n_other == 0 else None


def scan(folder, max_depth: int = 3) -> List[Source]:
    """All importable sources under *folder*."""
    root = Path(folder)
    if not root.is_dir():
        raise FileNotFoundError(f"not a folder: {root}")
    found: List[Source] = []

    def classify_dir(d: Path) -> Optional[Source]:
        r, s, desc = detect(d)
        if r is not None and r.name == "usgs":
            return _make(d, r, desc)
        n = _instrument_folder(d)
        if n:
            folder_reader = next(x for x in REGISTRY if x.name == "folder")
            src = _make(d, folder_reader, folder_reader.label)
            src.note = f"{n} instrument files; subfolders → categories"
            return src
        return None

    def walk(d: Path, depth: int):
        try:
            entries = sorted(d.iterdir(), key=lambda e: e.name.lower())
        except OSError:
            return
        seen_stems = set()
        for e in entries:
            if e.name.lower() in _SKIP_DIRS or e.name.startswith((".", "~$")):
                continue
            if e.is_dir():
                if e.suffix.lower() == ".zarr":
                    r, s, desc = detect(e)
                    if r is not None:
                        found.append(_make(e, r, desc))
                    continue
                src = classify_dir(e)
                if src:
                    found.append(src)
                elif depth < max_depth:
                    walk(e, depth + 1)
                continue
            if e.name.lower() == "wavelengths.parquet":
                continue
            r, s, desc = detect(e)
            if r is None or r.name == "folder":
                continue
            if r.name in ("envi_sli", "image"):          # header + data file = one source
                key = (e.stem.lower(), r.name)
                if key in seen_stems:
                    continue
                seen_stems.add(key)
                if e.suffix.lower() != ".hdr" and (e.with_suffix(".hdr")).is_file():
                    e = e.with_suffix(".hdr")
            found.append(_make(e, r, desc))

    me = classify_dir(root)
    if me:
        return [me]
    walk(root, 1)
    return found


# ================================================================== status
_SUM_CACHE: Dict[tuple, str] = {}


def _cached_sha(path: str) -> Optional[str]:
    try:
        st = os.stat(path)
    except OSError:
        return None
    key = (os.path.abspath(path), st.st_size, int(st.st_mtime))
    if key not in _SUM_CACHE:
        _SUM_CACHE[key] = file_sha256(path)
    return _SUM_CACHE[key]


def status(db: Session, sources: List[Source], progress: Optional[Callable[[int, int, str], None]] = None):
    """Fill .status / .imported_into / .checksum in place and return the list."""
    for i, s in enumerate(sources, 1):
        if progress:
            progress(i, len(sources), s.name)
        if s.status == "error":
            continue
        where = set()
        if os.path.isfile(s.path):
            s.checksum = _cached_sha(s.path)
            rows = (db.query(Library.name, Dataset.name).join(Dataset, Dataset.library_id == Library.id)
                    .join(DataFile, DataFile.dataset_id == Dataset.id).filter(DataFile.checksum == s.checksum).all())
        else:
            rows = (db.query(Library.name, Dataset.name).join(Dataset, Dataset.library_id == Library.id)
                    .join(DataFile, DataFile.dataset_id == Dataset.id)
                    .filter(DataFile.storage_path == str(Path(s.path).resolve())).all())
        where.update(f"{l} / {d}" for l, d in rows)
        lib = db.query(Library).filter(Library.slug == repo.slugify(s.library)).first()
        if lib is not None and db.query(Dataset.id).filter(Dataset.library_id == lib.id,
                                                           Dataset.slug == repo.slugify(s.dataset, "dataset")).first():
            where.add(f"{lib.name} / {s.dataset}")
        s.imported_into = sorted(where)
        s.status = "imported" if where else "new"
    return sources


# ================================================================== import
_OPT_KEYS = set(ReadOptions.__dataclass_fields__) - {"warnings", "reader"}


def import_source(db: Session, s: Source, *, library_name: Optional[str] = None,
                  progress: Optional[Callable[[int, int, str], None]] = None, **opts) -> dict:
    opts = {("category" if k == "csv_category" else k): v for k, v in opts.items()}
    if "include_tir" in opts:
        opts["spectral_range"] = "all" if opts.pop("include_tir") else opts.get("spectral_range", "auto")
    ro = ReadOptions(**{k: v for k, v in opts.items() if k in _OPT_KEYS and v is not None})
    if s.kind == "usgs" and not ro.chapters:
        ro.chapters = "M,S,V,A"
    return import_path(db, s.path, library_name=(library_name or s.library).strip(), dataset_name=s.dataset,
                       options=ro, progress=progress)


def import_many(db: Session, items: List[dict], progress: Optional[Callable[[int, int, str], None]] = None,
                on_item: Optional[Callable[[int, int, Source], None]] = None) -> List[dict]:
    """items = [{"source": Source or dict, "library": name, ...ReadOptions fields}] -> one result per item."""
    out = []
    for k, it in enumerate(items, 1):
        s = it["source"] if isinstance(it["source"], Source) else Source(**{
            f: it["source"][f] for f in Source.__dataclass_fields__ if f in it["source"]})
        if on_item:
            on_item(k, len(items), s)
        opts = {o: v for o, v in it.items() if o not in ("source", "library") and v is not None}
        try:
            r = import_source(db, s, library_name=it.get("library"), progress=progress, **opts)
            out.append({"source": s.name, "ok": True, **r})
        except Exception as e:                                              # noqa: BLE001
            db.rollback()
            out.append({"source": s.name, "ok": False, "error": str(e).splitlines()[0][:300]})
    return out


def import_new(db: Session, folder, progress=None, on_item=None, **opts) -> List[dict]:
    srcs = status(db, scan(folder))
    todo = [{"source": s, "library": s.library, **opts} for s in srcs if s.status == "new"]
    return import_many(db, todo, progress, on_item)


# ================================================================== CLI
def main(argv=None):
    from .database import connect, default_url
    p = argparse.ArgumentParser(description="Scan a local data folder and import new spectral sources (any format)")
    p.add_argument("folder")
    p.add_argument("--import", dest="do_import", action="store_true", help="import every new source")
    p.add_argument("--only", help="comma-separated file/folder names to import (also re-imports 'imported' ones)")
    p.add_argument("--acquisition", choices=repo.ACQUISITION_TYPES, help="tag for spectra without one")
    p.add_argument("--force-acquisition", action="store_true", help="use --acquisition for every spectrum")
    p.add_argument("--limit", type=int, help="at most N spectra per source")
    p.add_argument("--url", default=None, help=f"database URL (default: $DATABASE_URL or {default_url()})")
    a = p.parse_args(argv)
    db = connect(a.url)
    with db.session() as s:
        srcs = status(s, scan(a.folder))
        if not srcs:
            print("No supported spectral data found.")
            return
        w = max(len(x.name) for x in srcs)
        for x in srcs:
            st = x.status if x.status != "imported" else "imported -> " + "; ".join(x.imported_into)
            print(f"{x.name:<{w}}  {kind_label(x.kind)[:34]:<34} -> {x.library:<26} {st}" + (f"  ({x.note})" if x.note else ""))
        if not a.do_import:
            print("\nRun again with --import to load the new ones.")
            return
        only = {o.strip().lower() for o in a.only.split(",")} if a.only else None
        todo = [x for x in srcs if x.status != "error" and ((only and x.name.lower() in only) or
                                                           (not only and x.status == "new"))]
        if not todo:
            print("\nNothing new to import.")
            return
        opts = {k: v for k, v in (("acquisition", a.acquisition), ("limit", a.limit),
                                  ("force_acquisition", a.force_acquisition or None)) if v}
        res = import_many(s, [{"source": x, "library": x.library, **opts} for x in todo],
                          progress=lambda i, n, name: print(f"   [{i}/{n}] {name}") if n and i % 500 == 0 else None,
                          on_item=lambda k, n, x: print(f"[{k}/{n}] {x.name} -> {x.library}"))
        print()
        for r in res:
            print(f"{r['source']}: " + (f"{r['spectra_added']} added, {r.get('skipped', 0)} already there  [{r['detected']}]"
                                        + "".join(f"\n   ! {w}" for w in r.get("warnings", [])[:5])
                                        if r["ok"] else f"FAILED - {r['error']}"))


if __name__ == "__main__":
    main()
