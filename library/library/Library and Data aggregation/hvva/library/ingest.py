"""One import path for every format: detect -> read -> normalise -> write.

    python -m hvva.library.ingest FILE_OR_FOLDER [...] --library "My Library" [options]
    python -m hvva.library.ingest leaves.csv --preview                 # what would be imported, nothing written

Options (all optional, everything is auto-detected otherwise):
    --unit nm|um|cm-1            wavelength unit of the file
    --scale unit|percent|x10000|none   value scale (none = keep as measured, e.g. radiance)
    --orientation columns|rows|long    table layout
    --range all|vswir            keep everything, or only 350-2500 nm
    --category PATH              category for spectra that have none (Vegetation/Tree/...)
    --acquisition lab|field|airborne|satellite [--force-acquisition]
    --image-mode classes|mean|sample|all  --samples N     (hyperspectral images)
    --wl-range START STOP        files without wavelengths (nm)
    --wavelength-column / --name-column / --category-column   table column overrides
    --limit N                    at most N spectra per source
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
from sqlalchemy.orm import Session

from . import repository as repo
from .formats import ReadOptions, SpectrumRecord, detect, read_path
from .formats.refcollections import _has_ecostress, _walk_files
from .models import DataFile, Library, Material, Sample, Spectrum


def default_dataset_name(path, reader_name: str) -> str:
    p = Path(path)
    if reader_name == "openspeclib":
        return f"OpenSpecLib {p.stem}"
    if reader_name == "usgs":
        return f"USGS Spectral Library v7 ({p.name})"
    if reader_name in ("folder", "zip"):
        try:
            names = _walk_files(p, 2000) if p.is_dir() else __import__("zipfile").ZipFile(p).namelist()[:2000]
        except Exception:                                                   # noqa: BLE001
            names = []
        if _has_ecostress(names):
            return f"ECOSTRESS spectral library ({p.name})"
    return re.sub(r"[_\s]+", " ", p.stem if p.is_file() else p.name).strip().title() or "Imported spectra"


def file_sha256(path) -> Optional[str]:
    from .speclib import file_sha256 as _f
    return _f(path)


def write_records(db: Session, records: List[SpectrumRecord], *, library_id=None, library_name=None,
                  dataset_name: str, source_label: str, provenance=None, kind: Optional[str] = None,
                  progress: Optional[Callable[[int, int, str], None]] = None) -> dict:
    """Library -> Dataset -> (DataFile) -> Material -> Sample -> Spectrum; skips spectra already in the dataset."""
    try:
        if library_id is not None:
            lib = db.query(Library).filter(Library.id == repo._uuid(library_id, "Library")).first()
            if lib is None:
                raise repo.NotFound("Library not found")
        else:
            library_name = (library_name or "").strip()
            if not library_name:
                raise ValueError("library name is required")
            lib = db.query(Library).filter(Library.slug == repo.slugify(library_name)).first()
            if lib is None:
                lib = Library(name=library_name[:255], slug=repo._unique_slug(db, repo.slugify(library_name)),
                              description=f"Spectra imported from {source_label}.",
                              library_metadata={"type": "spectral", "source": source_label})
                db.add(lib)
                db.flush()
        ds = repo.get_or_create_dataset(db, lib.id, dataset_name, description=f"Imported from {source_label}.",
                                        source_name=source_label[:500], commit=False)
        if provenance is not None:
            prov = Path(provenance)
            cs = file_sha256(prov)
            spath = str(prov.resolve())
            q = db.query(DataFile).filter(DataFile.dataset_id == ds.id)
            q = q.filter(DataFile.checksum == cs) if cs else q.filter(DataFile.storage_path == spath)
            if q.first() is None:
                db.add(DataFile(dataset_id=ds.id, filename=prov.name[:500],
                                file_type=(prov.suffix.lstrip(".") or "file") if prov.is_file() else "folder",
                                file_size=prov.stat().st_size if prov.is_file() else None, checksum=cs,
                                storage_path=spath, file_metadata={"source": source_label, "kind": kind}))
        # duplicate check over the whole library (not just this dataset): re-importing a file under another
        # dataset name, or a second copy of the same file, must not add the same spectra twice
        have = {(m or {}).get("uid") for (m,) in db.query(Spectrum.spectrum_metadata)
                .join(Sample, Sample.id == Spectrum.sample_id).join(Material, Material.id == Sample.material_id)
                .filter(Material.library_id == lib.id)}
        added = skipped = 0
        cats = set()
        for i, r in enumerate(records, 1):
            if progress and (i % 25 == 0 or i == len(records)):
                progress(i, len(records), r.name)
            uid = (r.spectrum_metadata or {}).get("uid")
            if uid and uid in have:
                skipped += 1
                continue
            have.add(uid)
            repo.add_spectrum(db, lib.id, ds.id, r.name, r.wavelengths, r.values, category=r.category, std=r.std,
                              material_metadata=_jsonable(r.material_metadata),
                              spectrum_metadata=_jsonable(r.spectrum_metadata), commit=False,
                              acquisition=r.acquisition)
            if r.category:
                cats.add(r.category)
            added += 1
            if added % 200 == 0:
                db.flush()
        db.commit()
    except Exception:
        db.rollback()
        raise
    return {"library_id": str(lib.id), "library_name": lib.name, "dataset_name": ds.name, "found": len(records),
            "spectra_added": added, "skipped": skipped, "categories": len(cats)}


def _jsonable(d):
    out = {}
    for k, v in (d or {}).items():
        if isinstance(v, (np.floating, np.integer)):
            v = v.item()
        elif isinstance(v, np.ndarray):
            v = v.tolist()
        elif isinstance(v, (list, tuple)):
            v = [x.item() if isinstance(x, (np.floating, np.integer)) else x for x in v]
        elif isinstance(v, float) and not np.isfinite(v):
            v = None
        elif not isinstance(v, (str, int, float, bool, list, dict, type(None))):
            v = str(v)
        out[str(k)] = v
    return out


def import_path(db: Session, path, *, library_id=None, library_name=None, dataset_name=None,
                options: Optional[ReadOptions] = None,
                progress: Optional[Callable[[int, int, str], None]] = None) -> dict:
    """Any supported file / folder / zip -> library. Returns counts plus what was detected."""
    opts = options or ReadOptions()
    p = Path(path)
    recs, info = read_path(p, opts, progress)
    ds = dataset_name or default_dataset_name(p, info.reader)
    if library_id is None and not library_name:
        library_name = ds
    label = f"{info.label}: {p.name}"
    res = write_records(db, recs, library_id=library_id, library_name=library_name, dataset_name=ds,
                        source_label=label, provenance=p, kind=info.reader, progress=progress)
    res.update({"reader": info.reader, "reader_label": info.label, "detected": info.summary(),
                "warnings": info.warnings[:50], "source": p.name})
    return res


def preview_path(path, options: Optional[ReadOptions] = None, n_spectra: int = 12) -> dict:
    """What an import would produce, without writing: detection, units, scale, sample spectra."""
    opts = options or ReadOptions()
    recs, info = read_path(Path(path), opts)
    return {"reader": info.reader, "reader_label": info.label, "summary": info.summary(), "n": len(recs),
            "found": info.n_found, "wl_unit": info.wl_unit, "scale": info.scale, "wl_min": info.wl_min,
            "wl_max": info.wl_max, "categories": info.categories[:200], "acquisitions": info.acquisitions,
            "warnings": info.warnings[:50],
            "dataset": default_dataset_name(path, info.reader),
            "sample": [{"name": r.name, "category": r.category, "acquisition": r.acquisition,
                        "wavelengths": np.asarray(r.wavelengths).tolist(),
                        "values": [None if not np.isfinite(v) else float(v) for v in r.values],
                        "std": None if r.std is None else [None if not np.isfinite(v) else float(v) for v in r.std],
                        "points": int(np.asarray(r.values).size)} for r in recs[:n_spectra]]}


def options_from_args(a) -> ReadOptions:
    return ReadOptions(wavelength_unit=a.unit, value_scale=a.scale, orientation=a.orientation,
                       spectral_range=a.range, category=a.category, acquisition=a.acquisition,
                       force_acquisition=a.force_acquisition, image_mode=a.image_mode, image_samples=a.samples,
                       wl_range=tuple(a.wl_range) if a.wl_range else None, wavelength_column=a.wavelength_column,
                       name_column=a.name_column, category_column=a.category_column, limit=a.limit,
                       reader=a.reader)


def add_option_args(p):
    p.add_argument("--unit", default="auto", choices=["auto", "nm", "um", "cm-1"])
    p.add_argument("--scale", default="auto", choices=["auto", "unit", "percent", "x10000", "none"])
    p.add_argument("--orientation", default="auto", choices=["auto", "columns", "rows", "long"])
    p.add_argument("--range", default="auto", choices=["auto", "all", "vswir"])
    p.add_argument("--category")
    p.add_argument("--acquisition", choices=repo.ACQUISITION_TYPES)
    p.add_argument("--force-acquisition", action="store_true")
    p.add_argument("--image-mode", default="auto", choices=["auto", "classes", "mean", "sample", "all"])
    p.add_argument("--samples", type=int, default=100)
    p.add_argument("--wl-range", type=float, nargs=2, metavar=("START", "STOP"))
    p.add_argument("--wavelength-column")
    p.add_argument("--name-column")
    p.add_argument("--category-column")
    p.add_argument("--limit", type=int)
    p.add_argument("--reader", help="force a reader (table, excel, json, parquet, envi_sli, image, arrays, asd, sig, "
                                    "sed, usgs, openspeclib, folder, zip)")


def main(argv=None):
    from .database import connect, default_url
    p = argparse.ArgumentParser(description="Import any supported spectral data into the material library")
    p.add_argument("paths", nargs="+")
    p.add_argument("--library", help="target library (created if missing; default: named after each source)")
    p.add_argument("--dataset", help="dataset name (default: from each source)")
    p.add_argument("--preview", action="store_true", help="only show what would be imported")
    p.add_argument("--url", default=None, help=f"database URL (default: $DATABASE_URL or {default_url()})")
    add_option_args(p)
    a = p.parse_args(argv)
    if a.preview:
        for path in a.paths:
            try:
                pv = preview_path(path, options_from_args(a), 5)
            except Exception as e:                                          # noqa: BLE001
                print(f"{path}: cannot read - {e}")
                continue
            print(f"{path}\n  {pv['summary']}\n  dataset: {pv['dataset']}   acquisition: {pv['acquisitions']}")
            for s in pv["sample"]:
                print(f"    {s['name'][:50]:<50} {s['points']:>6} pts  {s['category'] or ''}  {s['acquisition'] or ''}")
            for w in pv["warnings"][:10]:
                print(f"  ! {w}")
        return
    db = connect(a.url)
    for path in a.paths:
        with db.session() as s:
            try:
                r = import_path(s, path, library_name=a.library, dataset_name=a.dataset,
                                options=options_from_args(a),
                                progress=lambda i, n, name: print(f"   [{i}/{n}] {name}") if n and i % 250 == 0 else None)
            except Exception as e:                                          # noqa: BLE001
                print(f"{path}: FAILED - {e}")
                continue
        print(f"{path}: {r['detected']}\n   -> {r['library_name']} / {r['dataset_name']}: {r['spectra_added']} added, "
              f"{r['skipped']} already there" + "".join(f"\n   ! {w}" for w in r["warnings"][:10]))


if __name__ == "__main__":
    main()
