"""Import a wide spectral CSV (one wavelength column + one column per material) into the library database.

Generalised from new_hyperspectral/material_library/backend/app/services/csv_importer.py:
same get-or-create logic (Library -> Dataset -> DataFile provenance -> Material -> Sample -> Spectrum) and the
same idempotency (re-importing a file adds nothing that is already there), but

* library, dataset, file and category are parameters instead of hard-coded constants;
* metadata is written to the real JSON columns (the original passed ``metadata=`` to the model constructors,
  which SQLAlchemy silently treats as the reserved ``Base.metadata`` attribute, so it was never stored);
* no pandas dependency (csv + numpy); non-numeric cells become NULL;
* a progress callback lets the GUI show a progress bar.

Command line:
    python -m hvva.library.importer data/material_spectra.csv --library "Material Library" --dataset "Material Spectra"
"""
from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from sqlalchemy.orm import Session

from . import repository as repo
from .models import DataFile, Library, Material, Sample, Spectrum


def checksum(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def _num(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return np.nan


def read_wide_csv(path, wavelength_column: Optional[str] = None):
    """Return (wavelengths, {column: values}) from a wide CSV."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))
    if len(rows) < 3:
        raise ValueError("CSV must contain a header and at least two wavelength rows")
    header = [h.strip() for h in rows[0]]
    if wavelength_column is None:
        wl_idx = next((i for i, h in enumerate(header) if h.lower().startswith("wavelength")), 0)
    else:
        if wavelength_column not in header:
            raise ValueError(f"CSV must contain a '{wavelength_column}' column")
        wl_idx = header.index(wavelength_column)
    body = [r for r in rows[1:] if any(c.strip() for c in r)]
    data = np.array([[_num(c) for c in (r + [""] * (len(header) - len(r)))[:len(header)]] for r in body], float)
    wl = data[:, wl_idx]
    keep = ~np.isnan(wl)
    data, wl = data[keep], wl[keep]
    if wl.size < 2:
        raise ValueError("CSV must contain at least two numeric wavelength values")
    order = np.argsort(wl)
    cols = {h: data[order, i] for i, h in enumerate(header) if i != wl_idx and h}
    return wl[order], cols


def import_wide_csv(db: Session, path, *, library_id=None, library_name: Optional[str] = None,
                    library_description: Optional[str] = None, dataset_name: Optional[str] = None,
                    category: Optional[str] = None, wavelength_column: Optional[str] = None,
                    wavelength_unit="nm", progress: Optional[Callable[[int, int, str], None]] = None,
                    acquisition: Optional[str] = None):
    """category: tree path for new materials (Vegetation/Tree/Tsuga); acquisition: lab|field|airborne|satellite
    for every imported spectrum (spectra already in the database keep their tag)."""
    category = repo.normalize_category(category)
    acquisition = repo.normalize_acquisition(acquisition)
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"CSV file not found: {path}")
    wl, cols = read_wide_csv(path, wavelength_column)
    if not cols:
        raise ValueError("no material columns next to the wavelength column")
    start, end, step, explicit = repo._grid(wl)
    try:
        # 1. library ----------------------------------------------------------------
        if library_id is not None:
            lib = db.query(Library).filter(Library.id == repo._uuid(library_id, "Library")).first()
            if lib is None:
                raise repo.NotFound("Library not found")
        else:
            library_name = (library_name or path.stem.replace("_", " ").title()).strip()
            lib = db.query(Library).filter(Library.slug == repo.slugify(library_name)).first()
            if lib is None:
                lib = Library(name=library_name, slug=repo._unique_slug(db, repo.slugify(library_name)),
                              description=library_description or "Library containing spectral measurements of material samples.",
                              library_metadata={"type": "spectral"})
                db.add(lib)
                db.flush()
        # 2. dataset ----------------------------------------------------------------
        ds = repo.get_or_create_dataset(
            db, lib.id, dataset_name or path.stem.replace("_", " ").title(),
            description=f"Spectral measurements imported from {path.name}.", source_name=path.name,
            source_version="1.0", commit=False,
            metadata={"row_count": int(wl.size), "column_count": len(cols) + 1, "wavelength_start": start,
                      "wavelength_end": end, "wavelength_step": step, "wavelength_unit": wavelength_unit})
        # 3. provenance -------------------------------------------------------------
        cs = checksum(path)
        if db.query(DataFile).filter(DataFile.dataset_id == ds.id, DataFile.filename == path.name,
                                     DataFile.checksum == cs).first() is None:
            db.add(DataFile(dataset_id=ds.id, filename=path.name, file_type=path.suffix.lstrip(".").lower() or "csv",
                            file_size=path.stat().st_size, checksum=cs, storage_path=str(path.resolve()),
                            file_metadata={"columns": len(cols) + 1, "rows": int(wl.size)}))
            db.flush()
        # 4. one material / sample / spectrum per column -----------------------------
        added = 0
        n = len(cols)
        for i, (name, values) in enumerate(cols.items(), 1):
            if progress:
                progress(i, n, name)
            mat = db.query(Material).filter(Material.library_id == lib.id, Material.name == name).first()
            if mat is None:
                mat = Material(library_id=lib.id, name=name[:500], category=category or None,
                               material_metadata={"source_column": name, "source_file": path.name})
                db.add(mat)
                db.flush()
            smp = db.query(Sample).filter(Sample.material_id == mat.id, Sample.dataset_id == ds.id).first()
            if smp is None:
                nxt = (db.query(Sample.sample_number).filter(Sample.material_id == mat.id)
                       .order_by(Sample.sample_number.desc()).limit(1).scalar() or 0) + 1
                smp = Sample(material_id=mat.id, dataset_id=ds.id, name=name, sample_number=nxt,
                             sample_metadata={"source_column": name})
                db.add(smp)
                db.flush()
            if db.query(Spectrum.id).filter(Spectrum.sample_id == smp.id).first() is None:
                meta = {"wavelength_unit": wavelength_unit, "source_column": name, "point_count": int(values.size)}
                if explicit is not None:
                    meta["wavelengths"] = explicit
                db.add(Spectrum(sample_id=smp.id, wavelength_start=start, wavelength_end=end, wavelength_step=step,
                                measurement_unit=None, values=repo._clean(values), spectrum_metadata=meta,
                                acquisition_type=acquisition))
                added += 1
        db.commit()
    except Exception:
        db.rollback()
        raise
    return {"library_id": str(lib.id), "library_name": lib.name, "dataset_id": str(ds.id), "dataset_name": ds.name,
            "materials": len(cols), "spectra_added": added, "points": int(wl.size), "range": (start, end), "step": step}


def main(argv=None):
    from .database import connect, default_url
    p = argparse.ArgumentParser(description="Import a wide spectral CSV into the material library database")
    p.add_argument("csv")
    p.add_argument("--url", default=None, help=f"database URL (default: $DATABASE_URL or {default_url()})")
    p.add_argument("--library", help="library name (created if missing; default: from file name)")
    p.add_argument("--dataset", help="dataset name (default: from file name)")
    p.add_argument("--category", help="category path for new materials, e.g. 'Vegetation/Tree/Tsuga'")
    p.add_argument("--acquisition", choices=repo.ACQUISITION_TYPES, help="tag every imported spectrum")
    p.add_argument("--wavelength-column", default=None)
    p.add_argument("--init", action="store_true", help="create the tables first if the database is empty")
    a = p.parse_args(argv)
    db = connect(a.url, init=a.init)
    with db.session() as s:
        r = import_wide_csv(s, a.csv, library_name=a.library, dataset_name=a.dataset, category=a.category,
                            acquisition=a.acquisition,
                            wavelength_column=a.wavelength_column,
                            progress=lambda i, n, name: print(f"[{i}/{n}] {name}"))
    print("\n======================================\nCSV IMPORT COMPLETED\n======================================")
    print(f"Library  : {r['library_name']}\nDataset  : {r['dataset_name']}\nMaterials: {r['materials']} "
          f"({r['spectra_added']} new spectra)\nPoints   : {r['points']}\nRange    : {r['range'][0]} - {r['range'][1]}"
          f"\nStep     : {r['step']}")


if __name__ == "__main__":
    main()
