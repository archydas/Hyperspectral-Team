"""Generic importer: any supported file (or folder) -> Library > Dataset > Material > Sample > Spectrum.

Nothing is hard-coded: the file, the library name, the category and the unit all come from the caller.
Re-importing the same file into the same library is detected by SHA-256 and skipped.
"""
from __future__ import annotations
import hashlib
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median

from sqlalchemy.orm import Session

from ..database import SessionLocal, init_db
from ..models import Library, Dataset, Material, Sample, Spectrum, DataFile
from .readers import load_spectra, is_candidate, guess_category, HAVE_DS3
from .taxonomy import normalize_category, normalize_acquisition, infer_acquisition


def slugify(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")
    return s or "item"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def grid_info(wl):
    """(step, is_uniform). Step is only meaningful for an evenly spaced grid."""
    if len(wl) < 2:
        return None, True
    d = [b - a for a, b in zip(wl, wl[1:])]
    step = median(d)
    tol = max(1e-6, 1e-4 * abs(step))
    return (step, True) if all(abs(x - step) <= tol for x in d) else (None, False)


@dataclass
class ImportReport:
    library: str
    files: list = field(default_factory=list)      # dicts: file, status (imported|duplicate|error), spectra, detail

    def as_dict(self):
        return {"library": self.library, "files": self.files,
                "imported": sum(f["spectra"] for f in self.files if f["status"] == "imported"),
                "images": sum(1 for f in self.files if f["status"] == "imported" and f.get("kind") == "image"),
                "skipped": sum(1 for f in self.files if f["status"] == "skipped"),
                "duplicates": sum(1 for f in self.files if f["status"] == "duplicate"),
                "errors": sum(1 for f in self.files if f["status"] == "error")}


def _get_library(db: Session, name: str) -> Library:
    slug = slugify(name)
    lib = db.query(Library).filter(Library.slug == slug).first()
    if lib is None:
        lib = Library(id=uuid.uuid4(), name=name, slug=slug, description=f"Spectral library '{name}'",
                      library_metadata={"type": "spectral"})
        db.add(lib)
        db.flush()
    return lib


def import_file(db: Session, path: Path, library: str, *, dataset: str | None = None, category: str | None = None,
                unit: str | None = None, acquisition: str | None = None) -> dict:
    path = Path(path)
    category = normalize_category(category)
    acquisition = normalize_acquisition(acquisition)
    digest = sha256_of(path)
    lib = _get_library(db, library)
    dup = (db.query(DataFile).join(Dataset, DataFile.dataset_id == Dataset.id)
           .filter(Dataset.library_id == lib.id, DataFile.checksum == digest).first())
    if dup is not None:
        return dict(file=path.name, status="duplicate", spectra=0, detail="already imported into this library")
    records = load_spectra(path, unit=unit)
    if not records:
        raise ValueError(f"{path.name}: no spectra found")

    ds_name = dataset or path.stem
    slug = slugify(ds_name)
    if db.query(Dataset).filter(Dataset.library_id == lib.id, Dataset.slug == slug).first():
        slug = f"{slug}-{digest[:8]}"
    wl0 = records[0]["wavelengths"]
    ds = Dataset(id=uuid.uuid4(), library_id=lib.id, name=ds_name, slug=slug, source_name=path.name,
                 description=f"Imported from {path.name}",
                 dataset_metadata={"row_count": len(records), "wavelength_start": wl0[0], "wavelength_end": wl0[-1],
                                   "wavelength_unit": "nm", "points": len(wl0)})
    db.add(ds)
    db.flush()
    db.add(DataFile(dataset_id=ds.id, filename=path.name, file_type=path.suffix.lstrip(".").lower(),
                    file_size=path.stat().st_size, checksum=digest, storage_path=str(path.resolve()),
                    file_metadata={"spectra": len(records)}))

    mats = {m.name: m for m in db.query(Material).filter(Material.library_id == lib.id).all()}
    n_samples: dict = {}
    for r in records:
        cat = category or normalize_category(r.get("category")) or guess_category(r["name"])
        cat_src = "user" if category else ("provider" if r.get("category") else "keyword_rule")
        m = mats.get(r["name"])
        if m is None:
            m = Material(id=uuid.uuid4(), library_id=lib.id, name=r["name"], category=cat,
                         material_metadata={"source_file": path.name, **({"category_source": cat_src} if cat else {})})
            db.add(m)
            mats[r["name"]] = m
        elif m.category is None and cat:
            m.category = cat
        n_samples[m.id] = n_samples.get(m.id, 0) + 1
        s = Sample(id=uuid.uuid4(), material_id=m.id, dataset_id=ds.id, name=r["name"], sample_number=n_samples[m.id],
                   sample_metadata={"source_file": path.name})
        db.add(s)
        wl = r["wavelengths"]
        step, uniform = grid_info(wl)
        acq = acquisition or normalize_acquisition(r.get("acquisition")) or infer_acquisition(path.name, path.suffix)
        db.add(Spectrum(id=uuid.uuid4(), sample_id=s.id, wavelength_start=wl[0], wavelength_end=wl[-1],
                        wavelength_step=step, measurement_unit=r["unit"], values=r["values"],
                        wavelengths=None if uniform else wl, acquisition_type=acq,
                        spectrum_metadata={"wavelength_unit": "nm", "point_count": len(r["values"]),
                                           "source_file": path.name, **r.get("meta", {})}))
    db.flush()
    return dict(file=path.name, status="imported", spectra=len(records), detail=f"{len(records)} spectra, {len(wl0)} points")


def _walk(p: Path):
    """Files under a folder that may be importable. With ds3: every file except companions (the data file of an
    ENVI image is read through its .hdr; wavelength lists belong to USGS value files)."""
    if not HAVE_DS3:
        return [f for f in sorted(p.rglob("*")) if f.is_file() and is_candidate(f)]
    from ds3.pipeline import SKIP_EXT, CUBE_BIN_EXT, _has_hdr
    out = []
    for f in sorted(p.rglob("*")):
        if not f.is_file() or f.name.startswith(".") or f.suffix.lower() in SKIP_EXT:
            continue
        if "wavelength" in f.name.lower() and f.suffix.lower() == ".txt":
            continue
        if (f.suffix.lower() in CUBE_BIN_EXT | {".sli"}) and _has_hdr(f):
            continue
        out.append(f)
    return out


def _route_file(db: Session, f: Path, lib_name: str, *, dataset, category, unit, reader_kw, acquisition=None) -> dict:
    """spectra file -> library rows; image file -> registered image; anything else -> 'skipped' with the reason."""
    if not HAVE_DS3:
        return import_file(db, f, lib_name, dataset=dataset, category=category, unit=unit, acquisition=acquisition)
    from ds3.pipeline import classify
    from ds3.ingest import IngestError
    try:
        kind = classify(f)
    except IngestError as e:
        return dict(file=f.name, status="skipped", spectra=0, kind=None, detail=str(e))
    if kind == "cube":
        from .images import register_image
        return register_image(db, f, **reader_kw)
    res = import_file(db, f, lib_name, dataset=dataset, category=category, unit=unit, acquisition=acquisition)
    res.setdefault("kind", "spectra")
    return res


def import_path(path, *, library: str | None = None, dataset: str | None = None, category: str | None = None,
                unit: str | None = None, reader_kw: dict | None = None, acquisition: str | None = None) -> ImportReport:
    """File, .zip or folder. Spectra files go into `library` (default: folder / file name); hyperspectral images
    (ENVI .hdr, GeoTIFF/EnMAP, .mat, HDF5, ...) are registered in the Images table."""
    p = Path(path)
    if not p.exists():
        raise ValueError(f"path not found: {p}")
    init_db()
    reader_kw = reader_kw or {}
    lib_name = library or (p.name if p.is_dir() else p.stem)
    if p.is_file() and p.suffix.lower() == ".zip":
        if not HAVE_DS3:
            raise ValueError("zip archives need the ds3 package: pip install -e <path>/ds3-platform")
        from ds3.pipeline import extract_archive, _enmap_only
        folder = extract_archive(p, dest_root=p.parent)
        only = _enmap_only(folder)
        files = only or _walk(folder)
        lib_name = library or p.stem
    elif p.is_file():
        files = [p]
    else:
        files = _walk(p)
    if not files:
        raise ValueError(f"no importable files found in {p}")
    rep = ImportReport(lib_name)
    for f in files:
        db = SessionLocal()
        try:
            res = _route_file(db, f, lib_name, dataset=dataset, category=category, unit=unit, reader_kw=reader_kw, acquisition=acquisition)
            db.commit()
        except Exception as e:
            db.rollback()
            msg = f"{type(e).__name__}: {e}"
            # a stray notes/readme .txt in a folder is not an error of the import: report it as skipped
            soft = len(files) > 1 and f.suffix.lower() in (".txt", ".tsv", ".dat") and ("layout" in msg or "fewer than" in msg)
            res = dict(file=f.name, status="skipped" if soft else "error", spectra=0, detail=msg)
        finally:
            db.close()
        rep.files.append(res)
    if len(files) == 1 and rep.files[0]["status"] in ("error", "skipped"):
        raise ValueError(rep.files[0]["detail"])
    return rep


def pretty_name(stem: str) -> str:
    """ecostress_library_all_spectra -> 'Ecostress library all spectra' (only used when no name is given)."""
    t = re.sub(r"[_\s]+", " ", stem).strip()
    return t[:1].upper() + t[1:] if t else stem


def mark_main(library_name: str, file_name: str) -> None:
    """Flag a library as a MAIN library (loaded from the libraries folder) so the interface can list it apart from
    libraries the user builds. Stored in the library's JSON metadata - no schema change."""
    db = SessionLocal()
    try:
        lib = db.query(Library).filter(Library.slug == slugify(library_name)).first()
        if lib is not None and not (lib.library_metadata or {}).get("main"):
            lib.library_metadata = {**(lib.library_metadata or {}), "main": True, "source_file": file_name}
            db.commit()
    finally:
        db.close()


def import_folder_as_libraries(folder, *, category=None, unit=None) -> list:
    """Every file directly inside `folder` becomes its own library (named after the file). Images go to the Images table.
    Re-running is safe: files already imported (same SHA-256) are reported as duplicates. Never raises for one bad file."""
    folder = Path(folder)
    out = []
    if not folder.is_dir():
        return out
    for f in sorted(p for p in folder.iterdir() if p.is_file() and not p.name.startswith(".") and not p.name.lower().startswith("readme") and p.suffix.lower() not in (".md", ".json", ".py", ".png", ".jpg")):
        try:
            rep = import_path(f, library=pretty_name(f.stem), category=category, unit=unit)
            out.extend(dict(x, library=rep.library) for x in rep.files)
            mark_main(rep.library, f.name)
        except Exception as e:
            out.append(dict(file=f.name, library=pretty_name(f.stem), status="error", spectra=0, detail=f"{type(e).__name__}: {e}"))
    return out
