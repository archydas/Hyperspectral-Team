import uuid

"""Categories, statistics, and importing/deleting libraries through the API."""
import os, re, shutil
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel
from sqlalchemy import func
from sqlalchemy.orm import Session

from ..database import get_db, display_url, PROJECT_ROOT, DATA_HOME
from ..models import Library, Material, Sample, Spectrum, Image
from ..services.importer import import_path

router = APIRouter(prefix="/api", tags=["Admin"])
UPLOAD_DIR = DATA_HOME / "data" / "uploads"


@router.get("/categories")
def categories(db: Session = Depends(get_db)):
    rows = (db.query(Material.category, func.count(Material.id)).filter(Material.category.is_not(None))
            .group_by(Material.category).order_by(Material.category).all())
    return [{"category": c, "materials": n} for c, n in rows]


@router.get("/stats")
def stats(db: Session = Depends(get_db)):
    return {"database": display_url(), "libraries": db.query(func.count(Library.id)).scalar(),
            "materials": db.query(func.count(Material.id)).scalar(), "spectra": db.query(func.count(Spectrum.id)).scalar(),
            "images": db.query(func.count(Image.id)).scalar()}


def _reader_kw(wavelengths, divisor):
    kw = {}
    if wavelengths and wavelengths.strip():
        m = re.fullmatch(r"\s*([0-9.]+)\s*[-\u2013]\s*([0-9.]+)\s*(nm)?\s*", wavelengths)
        if not m:
            raise ValueError("wavelengths: give a range such as 430-860 (nm), for images that carry no wavelengths")
        kw["wavelength_range"] = (float(m.group(1)), float(m.group(2)))
    if divisor and divisor.strip():
        try:
            kw["reflectance_divisor"] = float(divisor)
        except ValueError:
            raise ValueError("reflectance divisor must be a number, e.g. 10000")
    return kw


@router.post("/import")
async def import_upload(files: List[UploadFile] = File(default=[]), file: UploadFile | None = File(None), library: str | None = Form(None), category: str | None = Form(None),
                        unit: str | None = Form(None), wavelengths: str | None = Form(None),
                        reflectance_divisor: str | None = Form(None), acquisition: str | None = Form(None)):
    """Upload one or more files into a library. Accepts spectra (CSV, TXT tables, .sli, .sig, USGS, ECOSTRESS ...),
    hyperspectral images (ENVI .hdr + data file together, GeoTIFF / EnMAP, .mat, HDF5/NetCDF) and .zip archives of either.
    Files are kept in data/uploads/<id>/ because images are read from disk on demand."""
    files = list(files) + ([file] if file is not None else [])      # `file` = the original single-file field
    if not files:
        raise HTTPException(status_code=400, detail="no files received")
    try:
        kw = _reader_kw(wavelengths, reflectance_divisor)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    dest = UPLOAD_DIR / uuid.uuid4().hex[:12]
    dest.mkdir(parents=True, exist_ok=True)
    try:
        saved = []
        for f in files:
            name = Path(f.filename or "upload").name            # never trust a client path
            if not name:
                continue
            with open(dest / name, "wb") as out:
                shutil.copyfileobj(f.file, out)
            saved.append(dest / name)
        if not saved:
            raise ValueError("no files received")
        target = saved[0] if len(saved) == 1 else dest
        rep = import_path(target, library=library or (saved[0].stem if len(saved) == 1 else None), category=category,
                          unit=unit, reader_kw=kw, acquisition=acquisition or None)
        return rep.as_dict()
    except ValueError as e:
        shutil.rmtree(dest, ignore_errors=True)
        raise HTTPException(status_code=400, detail=str(e))


class PathImport(BaseModel):
    paths: List[str]
    library: Optional[str] = None
    category: Optional[str] = None
    unit: Optional[str] = None
    wavelengths: Optional[str] = None
    reflectance_divisor: Optional[str] = None
    acquisition: Optional[str] = None


@router.get("/local")
def local_mode():
    """True when the app runs as a desktop program on the user's own computer (files can then be opened in place)."""
    return {"local": os.environ.get("DS3_LOCAL_APP") == "1"}


@router.post("/import/path")
def import_local_path(body: PathImport):
    """Open files or folders that already exist on this computer WITHOUT copying them (multi-GB scenes).
    Only available in the desktop program (DS3_LOCAL_APP=1); a shared server must use the upload route."""
    if os.environ.get("DS3_LOCAL_APP") != "1":
        raise HTTPException(status_code=403, detail="opening files by path is only available in the desktop program; use upload")
    try:
        kw = _reader_kw(body.wavelengths, body.reflectance_divisor)
        paths = [Path(p).expanduser() for p in body.paths if p and p.strip()]
        if not paths:
            raise ValueError("no file chosen")
        missing = [str(p) for p in paths if not p.exists()]
        if missing:
            raise ValueError("not found: " + ", ".join(missing))
        reports = [import_path(p, library=body.library or p.stem, category=body.category, unit=body.unit,
                               reader_kw=kw, acquisition=body.acquisition or None).as_dict() for p in paths]
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    first = reports[0]
    for r in reports[1:]:
        for k in ("files",):
            first[k] += r[k]
        for k in ("imported", "images", "skipped", "duplicates", "errors"):
            first[k] += r[k]
    return first


@router.delete("/libraries/{library_id}")
def delete_library(library_id: uuid.UUID, db: Session = Depends(get_db)):
    lib = db.query(Library).filter(Library.id == library_id).first()
    if lib is None:
        raise HTTPException(status_code=404, detail="Library not found")
    db.delete(lib)
    db.commit()
    return {"deleted": library_id}
