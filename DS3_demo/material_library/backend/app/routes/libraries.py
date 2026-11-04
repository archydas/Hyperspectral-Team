import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func
from sqlalchemy.orm import Session

from ..database import get_db
from ..models import Library, Material, Sample, Spectrum, Dataset

router = APIRouter(prefix="/api/libraries", tags=["Libraries"])


def _counts(db: Session, library_id):
    mats = db.query(func.count(Material.id)).filter(Material.library_id == library_id).scalar() or 0
    specs = (db.query(func.count(Spectrum.id)).join(Sample, Spectrum.sample_id == Sample.id)
             .join(Dataset, Sample.dataset_id == Dataset.id).filter(Dataset.library_id == library_id).scalar() or 0)
    return mats, specs


def _lib(db, lib):
    mats, specs = _counts(db, lib.id)
    return {"id": str(lib.id), "name": lib.name, "slug": lib.slug, "description": lib.description,
            "metadata": lib.library_metadata, "material_count": mats, "spectrum_count": specs}


@router.get("/")
def get_libraries(db: Session = Depends(get_db)):
    return [_lib(db, l) for l in db.query(Library).order_by(Library.name).all()]


@router.get("/{library_id}")
def get_library(library_id: uuid.UUID, db: Session = Depends(get_db)):
    lib = db.query(Library).filter(Library.id == library_id).first()
    if lib is None:
        raise HTTPException(status_code=404, detail="Library not found")
    return _lib(db, lib)
