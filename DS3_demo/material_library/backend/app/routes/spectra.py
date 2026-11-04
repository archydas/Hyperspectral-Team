import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..database import get_db
from ..models import Spectrum, Sample, Material, Dataset, Library
from ._serialize import spectrum_dict
from ..services.taxonomy import category_filter, normalize_acquisition, normalize_category, NONE_TOKEN

router = APIRouter(prefix="/api", tags=["Spectra"])


@router.get("/samples/{sample_id}/spectra")
def get_sample_spectra(sample_id: uuid.UUID, db: Session = Depends(get_db)):
    return [spectrum_dict(s) for s in db.query(Spectrum).filter(Spectrum.sample_id == sample_id).all()]


@router.get("/spectra/search")
def search_spectra(q: str | None = None, category: str | None = None, category_path: str | None = None,
                   acquisition: str | None = None, library_id: uuid.UUID | None = None,
                   unit: str | None = None, limit: int = 50, offset: int = 0, db: Session = Depends(get_db)):
    """Search across ALL libraries by material name / category / library, without loading the values."""
    limit = max(1, min(limit, 500))
    qry = (db.query(Spectrum, Material, Library)
           .join(Sample, Spectrum.sample_id == Sample.id)
           .join(Material, Sample.material_id == Material.id)
           .join(Dataset, Sample.dataset_id == Dataset.id)
           .join(Library, Dataset.library_id == Library.id))
    if q:
        qry = qry.filter(Material.name.ilike(f"%{q}%") | Material.category.ilike(f"%{q}%") | Library.name.ilike(f"%{q}%"))
    if category:
        qry = qry.filter(Material.category == category)
    if category_path:                                   # this node and everything below it
        qry = qry.filter(category_filter(Material.category, category_path))
    if acquisition:
        qry = qry.filter(Spectrum.acquisition_type.is_(None) if acquisition == NONE_TOKEN
                         else Spectrum.acquisition_type == acquisition)
    if library_id:
        qry = qry.filter(Library.id == library_id)
    if unit:
        qry = qry.filter(Spectrum.measurement_unit == unit)
    total = qry.count()
    rows = qry.order_by(Library.name, Material.name).limit(limit).offset(offset).all()
    return {"total": total, "limit": limit, "offset": offset, "results": [
        {"spectrum_id": str(sp.id), "sample_id": str(sp.sample_id), "material_id": str(m.id), "material": m.name,
         "category": m.category, "acquisition_type": sp.acquisition_type, "library_id": str(l.id), "library": l.name, "unit": sp.measurement_unit,
         "wavelength_start": sp.wavelength_start, "wavelength_end": sp.wavelength_end,
         "n_points": len(sp.values or [])} for sp, m, l in rows]}


@router.get("/spectra/{spectrum_id}")
def get_spectrum(spectrum_id: uuid.UUID, db: Session = Depends(get_db)):
    sp = db.query(Spectrum).filter(Spectrum.id == spectrum_id).first()
    if sp is None:
        raise HTTPException(status_code=404, detail="Spectrum not found")
    return spectrum_dict(sp)


# ---- save spectra built in the page (Spectral Library Builder) -------------------------------------------------
import uuid as _uuid
from typing import List, Optional
from pydantic import BaseModel, Field


class SaveItem(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    wavelengths_nm: List[float] = Field(..., min_length=3)
    values: List[Optional[float]]
    unit: str = "reflectance"
    category: Optional[str] = None
    origin: str = "builder"
    acquisition_type: Optional[str] = None


class SaveBody(BaseModel):
    library: str = Field(..., min_length=1, max_length=200)
    items: List[SaveItem] = Field(..., min_length=1, max_length=2000)


@router.post("/spectra/save")
def save_spectra(body: SaveBody, db: Session = Depends(get_db)):
    """Store spectra made in the page (builder, browser library) in a database library. Gaps (null) are dropped, not invented."""
    from ..services.importer import _get_library, slugify, grid_info
    lib = _get_library(db, body.library.strip())
    ds_name = "built in the interface"
    ds = db.query(Dataset).filter(Dataset.library_id == lib.id, Dataset.slug == slugify(ds_name)).first()
    if ds is None:
        ds = Dataset(id=_uuid.uuid4(), library_id=lib.id, name=ds_name, slug=slugify(ds_name), source_name="interface",
                     description="Spectra saved from the Library Builder", dataset_metadata={"origin": "builder"})
        db.add(ds); db.flush()
    saved, rejected = 0, []
    for it in body.items:
        if len(it.values) != len(it.wavelengths_nm):
            rejected.append({"name": it.name, "reason": "values and wavelengths differ in length"}); continue
        pts = [(w, v) for w, v in zip(it.wavelengths_nm, it.values) if v is not None and v == v]
        if len(pts) < 3:
            rejected.append({"name": it.name, "reason": "fewer than 3 valid points"}); continue
        if any(b[0] <= a[0] for a, b in zip(pts, pts[1:])):
            rejected.append({"name": it.name, "reason": "wavelengths must be strictly increasing"}); continue
        w = [p[0] for p in pts]; v = [p[1] for p in pts]
        try:
            acq = normalize_acquisition(it.acquisition_type)
        except ValueError as e:
            rejected.append({"name": it.name, "reason": str(e)}); continue
        mat = Material(id=_uuid.uuid4(), library_id=lib.id, name=it.name.strip(), category=normalize_category(it.category),
                       material_metadata={"origin": it.origin})
        db.add(mat); db.flush()
        s = Sample(id=_uuid.uuid4(), material_id=mat.id, dataset_id=ds.id, name=mat.name, sample_number=1, sample_metadata={})
        db.add(s); db.flush()
        step, uniform = grid_info(w)
        db.add(Spectrum(id=_uuid.uuid4(), sample_id=s.id, wavelength_start=w[0], wavelength_end=w[-1], wavelength_step=step,
                        measurement_unit=it.unit, values=v, wavelengths=None if uniform else w, acquisition_type=acq,
                        spectrum_metadata={"purity": it.origin, "wavelength_unit": "nm", "point_count": len(v)}))
        saved += 1
    db.commit()
    return {"library": lib.name, "library_id": str(lib.id), "saved": saved, "rejected": rejected}
