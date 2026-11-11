"""Category tree, acquisition tags and bulk edits."""
import uuid
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import func
from sqlalchemy.orm import Session

from ..database import get_db
from ..models import Material, Sample, Spectrum, Dataset
from ..services.taxonomy import (ACQUISITION_TYPES, NONE_TOKEN, normalize_acquisition, normalize_category,
                                 category_filter, rollup)

router = APIRouter(prefix="/api", tags=["Categories and acquisition"])


@router.get("/facets")
def facets(library_id: uuid.UUID | None = None, db: Session = Depends(get_db)):
    """Category tree with counts (spectra per node, including everything below it) and counts per acquisition type."""
    q = (db.query(Material.category, Spectrum.acquisition_type, func.count(Spectrum.id))
         .join(Sample, Sample.material_id == Material.id).join(Spectrum, Spectrum.sample_id == Sample.id))
    if library_id:
        q = q.filter(Material.library_id == library_id)
    rows = q.group_by(Material.category, Spectrum.acquisition_type).all()
    tree = rollup([(c, n) for c, _, n in rows])
    acq: dict = {}
    for _, a, n in rows:
        acq[a or NONE_TOKEN] = acq.get(a or NONE_TOKEN, 0) + n
    return {"categories": [{"path": k, "count": v} for k, v in sorted(tree.items())],
            "acquisition": acq, "acquisition_types": list(ACQUISITION_TYPES)}


class BulkMaterials(BaseModel):
    material_ids: List[uuid.UUID] = Field(..., min_length=1, max_length=5000)
    category: Optional[str] = None
    set_category: bool = True
    acquisition_type: Optional[str] = None
    set_acquisition: bool = False


@router.patch("/materials")
def edit_materials(body: BulkMaterials, db: Session = Depends(get_db)):
    """Set the category of several materials (and optionally the acquisition tag of all their spectra) at once."""
    try:
        acq = normalize_acquisition(body.acquisition_type) if body.set_acquisition else None
    except ValueError as e:
        raise HTTPException(400, str(e))
    mats = db.query(Material).filter(Material.id.in_(body.material_ids)).all()
    for m in mats:
        if body.set_category:
            m.category = normalize_category(body.category)
            m.material_metadata = {**(m.material_metadata or {}), "category_source": "user"}
    n_sp = 0
    if body.set_acquisition and mats:
        ids = [m.id for m in mats]
        sids = [s.id for s in db.query(Sample.id).filter(Sample.material_id.in_(ids)).all()]
        n_sp = db.query(Spectrum).filter(Spectrum.sample_id.in_(sids)).update({Spectrum.acquisition_type: acq}, synchronize_session=False)
    db.commit()
    return {"materials": len(mats), "spectra_tagged": n_sp}


class SpectrumPatch(BaseModel):
    acquisition_type: Optional[str] = None


@router.patch("/spectra/{spectrum_id}")
def edit_spectrum(spectrum_id: uuid.UUID, body: SpectrumPatch, db: Session = Depends(get_db)):
    sp = db.query(Spectrum).filter(Spectrum.id == spectrum_id).first()
    if sp is None:
        raise HTTPException(404, "Spectrum not found")
    try:
        sp.acquisition_type = normalize_acquisition(body.acquisition_type)
    except ValueError as e:
        raise HTTPException(400, str(e))
    db.commit()
    return {"id": str(sp.id), "acquisition_type": sp.acquisition_type}


class RenameBody(BaseModel):
    old: str = Field(..., min_length=1)
    new: Optional[str] = None
    library_id: Optional[uuid.UUID] = None


@router.post("/categories/rename")
def rename_category(body: RenameBody, db: Session = Depends(get_db)):
    """Rename or move a category node; every material below it moves with it. Empty `new` = remove the category."""
    old = normalize_category(body.old)
    new = normalize_category(body.new)
    if not old:
        raise HTTPException(400, "old category is empty")
    q = db.query(Material).filter(category_filter(Material.category, old))
    if body.library_id:
        q = q.filter(Material.library_id == body.library_id)
    n = 0
    for m in q.all():
        rest = m.category[len(old):]                           # '' or '/child/...'
        m.category = normalize_category((new or "") + rest) if (new or rest) else None
        n += 1
    db.commit()
    return {"materials": n, "old": old, "new": new}
