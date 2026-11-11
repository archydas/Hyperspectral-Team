import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..database import get_db
from ..models import Material, Sample, Spectrum
from ..services.taxonomy import category_filter, NONE_TOKEN

router = APIRouter(
    prefix="/api",
    tags=["Materials"]
)


@router.get("/libraries/{library_id}/materials")
def get_library_materials(
    library_id: uuid.UUID,
    search: str | None = None,
    category: str | None = None,
    category_path: str | None = None,
    acquisition: str | None = None,
    db: Session = Depends(get_db)
):
    query = (
        db.query(Material)
        .filter(Material.library_id == library_id)
    )

    if search:
        query = query.filter(
            Material.name.ilike(f"%{search}%")
        )

    if category:
        query = query.filter(
            Material.category == category
        )

    if category_path:
        query = query.filter(category_filter(Material.category, category_path))
    if acquisition:
        sub = (db.query(Sample.material_id).join(Spectrum, Spectrum.sample_id == Sample.id)
               .filter(Spectrum.acquisition_type.is_(None) if acquisition == NONE_TOKEN
                       else Spectrum.acquisition_type == acquisition))
        query = query.filter(Material.id.in_(sub))

    materials = query.order_by(Material.name).all()
    tags = {}
    if materials:
        rows = (db.query(Sample.material_id, Spectrum.acquisition_type).join(Spectrum, Spectrum.sample_id == Sample.id)
                .filter(Sample.material_id.in_([m.id for m in materials])).distinct().all())
        for mid, a in rows:
            if a:
                tags.setdefault(mid, []).append(a)

    return [
        {
            "id": str(material.id),
            "library_id": str(material.library_id),
            "name": material.name,
            "category": material.category,
            "description": material.description,
            "acquisition": sorted(tags.get(material.id, [])),
            "metadata": material.material_metadata
        }
        for material in materials
    ]


@router.get("/materials/{material_id}")
def get_material(
    material_id: uuid.UUID,
    db: Session = Depends(get_db)
):
    material = (
        db.query(Material)
        .filter(Material.id == material_id)
        .first()
    )

    if material is None:
        raise HTTPException(
            status_code=404,
            detail="Material not found"
        )

    return {
        "id": str(material.id),
        "library_id": str(material.library_id),
        "name": material.name,
        "category": material.category,
        "description": material.description,
        "metadata": material.material_metadata
    }