import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..database import get_db
from ..models import Sample

router = APIRouter(
    prefix="/api",
    tags=["Samples"]
)


@router.get("/materials/{material_id}/samples")
def get_material_samples(
    material_id: uuid.UUID,
    db: Session = Depends(get_db)
):
    samples = (
        db.query(Sample)
        .filter(Sample.material_id == material_id)
        .order_by(Sample.sample_number)
        .all()
    )

    return [
        {
            "id": str(sample.id),
            "material_id": str(sample.material_id),
            "dataset_id": str(sample.dataset_id),
            "name": sample.name,
            "sample_number": sample.sample_number,
            "metadata": sample.sample_metadata
        }
        for sample in samples
    ]


@router.get("/samples/{sample_id}")
def get_sample(
    sample_id: uuid.UUID,
    db: Session = Depends(get_db)
):
    sample = (
        db.query(Sample)
        .filter(Sample.id == sample_id)
        .first()
    )

    if sample is None:
        raise HTTPException(
            status_code=404,
            detail="Sample not found"
        )

    return {
        "id": str(sample.id),
        "material_id": str(sample.material_id),
        "dataset_id": str(sample.dataset_id),
        "name": sample.name,
        "sample_number": sample.sample_number,
        "metadata": sample.sample_metadata
    }