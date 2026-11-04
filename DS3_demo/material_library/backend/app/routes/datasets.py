import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..database import get_db
from ..models import Dataset

router = APIRouter(
    prefix="/api",
    tags=["Datasets"]
)


@router.get("/libraries/{library_id}/datasets")
def get_library_datasets(
    library_id: uuid.UUID,
    db: Session = Depends(get_db)
):
    datasets = (
        db.query(Dataset)
        .filter(Dataset.library_id == library_id)
        .all()
    )

    return [
        {
            "id": str(dataset.id),
            "library_id": str(dataset.library_id),
            "name": dataset.name,
            "slug": dataset.slug,
            "description": dataset.description,
            "source_name": dataset.source_name,
            "source_version": dataset.source_version,
            "metadata": dataset.dataset_metadata
        }
        for dataset in datasets
    ]


@router.get("/datasets/{dataset_id}")
def get_dataset(
    dataset_id: uuid.UUID,
    db: Session = Depends(get_db)
):
    dataset = (
        db.query(Dataset)
        .filter(Dataset.id == dataset_id)
        .first()
    )

    if dataset is None:
        raise HTTPException(
            status_code=404,
            detail="Dataset not found"
        )

    return {
        "id": str(dataset.id),
        "library_id": str(dataset.library_id),
        "name": dataset.name,
        "slug": dataset.slug,
        "description": dataset.description,
        "source_name": dataset.source_name,
        "source_version": dataset.source_version,
        "metadata": dataset.dataset_metadata
    }