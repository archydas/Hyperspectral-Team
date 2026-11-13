"""Data aggregation: pool ROIs of several scenes per class on a common wavelength grid."""
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from ..database import get_db
from ..services import aggregate as svc
from ds3lib.aggregate.export import to_dict

router = APIRouter(prefix="/api/aggregate", tags=["Aggregation"])


class Roi(BaseModel):
    image_id: str
    class_name: str = Field("class", max_length=200)
    polygon: List[List[float]] = Field(..., min_length=3)


class SaveOpts(BaseModel):
    library: str = Field(..., min_length=1, max_length=200)
    category: Optional[str] = None
    acquisition_type: Optional[str] = None


class AggBody(BaseModel):
    rois: List[Roi] = Field(..., min_length=1, max_length=500)
    mode: str = "intersection"
    step: Optional[float] = None
    mask_bad_bands: bool = True
    save: Optional[SaveOpts] = None


def _compute(body: AggBody, db: Session):
    try:
        items = [r.model_dump() for r in body.rois]
        grid, stats, notes = svc.run(db, items, mode=body.mode, step=body.step, mask_bad_bands=body.mask_bad_bands)
        units = svc.units_of(db, items)
        return grid, stats, notes, units
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.post("")
def aggregate(body: AggBody, db: Session = Depends(get_db)):
    grid, stats, notes, units = _compute(body, db)
    out = to_dict(grid, stats, units[0] if len(units) == 1 else None)
    out.update(units=units, items=notes, mixed_units=len(units) > 1)
    if body.save:
        try:
            out["saved"] = svc.save_to_library(db, grid, stats, library=body.save.library, category=body.save.category,
                                               acquisition=body.save.acquisition_type, unit=units[0] if len(units) == 1 else None)
        except ValueError as e:
            raise HTTPException(400, str(e))
    return out


@router.post("/export/{fmt}")
def export(fmt: str, body: AggBody, db: Session = Depends(get_db)):
    grid, stats, _, units = _compute(body, db)
    try:
        data, media, name = svc.export_bytes(grid, stats, fmt, units[0] if len(units) == 1 else None)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return Response(data, media_type=media, headers={"Content-Disposition": f'attachment; filename="{name}"'})
