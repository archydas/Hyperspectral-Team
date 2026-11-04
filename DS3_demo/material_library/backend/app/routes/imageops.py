"""Analysis on a registered image: PCA, k-means, SAM, band math / indices, anomaly, SNR, unmixing, spectral analyst."""
import uuid
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..database import get_db
from ..models import Image
from ..services import imageops, images as svc

router = APIRouter(prefix="/api/images", tags=["Image analysis"])


class AnalyzeBody(BaseModel):
    params: Dict[str, Any] = {}
    library_id: Optional[uuid.UUID] = None          # endmembers / SAM spectra from a library …
    spectrum_ids: Optional[list] = None             # … or from some of its spectra (collected spectra come in params.endmembers)


@router.get("/analysis/operations")
def operations():
    return {"operations": list(imageops.OPS)}


@router.post("/{image_id}/analyze/{op}")
def analyze(image_id: uuid.UUID, op: str, body: AnalyzeBody, db: Session = Depends(get_db)):
    img = db.query(Image).filter(Image.id == image_id).first()
    if img is None:
        raise HTTPException(status_code=404, detail="Image not found")
    try:
        p = dict(body.params or {})
        if op in ("sam", "unmix") and (body.library_id or body.spectrum_ids):
            sp = svc.library_spectra(db, library_id=body.library_id, spectrum_ids=body.spectrum_ids, limit=60)
            p["endmembers"] = list(p.get("endmembers") or []) + [
                dict(name=s["material"], wavelengths_nm=[float(x) for x in s["wavelengths_nm"]],
                     values=[None if x != x else float(x) for x in s["values"]]) for s in sp]
        return imageops.run(op, img, p)
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"{type(e).__name__}: {e}")
