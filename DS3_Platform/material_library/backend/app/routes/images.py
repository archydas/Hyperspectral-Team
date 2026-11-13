"""Hyperspectral images: list, preview, read a pixel, bring a library into the image (match / classify), save a pixel."""
import uuid
from typing import List, Optional

import shutil, tempfile
from pathlib import Path
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from ..database import get_db
from ..models import Image
from ..services import images as svc

router = APIRouter(prefix="/api/images", tags=["Images"])


def _img(db, image_id):
    img = db.query(Image).filter(Image.id == image_id).first()
    if img is None:
        raise HTTPException(status_code=404, detail="Image not found")
    return img


def _wrap(fn, *a, **k):
    try:
        return fn(*a, **k)
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"{type(e).__name__}: {e}")


def _dict(i: Image):
    return {"id": str(i.id), "name": i.name, "path": i.path, "format": i.format, "rows": i.rows, "cols": i.cols, "bands": i.bands,
            "unit": i.unit, "sensor": i.sensor, "wavelength_min": i.wl_min_nm, "wavelength_max": i.wl_max_nm,
            "notes": (i.image_metadata or {}).get("notes", []),
            "assumed_wavelengths": any("ASSUMED" in n for n in (i.image_metadata or {}).get("notes", []))}


@router.get("/")
def list_images(db: Session = Depends(get_db)):
    return [_dict(i) for i in db.query(Image).order_by(Image.created_at.desc()).all()]


@router.post("/{image_id}/reveal")
def reveal_file(image_id: uuid.UUID, request: Request, db: Session = Depends(get_db)):
    """Show the image file in the system file browser (desktop use only: the request must come from this computer)."""
    import os, subprocess, sys
    from pathlib import Path
    if (request.client.host if request.client else "") not in ("127.0.0.1", "::1", "localhost", "testclient"):
        raise HTTPException(403, "Locating a file works only when the app runs on your own computer")
    img = _img(db, image_id)
    p = Path(img.path)
    if not p.exists():
        raise HTTPException(404, f"The file is not at its stored address any more: {img.path}")
    try:
        if sys.platform.startswith("win"):
            subprocess.Popen(["explorer", "/select,", str(p)])
        elif sys.platform == "darwin":
            subprocess.Popen(["open", "-R", str(p)])
        else:
            subprocess.Popen(["xdg-open", str(p.parent)])
    except Exception as e:                                   # no file browser available
        raise HTTPException(400, f"Could not open the file browser: {e}")
    return {"path": str(p)}


@router.get("/{image_id}")
def get_image(image_id: uuid.UUID, db: Session = Depends(get_db)):
    return _dict(_img(db, image_id))


@router.get("/{image_id}/preview")
def preview(image_id: uuid.UUID, r: Optional[int] = None, g: Optional[int] = None, b: Optional[int] = None,
            alpha: bool = False, max_px: int = 300_000, db: Session = Depends(get_db)):
    """Colour composite as a PNG data URL. r,g,b = 1-based band numbers (all three, or none for true colour).
    max_px = how many pixels may be drawn (the page asks for more when the user zooms in)."""
    bands = [r, g, b] if None not in (r, g, b) else None
    return _wrap(svc.preview, _img(db, image_id), bands, max_px=max(50_000, min(max_px, 6_000_000)), alpha=alpha)


@router.get("/{image_id}/pixel")
def pixel(image_id: uuid.UUID, row: int, col: int, size: int = 1, db: Session = Depends(get_db)):
    return _wrap(lambda: svc.pixel(_img(db, image_id), row, col, max(1, min(size, 25)))[0])


@router.get("/{image_id}/bands")
def bands(image_id: uuid.UUID, db: Session = Depends(get_db)):
    """Band centres (nm) and the bands flagged as bad, for the band-selection boxes."""
    def go():
        cube = svc.open_image(_img(db, image_id))
        return {"wavelengths": [float(x) for x in cube.wavelengths_nm], "bad": [bool(x) for x in cube.bad_band_mask]}
    return _wrap(go)


@router.get("/{image_id}/geo")
def geo(image_id: uuid.UUID, db: Session = Depends(get_db)):
    """Corner coordinates (lat/lon) from the file's own map information, if it has any."""
    return _wrap(svc.geo, _img(db, image_id))


class RoiBody(BaseModel):
    polygon: List[List[float]] = Field(..., min_length=3)     # [[row, col], ...] full-resolution pixel coordinates


@router.post("/{image_id}/roi")
def roi(image_id: uuid.UUID, body: RoiBody, db: Session = Depends(get_db)):
    return _wrap(lambda: svc.roi_spectrum(_img(db, image_id), body.polygon)[0])


class MatchBody(BaseModel):
    row: Optional[int] = None
    col: Optional[int] = None
    polygon: Optional[List[List[float]]] = None
    size: int = 1
    library_id: uuid.UUID
    ignore_unit: bool = False
    top_k: int = Field(10, ge=1, le=100)


@router.post("/{image_id}/match")
def match(image_id: uuid.UUID, body: MatchBody, db: Session = Depends(get_db)):
    def go():
        img = _img(db, image_id)
        _, vals = svc.target(img, row=body.row, col=body.col, size=max(1, min(body.size, 25)), polygon=body.polygon)
        res, note = svc.best_matches(img, vals, svc.library_spectra(db, library_id=body.library_id),
                                     ignore_unit=body.ignore_unit, top_k=body.top_k)
        return {"results": res, "note": note}
    return _wrap(go)


class MapBody(BaseModel):
    library_id: Optional[uuid.UUID] = None
    spectrum_ids: Optional[List[uuid.UUID]] = None
    max_angle_deg: Optional[float] = Field(None, ge=0, le=90)
    limit: int = Field(60, ge=1, le=200)


@router.post("/{image_id}/sam_map")
def sam_map(image_id: uuid.UUID, body: MapBody, db: Session = Depends(get_db)):
    """Classify the image with the chosen library spectra (spectral angle). Returns a transparent PNG overlay + legend."""
    def go():
        img = _img(db, image_id)
        sp = svc.library_spectra(db, library_id=body.library_id, spectrum_ids=[str(i) for i in body.spectrum_ids or []] or None,
                                 limit=body.limit)
        if not sp:
            raise ValueError("the library has no spectra")
        return svc.sam_overlay(img, sp, body.max_angle_deg or None)
    return _wrap(go)


class SaveBody(BaseModel):
    row: Optional[int] = None
    col: Optional[int] = None
    polygon: Optional[List[List[float]]] = None
    size: int = 1
    library: str = Field(..., min_length=1, max_length=200)
    name: str = Field("pixel", min_length=1, max_length=200)
    category: Optional[str] = None


@router.post("/{image_id}/save_pixel")
def save_pixel(image_id: uuid.UUID, body: SaveBody, db: Session = Depends(get_db)):
    def go():
        res = svc.save_pixel(db, _img(db, image_id), row=body.row, col=body.col, size=max(1, min(body.size, 25)),
                             library=body.library, name=body.name, category=body.category, polygon=body.polygon)
        db.commit()
        return res
    return _wrap(go)


@router.post("/{image_id}/labels")
def class_spectra_from_labels(image_id: uuid.UUID, file: UploadFile = File(...), library: Optional[str] = Form(None),
                              db: Session = Depends(get_db)):
    """Upload a ground-truth label map (PaviaU_gt.mat, .npy, .tif; 0 = unlabelled): one mean spectrum per class is stored in a library."""
    def go():
        img = _img(db, image_id)
        tmp = Path(tempfile.mkdtemp())
        try:
            dest = tmp / (Path(file.filename or "labels.mat").name)
            with open(dest, "wb") as f:
                shutil.copyfileobj(file.file, f)
            res = svc.save_label_spectra(db, img, dest, library or f"{img.name} classes")
            db.commit()
            return res
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    return _wrap(go)


@router.delete("/{image_id}")
def delete_image(image_id: uuid.UUID, db: Session = Depends(get_db)):
    img = _img(db, image_id)
    svc.forget(img.id)
    db.delete(img); db.commit()
    return {"deleted": str(image_id), "note": "the image file itself is not deleted"}
