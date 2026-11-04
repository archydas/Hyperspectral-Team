"""Library functions that work on a whole library: resampling to a new wavelength grid."""
import uuid
from typing import List, Optional

import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from ..database import get_db
from ..models import Library, Material, Sample, Spectrum, Dataset, Image
from ..services import images as imgsvc
from ..services.importer import _get_library, slugify, grid_info
from ..services.resampling import resample_spectrum, without_water_bands, WATER_NM

router = APIRouter(prefix="/api/libraries", tags=["Library functions"])


class ResampleBody(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    mode: str = Field(..., pattern="^(range|image|list)$")
    start: Optional[float] = None
    end: Optional[float] = None
    step: Optional[float] = Field(None, gt=0)
    image_id: Optional[uuid.UUID] = None
    wavelengths: Optional[List[float]] = None
    min_coverage: float = Field(0.5, ge=0.05, le=1.0)
    mask_water: bool = False


def grid_of(body: ResampleBody, db: Session):
    if body.mode == "range":
        if None in (body.start, body.end, body.step) or body.end <= body.start:
            raise HTTPException(400, "give start, end (larger than start) and step in nm")
        n = int(round((body.end - body.start) / body.step)) + 1
        if n < 3 or n > 20000:
            raise HTTPException(400, "the grid must have between 3 and 20000 bands")
        return body.start + body.step * np.arange(n), "range"
    if body.mode == "list":
        w = sorted(set(body.wavelengths or []))
        if len(w) < 3:
            raise HTTPException(400, "give at least 3 wavelengths")
        return np.asarray(w, float), "list"
    img = db.query(Image).filter(Image.id == body.image_id).first()
    if img is None:
        raise HTTPException(404, "image not found")
    cube = imgsvc.open_image(img)
    w = np.asarray(cube.wavelengths_nm, float)[~np.asarray(cube.bad_band_mask, bool)]
    return w, f"image {img.name}"


@router.post("/{library_id}/resample")
def resample(library_id: uuid.UUID, body: ResampleBody, db: Session = Depends(get_db)):
    """New library = every spectrum of this one, linearly resampled to the chosen grid.
    No extrapolation, gaps are not bridged; a spectrum must cover at least `min_coverage` of the grid."""
    src = db.query(Library).filter(Library.id == library_id).first()
    if src is None:
        raise HTTPException(404, "library not found")
    grid, grid_src = grid_of(body, db)
    if body.mask_water:
        grid = without_water_bands(grid)
    if len(grid) < 3:
        raise HTTPException(400, "fewer than 3 grid bands left")
    if body.name.strip().lower() == src.name.lower():
        raise HTTPException(400, "choose a new name; the original library is never overwritten")
    spectra = imgsvc.library_spectra(db, library_id=library_id, limit=50000)
    if not spectra:
        raise HTTPException(400, "the library has no spectra")
    lib = _get_library(db, body.name.strip())
    ds_name = f"resampled from {src.name}"
    ds = Dataset(id=uuid.uuid4(), library_id=lib.id, name=ds_name, slug=slugify(ds_name) + "-" + uuid.uuid4().hex[:6], source_name=src.name,
                 description=f"Linear resampling to {len(grid)} bands ({grid[0]:.1f}-{grid[-1]:.1f} nm), grid from {grid_src}",
                 dataset_metadata={"resampled_from": str(src.id), "grid_source": grid_src, "min_coverage": body.min_coverage})
    db.add(ds); db.flush()
    done, rejected = 0, []
    for sp in spectra:
        try:
            r, cov = resample_spectrum(sp["wavelengths_nm"], sp["values"], grid)
        except ValueError as e:
            rejected.append({"name": sp["material"], "reason": str(e)}); continue
        if cov < body.min_coverage:
            rejected.append({"name": sp["material"], "reason": f"covers only {cov*100:.0f}% of the grid"}); continue
        g2 = grid[np.isfinite(r)]; v2 = r[np.isfinite(r)]
        m = Material(id=uuid.uuid4(), library_id=lib.id, name=sp["material"], category=sp["category"],
                     material_metadata={"resampled_from": src.name})
        db.add(m); db.flush()
        s = Sample(id=uuid.uuid4(), material_id=m.id, dataset_id=ds.id, name=sp["material"], sample_number=1, sample_metadata={})
        db.add(s); db.flush()
        step, uniform = grid_info([float(x) for x in g2])
        db.add(Spectrum(id=uuid.uuid4(), sample_id=s.id, wavelength_start=float(g2[0]), wavelength_end=float(g2[-1]), wavelength_step=step,
                        measurement_unit=sp["unit"], values=[float(x) for x in v2], wavelengths=None if uniform else [float(x) for x in g2],
                        spectrum_metadata={"purity": "resampled", "wavelength_unit": "nm", "point_count": len(v2), "coverage": float(cov)}))
        done += 1
    if done == 0:
        db.rollback()
        raise HTTPException(400, "no spectrum could be resampled: " + (rejected[0]["reason"] if rejected else "empty"))
    db.commit()
    return {"library": lib.name, "library_id": str(lib.id), "resampled": done, "rejected": rejected[:50],
            "n_rejected": len(rejected), "bands": int(len(grid)), "range_nm": [float(grid[0]), float(grid[-1])], "grid_source": grid_src}
