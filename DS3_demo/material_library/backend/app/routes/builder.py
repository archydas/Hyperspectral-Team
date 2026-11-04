"""Spectral Library Builder: where the output wavelengths come from, reading spectra files, resampling onto the grid."""
import re, uuid
from typing import List, Optional

import numpy as np
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel, Field
from sqlalchemy import func
from sqlalchemy.orm import Session

from ..database import get_db
from ..models import Image, Library, Material, Sample, Spectrum, Dataset
from ..services import images as imgsvc
from ..services.readers import load_spectra
from ..services.resampling import resample_spectrum, without_water_bands
from ._serialize import wavelengths_of

router = APIRouter(prefix="/api/builder", tags=["Library builder"])


# ---- "Data File": what can supply the output wavelengths ---------------------------------------------------------------
@router.get("/sources")
def sources(db: Session = Depends(get_db)):
    """Everything the 'Data File' dialog lists: registered images and spectral libraries, with their wavelength range."""
    out = []
    for im in db.query(Image).order_by(Image.created_at.desc()).all():
        out.append({"kind": "image", "id": str(im.id), "name": im.name, "type": f"Hyperspectral image ({im.format})",
                    "bands": im.bands, "wl_min": im.wl_min_nm, "wl_max": im.wl_max_nm, "unit": im.unit,
                    "info": f"{im.rows} x {im.cols} x {im.bands}"})
    for lib in db.query(Library).order_by(Library.name).all():
        n = (db.query(func.count(Spectrum.id)).join(Sample, Spectrum.sample_id == Sample.id)
             .join(Dataset, Sample.dataset_id == Dataset.id).filter(Dataset.library_id == lib.id).scalar()) or 0
        if not n:
            continue
        sp = (db.query(Spectrum).join(Sample, Spectrum.sample_id == Sample.id).join(Dataset, Sample.dataset_id == Dataset.id)
              .filter(Dataset.library_id == lib.id).first())
        out.append({"kind": "library", "id": str(lib.id), "name": lib.name, "type": "Spectral library", "bands": len(sp.values or []),
                    "wl_min": sp.wavelength_start, "wl_max": sp.wavelength_end, "unit": sp.measurement_unit,
                    "info": f"{n} spectra x {len(sp.values or [])} points (first spectrum)"})
    return out


@router.get("/sources/{kind}/{source_id}/wavelengths")
def source_wavelengths(kind: str, source_id: uuid.UUID, band_from: int | None = None, band_to: int | None = None,
                       db: Session = Depends(get_db)):
    """Wavelengths (nm) of an image (bad bands left out) or of the first spectrum of a library; optional band subset
    (1-based, inclusive) like ENVI's 'Spectral Subset'."""
    if kind == "image":
        im = db.query(Image).filter(Image.id == source_id).first()
        if im is None:
            raise HTTPException(404, "image not found")
        cube = imgsvc.open_image(im)
        w = np.asarray(cube.wavelengths_nm, float)[~np.asarray(cube.bad_band_mask, bool)]
        name = im.name
    elif kind == "library":
        sp = (db.query(Spectrum).join(Sample, Spectrum.sample_id == Sample.id).join(Dataset, Sample.dataset_id == Dataset.id)
              .filter(Dataset.library_id == source_id).first())
        if sp is None:
            raise HTTPException(404, "library has no spectra")
        w = np.asarray(wavelengths_of(sp), float)
        name = db.query(Library.name).filter(Library.id == source_id).scalar()
    else:
        raise HTTPException(400, "kind must be image or library")
    total = len(w)
    lo, hi = (band_from or 1), (band_to or total)
    if not 1 <= lo <= hi <= total:
        raise HTTPException(400, f"band subset must lie within 1-{total}")
    w = w[lo - 1:hi]
    return {"name": name, "wavelengths": [float(x) for x in w], "total_bands": total, "band_from": lo, "band_to": hi}


_NUM = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


@router.post("/wavelengths/ascii")
async def wavelengths_from_ascii(file: UploadFile = File(...), unit: Optional[str] = None):
    """Read output wavelengths from a text file: one number per line, or the first number of every line (extra columns,
    comma/tab/space separated, header or comment lines are skipped). Micrometres are converted to nm (unit=um or auto)."""
    text = (await file.read()).decode("utf-8-sig", errors="replace")
    vals = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line[0] in "#;%/":
            continue
        m = _NUM.match(line)
        if m and (len(line) == m.end() or line[m.end()] in " \t,;"):
            vals.append(float(m.group()))
    if len(vals) < 3:
        raise HTTPException(400, f"{file.filename}: found {len(vals)} wavelengths; need at least 3 (one per line)")
    w = np.asarray(vals)
    u = (unit or "auto").lower()
    if u in ("um", "µm", "micrometers", "micrometres") or (u == "auto" and w.max() < 30):
        w = w * 1000.0
    if np.any(np.diff(w) <= 0):
        raise HTTPException(400, "wavelengths must be strictly increasing")
    return {"name": file.filename, "wavelengths": [float(x) for x in w], "converted_from_um": bool(w.max() != max(vals))}


# ---- add spectra --------------------------------------------------------------------------------------------------------
@router.post("/parse")
async def parse_files(files: List[UploadFile] = File(...)):
    """Read spectra files of any supported spectra format (CSV/TXT tables, .sli, .sig, .asd, .xlsx, .json, USGS, ECOSTRESS ...)
    into plain spectra for the builder. Nothing is stored."""
    import tempfile
    from pathlib import Path
    out, errors = [], []
    with tempfile.TemporaryDirectory() as td:
        for f in files:
            p = Path(td) / Path(f.filename or "upload").name
            p.write_bytes(await f.read())
            try:
                for r in load_spectra(p):
                    out.append({"name": r["name"], "wavelengths": r["wavelengths"], "values": r["values"], "unit": r["unit"],
                                "category": r.get("category"), "source": p.name})
            except (ValueError, Exception) as e:                  # one unreadable file must not stop the others
                errors.append({"file": p.name, "reason": str(e)})
    return {"spectra": out, "errors": errors}


class RawSpectrum(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    wavelengths: List[float]
    values: List[Optional[float]]
    unit: str = "reflectance"
    origin: str = "file"
    source: str = ""


class ResampleBody(BaseModel):
    spectra: List[RawSpectrum] = Field(..., min_length=1, max_length=5000)
    grid: Optional[List[float]] = None          # None = take the grid of the first spectrum ("First Input Spectrum")
    mask_water: bool = False
    min_coverage: float = Field(0.5, ge=0.05, le=1.0)


@router.post("/resample")
def resample(body: ResampleBody):
    """Put spectra on the builder's output grid. Returns the grid (the first spectrum's when none was given), the
    resampled spectra with their coverage, and the ones that were rejected (with the reason)."""
    grid = None
    if body.grid:
        grid = np.asarray(sorted(set(body.grid)), float)
        if len(grid) < 3:
            raise HTTPException(400, "the output grid needs at least 3 wavelengths")
    out, rejected = [], []
    for sp in body.spectra:
        if grid is None:
            pts = sorted((w, v) for w, v in zip(sp.wavelengths, sp.values) if v is not None and np.isfinite(v))
            if len(pts) < 3:
                rejected.append({"name": sp.name, "reason": "fewer than 3 valid points to define the grid"}); continue
            grid = np.asarray([p[0] for p in pts], float)
            if body.mask_water:
                grid = without_water_bands(grid)
        elif body.mask_water:
            grid = without_water_bands(grid)
        try:
            r, cov = resample_spectrum(sp.wavelengths, sp.values, grid)
        except ValueError as e:
            rejected.append({"name": sp.name, "reason": str(e)}); continue
        if cov < body.min_coverage:
            rejected.append({"name": sp.name, "reason": f"covers only {cov * 100:.0f}% of the grid (minimum {body.min_coverage * 100:.0f}%)"}); continue
        out.append({"name": sp.name, "values": [None if not np.isfinite(x) else float(x) for x in r], "coverage": cov,
                    "unit": sp.unit, "origin": sp.origin, "source": sp.source, "n_source": len(sp.wavelengths)})
    return {"grid": None if grid is None else [float(x) for x in grid], "spectra": out, "rejected": rejected}
