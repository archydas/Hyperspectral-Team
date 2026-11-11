"""REST API. Run:  ds3 serve   (docs at http://127.0.0.1:8000/docs)"""
from __future__ import annotations
import shutil
from contextlib import asynccontextmanager
from pathlib import Path
import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File, Query, Depends
from pydantic import BaseModel
from sqlalchemy import select, func
from sqlalchemy.orm import Session
from .. import __version__
from ..db import init_db, get_engine
from ..db.models import CubeRecord, SpectrumRecord, Sample, Material, Dataset, Library, Source
from ..ingest import open_cube, extract_roi, IngestError
from ..ingest.contract import CONTRACT_VERSION
from ..pipeline import ingest_path

@asynccontextmanager
async def _lifespan(_):
    init_db(); yield


app = FastAPI(title="DS3 - hyperspectral aggregation", version=__version__, lifespan=_lifespan)





def db():
    from sqlalchemy.orm import sessionmaker
    s = sessionmaker(get_engine(), expire_on_commit=False)()
    try: yield s
    finally: s.close()


def _cube(s: Session, cube_id: int):
    rec = s.get(CubeRecord, cube_id)
    if rec is None: raise HTTPException(404, f"cube {cube_id} not found")
    try:
        return rec, open_cube(rec.path, qc=False, unit=rec.unit if rec.unit != "unknown" else None)
    except (IngestError, FileNotFoundError) as e:
        raise HTTPException(410, f"source file unavailable: {e}")


def _cube_dict(c: CubeRecord, full=False):
    d = dict(id=c.id, name=c.name, format=c.format, rows=c.rows, cols=c.cols, bands=c.bands, unit=c.unit,
             sensor=c.sensor, crs=c.crs, wl_min_nm=c.wl_min_nm, wl_max_nm=c.wl_max_nm,
             n_bad_bands=int(sum(c.bad_band_mask)), source=c.source.name if c.source else None, qc=c.qc)
    if full: d.update(wavelengths_nm=c.wavelengths_nm, bad_band_mask=c.bad_band_mask, fwhm_nm=c.fwhm_nm, transform=c.transform)
    return d


def _nan(a): return [None if not np.isfinite(x) else float(x) for x in np.asarray(a, float).ravel()]


@app.get("/health")
def health(s: Session = Depends(db)):
    return dict(status="ok", version=__version__, contract=CONTRACT_VERSION, db=get_engine().dialect.name)


@app.get("/formats")
def formats():
    from ..ingest.api import _READERS
    from ..spectra import SPECTRA_FORMATS
    return dict(cubes=sorted(_READERS), spectra=SPECTRA_FORMATS)


class IngestReq(BaseModel):
    path: str
    source: str = "unspecified"
    kind: str | None = None
    library: str | None = None


def _res(rs): return [r.__dict__ for r in rs]


@app.post("/ingest")
def ingest(req: IngestReq):
    """Ingest a file/folder that already exists on the API server's disk."""
    if not Path(req.path).exists(): raise HTTPException(404, f"{req.path} not found on server")
    return _res(ingest_path(req.path, source_name=req.source, kind=req.kind, library=req.library))


@app.post("/ingest/upload")
async def ingest_upload(file: UploadFile = File(...), source: str = "uploaded", kind: str | None = None):
    """Upload ONE self-contained file (.tif, .nc, .h5, .csv, .sig ...). For ENVI use /ingest with a server path."""
    from ..config import data_dir
    dest = data_dir() / "raw" / "uploads"; dest.mkdir(parents=True, exist_ok=True)
    target = dest / Path(file.filename).name
    with open(target, "wb") as f: shutil.copyfileobj(file.file, f)
    return _res(ingest_path(target, source_name=source, kind=kind))


@app.get("/sources")
def sources(s: Session = Depends(db)):
    return [dict(id=x.id, name=x.name, kind=x.kind, url=x.url) for x in s.scalars(select(Source))]


@app.get("/cubes")
def cubes(s: Session = Depends(db)):
    return [_cube_dict(c) for c in s.scalars(select(CubeRecord).order_by(CubeRecord.id))]


@app.get("/cubes/{cube_id}")
def cube_detail(cube_id: int, s: Session = Depends(db)):
    rec = s.get(CubeRecord, cube_id)
    if rec is None: raise HTTPException(404)
    return _cube_dict(rec, full=True)


@app.get("/cubes/{cube_id}/spectrum")
def cube_spectrum(cube_id: int, row: int, col: int, s: Session = Depends(db)):
    rec, cube = _cube(s, cube_id)
    if not (0 <= row < rec.rows and 0 <= col < rec.cols): raise HTTPException(422, "pixel outside the cube")
    v = cube.read((row, row + 1), (col, col + 1))[0, 0]
    return dict(wavelengths_nm=rec.wavelengths_nm, values=_nan(v), bad_band_mask=rec.bad_band_mask, unit=rec.unit)


class ROIReq(BaseModel):
    bbox_px: tuple[int, int, int, int] | None = None     # (row0, col0, row1, col1), r1/c1 exclusive
    polygon: list[tuple[float, float]] | None = None     # pixel (col,row) vertices


@app.post("/cubes/{cube_id}/roi")
def cube_roi(cube_id: int, req: ROIReq, s: Session = Depends(db)):
    rec, cube = _cube(s, cube_id)
    try:
        r = extract_roi(cube, bbox_px=req.bbox_px, polygon=req.polygon, polygon_coords="pixel")
    except (IngestError, ValueError) as e:
        raise HTTPException(422, str(e))
    return dict(n_pixels=r.n_pixels, n_excluded_nodata=r.n_excluded_nodata, wavelengths_nm=rec.wavelengths_nm,
                mean=_nan(r.mean), std=_nan(r.std), bad_band_mask=rec.bad_band_mask, unit=r.unit)


@app.get("/libraries")
def libraries(s: Session = Depends(db)):
    out = []
    for l in s.scalars(select(Library)):
        n = s.scalar(select(func.count(SpectrumRecord.id)).select_from(SpectrumRecord)
                     .join(Sample, SpectrumRecord.sample_id == Sample.id)
                     .join(Dataset, Sample.dataset_id == Dataset.id).where(Dataset.library_id == l.id))
        out.append(dict(id=l.id, name=l.name, n_spectra=n))
    return out


@app.get("/spectra")
def spectra(material: str | None = None, unit: str | None = None, limit: int = Query(50, le=500), s: Session = Depends(db)):
    q = (select(SpectrumRecord, Material.name).select_from(SpectrumRecord)
         .join(Sample, SpectrumRecord.sample_id == Sample.id).join(Material, Sample.material_id == Material.id))
    if material: q = q.where(Material.name.ilike(f"%{material}%"))
    if unit: q = q.where(SpectrumRecord.unit == unit)
    return [dict(id=r.id, material=m, unit=r.unit, purity=r.purity, n_bands=r.n_bands,
                 wl_min_nm=r.wl_min_nm, wl_max_nm=r.wl_max_nm) for r, m in s.execute(q.limit(limit))]


# ---------------- library builder + search ----------------
class BuildReq(BaseModel):
    name: str
    inputs: list[str]                      # paths on the server: spectra files, ENVI ASCII ROI, GeoJSON
    grid: str = "first"                    # first | cube:<path> | ascii:<path>
    cube: str | None = None
    method: str = "linear"
    unit: str | None = None
    min_coverage: float = 0.5


@app.post("/libraries/build")
def libraries_build(req: BuildReq):
    from ..library.builder import build_library
    from ..library.grid import Grid
    try:
        grid = "first" if req.grid == "first" else Grid.from_cube(req.grid[5:]) if req.grid.startswith("cube:") \
            else Grid.from_ascii(req.grid[6:]) if req.grid.startswith("ascii:") else None
        if grid is None: raise HTTPException(422, "grid must be first | cube:<path> | ascii:<path>")
        r = build_library(req.name, req.inputs, grid, cube=req.cube, method=req.method, unit=req.unit, min_coverage=req.min_coverage)
    except (IngestError, FileNotFoundError, ValueError) as e:
        raise HTTPException(422, str(e))
    return dict(library=r.library, grid=r.grid_origin, n_bands=r.n_bands, added=r.added, rejected=r.rejected, spectrum_ids=r.spectrum_ids)


@app.get("/spectra/search")
def spectra_search(text: str | None = None, category: str | None = None, unit: str | None = None,
                   purity: str | None = None, library: str | None = None, covers_lo: float | None = None,
                   covers_hi: float | None = None, limit: int = Query(50, le=500), offset: int = 0, s: Session = Depends(db)):
    from ..library.search import find_spectra
    cov = (covers_lo, covers_hi) if covers_lo is not None and covers_hi is not None else None
    return find_spectra(s, text=text, category=category, unit=unit, purity=purity, library=library, covers_nm=cov, limit=limit, offset=offset)


class MatchReq(BaseModel):
    spectrum_id: int | None = None
    wavelengths_nm: list[float] | None = None
    values: list[float | None] | None = None
    top_k: int = 5
    unit: str | None = None
    library: str | None = None
    metric: str = "sam"


@app.post("/spectra/match")
def spectra_match(req: MatchReq, s: Session = Depends(db)):
    from ..library.search import match_spectrum
    if req.spectrum_id is not None:
        rec = s.get(SpectrumRecord, req.spectrum_id)
        if rec is None: raise HTTPException(404, "spectrum not found")
        wl, vals, unit, ex = rec.wavelengths_nm, rec.values, req.unit or rec.unit, rec.id
    elif req.wavelengths_nm and req.values and len(req.values) == len(req.wavelengths_nm):
        wl, vals, unit, ex = req.wavelengths_nm, req.values, req.unit, None
        if unit is None: raise HTTPException(422, "give unit when sending a raw spectrum")
    else:
        raise HTTPException(422, "give spectrum_id, or equal-length wavelengths_nm + values")
    q = np.array([np.nan if v is None else v for v in vals], float)
    return match_spectrum(s, wl, q, top_k=req.top_k, metric=req.metric, unit=unit, library=req.library, exclude_id=ex)


@app.get("/spectra/{spectrum_id}")
def spectrum(spectrum_id: int, s: Session = Depends(db)):
    r = s.get(SpectrumRecord, spectrum_id)
    if r is None: raise HTTPException(404)
    return dict(id=r.id, material=r.sample.material.name, unit=r.unit, purity=r.purity,
                wavelengths_nm=r.wavelengths_nm, values=r.values, meta=r.sample.meta)


