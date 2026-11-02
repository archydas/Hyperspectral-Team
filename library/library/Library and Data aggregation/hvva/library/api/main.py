"""FastAPI app - the new_hyperspectral material_library backend (app/main.py + app/routes/*) on top of
hvva.library.repository, so the HTTP API and the desktop GUI share one implementation.

Original endpoints (unchanged paths and response shapes):
    GET  /                                   GET  /health
    GET  /api/libraries/                     POST /api/libraries/            GET /api/libraries/{id}
    GET  /api/libraries/{id}/datasets        GET  /api/datasets/{id}
    GET  /api/libraries/{id}/materials?search=&category=&limit=&offset=
    POST /api/libraries/{id}/materials       GET  /api/materials/search?q=&limit=
    GET  /api/materials/{id}                 GET  /api/materials/{id}/samples
    GET  /api/samples/{id}                   GET  /api/samples/{id}/spectra   GET /api/spectra/{id}
Added for HyperVista:
    DELETE /api/libraries/{id}               DELETE /api/materials/{id}
    GET  /api/libraries/{id}/categories      GET  /api/libraries/{id}/spectra   (bulk, for matching)
    POST /api/libraries/{id}/spectra         (save signatures: Material -> Sample -> Spectrum)
    POST /api/libraries/{id}/import-csv?filename=&dataset_name=&category=&acquisition=   (body = raw CSV)
    Category tree + acquisition type (migration 001):
    GET  /api/libraries/{id}/materials?...&category_path=Vegetation/Tree&acquisition=lab|field|airborne|satellite|__none__
    GET  /api/libraries/{id}/facets          tree paths with counts + materials per acquisition type
    POST /api/libraries/{id}/categories/rename   {"old_path", "new_path"}
    PATCH /api/materials                     {"ids": [...], "category"?: path|null, "acquisition_type"?: tag|null}
    PATCH /api/spectra/{id}                  {"acquisition_type": tag|null}
"""
from __future__ import annotations

import os
import tempfile
from typing import List, Optional

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from .. import repository as repo
from ..database import Database
from ..importer import import_wide_csv

_db: Optional[Database] = None


def database() -> Database:
    global _db
    if _db is None:
        db = Database()                      # DATABASE_URL from the environment / .env, else the default URL
        db.migrate()                         # adds spectra.acquisition_type etc. to an older database
        _db = db
    return _db


def get_db():
    yield from database().get_db()


def _nf(fn, *a, **kw):
    try:
        return fn(*a, **kw)
    except repo.NotFound as e:
        raise HTTPException(status_code=404, detail=str(e)) from None
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from None


app = FastAPI(title="Material Spectral Library API", version="1.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])


@app.get("/")
def root():
    return {"message": "Material Spectral Library API"}


@app.get("/health")
def health():
    return {"status": "ok"}


# ------------------------------------------------------------------ schemas
class LibraryCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    metadata: Optional[dict] = None


class MaterialCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=500)
    category: Optional[str] = None
    description: Optional[str] = None
    metadata: Optional[dict] = None


class SpectrumItem(BaseModel):
    name: str = Field(..., min_length=1, max_length=500)
    wavelengths: List[float]
    values: List[Optional[float]]
    std: Optional[List[Optional[float]]] = None
    category: Optional[str] = None
    material_metadata: Optional[dict] = None
    spectrum_metadata: Optional[dict] = None
    acquisition_type: Optional[str] = None


class SpectraSave(BaseModel):
    dataset_name: str = Field("HyperVista signatures", min_length=1, max_length=255)
    category: Optional[str] = None
    acquisition_type: Optional[str] = None
    items: List[SpectrumItem]


class MaterialsPatch(BaseModel):
    ids: List[str] = Field(..., min_length=1)
    category: Optional[str] = None             # only applied when present in the body (null = uncategorised)
    acquisition_type: Optional[str] = None     # only applied when present in the body (null = untagged)


class SpectrumPatch(BaseModel):
    acquisition_type: Optional[str] = None


class CategoryRename(BaseModel):
    old_path: str = Field(..., min_length=1)
    new_path: Optional[str] = None


# ------------------------------------------------------------------ libraries
libraries = APIRouter(prefix="/api/libraries", tags=["Libraries"])


@libraries.get("/")
def get_libraries(db: Session = Depends(get_db)):
    return repo.list_libraries(db)


@libraries.post("/")
def create_library(payload: LibraryCreate, db: Session = Depends(get_db)):
    return _nf(repo.create_library, db, payload.name, payload.description, payload.metadata)


@libraries.get("/{library_id}")
def get_library(library_id: str, db: Session = Depends(get_db)):
    return _nf(repo.get_library, db, library_id)


@libraries.delete("/{library_id}")
def delete_library(library_id: str, db: Session = Depends(get_db)):
    _nf(repo.delete_library, db, library_id)
    return {"deleted": library_id}


@libraries.get("/{library_id}/categories")
def get_categories(library_id: str, db: Session = Depends(get_db)):
    return _nf(repo.list_categories, db, library_id)


@libraries.get("/{library_id}/spectra")
def get_library_spectra(library_id: str, limit: Optional[int] = Query(None, ge=1), acquisition: Optional[str] = None,
                        category_path: Optional[str] = None, db: Session = Depends(get_db)):
    return _nf(repo.library_spectra, db, library_id, limit, acquisition, category_path)


@libraries.get("/{library_id}/facets")
def get_facets(library_id: str, db: Session = Depends(get_db)):
    return _nf(repo.library_facets, db, library_id)


@libraries.post("/{library_id}/categories/rename")
def rename_category(library_id: str, payload: CategoryRename, db: Session = Depends(get_db)):
    return _nf(repo.rename_category, db, library_id, payload.old_path, payload.new_path)


@libraries.post("/{library_id}/spectra")
def save_spectra(library_id: str, payload: SpectraSave, db: Session = Depends(get_db)):
    def go():
        ds = repo.get_or_create_dataset(db, library_id, payload.dataset_name, source_name="HyperVista",
                                        description="Signatures saved from HyperVista.", commit=False)
        out = [repo.add_spectrum(db, library_id, ds.id, it.name, it.wavelengths,
                                 [float("nan") if v is None else v for v in it.values],
                                 category=it.category or payload.category,
                                 std=None if it.std is None else [float("nan") if v is None else v for v in it.std],
                                 material_metadata=it.material_metadata, spectrum_metadata=it.spectrum_metadata,
                                 commit=False, acquisition=it.acquisition_type or payload.acquisition_type)
               for it in payload.items]
        db.commit()
        return out
    try:
        return _nf(go)
    except HTTPException:
        db.rollback()
        raise


@libraries.post("/{library_id}/import-csv")
async def import_csv(library_id: str, request: Request, filename: str = "upload.csv",
                     dataset_name: Optional[str] = None, category: Optional[str] = None,
                     acquisition: Optional[str] = None, db: Session = Depends(get_db)):
    body = await request.body()
    if not body:
        raise HTTPException(status_code=422, detail="empty CSV body")
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, os.path.basename(filename) or "upload.csv")
        with open(p, "wb") as f:
            f.write(body)
        return _nf(import_wide_csv, db, p, library_id=library_id, dataset_name=dataset_name, category=category,
                   acquisition=acquisition)


# ------------------------------------------------------------------ datasets / materials / samples / spectra
api = APIRouter(prefix="/api")


@api.get("/libraries/{library_id}/datasets", tags=["Datasets"])
def get_library_datasets(library_id: str, db: Session = Depends(get_db)):
    return _nf(repo.list_datasets, db, library_id)


@api.get("/datasets/{dataset_id}", tags=["Datasets"])
def get_dataset(dataset_id: str, db: Session = Depends(get_db)):
    return _nf(repo.get_dataset, db, dataset_id)


@api.get("/libraries/{library_id}/materials", tags=["Materials"])
def get_library_materials(library_id: str, search: Optional[str] = None, category: Optional[str] = None,
                          limit: int = Query(20, ge=1, le=100), offset: int = Query(0, ge=0),
                          category_path: Optional[str] = None, acquisition: Optional[str] = None,
                          db: Session = Depends(get_db)):
    return _nf(repo.list_materials, db, library_id, search, category, limit, offset, category_path, acquisition)


@api.post("/libraries/{library_id}/materials", tags=["Materials"])
def create_material(library_id: str, payload: MaterialCreate, db: Session = Depends(get_db)):
    return _nf(repo.create_material, db, library_id, payload.name, payload.category, payload.description,
               payload.metadata)


# must stay above /materials/{material_id} (route order)
@api.get("/materials/search", tags=["Materials"])
def search_materials_global(q: str = Query(..., min_length=2), limit: int = Query(20, ge=1, le=100),
                            acquisition: Optional[str] = None, db: Session = Depends(get_db)):
    return _nf(repo.search_materials_global, db, q, limit, acquisition)


@api.patch("/materials", tags=["Materials"])
def patch_materials(payload: MaterialsPatch, db: Session = Depends(get_db)):
    kw = {k: getattr(payload, k) for k in ("category",) if k in payload.model_fields_set}
    if "acquisition_type" in payload.model_fields_set:
        kw["acquisition"] = payload.acquisition_type
    return _nf(repo.update_materials, db, payload.ids, **kw)


@api.get("/materials/{material_id}", tags=["Materials"])
def get_material(material_id: str, db: Session = Depends(get_db)):
    return _nf(repo.get_material, db, material_id)


@api.delete("/materials/{material_id}", tags=["Materials"])
def delete_material(material_id: str, db: Session = Depends(get_db)):
    _nf(repo.delete_material, db, material_id)
    return {"deleted": material_id}


@api.get("/materials/{material_id}/samples", tags=["Samples"])
def get_material_samples(material_id: str, db: Session = Depends(get_db)):
    return _nf(repo.list_samples, db, material_id)


@api.get("/samples/{sample_id}", tags=["Samples"])
def get_sample(sample_id: str, db: Session = Depends(get_db)):
    return _nf(repo.get_sample, db, sample_id)


@api.get("/samples/{sample_id}/spectra", tags=["Spectra"])
def get_sample_spectra(sample_id: str, db: Session = Depends(get_db)):
    return _nf(repo.list_spectra, db, sample_id)


@api.patch("/spectra/{spectrum_id}", tags=["Spectra"])
def patch_spectrum(spectrum_id: str, payload: SpectrumPatch, db: Session = Depends(get_db)):
    return _nf(repo.set_spectrum_acquisition, db, spectrum_id, payload.acquisition_type)


@api.get("/spectra/{spectrum_id}", tags=["Spectra"])
def get_spectrum(spectrum_id: str, db: Session = Depends(get_db)):
    return _nf(repo.get_spectrum, db, spectrum_id)


app.include_router(libraries)
app.include_router(api)
