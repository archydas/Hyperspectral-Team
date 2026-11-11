"""Query/command layer of the material spectral library.

This is the logic of the FastAPI routes in new_hyperspectral/material_library/backend/app/routes/
(libraries, datasets, materials, samples, spectra) pulled out of the HTTP handlers so the desktop GUI can call
it directly over a SQLAlchemy session, and the bundled API (hvva/library/api) can call exactly the same code.
Every function returns plain dicts shaped like the original API responses.

Additions needed by HyperVista (not in the original routes):
    list_categories, delete_library, delete_material   - library management from the GUI
    get_or_create_dataset, add_spectrum                 - save aggregated signatures / clicked pixels
    library_spectra                                     - bulk fetch for spectral-angle matching
    library_facets, rename_category, update_materials   - category tree (path in materials.category)
    set_spectrum_acquisition                            - lab / field / airborne / satellite tag per spectrum
    save_roi_spectra, list_rois, get_roi, roi_classes,  - ROI analysis (migration 002): ROIs class by class, from
    delete_roi, roi_spectra                               one or many images, with the spectra computed from them
"""
from __future__ import annotations

import math
import re
import uuid
from typing import Iterable, Optional

import numpy as np
from sqlalchemy import and_, exists, func, or_
from sqlalchemy.orm import Session

from .models import DataFile, Dataset, Library, Material, Roi, Sample, Spectrum, SpectrumRoi

MAX_PAGE = 100          # same upper bound as the API's Query(le=100)

ACQUISITION_TYPES = ("lab", "field", "airborne", "satellite")
UNTAGGED = "__none__"           # filter value: spectra without an acquisition tag
UNCATEGORISED = "__none__"      # filter value: materials without a category
SEP = "/"                       # category path separator: Vegetation/Tree/Tsuga/Canadensis
_UNSET = object()
_SPLIT = re.compile(r"\s*(?:/|\\|>|->|›|»|\|)\s*")


class NotFound(LookupError):
    pass


# ------------------------------------------------------------------ helpers
def _uuid(value, what="Object"):
    if isinstance(value, uuid.UUID):
        return value
    try:
        return uuid.UUID(str(value))
    except (ValueError, TypeError, AttributeError):
        raise NotFound(f"{what} not found") from None


def slugify(name: str, fallback="library") -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.strip().lower()).strip("-")
    return slug or fallback


def _unique_slug(db: Session, base: str, model=Library, **scope) -> str:
    slug, k = base, 2
    while db.query(model).filter_by(slug=slug, **scope).first() is not None:
        slug, k = f"{base}-{k}", k + 1
    return slug


def normalize_category(path) -> Optional[str]:
    """'Vegetation > Tree > Tsuga > Canadensis' or 'Vegetation/Tree//Tsuga ' -> 'Vegetation/Tree/Tsuga/Canadensis'.

    Level separators accepted: /  \\  >  ->  ›  »  |   Levels are trimmed, empty levels dropped, case kept as typed
    (the GUI offers existing paths for completion so spellings stay consistent).  Empty -> None (uncategorised).
    """
    if path is None:
        return None
    parts = [p.strip() for p in _SPLIT.split(str(path)) if p and p.strip()]
    out = SEP.join(parts)
    if len(out) > 255:
        raise ValueError("category path is longer than 255 characters")
    return out or None


def category_levels(path) -> list:
    return [p for p in (path or "").split(SEP) if p]


def normalize_acquisition(value) -> Optional[str]:
    if value is None:
        return None
    v = str(value).strip().lower()
    if v in ("", "none", "untagged", UNTAGGED):
        return None
    if v not in ACQUISITION_TYPES:
        raise ValueError(f"acquisition type must be one of {', '.join(ACQUISITION_TYPES)} (got {value!r})")
    return v


def _like_escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _category_filter(path):
    """Materials in a tree node: the node itself and everything below it."""
    if path == UNCATEGORISED:
        return Material.category.is_(None)
    p = normalize_category(path)
    if p is None:
        return None
    return or_(Material.category == p, Material.category.like(_like_escape(p) + SEP + "%", escape="\\"))


def _acquisition_filter(acq):
    """Materials having at least one spectrum with this tag (UNTAGGED: at least one untagged spectrum)."""
    cond = Spectrum.acquisition_type.is_(None) if acq == UNTAGGED else Spectrum.acquisition_type == normalize_acquisition(acq)
    return exists().where(and_(Sample.material_id == Material.id, Spectrum.sample_id == Sample.id, cond))


def _clean(values) -> list:
    """float list with NaN/inf -> None (NULL in the array; JSON-safe for the API)."""
    return [None if v is None or not math.isfinite(float(v)) else float(v) for v in np.asarray(values, float).ravel()]


def library_dict(x: Library, **extra):
    d = {"id": str(x.id), "name": x.name, "slug": x.slug, "description": x.description,
         "metadata": x.library_metadata or {}}
    d.update(extra)
    return d


def dataset_dict(x: Dataset):
    return {"id": str(x.id), "library_id": str(x.library_id), "name": x.name, "slug": x.slug,
            "description": x.description, "source_name": x.source_name, "source_version": x.source_version,
            "metadata": x.dataset_metadata or {}}


def material_dict(x: Material):
    return {"id": str(x.id), "library_id": str(x.library_id), "name": x.name, "category": x.category,
            "description": x.description, "metadata": x.material_metadata or {}}


def sample_dict(x: Sample):
    return {"id": str(x.id), "material_id": str(x.material_id), "dataset_id": str(x.dataset_id), "name": x.name,
            "sample_number": x.sample_number, "metadata": x.sample_metadata or {}}


def spectrum_dict(x: Spectrum):
    return {"id": str(x.id), "sample_id": str(x.sample_id), "wavelength_start": x.wavelength_start,
            "wavelength_end": x.wavelength_end, "wavelength_step": x.wavelength_step,
            "measurement_unit": x.measurement_unit, "values": list(x.values or []),
            "acquisition_type": x.acquisition_type, "metadata": x.spectrum_metadata or {},
            "roi_id": None if x.roi_id is None else str(x.roi_id), "roi_class": x.roi_class}


def roi_dict(x: Roi, **extra):
    d = {"id": str(x.id), "library_id": str(x.library_id),
         "dataset_id": None if x.dataset_id is None else str(x.dataset_id), "roi_uid": x.roi_uid,
         "name": x.roi_name, "class_name": x.class_name, "scene_name": x.scene_name, "scene_path": x.scene_path,
         "image_rows": x.image_rows, "image_cols": x.image_cols, "geometry_type": x.geometry_type,
         "geometry": x.geometry or {}, "geometry_map": x.geometry_map, "crs": x.crs,
         "bbox": None if x.bbox_row_min is None else [x.bbox_row_min, x.bbox_row_max, x.bbox_col_min, x.bbox_col_max],
         "n_pixels": x.n_pixels, "metadata": x.roi_metadata or {},
         "created_at": x.created_at.isoformat() if x.created_at else None}
    d.update(extra)
    return d


def spectrum_wavelengths(spec: dict) -> np.ndarray:
    """Wavelength axis of a spectrum dict: explicit list in metadata, else start + i*step, else linspace."""
    n = len(spec.get("values") or [])
    wl = (spec.get("metadata") or {}).get("wavelengths")
    if wl is not None and len(wl) == n:
        return np.asarray(wl, float)
    start, end, step = spec.get("wavelength_start"), spec.get("wavelength_end"), spec.get("wavelength_step")
    if step:
        return float(start) + float(step) * np.arange(n)
    return np.linspace(float(start), float(end), n)


def spectrum_values(spec: dict) -> np.ndarray:
    return np.array([np.nan if v is None else v for v in (spec.get("values") or [])], float)


# ------------------------------------------------------------------ libraries
def list_libraries(db: Session):
    mats = (db.query(Material.library_id, func.count(Material.id)).group_by(Material.library_id).all())
    dsets = (db.query(Dataset.library_id, func.count(Dataset.id)).group_by(Dataset.library_id).all())
    mc, dc = {k: v for k, v in mats}, {k: v for k, v in dsets}
    return [library_dict(l, material_count=mc.get(l.id, 0), dataset_count=dc.get(l.id, 0))
            for l in db.query(Library).order_by(Library.name).all()]


def get_library(db: Session, library_id):
    lib = db.query(Library).filter(Library.id == _uuid(library_id, "Library")).first()
    if lib is None:
        raise NotFound("Library not found")
    return library_dict(lib)


def create_library(db: Session, name: str, description: Optional[str] = None, metadata: Optional[dict] = None):
    name = (name or "").strip()
    if not name:
        raise ValueError("Library name is required")
    lib = Library(name=name[:255], slug=_unique_slug(db, slugify(name)), description=description or None,
                  library_metadata=metadata or {})
    db.add(lib)
    db.commit()
    db.refresh(lib)
    return library_dict(lib, material_count=0, dataset_count=0)


def delete_library(db: Session, library_id):
    lib = db.query(Library).filter(Library.id == _uuid(library_id, "Library")).first()
    if lib is None:
        raise NotFound("Library not found")
    # explicit child deletes so SQLite (no FK cascade by default) behaves like PostgreSQL's ON DELETE CASCADE
    mat_ids = [m for (m,) in db.query(Material.id).filter(Material.library_id == lib.id)]
    ds_ids = [d for (d,) in db.query(Dataset.id).filter(Dataset.library_id == lib.id)]
    _delete_samples(db, db.query(Sample.id).filter(Sample.material_id.in_(mat_ids) | Sample.dataset_id.in_(ds_ids)))
    db.query(DataFile).filter(DataFile.dataset_id.in_(ds_ids)).delete(synchronize_session=False)
    roi_ids = db.query(Roi.id).filter(Roi.library_id == lib.id).scalar_subquery()
    db.query(SpectrumRoi).filter(SpectrumRoi.roi_id.in_(roi_ids)).delete(synchronize_session=False)
    db.query(Roi).filter(Roi.library_id == lib.id).delete(synchronize_session=False)
    db.query(Material).filter(Material.id.in_(mat_ids)).delete(synchronize_session=False)
    db.query(Dataset).filter(Dataset.id.in_(ds_ids)).delete(synchronize_session=False)
    db.delete(lib)
    db.commit()


def _delete_samples(db, sample_id_query):
    ids = [s for (s,) in sample_id_query]
    if ids:
        spec_ids = db.query(Spectrum.id).filter(Spectrum.sample_id.in_(ids)).scalar_subquery()
        db.query(SpectrumRoi).filter(SpectrumRoi.spectrum_id.in_(spec_ids)).delete(synchronize_session=False)
        db.query(Spectrum).filter(Spectrum.sample_id.in_(ids)).delete(synchronize_session=False)
        db.query(Sample).filter(Sample.id.in_(ids)).delete(synchronize_session=False)


# ------------------------------------------------------------------ datasets
def list_datasets(db: Session, library_id):
    q = db.query(Dataset).filter(Dataset.library_id == _uuid(library_id, "Library")).order_by(Dataset.name)
    return [dataset_dict(d) for d in q.all()]


def get_dataset(db: Session, dataset_id):
    d = db.query(Dataset).filter(Dataset.id == _uuid(dataset_id, "Dataset")).first()
    if d is None:
        raise NotFound("Dataset not found")
    return dataset_dict(d)


def get_or_create_dataset(db: Session, library_id, name: str, description=None, source_name=None,
                          source_version=None, metadata=None, commit=True) -> Dataset:
    lid = _uuid(library_id, "Library")
    if db.query(Library).filter(Library.id == lid).first() is None:
        raise NotFound("Library not found")
    slug = slugify(name, "dataset")
    d = db.query(Dataset).filter(Dataset.library_id == lid, Dataset.slug == slug).first()
    if d is None:
        d = Dataset(library_id=lid, name=name.strip()[:255], slug=slug, description=description,
                    source_name=source_name, source_version=source_version, dataset_metadata=metadata or {})
        db.add(d)
        db.flush()
        if commit:
            db.commit()
            db.refresh(d)
    return d


# ------------------------------------------------------------------ materials
def _acq_by_material(db: Session, material_ids) -> dict:
    """{material_id: sorted acquisition tags of its spectra ('untagged' for NULL)}."""
    if not material_ids:
        return {}
    rows = (db.query(Sample.material_id, Spectrum.acquisition_type).join(Spectrum, Spectrum.sample_id == Sample.id)
            .filter(Sample.material_id.in_(material_ids)).distinct().all())
    out = {}
    for mid, a in rows:
        out.setdefault(mid, set()).add(a or "untagged")
    order = {t: i for i, t in enumerate(ACQUISITION_TYPES + ("untagged",))}
    return {k: sorted(v, key=order.get) for k, v in out.items()}


def list_materials(db: Session, library_id, search: Optional[str] = None, category: Optional[str] = None,
                   limit: int = 20, offset: int = 0, category_path: Optional[str] = None,
                   acquisition: Optional[str] = None):
    """Server-side search + pagination, same contract as GET /api/libraries/{id}/materials.

    category       exact category (original API)
    category_path  category tree node: that path and all paths below it ('__none__' = uncategorised)
    acquisition    lab | field | airborne | satellite | '__none__' (untagged) - materials with such a spectrum
    Each item additionally carries 'acquisition_types' (tags of its spectra).
    """
    q = db.query(Material).filter(Material.library_id == _uuid(library_id, "Library"))
    if search:
        q = q.filter(Material.name.ilike(f"%{search}%"))
    if category:
        q = q.filter(Material.category == category)
    if category_path:
        cond = _category_filter(category_path)
        if cond is not None:
            q = q.filter(cond)
    if acquisition:
        q = q.filter(_acquisition_filter(acquisition))
    total = q.count()
    limit = max(1, min(int(limit), MAX_PAGE))
    offset = max(0, int(offset))
    items = q.order_by(Material.name).offset(offset).limit(limit).all()
    acq = _acq_by_material(db, [m.id for m in items])
    return {"items": [dict(material_dict(m), acquisition_types=acq.get(m.id, [])) for m in items],
            "total": total, "limit": limit, "offset": offset}


def library_facets(db: Session, library_id):
    """What the category tree and the acquisition filter need, in one call:

    {"total": materials in the library,
     "categories": [{"path": "Vegetation/Tree/Tsuga/Canadensis", "count": materials with exactly that path}, ...],
     "uncategorised": materials without a category,
     "acquisition": {"lab": n, "field": n, "airborne": n, "satellite": n, "untagged": n}}   # materials per tag
    """
    lid = _uuid(library_id, "Library")
    if db.query(Library.id).filter(Library.id == lid).first() is None:
        raise NotFound("Library not found")
    rows = (db.query(Material.category, func.count(Material.id)).filter(Material.library_id == lid)
            .group_by(Material.category).all())
    cats = sorted(({"path": c, "count": n} for c, n in rows if c), key=lambda d: d["path"].lower())
    acq_rows = (db.query(Spectrum.acquisition_type, func.count(func.distinct(Material.id)))
                .join(Sample, Sample.id == Spectrum.sample_id).join(Material, Material.id == Sample.material_id)
                .filter(Material.library_id == lid).group_by(Spectrum.acquisition_type).all())
    acq = {t: 0 for t in ACQUISITION_TYPES + ("untagged",)}
    for a, n in acq_rows:
        acq[a or "untagged"] = n
    return {"total": sum(n for _c, n in rows), "categories": cats,
            "uncategorised": sum(n for c, n in rows if not c), "acquisition": acq}


def update_materials(db: Session, material_ids, category=_UNSET, acquisition=_UNSET):
    """Bulk edit: set the category path and/or tag every spectrum of the given materials.

    Pass category=None to make them uncategorised, acquisition=None to untag. Omitted = unchanged.
    """
    ids = [_uuid(m, "Material") for m in material_ids]
    if not ids:
        return {"materials": 0, "spectra": 0}
    found = db.query(Material.id).filter(Material.id.in_(ids)).count()
    if found != len(set(ids)):
        raise NotFound("Material not found")
    n_spec = 0
    if category is not _UNSET:
        db.query(Material).filter(Material.id.in_(ids)).update(
            {Material.category: normalize_category(category)}, synchronize_session=False)
    if acquisition is not _UNSET:
        a = normalize_acquisition(acquisition)
        sids = db.query(Spectrum.id).join(Sample, Sample.id == Spectrum.sample_id).filter(Sample.material_id.in_(ids))
        n_spec = db.query(Spectrum).filter(Spectrum.id.in_(sids.scalar_subquery())).update(
            {Spectrum.acquisition_type: a}, synchronize_session=False)
    db.commit()
    return {"materials": found, "spectra": n_spec}


def rename_category(db: Session, library_id, old_path: str, new_path: Optional[str]):
    """Rename / move a tree node: 'Vegetation/Tree' -> 'Plants/Trees' also rewrites every path below it.
    new_path empty -> the node and everything below it become uncategorised."""
    lid = _uuid(library_id, "Library")
    old = normalize_category(old_path)
    if old is None:
        raise ValueError("choose a category to rename")
    new = normalize_category(new_path)
    if new is not None and (new == old or new.startswith(old + SEP)):
        if new == old:
            return {"materials": 0}
        raise ValueError("a category cannot be moved inside itself")
    mats = db.query(Material).filter(Material.library_id == lid, _category_filter(old)).all()
    for m in mats:
        rest = m.category[len(old):]                       # '' or '/Tsuga/Canadensis'
        m.category = normalize_category((new or "") + rest) if new else None
    db.commit()
    return {"materials": len(mats)}


def list_categories(db: Session, library_id):
    q = (db.query(Material.category).filter(Material.library_id == _uuid(library_id, "Library"),
                                            Material.category.isnot(None)).distinct())
    return sorted(c for (c,) in q.all() if c)


def create_material(db: Session, library_id, name: str, category=None, description=None, metadata=None):
    lid = _uuid(library_id, "Library")
    if db.query(Library).filter(Library.id == lid).first() is None:
        raise NotFound("Library not found")
    name = (name or "").strip()
    if not name:
        raise ValueError("Material name is required")
    m = Material(library_id=lid, name=name[:500], category=normalize_category(category),
                 description=description or None, material_metadata=metadata or {})
    db.add(m)
    db.commit()
    db.refresh(m)
    return dict(material_dict(m), acquisition_types=[])


def search_materials_global(db: Session, q: str, limit: int = 20, acquisition: Optional[str] = None):
    """Material name search across every library (GET /api/materials/search)."""
    q = (q or "").strip()
    if len(q) < 2:
        return {"items": []}
    query = (db.query(Material, Library.name.label("library_name"))
             .join(Library, Library.id == Material.library_id)
             .filter(Material.name.ilike(f"%{q}%")))
    if acquisition:
        query = query.filter(_acquisition_filter(acquisition))
    rows = query.order_by(Material.name).limit(max(1, min(int(limit), MAX_PAGE))).all()
    acq = _acq_by_material(db, [m.id for m, _ in rows])
    return {"items": [{"id": str(m.id), "library_id": str(m.library_id), "library_name": ln, "name": m.name,
                       "category": m.category, "acquisition_types": acq.get(m.id, [])} for m, ln in rows]}


def get_material(db: Session, material_id):
    m = db.query(Material).filter(Material.id == _uuid(material_id, "Material")).first()
    if m is None:
        raise NotFound("Material not found")
    return material_dict(m)


def delete_material(db: Session, material_id):
    m = db.query(Material).filter(Material.id == _uuid(material_id, "Material")).first()
    if m is None:
        raise NotFound("Material not found")
    _delete_samples(db, db.query(Sample.id).filter(Sample.material_id == m.id))
    db.delete(m)
    db.commit()


# ------------------------------------------------------------------ samples / spectra
def list_samples(db: Session, material_id):
    q = db.query(Sample).filter(Sample.material_id == _uuid(material_id, "Material")).order_by(Sample.sample_number)
    return [sample_dict(s) for s in q.all()]


def get_sample(db: Session, sample_id):
    s = db.query(Sample).filter(Sample.id == _uuid(sample_id, "Sample")).first()
    if s is None:
        raise NotFound("Sample not found")
    return sample_dict(s)


def list_spectra(db: Session, sample_id):
    return [spectrum_dict(s) for s in db.query(Spectrum).filter(Spectrum.sample_id == _uuid(sample_id, "Sample")).all()]


def set_spectrum_acquisition(db: Session, spectrum_id, acquisition):
    s = db.query(Spectrum).filter(Spectrum.id == _uuid(spectrum_id, "Spectrum")).first()
    if s is None:
        raise NotFound("Spectrum not found")
    s.acquisition_type = normalize_acquisition(acquisition)
    db.commit()
    return spectrum_dict(s)


def get_spectrum(db: Session, spectrum_id):
    s = db.query(Spectrum).filter(Spectrum.id == _uuid(spectrum_id, "Spectrum")).first()
    if s is None:
        raise NotFound("Spectrum not found")
    return spectrum_dict(s)


def _grid(wavelengths):
    """(start, end, step or None, explicit list or None) - step is only stored for a regular grid."""
    wl = np.asarray(wavelengths, float).ravel()
    if wl.size < 2:
        raise ValueError("a spectrum needs at least two wavelengths")
    d = np.diff(wl)
    step = float(np.median(d))
    regular = step > 0 and np.allclose(d, step, rtol=1e-4, atol=1e-9)
    if regular:
        return float(wl[0]), float(wl[-1]), round(step, 9), None
    return float(wl[0]), float(wl[-1]), None, [float(v) for v in wl]


def add_spectrum(db: Session, library_id, dataset_id, name: str, wavelengths, values, *, category=None,
                 description=None, std=None, unit=None, wavelength_unit="nm", material_metadata=None,
                 spectrum_metadata=None, reuse_material=True, commit=True, acquisition=None, sample_name=None,
                 roi_id=None, roi_class=None, roi_ids=None):
    """Store one spectrum as Material -> Sample -> Spectrum (the backend's data model).

    If a material of that name already exists in the library (and reuse_material), the spectrum becomes its
    next sample - "a material can have multiple measurement samples" in schema.sql.
    roi_id / roi_class / roi_ids (database ids of rois rows) record which ROI(s) the spectrum came from.
    """
    lid, did = _uuid(library_id, "Library"), _uuid(dataset_id, "Dataset")
    vals = _clean(values)
    if len(vals) != len(np.asarray(wavelengths).ravel()):
        raise ValueError("wavelengths and values differ in length")
    start, end, step, explicit = _grid(wavelengths)
    mat = None
    if reuse_material:
        mat = db.query(Material).filter(Material.library_id == lid, Material.name == name).first()
    if mat is None:
        mat = Material(library_id=lid, name=name[:500], category=normalize_category(category),
                       description=description or None, material_metadata=material_metadata or {})
        db.add(mat)
        db.flush()
    n = (db.query(func.max(Sample.sample_number)).filter(Sample.material_id == mat.id).scalar() or 0) + 1
    sname = sample_name or (name if n == 1 else f"{name} #{n}")
    sample = Sample(material_id=mat.id, dataset_id=did, name=sname[:500], sample_number=n,
                    sample_metadata=dict(spectrum_metadata or {}))
    db.add(sample)
    db.flush()
    meta = {"wavelength_unit": wavelength_unit, "point_count": len(vals), **(spectrum_metadata or {})}
    if explicit is not None:
        meta["wavelengths"] = explicit
    if std is not None:
        meta["std"] = _clean(std)
    sp = Spectrum(sample_id=sample.id, wavelength_start=start, wavelength_end=end, wavelength_step=step,
                  measurement_unit=unit, values=vals, spectrum_metadata=meta,
                  acquisition_type=normalize_acquisition(acquisition),
                  roi_id=None if roi_id is None else _uuid(roi_id, "ROI"), roi_class=roi_class or None)
    db.add(sp)
    links = {_uuid(r, "ROI") for r in (roi_ids or [])} | ({sp.roi_id} if sp.roi_id else set())
    if links:
        db.flush()
        db.add_all(SpectrumRoi(spectrum_id=sp.id, roi_id=r) for r in links)
    if commit:
        db.commit()
    return {"material": material_dict(mat), "sample": sample_dict(sample), "spectrum_id": str(sp.id)}


# ------------------------------------------------------------------ ROIs (migration 002)
ROI_MODES = ("class", "roi", "pixels")


def upsert_roi(db: Session, library_id, rec: dict, dataset_id=None) -> Roi:
    """Insert or update one ROI (keyed by library + roi_uid).

    rec: {uid, name, class, scene_name, scene_path, image_rows, image_cols, kind, geometry, geometry_map, crs,
          bbox [rmin, rmax, cmin, cmax], n_pixels, metadata}
    """
    lid = _uuid(library_id, "Library")
    uid = str(rec.get("uid") or "").strip()
    cls = str(rec.get("class") or "").strip()
    kind = rec.get("kind")
    if not uid or not cls:
        raise ValueError("a ROI needs an id and a class name")
    if kind not in ("rect", "polygon"):
        raise ValueError(f"ROI kind must be rect or polygon (got {kind!r})")
    geom = rec.get("geometry") or {}
    if (kind == "rect" and len(geom.get("rect") or []) != 4) or (kind == "polygon" and len(geom.get("vertices") or []) < 3):
        raise ValueError(f"ROI {rec.get('name') or uid}: geometry does not match kind {kind}")
    r = db.query(Roi).filter(Roi.library_id == lid, Roi.roi_uid == uid).first()
    if r is None:
        r = Roi(library_id=lid, roi_uid=uid[:64])
        db.add(r)
    bbox = rec.get("bbox") or [None] * 4
    r.dataset_id = None if dataset_id is None else _uuid(dataset_id, "Dataset")
    r.roi_name = (rec.get("name") or None) and str(rec["name"])[:255]
    r.class_name = cls[:255]
    r.scene_name = (rec.get("scene_name") or None) and str(rec["scene_name"])[:500]
    r.scene_path = rec.get("scene_path") or None
    r.image_rows, r.image_cols = rec.get("image_rows"), rec.get("image_cols")
    r.geometry_type, r.geometry = kind, geom
    r.geometry_map = rec.get("geometry_map") or None
    r.crs = rec.get("crs") or None
    r.bbox_row_min, r.bbox_row_max, r.bbox_col_min, r.bbox_col_max = bbox
    r.n_pixels = rec.get("n_pixels")
    r.roi_metadata = rec.get("metadata") or {}
    db.flush()
    return r


def save_roi_spectra(db: Session, library_id, dataset_name: str, rois: list, items: list, category=None,
                     acquisition=None, skip_existing=True, commit=True):
    """Store ROIs and the spectra computed from them, class by class, in one transaction.

    rois   [roi record (see upsert_roi), ...]  - saved/updated in the rois table
    items  [{name, class, wavelengths, values, std?, sample_name?, category?, roi_uids: [...],
             uid?, material_metadata?, spectrum_metadata?}, ...]
           one item = one spectrum: a pooled class mean (roi_uids = all ROIs of the class), the mean of one ROI
           (roi_uids = [that ROI]) or one ROI pixel.  Items with a single ROI get spectra.roi_id set.
    uid    optional idempotency key: an item whose uid already exists in this library is skipped.
    """
    lid = _uuid(library_id, "Library")
    ds = get_or_create_dataset(db, lid, dataset_name, source_name="HyperVista ROI analysis",
                               description="Spectra of regions of interest saved from HyperVista.", commit=False)
    by_uid = {}
    for rec in rois:
        by_uid[str(rec["uid"])] = upsert_roi(db, lid, rec, ds.id)
    existing = set()
    uids = {it["uid"] for it in items if it.get("uid")}
    if skip_existing and uids:
        for (meta,) in (db.query(Spectrum.spectrum_metadata).join(Sample, Sample.id == Spectrum.sample_id)
                        .join(Material, Material.id == Sample.material_id).filter(Material.library_id == lid)
                        .filter(Spectrum.roi_class.isnot(None))):
            if meta and meta.get("uid") in uids:
                existing.add(meta["uid"])
    saved, skipped = [], 0
    for it in items:
        if it.get("uid") in existing:
            skipped += 1
            continue
        ids = []
        for u in it.get("roi_uids") or []:
            r = by_uid.get(str(u)) or db.query(Roi).filter(Roi.library_id == lid, Roi.roi_uid == str(u)).first()
            if r is None:
                raise NotFound(f"ROI {u} is not in this library")
            ids.append(r.id)
        meta = dict(it.get("spectrum_metadata") or {})
        if it.get("uid"):
            meta["uid"] = it["uid"]
        res = add_spectrum(db, lid, ds.id, it["name"], it["wavelengths"],
                           [np.nan if v is None else v for v in it["values"]],
                           category=it.get("category") or category,
                           std=None if it.get("std") is None else [np.nan if v is None else v for v in it["std"]],
                           material_metadata=it.get("material_metadata"), spectrum_metadata=meta, commit=False,
                           acquisition=it.get("acquisition_type") or acquisition, sample_name=it.get("sample_name"),
                           roi_id=ids[0] if len(ids) == 1 else None, roi_class=it.get("class") or it["name"],
                           roi_ids=ids)
        saved.append(res)
    if commit:
        db.commit()
    return {"dataset": dataset_dict(ds), "rois": len(by_uid), "spectra": len(saved), "skipped": skipped,
            "classes": sorted({it.get("class") or it["name"] for it in items}),
            "roi_ids": {u: str(r.id) for u, r in by_uid.items()}}


def _roi_spectrum_counts(db: Session, roi_ids) -> dict:
    if not roi_ids:
        return {}
    rows = (db.query(SpectrumRoi.roi_id, func.count(SpectrumRoi.spectrum_id))
            .filter(SpectrumRoi.roi_id.in_(roi_ids)).group_by(SpectrumRoi.roi_id).all())
    return {k: n for k, n in rows}


def list_rois(db: Session, library_id=None, scene_name: Optional[str] = None, class_name: Optional[str] = None):
    """ROIs of a library (or of all libraries), optionally only one image / one class.
    Each dict has 'n_spectra' (spectra computed from it) and 'library_name'."""
    q = db.query(Roi, Library.name).join(Library, Library.id == Roi.library_id)
    if library_id is not None:
        q = q.filter(Roi.library_id == _uuid(library_id, "Library"))
    if scene_name:
        q = q.filter(Roi.scene_name == scene_name)
    if class_name:
        q = q.filter(Roi.class_name == class_name)
    rows = q.order_by(Library.name, Roi.class_name, Roi.scene_name, Roi.roi_name).all()
    counts = _roi_spectrum_counts(db, [r.id for r, _ in rows])
    return [roi_dict(r, library_name=ln, n_spectra=counts.get(r.id, 0)) for r, ln in rows]


def get_roi(db: Session, roi_id):
    r = db.query(Roi).filter(Roi.id == _uuid(roi_id, "ROI")).first()
    if r is None:
        raise NotFound("ROI not found")
    return roi_dict(r, n_spectra=_roi_spectrum_counts(db, [r.id]).get(r.id, 0))


def roi_classes(db: Session, library_id):
    """Class-by-class summary: [{class_name, n_rois, n_scenes, n_pixels, n_spectra, scenes}, ...]."""
    lid = _uuid(library_id, "Library")
    rows = db.query(Roi).filter(Roi.library_id == lid).all()
    spec = dict(db.query(Spectrum.roi_class, func.count(Spectrum.id)).join(Sample, Sample.id == Spectrum.sample_id)
                .join(Material, Material.id == Sample.material_id)
                .filter(Material.library_id == lid, Spectrum.roi_class.isnot(None))
                .group_by(Spectrum.roi_class).all())
    out = {}
    for r in rows:
        d = out.setdefault(r.class_name, {"class_name": r.class_name, "n_rois": 0, "n_pixels": 0, "scenes": set()})
        d["n_rois"] += 1
        d["n_pixels"] += r.n_pixels or 0
        if r.scene_name:
            d["scenes"].add(r.scene_name)
    res = []
    for name in sorted(set(out) | set(spec), key=str.lower):
        d = out.get(name, {"class_name": name, "n_rois": 0, "n_pixels": 0, "scenes": set()})
        res.append(dict(d, scenes=sorted(d["scenes"]), n_scenes=len(d["scenes"]), n_spectra=spec.get(name, 0)))
    return res


def roi_spectra(db: Session, roi_id=None, library_id=None, class_name: Optional[str] = None):
    """Spectra computed from one ROI (roi_id), or of one class in a library (library_id + class_name).
    -> [{"material", "sample", "spectrum", "roi_ids"}]"""
    q = (db.query(Spectrum, Sample, Material).join(Sample, Sample.id == Spectrum.sample_id)
         .join(Material, Material.id == Sample.material_id))
    if roi_id is not None:
        rid = _uuid(roi_id, "ROI")
        q = q.filter(Spectrum.id.in_(db.query(SpectrumRoi.spectrum_id).filter(SpectrumRoi.roi_id == rid)
                                     .scalar_subquery()))
    else:
        if library_id is None or not class_name:
            raise ValueError("give a roi_id, or a library_id and a class_name")
        q = q.filter(Material.library_id == _uuid(library_id, "Library"), Spectrum.roi_class == class_name)
    rows = q.order_by(Material.name, Sample.sample_number).all()
    links = {}
    if rows:
        for sid, rid in db.query(SpectrumRoi.spectrum_id, SpectrumRoi.roi_id).filter(
                SpectrumRoi.spectrum_id.in_([s.id for s, _, _ in rows])):
            links.setdefault(sid, []).append(str(rid))
    return [{"material": material_dict(m), "sample": sample_dict(sa), "spectrum": spectrum_dict(sp),
             "roi_ids": links.get(sp.id, [])} for sp, sa, m in rows]


def spectrum_roi_info(db: Session, spectrum_id):
    """The ROIs a spectrum was computed from (for the Spectrum info panel)."""
    sid = _uuid(spectrum_id, "Spectrum")
    rows = (db.query(Roi).join(SpectrumRoi, SpectrumRoi.roi_id == Roi.id).filter(SpectrumRoi.spectrum_id == sid)
            .order_by(Roi.scene_name, Roi.roi_name).all())
    return [roi_dict(r) for r in rows]


def delete_roi(db: Session, roi_id, delete_spectra: bool = False):
    """Delete a ROI. Its spectra stay (roi_id -> NULL) unless delete_spectra: then the spectra computed only
    from this ROI are deleted too (a pooled class mean that also used other ROIs is kept)."""
    r = db.query(Roi).filter(Roi.id == _uuid(roi_id, "ROI")).first()
    if r is None:
        raise NotFound("ROI not found")
    n_deleted = 0
    if delete_spectra:
        spec_ids = [s for (s,) in db.query(SpectrumRoi.spectrum_id).filter(SpectrumRoi.roi_id == r.id)]
        only = [s for s in spec_ids
                if db.query(SpectrumRoi).filter(SpectrumRoi.spectrum_id == s, SpectrumRoi.roi_id != r.id).count() == 0]
        if only:
            sample_ids = [x for (x,) in db.query(Spectrum.sample_id).filter(Spectrum.id.in_(only))]
            db.query(SpectrumRoi).filter(SpectrumRoi.spectrum_id.in_(only)).delete(synchronize_session=False)
            n_deleted = db.query(Spectrum).filter(Spectrum.id.in_(only)).delete(synchronize_session=False)
            empty = [x for x in set(sample_ids) if db.query(Spectrum.id).filter(Spectrum.sample_id == x).count() == 0]
            mats = {m for (m,) in db.query(Sample.material_id).filter(Sample.id.in_(empty))}
            db.query(Sample).filter(Sample.id.in_(empty)).delete(synchronize_session=False)
            for m in mats:
                if db.query(Sample.id).filter(Sample.material_id == m).count() == 0:
                    db.query(Material).filter(Material.id == m).delete(synchronize_session=False)
    db.query(SpectrumRoi).filter(SpectrumRoi.roi_id == r.id).delete(synchronize_session=False)
    db.query(Spectrum).filter(Spectrum.roi_id == r.id).update({Spectrum.roi_id: None}, synchronize_session=False)
    db.delete(r)
    db.commit()
    return {"deleted_spectra": n_deleted}


def library_spectra(db: Session, library_id, limit: Optional[int] = None, acquisition: Optional[str] = None,
                    category_path: Optional[str] = None):
    """First spectrum of the first sample of every material in a library (for matching / bulk plotting).
    acquisition / category_path restrict it like list_materials does (the spectrum itself must carry the tag)."""
    lid = _uuid(library_id, "Library")
    q = (db.query(Material, Spectrum)
         .join(Sample, Sample.material_id == Material.id)
         .join(Spectrum, Spectrum.sample_id == Sample.id)
         .filter(Material.library_id == lid))
    if acquisition:
        q = q.filter(Spectrum.acquisition_type.is_(None) if acquisition == UNTAGGED
                     else Spectrum.acquisition_type == normalize_acquisition(acquisition))
    if category_path:
        cond = _category_filter(category_path)
        if cond is not None:
            q = q.filter(cond)
    q = (q
         .order_by(Material.name, Sample.sample_number, Spectrum.created_at))
    out, seen = [], set()
    for m, s in q.yield_per(500):
        if m.id in seen:
            continue
        seen.add(m.id)
        out.append({"material": material_dict(m), "spectrum": spectrum_dict(s)})
        if limit and len(out) >= limit:
            break
    return out


def spectral_angle_rank(spectrum, wavelengths, entries: Iterable[dict], top=5):
    """Rank library entries ({'material','spectrum'}) by spectral angle (radians) to *spectrum*.

    The query spectrum is resampled onto each reference's grid; only the overlapping bands are compared.
    """
    from ..core.resample import resample_spectra
    from ..core.signature import spectral_angle
    x = np.asarray(spectrum, np.float32)[None]
    res = []
    for e in entries:
        wl_ref, ref = spectrum_wavelengths(e["spectrum"]), spectrum_values(e["spectrum"])
        q = resample_spectra(x, wavelengths, wl_ref)[0]
        ok = ~(np.isnan(q) | np.isnan(ref))
        if ok.sum() < 3:
            continue
        res.append((spectral_angle(q[ok], ref[ok]), e["material"], int(ok.sum())))
    res.sort(key=lambda r: r[0])
    return res[:top]
