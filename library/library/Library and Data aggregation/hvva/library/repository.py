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
"""
from __future__ import annotations

import math
import re
import uuid
from typing import Iterable, Optional

import numpy as np
from sqlalchemy import and_, exists, func, or_
from sqlalchemy.orm import Session

from .models import DataFile, Dataset, Library, Material, Sample, Spectrum

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
            "acquisition_type": x.acquisition_type, "metadata": x.spectrum_metadata or {}}


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
    db.query(Material).filter(Material.id.in_(mat_ids)).delete(synchronize_session=False)
    db.query(Dataset).filter(Dataset.id.in_(ds_ids)).delete(synchronize_session=False)
    db.delete(lib)
    db.commit()


def _delete_samples(db, sample_id_query):
    ids = [s for (s,) in sample_id_query]
    if ids:
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
                 spectrum_metadata=None, reuse_material=True, commit=True, acquisition=None):
    """Store one spectrum as Material -> Sample -> Spectrum (the backend's data model).

    If a material of that name already exists in the library (and reuse_material), the spectrum becomes its
    next sample - "a material can have multiple measurement samples" in schema.sql.
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
    sample = Sample(material_id=mat.id, dataset_id=did, name=name if n == 1 else f"{name} #{n}", sample_number=n,
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
                  acquisition_type=normalize_acquisition(acquisition))
    db.add(sp)
    if commit:
        db.commit()
    return {"material": material_dict(mat), "sample": sample_dict(sample), "spectrum_id": str(sp.id)}


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
