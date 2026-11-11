"""Search the spectra database.

find_spectra : metadata filters (text, category, unit, purity, library, wavelength coverage) with paging - runs in SQL.
match_spectrum: spectral similarity (SAM / Euclidean) of a query against stored spectra - streamed in chunks.
SCALING NOTE: match_spectrum reads candidates in chunks of `chunk` rows (memory stays flat), but it is still
O(N) per query. When a library passes ~10^5 spectra the next step is a pre-resampled matrix cache or pgvector.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from sqlalchemy import select, or_, func
from ..db.models import SpectrumRecord, Sample, Material, Dataset, Library
from .grid import Grid, resample


def _base():
    return (select(SpectrumRecord, Material.name, Material.category, Library.name)
            .select_from(SpectrumRecord)
            .join(Sample, SpectrumRecord.sample_id == Sample.id)
            .join(Material, Sample.material_id == Material.id)
            .join(Dataset, Sample.dataset_id == Dataset.id)
            .join(Library, Dataset.library_id == Library.id))


def _filters(q, text=None, category=None, unit=None, purity=None, library=None, covers_nm=None):
    if text:
        t = f"%{text}%"
        q = q.where(or_(Material.name.ilike(t), Material.category.ilike(t), Sample.label.ilike(t), Library.name.ilike(t)))
    if category: q = q.where(Material.category.ilike(category))
    if unit: q = q.where(SpectrumRecord.unit == unit)
    if purity: q = q.where(SpectrumRecord.purity == purity)
    if library: q = q.where(Library.name == library)
    if covers_nm:                                   # only spectra spanning the whole [lo, hi] nm window
        lo, hi = covers_nm
        q = q.where(SpectrumRecord.wl_min_nm <= lo, SpectrumRecord.wl_max_nm >= hi)
    return q


def _row(r):
    rec, mat, cat, lib = r
    return dict(id=rec.id, material=mat, category=cat, library=lib, unit=rec.unit, purity=rec.purity,
                n_bands=rec.n_bands, wl_min_nm=rec.wl_min_nm, wl_max_nm=rec.wl_max_nm)


def find_spectra(s, *, limit=50, offset=0, **filt):
    q = _filters(_base(), **filt).order_by(SpectrumRecord.id).limit(limit).offset(offset)
    total = s.scalar(_filters(select(func.count(SpectrumRecord.id)).select_from(SpectrumRecord)
                              .join(Sample, SpectrumRecord.sample_id == Sample.id)
                              .join(Material, Sample.material_id == Material.id)
                              .join(Dataset, Sample.dataset_id == Dataset.id)
                              .join(Library, Dataset.library_id == Library.id), **filt))
    return dict(total=total, limit=limit, offset=offset, results=[_row(r) for r in s.execute(q)])


def spectral_angle(a, b):
    """SAM in radians over bands where both are finite."""
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3: return np.nan
    a, b = a[m], b[m]
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.arccos(np.clip(np.dot(a, b) / d, -1, 1))) if d > 0 else np.nan


def match_spectrum(s, wavelengths_nm, values, *, top_k=5, metric="sam", method="linear", min_overlap=0.5,
                   chunk=500, exclude_id=None, **filt):
    """Rank stored spectra by similarity to the query. Both are put on the QUERY's wavelength grid first,
    so any unit-matched spectra from any sensor/instrument can be compared. unit filter defaults to the query-unit
    only if you pass unit=...; mixing radiance and reflectance gives meaningless scores - always pass unit."""
    grid = Grid(np.asarray(wavelengths_nm, float), origin="query")
    q = np.asarray(values, float)
    best = []
    last = 0
    while True:
        stmt = _filters(_base(), **filt).where(SpectrumRecord.id > last).order_by(SpectrumRecord.id).limit(chunk)
        rows = s.execute(stmt).all()
        if not rows: break
        for r in rows:
            rec = r[0]; last = max(last, rec.id)
            if exclude_id is not None and rec.id == exclude_id: continue
            v = np.array([np.nan if x is None else x for x in rec.values], float)
            on, cov = resample(rec.wavelengths_nm, v, grid, method)
            overlap = np.isfinite(on) & np.isfinite(q)
            if overlap.mean() < min_overlap: continue
            sc = spectral_angle(q, on) if metric == "sam" else float(np.sqrt(np.nanmean((q[overlap] - on[overlap]) ** 2)))
            if np.isfinite(sc): best.append((sc, dict(_row(r), score=sc, overlap=float(overlap.mean()))))
        best.sort(key=lambda t: t[0]); best = best[:top_k]
    return [b[1] for b in best]
