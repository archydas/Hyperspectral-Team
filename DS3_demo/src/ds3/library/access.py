"""Read-side helpers shared by the desktop app, CLI and API (no GUI imports)."""
from __future__ import annotations
import numpy as np
from sqlalchemy import select, func
from ..db.models import (SpectrumRecord, Sample, Material, Dataset, Library, CubeRecord, Source)
from ..ingest.contract import Spectrum


def list_libraries(s):
    q = (select(Library.name, func.count(SpectrumRecord.id))
         .select_from(Library)
         .outerjoin(Dataset, Dataset.library_id == Library.id)
         .outerjoin(Sample, Sample.dataset_id == Dataset.id)
         .outerjoin(SpectrumRecord, SpectrumRecord.sample_id == Sample.id)
         .group_by(Library.name).order_by(Library.name))
    return [dict(name=n, n_spectra=c) for n, c in s.execute(q)]


def list_categories(s):
    return [c for (c,) in s.execute(select(Material.category).where(Material.category.is_not(None))
                                    .distinct().order_by(Material.category))]


def list_cubes(s):
    out = []
    for c, src in s.execute(select(CubeRecord, Source.name).outerjoin(Source, CubeRecord.source_id == Source.id)
                            .order_by(CubeRecord.id)):
        out.append(dict(id=c.id, name=c.name, path=c.path, source=src, format=c.format, rows=c.rows, cols=c.cols,
                        bands=c.bands, unit=c.unit, sensor=c.sensor, wl_min_nm=c.wl_min_nm, wl_max_nm=c.wl_max_nm,
                        qc_passed=None if not c.qc else c.qc.get("passed"), crs=c.crs))
    return out


def db_stats(s):
    one = lambda q: s.scalar(q) or 0
    return dict(libraries=one(select(func.count(Library.id))), materials=one(select(func.count(Material.id))),
                spectra=one(select(func.count(SpectrumRecord.id))), cubes=one(select(func.count(CubeRecord.id))),
                sources=one(select(func.count(Source.id))))


def get_spectra(s, ids):
    """Full stored spectra (wavelengths + values, NaN restored) for the given spectrum ids, in the order given."""
    ids = [int(i) for i in ids]
    if not ids:
        return []
    rows = s.execute(select(SpectrumRecord, Material.name, Material.category, Library.name)
                     .select_from(SpectrumRecord)
                     .join(Sample, SpectrumRecord.sample_id == Sample.id)
                     .join(Material, Sample.material_id == Material.id)
                     .join(Dataset, Sample.dataset_id == Dataset.id)
                     .join(Library, Dataset.library_id == Library.id)
                     .where(SpectrumRecord.id.in_(ids))).all()
    by_id = {}
    for rec, mat, cat, lib in rows:
        by_id[rec.id] = dict(id=rec.id, material=mat, category=cat, library=lib, unit=rec.unit, purity=rec.purity,
                             wavelengths_nm=np.asarray(rec.wavelengths_nm, float),
                             values=np.array([np.nan if x is None else x for x in rec.values], float))
    return [by_id[i] for i in ids if i in by_id]


def to_spectrum(d) -> Spectrum:
    return Spectrum(material=d["material"], wavelengths_nm=d["wavelengths_nm"], values=d["values"], unit=d["unit"],
                    purity=d["purity"], category=d.get("category"), metadata=dict(origin="database", source_id=d["id"]))


def delete_library(s, name):
    lib = s.scalar(select(Library).where(Library.name == name))
    if lib is None:
        return False
    s.delete(lib)
    return True


def open_registered_cube(s, cube_id):
    """Re-open a cube registered in the database with the SAME reader options it was ingested with
    (unit, wavelengths, reflectance_divisor) and the stored bad-band mask. Returns (cube, record_dict)."""
    from ..ingest import open_cube, IngestError
    c = s.get(CubeRecord, int(cube_id))
    if c is None:
        raise IngestError(f"cube id {cube_id} not found")
    kw = dict((c.meta or {}).get("reader_kw") or {})
    kw.setdefault("unit", c.unit)
    cube = open_cube(c.path, qc=False, **kw)
    if c.bad_band_mask and len(c.bad_band_mask) == cube.bands:
        cube.bad_band_mask = np.asarray(c.bad_band_mask, bool)
    return cube, dict(id=c.id, name=c.name, path=c.path, rows=c.rows, cols=c.cols, bands=c.bands, unit=c.unit)
