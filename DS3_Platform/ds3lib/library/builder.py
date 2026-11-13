"""Spectral Library Builder: many inputs -> ONE library on ONE wavelength grid."""
from __future__ import annotations
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
from ..ingest import open_cube, IngestError
from ..ingest.contract import Spectrum
from ..spectra import read_spectra, detect_spectra_format
from ..db import session_scope, init_db
from ..db.models import Source
from ..pipeline import store_spectra, _get_source, sha256_of
from .grid import Grid, resample
from .sources import is_envi_roi, spectra_from_roi_ascii, spectra_from_geojson, read_evf


@dataclass
class BuildReport:
    library: str
    grid_origin: str
    n_bands: int
    method: str
    added: list = field(default_factory=list)      # dicts: material, from, coverage
    rejected: list = field(default_factory=list)   # dicts: input, reason
    spectrum_ids: list = field(default_factory=list)

    def summary(self):
        lines = [f"library '{self.library}' | grid {self.grid_origin} ({self.n_bands} bands) | {self.method}",
                 f"  added {len(self.added)}, rejected {len(self.rejected)}"]
        lines += [f"  + {a['material']:30s} <- {a['from']}  coverage {a['coverage']*100:.0f}%" for a in self.added]
        lines += [f"  ! {r['input']}: {r['reason']}" for r in self.rejected]
        return "\n".join(lines)


def _inputs_to_spectra(item, cube, one_based):
    p = Path(item)
    if p.suffix.lower() == ".evf": read_evf(p)
    if p.suffix.lower() in (".json", ".geojson"):
        if cube is None: raise IngestError("a --cube is needed to measure spectra from GeoJSON polygons")
        return spectra_from_geojson(p, cube)
    if is_envi_roi(p):
        if cube is None: raise IngestError("a --cube is needed to measure spectra from an ROI file")
        return spectra_from_roi_ascii(p, cube, one_based)
    kw = {}
    if detect_spectra_format(p) == "usgs":
        sib = sorted(p.parent.glob("*avelength*.txt"))
        if sib: kw["wavelengths_file"] = sib[0]
    out = read_spectra(p, **kw)
    for s in out: s.metadata.setdefault("origin", "field" if s.purity == "ground_measured" else "library_import")
    return out


def build_library(name, inputs, grid: Grid | str, *, cube=None, method="linear", unit=None, one_based=True,
                  min_coverage=0.5, source_name="library-builder", db_url=None, spectrum_ids=None) -> BuildReport:
    """grid: a Grid, or the literal 'first' (ENVI's 'First Input Spectrum').
    cube: path to the image the ROI/GeoJSON inputs are measured from.
    Spectra with < min_coverage of the grid covered, or in a different unit from the library, are REJECTED (reported)."""
    init_db(db_url)
    cube_obj = open_cube(cube) if cube else None
    collected, rejected = [], []
    for item in inputs:
        try:
            for sp in _inputs_to_spectra(item, cube_obj, one_based):
                collected.append((str(item), sp))
        except (IngestError, OSError, ValueError, KeyError) as e:
            rejected.append(dict(input=str(item), reason=f"{type(e).__name__}: {e}"))
    if spectrum_ids:                                   # spectra already in the database (e.g. rows picked in the app)
        from .access import get_spectra, to_spectrum
        with session_scope(db_url) as s:
            collected += [(f"db#{d['id']}", to_spectrum(d)) for d in get_spectra(s, spectrum_ids)]
    if not collected:
        raise IngestError("no usable input spectra:\n" + "\n".join(r["input"] + ": " + r["reason"] for r in rejected))
    return _finish_build(name, collected, grid, rejected, method=method, unit=unit, min_coverage=min_coverage,
                         source_name=source_name, db_url=db_url,
                         digest_seed="|".join(sorted(sha256_of(i) for i in inputs if Path(i).exists()))
                         + ("|db:" + ",".join(str(i) for i in spectrum_ids) if spectrum_ids else ""))


def build_library_from_ids(name, spectrum_ids, grid: Grid | str, *, method="linear", unit=None, min_coverage=0.5,
                           source_name="library-builder", db_url=None) -> BuildReport:
    """Compose a NEW library from spectra ALREADY in the database (e.g. rows picked in the app)."""
    from .access import get_spectra, to_spectrum
    init_db(db_url)
    with session_scope(db_url) as s:
        recs = get_spectra(s, spectrum_ids)
    if not recs:
        raise IngestError("none of the selected spectrum ids exist")
    collected = [(f"db#{d['id']}", to_spectrum(d)) for d in recs]
    return _finish_build(name, collected, grid, [], method=method, unit=unit, min_coverage=min_coverage,
                         source_name=source_name, db_url=db_url,
                         digest_seed="db:" + ",".join(str(d["id"]) for d in recs))


def _finish_build(name, collected, grid, rejected, *, method, unit, min_coverage, source_name, db_url, digest_seed):
    if isinstance(grid, str):
        if grid != "first": raise ValueError("grid must be a Grid or 'first'")
        grid = Grid.from_spectrum(collected[0][1])
    want_unit = unit or collected[0][1].unit
    rep = BuildReport(name, grid.origin, len(grid), method)
    rep.rejected.extend(rejected)
    out = []
    for src_item, sp in collected:
        if sp.unit != want_unit:
            rep.rejected.append(dict(input=f"{Path(src_item).name}:{sp.material}", reason=f"unit {sp.unit} != library unit {want_unit} (not converted - radiance and reflectance cannot be mixed)"))
            continue
        v = np.asarray(sp.values, float).copy()
        if sp.bad_band_mask is not None: v[np.asarray(sp.bad_band_mask, bool)] = np.nan
        vals, cov = resample(sp.wavelengths_nm, v, grid, method)
        if cov < min_coverage:
            rep.rejected.append(dict(input=f"{Path(src_item).name}:{sp.material}", reason=f"only {cov*100:.0f}% of the grid is covered by this spectrum ({sp.wavelengths_nm[0]:.0f}-{sp.wavelengths_nm[-1]:.0f} nm)"))
            continue
        meta = dict(sp.metadata, resampled=dict(method=method, grid=grid.origin, coverage=round(cov, 4),
                    source_bands=len(sp.values), source_range_nm=[float(sp.wavelengths_nm[0]), float(sp.wavelengths_nm[-1])]))
        out.append(Spectrum(material=sp.material, wavelengths_nm=grid.wavelengths_nm, values=vals, unit=sp.unit,
                            purity=sp.purity, fwhm_nm=grid.fwhm_nm, category=sp.category, metadata=meta))
        rep.added.append(dict(material=sp.material, **{"from": Path(src_item).name}, coverage=cov))
    if not out:
        raise IngestError("every input was rejected:\n" + rep.summary())
    digest = hashlib.sha256((digest_seed + name + method +
                             ",".join(f"{w:.3f}" for w in grid.wavelengths_nm)).encode()).hexdigest()
    from sqlalchemy import select
    from ..db.models import Dataset
    with session_scope(db_url) as s:
        if s.scalar(select(Dataset).where(Dataset.sha256 == digest)):
            raise IngestError(f"library '{name}' was already built from these exact inputs/grid/method (nothing changed)")
        src = _get_source(s, source_name, "library")
        rep.spectrum_ids = store_spectra(s, out, source=src, library_name=name, dataset_name=f"build:{len(collected)} inputs",
                                         digest=digest, fmt="builder", extra_meta=None)
    return rep
