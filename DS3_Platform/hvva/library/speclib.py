"""Import public reference spectral libraries straight into the material library database.

Supported sources (download them yourself - links in README / docs):

  usgs       USGS Spectral Library Version 7 (splib07), ASCII data release, e.g.
             ASCIIdata_splib07b_cvASD.zip (convolved to ASD, 2151 ch, 350-2500 nm) or ASCIIdata_splib07a.zip.
             Layout: Chapter folders (ChapterM_Minerals, ChapterV_Vegetation, ...), one spectrum per .txt
             (title line, then one value per line, bad bands = -1.23e34) and separate *Wavelengths*.txt files
             (microns).  Category  = <Chapter>/<material>, e.g. Minerals/Alunite, Vegetation/Aspen.
             Acquisition = airborne for AVIRIS measurements, otherwise lab (Beckman, Nicolet, ASD in the lab);
             override with --acquisition when you know better (some ASD spectra were measured in the field).

  ecostress  ECOSTRESS / ASTER spectral library (JPL), *.spectrum.txt files with "Key: value" headers
             (Name, Type, Class, Subclass / Genus, Species, X Units, Y Units, ...) and two data columns.
             Category  = Type/Class/Subclass (minerals, rocks, man-made ...) or Type/Class/Genus/Species
             (vegetation), e.g. Vegetation/Tree/Acer/Rubrum.  Acquisition = lab (all laboratory measurements).
             Wavelengths are converted to nm, percent reflectance to 0-1.  TIR-only files are skipped by default.

  openspeclib  OpenSpecLib release Parquet files (https://github.com/null-jones/openspeclib/releases) - the
             USGS splib07, ECOSTRESS and EcoSIS libraries already normalised, one file per source, direct download,
             no form.  Needs pyarrow and the release's wavelengths.parquet next to the file (or --wavelengths).
             Category = Category/Subcategory/Genus/Species for vegetation (from the species name), else
             Category/Subcategory/Material; acquisition = airborne for AVIRIS instruments, otherwise lab.

usgs / ecostress accept a folder or the downloaded .zip (read in place, no extraction).  Re-running adds nothing
that is already in the dataset.  By default only the VSWIR part (<= 2500 nm) of spectra starting below 1000 nm is
kept (that is what imaging spectrometers see); --include-tir keeps thermal-infrared spectra unchanged.

    python -m hvva.library.speclib usgs ASCIIdata_splib07b_cvASD.zip --library "USGS splib07" --chapters M,V,A
    python -m hvva.library.speclib ecostress ecospeclib-all/ --library "ECOSTRESS" --limit 300
    python -m hvva.library.speclib openspeclib ecostress.parquet --library "ECOSTRESS" --categories vegetation,npv
"""
from __future__ import annotations

import argparse
import hashlib
import io
import os
import re
import zipfile
from pathlib import Path, PurePosixPath
from typing import Callable, Dict, Iterator, List, Optional

import numpy as np
from sqlalchemy.orm import Session

from . import repository as repo
from .models import DataFile, Library, Material, Sample, Spectrum

BAD = -1.0e30                      # USGS deleted channels are -1.23e34; anything below this is a bad band
USGS_CHAPTERS = {"M": "Minerals", "S": "Soils and Mixtures", "C": "Coatings", "L": "Liquids",
                 "O": "Organic Compounds", "A": "Artificial Materials", "V": "Vegetation"}
_SENSOR = re.compile(r"(BECK[a-z]?|ASD[A-Z]{2}[a-z]?|NIC4[a-z]*|AVIRIS[a-z0-9]*|AV\d\d)", re.I)


# ================================================================== file access (folder or zip)
class _Source:
    """Uniform listing/reading of files in a folder or a zip archive."""

    def __init__(self, path):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"not found: {self.path}")
        self.zip = zipfile.ZipFile(self.path) if self.path.is_file() and zipfile.is_zipfile(self.path) else None
        if self.zip is None and not self.path.is_dir():
            raise ValueError(f"{self.path} is neither a folder nor a .zip archive")

    def names(self) -> List[str]:
        if self.zip:
            return [n for n in self.zip.namelist() if not n.endswith("/")]
        return [str(p.relative_to(self.path)).replace(os.sep, "/") for p in self.path.rglob("*") if p.is_file()]

    def text(self, name) -> str:
        raw = self.zip.read(name) if self.zip else (self.path / name).read_bytes()
        return raw.decode("utf-8", errors="replace")

    def checksum(self) -> Optional[str]:
        return file_sha256(self.path)


def _floats(lines) -> List[float]:
    out = []
    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue
        try:
            out.append(float(ln.split()[0]))
        except ValueError:
            continue
    return out


# ================================================================== USGS splib07 ASCII
def _usgs_wavelength_files(src: _Source, names) -> Dict[str, np.ndarray]:
    """{file name: wavelengths in nm} for every *Wavelength* file (values in microns in the release)."""
    out = {}
    for n in names:
        base = PurePosixPath(n).name
        if n.lower().endswith(".txt") and "wavelength" in base.lower():
            v = np.array(_floats(src.text(n).splitlines()[1:]), float)
            if v.size >= 2:
                out[n] = v * 1000.0 if np.nanmax(v) < 100 else v          # microns -> nm
    return out


def _pick_wavelengths(spec_name: str, n: int, wfiles: Dict[str, np.ndarray]):
    cands = [(k, v) for k, v in wfiles.items() if v.size == n]
    if not cands:
        return None, None
    if len(cands) == 1:
        return cands[0]
    m = _SENSOR.search(PurePosixPath(spec_name).name)
    if m:
        code = m.group(1)[:4].upper()
        for k, v in cands:
            if code in PurePosixPath(k).name.upper():
                return k, v
    return cands[0]


def iter_usgs(src: _Source, chapters: Optional[str] = None) -> Iterator[dict]:
    names = src.names()
    wfiles = _usgs_wavelength_files(src, names)
    if not wfiles:
        raise ValueError("no *Wavelengths*.txt file found - is this an ASCIIdata_splib07* download?")
    want = {c.strip().upper() for c in chapters.split(",")} if chapters else None
    for n in sorted(names):
        p = PurePosixPath(n)
        if not n.lower().endswith(".txt") or n in wfiles or "bandpass" in p.name.lower() \
                or "wavelength" in p.name.lower() or "readme" in p.name.lower():
            continue
        chap = next((part for part in p.parts if part.lower().startswith("chapter")), None)
        if chap is None:
            continue
        letter = chap[len("Chapter"):len("Chapter") + 1].upper()
        if want and letter not in want:
            continue
        lines = src.text(n).splitlines()
        if not lines:
            continue
        title = lines[0].strip()
        vals = np.array(_floats(lines[1:]), float)
        if vals.size < 2:
            continue
        wname, wl = _pick_wavelengths(n, vals.size, wfiles)
        if wl is None:
            continue
        vals[vals < BAD] = np.nan
        if np.all(np.isnan(vals)):
            continue
        stem = p.stem
        parts = stem.split("_")
        # drop the library/sensor prefix: splib07a_ / splib07b_ / s07_ASD_ / s07ASD_ / s07_AV14_ ...
        while parts and re.fullmatch(r"(splib07[ab]?|s07|s07[A-Za-z0-9]+|ASD|AV\d\d|HY\d\d|VIMS|M3t|WV3|cv\w+|rs\w+)",
                                     parts[0]):
            parts = parts[1:]
        sensor = next((x for x in parts[1:] if _SENSOR.match(x)), None)      # BECKb, ASDFRa, NIC4, AVIRIS-1996
        core = [x for x in parts if x != sensor and not re.fullmatch(r"[A-Z]{4}", x)]   # drop AREF/RREF/RTGC ...
        if not core:
            continue
        material = core[0].replace("-", " ").strip()                  # Lodgepole-Pine -> Lodgepole Pine
        name = " ".join([material] + core[1:])                        # sample ids keep hyphens (KGa-1)
        acq = "airborne" if sensor and sensor.upper().startswith(("AVIRIS", "AV")) else "lab"
        yield {"name": name[:500], "category": f"{USGS_CHAPTERS.get(letter, chap[8:])}/{material}",
               "acquisition": acq, "wavelengths": wl, "values": vals,
               "material_metadata": {"source": "USGS splib07", "file": p.name, "title": title,
                                     "sensor": sensor, "chapter": chap},
               "spectrum_metadata": {"uid": f"usgs:{n}", "wavelength_file": PurePosixPath(wname).name,
                                     "source_file": p.name}}


# ================================================================== ECOSTRESS / ASTER
def _ecostress_parse(text: str):
    head, data = {}, []
    for ln in text.splitlines():
        s = ln.strip()
        if not s:
            continue
        nums = s.replace(",", " ").split()
        if len(nums) >= 2:
            try:
                data.append((float(nums[0]), float(nums[1])))
                continue
            except ValueError:
                pass
        if not data and ":" in s:
            k, _, v = s.partition(":")
            head[k.strip().lower()] = v.strip()
    return head, data


def iter_ecostress(src: _Source, include_tir=False) -> Iterator[dict]:
    for n in sorted(src.names()):
        p = PurePosixPath(n)
        low = p.name.lower()
        if not low.endswith(".txt") or "ancillary" in low:
            continue
        head, data = _ecostress_parse(src.text(n))
        if len(data) < 2 or "name" not in head:
            continue
        x = np.array([d[0] for d in data], float)
        y = np.array([d[1] for d in data], float)
        xu, yu = head.get("x units", "").lower(), head.get("y units", "").lower()
        if "micro" in xu or (not xu and np.nanmax(x) < 100):
            x = x * 1000.0
        if "percent" in yu:
            y = y / 100.0
        y[(y < -1e3) | (y > 1e3)] = np.nan
        order = np.argsort(x)
        x, y = x[order], y[order]
        keep = np.concatenate([[True], np.diff(x) > 0])               # drop duplicate wavelengths
        x, y = x[keep], y[keep]
        if not include_tir and x.min() >= 2400:                       # thermal-only spectrum
            continue
        if not include_tir:
            sel = x <= 2500.5
            x, y = x[sel], y[sel]
        if x.size < 2:
            continue
        typ = head.get("type", "").title()
        cls = head.get("class", "").title()
        if head.get("genus"):
            levels = [typ, cls, head.get("genus", "").title(), head.get("species", "").title()]
        else:
            levels = [typ, cls, head.get("subclass", "").title()]
        name = head["name"]
        if head.get("sample no."):
            name = f"{name} {head['sample no.']}"
        yield {"name": name[:500], "category": "/".join(l for l in levels if l and l.lower() != "none"),
               "acquisition": "lab", "wavelengths": x, "values": y,
               "material_metadata": {"source": "ECOSTRESS spectral library", "file": p.name,
                                     **{k: v for k, v in head.items() if k in (
                                         "type", "class", "subclass", "genus", "species", "sample no.", "owner",
                                         "particle size", "origin", "description", "measurement")}},
               "spectrum_metadata": {"uid": f"ecostress:{p.name}", "source_file": p.name,
                                     "wavelength_range": head.get("wavelength range")}}


# ================================================================== OpenSpecLib Parquet
_OSL_TOP = {"mineral": "Mineral", "rock": "Rock", "soil": "Soil", "vegetation": "Vegetation",
            "npv": "Non-photosynthetic Vegetation", "man_made": "Man-made", "meteorite": "Meteorite",
            "water": "Water", "organic_compound": "Organic Compound", "mixture": "Mixture", "other": "Other",
            "lunar": "Lunar"}
_LATIN = re.compile(r"^([A-Z][a-z]+)\s+([a-z][a-z-]+)\b")


def _osl_category(r) -> str:
    cat = r.get("material.category") or "other"
    top = _OSL_TOP.get(cat, cat.replace("_", " ").title())
    sub = (r.get("material.subcategory") or "").strip()
    sub = "" if sub.lower() in ("", "none", "null") else sub.title() if sub.islower() else sub
    mname = (r.get("material.name") or r.get("name") or "").strip()
    levels = [top, sub]
    m = _LATIN.match(mname)
    if cat in ("vegetation", "npv") and m:
        levels += [m.group(1), m.group(2).capitalize()]                # genus / species
    else:
        first = re.split(r"[\s(]", mname, maxsplit=1)[0].strip(" ,.-")          # 'Kaolinite KGa-1' -> Kaolinite
        if first and first.lower() not in sub.lower():
            levels.append(first[:1].upper() + first[1:])
    return "/".join(l for l in levels if l)


def iter_openspeclib(path, wavelengths_path=None, categories: Optional[str] = None,
                     include_tir=False) -> Iterator[dict]:
    try:
        import pyarrow.parquet as pq
    except ImportError:
        raise RuntimeError("reading OpenSpecLib Parquet files needs pyarrow - pip install pyarrow") from None
    path = Path(path)
    wpath = Path(wavelengths_path) if wavelengths_path else path.with_name("wavelengths.parquet")
    if not wpath.is_file():
        raise FileNotFoundError(f"{wpath.name} not found next to {path.name} - download it from the same release")
    grids = {g["grid_id"]: (g["wavelength_unit"], g["wavelengths"])
             for g in pq.read_table(wpath, columns=["grid_id", "wavelength_unit", "wavelengths"]).to_pylist()}
    want = {c.strip().lower() for c in categories.split(",")} if categories else None
    pf = pq.ParquetFile(path)
    # Scale check for the whole file: the v0.0.6 ECOSTRESS file says reflectance_scale='unit' but holds percent
    # (median 99th percentile ~54), USGS really is 0-1 (~0.6). Decide once per file so dark samples
    # (a few percent reflectance) of a percent file are still divided by 100.
    first = next(pf.iter_batches(batch_size=200, columns=["spectral_data.values"])).to_pylist()
    p99 = [np.nanpercentile(np.array([np.nan if v is None else v for v in r["spectral_data.values"]], float)
                            .clip(-1e6, 1e6), 99) for r in first if r["spectral_data.values"]]
    percent_file = bool(p99) and float(np.nanmedian(p99)) > 1.5
    for batch in pf.iter_batches(batch_size=256):
        for r in batch.to_pylist():
            if want and (r.get("material.category") or "").lower() not in want:
                continue
            unit, wl = grids.get(r["spectral_data.wavelength_grid_id"], (None, None))
            if wl is None:
                continue
            unit = (r.get("spectral_data.wavelength_unit") or unit or "").lower()
            x = np.asarray(wl, float) * (1000.0 if unit in ("um", "micrometer", "micron", "microns") else 1.0)
            y = np.array([np.nan if v is None else v for v in r["spectral_data.values"]], float)
            if x.size != y.size or x.size < 2:
                continue
            y[(y < BAD) | (np.abs(y) > 1e6)] = np.nan
            if percent_file or (r.get("spectral_data.reflectance_scale") or "unit") == "percent":
                y = y / 100.0
            order = np.argsort(x)
            x, y = x[order], y[order]
            keep = np.concatenate([[True], np.diff(x) > 0])
            x, y = x[keep], y[keep]
            if not include_tir:
                if x.min() > 1000:                                      # thermal / MIR-only measurement
                    continue
                sel = x <= 2500.5
                x, y = x[sel], y[sel]
            if x.size < 2 or np.all(np.isnan(y)):
                continue
            inst = r.get("measurement.instrument") or ""
            acq = "airborne" if inst.upper().startswith("AVIRIS") else "lab"
            name = (r.get("name") or r.get("material.name") or "").strip()
            sid = (r.get("sample.id") or "").strip()
            if sid and sid.lower() not in name.lower():
                name = f"{name} {sid}"
            meta = {k.split(".", 1)[1] if k.startswith(("source.", "sample.")) else k: r.get(k) for k in (
                "source.library", "source.library_version", "source.original_id", "source.license",
                "source.citation", "sample.id", "sample.description", "sample.particle_size", "sample.origin",
                "sample.owner", "material.formula") if r.get(k)}
            yield {"name": re.sub(r"\s+", " ", name)[:500], "category": _osl_category(r), "acquisition": acq,
                   "wavelengths": x, "values": y,
                   "material_metadata": {"source": "OpenSpecLib", **meta, "instrument": inst or None},
                   "spectrum_metadata": {"uid": f"openspeclib:{r.get('id')}", "openspeclib_id": r.get("id"),
                                         "rescaled_from_percent": percent_file or None, "technique": r.get("measurement.technique"),
                                         "source_file": r.get("source.filename")}}


# ================================================================== import
def file_sha256(path) -> Optional[str]:
    path = Path(path)
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def import_records(db: Session, records: Iterator[dict], *, library_id=None, library_name=None, dataset_name: str,
                   source_label: str, src: Optional[_Source] = None, acquisition: Optional[str] = None,
                   limit: Optional[int] = None, progress: Optional[Callable[[int, int, str], None]] = None,
                   provenance=None, kind: Optional[str] = None):
    """Write records into Library -> Dataset -> Material -> Sample -> Spectrum (idempotent per dataset)."""
    acquisition = repo.normalize_acquisition(acquisition)
    try:
        if library_id is not None:
            lib = db.query(Library).filter(Library.id == repo._uuid(library_id, "Library")).first()
            if lib is None:
                raise repo.NotFound("Library not found")
        else:
            lib = db.query(Library).filter(Library.slug == repo.slugify(library_name)).first()
            if lib is None:
                lib = Library(name=library_name, slug=repo._unique_slug(db, repo.slugify(library_name)),
                              description=f"Reference spectra imported from {source_label}.",
                              library_metadata={"type": "spectral", "source": source_label})
                db.add(lib)
                db.flush()
        ds = repo.get_or_create_dataset(db, lib.id, dataset_name, description=f"Imported from {source_label}.",
                                        source_name=source_label, commit=False)
        prov = Path(provenance) if provenance is not None else (src.path if src is not None else None)
        if prov is not None:                         # provenance: a file (with SHA-256) or a folder (by path)
            cs = file_sha256(prov)
            spath = str(prov.resolve())
            q = db.query(DataFile).filter(DataFile.dataset_id == ds.id)
            q = q.filter(DataFile.checksum == cs) if cs else q.filter(DataFile.storage_path == spath)
            if q.first() is None:
                db.add(DataFile(dataset_id=ds.id, filename=prov.name,
                                file_type=(prov.suffix.lstrip(".") or "folder") if prov.is_file() else "folder",
                                file_size=prov.stat().st_size if prov.is_file() else None, checksum=cs,
                                storage_path=spath, file_metadata={"source": source_label, "kind": kind}))
        recs = list(records)
        if limit:
            recs = recs[:limit]
        # every record carries a source-unique id ("uid"); the same sample measured twice (two instruments,
        # two ranges) becomes two samples of one material, while re-running the import adds nothing
        have = {(m or {}).get("uid") for (m,) in db.query(Spectrum.spectrum_metadata)
                .join(Sample, Sample.id == Spectrum.sample_id).filter(Sample.dataset_id == ds.id)}
        added = skipped = 0
        cats = set()
        for i, r in enumerate(recs, 1):
            if progress:
                progress(i, len(recs), r["name"])
            uid = (r.get("spectrum_metadata") or {}).get("uid")
            if uid and uid in have:
                skipped += 1
                continue
            have.add(uid)
            repo.add_spectrum(db, lib.id, ds.id, r["name"], r["wavelengths"], r["values"], category=r["category"],
                              material_metadata=r.get("material_metadata"),
                              spectrum_metadata=r.get("spectrum_metadata"), commit=False,
                              acquisition=acquisition or r.get("acquisition"), unit=r.get("unit"),
                              purity=r.get("purity", "ground_measured"))
            cats.add(r["category"])
            added += 1
            if added % 200 == 0:
                db.flush()
        db.commit()
    except Exception:
        db.rollback()
        raise
    return {"library_id": str(lib.id), "library_name": lib.name, "dataset_name": ds.name, "found": len(recs),
            "spectra_added": added, "skipped": skipped, "categories": len(cats)}


def import_speclib(db: Session, kind: str, path, *, chapters=None, include_tir=False, categories=None,
                   wavelengths_path=None, **kw):
    """Kept for compatibility - goes through the universal pipeline (hvva.library.ingest)."""
    if wavelengths_path is None:
        from .formats import ReadOptions
        from .ingest import import_path
        acq = kw.pop("acquisition", None)
        opts = ReadOptions(reader={"openspeclib": "openspeclib", "usgs": "usgs"}.get(kind),
                           chapters=chapters, categories=categories, spectral_range="all" if include_tir else "auto",
                           acquisition=acq, force_acquisition=bool(acq), limit=kw.pop("limit", None))
        if kind not in ("openspeclib", "usgs", "ecostress"):
            raise ValueError(f"unknown library kind {kind!r} (usgs | ecostress | openspeclib)")
        r = import_path(db, path, library_id=kw.pop("library_id", None), library_name=kw.pop("library_name", None),
                        dataset_name=kw.pop("dataset_name", None), options=opts, progress=kw.pop("progress", None))
        r.setdefault("categories", 0)
        return r
    if kind == "openspeclib":
        p = Path(path)
        recs = iter_openspeclib(p, wavelengths_path, categories, include_tir)
        label = f"OpenSpecLib {p.stem}"
        kw.setdefault("dataset_name", label)
        return import_records(db, recs, provenance=p, kind=kind, source_label=label, **kw)
    src = _Source(path)
    if kind == "usgs":
        recs = iter_usgs(src, chapters)
        label = f"USGS Spectral Library v7 ({src.path.name})"
    elif kind == "ecostress":
        recs = iter_ecostress(src, include_tir)
        label = f"ECOSTRESS spectral library ({src.path.name})"
    else:
        raise ValueError(f"unknown library kind {kind!r} (usgs | ecostress)")
    kw.setdefault("dataset_name", label)
    return import_records(db, recs, src=src, kind=kind, source_label=label, **kw)


def main(argv=None):
    from .database import connect, default_url
    p = argparse.ArgumentParser(description="Import USGS splib07 / ECOSTRESS spectra into the material library")
    p.add_argument("kind", choices=["usgs", "ecostress", "openspeclib"])
    p.add_argument("path", help="downloaded .zip or extracted folder (openspeclib: the source .parquet)")
    p.add_argument("--library", help="library name (created if missing)")
    p.add_argument("--dataset", help="dataset name (default: from the source)")
    p.add_argument("--chapters", help="USGS only: chapter letters, e.g. M,S,V,A "
                                      "(Minerals, Soils, Coatings, Liquids, Organics, Artificial, Vegetation)")
    p.add_argument("--acquisition", choices=repo.ACQUISITION_TYPES, help="override the automatic tag")
    p.add_argument("--limit", type=int, help="import at most N spectra (quick demo)")
    p.add_argument("--include-tir", action="store_true", help="keep thermal-infrared data (> 2.5 um)")
    p.add_argument("--categories", help="openspeclib only: source categories, e.g. mineral,vegetation,npv,soil,"
                                        "rock,man_made,water,organic_compound")
    p.add_argument("--wavelengths", help="openspeclib only: path to wavelengths.parquet (default: next to the file)")
    p.add_argument("--url", default=None, help=f"database URL (default: $DATABASE_URL or {default_url()})")
    a = p.parse_args(argv)
    lib = a.library or {"usgs": "USGS splib07", "ecostress": "ECOSTRESS"}.get(a.kind, "OpenSpecLib")
    db = connect(a.url)
    with db.session() as s:
        kw = {"dataset_name": a.dataset} if a.dataset else {}
        r = import_speclib(s, a.kind, a.path, chapters=a.chapters, include_tir=a.include_tir, library_name=lib,
                           categories=a.categories, wavelengths_path=a.wavelengths,
                           acquisition=a.acquisition, limit=a.limit,
                           progress=lambda i, n, name: print(f"[{i}/{n}] {name}") if i % 50 == 0 or i == n else None,
                           **kw)
    print(f"\nLibrary {r['library_name']} / {r['dataset_name']}: {r['spectra_added']} spectra added, "
          f"{r['skipped']} already there, {r['categories']} categories.")


if __name__ == "__main__":
    main()
