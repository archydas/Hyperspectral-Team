"""Ingest pipeline: file or folder -> reader -> standard form -> database. Idempotent (sha256)."""
from __future__ import annotations
import hashlib, re, zipfile
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
from sqlalchemy import select
from .ingest import open_cube, IngestError, UnsupportedFormatError
from .ingest.api import detect_format
from .ingest.readers.envi import parse_envi_header, BIN_EXT
from .spectra import read_spectra, detect_spectra_format
from .spectra.categories import guess_category
from .library.sources import is_envi_roi
from .db import session_scope, init_db
from .db.models import Source, CubeRecord, Library, Dataset, Material, Sample, SpectrumRecord

CUBE_BIN_EXT = {e for e in BIN_EXT if e and e != ".sli"}
KINDS = ("satellite", "airborne", "drone", "field", "library")
SKIP_EXT = {".md", ".json", ".xml", ".py", ".png", ".jpg", ".gitkeep", ".gz", ".tar", ".7z"}
# AVIRIS-NG / AVIRIS-C ancillary rasters (geometry, observation, water vapour...) have no spectral axis -> not images to import
ANCILLARY = re.compile(r"(_(glt|igm|loc|obs|loc_ort|obs_ort|ort|mask|qa|qual\w*)|_h2o(_[a-z0-9]+_img)?)(\.hdr)?$", re.I)
ARCHIVE_EXT = {".zip"}


@dataclass
class IngestResult:
    path: str
    status: str            # ingested | duplicate | skipped | error
    kind: str | None = None
    detail: str = ""
    ids: list = field(default_factory=list)


def sha256_of(path) -> str:
    p = Path(path); h = hashlib.sha256()
    files = sorted(x for x in p.rglob("*") if x.is_file()) if p.is_dir() else [p]
    if p.suffix.lower() == ".hdr":                # include the binary that goes with the header
        files += [c for c in (p.with_suffix(e) for e in ("",) + tuple(CUBE_BIN_EXT)) if c.is_file() and c != p]
    for f in files:
        if p.is_dir(): h.update(str(f.relative_to(p)).encode())
        with open(f, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
    return h.hexdigest()


def _has_hdr(p: Path) -> bool:
    return Path(str(p) + ".hdr").is_file() or p.with_suffix(".hdr").is_file()


def classify(path) -> str:
    """'cube' | 'spectra' | raises UnsupportedFormatError"""
    p = Path(path)
    if p.is_dir():
        detect_format(p); return "cube"
    if p.is_file() and ANCILLARY.search(p.name if p.suffix.lower() == ".hdr" else p.stem if p.suffix.lower() in CUBE_BIN_EXT else p.name) \
            and (p.suffix.lower() == ".hdr" or p.suffix.lower() in CUBE_BIN_EXT or not p.suffix):
        raise UnsupportedFormatError(f"{p.name}: ancillary product (geometry / observation / water-vapour), not a spectral image")
    if p.suffix.lower() == ".hdr":
        try:
            ft = parse_envi_header(p).get("file type", "").lower()
        except IngestError:
            ft = ""
        return "spectra" if "spectral library" in ft else "cube"
    if p.suffix.lower() == ".mat":
        from .ingest.readers.mat import inspect_mat
        info = inspect_mat(p)
        if any(len(sh) == 3 for _, sh, _ in info):
            return "cube"
        raise UnsupportedFormatError(f"{p.name}: no 3-D array ({[(k, sh) for k, sh, _ in info]}) - looks like a label/ground-truth "
                                     "map; import the image, then use this file as 'labels' to build class-mean spectra")
    if p.suffix.lower() in (".tif", ".tiff", ".h5", ".he5", ".hdf5", ".hdf", ".nc", ".nc4"):
        return "cube"
    if p.suffix.lower() in CUBE_BIN_EXT and _has_hdr(p):
        return "cube"
    if detect_spectra_format(p):
        return "spectra"
    raise UnsupportedFormatError(f"don't know how to ingest {p.name}")


def _get_source(s, name, kind, url=None, license=None):
    src = s.scalar(select(Source).where(Source.name == name))
    if src is None:
        src = Source(name=name, kind=kind, url=url, license=license); s.add(src); s.flush()
    return src


def _nn(a):
    return None if a is None else [None if (isinstance(x, float) and not np.isfinite(x)) else x for x in np.asarray(a, float).tolist()]


def store_spectra(s, spectra, *, source, library_name, dataset_name, digest, fmt, extra_meta=None):
    """Write Spectrum objects to library -> dataset -> material -> sample -> spectrum. Returns spectrum ids."""
    lib = s.scalar(select(Library).where(Library.name == library_name))
    if lib is None:
        lib = Library(name=library_name, source_id=source.id if source else None); s.add(lib); s.flush()
    ds = Dataset(library_id=lib.id, name=dataset_name, sha256=digest, format=fmt)
    s.add(ds); s.flush()
    ids = []
    for sp in spectra:
        mat = s.scalar(select(Material).where(Material.name == sp.material))
        cat, cat_src = sp.category, "provider"
        if cat is None:
            cat, cat_src = guess_category(sp.material), "keyword_rule"
        if mat is None:
            mat = Material(name=sp.material, category=cat); s.add(mat); s.flush()
        elif mat.category is None and cat is not None:
            mat.category = cat
        meta = dict(_jsonable(sp.metadata) or {}, **(extra_meta or {}))
        if cat is not None:
            meta["category_source"] = cat_src
        smp = Sample(dataset_id=ds.id, material_id=mat.id, label=sp.material, meta=meta)
        s.add(smp); s.flush()
        v = np.asarray(sp.values, float)
        wl = np.asarray(sp.wavelengths_nm, float)
        rec = SpectrumRecord(sample_id=smp.id, unit=sp.unit, purity=sp.purity,
                             wl_min_nm=float(wl[0]), wl_max_nm=float(wl[-1]), n_bands=len(v),
                             wavelengths_nm=wl.tolist(), values=_nn(v), meta=None)
        s.add(rec); s.flush(); ids.append(rec.id)
    return ids


def ingest_file(path, *, source_name="unspecified", kind=None, url=None, license=None,
                library=None, db_url=None, **reader_kw) -> IngestResult:
    p = Path(path)
    if kind is None and p.parent.name in KINDS:      # data/raw/<kind>/file -> provenance for free
        kind = p.parent.name
        source_name = f"{source_name}/{kind}" if source_name != "unspecified" else kind
    try:
        what = classify(p)
    except UnsupportedFormatError as e:
        return IngestResult(str(p), "skipped", detail=str(e))
    init_db(db_url)
    digest = sha256_of(p)
    try:
        with session_scope(db_url) as s:
            if what == "cube":
                if s.scalar(select(CubeRecord).where(CubeRecord.sha256 == digest)):
                    return IngestResult(str(p), "duplicate", "cube")
                cube = open_cube(p, **reader_kw)
                src = _get_source(s, source_name, kind or "airborne", url, license)
                bad = cube.bad_band_mask
                rec = CubeRecord(
                    source_id=src.id, name=p.stem if p.is_file() else p.name, path=str(p.resolve()), sha256=digest,
                    format=cube.source_format or "?", rows=cube.rows, cols=cube.cols, bands=cube.bands,
                    unit=cube.unit, sensor=cube.sensor, processing_level=cube.processing_level, crs=cube.crs,
                    transform=list(cube.transform) if cube.transform else None,
                    wl_min_nm=float(cube.wavelengths_nm[0]), wl_max_nm=float(cube.wavelengths_nm[-1]),
                    wavelengths_nm=cube.wavelengths_nm.tolist(),
                    fwhm_nm=None if cube.fwhm_nm is None else cube.fwhm_nm.tolist(),
                    bad_band_mask=[bool(x) for x in bad],
                    qc=_qc_dict(cube), meta=dict(_jsonable(cube.metadata) or {}, reader_kw=_jsonable({k: v for k, v in reader_kw.items() if v is not None})))
                s.add(rec); s.flush()
                return IngestResult(str(p), "ingested", "cube", cube.describe(), [rec.id])
            # ---- spectra
            if s.scalar(select(Dataset).where(Dataset.sha256 == digest)):
                return IngestResult(str(p), "duplicate", "spectra")
            if detect_spectra_format(p) == "usgs" and "wavelengths_file" not in reader_kw:
                sib = sorted(p.parent.glob("*avelength*.txt"))
                if sib: reader_kw = dict(reader_kw, wavelengths_file=sib[0])
            spectra = read_spectra(p, **reader_kw)
            src = _get_source(s, source_name, kind or "library", url, license)
            lib_name = library or (source_name if source_name != "unspecified" else p.parent.name or "unspecified")
            ids = store_spectra(s, spectra, source=src, library_name=lib_name, dataset_name=p.name,
                                digest=digest, fmt=detect_spectra_format(p) or "?")
            return IngestResult(str(p), "ingested", "spectra", f"{len(ids)} spectra", ids)
    except IngestError as e:
        return IngestResult(str(p), "error", detail=f"{type(e).__name__}: {e}")


def extract_archive(path, dest_root=None) -> Path:
    """Unzip into <dest_root>/<zip-stem>-<sha8>/ (default: ~/.ds3/extracted). Refuses entries that escape the folder."""
    p = Path(path)
    root = Path(dest_root) if dest_root else Path.home() / ".ds3" / "extracted"
    dest = root / f"{p.stem}-{sha256_of(p)[:8]}"
    if dest.is_dir() and any(dest.iterdir()):
        return dest
    dest.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(p) as z:
            base = dest.resolve()
            for m in z.infolist():
                tgt = (dest / m.filename).resolve()
                if base != tgt and base not in tgt.parents:
                    raise IngestError(f"{p.name}: unsafe path in archive: {m.filename}")
            z.extractall(dest)
    except zipfile.BadZipFile as e:
        raise IngestError(f"{p.name}: not a valid zip archive ({e})") from e
    return dest


def _enmap_only(folder: Path):
    """EnMAP L2A zips also hold quality / cloud / mask rasters: if a *SPECTRAL_IMAGE.TIF is present, import only that."""
    si = sorted(folder.rglob("*SPECTRAL_IMAGE.TIF")) + sorted(folder.rglob("*SPECTRAL_IMAGE.tif"))
    return si or None


def ingest_path(path, **kw) -> list[IngestResult]:
    """A file, a .zip archive, a .zarr store, or a folder (walked recursively)."""
    p = Path(path)
    if p.is_file() and p.suffix.lower() in ARCHIVE_EXT:
        try:
            folder = extract_archive(p)
        except IngestError as e:
            return [IngestResult(str(p), "error", detail=f"{type(e).__name__}: {e}")]
        only = _enmap_only(folder)
        if only:
            return [ingest_file(f, **kw) for f in only]
        return ingest_path(folder, **kw)
    if p.is_file() or (p.is_dir() and p.suffix.lower() == ".zarr"):
        return [ingest_file(p, **kw)]
    out = []
    for f in sorted(p.rglob("*")):
        if f.is_dir() and f.suffix.lower() != ".zarr": continue
        if any(par.suffix.lower() == ".zarr" for par in f.parents if par != p) : continue
        if f.name.startswith(".") or f.suffix.lower() in SKIP_EXT: continue
        if "wavelength" in f.name.lower() and f.suffix.lower() == ".txt": continue  # USGS / grid companion file
        if f.suffix.lower() in (".txt", ".roi") and is_envi_roi(f): continue        # use `ds3 library build`
        if (f.suffix.lower() in CUBE_BIN_EXT | {".sli"}) and _has_hdr(f): continue  # handled via its .hdr
        out.append(ingest_file(f, **kw))
    return out


def _qc_dict(cube):
    q = cube.qc
    if q is None: return None
    return dict(passed=bool(q.passed), n_nodata_pixels=int(q.n_nodata_pixels),
                frac_nodata_pixels=float(q.frac_nodata_pixels),
                bad_bands=dict(source=int(q.source_bad.sum()), water=int(q.water_bad.sum()),
                               auto=int(q.auto_bad.sum()), combined=int(q.combined_bad.sum())),
                issues=_jsonable(q.issues))


def _jsonable(o):
    if isinstance(o, dict): return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [_jsonable(x) for x in o]
    if isinstance(o, np.ndarray): return _jsonable(o.tolist())
    if isinstance(o, (np.floating, float)): return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.integer): return int(o)
    if isinstance(o, np.bool_): return bool(o)
    if o is None or isinstance(o, (str, int, bool)): return o
    if hasattr(o, "__dict__"): return _jsonable(vars(o))
    return str(o)


def library_name_from_file(path) -> str:
    """ecostress_library_all_spectra.txt -> 'Ecostress library all spectra'"""
    t = re.sub(r"[_\s]+", " ", Path(path).stem).strip()
    return t[:1].upper() + t[1:] if t else Path(path).stem


def ingest_library_folder(folder, *, db_url=None, **kw) -> list[IngestResult]:
    """Every file directly inside `folder` becomes its own library (named after the file); images are registered as cubes.
    Re-running is safe (same SHA-256 -> 'duplicate'). One unreadable file never stops the others."""
    folder = Path(folder)
    out: list[IngestResult] = []
    if not folder.is_dir():
        return out
    for f in sorted(p for p in folder.iterdir() if p.is_file() and not p.name.startswith(".")):
        if f.name.lower().startswith("readme") or f.suffix.lower() in SKIP_EXT:
            continue
        try:
            if f.suffix.lower() in ARCHIVE_EXT:
                out.extend(ingest_path(f, db_url=db_url, library=library_name_from_file(f), **kw))
            else:
                out.append(ingest_file(f, library=library_name_from_file(f), db_url=db_url, **kw))
        except Exception as e:                       # never let one file stop the rest
            out.append(IngestResult(str(f), "error", detail=f"{type(e).__name__}: {e}"))
    return out
