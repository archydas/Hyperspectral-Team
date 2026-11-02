"""Reference collections and containers: USGS splib07 ASCII, OpenSpecLib Parquet, any folder, any zip."""
from __future__ import annotations

import os
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import List

import numpy as np

from .base import ReadOptions, Reader, SpectrumRecord, detect, register, rel_category

_SKIP = {"__pycache__", ".git", "__macosx", "$recycle.bin", "system volume information"}
_SIDE = {".hdr"}                     # ENVI pairs: read once per stem


def _dict_to_record(d: dict, group: str) -> SpectrumRecord:
    return SpectrumRecord(d["name"], np.asarray(d["wavelengths"], float), np.asarray(d["values"], float),
                          category=d.get("category"), acquisition=d.get("acquisition"), wl_unit="nm", scale="unit",
                          quantity="reflectance", material_metadata=d.get("material_metadata") or {},
                          spectrum_metadata=d.get("spectrum_metadata") or {}, group=group)


# ================================================================== USGS splib07 ASCII release
@register
class UsgsReader(Reader):
    name = "usgs"
    label = "USGS Spectral Library v7 (ASCII)"
    extensions = (".zip",)
    is_container = True
    reference_collection = True

    def sniff(self, path: Path) -> float:
        if path.is_file() and zipfile.is_zipfile(path):
            try:
                with zipfile.ZipFile(path) as z:
                    names = [n.lower() for n in z.namelist()[:20000]]
            except (zipfile.BadZipFile, OSError):
                return 0.0
            chap = any("/chapter" in n or n.startswith("chapter") for n in names)
            wl = any("wavelength" in n and n.endswith(".txt") for n in names)
            return 0.95 if chap and wl else 0.0
        if path.is_dir():
            try:
                entries = list(path.iterdir())
            except OSError:
                return 0.0
            chap = any(e.is_dir() and e.name.lower().startswith("chapter") for e in entries)
            wl = any(e.is_file() and "wavelength" in e.name.lower() and e.suffix.lower() == ".txt" for e in entries)
            return 0.95 if chap and wl else 0.0
        return 0.0

    def read(self, path: Path, opts: ReadOptions):
        from ..speclib import _Source, iter_usgs
        g = str(path)
        return [_dict_to_record(d, g) for d in iter_usgs(_Source(path), opts.chapters or "M,S,V,A")]


# ================================================================== OpenSpecLib release Parquet
@register
class OpenSpecLibReader(Reader):
    name = "openspeclib"
    label = "OpenSpecLib Parquet (USGS / ECOSTRESS / EcoSIS)"
    extensions = (".parquet",)
    reference_collection = True

    def sniff(self, path: Path) -> float:
        if not path.is_file() or path.suffix.lower() != ".parquet" or path.name.lower() == "wavelengths.parquet":
            return 0.0
        try:
            import pyarrow.parquet as pq
            names = pq.ParquetFile(path).schema_arrow.names
        except Exception:                                                   # noqa: BLE001
            return 0.0
        return 0.95 if "spectral_data.values" in names and "spectral_data.wavelength_grid_id" in names else 0.0

    def describe(self, path: Path) -> str:
        if not (path.parent / "wavelengths.parquet").is_file():
            return f"{self.label} - wavelengths.parquet missing in this folder"
        return self.label

    def read(self, path: Path, opts: ReadOptions):
        from ..speclib import iter_openspeclib
        g = str(path)
        return [_dict_to_record(d, g) for d in iter_openspeclib(path, None, opts.categories,
                                                                include_tir=opts.spectral_range == "all")]


# ================================================================== any folder
def _walk_files(root: Path, max_files=200000) -> List[Path]:
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d.lower() not in _SKIP and not d.startswith("."))
        for f in sorted(filenames):
            if not f.startswith(".") and not f.startswith("~$"):
                out.append(Path(dirpath) / f)
                if len(out) >= max_files:
                    return out
    return out


def read_folder(root: Path, opts: ReadOptions, label_root: str) -> List[SpectrumRecord]:
    out: List[SpectrumRecord] = []
    done_stems = set()
    files = _walk_files(root)
    for f in files:
        if f.name.lower() == "wavelengths.parquet":
            continue
        reader, score, _ = detect(f, exclude_containers=False)
        if reader is None or reader.is_container:
            continue
        key = (str(f.parent), f.stem.lower(), reader.name)
        if reader.name in ("envi_sli", "image"):           # .hdr + .sli/.img are one dataset
            if key in done_stems:
                continue
            done_stems.add(key)
        try:
            recs = list(reader.read(f, opts))
        except Exception as e:                                              # noqa: BLE001
            opts.warnings.append(f"{f.relative_to(root)}: {str(e).splitlines()[0][:200]}")
            continue
        rel = str(f.relative_to(root)).replace(os.sep, "/")
        cat = rel_category(root, f) if opts.category_from_folders else None
        for r in recs:
            r.spectrum_metadata = dict(r.spectrum_metadata or {})
            r.spectrum_metadata["uid"] = f"{label_root}/{rel}|{r.spectrum_metadata.get('uid', r.name)}"
            r.spectrum_metadata.setdefault("source_file", rel)
            if not r.category and cat:
                r.category = cat
            r.group = str(f)
            out.append(r)
    return out


def _has_ecostress(paths) -> bool:
    return any(str(p).lower().endswith(".spectrum.txt") for p in paths)


@register
class FolderReader(Reader):
    name = "folder"
    label = "Folder of spectral files"
    is_container = True

    def sniff(self, path: Path) -> float:
        if not path.is_dir() or path.suffix.lower() == ".zarr":
            return 0.0
        for f in _walk_files(path, max_files=400):
            r, s, _ = detect(f, exclude_containers=True)
            if r is not None:
                return 0.5
        return 0.0

    def describe(self, path: Path) -> str:
        files = _walk_files(path, max_files=20000)
        kinds = {}
        for f in files[:3000]:
            r, s, _ = detect(f, exclude_containers=True)
            if r is not None:
                kinds[r.label] = kinds.get(r.label, 0) + 1
        return "Folder: " + ", ".join(f"{n} × {k}" for k, n in sorted(kinds.items(), key=lambda kv: -kv[1])) \
            if kinds else self.label

    def wants_vswir(self, path: Path) -> bool:
        return _has_ecostress(_walk_files(path, max_files=2000))

    def read(self, path: Path, opts: ReadOptions):
        return read_folder(path, opts, path.name)


@register
class ZipReader(Reader):
    name = "zip"
    label = "Zip archive of spectral files"
    extensions = (".zip",)
    is_container = True

    def sniff(self, path: Path) -> float:
        return 0.4 if path.is_file() and zipfile.is_zipfile(path) else 0.0

    def wants_vswir(self, path: Path) -> bool:
        try:
            with zipfile.ZipFile(path) as z:
                return _has_ecostress(z.namelist()[:5000])
        except (zipfile.BadZipFile, OSError):
            return False

    def read(self, path: Path, opts: ReadOptions):
        tmp = tempfile.mkdtemp(prefix="hvva_zip_")
        try:
            with zipfile.ZipFile(path) as z:
                for m in z.infolist():
                    if m.is_dir() or m.filename.startswith(("__MACOSX/", "/")) or ".." in m.filename:
                        continue
                    z.extract(m, tmp)
            return read_folder(Path(tmp), opts, path.name)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
