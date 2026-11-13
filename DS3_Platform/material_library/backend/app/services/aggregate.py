"""Data aggregation across scenes: ROI pixels of several registered images -> one signature per class on a common grid."""
from __future__ import annotations
import io, uuid, zipfile
from pathlib import Path
import tempfile

import numpy as np
from sqlalchemy.orm import Session

from ..models import Image, Library, Dataset, Material, Sample, Spectrum
from .images import open_image, _roi_blocks
from .importer import _get_library, slugify, grid_info
from .taxonomy import normalize_category, normalize_acquisition
from ds3lib.aggregate import (common_grid, interp_weights, resample_rows, ClassAccumulator, write_csv, write_json, write_sli)
from ds3lib.aggregate.export import to_dict


def run(db: Session, items: list[dict], *, mode: str = "intersection", step: float | None = None,
        mask_bad_bands: bool = True) -> tuple[np.ndarray, list, list]:
    """items: [{image_id, class_name, polygon}]. Returns (grid_nm, [ClassStats], [per-item notes]).
    Each ROI is read at full resolution (sub-sampled above ~120k pixels), bad bands are masked, the pixels are
    resampled to the common grid and pooled per class name across all scenes."""
    if not items:
        raise ValueError("add at least one ROI first")
    imgs = {}
    for it in items:
        iid = uuid.UUID(str(it["image_id"]))
        if iid not in imgs:
            im = db.query(Image).filter(Image.id == iid).first()
            if im is None:
                raise ValueError(f"image {iid} not found")
            imgs[iid] = im
    grid = common_grid([im.wavelengths for im in imgs.values()], mode=mode, step=step)
    accs: dict[str, ClassAccumulator] = {}
    notes = []
    for it in items:
        im = imgs[uuid.UUID(str(it["image_id"]))]
        name = (it.get("class_name") or "").strip() or "class"
        cube = open_image(im)
        w = interp_weights(im.wavelengths, grid)
        bad = np.asarray(cube.bad_band_mask, bool) if mask_bad_bands else np.zeros(cube.bands, bool)
        acc = accs.setdefault(name, ClassAccumulator(grid.size))
        blocks, stride = _roi_blocks(cube, it["polygon"])
        before = acc.n_pixels
        for sel in blocks:
            sel[:, bad] = np.nan
            acc.add(resample_rows(sel, w), scene=im.name)
        notes.append({"image": im.name, "class": name, "pixels": acc.n_pixels - before, "stride": stride})
    stats = [a.result(n) for n, a in accs.items() if a.n_pixels > 0]
    if not stats:
        raise ValueError("none of the ROIs contains valid pixels")
    return grid, stats, notes


def units_of(db: Session, items) -> list[str]:
    ids = {uuid.UUID(str(i["image_id"])) for i in items}
    return sorted({im.unit or "" for im in db.query(Image).filter(Image.id.in_(ids)).all()})


def save_to_library(db: Session, grid, stats, *, library: str, category=None, acquisition=None, unit=None,
                    notes=None) -> dict:
    """Store the mean spectrum of every class (std, pixel count and source scenes go into the metadata)."""
    lib = _get_library(db, library.strip())
    ds_name = "aggregated signatures"
    ds = db.query(Dataset).filter(Dataset.library_id == lib.id, Dataset.slug == slugify(ds_name)).first()
    if ds is None:
        ds = Dataset(id=uuid.uuid4(), library_id=lib.id, name=ds_name, slug=slugify(ds_name), source_name="aggregation",
                     description="Per-class mean spectra pooled over several scenes", dataset_metadata={"origin": "aggregation"})
        db.add(ds); db.flush()
    cat, acq = normalize_category(category), normalize_acquisition(acquisition)
    saved = []
    for c in stats:
        ok = np.isfinite(c.mean)
        if ok.sum() < 3:
            continue
        w = [float(x) for x in grid[ok]]
        vals = [float(x) for x in c.mean[ok]]
        mat = Material(id=uuid.uuid4(), library_id=lib.id, name=c.name, category=cat,
                       material_metadata={"origin": "aggregation", **({"category_source": "user"} if cat else {})})
        db.add(mat); db.flush()
        smp = Sample(id=uuid.uuid4(), material_id=mat.id, dataset_id=ds.id, name=c.name, sample_number=1, sample_metadata={})
        db.add(smp); db.flush()
        step, uniform = grid_info(w)
        db.add(Spectrum(id=uuid.uuid4(), sample_id=smp.id, wavelength_start=w[0], wavelength_end=w[-1], wavelength_step=step,
                        measurement_unit=unit or "reflectance", values=vals, wavelengths=None if uniform else w,
                        acquisition_type=acq,
                        spectrum_metadata={"purity": "aggregated", "wavelength_unit": "nm", "point_count": len(vals),
                                           "std": [None if not np.isfinite(x) else float(x) for x in c.std[ok]],
                                           "n_pixels": c.n_pixels, "source_scenes": c.scenes}))
        saved.append(c.name)
    db.commit()
    return {"library": lib.name, "library_id": str(lib.id), "saved": saved}


def export_bytes(grid, stats, fmt: str, unit=None) -> tuple[bytes, str, str]:
    """(content, media type, file name) for fmt in csv | json | sli (sli = zip with .sli/.hdr and the _std pair)."""
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        if fmt == "csv":
            write_csv(td / "a.csv", grid, stats); return (td / "a.csv").read_bytes(), "text/csv", "aggregated_signatures.csv"
        if fmt == "json":
            write_json(td / "a.json", grid, stats, unit); return (td / "a.json").read_bytes(), "application/json", "aggregated_signatures.json"
        if fmt == "sli":
            write_sli(td / "aggregated_signatures.sli", grid, stats)
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
                for f in sorted(td.iterdir()):
                    z.write(f, f.name)
            return buf.getvalue(), "application/zip", "aggregated_signatures_sli.zip"
    raise ValueError("format must be csv, json or sli")
