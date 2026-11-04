"""Image -> library: measure ROI spectra from a registered cube and store them (purity = image_pixel)."""
from __future__ import annotations
import hashlib, json
from ..db import session_scope, init_db
from ..ingest import extract_roi, IngestError
from ..ingest.contract import Spectrum
from ..pipeline import store_spectra, _get_source
from .access import open_registered_cube
from .sources import (spectra_from_roi_ascii, spectra_from_geojson, is_envi_roi, read_label_map, spectra_from_labels,
                      PAVIA_U_CLASSES)


def measure(cube_id, *, pixel=None, bbox_px=None, roi_file=None, name="ROI", category=None, one_based=True,
            db_url=None):
    """Return (list[Spectrum], description). Exactly one of pixel=(row, col), bbox_px=(r0, c0, r1, c1) (end-exclusive),
    roi_file=path (ENVI ASCII ROI export or GeoJSON). Pixels that are nodata in every band are excluded and counted."""
    given = [k for k, v in dict(pixel=pixel, bbox_px=bbox_px, roi_file=roi_file).items() if v]
    if len(given) != 1:
        raise IngestError(f"give exactly one of pixel / bbox_px / roi_file, got {given or 'none'}")
    with session_scope(db_url) as s:
        cube, rec = open_registered_cube(s, cube_id)
    if roi_file:
        spectra = (spectra_from_roi_ascii(roi_file, cube, one_based, category) if is_envi_roi(roi_file)
                   else spectra_from_geojson(roi_file, cube, category=category))
        desc = f"roi_file={roi_file}"
    else:
        sel = dict(pixel=pixel) if pixel else dict(bbox_px=bbox_px)
        res = extract_roi(cube, **sel)
        if res.n_pixels == 0:
            raise IngestError(f"the selection covers no valid pixels (cube is {cube.rows} rows x {cube.cols} cols; "
                              f"{res.n_excluded_nodata} nodata pixels excluded)")
        sp = res.to_spectrum(name, purity="image_pixel", category=category)
        sp.metadata.update(origin="image_pixel", n_excluded_nodata=res.n_excluded_nodata)
        spectra, desc = [sp], json.dumps(sel)
    for sp in spectra:
        sp.metadata.update(cube_id=rec["id"], cube_name=rec["name"], roi=desc)
    return spectra, desc


def save(spectra, cube_id, *, library="ROI spectra", db_url=None):
    """Store measured spectra in `library`. Saving the same ROI under the same name twice is refused (idempotent)."""
    init_db(db_url)
    digest = hashlib.sha256(("roi|" + str(cube_id) + "|" + library + "|" +
                             "|".join(f"{sp.material}:{sp.metadata.get('roi')}" for sp in spectra)).encode()).hexdigest()
    from sqlalchemy import select
    from ..db.models import Dataset
    with session_scope(db_url) as s:
        if s.scalar(select(Dataset).where(Dataset.sha256 == digest)):
            raise IngestError("these ROI spectra were already saved to this library (nothing changed)")
        src = _get_source(s, "image-roi", "library")
        return store_spectra(s, spectra, source=src, library_name=library,
                             dataset_name=f"roi:{spectra[0].metadata.get('cube_name', cube_id)}",
                             digest=digest, fmt="roi")


def measure_labels(cube_id, labels_file, *, names=None, category=None, db_url=None):
    """Mean spectrum per class of a ground-truth label map (.mat/.npy/.tif, 0 = unlabelled) over a registered cube.
    names: {id: name} / list; default = the published Pavia University class names when the map has classes 1-9 and the
    cube/label file is called 'pavia', else "class N". Returns (list[Spectrum], description)."""
    with session_scope(db_url) as s:
        cube, rec = open_registered_cube(s, cube_id)
    labels, var = read_label_map(labels_file)
    if names is None and labels.max() <= len(PAVIA_U_CLASSES) and "pavia" in (str(labels_file) + rec["name"]).lower():
        names = PAVIA_U_CLASSES
    spectra = spectra_from_labels(cube, labels, names, category)
    desc = f"labels={labels_file}"
    for sp in spectra:
        sp.metadata.update(cube_id=rec["id"], cube_name=rec["name"], roi=f"{desc}#{sp.metadata.get('label_id')}")
    return spectra, desc
