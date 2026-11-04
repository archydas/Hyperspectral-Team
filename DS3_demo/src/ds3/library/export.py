"""Write a stored library back out as an ENVI spectral library (.sli + .hdr) - opens in ENVI's Spectral Library Viewer."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from sqlalchemy import select
from ..db import session_scope
from ..db.models import SpectrumRecord, Sample, Material, Dataset, Library
from ..ingest import IngestError


def export_envi_sli(library_name, out_path, db_url=None):
    out = Path(out_path).with_suffix(".sli")
    with session_scope(db_url) as s:
        rows = s.execute(select(SpectrumRecord, Material.name).select_from(SpectrumRecord)
                         .join(Sample, SpectrumRecord.sample_id == Sample.id).join(Material, Sample.material_id == Material.id)
                         .join(Dataset, Sample.dataset_id == Dataset.id).join(Library, Dataset.library_id == Library.id)
                         .where(Library.name == library_name).order_by(SpectrumRecord.id)).all()
    if not rows: raise IngestError(f"no spectra in library '{library_name}'")
    wl0 = rows[0][0].wavelengths_nm
    if any(r.wavelengths_nm != wl0 for r, _ in rows):
        raise IngestError(f"library '{library_name}' mixes wavelength grids; ENVI .sli needs one grid - rebuild it with `ds3 library build`")
    units = {r.unit for r, _ in rows}
    arr = np.array([[np.nan if x is None else x for x in r.values] for r, _ in rows], "<f4")
    arr = np.where(np.isfinite(arr), arr, -9999.0).astype("<f4")
    arr.tofile(out)
    names = [m for _, m in rows]
    hdr = ["ENVI", "description = {DS3 export of library %s (unit: %s)}" % (library_name, "/".join(sorted(units))),
           f"samples = {arr.shape[1]}", f"lines = {arr.shape[0]}", "bands = 1", "header offset = 0",
           "file type = ENVI Spectral Library", "data type = 4", "interleave = bsq", "byte order = 0",
           "data ignore value = -9999", "wavelength units = Nanometers",
           "spectra names = {" + ", ".join(names) + "}",
           "wavelength = {" + ", ".join(f"{w:.4f}" for w in wl0) + "}"]
    out.with_suffix(".hdr").write_text("\n".join(hdr))
    return str(out)


def export_csv(library_name, out_path, db_url=None):
    """Wide CSV: first column 'Wavelength' (nm), then one column per spectrum. Needs ONE wavelength grid.
    Same layout the team's material_spectra.csv uses, so it can be re-imported by `ds3 ingest`."""
    out = Path(out_path).with_suffix(".csv")
    with session_scope(db_url) as s:
        rows = s.execute(select(SpectrumRecord, Material.name).select_from(SpectrumRecord)
                         .join(Sample, SpectrumRecord.sample_id == Sample.id).join(Material, Sample.material_id == Material.id)
                         .join(Dataset, Sample.dataset_id == Dataset.id).join(Library, Dataset.library_id == Library.id)
                         .where(Library.name == library_name).order_by(SpectrumRecord.id)).all()
        if not rows: raise IngestError(f"no spectra in library '{library_name}'")
        wl0 = rows[0][0].wavelengths_nm
        if any(r.wavelengths_nm != wl0 for r, _ in rows):
            raise IngestError(f"library '{library_name}' mixes wavelength grids; CSV needs one grid - rebuild it with the Library Builder")
        names, seen = [], {}
        for _, m in rows:
            seen[m] = seen.get(m, 0) + 1
            names.append(m if seen[m] == 1 else f"{m}_{seen[m]}")
        import csv
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["Wavelength"] + names)
            for i, wl in enumerate(wl0):
                w.writerow([wl] + ["" if r.values[i] is None else r.values[i] for r, _ in rows])
    return str(out)
