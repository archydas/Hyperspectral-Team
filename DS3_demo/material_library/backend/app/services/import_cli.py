"""Command line importer.

  python -m app.services.import_cli data/material_spectra.csv --library "Team materials"
  python -m app.services.import_cli D:\\libraries\\usgs --library "USGS splib07" --unit reflectance
  python -m app.services.import_cli D:\\data\\PaviaU.mat --wavelengths 430-860      (an image -> Images table)
  python -m app.services.import_cli D:\\data\\enmap_scene.zip                          (EnMAP L2A zip)
  python -m app.services.import_cli --list
  python -m app.services.import_cli --delete-library "Team materials"
Run from the backend folder.
"""
from __future__ import annotations
import argparse
import sys

from sqlalchemy import func

from ..database import SessionLocal, init_db, display_url
from ..models import Library, Material, Spectrum, Sample, Dataset
from .importer import import_path, import_folder_as_libraries


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="import_cli", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="*", help="spectra file(s) or folder(s) to import")
    ap.add_argument("--library", help="library name (default: folder name or file name)")
    ap.add_argument("--dataset", help="dataset name (default: file name)")
    ap.add_argument("--category", help="material category for every spectrum (default: guessed from the name when possible)")
    ap.add_argument("--unit", choices=["reflectance", "radiance"], help="unit of the values if the file does not say")
    ap.add_argument("--wavelengths", metavar="LO-HI", help="wavelength range in nm for images that carry none (e.g. a .mat cube): 430-860")
    ap.add_argument("--reflectance-divisor", type=float, help="e.g. 10000 for int16 reflectance x10000 images")
    ap.add_argument("--each-file-a-library", action="store_true", help="folders: every file becomes its own library, named after the file")
    ap.add_argument("--list", action="store_true", help="list libraries in the database")
    ap.add_argument("--delete-library", metavar="NAME", help="delete a library and everything in it")
    a = ap.parse_args(argv)
    init_db()
    print(f"Database: {display_url()}")
    if a.list or a.delete_library:
        db = SessionLocal()
        try:
            if a.delete_library:
                lib = db.query(Library).filter(Library.name == a.delete_library).first()
                if lib is None:
                    print(f"no library named '{a.delete_library}'"); return 1
                db.delete(lib); db.commit(); print(f"deleted '{a.delete_library}'")
            for lib in db.query(Library).order_by(Library.name):
                m = db.query(func.count(Material.id)).filter(Material.library_id == lib.id).scalar()
                s = (db.query(func.count(Spectrum.id)).join(Sample, Spectrum.sample_id == Sample.id)
                     .join(Dataset, Sample.dataset_id == Dataset.id).filter(Dataset.library_id == lib.id).scalar())
                print(f"  {lib.name:40s} {m:5d} materials  {s:5d} spectra")
        finally:
            db.close()
        return 0
    if not a.paths:
        ap.error("give a file/folder to import, or --list")
    code = 0
    kw = {}
    if a.wavelengths:
        try:
            lo, hi = (float(x) for x in a.wavelengths.replace("\u2013", "-").split("-"))
        except ValueError:
            ap.error("--wavelengths must look like 430-860")
        kw["wavelength_range"] = (lo, hi)
    if a.reflectance_divisor:
        kw["reflectance_divisor"] = a.reflectance_divisor
    for p in a.paths:
        if a.each_file_a_library:
            for f in import_folder_as_libraries(p, category=a.category, unit=a.unit):
                print(f"  [{f['status']:9s}] {f['library']}  <- {f['file']}  {f['detail']}")
                code = code or (1 if f["status"] == "error" else 0)
            continue
        try:
            rep = import_path(p, library=a.library, dataset=a.dataset, category=a.category, unit=a.unit, reader_kw=kw)
        except ValueError as e:
            print(f"ERROR {e}"); code = 1; continue
        print(f"Library '{rep.library}':")
        for f in rep.files:
            print(f"  [{f['status']:9s}] {f['file']}  {f['detail']}")
            code = code or (1 if f["status"] == "error" else 0)
    return code


if __name__ == "__main__":
    sys.exit(main())
