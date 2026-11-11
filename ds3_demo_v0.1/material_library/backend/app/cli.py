"""Command line: import files, list libraries, aggregate ROIs of several images, start the app.

    python run_cli.py import D:\\data\\soil_spectra.txt --library Soil --category Soil --acquisition lab
    python run_cli.py list
    python run_cli.py formats
    python run_cli.py aggregate rois.json --out results          (see `aggregate --help` for the file layout)
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path


def _db():
    from .database import init_db, SessionLocal
    init_db()
    return SessionLocal()


def cmd_import(a):
    """Every path is tried; one bad path does not stop the others. For an ENVI image give the .hdr (its data file is found beside it)."""
    from .services.importer import import_path
    failed = 0
    for path in a.paths:
        try:
            rep = import_path(path, library=a.library, dataset=a.dataset, category=a.category, unit=a.unit, acquisition=a.acquisition)
        except ValueError as e:
            print(f"error     {path}: {e}", file=sys.stderr); failed += 1; continue
        for f in rep.files:
            print(f"{f['status']:9s} {f['file']}: {f['detail']}")
        d = rep.as_dict()
        print(f"-> '{d['library']}': {d['imported']} spectra, {d['images']} image(s), {d['duplicates']} duplicate(s), {d['errors']} error(s)")
    return 1 if failed else 0


def cmd_list(a):
    from .models import Library, Image
    from .routes.libraries import get_libraries
    db = _db()
    try:
        for l in get_libraries(db=db):
            print(f"library  {l['name']:40s} {l['material_count']:6d} materials {l['spectrum_count']:7d} spectra")
        for im in db.query(Image).all():
            print(f"image    {im.name:40s} {im.rows}x{im.cols}x{im.bands}  {im.wl_min_nm:.0f}-{im.wl_max_nm:.0f} nm  [{im.format}]")
    finally:
        db.close()
    return 0


def cmd_formats(a):
    try:
        from ds3.spectra import SPECTRA_FORMATS
    except Exception:
        print("the ds3 package is not installed (pip install -e <path to ds3-platform>)"); return 1
    print("Spectra formats:")
    for k, v in SPECTRA_FORMATS.items():
        print(f"  {k:10s} {v}")
    print("Image formats: ENVI (.hdr + data), GeoTIFF / EnMAP, MATLAB .mat, HDF5, NetCDF, Zarr; .zip archives of any of these")
    return 0


def cmd_aggregate(a):
    """rois.json:  {"rois": [{"image": "<image name or id>", "class_name": "Water", "polygon": [[row, col], ...]}, ...],
                    "mode": "intersection" | "union", "step": null, "save": {"library": "...", "category": "...", "acquisition_type": "lab"}}"""
    from .models import Image
    from .services import aggregate as svc
    spec = json.loads(Path(a.spec).read_text(encoding="utf-8"))
    db = _db()
    try:
        items = []
        for r in spec["rois"]:
            key = str(r.get("image") or r.get("image_id"))
            im = next((i for i in db.query(Image).all() if str(i.id) == key or i.name == key), None)
            if im is None:
                print(f"image not found: {key}", file=sys.stderr); return 1
            items.append({"image_id": str(im.id), "class_name": r.get("class_name", "class"), "polygon": r["polygon"]})
        grid, stats, notes = svc.run(db, items, mode=spec.get("mode", "intersection"), step=spec.get("step"))
        units = svc.units_of(db, items)
        out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
        for fmt in ("csv", "json", "sli"):
            data, _, name = svc.export_bytes(grid, stats, fmt, units[0] if len(units) == 1 else None)
            (out / name).write_bytes(data)
        for n in notes:
            print(f"  {n['class']:20s} {n['image']:30s} {n['pixels']} pixels")
        print(f"{len(stats)} class(es), {len(grid)} bands ({grid[0]:.1f}-{grid[-1]:.1f} nm) -> {out}")
        if spec.get("save"):
            s = spec["save"]
            r = svc.save_to_library(db, grid, stats, library=s["library"], category=s.get("category"),
                                    acquisition=s.get("acquisition_type"), unit=units[0] if len(units) == 1 else None)
            print(f"saved {len(r['saved'])} signature(s) to library '{r['library']}'")
    finally:
        db.close()
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="run_cli.py", description="DS3 Library command line")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("import", help="import spectra files, images, folders or zips")
    p.add_argument("paths", nargs="+")
    p.add_argument("--library"); p.add_argument("--dataset"); p.add_argument("--category", help="e.g. Vegetation/Tree")
    p.add_argument("--unit"); p.add_argument("--acquisition", choices=["lab", "field", "airborne", "satellite"])
    p.set_defaults(fn=cmd_import)
    sub.add_parser("list", help="list libraries and images").set_defaults(fn=cmd_list)
    sub.add_parser("formats", help="supported formats").set_defaults(fn=cmd_formats)
    p = sub.add_parser("aggregate", help="aggregate ROIs of several images (see module docstring for the JSON layout)")
    p.add_argument("spec"); p.add_argument("--out", default="aggregation_out")
    p.set_defaults(fn=cmd_aggregate)
    a = ap.parse_args(argv)
    try:
        return a.fn(a)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr); return 1


if __name__ == "__main__":
    sys.exit(main())
