"""ds3 command line: formats | info | db | ingest | serve | sample-data | export-schema"""
from __future__ import annotations
import argparse, json, sys
from . import __version__


def _fmt_cmd(a):
    from .ingest.api import _READERS
    from .spectra import SPECTRA_FORMATS
    print("Hyperspectral cubes:  " + ", ".join(sorted(_READERS)) + "   (.hdr/.img, .tif, .zarr, .h5/.nc)")
    print("Point spectra / libraries:")
    for k, v in SPECTRA_FORMATS.items(): print(f"  {k:10s} {v}")


def _info(a):
    from .pipeline import classify
    k = classify(a.path)
    if k == "cube":
        from .ingest import open_cube
        c = open_cube(a.path, unit=a.unit)
        print(c.describe()); print(c.qc.summary() if c.qc else "")
    else:
        from .spectra import read_spectra
        for s in read_spectra(a.path):
            print(f"{s.material}: {len(s.values)} pts {s.wavelengths_nm[0]:.0f}-{s.wavelengths_nm[-1]:.0f} nm unit={s.unit}")


def _convert(a):
    from .ingest import open_cube, write_zarr
    c = open_cube(a.src, unit=a.unit); write_zarr(c, a.dst); print("wrote", a.dst)


def _db(a):
    from .db import init_db, reset_db, get_engine
    from .config import database_url
    if a.action == "init": init_db(); print("tables created on", _safe(database_url()))
    elif a.action == "reset":
        if not a.yes: sys.exit("refusing to drop all tables without --yes")
        reset_db(); print("database reset")
    else:
        from sqlalchemy import text
        init_db()
        with get_engine().connect() as c:
            for t in ("sources", "cubes", "libraries", "datasets", "materials", "samples", "spectra"):
                print(f"{t:10s} {c.execute(text(f'select count(*) from {t}')).scalar()}")


def _safe(url):
    import re; return re.sub(r"://([^:]+):[^@]+@", r"://\1:***@", url)


def _ingest(a):
    from .pipeline import ingest_path
    kw = {}
    if a.unit: kw["unit"] = a.unit
    res = ingest_path(a.path, source_name=a.source, kind=a.kind, url=a.url, license=a.license, library=a.library, **kw)
    for r in res: print(f"[{r.status:9s}] {r.kind or '-':7s} {r.path}  {r.detail}")
    n = {k: sum(r.status == k for r in res) for k in ("ingested", "duplicate", "skipped", "error")}
    print(n)
    sys.exit(1 if n["error"] else 0)


def _lib(a):
    import numpy as np
    from .db import session_scope, init_db
    init_db()
    if a.action == "build":
        from .library.builder import build_library
        from .library.grid import Grid
        g = a.grid
        if g == "first": grid = "first"
        elif g.startswith("cube:"): grid = Grid.from_cube(g[5:])
        elif g.startswith("ascii:"): grid = Grid.from_ascii(g[6:])
        else: sys.exit("--grid must be first | cube:<path> | ascii:<path>")
        rep = build_library(a.name, a.inputs, grid, cube=a.cube, method=a.method, unit=a.unit, min_coverage=a.min_coverage)
        print(rep.summary())
    elif a.action == "export":
        from .library.export import export_envi_sli
        print("wrote", export_envi_sli(a.name, a.out))
    elif a.action == "list":
        from .api.app import libraries
        from sqlalchemy.orm import Session
        with session_scope() as s:
            for l in libraries(s): print(f"{l['id']:4d} {l['name']:30s} {l['n_spectra']} spectra")
    elif a.action == "search":
        from .library.search import find_spectra
        with session_scope() as s:
            r = find_spectra(s, text=a.text, category=a.category, unit=a.unit, purity=a.purity, library=a.library,
                             covers_nm=tuple(a.covers) if a.covers else None, limit=a.limit)
        print(f"{r['total']} match(es)")
        for x in r["results"]:
            print(f"  #{x['id']:<5d} {x['material']:28s} {x['library']:20s} {x['unit']:11s} {x['purity']:15s} {x['wl_min_nm']:.0f}-{x['wl_max_nm']:.0f} nm")
    elif a.action == "match":
        from .library.search import match_spectrum
        from .db.models import SpectrumRecord
        with session_scope() as s:
            rec = s.get(SpectrumRecord, a.spectrum_id)
            if rec is None: sys.exit(f"no spectrum #{a.spectrum_id}")
            q = np.array([np.nan if x is None else x for x in rec.values], float)
            res = match_spectrum(s, rec.wavelengths_nm, q, top_k=a.top, unit=a.unit or rec.unit, library=a.library, exclude_id=rec.id)
        for x in res:
            print(f"  #{x['id']:<5d} {x['material']:28s} angle={x['score']:.4f} rad  overlap={x['overlap']*100:.0f}%  [{x['library']}]")


def _serve(a):
    import uvicorn; uvicorn.run("ds3lib.api.app:app", host=a.host, port=a.port, reload=a.reload)


def _sample(a):
    from .sampledata import generate
    for f in generate(a.out): print("  ", f)
    print("synthetic sample data written to", a.out)


def _schema(a):
    from sqlalchemy.schema import CreateTable, CreateIndex
    from sqlalchemy.dialects import postgresql
    from .db.models import Base
    out = ["-- DS3 PostgreSQL schema (generated from src/ds3/db/models.py; do not hand-edit)\n"]
    for t in Base.metadata.sorted_tables:
        out.append(str(CreateTable(t).compile(dialect=postgresql.dialect())).strip() + ";\n")
        for i in t.indexes: out.append(str(CreateIndex(i).compile(dialect=postgresql.dialect())) + ";")
    txt = "\n".join(out)
    if a.out: open(a.out, "w").write(txt); print("wrote", a.out)
    else: print(txt)


def main(argv=None):
    p = argparse.ArgumentParser(prog="ds3", description="DS3 Task 1 - hyperspectral data aggregation")
    p.add_argument("--version", action="version", version=__version__)
    sp = p.add_subparsers(dest="cmd", required=True)
    sp.add_parser("formats", help="list supported formats").set_defaults(f=_fmt_cmd)
    i = sp.add_parser("info", help="inspect one file"); i.add_argument("path"); i.add_argument("--unit"); i.set_defaults(f=_info)
    cv = sp.add_parser("convert", help="any cube -> standard DS3 Zarr"); cv.add_argument("src"); cv.add_argument("dst"); cv.add_argument("--unit"); cv.set_defaults(f=_convert)
    d = sp.add_parser("db", help="database admin"); d.add_argument("action", choices=["init", "reset", "status"])
    d.add_argument("--yes", action="store_true"); d.set_defaults(f=_db)
    g = sp.add_parser("ingest", help="ingest a file or folder into the database")
    g.add_argument("path"); g.add_argument("--source", default="unspecified"); g.add_argument("--kind", choices=["satellite", "airborne", "drone", "field", "library"])
    g.add_argument("--url"); g.add_argument("--license"); g.add_argument("--library"); g.add_argument("--unit"); g.set_defaults(f=_ingest)
    lb = sp.add_parser("library", help="ENVI-style library builder + search")
    lb.add_argument("action", choices=["build", "list", "search", "match", "export"])
    lb.add_argument("name", nargs="?", help="(build) library name")
    lb.add_argument("--inputs", nargs="+", help="(build) spectra files, ENVI ASCII ROI files, GeoJSON")
    lb.add_argument("--grid", default="first", help="(build) first | cube:<path> | ascii:<path>")
    lb.add_argument("--cube", help="(build) image the ROI/GeoJSON inputs are measured from")
    lb.add_argument("--method", default="linear", choices=["linear", "gaussian"])
    lb.add_argument("--min-coverage", type=float, default=0.5)
    lb.add_argument("--text"); lb.add_argument("--category"); lb.add_argument("--unit"); lb.add_argument("--purity")
    lb.add_argument("--library"); lb.add_argument("--covers", nargs=2, type=float, metavar=("LO_NM", "HI_NM"))
    lb.add_argument("--out", help="(export) output .sli path")
    lb.add_argument("--limit", type=int, default=20); lb.add_argument("--spectrum-id", type=int); lb.add_argument("--top", type=int, default=5)
    lb.set_defaults(f=_lib)
    s = sp.add_parser("serve", help="run the REST API"); s.add_argument("--host", default="127.0.0.1"); s.add_argument("--port", type=int, default=8000)
    s.add_argument("--reload", action="store_true"); s.set_defaults(f=_serve)
    sd = sp.add_parser("sample-data", help="write synthetic sample files"); sd.add_argument("--out", default="data/sample"); sd.set_defaults(f=_sample)
    e = sp.add_parser("export-schema", help="print PostgreSQL DDL"); e.add_argument("--out"); e.set_defaults(f=_schema)
    a = p.parse_args(argv)
    return a.f(a)


if __name__ == "__main__":
    main()
