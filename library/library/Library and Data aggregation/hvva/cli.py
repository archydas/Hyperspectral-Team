"""Headless command-line entry point:  python -m hvva <command> ...   (no Qt needed)."""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

from . import APP_NAME, __version__
from .core import export, plugins as plugmod
from .core.classify import kmeans_labels
from .core.mosaic import composite, mosaic, stack_cube
from .core.resample import common_grid
from .core.scene import load_scene, save_scene, save_envi
from .core.signature import signatures_from_result
from .core.stats import aggregate_classes
from .core.synthetic import make_demo_scenes
from .library.db import SignatureLibrary


def _load(paths, a=None):
    """Load scenes of any format; the --cube-key/--band-axis/... options are guessed when omitted."""
    opts = {}
    if a is not None:
        opts = dict(cube_key=a.cube_key, band_axis=a.band_axis, labels_key=a.labels_key,
                    wavelengths_key=a.wl_key, wavelength_range=tuple(a.wl_range) if a.wl_range else None)
        if a.raw:
            ns, nl, nb, dt, il, off = (a.raw.split(",") + ["uint16", "bsq", "0"][len(a.raw.split(",")) - 3:])[:6]
            opts["raw"] = dict(samples=int(ns), lines=int(nl), bands=int(nb), dtype=dt, interleave=il, offset=int(off))
    scenes = [load_scene(p, **opts) for p in paths]
    if a is not None and getattr(a, "labels_file", None):
        from .core.loaders import open_source
        src = open_source(a.labels_file)
        key = a.labels_key or next(k for k, (s, _) in src.arrays.items() if len(s) == 2)
        lab = np.squeeze(np.asarray(src.reader(key)))
        for sc in scenes:
            if lab.shape != sc.shape[:2]:
                raise ValueError(f"label map {lab.shape} does not match '{sc.name}' {sc.shape[:2]}")
            sc.labels = np.nan_to_num(lab).astype(np.int32)
            sc.class_names = {int(i): f"class_{int(i)}" for i in np.unique(sc.labels) if i}
    return scenes


def _grid(scenes, a):
    return common_grid([s.wavelengths for s in scenes], a.grid, a.step)


def cmd_demo(a):
    os.makedirs(a.out, exist_ok=True)
    for sc in make_demo_scenes(a.n):
        p = os.path.join(a.out, sc.name + ".npz")
        save_scene(sc, p)
        print("wrote", p)


def cmd_info(a):
    for sc in _load(a.scenes, a):
        print(sc.summary())
        if sc.geo:
            print(sc.geo.describe(*sc.shape[:2]))


def cmd_aggregate(a):
    scenes = _load(a.scenes, a)
    if a.kmeans:
        for sc in scenes:
            sc.labels = kmeans_labels(sc.cube, a.kmeans)
            sc.class_names = {i: f"class_{i}" for i in range(1, a.kmeans + 1)}
    grid = _grid(scenes, a)
    res = aggregate_classes(scenes, grid, a.max_pixels)
    sigs = signatures_from_result(res)
    for s in sigs:
        print(f"{s.name:<14} n={s.n_pixels:<8} scenes={s.source}")
    for p in export.export_all(a.out, sigs, tuple(a.formats.split(","))):
        print("wrote", p)
    if a.db:
        lib = SignatureLibrary(a.db)
        lib.add_many(sigs)
        print(f"library {a.db}: {lib.count()} signatures")


def cmd_mosaic(a):
    scenes = _load(a.scenes, a)
    res = mosaic(scenes, _grid(scenes, a), a.blend)
    save_scene(res, a.out)
    print("mosaic", res.shape, "->", a.out)
    if a.envi:
        save_envi(res, os.path.splitext(a.out)[0])


def cmd_stack(a):
    scenes = _load(a.scenes, a)
    g = _grid(scenes, a)
    if a.method == "cube":
        arr, geo = stack_cube(scenes, g)
        np.save(a.out, arr)
        print("stack", arr.shape, "->", a.out)
    else:
        res = composite(scenes, g, a.method)
        save_scene(res, a.out)
        print("composite", res.shape, "->", a.out)


def cmd_plugins(a):
    pl, errs = plugmod.discover(a.dir)
    for p in pl:
        print(f"{p.kind:<11} {p.name:<28} {p.description}")
    for e in errs:
        print("ERROR", e, file=sys.stderr)


def cmd_run_plugin(a):
    pl, _ = plugmod.discover(a.dir)
    plug = next((p for p in pl if p.name.lower() == a.name.lower()), None)
    if plug is None:
        sys.exit(f"no plugin named {a.name!r}")
    params = {}
    for kv in a.param or []:
        k, v = kv.split("=", 1)
        params[k] = plugmod._coerce(plug.params[k], v)
    sc = load_scene(a.scene)
    out = plug.apply(sc, **params)
    if plug.kind == "correction":
        save_scene(out, a.out)
        print("wrote", a.out)
    else:
        np.save(a.out, out["image"])
        print("table:", out.get("table"), "| image ->", a.out)


def cmd_library(a):
    lib = SignatureLibrary(a.db)
    if a.action == "list":
        for r in lib.rows(a.filter or ""):
            print(r)
    elif a.action == "export":
        sigs = [lib.get(r[0]) for r in lib.rows(a.filter or "")]
        for p in export.export_all(a.out, sigs):
            print("wrote", p)


def build_parser():
    p = argparse.ArgumentParser("hvva", description=f"{APP_NAME} {__version__} - headless tools")
    sub = p.add_subparsers(dest="cmd", required=True)

    def load_args(q):
        q.add_argument("--cube-key", help="array name inside .mat/.h5/.nc/.npz/.zarr (default: largest 3-D array)")
        q.add_argument("--band-axis", type=int, help="spectral axis of the array (default: the shortest one)")
        q.add_argument("--labels-key", help="ground-truth / class map array (default: guessed)")
        q.add_argument("--labels-file", help="separate ground-truth file (any format), applied to every scene")
        q.add_argument("--wl-key", help="wavelength array (default: guessed)")
        q.add_argument("--wl-range", type=float, nargs=2, metavar=("START", "STOP"),
                       help="nm range to spread over the bands when the file has no wavelengths")
        q.add_argument("--raw", metavar="SAMPLES,LINES,BANDS[,DTYPE,INTERLEAVE,OFFSET]",
                       help="headerless binary, e.g. 610,340,103,uint16,bil,0")

    def grid_args(q):
        q.add_argument("--grid", choices=["intersection", "union"], default="intersection")
        q.add_argument("--step", type=float, default=None, help="grid step in nm (default: coarsest scene)")

    q = sub.add_parser("demo", help="write synthetic demo scenes"); q.add_argument("--out", default="demo_scenes")
    q.add_argument("--n", type=int, default=3); q.set_defaults(f=cmd_demo)
    q = sub.add_parser("info", help="describe scenes"); q.add_argument("scenes", nargs="+"); load_args(q); q.set_defaults(f=cmd_info)
    q = sub.add_parser("aggregate", help="per-class statistics across scenes + export")
    q.add_argument("scenes", nargs="+"); grid_args(q); load_args(q)
    q.add_argument("--out", required=True, help="output base name (no extension)")
    q.add_argument("--formats", default="csv,json,sli"); q.add_argument("--db", help="also store in this SQLite library")
    q.add_argument("--kmeans", type=int, default=0, help="auto-classify each scene into K classes first")
    q.add_argument("--max-pixels", type=int, default=None); q.set_defaults(f=cmd_aggregate)
    q = sub.add_parser("mosaic", help="merge scenes spatially"); q.add_argument("scenes", nargs="+"); grid_args(q); load_args(q)
    q.add_argument("--blend", default="mean", choices=["mean", "first", "last", "max"])
    q.add_argument("--out", required=True); q.add_argument("--envi", action="store_true", help="also write ENVI .img/.hdr")
    q.set_defaults(f=cmd_mosaic)
    q = sub.add_parser("stack", help="stack / composite scenes of the same area"); q.add_argument("scenes", nargs="+")
    grid_args(q); load_args(q); q.add_argument("--method", default="mean", choices=["cube", "mean", "median", "max", "min"])
    q.add_argument("--out", required=True); q.set_defaults(f=cmd_stack)
    q = sub.add_parser("plugins", help="list plugins"); q.add_argument("--dir"); q.set_defaults(f=cmd_plugins)
    q = sub.add_parser("run-plugin", help="run a plugin on a scene"); q.add_argument("name"); q.add_argument("scene")
    q.add_argument("--param", action="append", help="key=value"); q.add_argument("--dir")
    q.add_argument("--out", required=True); q.set_defaults(f=cmd_run_plugin)
    q = sub.add_parser("library", help="inspect / export the SQLite library"); q.add_argument("action", choices=["list", "export"])
    q.add_argument("--db", required=True); q.add_argument("--filter"); q.add_argument("--out", default="library_export")
    q.set_defaults(f=cmd_library)
    return p


def main(argv=None):
    a = build_parser().parse_args(argv)
    a.f(a)


if __name__ == "__main__":
    main()
