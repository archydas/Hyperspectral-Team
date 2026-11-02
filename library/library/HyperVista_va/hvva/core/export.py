"""Signature export: CSV, JSON and ENVI spectral library (.sli + .hdr)."""
from __future__ import annotations

import csv
import json
import os
from typing import List

import numpy as np

from .scene import parse_envi_header, _hdr_list
from .signature import Signature


def _check(sigs: List[Signature]):
    if not sigs:
        raise ValueError("nothing to export")
    wl = sigs[0].wavelengths
    for s in sigs[1:]:
        if len(s.wavelengths) != len(wl) or not np.allclose(s.wavelengths, wl):
            raise ValueError("signatures are on different wavelength grids; aggregate on one grid first")
    return wl


def _num(v):
    return "" if v is None or not np.isfinite(v) else f"{float(v):.8g}"


def write_csv(path, sigs: List[Signature]):
    wl = _check(sigs)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["wavelength_nm"] + [f"{s.name}_{k}" for s in sigs for k in ("mean", "std")])
        w.writerow(["n_pixels"] + [v for s in sigs for v in (s.n_pixels, s.n_pixels)])
        for i, x in enumerate(wl):
            w.writerow([f"{x:.4f}"] + [_num(v[i]) for s in sigs for v in (s.mean, s.std)])
    return path


def write_json(path, sigs: List[Signature]):
    wl = _check(sigs)
    clean = lambda a: [None if not np.isfinite(v) else float(v) for v in a]
    doc = {"format": "hvva-signatures-1", "wavelength_units": "nm", "wavelengths": [float(v) for v in wl],
           "classes": {s.name: {"n_pixels": s.n_pixels, "source": s.source, "mean": clean(s.mean),
                                "std": clean(s.std), "meta": s.meta} for s in sigs}}
    with open(path, "w") as f:
        json.dump(doc, f, indent=1)
    return path


def _write_sli(base, names, wl, data, desc):
    data = np.nan_to_num(np.asarray(data, "<f4"), nan=-9999.0)
    data.tofile(base + ".sli")
    safe = [n.replace(",", "_").replace("{", "(").replace("}", ")") for n in names]
    lines = ["ENVI", f"description = {{{desc}}}", f"samples = {data.shape[1]}", f"lines = {data.shape[0]}",
            "bands = 1", "header offset = 0", "file type = ENVI Spectral Library", "data type = 4",
            "interleave = bsq", "byte order = 0", "data ignore value = -9999", "wavelength units = Nanometers",
            "spectra names = {" + ", ".join(safe) + "}",
            "wavelength = {" + ", ".join(f"{v:.4f}" for v in wl) + "}"]
    open(base + ".hdr", "w").write("\n".join(lines) + "\n")
    return base + ".sli"


def write_sli(base, sigs: List[Signature]):
    """base.sli/.hdr hold the class means, base_std.sli/.hdr the standard deviations."""
    wl = _check(sigs)
    base = os.path.splitext(base)[0]
    names = [s.name for s in sigs]
    a = _write_sli(base, names, wl, [s.mean for s in sigs], "class mean spectra")
    b = _write_sli(base + "_std", names, wl, [s.std for s in sigs], "class standard deviation spectra")
    return [a, b]


def read_sli(path):
    """Return (names, wavelengths_nm, data[n_spectra, n_bands]) from an ENVI spectral library."""
    base = os.path.splitext(path)[0]
    h = parse_envi_header(base + ".hdr")
    ns, nl = int(h["samples"]), int(h["lines"])
    data = np.fromfile(base + ".sli", "<f4").reshape(nl, ns)
    ign = float(h.get("data ignore value", "nan"))
    data = np.where(data == ign, np.nan, data)
    return _hdr_list(h["spectra names"]), np.array([float(v) for v in _hdr_list(h["wavelength"])]), data


def export_all(base, sigs, formats=("csv", "json", "sli")):
    paths = []
    if "csv" in formats:
        paths.append(write_csv(base + ".csv", sigs))
    if "json" in formats:
        paths.append(write_json(base + ".json", sigs))
    if "sli" in formats:
        paths += write_sli(base, sigs)
    return paths
