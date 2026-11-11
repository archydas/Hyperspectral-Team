"""Write aggregated signatures as CSV, JSON or an ENVI spectral library (.sli + .hdr, plus a _std library)."""
from __future__ import annotations
import csv, json
from pathlib import Path
import numpy as np


def _f(x):
    return None if x is None or not np.isfinite(x) else float(x)


def write_csv(path, wavelengths, stats):
    """One row per wavelength; columns <class>_mean, <class>_std, <class>_n for every class."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["wavelength_nm"] + [f"{c.name}_{k}" for c in stats for k in ("mean", "std", "n")])
        for i, wl in enumerate(wavelengths):
            row = [f"{wl:.4f}"]
            for c in stats:
                row += ["" if not np.isfinite(c.mean[i]) else f"{c.mean[i]:.6g}",
                        "" if not np.isfinite(c.std[i]) else f"{c.std[i]:.6g}", int(c.n[i])]
            w.writerow(row)


def to_dict(wavelengths, stats, unit=None):
    return {"wavelength_unit": "nm", "unit": unit, "wavelengths": [float(x) for x in wavelengths],
            "classes": [{"name": c.name, "n_pixels": c.n_pixels, "scenes": c.scenes,
                         "mean": [_f(x) for x in c.mean], "std": [_f(x) for x in c.std],
                         "min": [_f(x) for x in c.minimum], "max": [_f(x) for x in c.maximum],
                         "n": [int(x) for x in c.n]} for c in stats]}


def write_json(path, wavelengths, stats, unit=None):
    Path(path).write_text(json.dumps(to_dict(wavelengths, stats, unit), indent=1), encoding="utf-8")


def _sli(path, wavelengths, names, rows, description):
    arr = np.asarray(rows, "<f4")
    arr = np.where(np.isfinite(arr), arr, -9999.0).astype("<f4")
    p = Path(path)
    p.write_bytes(arr.tobytes())
    hdr = ("ENVI\n"
           f"description = {{{description}}}\n"
           f"samples = {arr.shape[1]}\nlines = {arr.shape[0]}\nbands = 1\nheader offset = 0\n"
           "file type = ENVI Spectral Library\ndata type = 4\ninterleave = bsq\nbyte order = 0\n"
           "data ignore value = -9999\nwavelength units = Nanometers\n"
           "spectra names = {" + ", ".join(n.replace(",", ";") for n in names) + "}\n"
           "wavelength = {" + ", ".join(f"{w:.4f}" for w in wavelengths) + "}\n")
    p.with_suffix(".hdr").write_text(hdr, encoding="utf-8")


def write_sli(path, wavelengths, stats):
    """`path` (mean spectra) and `<path stem>_std.sli` (standard deviation), each with its .hdr."""
    p = Path(path).with_suffix(".sli")
    names = [c.name for c in stats]
    _sli(p, wavelengths, names, [c.mean for c in stats], "DS3 aggregated mean spectra")
    _sli(p.with_name(p.stem + "_std.sli"), wavelengths, names, [c.std for c in stats], "DS3 aggregated standard deviation")
    return p
