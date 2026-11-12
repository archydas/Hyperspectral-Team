"""Spectral Library Builder, library resampling and whole-library export (no GUI code).

The ENVI "Spectral Library Builder" workflow:
    1. output wavelength grid  - from an image (its bands, optionally a band subset), an ASCII file
                                 (column 1 = wavelength, optional column 2 = FWHM; nm or µm), the first input
                                 spectrum, or a regular range (start, stop, step)
    2. inputs                  - library spectra, spectra files, image ROIs / pixels / collected spectra
    3. resample                - linear or Gaussian (FWHM).  No extrapolation; gaps wider than 3x the input's
                                 median band spacing (masked water bands ...) are not bridged
    4. check                   - one value unit per library; spectra covering less than *min_coverage* of the grid
                                 are rejected - every rejection is reported with its reason
    5. save                    - as a library in the database (new or existing)
"""
from __future__ import annotations

import csv
import json
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..core import quality as Q

METHODS = {"linear": "Linear interpolation", "gaussian": "Gaussian (band FWHM)"}


# ================================================================== grids
def grid_from_range(start: float, stop: float, step: float) -> np.ndarray:
    if step <= 0 or stop <= start:
        raise ValueError("the range needs start < stop and a positive step")
    n = int(np.floor((stop - start) / step + 1e-9)) + 1
    if n > 100000:
        raise ValueError("that grid would have more than 100 000 bands")
    return start + step * np.arange(n)


def grid_from_text(text: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Wavelengths (and optional FWHM) from typed text or an ASCII file's content.

    Accepts '450, 460, 470' / one value per line / two columns (wavelength, FWHM).  Header lines and comments
    (';' '#') are skipped; values below 30 are taken as µm and converted to nm."""
    wl, fw = [], []
    lines = [ln for ln in re.split(r"[\r\n]+", text or "") if ln.strip() and not ln.strip().startswith((";", "#"))]
    if len(lines) == 1:                                           # one line: a list
        nums = [float(v) for v in re.findall(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", lines[0])]
        wl = nums
    else:
        for ln in lines:
            nums = re.findall(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", ln)
            if not nums:
                continue
            wl.append(float(nums[0]))
            if len(nums) > 1:
                fw.append(float(nums[1]))
    wl = np.asarray(wl, float)
    if wl.size < 2:
        raise ValueError("at least two wavelengths are needed")
    f = np.asarray(fw, float) if len(fw) == wl.size else None
    if np.nanmax(wl) < 30:
        wl = wl * 1000.0
        f = None if f is None else f * 1000.0
    order = np.argsort(wl)
    wl = wl[order]
    f = None if f is None else f[order]
    keep = np.concatenate([[True], np.diff(wl) > 0])
    return wl[keep], (None if f is None else f[keep])


def grid_from_scene(scene, first: Optional[int] = None, last: Optional[int] = None, skip_bad: bool = True):
    """(wavelengths, fwhm or None) of an image's bands - optionally a band subset (1-based, inclusive)."""
    if not scene.has_wavelengths:
        raise ValueError(f"{scene.name} has no wavelengths")
    i0 = 0 if first is None else max(0, int(first) - 1)
    i1 = scene.n_bands if last is None else min(scene.n_bands, int(last))
    idx = np.arange(i0, i1)
    if skip_bad and hasattr(scene, "bad_bands"):
        idx = idx[~scene.bad_bands[idx]]
    wl = np.asarray(scene.wavelengths, float)[idx]
    fwhm = getattr(scene, "fwhm", None)
    return wl, (None if fwhm is None else np.asarray(fwhm, float)[idx])


# ================================================================== resampling
def _median_step(x):
    d = np.diff(np.asarray(x, float))
    d = d[d > 0]
    return float(np.median(d)) if d.size else 1.0


def resample_linear(wl, y, grid, max_gap_factor: float = 3.0) -> np.ndarray:
    """Linear; NaN outside the input range and inside gaps wider than max_gap_factor x the median spacing."""
    wl, y, grid = np.asarray(wl, float), np.asarray(y, float), np.asarray(grid, float)
    ok = np.isfinite(wl) & np.isfinite(y)
    wl, y = wl[ok], y[ok]
    out = np.full(grid.shape, np.nan)
    if wl.size < 2:
        return out
    order = np.argsort(wl)
    wl, y = wl[order], y[order]
    inside = (grid >= wl[0]) & (grid <= wl[-1])
    out[inside] = np.interp(grid[inside], wl, y)
    gap = max_gap_factor * _median_step(wl)
    j = np.clip(np.searchsorted(wl, grid), 1, wl.size - 1)
    wide = (wl[j] - wl[j - 1]) > gap
    out[inside & wide & ~np.isin(grid, wl)] = np.nan
    return out


def resample_gaussian(wl, y, grid, fwhm=None, max_gap_factor: float = 3.0) -> np.ndarray:
    """Each output band = Gaussian-weighted mean of the input around it (FWHM of the output band; default the
    grid spacing).  Bands whose Gaussian falls mostly outside the input / into a gap become NaN."""
    wl, y, grid = np.asarray(wl, float), np.asarray(y, float), np.asarray(grid, float)
    ok = np.isfinite(wl) & np.isfinite(y)
    wl, y = wl[ok], y[ok]
    out = np.full(grid.shape, np.nan)
    if wl.size < 2:
        return out
    if fwhm is None:
        step = np.gradient(grid) if grid.size > 1 else np.array([1.0])
        fwhm = np.abs(step)
    fwhm = np.broadcast_to(np.asarray(fwhm, float), grid.shape)
    sig = np.maximum(fwhm, 1e-6) / 2.3548
    lin = resample_linear(wl, y, grid, max_gap_factor)
    for k, (g, s) in enumerate(zip(grid, sig)):
        if not np.isfinite(lin[k]):
            continue
        w = np.exp(-0.5 * ((wl - g) / s) ** 2)
        if w.sum() < 1e-6 or (np.abs(wl - g) <= 2 * s).sum() == 0:
            out[k] = lin[k]                                       # input coarser than the band: interpolate
        else:
            out[k] = float((w * y).sum() / w.sum())
    return out


def coverage(y_on_grid) -> float:
    y = np.asarray(y_on_grid, float)
    return float(np.isfinite(y).mean()) if y.size else 0.0


def resample_one(wl, y, grid, method="linear", fwhm=None, min_coverage=0.5):
    """(values on grid or None, coverage, reason if rejected)."""
    if method not in METHODS:
        raise ValueError(f"method must be one of {', '.join(METHODS)}")
    out = resample_gaussian(wl, y, grid, fwhm) if method == "gaussian" else resample_linear(wl, y, grid)
    cov = coverage(out)
    if cov < min_coverage:
        wl = np.asarray(wl, float)
        return None, cov, (f"covers only {cov * 100:.0f}% of the grid (spectrum {np.nanmin(wl):g}-{np.nanmax(wl):g} nm, "
                           f"grid {grid[0]:g}-{grid[-1]:g} nm; minimum {min_coverage * 100:.0f}%)")
    return out, cov, ""


# ================================================================== builder
@dataclass
class BuildInput:
    name: str
    wavelengths: np.ndarray
    values: np.ndarray
    source: str = ""                         # "library X", "file.csv", "image A / ROI 3" ...
    category: Optional[str] = None
    acquisition: Optional[str] = None
    unit: Optional[str] = None
    purity: Optional[str] = None
    metadata: Dict = field(default_factory=dict)


@dataclass
class BuildResult:
    grid: np.ndarray
    added: List[Tuple[BuildInput, np.ndarray, float]]            # (input, values on grid, coverage)
    rejected: List[Tuple[BuildInput, str]]                        # (input, reason)
    unit: str


def build(inputs: Sequence[BuildInput], grid, method="linear", fwhm=None, min_coverage=0.5,
          unit: Optional[str] = None) -> BuildResult:
    """Resample every input onto *grid*; reject (with the reason) what does not fit.

    unit: the library's unit (None = taken from the first input with a known unit).  Inputs in another known
    unit are rejected - radiance and reflectance are never mixed or converted."""
    grid = np.asarray(grid, float)
    if grid.size < 2:
        raise ValueError("choose an output wavelength grid first")
    lib_unit = Q.normalize_unit(unit) if unit else None
    added, rejected = [], []
    seen = set()
    for it in inputs:
        u = Q.normalize_unit(it.unit) or "unknown"
        if lib_unit in (None, "unknown") and u != "unknown":
            lib_unit = u
        if u not in ("unknown",) and lib_unit not in (None, "unknown") and u != lib_unit:
            rejected.append((it, f"{u}, but the library is {lib_unit} - units are never mixed"))
            continue
        v = np.asarray(it.values, float)
        key = (it.name, it.source, v.size, round(float(np.nansum(v)), 6), round(float(np.nansum(v * v)), 6))
        if key in seen:
            rejected.append((it, "already in the list (same name, source and values)"))
            continue
        y, cov, why = resample_one(it.wavelengths, it.values, grid, method, fwhm, min_coverage)
        if y is None:
            rejected.append((it, why))
            continue
        seen.add(key)
        added.append((it, y, cov))
    return BuildResult(grid, added, rejected, lib_unit or "unknown")


def result_items(res: BuildResult, method: str) -> List[dict]:
    """Items for client.save_spectra()."""
    out = []
    for it, y, cov in res.added:
        out.append({"name": it.name, "wavelengths": res.grid.tolist(),
                    "values": [None if not np.isfinite(v) else float(v) for v in y],
                    "category": it.category, "acquisition_type": it.acquisition,
                    "unit": it.unit if it.unit not in (None, "unknown") else res.unit,
                    "purity": it.purity,
                    "material_metadata": {"created_by": "HyperVista Library Builder", "source": it.source},
                    "spectrum_metadata": {**{k: v for k, v in (it.metadata or {}).items()
                                             if isinstance(v, (str, int, float, bool)) or v is None},
                                          "builder_source": it.source, "resampling": method,
                                          "coverage": round(cov, 3)}})
    return out


# ================================================================== whole-library export
def common_library_grid(entries) -> Tuple[Optional[np.ndarray], int]:
    """(the grid every entry already shares, or None, number of distinct grids)."""
    from .repository import spectrum_wavelengths
    grids = {}
    for e in entries:
        wl = spectrum_wavelengths(e["spectrum"])
        key = (len(wl), round(float(wl[0]), 4), round(float(wl[-1]), 4), round(float(np.sum(wl)), 2))
        grids.setdefault(key, wl)
    if len(grids) == 1:
        return next(iter(grids.values())), 1
    return None, len(grids)


def most_common_grid(entries) -> np.ndarray:
    from .repository import spectrum_wavelengths
    counts, grids = {}, {}
    for e in entries:
        wl = spectrum_wavelengths(e["spectrum"])
        key = (len(wl), round(float(wl[0]), 4), round(float(wl[-1]), 4))
        counts[key] = counts.get(key, 0) + 1
        grids[key] = wl
    return grids[max(counts, key=counts.get)]


def export_library(entries, path: str, fmt: str, grid=None, method="linear", library_name="library") -> dict:
    """Write library entries ({'material', 'spectrum'}) to CSV (one column per spectrum) or ENVI .sli.

    All spectra must share one grid; when they do not, give *grid* and they are resampled onto it (spectra
    that do not cover half of it are left out and listed)."""
    from .repository import spectrum_values, spectrum_wavelengths
    if not entries:
        raise ValueError("the library has no spectra")
    shared, n_grids = common_library_grid(entries)
    target = shared if grid is None else np.asarray(grid, float)
    if target is None:
        raise ValueError(f"the spectra are on {n_grids} different wavelength grids - choose a grid to resample to")
    names, rows, skipped, units = [], [], [], set()
    for e in entries:
        wl, y = spectrum_wavelengths(e["spectrum"]), spectrum_values(e["spectrum"])
        if shared is None or grid is not None:
            y, cov, why = resample_one(wl, y, target, method, None, 0.5)
            if y is None:
                skipped.append(f"{e['material']['name']}: {why}")
                continue
        names.append(e["material"]["name"])
        rows.append(np.asarray(y, float))
        if e["spectrum"].get("measurement_unit"):
            units.add(e["spectrum"]["measurement_unit"])
    if not rows:
        raise ValueError("no spectrum could be put on that grid")
    data = np.vstack(rows)
    fmt = fmt.lower()
    unit = "/".join(sorted(units)) or "unknown"
    if fmt == "csv":
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["wavelength_nm"] + names)
            for i, x in enumerate(target):
                w.writerow([f"{x:.4f}"] + ["" if not np.isfinite(v) else f"{v:.8g}" for v in data[:, i]])
        files = [path]
    elif fmt in ("sli", "envi"):
        from ..core.export import _write_sli
        base = os.path.splitext(path)[0]
        files = [_write_sli(base, names, target, data, f"{library_name} ({len(names)} spectra)", unit), base + ".hdr"]
    elif fmt == "json":
        doc = {"format": "hvva-library-1", "library": library_name, "unit": unit, "wavelength_units": "nm",
               "wavelengths": [float(v) for v in target],
               "spectra": [{"name": n, "values": [None if not np.isfinite(v) else float(v) for v in r]}
                           for n, r in zip(names, rows)]}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(doc, f)
        files = [path]
    else:
        raise ValueError("format must be csv, sli or json")
    return {"files": files, "spectra": len(names), "skipped": skipped, "bands": int(target.size),
            "resampled": shared is None or grid is not None}
