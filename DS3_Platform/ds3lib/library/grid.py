"""Target wavelength grid and resampling (what ENVI's 'Input Spectral Wavelength From' step defines)."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from ..ingest.metadata import to_nm, IngestError, MissingWavelengthsError, InvalidHeaderError

FWHM_TO_SIGMA = 1 / (2 * np.sqrt(2 * np.log(2)))


@dataclass
class Grid:
    wavelengths_nm: np.ndarray
    fwhm_nm: np.ndarray | None = None
    origin: str = "unknown"        # cube:<path> | ascii:<path> | first_spectrum | explicit

    def __post_init__(self):
        self.wavelengths_nm = np.asarray(self.wavelengths_nm, float)
        if self.wavelengths_nm.ndim != 1 or len(self.wavelengths_nm) < 2:
            raise InvalidHeaderError("grid needs at least 2 wavelengths")
        if np.any(np.diff(self.wavelengths_nm) <= 0):
            raise InvalidHeaderError("grid wavelengths must be strictly ascending")
        if self.fwhm_nm is not None:
            self.fwhm_nm = np.asarray(self.fwhm_nm, float)

    def __len__(self): return len(self.wavelengths_nm)

    @classmethod
    def from_cube(cls, path, **kw):
        from ..ingest import open_cube
        c = open_cube(path, qc=False, **kw)
        return cls(c.wavelengths_nm, c.fwhm_nm, f"cube:{Path(path).name}")

    @classmethod
    def from_ascii(cls, path, units=None):
        """One column = wavelengths; a 2nd column, if present, = FWHM. '#', ';' and text header lines are ignored."""
        rows = []
        for line in Path(path).read_text(errors="replace").splitlines():
            parts = line.replace(",", " ").split()
            try: rows.append([float(x) for x in parts[:2]])
            except ValueError: continue
        if len(rows) < 2: raise MissingWavelengthsError(f"{path}: no numeric wavelength rows")
        width = min(len(r) for r in rows)
        a = np.array([r[:width] for r in rows])
        wl, f, _ = to_nm(a[:, 0], units)
        fw = a[:, 1] * f if width > 1 else None
        o = np.argsort(wl)
        return cls(wl[o], None if fw is None else fw[o], f"ascii:{Path(path).name}")

    @classmethod
    def from_spectrum(cls, spectrum):
        return cls(spectrum.wavelengths_nm, spectrum.fwhm_nm, "first_spectrum")


def resample(wl, values, grid: Grid, method="linear", max_gap_nm=None):
    """Resample one spectrum onto `grid`. Returns (values_on_grid float64, coverage 0-1).

    NaN source samples are ignored. Target bands are NaN (never extrapolated, never guessed) when
      linear   : outside the source range, or inside a gap wider than max_gap_nm (default 3x median spacing)
      gaussian : the band's FWHM window is not fully inside the source range / has no source samples
    """
    wl = np.asarray(wl, float); v = np.asarray(values, float)
    ok = np.isfinite(wl) & np.isfinite(v)
    wl, v = wl[ok], v[ok]
    if len(wl) < 2:
        return np.full(len(grid), np.nan), 0.0
    o = np.argsort(wl); wl, v = wl[o], v[o]
    g = grid.wavelengths_nm
    if method == "linear":
        out = np.interp(g, wl, v, left=np.nan, right=np.nan)
        gap = max_gap_nm if max_gap_nm is not None else 3.0 * float(np.median(np.diff(wl)))
        j = np.clip(np.searchsorted(wl, g), 1, len(wl) - 1)
        wide = (wl[j] - wl[j - 1]) > gap
        exact = np.isin(g, wl)
        out[wide & ~exact] = np.nan
    elif method == "gaussian":
        fw = grid.fwhm_nm
        if fw is None:
            fw = np.gradient(g)                    # assume FWHM ~ band spacing when the grid has none
        out = np.full(len(g), np.nan)
        for i, (c, f) in enumerate(zip(g, fw)):
            if c - f / 2 < wl[0] or c + f / 2 > wl[-1]:
                continue
            sig = f * FWHM_TO_SIGMA
            m = np.abs(wl - c) <= 3 * sig
            if not m.any():
                continue
            w = np.exp(-0.5 * ((wl[m] - c) / sig) ** 2)
            out[i] = float((w * v[m]).sum() / w.sum())
    else:
        raise ValueError("method must be 'linear' or 'gaussian'")
    return out, float(np.isfinite(out).mean())
