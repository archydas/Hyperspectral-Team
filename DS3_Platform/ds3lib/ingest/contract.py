"""The ds3lib.ingest output contract (v1.0).

Axis order is ALWAYS [rows, cols, bands]; wavelengths are ALWAYS nm, ascending;
values are float32 physical values (scale/offset applied, nodata -> NaN);
bad_band_mask is True where a band should be excluded from analysis.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Optional
import numpy as np

CONTRACT_VERSION = "1.0"
UNITS = ("reflectance", "radiance", "dn", "unknown")
PURITY = ("ground_measured", "image_pixel", "synthetic")


@dataclass
class Spectrum:
    material: str
    wavelengths_nm: np.ndarray
    values: np.ndarray
    unit: str
    purity: str
    fwhm_nm: Optional[np.ndarray] = None
    bad_band_mask: Optional[np.ndarray] = None
    uncertainty: Optional[np.ndarray] = None
    category: Optional[str] = None
    id: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.purity not in PURITY:
            raise ValueError(f"purity must be one of {PURITY}")
        if self.unit not in UNITS:
            raise ValueError(f"unit must be one of {UNITS}")
        if len(self.wavelengths_nm) != len(self.values):
            raise ValueError("wavelengths_nm and values must have the same length")

    def to_dict(self):
        f = lambda a: None if a is None else np.asarray(a).tolist()
        return dict(id=self.id, material=self.material, category=self.category,
                    wavelengths_nm=f(self.wavelengths_nm), values=f(self.values),
                    unit=self.unit, purity=self.purity, fwhm_nm=f(self.fwhm_nm),
                    bad_band_mask=f(self.bad_band_mask), uncertainty=f(self.uncertainty),
                    metadata=self.metadata)


@dataclass
class ROIResult:
    spectra: np.ndarray            # [n_pixels, n_bands] float32
    pixel_coords: np.ndarray       # [n_pixels, 2] int (row, col)
    mean: np.ndarray               # [n_bands]
    std: np.ndarray                # [n_bands]
    wavelengths_nm: np.ndarray
    bad_band_mask: np.ndarray
    unit: str
    n_excluded_nodata: int = 0
    source: Optional[str] = None

    @property
    def n_pixels(self):
        return int(self.spectra.shape[0])

    def to_spectrum(self, material, purity="image_pixel", category=None, id=None):
        """Mean ROI spectrum as a library-ready Spectrum (std stored as uncertainty)."""
        return Spectrum(material=material, wavelengths_nm=self.wavelengths_nm,
                        values=self.mean, unit=self.unit, purity=purity,
                        bad_band_mask=self.bad_band_mask, uncertainty=self.std,
                        category=category, id=id,
                        metadata=dict(source=self.source, n_pixels=self.n_pixels))


@dataclass
class Cube:
    wavelengths_nm: np.ndarray
    shape: tuple                   # (rows, cols, bands)
    unit: str
    bad_band_mask: np.ndarray
    fwhm_nm: Optional[np.ndarray] = None
    nodata: Optional[float] = None
    crs: Optional[str] = None
    transform: Optional[tuple] = None    # affine (a,b,c,d,e,f): x=a*col+b*row+c, y=d*col+e*row+f
    sensor: Optional[str] = None
    processing_level: Optional[str] = None
    source_path: Optional[str] = None
    source_format: Optional[str] = None
    metadata: dict = field(default_factory=dict)
    qc: object = None
    _loader: Optional[Callable] = field(default=None, repr=False)

    @property
    def rows(self): return self.shape[0]
    @property
    def cols(self): return self.shape[1]
    @property
    def bands(self): return self.shape[2]

    def read(self, rows=None, cols=None, bands=None):
        """Lazy windowed read -> float32 [r, c, b]. rows/cols: slice or (start, stop)."""
        def norm(x, n):
            if x is None: return 0, n
            if isinstance(x, slice):
                s, e, _ = x.indices(n); return s, e
            s, e = x; return max(0, s), min(n, e)
        r0, r1 = norm(rows, self.rows)
        c0, c1 = norm(cols, self.cols)
        nb = self.bands if bands is None else len(np.arange(self.bands)[bands])
        if r1 <= r0 or c1 <= c0:
            return np.empty((0, 0, nb), np.float32)
        a = self._loader(r0, r1, c0, c1)
        return a if bands is None else a[..., bands]

    @property
    def data(self):
        """Full cube in memory. Use read() for big scenes."""
        return self.read()

    def xy_to_pixel(self, x, y):
        if self.transform is None:
            raise ValueError("cube has no georeferencing transform")
        from affine import Affine
        col, row = ~Affine(*self.transform) @ (x, y)
        return int(np.floor(row)), int(np.floor(col))

    def pixel_to_xy(self, row, col, center=True):
        if self.transform is None:
            raise ValueError("cube has no georeferencing transform")
        from affine import Affine
        off = 0.5 if center else 0.0
        return tuple(Affine(*self.transform) @ (col + off, row + off))

    def describe(self):
        wl = self.wavelengths_nm
        return (f"Cube {self.rows}x{self.cols}x{self.bands} | {self.source_format} | unit={self.unit} | "
                f"{wl[0]:.1f}-{wl[-1]:.1f} nm | bad bands={int(self.bad_band_mask.sum())} | "
                f"fwhm={'yes' if self.fwhm_nm is not None else 'no'} | "
                f"georef={'yes' if self.transform is not None else 'no'} | sensor={self.sensor}")

    @classmethod
    def make(cls, raw_loader, shape, spectral, unit, **kw):
        """Build a Cube from a reader's raw loader (file band order) + the dict returned
        by metadata.standardize_spectral_axis."""
        order = spectral["order"]
        if order is None:
            loader = raw_loader
        else:
            loader = lambda r0, r1, c0, c1: raw_loader(r0, r1, c0, c1)[..., order]
        kw.setdefault("metadata", {})
        kw["metadata"].setdefault("notes", []).extend(spectral["notes"])
        return cls(wavelengths_nm=spectral["wavelengths_nm"], shape=tuple(shape), unit=unit,
                   bad_band_mask=spectral["bad_band_mask"], fwhm_nm=spectral["fwhm_nm"],
                   _loader=loader, **kw)


def to_physical(raw, nodata=None, scale=1.0, offset=0.0):
    """raw array -> float32 physical values; nodata (compared BEFORE scaling) -> NaN."""
    raw = np.asarray(raw)
    mask = None
    if nodata is not None:
        mask = np.isnan(raw) if (isinstance(nodata, float) and np.isnan(nodata)) else (raw == nodata)
    out = raw.astype(np.float32)
    if scale != 1.0:
        out *= np.float32(scale)
    if offset != 0.0:
        out += np.float32(offset)
    if mask is not None:
        out[mask] = np.nan
    return out
