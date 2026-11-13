"""Zarr: reader for the DS3 standard cube layout + writer (the 'standardised' form)."""
from __future__ import annotations
import numpy as np
import zarr
from ..contract import Cube, CONTRACT_VERSION
from ..metadata import standardize_spectral_axis, MissingWavelengthsError, InvalidHeaderError

ARRAY_NAMES = ("data", "reflectance", "cube", "radiance")


def _find_array(g):
    for n in ARRAY_NAMES:
        if n in g:
            return n
    raise InvalidHeaderError(f"zarr group has none of the arrays {ARRAY_NAMES}")


def read_zarr(path, unit=None, wavelengths=None, **_):
    g = zarr.open_group(str(path), mode="r")
    name = _find_array(g)
    a = g[name]
    at = dict(g.attrs)
    axes = at.get("axes", "yxb")
    if sorted(axes) != sorted("yxb"):
        raise InvalidHeaderError(f"bad 'axes' attribute {axes!r}")
    ry, rx, rb = axes.index("y"), axes.index("x"), axes.index("b")
    shape = (a.shape[ry], a.shape[rx], a.shape[rb])
    wl = wavelengths if wavelengths is not None else at.get("wavelengths_nm", at.get("wavelengths"))
    if wl is None and "wavelengths" in g:
        wl = g["wavelengths"][:]
    if wl is None:
        raise MissingWavelengthsError("zarr store has no 'wavelengths_nm' attribute or 'wavelengths' array")
    if len(wl) != shape[2]:
        raise InvalidHeaderError(f"{len(wl)} wavelengths but {shape[2]} bands")
    bad = at.get("bad_band_mask")
    spec = standardize_spectral_axis(wl, at.get("wavelength_units", "nm"), at.get("fwhm_nm"),
                                     None if bad is None else np.array(bad, bool))
    nodata = at.get("nodata")

    def raw_loader(r0, r1, c0, c1):
        idx = [slice(None)] * 3
        idx[ry], idx[rx] = slice(r0, r1), slice(c0, c1)
        sub = np.asarray(a[tuple(idx)], dtype=np.float32)
        sub = np.transpose(sub, (ry, rx, rb))
        if nodata is not None:
            sub = sub.copy(); sub[sub == np.float32(nodata)] = np.nan
        return sub

    tr = at.get("transform")
    return Cube.make(raw_loader, shape, spec, unit or at.get("unit", "unknown"), nodata=nodata,
                     crs=at.get("crs"), transform=tuple(tr) if tr else None, sensor=at.get("sensor"),
                     processing_level=at.get("processing_level"), source_path=str(path),
                     source_format="zarr", metadata=dict(notes=[], contract_version=at.get("contract_version")))


def write_zarr(cube, path, chunks=(64, 64), block_mb=64):
    """Write any Cube to the standard DS3 Zarr layout (rows, cols, bands; nm; float32; NaN nodata)."""
    g = zarr.open_group(str(path), mode="w")
    r, c, b = cube.shape
    arr = g.create_array("data", shape=(r, c, b), chunks=(min(chunks[0], r), min(chunks[1], c), b),
                         dtype="float32", fill_value=float("nan"))
    step = max(1, int(block_mb * 1e6 / (c * b * 4)))
    for r0 in range(0, r, step):
        arr[r0:r0 + step, :, :] = cube.read(rows=(r0, min(r, r0 + step)))
    j = lambda x: None if x is None else np.asarray(x).astype(float).tolist()
    g.attrs.update(dict(
        contract_version=CONTRACT_VERSION, axes="yxb", wavelengths_nm=j(cube.wavelengths_nm),
        wavelength_units="nm", fwhm_nm=j(cube.fwhm_nm),
        bad_band_mask=np.asarray(cube.bad_band_mask).astype(bool).tolist(),
        unit=cube.unit, nodata=None, crs=cube.crs,
        transform=None if cube.transform is None else list(cube.transform),
        sensor=cube.sensor, processing_level=cube.processing_level,
        source_format=cube.source_format, source_path=cube.source_path))
    return str(path)
