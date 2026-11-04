from __future__ import annotations
from pathlib import Path
from .metadata import UnsupportedFormatError, water_absorption_mask
from .qc import run_qc, apply_bad_bands
from .readers.envi import read_envi
from .readers.geotiff import read_geotiff
from .readers.zarr_io import read_zarr
from .readers.hdf import read_hdf
from .readers.mat import read_mat


def detect_format(path):
    p = Path(path)
    if p.is_dir():
        if p.suffix.lower() == ".zarr" or (p / "zarr.json").exists() or (p / ".zgroup").exists():
            return "zarr"
        raise UnsupportedFormatError(f"{path} is a directory but not a Zarr store")
    ext = p.suffix.lower()
    if ext == ".hdr":
        return "envi"
    if ext in (".tif", ".tiff"):
        return "geotiff"
    if ext == ".mat":
        return "mat"
    if ext in (".h5", ".he5", ".hdf5", ".hdf", ".nc", ".nc4", ".h5l"):
        return "hdf"
    if Path(str(p) + ".hdr").is_file() or p.with_suffix(".hdr").is_file():
        return "envi"
    raise UnsupportedFormatError(f"cannot determine format of {path}")


_READERS = dict(envi=read_envi, geotiff=read_geotiff, zarr=read_zarr, hdf=read_hdf, mat=read_mat)


def open_cube(path, *, qc=True, unit=None, wavelengths=None, reflectance_divisor=None, **kw):
    """Open any supported hyperspectral file as a standard Cube (lazy).

    qc=True runs streaming QC and merges source + water-absorption + automatic flags into
    cube.bad_band_mask. qc=False merges only source + water-absorption flags (no data pass).
    Extra keyword args go to the reader (e.g. dataset=, wavelengths_dataset= for HDF5)."""
    fmt = detect_format(path)
    cube = _READERS[fmt](path, unit=unit, wavelengths=wavelengths, reflectance_divisor=reflectance_divisor, **kw)
    if qc:
        apply_bad_bands(cube, run_qc(cube))
    else:
        cube.bad_band_mask = cube.bad_band_mask | water_absorption_mask(cube.wavelengths_nm)
    return cube
