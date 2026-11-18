from __future__ import annotations
from pathlib import Path
from .metadata import UnsupportedFormatError, water_absorption_mask
from .qc import run_qc, apply_bad_bands
from .readers.envi import read_envi
from .readers.geotiff import read_geotiff
from .readers.zarr_io import read_zarr
from .readers.hdf import read_hdf
from .readers.mat import read_mat


# Extensions the Image module opens (see detect_format / open_cube)
HDF_EXT = (".h5", ".he5", ".hdf5", ".hdf", ".h4", ".hdf4", ".nc", ".nc4", ".cdf", ".h5l")
ANY_EXT = (".npy", ".npz")                                                     # NumPy cubes -> general loader
GDAL_EXT = (".img", ".pix", ".vrt", ".jp2", ".j2k", ".ntf", ".nitf", ".lbl", ".fits", ".fit", ".fts",
            ".l1r", ".l1b", ".l1t", ".bil", ".bip", ".bsq", ".kea", ".rst", ".mpr", ".ecw", ".sid")
CUBE_EXTENSIONS = (".hdr", ".tif", ".tiff", ".mat", ".zarr") + HDF_EXT + ANY_EXT + GDAL_EXT


def _sniff(p):
    try:
        with open(p, "rb") as f:
            head = f.read(8)
    except OSError:
        return None
    if head[:4] in (b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+"):
        return "geotiff"
    if head[:4] == b"\x89HDF" or head[:4] == b"\x0e\x03\x13\x01" or head[:3] == b"CDF":
        return "hdf"
    if head[:6] == b"\x93NUMPY" or head[:2] == b"PK":
        return "any"
    if head[:6] == b"MATLAB":
        return "mat"
    return None


def detect_format(path):
    p = Path(path)
    if p.is_dir():
        if p.suffix.lower() == ".zarr" or (p / "zarr.json").exists() or (p / ".zgroup").exists():
            return "zarr"
        raise UnsupportedFormatError(f"{path} is a directory but not a Zarr store")
    ext = p.suffix.lower()
    if ext == ".hdr":
        return "envi"
    if ext in (".tif", ".tiff", ".gtif", ".btf", ".tf8"):
        return "geotiff"
    if ext == ".mat":
        return "mat"
    if ext in HDF_EXT:
        return "hdf"
    if ext in ANY_EXT:
        return "any"
    if Path(str(p) + ".hdr").is_file() or p.with_suffix(".hdr").is_file():
        return "envi"
    if ext in GDAL_EXT:
        return "gdal"
    sniffed = _sniff(p) if p.is_file() else None
    if sniffed:
        return sniffed
    raise UnsupportedFormatError(f"cannot determine format of {p.name} - supported: ENVI (.hdr + data), "
                                 f"GeoTIFF / TIFF stacks, MATLAB .mat, HDF5 / HDF-EOS / NetCDF, Zarr, NumPy .npy / .npz, "
                                 f"and GDAL rasters ({', '.join(GDAL_EXT)})")


from .readers.anyformat import read_any, read_gdal
_READERS = dict(envi=read_envi, geotiff=read_geotiff, zarr=read_zarr, hdf=read_hdf, mat=read_mat, any=read_any, gdal=read_gdal)
# when the first reader cannot read a file, these are tried next (wavelength errors are never retried: the user is asked)
_FALLBACK = dict(hdf=("any", "gdal"), mat=("any",), gdal=("any",), geotiff=("gdal",))


def open_cube(path, *, qc=True, unit=None, wavelengths=None, reflectance_divisor=None, **kw):
    """Open any supported hyperspectral file as a standard Cube (lazy).

    qc=True runs streaming QC and merges source + water-absorption + automatic flags into
    cube.bad_band_mask. qc=False merges only source + water-absorption flags (no data pass).
    Extra keyword args go to the reader (e.g. dataset=, wavelengths_dataset= for HDF5)."""
    from .metadata import MissingWavelengthsError
    fmt = detect_format(path)
    try:
        cube = _READERS[fmt](path, unit=unit, wavelengths=wavelengths, reflectance_divisor=reflectance_divisor, **kw)
    except MissingWavelengthsError:
        if fmt != "hdf" or not kw.get("wavelength_range"):
            raise
        cube = read_any(path, unit=unit, wavelengths=wavelengths, reflectance_divisor=reflectance_divisor, **kw)
    except Exception as first:                                   # noqa: BLE001
        cube = None
        for alt in _FALLBACK.get(fmt, ()):
            try:
                cube = _READERS[alt](path, unit=unit, wavelengths=wavelengths, reflectance_divisor=reflectance_divisor, **kw)
                cube.metadata.setdefault("notes", []).append(f"{fmt} reader failed ({type(first).__name__}); read by the {alt} reader")
                break
            except MissingWavelengthsError:
                raise
            except Exception:                                    # noqa: BLE001
                continue
        if cube is None:
            raise first
    if qc:
        apply_bad_bands(cube, run_qc(cube))
    else:
        cube.bad_band_mask = cube.bad_band_mask | water_absorption_mask(cube.wavelengths_nm)
    return cube
