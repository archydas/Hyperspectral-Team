"""Universal spectral-data readers for the library.

Every supported file family is a *reader*: it says how sure it is that it can read a path (sniff) and turns the path
into SpectrumRecords (read).  normalize() then makes every record look the same no matter where it came from
(wavelengths in nm ascending, reflectance 0-1, bad values NaN, category / acquisition filled in), and
hvva.library.ingest writes them to the database.  Nothing here imports Qt or touches the database.

    from hvva.library import formats
    reader, how_sure, what = formats.detect("D:/data/leaves.sed")
    records, info = formats.read_path("D:/data/leaves.sed", formats.ReadOptions(acquisition="field"))

Readers (see formats/text.py, binary.py, collections.py):

    table         CSV / TSV / TXT / DAT - spectra in columns, in rows (+ metadata columns) or long format; any
                  delimiter, decimal comma, comment lines, "key: value" header block (ECOSTRESS, generic two-column)
    excel         .xlsx / .xlsm (openpyxl) - every sheet with a numeric table, same layouts as above
    json          HyperVista signature JSON, lists of {name, wavelengths, values, ...}, {wavelengths, spectra{...}}
    parquet       generic tables (wide / long / list columns) - OpenSpecLib files have their own reader
    envi_sli      ENVI spectral library (.sli/.hdr): any data type, byte order, header offset, scale factor, µm/nm
    image         hyperspectral cubes (ENVI image, GeoTIFF, .mat/.h5/.nc/.npy/.npz/zarr via hvva.core.loaders):
                  class means (+/- std) when the file has a label map, else image mean and/or sampled pixels
    arrays        .mat / .h5 / .npz / .npy holding a spectra matrix (N x bands) + wavelength vector (+ names)
    asd, sig, sed field spectrometers: ASD FieldSpec (.asd, v1-v8, reflectance or DN with white reference),
                  Spectra Vista SVC (.sig), Spectral Evolution (.sed)
    usgs          USGS Spectral Library v7 ASCII release (zip or folder)
    openspeclib   OpenSpecLib release Parquet (USGS / ECOSTRESS / EcoSIS ...)
    folder / zip  any folder or zip of the above; subfolder names become category levels
"""
from .base import (ReadOptions, SpectrumRecord, ReadInfo, Reader, REGISTRY, register, detect, read_path,
                   normalize, infer_acquisition, readers_info, file_dialog_filter, SUPPORTED_EXTENSIONS)
from . import text, binary, refcollections  # noqa: F401  (register the readers)

__all__ = ["ReadOptions", "SpectrumRecord", "ReadInfo", "Reader", "REGISTRY", "register", "detect", "read_path",
           "normalize", "infer_acquisition", "readers_info", "file_dialog_filter", "SUPPORTED_EXTENSIONS"]
