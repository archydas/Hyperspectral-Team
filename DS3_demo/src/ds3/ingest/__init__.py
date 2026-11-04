"""ds3.ingest - open-framework ingestion of hyperspectral data into one standard form."""
from .contract import Cube, Spectrum, ROIResult, CONTRACT_VERSION
from .api import open_cube, detect_format
from .roi import extract_roi
from .qc import run_qc, QCReport
from .readers.zarr_io import write_zarr
from .metadata import (IngestError, MissingWavelengthsError, InvalidHeaderError, UnsupportedFormatError)
__version__ = "0.1.0"
