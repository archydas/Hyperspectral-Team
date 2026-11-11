from sqlalchemy import Column, String, Text, DateTime, Float, ForeignKey
from sqlalchemy import Uuid
from sqlalchemy.sql import func
import uuid

from ..database import Base
from ..db_types import JSONType, FloatArray


class Spectrum(Base):
    __tablename__ = "spectra"

    id = Column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    sample_id = Column(Uuid(as_uuid=True), ForeignKey("samples.id", ondelete="CASCADE"), nullable=False)
    wavelength_start = Column(Float, nullable=False)
    wavelength_end = Column(Float, nullable=False)
    wavelength_step = Column(Float)          # only set when the grid is evenly spaced
    measurement_unit = Column(String(100))   # unit of the VALUES: reflectance | radiance | ...
    values = Column(FloatArray, nullable=False)
    # Explicit wavelengths (nm), one per value. Set when the grid is NOT evenly spaced (AVIRIS-NG, EnMAP, USGS, ...).
    # NULL means: wavelength_start + i * wavelength_step.
    wavelengths = Column(FloatArray)
    # How the spectrum was measured: lab | field | airborne | satellite (NULL = not tagged)
    acquisition_type = Column(String(20))
    spectrum_metadata = Column("metadata", JSONType, nullable=False, default=dict)

    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
