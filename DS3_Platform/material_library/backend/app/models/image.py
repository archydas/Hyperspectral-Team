from sqlalchemy import Column, String, Text, DateTime, Float, Integer
from sqlalchemy import Uuid
from sqlalchemy.sql import func
import uuid

from ..database import Base
from ..db_types import JSONType, FloatArray


class Image(Base):
    """A registered hyperspectral image. Only the description lives in the database; pixel data stays in the file
    (`path`) and is read on demand, so a multi-GB cube never has to be copied into PostgreSQL."""
    __tablename__ = "images"

    id = Column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    path = Column(Text, nullable=False)
    sha256 = Column(String(64), nullable=False, unique=True)
    fingerprint = Column(String(64))          # the shared PostgreSQL schema (Archy's `images`) requires one; we store the sha256
    format = Column(String(50))
    rows = Column(Integer, nullable=False)
    cols = Column(Integer, nullable=False)
    bands = Column(Integer, nullable=False)
    unit = Column(String(50))
    sensor = Column(String(100))
    wl_min_nm = Column(Float)
    wl_max_nm = Column(Float)
    wavelengths = Column(FloatArray)          # nm, ascending
    bad_band_mask = Column(JSONType)          # list[bool], True = bad band
    reader_kw = Column(JSONType, nullable=False, default=dict)   # options needed to re-open the file the same way
    image_metadata = Column("metadata", JSONType, nullable=False, default=dict)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
