"""SQLAlchemy models of the material spectral library.

Taken from new_hyperspectral/material_library/backend/models/*.py (library, dataset, material, sample,
spectrum, datafile) and merged into one module.  Tables and columns are unchanged and match
database/schema.sql exactly:

    libraries 1-n datasets
    libraries 1-n materials 1-n samples (each sample also belongs to a dataset) 1-n spectra
    datasets  1-n data_files (provenance of imported files)

The only change: column types are declared portably (UUID / JSONB / DOUBLE PRECISION[] on PostgreSQL,
generic UUID / JSON on SQLite) so the same models also run on a local SQLite file for offline use and tests.
HyperVista additions (database/migrations/001_category_tree_acquisition.sql, applied automatically on connect):
``spectra.acquisition_type`` and the convention that ``materials.category`` holds a '/'-separated hierarchy path
(Vegetation/Tree/Tsuga/Canadensis).

The JSON ``metadata`` column is exposed as ``<table>_metadata`` because ``metadata`` is reserved by SQLAlchemy.
"""
from __future__ import annotations

import uuid

from sqlalchemy import JSON, BigInteger, Column, DateTime, Float, ForeignKey, Integer, String, Text, Uuid
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.sql import func

from .database import Base

JSONType = JSON().with_variant(JSONB(), "postgresql")
FloatArray = JSON().with_variant(ARRAY(Float), "postgresql")


def _id():
    return Column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)


def _fk(target):
    return Column(Uuid(as_uuid=True), ForeignKey(target, ondelete="CASCADE"), nullable=False)


def _meta():
    return Column("metadata", JSONType, nullable=False, default=dict)


def _created():
    return Column(DateTime(timezone=True), nullable=False, server_default=func.now())


def _updated():
    return Column(DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now())


class Library(Base):
    __tablename__ = "libraries"
    id = _id()
    name = Column(String(255), nullable=False)
    slug = Column(String(255), nullable=False, unique=True)
    description = Column(Text)
    library_metadata = _meta()
    created_at = _created()
    updated_at = _updated()


class Dataset(Base):
    __tablename__ = "datasets"
    id = _id()
    library_id = _fk("libraries.id")
    name = Column(String(255), nullable=False)
    slug = Column(String(255), nullable=False)
    description = Column(Text)
    source_name = Column(String(500))
    source_version = Column(String(100))
    dataset_metadata = _meta()
    created_at = _created()
    updated_at = _updated()


class Material(Base):
    __tablename__ = "materials"
    id = _id()
    library_id = _fk("libraries.id")
    name = Column(String(500), nullable=False)
    category = Column(String(255))
    description = Column(Text)
    material_metadata = _meta()
    created_at = _created()
    updated_at = _updated()


class Sample(Base):
    __tablename__ = "samples"
    id = _id()
    material_id = _fk("materials.id")
    dataset_id = _fk("datasets.id")
    name = Column(String(500))
    sample_number = Column(Integer)
    sample_metadata = _meta()
    created_at = _created()
    updated_at = _updated()


class Spectrum(Base):
    __tablename__ = "spectra"
    id = _id()
    sample_id = _fk("samples.id")
    wavelength_start = Column(Float, nullable=False)
    wavelength_end = Column(Float, nullable=False)
    wavelength_step = Column(Float)
    measurement_unit = Column(String(100))
    values = Column(FloatArray, nullable=False)
    spectrum_metadata = _meta()
    acquisition_type = Column(String(20))          # lab | field | airborne | satellite | NULL (migration 001)
    created_at = _created()


class DataFile(Base):
    __tablename__ = "data_files"
    id = _id()
    dataset_id = _fk("datasets.id")
    filename = Column(String(500), nullable=False)
    file_type = Column(String(100))
    file_size = Column(BigInteger)
    checksum = Column(String(128))
    storage_path = Column(Text)
    file_metadata = _meta()
    uploaded_at = _created()
