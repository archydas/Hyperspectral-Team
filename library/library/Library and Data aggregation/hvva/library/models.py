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
Migration 002 (ROIs): table ``rois`` (one row per region of interest drawn on an image, class by class),
``spectra.roi_id`` / ``spectra.roi_class`` and the link table ``spectrum_rois`` (which ROIs a spectrum was computed
from - a pooled class mean comes from many ROIs, possibly in several images).

The JSON ``metadata`` column is exposed as ``<table>_metadata`` because ``metadata`` is reserved by SQLAlchemy.
"""
from __future__ import annotations

import uuid

from sqlalchemy import (JSON, BigInteger, Column, DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint,
                        Uuid)
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
    roi_id = Column(Uuid(as_uuid=True), ForeignKey("rois.id", ondelete="SET NULL"))   # single source ROI (002)
    roi_class = Column(String(255))                # ROI class the spectrum represents (002)
    created_at = _created()


class Roi(Base):
    """A region of interest on an image (migration 002).  Geometry in pixel coordinates of that image:
    rect {"rect": [r0, c0, r1, c1]} (end-exclusive) or polygon {"vertices": [[x=col, y=row], ...]};
    geometry_map holds the same outline in map coordinates when the image is georeferenced."""
    __tablename__ = "rois"
    __table_args__ = (UniqueConstraint("library_id", "roi_uid", name="uq_rois_library_uid"),)
    id = _id()
    library_id = _fk("libraries.id")
    dataset_id = Column(Uuid(as_uuid=True), ForeignKey("datasets.id", ondelete="SET NULL"))
    roi_uid = Column(String(64), nullable=False)           # id of the ROI in HyperVista (scene file / ROI JSON)
    roi_name = Column(String(255))
    class_name = Column(String(255), nullable=False)
    scene_name = Column(String(500))
    scene_path = Column(Text)
    image_rows = Column(Integer)
    image_cols = Column(Integer)
    geometry_type = Column(String(20), nullable=False)     # rect | polygon
    geometry = Column(JSONType, nullable=False)
    geometry_map = Column(JSONType)
    crs = Column(Text)
    bbox_row_min = Column(Integer)
    bbox_row_max = Column(Integer)
    bbox_col_min = Column(Integer)
    bbox_col_max = Column(Integer)
    n_pixels = Column(Integer)
    roi_metadata = _meta()
    created_at = _created()
    updated_at = _updated()


class SpectrumRoi(Base):
    """spectrum <-> ROIs it was computed from (n-n; migration 002)."""
    __tablename__ = "spectrum_rois"
    spectrum_id = Column(Uuid(as_uuid=True), ForeignKey("spectra.id", ondelete="CASCADE"), primary_key=True)
    roi_id = Column(Uuid(as_uuid=True), ForeignKey("rois.id", ondelete="CASCADE"), primary_key=True)


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
