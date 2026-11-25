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
Migration 005 (classification): ``ground_truths`` + ``ground_truth_points`` (point / label-map ground truth placed
by lat/lon) and ``classification_models`` (trained Random Forest / SVM models with their settings, file in the row).
Migration 002 (ROIs): table ``rois`` (one row per region of interest drawn on an image, class by class),
``spectra.roi_id`` / ``spectra.roi_class`` and the link table ``spectrum_rois`` (which ROIs a spectrum was computed
from - a pooled class mean comes from many ROIs, possibly in several images).

The JSON ``metadata`` column is exposed as ``<table>_metadata`` because ``metadata`` is reserved by SQLAlchemy.
"""
from __future__ import annotations

import uuid

from sqlalchemy import (JSON, BigInteger, Column, DateTime, Float, ForeignKey, Integer, LargeBinary, String, Text,
                        UniqueConstraint, Uuid)
from sqlalchemy.orm import deferred
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
    purity = Column(String(20))                    # ground_measured | image_pixel | synthetic | NULL (003)
    wavelengths = Column(FloatArray)               # explicit grid for irregular spectra; NULL = start + i*step (004)
    created_at = _created()


class Image(Base):
    """A hyperspectral image the library knows (migration 004).  Pixels stay in the file."""
    __tablename__ = "images"
    id = _id()
    name = Column(String(500), nullable=False)
    path = Column(Text)
    fingerprint = Column(String(64), nullable=False, unique=True)   # size + mtime + head/tail hash of the file
    sha256 = Column(String(64))
    format = Column(String(50))
    rows = Column(Integer, nullable=False)
    cols = Column(Integer, nullable=False)
    bands = Column(Integer, nullable=False)
    unit = Column(String(20))
    sensor = Column(String(100))
    wl_min_nm = Column(Float)
    wl_max_nm = Column(Float)
    wavelengths = Column(FloatArray)
    bad_band_mask = Column(JSONType)
    crs = Column(Text)
    transform = Column(JSONType)
    image_metadata = _meta()
    created_at = _created()
    updated_at = _updated()


class Roi(Base):
    """A region of interest on an image (migration 002).  Geometry in pixel coordinates of that image:
    rect {"rect": [r0, c0, r1, c1]} (end-exclusive) or polygon {"vertices": [[x=col, y=row], ...]};
    geometry_map holds the same outline in map coordinates when the image is georeferenced."""
    __tablename__ = "rois"
    __table_args__ = (UniqueConstraint("library_id", "roi_uid", name="uq_rois_library_uid"),)
    id = _id()
    library_id = _fk("libraries.id")
    dataset_id = Column(Uuid(as_uuid=True), ForeignKey("datasets.id", ondelete="SET NULL"))
    image_id = Column(Uuid(as_uuid=True), ForeignKey("images.id", ondelete="SET NULL"))     # migration 004
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


class GroundTruth(Base):
    """Ground truth for classification (migration 005): point tables (lat/lon/value per layer, e.g. one abundance
    sheet per material) or a label map, with the label rule last used to turn values into classes."""
    __tablename__ = "ground_truths"
    id = _id()
    library_id = Column(Uuid(as_uuid=True), ForeignKey("libraries.id", ondelete="SET NULL"))
    image_id = Column(Uuid(as_uuid=True), ForeignKey("images.id", ondelete="SET NULL"))
    name = Column(String(255), nullable=False)
    kind = Column(String(20), nullable=False, default="points")          # points | label_map
    layers = Column(JSONType, nullable=False, default=list)              # ["Cadmium", "Lead", ...]
    value_meaning = Column(String(100))                                  # e.g. "abundance (fraction of pixel)"
    crs = Column(Text)
    image_rows = Column(Integer)
    image_cols = Column(Integer)
    source_files = Column(JSONType)
    rule = Column(JSONType)                                              # gt_points label rule
    class_names = Column(JSONType)                                       # {"1": "Cadmium", ...}
    n_points = Column(Integer)
    label_map = deferred(Column(LargeBinary))                            # npz bytes (kind = label_map)
    gt_metadata = _meta()
    created_at = _created()
    updated_at = _updated()


class GroundTruthPoint(Base):
    __tablename__ = "ground_truth_points"
    id = Column(BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True)
    ground_truth_id = _fk("ground_truths.id")
    layer = Column(String(255), nullable=False)
    lat = Column(Float)
    lon = Column(Float)
    pixel_row = Column(Integer)
    pixel_col = Column(Integer)
    value = Column(Float)
    class_name = Column(String(255))


class ClassificationModel(Base):
    """A trained classical model (migration 005). The fitted estimator + feature transform are stored as joblib
    bytes in ``model_file`` (sha256 checked before loading); everything needed to reuse it is in the other columns."""
    __tablename__ = "classification_models"
    id = _id()
    library_id = Column(Uuid(as_uuid=True), ForeignKey("libraries.id", ondelete="SET NULL"))
    image_id = Column(Uuid(as_uuid=True), ForeignKey("images.id", ondelete="SET NULL"))
    ground_truth_id = Column(Uuid(as_uuid=True), ForeignKey("ground_truths.id", ondelete="SET NULL"))
    name = Column(String(255), nullable=False)
    model_type = Column(String(50), nullable=False)                      # Random Forest | SVM (RBF) | ...
    family = Column(String(20), nullable=False, default="classical")
    class_names = Column(JSONType, nullable=False)
    n_bands = Column(Integer, nullable=False)
    wavelengths = Column(FloatArray)
    unit = Column(String(20))
    feature_kind = Column(String(20))
    window = Column(Integer)
    components = Column(Integer)
    label_source = Column(String(20))                                    # rois | gt | both
    rule = Column(JSONType)
    roi_uids = Column(JSONType)
    metrics = Column(JSONType)
    settings = Column(JSONType)
    config = Column(JSONType)
    model_file = deferred(Column(LargeBinary, nullable=False))
    file_format = Column(String(20), nullable=False, default="joblib")
    size_bytes = Column(BigInteger)
    sha256 = Column(String(64))
    model_metadata = _meta()
    created_at = _created()
