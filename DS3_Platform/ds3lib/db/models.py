"""Database schema. One schema for both PostgreSQL (production) and SQLite (zero-setup dev).

sources    : where data came from (satellite / drone / field / library) - provenance
cubes      : registered hyperspectral images (pixels stay in the file; metadata + QC here)
libraries  -> datasets -> materials -> samples -> spectra : point spectra
"""
from __future__ import annotations
from datetime import datetime, timezone
from sqlalchemy import (String, Integer, Float, Boolean, Text, DateTime, ForeignKey, JSON,
                        UniqueConstraint, Index)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

FloatArray = ARRAY(Float).with_variant(JSON(), "sqlite")
BoolArray = ARRAY(Boolean).with_variant(JSON(), "sqlite")
Json = JSON().with_variant(JSONB(), "postgresql")

def _now(): return datetime.now(timezone.utc)

class Base(DeclarativeBase): pass

class Source(Base):
    __tablename__ = "sources"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(200), unique=True)
    kind: Mapped[str] = mapped_column(String(30))       # satellite|airborne|drone|field|library
    url: Mapped[str | None] = mapped_column(String(500))
    license: Mapped[str | None] = mapped_column(String(200))
    notes: Mapped[str | None] = mapped_column(Text)

class CubeRecord(Base):
    __tablename__ = "cubes"
    id: Mapped[int] = mapped_column(primary_key=True)
    source_id: Mapped[int | None] = mapped_column(ForeignKey("sources.id"))
    name: Mapped[str] = mapped_column(String(300))
    path: Mapped[str] = mapped_column(String(1000))
    sha256: Mapped[str] = mapped_column(String(64), unique=True)
    format: Mapped[str] = mapped_column(String(30))
    rows: Mapped[int] = mapped_column(Integer)
    cols: Mapped[int] = mapped_column(Integer)
    bands: Mapped[int] = mapped_column(Integer)
    unit: Mapped[str] = mapped_column(String(20))
    sensor: Mapped[str | None] = mapped_column(String(100))
    processing_level: Mapped[str | None] = mapped_column(String(50))
    crs: Mapped[str | None] = mapped_column(String(100))
    transform: Mapped[list | None] = mapped_column(JSON)
    wl_min_nm: Mapped[float] = mapped_column(Float)
    wl_max_nm: Mapped[float] = mapped_column(Float)
    wavelengths_nm: Mapped[list] = mapped_column(FloatArray)
    fwhm_nm: Mapped[list | None] = mapped_column(FloatArray)
    bad_band_mask: Mapped[list] = mapped_column(BoolArray)
    qc: Mapped[dict | None] = mapped_column(Json)
    meta: Mapped[dict | None] = mapped_column(Json)
    contract_version: Mapped[str] = mapped_column(String(10), default="1.0")
    ingested_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    source: Mapped[Source | None] = relationship()

class Library(Base):
    __tablename__ = "libraries"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(200), unique=True)
    source_id: Mapped[int | None] = mapped_column(ForeignKey("sources.id"))
    description: Mapped[str | None] = mapped_column(Text)
    datasets: Mapped[list["Dataset"]] = relationship(back_populates="library", cascade="all, delete-orphan")

class Dataset(Base):
    __tablename__ = "datasets"
    id: Mapped[int] = mapped_column(primary_key=True)
    library_id: Mapped[int] = mapped_column(ForeignKey("libraries.id"), index=True)
    name: Mapped[str] = mapped_column(String(300))
    sha256: Mapped[str] = mapped_column(String(64), unique=True)
    format: Mapped[str] = mapped_column(String(30))
    ingested_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    library: Mapped[Library] = relationship(back_populates="datasets")
    samples: Mapped[list["Sample"]] = relationship(back_populates="dataset", cascade="all, delete-orphan")

class Material(Base):
    __tablename__ = "materials"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(300), unique=True)
    category: Mapped[str | None] = mapped_column(String(100), index=True)

class Sample(Base):
    __tablename__ = "samples"
    id: Mapped[int] = mapped_column(primary_key=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("datasets.id"), index=True)
    material_id: Mapped[int] = mapped_column(ForeignKey("materials.id"), index=True)
    label: Mapped[str] = mapped_column(String(300))
    meta: Mapped[dict | None] = mapped_column(Json)
    dataset: Mapped[Dataset] = relationship(back_populates="samples")
    material: Mapped[Material] = relationship()
    spectra: Mapped[list["SpectrumRecord"]] = relationship(back_populates="sample", cascade="all, delete-orphan")

class SpectrumRecord(Base):
    __tablename__ = "spectra"
    id: Mapped[int] = mapped_column(primary_key=True)
    sample_id: Mapped[int] = mapped_column(ForeignKey("samples.id"), index=True)
    unit: Mapped[str] = mapped_column(String(20))
    purity: Mapped[str] = mapped_column(String(20))
    wl_min_nm: Mapped[float] = mapped_column(Float)
    wl_max_nm: Mapped[float] = mapped_column(Float)
    n_bands: Mapped[int] = mapped_column(Integer)
    wavelengths_nm: Mapped[list] = mapped_column(FloatArray)
    values: Mapped[list] = mapped_column(FloatArray)     # NaN stored as NULL (JSON null)
    bad_band_mask: Mapped[list | None] = mapped_column(BoolArray)
    meta: Mapped[dict | None] = mapped_column(Json)
    sample: Mapped[Sample] = relationship(back_populates="spectra")
    __table_args__ = (Index("ix_spectra_unit_purity", "unit", "purity"), Index("ix_spectra_wl", "wl_min_nm", "wl_max_nm"))
