-- ============================================================
-- Material Spectral Library Database
-- PostgreSQL schema
-- ============================================================

-- Required extensions
CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE EXTENSION IF NOT EXISTS pg_trgm;


-- ============================================================
-- 1. LIBRARIES
-- ============================================================
-- A library is the top-level container.
-- Example:
--   Spectral Library
--   Chemical Library
--   Image Library
--   etc.

CREATE TABLE libraries (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    name VARCHAR(255) NOT NULL,
    slug VARCHAR(255) NOT NULL UNIQUE,

    description TEXT,

    -- Flexible library-specific information
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);


-- ============================================================
-- 2. DATASETS
-- ============================================================
-- A library can contain multiple datasets.
--
-- Example:
-- Spectral Library
--   ├── Material Spectra 2026
--   ├── Outdoor Materials
--   └── Building Materials

CREATE TABLE datasets (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    library_id UUID NOT NULL
        REFERENCES libraries(id)
        ON DELETE CASCADE,

    name VARCHAR(255) NOT NULL,
    slug VARCHAR(255) NOT NULL,

    description TEXT,

    source_name VARCHAR(500),
    source_version VARCHAR(100),

    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    UNIQUE(library_id, slug)
);


-- ============================================================
-- 2a. REGISTERED IMAGES (HyperVista, migration 004)
-- ============================================================
-- One row per hyperspectral image the library knows. The pixels stay in
-- the file; the row keeps where it is and what it is.

CREATE TABLE images (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(500) NOT NULL,
    path TEXT,
    fingerprint VARCHAR(64) NOT NULL UNIQUE,
    sha256 VARCHAR(64),
    format VARCHAR(50),
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    bands INTEGER NOT NULL,
    unit VARCHAR(20),
    sensor VARCHAR(100),
    wl_min_nm DOUBLE PRECISION,
    wl_max_nm DOUBLE PRECISION,
    wavelengths DOUBLE PRECISION[],
    bad_band_mask JSONB,
    crs TEXT,
    transform JSONB,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ============================================================
-- 2b. REGIONS OF INTEREST (HyperVista, migration 002)
-- ============================================================
-- A region drawn on an image (rectangle or polygon) labelled with a class.
-- Many ROIs per class, many classes per image, many images per library.
--
-- Example:
-- Field Campaign Library
--   ├── vegetation : ROI 1 (scene_A, rect), ROI 3 (scene_B, polygon)
--   └── soil       : ROI 2 (scene_A, polygon)

CREATE TABLE rois (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    library_id UUID NOT NULL
        REFERENCES libraries(id)
        ON DELETE CASCADE,

    dataset_id UUID
        REFERENCES datasets(id)
        ON DELETE SET NULL,

    -- the registered image the ROI was drawn on (migration 004)
    image_id UUID
        REFERENCES images(id)
        ON DELETE SET NULL,

    -- id of the ROI inside HyperVista; saving the same ROI again updates this row
    roi_uid VARCHAR(64) NOT NULL,
    roi_name VARCHAR(255),
    class_name VARCHAR(255) NOT NULL,

    scene_name VARCHAR(500),
    scene_path TEXT,
    image_rows INTEGER,
    image_cols INTEGER,

    -- rect {"rect": [r0, c0, r1, c1]} (end-exclusive) | polygon {"vertices": [[col, row], ...]}
    geometry_type VARCHAR(20) NOT NULL
        CONSTRAINT rois_geometry_type_check CHECK (geometry_type IN ('rect', 'polygon', 'pixels')),
    geometry JSONB NOT NULL,

    geometry_map JSONB,
    crs TEXT,

    bbox_row_min INTEGER,
    bbox_row_max INTEGER,
    bbox_col_min INTEGER,
    bbox_col_max INTEGER,
    n_pixels INTEGER,

    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT uq_rois_library_uid UNIQUE (library_id, roi_uid)
);


-- ============================================================
-- 3. MATERIALS
-- ============================================================
-- Represents the actual material.
--
-- Examples:
--   Copper Metal
--   Red Brick
--   Pine Wood
--   Grass
--   Concrete

CREATE TABLE materials (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    library_id UUID NOT NULL
        REFERENCES libraries(id)
        ON DELETE CASCADE,

    name VARCHAR(500) NOT NULL,

    category VARCHAR(255),

    description TEXT,

    -- Flexible properties specific to the material
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);


-- ============================================================
-- 4. SAMPLES
-- ============================================================
-- A material can have multiple physical/measurement samples.
--
-- Example:
--
-- Copper Metal
--   ├── Sample 1
--   ├── Sample 2
--   └── Sample 3

CREATE TABLE samples (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    material_id UUID NOT NULL
        REFERENCES materials(id)
        ON DELETE CASCADE,

    dataset_id UUID NOT NULL
        REFERENCES datasets(id)
        ON DELETE CASCADE,

    name VARCHAR(500),

    sample_number INTEGER,

    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);


-- ============================================================
-- 5. SPECTRA
-- ============================================================
-- Stores the actual spectral measurement.
--
-- Instead of creating hundreds/thousands of columns,
-- wavelength information and spectral values are stored here.
--
-- Example:
--
-- wavelength_start = 350
-- wavelength_end   = 2500
-- wavelength_step  = 1
--
-- values =
-- [0.123, 0.125, 0.129, ...]


CREATE TABLE spectra (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    sample_id UUID NOT NULL
        REFERENCES samples(id)
        ON DELETE CASCADE,

    wavelength_start DOUBLE PRECISION NOT NULL,

    wavelength_end DOUBLE PRECISION NOT NULL,

    wavelength_step DOUBLE PRECISION,

    measurement_unit VARCHAR(100),

    -- Spectral intensity/reflectance/etc.
    values DOUBLE PRECISION[] NOT NULL,

    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,

    -- How the spectrum was measured (HyperVista): lab | field | airborne | satellite, NULL = untagged
    acquisition_type VARCHAR(20)
        CONSTRAINT spectra_acquisition_type_check
        CHECK (acquisition_type IN ('lab', 'field', 'airborne', 'satellite')),

    -- ROI provenance (HyperVista): the ROI the spectrum was taken from (NULL for pooled class means)
    -- and the ROI class it represents
    roi_id UUID
        REFERENCES rois(id)
        ON DELETE SET NULL,

    roi_class VARCHAR(255),

    -- explicit wavelength (nm) of every value (migration 004); start / end / step are kept as a summary
    wavelengths DOUBLE PRECISION[],

    -- where the spectrum comes from (HyperVista, migration 003)
    purity VARCHAR(20)
        CONSTRAINT spectra_purity_check
        CHECK (purity IS NULL OR purity IN ('ground_measured', 'image_pixel', 'synthetic')),

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);


-- ============================================================
-- 5b. SPECTRUM <-> ROI LINKS (HyperVista, migration 002)
-- ============================================================
-- Every ROI a spectrum was computed from: a pooled class mean links
-- to all ROIs of its class, in one or several images.

CREATE TABLE spectrum_rois (
    spectrum_id UUID NOT NULL REFERENCES spectra(id) ON DELETE CASCADE,
    roi_id UUID NOT NULL REFERENCES rois(id) ON DELETE CASCADE,
    PRIMARY KEY (spectrum_id, roi_id)
);


-- ============================================================
-- 6. DATA FILES
-- ============================================================
-- Keeps track of the original files from which datasets
-- were imported.
--
-- This is important for provenance and reproducibility.

CREATE TABLE data_files (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    dataset_id UUID NOT NULL
        REFERENCES datasets(id)
        ON DELETE CASCADE,

    filename VARCHAR(500) NOT NULL,

    file_type VARCHAR(100),

    file_size BIGINT,

    checksum VARCHAR(128),

    storage_path TEXT,

    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,

    uploaded_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);


-- ============================================================
-- INDEXES
-- ============================================================

-- Dataset → Library
CREATE INDEX idx_datasets_library
    ON datasets(library_id);


-- Material → Library
CREATE INDEX idx_materials_library
    ON materials(library_id);


-- Material category
CREATE INDEX idx_materials_category
    ON materials(category);


-- Sample → Material
CREATE INDEX idx_samples_material
    ON samples(material_id);


-- Sample → Dataset
CREATE INDEX idx_samples_dataset
    ON samples(dataset_id);


-- Spectrum → Sample
CREATE INDEX idx_spectra_sample
    ON spectra(sample_id);


-- Acquisition type filter (HyperVista)
CREATE INDEX idx_spectra_acquisition
    ON spectra(acquisition_type);


-- Category hierarchy prefix filter: category holds 'Vegetation/Tree/Tsuga/Canadensis' (HyperVista)
CREATE INDEX idx_materials_category_path
    ON materials(category varchar_pattern_ops);


-- purity / unit (HyperVista)
CREATE INDEX idx_spectra_purity
    ON spectra(purity);

CREATE INDEX idx_spectra_unit
    ON spectra(measurement_unit);

-- images (HyperVista)
CREATE INDEX idx_images_name
    ON images(name);

CREATE INDEX idx_rois_image
    ON rois(image_id);

-- ROIs (HyperVista)
CREATE INDEX idx_rois_library
    ON rois(library_id);

CREATE INDEX idx_rois_class
    ON rois(library_id, class_name);

CREATE INDEX idx_rois_scene
    ON rois(scene_name);

CREATE INDEX idx_spectra_roi
    ON spectra(roi_id);

CREATE INDEX idx_spectra_roi_class
    ON spectra(roi_class);

CREATE INDEX idx_spectrum_rois_roi
    ON spectrum_rois(roi_id);


-- JSON metadata indexes
CREATE INDEX idx_materials_metadata
    ON materials USING GIN(metadata);

CREATE INDEX idx_datasets_metadata
    ON datasets USING GIN(metadata);

CREATE INDEX idx_samples_metadata
    ON samples USING GIN(metadata);

CREATE INDEX idx_spectra_metadata
    ON spectra USING GIN(metadata);


-- Faster material name searching
CREATE INDEX idx_materials_name_trgm
    ON materials USING GIN(name gin_trgm_ops);


-- ============================================================
-- AUTOMATIC updated_at FUNCTION
-- ============================================================

CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;


-- Apply updated_at automatically

CREATE TRIGGER update_libraries_updated_at
BEFORE UPDATE ON libraries
FOR EACH ROW
EXECUTE FUNCTION update_updated_at_column();


CREATE TRIGGER update_datasets_updated_at
BEFORE UPDATE ON datasets
FOR EACH ROW
EXECUTE FUNCTION update_updated_at_column();


CREATE TRIGGER update_materials_updated_at
BEFORE UPDATE ON materials
FOR EACH ROW
EXECUTE FUNCTION update_updated_at_column();


CREATE TRIGGER update_samples_updated_at
BEFORE UPDATE ON samples
FOR EACH ROW
EXECUTE FUNCTION update_updated_at_column();


CREATE TRIGGER update_rois_updated_at
BEFORE UPDATE ON rois
FOR EACH ROW
EXECUTE FUNCTION update_updated_at_column();


CREATE TRIGGER update_images_updated_at
BEFORE UPDATE ON images
FOR EACH ROW
EXECUTE FUNCTION update_updated_at_column();


-- ============================================================
-- Migration 005 - classification: ground truth + trained models   (DS3 Viewer and Analysis)
-- Safe to run any number of times. DS3 applies it automatically
-- when it connects; to apply by hand:
--   docker exec -i material_library_db psql -U postgres -d material_library < database/migrations/005_classification.sql
-- ============================================================

-- Ground truth: point tables placed by lat/lon (one layer per file, e.g. one abundance sheet per material)
-- or a label map; the label rule last used to turn values into classes is kept with it.
CREATE TABLE IF NOT EXISTS ground_truths (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    library_id UUID REFERENCES libraries(id) ON DELETE SET NULL,
    image_id UUID REFERENCES images(id) ON DELETE SET NULL,
    name VARCHAR(255) NOT NULL,
    kind VARCHAR(20) NOT NULL DEFAULT 'points'
        CONSTRAINT ground_truths_kind_check CHECK (kind IN ('points', 'label_map')),
    layers JSONB NOT NULL DEFAULT '[]'::jsonb,
    value_meaning VARCHAR(100),
    crs TEXT,
    image_rows INTEGER,
    image_cols INTEGER,
    source_files JSONB,
    rule JSONB,
    class_names JSONB,
    n_points INTEGER,
    label_map BYTEA,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_ground_truths_image ON ground_truths(image_id);
CREATE INDEX IF NOT EXISTS idx_ground_truths_library ON ground_truths(library_id);

CREATE TABLE IF NOT EXISTS ground_truth_points (
    id BIGSERIAL PRIMARY KEY,
    ground_truth_id UUID NOT NULL REFERENCES ground_truths(id) ON DELETE CASCADE,
    layer VARCHAR(255) NOT NULL,
    lat DOUBLE PRECISION,
    lon DOUBLE PRECISION,
    pixel_row INTEGER,
    pixel_col INTEGER,
    value DOUBLE PRECISION,
    class_name VARCHAR(255)
);
CREATE INDEX IF NOT EXISTS idx_gt_points_gt ON ground_truth_points(ground_truth_id);
CREATE INDEX IF NOT EXISTS idx_gt_points_latlon ON ground_truth_points(lat, lon);

-- Trained classical models (Random Forest / SVM). The fitted estimator and its feature transform are stored as
-- joblib bytes (checked with sha256 before loading); the other columns say how to reuse it.
CREATE TABLE IF NOT EXISTS classification_models (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    library_id UUID REFERENCES libraries(id) ON DELETE SET NULL,
    image_id UUID REFERENCES images(id) ON DELETE SET NULL,
    ground_truth_id UUID REFERENCES ground_truths(id) ON DELETE SET NULL,
    name VARCHAR(255) NOT NULL,
    model_type VARCHAR(50) NOT NULL,
    family VARCHAR(20) NOT NULL DEFAULT 'classical',
    class_names JSONB NOT NULL,
    n_bands INTEGER NOT NULL,
    wavelengths DOUBLE PRECISION[],
    unit VARCHAR(20),
    feature_kind VARCHAR(20),
    "window" INTEGER,
    components INTEGER,
    label_source VARCHAR(20),
    rule JSONB,
    roi_uids JSONB,
    metrics JSONB,
    settings JSONB,
    config JSONB,
    model_file BYTEA NOT NULL,
    file_format VARCHAR(20) NOT NULL DEFAULT 'joblib',
    size_bytes BIGINT,
    sha256 CHAR(64),
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_clf_models_image ON classification_models(image_id);
CREATE INDEX IF NOT EXISTS idx_clf_models_gt ON classification_models(ground_truth_id);

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'update_ground_truths_updated_at') THEN
        CREATE TRIGGER update_ground_truths_updated_at
        BEFORE UPDATE ON ground_truths
        FOR EACH ROW
        EXECUTE FUNCTION update_updated_at_column();
    END IF;
END $$;
