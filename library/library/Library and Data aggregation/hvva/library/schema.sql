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

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
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