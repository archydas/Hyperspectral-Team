-- ============================================================
-- Migration 001 - category tree + acquisition type   (HyperVista)
-- Safe to run any number of times. HyperVista applies it automatically
-- when it connects; to apply by hand:
--   docker exec -i material_library_db psql -U postgres -d material_library < database/migrations/001_category_tree_acquisition.sql
-- ============================================================

-- How each spectrum was measured. NULL = not tagged yet.
ALTER TABLE spectra ADD COLUMN IF NOT EXISTS acquisition_type VARCHAR(20);

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'spectra_acquisition_type_check') THEN
        ALTER TABLE spectra ADD CONSTRAINT spectra_acquisition_type_check
            CHECK (acquisition_type IN ('lab', 'field', 'airborne', 'satellite'));
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS idx_spectra_acquisition ON spectra(acquisition_type);

-- materials.category holds a hierarchy path, levels separated by '/':
--   Vegetation/Tree/Tsuga/Canadensis
-- This index makes the prefix filter (category LIKE 'Vegetation/Tree/%') fast.
CREATE INDEX IF NOT EXISTS idx_materials_category_path ON materials(category varchar_pattern_ops);


-- ============================================================
-- Migration 002 - regions of interest (ROIs), class by class   (HyperVista)
-- Safe to run any number of times. HyperVista applies it automatically
-- when it connects; to apply by hand:
--   docker exec -i material_library_db psql -U postgres -d material_library < database/migrations/002_rois.sql
--
--   rois           one row per ROI drawn on an image (rectangle or polygon), with its class,
--                  image, pixel geometry, optional map geometry and pixel count.
--                  Several ROIs per class, several classes per image, several images per library.
--   spectra.roi_id     the ROI a spectrum was taken from (ROI mean or ROI pixel); NULL for pooled class means
--   spectra.roi_class  the ROI class a spectrum represents (class-by-class queries without joins)
--   spectrum_rois      every ROI a spectrum was computed from (a pooled class mean -> all ROIs of that class)
-- ============================================================

CREATE TABLE IF NOT EXISTS rois (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    library_id UUID NOT NULL
        REFERENCES libraries(id)
        ON DELETE CASCADE,

    dataset_id UUID
        REFERENCES datasets(id)
        ON DELETE SET NULL,

    -- id of the ROI inside HyperVista (scene file / ROI JSON); saving the same ROI again updates this row
    roi_uid VARCHAR(64) NOT NULL,
    roi_name VARCHAR(255),
    class_name VARCHAR(255) NOT NULL,

    -- the image the ROI was drawn on
    scene_name VARCHAR(500),
    scene_path TEXT,
    image_rows INTEGER,
    image_cols INTEGER,

    -- pixel geometry: rect {"rect": [r0, c0, r1, c1]} (end-exclusive) | polygon {"vertices": [[col, row], ...]}
    geometry_type VARCHAR(20) NOT NULL
        CONSTRAINT rois_geometry_type_check CHECK (geometry_type IN ('rect', 'polygon')),
    geometry JSONB NOT NULL,

    -- same outline in map coordinates (georeferenced images only) - lets a ROI be placed on another image
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

ALTER TABLE spectra ADD COLUMN IF NOT EXISTS roi_id UUID REFERENCES rois(id) ON DELETE SET NULL;
ALTER TABLE spectra ADD COLUMN IF NOT EXISTS roi_class VARCHAR(255);

CREATE TABLE IF NOT EXISTS spectrum_rois (
    spectrum_id UUID NOT NULL REFERENCES spectra(id) ON DELETE CASCADE,
    roi_id UUID NOT NULL REFERENCES rois(id) ON DELETE CASCADE,
    PRIMARY KEY (spectrum_id, roi_id)
);

CREATE INDEX IF NOT EXISTS idx_rois_library ON rois(library_id);
CREATE INDEX IF NOT EXISTS idx_rois_class ON rois(library_id, class_name);
CREATE INDEX IF NOT EXISTS idx_rois_scene ON rois(scene_name);
CREATE INDEX IF NOT EXISTS idx_spectra_roi ON spectra(roi_id);
CREATE INDEX IF NOT EXISTS idx_spectra_roi_class ON spectra(roi_class);
CREATE INDEX IF NOT EXISTS idx_spectrum_rois_roi ON spectrum_rois(roi_id);

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'update_rois_updated_at') THEN
        CREATE TRIGGER update_rois_updated_at
        BEFORE UPDATE ON rois
        FOR EACH ROW
        EXECUTE FUNCTION update_updated_at_column();
    END IF;
END $$;


-- ============================================================
-- Migration 003 - spectrum purity, unit index   (HyperVista)
-- Safe to run any number of times. HyperVista applies it automatically
-- when it connects; to apply by hand:
--   docker exec -i material_library_db psql -U postgres -d material_library < database/migrations/003_units_purity.sql
--
--   spectra.purity            ground_measured (lab / field / reference libraries)
--                             | image_pixel (taken from an image: pixel, ROI, class mean)
--                             | synthetic (simulated / demo data) | NULL (not known)
--   spectra.measurement_unit  (existing column) now holds reflectance | radiance | dn |
--                             transmittance | emissivity | unknown. A library keeps one unit:
--                             HyperVista refuses to add radiance to a reflectance library and
--                             records the library's unit in libraries.metadata->>'unit'.
-- ============================================================

ALTER TABLE spectra ADD COLUMN IF NOT EXISTS purity VARCHAR(20);

ALTER TABLE spectra DROP CONSTRAINT IF EXISTS spectra_purity_check;
ALTER TABLE spectra ADD CONSTRAINT spectra_purity_check
    CHECK (purity IS NULL OR purity IN ('ground_measured', 'image_pixel', 'synthetic'));

CREATE INDEX IF NOT EXISTS idx_spectra_purity ON spectra(purity);
CREATE INDEX IF NOT EXISTS idx_spectra_unit ON spectra(measurement_unit);


-- ============================================================
-- Migration 004 - registered images, explicit wavelengths   (HyperVista)
-- Safe to run any number of times. HyperVista applies it automatically
-- when it connects; to apply by hand:
--   docker exec -i material_library_db psql -U postgres -d material_library < database/migrations/004_images_wavelengths.sql
--
--   images              one row per hyperspectral image the library knows (the pixels stay in the
--                       file; the row keeps where it is and what it is: size, wavelengths, unit,
--                       bad bands, CRS / transform). Same image = same fingerprint.
--   rois.image_id       the image a ROI was drawn on (NULL for ROIs saved before this migration)
--   spectra.wavelengths explicit wavelength of every value, for every spectrum (irregular grids such as
--                       EnMAP / AVIRIS / USGS and regular ones). Older rows are filled from
--                       metadata->'wavelengths', or from wavelength_start + i * wavelength_step.
--   rois.geometry_type  may now also be 'pixels' (pixel lists, e.g. from ENVI ASCII ROI files)
-- ============================================================

CREATE TABLE IF NOT EXISTS images (
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

CREATE INDEX IF NOT EXISTS idx_images_name ON images(name);

ALTER TABLE rois ADD COLUMN IF NOT EXISTS image_id UUID REFERENCES images(id) ON DELETE SET NULL;
CREATE INDEX IF NOT EXISTS idx_rois_image ON rois(image_id);

ALTER TABLE rois DROP CONSTRAINT IF EXISTS rois_geometry_type_check;
ALTER TABLE rois ADD CONSTRAINT rois_geometry_type_check CHECK (geometry_type IN ('rect', 'polygon', 'pixels'));

ALTER TABLE spectra ADD COLUMN IF NOT EXISTS wavelengths DOUBLE PRECISION[];

UPDATE spectra
SET wavelengths = ARRAY(SELECT (jsonb_array_elements_text(metadata->'wavelengths'))::double precision)
WHERE wavelengths IS NULL
  AND jsonb_typeof(metadata->'wavelengths') = 'array'
  AND jsonb_array_length(metadata->'wavelengths') = cardinality("values");

-- every other spectrum: wavelength of each value from start + i * step (or spread evenly start..end),
-- so the column is filled for all rows and new code can read it directly
UPDATE spectra
SET wavelengths = ARRAY(
        SELECT round((CASE WHEN wavelength_step IS NOT NULL THEN wavelength_start + wavelength_step * g
                           ELSE wavelength_start + (wavelength_end - wavelength_start) * g
                                / GREATEST(cardinality("values") - 1, 1) END)::numeric, 6)::double precision
        FROM generate_series(0, cardinality("values") - 1) AS g ORDER BY g)
WHERE wavelengths IS NULL
  AND cardinality("values") >= 2;

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'update_images_updated_at') THEN
        CREATE TRIGGER update_images_updated_at
        BEFORE UPDATE ON images
        FOR EACH ROW
        EXECUTE FUNCTION update_updated_at_column();
    END IF;
END $$;
