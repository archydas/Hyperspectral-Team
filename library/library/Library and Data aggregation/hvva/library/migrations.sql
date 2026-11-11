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
