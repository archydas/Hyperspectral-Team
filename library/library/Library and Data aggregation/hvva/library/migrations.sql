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
