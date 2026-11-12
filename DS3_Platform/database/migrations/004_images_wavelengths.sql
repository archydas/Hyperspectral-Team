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
