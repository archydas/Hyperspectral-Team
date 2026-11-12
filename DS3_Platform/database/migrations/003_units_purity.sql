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
