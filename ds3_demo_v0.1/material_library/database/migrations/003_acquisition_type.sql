-- Acquisition type per spectrum (lab | field | airborne | satellite). Safe to run more than once.
-- (The API also adds the column automatically at start-up; no data is changed.)
ALTER TABLE spectra ADD COLUMN IF NOT EXISTS acquisition_type VARCHAR(20);
ALTER TABLE spectra DROP CONSTRAINT IF EXISTS spectra_acquisition_type_check;
ALTER TABLE spectra ADD CONSTRAINT spectra_acquisition_type_check
    CHECK (acquisition_type IS NULL OR acquisition_type IN ('lab', 'field', 'airborne', 'satellite'));
CREATE INDEX IF NOT EXISTS idx_spectra_acquisition ON spectra (acquisition_type);
CREATE INDEX IF NOT EXISTS idx_materials_category ON materials (category);
