-- For a database created from the FIRST version of schema.sql. Safe to run more than once.
-- (The API also runs this automatically at start-up.)
ALTER TABLE spectra ADD COLUMN IF NOT EXISTS wavelengths DOUBLE PRECISION[];
