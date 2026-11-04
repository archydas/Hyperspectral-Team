-- Registered hyperspectral images (description only; pixel data stays in the file). Safe to run more than once.
-- (The API also creates this table automatically at start-up.)
CREATE TABLE IF NOT EXISTS images (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    path TEXT NOT NULL,
    sha256 VARCHAR(64) NOT NULL UNIQUE,
    format VARCHAR(50),
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    bands INTEGER NOT NULL,
    unit VARCHAR(50),
    sensor VARCHAR(100),
    wl_min_nm DOUBLE PRECISION,
    wl_max_nm DOUBLE PRECISION,
    wavelengths DOUBLE PRECISION[],
    bad_band_mask JSONB,
    reader_kw JSONB NOT NULL DEFAULT '{}'::jsonb,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
