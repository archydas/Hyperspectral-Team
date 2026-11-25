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
