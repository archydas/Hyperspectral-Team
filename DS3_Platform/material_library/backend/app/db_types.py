"""Column types that work on PostgreSQL (production) and SQLite (zero-install demo/tests)."""
from sqlalchemy import JSON, Float
from sqlalchemy.dialects.postgresql import JSONB, ARRAY

JSONType = JSON().with_variant(JSONB(), "postgresql")
FloatArray = ARRAY(Float).with_variant(JSON(), "sqlite")
