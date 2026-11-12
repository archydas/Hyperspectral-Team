"""Database connection for the material spectral library (PostgreSQL).

Adapted from new_hyperspectral/material_library/backend/app/database.py.  Differences:

* Nothing connects at import time and a missing DATABASE_URL is not fatal - a desktop app has to start
  without a server and let the user enter the URL in the GUI.
* ``connect(url)`` builds an engine + session factory on demand; ``init_schema`` creates the tables from
  the bundled schema.sql (identical to database/schema.sql) when the database is still empty.
* sqlite:/// URLs also work (offline/testing): the models use portable column types.
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import declarative_base, sessionmaker

DEFAULT_URL = "postgresql://postgres:postgres@localhost:5432/material_library"
SCHEMA_SQL = Path(__file__).with_name("schema.sql")
MIGRATIONS_SQL = Path(__file__).with_name("migrations.sql")

Base = declarative_base()


def _load_dotenv():
    """Tiny .env reader (DATABASE_URL=...) so python-dotenv is not required."""
    for p in (Path.cwd() / ".env", Path(__file__).resolve().parents[2] / ".env"):
        if p.is_file():
            for line in p.read_text(encoding="utf-8").splitlines():
                k, _, v = line.partition("=")
                if k.strip() and _ and k.strip() not in os.environ:
                    os.environ[k.strip()] = v.strip().strip('"').strip("'")


def default_url() -> str:
    _load_dotenv()
    return os.getenv("DATABASE_URL") or DEFAULT_URL


def resolve_driver(url: str) -> str:
    """'postgresql://' -> an installed driver.  SQLAlchemy 2.0 defaults to psycopg2, 2.1 to psycopg (v3);
    the backend's requirements install psycopg2-binary, so pick whichever is actually importable."""
    if url.startswith(("postgresql://", "postgres://")):
        rest = url.split("://", 1)[1]
        for mod, drv in (("psycopg2", "psycopg2"), ("psycopg", "psycopg")):
            try:
                __import__(mod)
                return f"postgresql+{drv}://{rest}"
            except ImportError:
                continue
        raise RuntimeError("no PostgreSQL driver installed - pip install psycopg2-binary")
    return url


class Database:
    """Engine + session factory for one database URL."""

    def __init__(self, url: str | None = None, connect_timeout: int = 5):
        self.url = resolve_driver((url or default_url()).strip())
        kw = {"pool_pre_ping": True}
        if self.url.startswith("postgresql"):
            kw["connect_args"] = {"connect_timeout": connect_timeout}
        self.engine = create_engine(self.url, **kw)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)

    @property
    def is_postgres(self) -> bool:
        return self.engine.dialect.name == "postgresql"

    def ping(self):
        with self.engine.connect() as c:
            c.execute(text("SELECT 1"))

    def _run_script(self, path: Path):
        """Run a .sql file verbatim on the raw driver cursor (no parameter parsing, so '%' in it is harmless)."""
        raw = self.engine.raw_connection()
        try:
            cur = raw.cursor()
            cur.execute(path.read_text(encoding="utf-8"))
            cur.close()
            raw.commit()
        except Exception:
            raw.rollback()
            raise
        finally:
            raw.close()

    def has_schema(self) -> bool:
        return inspect(self.engine).has_table("libraries")

    def init_schema(self) -> bool:
        """Create the tables if the database is empty. Returns True if it created them."""
        if self.has_schema():
            return False
        if self.is_postgres:
            self._run_script(SCHEMA_SQL)
        else:
            from . import models  # noqa: F401  (registers the tables)
            Base.metadata.create_all(self.engine)
        return True

    def migrate(self) -> list:
        """Bring an existing database up to date (idempotent). Returns what was changed.

        001: spectra.acquisition_type (+ CHECK constraint and indexes).
        002: rois table, spectra.roi_id / spectra.roi_class, spectrum_rois link table (ROI analysis).
        003: spectra.purity (ground_measured | image_pixel | synthetic) + unit / purity indexes.
        004: images table, rois.image_id, spectra.wavelengths (filled from metadata), 'pixels' ROIs.
        Existing data is never lost.
        """
        if not self.has_schema():
            return []
        insp = inspect(self.engine)
        cols = {c["name"] for c in insp.get_columns("spectra")}
        changed = [f"spectra.{c}" for c in ("acquisition_type", "roi_id", "roi_class", "purity", "wavelengths")
                   if c not in cols]
        changed += [t for t in ("rois", "spectrum_rois", "images") if not insp.has_table(t)]
        if insp.has_table("rois") and "image_id" not in {c["name"] for c in insp.get_columns("rois")}:
            changed.append("rois.image_id")
        if self.is_postgres:
            self._run_script(MIGRATIONS_SQL)
        elif changed:
            from . import models  # noqa: F401  (registers the tables)
            Base.metadata.create_all(self.engine, tables=[Base.metadata.tables[t] for t in
                                                          ("images", "rois", "spectrum_rois")], checkfirst=True)
            with self.engine.begin() as c:
                adds = {"spectra.acquisition_type": "ALTER TABLE spectra ADD COLUMN acquisition_type VARCHAR(20)",
                        "spectra.roi_id": "ALTER TABLE spectra ADD COLUMN roi_id CHAR(32) REFERENCES rois(id)",
                        "spectra.roi_class": "ALTER TABLE spectra ADD COLUMN roi_class VARCHAR(255)",
                        "spectra.purity": "ALTER TABLE spectra ADD COLUMN purity VARCHAR(20)",
                        "spectra.wavelengths": "ALTER TABLE spectra ADD COLUMN wavelengths JSON",
                        "rois.image_id": "ALTER TABLE rois ADD COLUMN image_id CHAR(32) REFERENCES images(id)"}
                for key, sql in adds.items():
                    if key in changed:
                        c.exec_driver_sql(sql)
                if "spectra.wavelengths" in changed:              # fill the new column from the metadata copy
                    c.exec_driver_sql("UPDATE spectra SET wavelengths = json_extract(metadata, '$.wavelengths') "
                                      "WHERE wavelengths IS NULL AND json_type(metadata, '$.wavelengths') = 'array' "
                                      "AND json_array_length(metadata, '$.wavelengths') = json_array_length(\"values\")")
                    _fill_regular_wavelengths_sqlite(c)
        return changed

    @contextmanager
    def session(self):
        db = self.SessionLocal()
        try:
            yield db
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def get_db(self):
        """FastAPI dependency (same shape as the original backend's get_db)."""
        db = self.SessionLocal()
        try:
            yield db
        finally:
            db.close()

    def dispose(self):
        self.engine.dispose()

    def safe_url(self) -> str:
        return self.engine.url.render_as_string(hide_password=True)


def connect(url: str | None = None, init: bool = False) -> Database:
    db = Database(url)
    db.ping()
    if init:
        db.init_schema()
    db.migrate()
    return db


def _fill_regular_wavelengths_sqlite(conn):
    """SQLite part of migration 004: spectra without explicit wavelengths get start + i * step (or an even
    spread start..end), so spectra.wavelengths is filled for every row."""
    import json
    rows = conn.exec_driver_sql("SELECT id, wavelength_start, wavelength_end, wavelength_step, \"values\" FROM spectra "
                                "WHERE wavelengths IS NULL").fetchall()
    for sid, start, end, step, vals in rows:
        try:
            n = len(json.loads(vals) if isinstance(vals, str) else vals or [])
        except (TypeError, ValueError):
            continue
        if n < 2 or start is None:
            continue
        if step:
            wl = [round(float(start) + float(step) * i, 6) for i in range(n)]
        else:
            wl = [round(float(start) + (float(end) - float(start)) * i / (n - 1), 6) for i in range(n)]
        conn.exec_driver_sql("UPDATE spectra SET wavelengths = ? WHERE id = ?", (json.dumps(wl), sid))
