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

        Databases created from the original new_hyperspectral schema.sql lack spectra.acquisition_type; this adds
        it (plus its CHECK constraint and indexes) without touching existing data.
        """
        if not self.has_schema():
            return []
        cols = {c["name"] for c in inspect(self.engine).get_columns("spectra")}
        changed = [] if "acquisition_type" in cols else ["spectra.acquisition_type"]
        if self.is_postgres:
            self._run_script(MIGRATIONS_SQL)
        elif changed:
            with self.engine.begin() as c:
                c.exec_driver_sql("ALTER TABLE spectra ADD COLUMN acquisition_type VARCHAR(20)")
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
