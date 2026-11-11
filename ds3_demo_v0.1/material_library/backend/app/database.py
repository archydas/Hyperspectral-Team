"""Database connection. DATABASE_URL (env or backend/.env) decides; with none set a local SQLite file is used,
so the app runs without Docker/PostgreSQL. Nothing here is tied to one machine."""
from __future__ import annotations
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import sessionmaker, declarative_base

def _roots():
    """(PROJECT_ROOT = read-only program files: frontend/, database/;  DATA_HOME = where data is written: database file,
    uploads, libraries folder, .env). They are the same folder when run from source; in a PyInstaller .exe the program
    files are inside the bundle and the data sits next to the .exe (override with the DS3_HOME environment variable)."""
    if getattr(sys, "frozen", False):
        bundle = Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent)) / "material_library"
        return bundle, Path(os.getenv("DS3_HOME") or Path(sys.executable).parent)
    root = Path(__file__).resolve().parents[2]
    return root, Path(os.getenv("DS3_HOME") or root)


PROJECT_ROOT, DATA_HOME = _roots()
load_dotenv(PROJECT_ROOT / "backend" / ".env")      # works whatever the current directory is
load_dotenv(DATA_HOME / ".env")
load_dotenv(PROJECT_ROOT.parent / ".env")           # the .env next to run_gui.py (repository root)
load_dotenv(Path(sys.executable).parent / ".env") if getattr(sys, "frozen", False) else None


def normalize_url(url: str) -> str:
    """postgres:// and postgresql:// -> postgresql+psycopg:// (psycopg 3, which has wheels for new Python versions)."""
    for old in ("postgres://", "postgresql://"):
        if url.startswith(old):
            return "postgresql+psycopg://" + url[len(old):]
    return url


def default_url() -> str:
    (DATA_HOME / "data" / "libraries").mkdir(parents=True, exist_ok=True)     # first start of a fresh install / .exe
    return "sqlite:///" + (DATA_HOME / "data" / "material_library.db").as_posix()


DATABASE_URL = normalize_url(os.getenv("DATABASE_URL") or default_url())
IS_SQLITE = DATABASE_URL.startswith("sqlite")

engine = create_engine(DATABASE_URL, pool_pre_ping=True,
                       **({"connect_args": {"check_same_thread": False}} if IS_SQLITE else {}))
if IS_SQLITE:
    @event.listens_for(engine, "connect")
    def _fk_on(dbapi_conn, _):
        dbapi_conn.execute("PRAGMA foreign_keys=ON")          # so ON DELETE CASCADE works

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def display_url() -> str:
    u = DATABASE_URL
    if "@" in u and "://" in u:
        scheme, rest = u.split("://", 1)
        cred, host = rest.rsplit("@", 1)
        return f"{scheme}://{cred.split(':', 1)[0]}:****@{host}"
    return u


def init_db():
    """Create missing tables and add columns introduced after the first schema (safe to run every start)."""
    from . import models  # noqa: F401  (registers the tables)
    Base.metadata.create_all(engine)
    from sqlalchemy import inspect
    have = {c["name"] for c in inspect(engine).get_columns("spectra")}
    with engine.begin() as c:
        if "wavelengths" not in have:
            c.execute(text("ALTER TABLE spectra ADD COLUMN wavelengths " + ("JSON" if IS_SQLITE else "DOUBLE PRECISION[]")))
        if "acquisition_type" not in have:
            c.execute(text("ALTER TABLE spectra ADD COLUMN acquisition_type VARCHAR(20)"))


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
