"""Where the app keeps its database. Order: --db argument > DATABASE_URL env var > last used (QSettings) > local SQLite."""
from __future__ import annotations
import os
from pathlib import Path

APP_DIR = Path.home() / ".ds3"


def default_db_url() -> str:
    APP_DIR.mkdir(parents=True, exist_ok=True)
    return "sqlite:///" + (APP_DIR / "ds3_library.db").as_posix()


def sqlite_url(path) -> str:
    return "sqlite:///" + Path(path).resolve().as_posix()


def pretty(url: str) -> str:
    """Hide the password in a postgres URL for display."""
    if "@" in url and "://" in url:
        scheme, rest = url.split("://", 1)
        cred, host = rest.rsplit("@", 1)
        if ":" in cred:
            cred = cred.split(":", 1)[0] + ":****"
        return f"{scheme}://{cred}@{host}"
    return url


def load_saved_url():
    try:
        from PySide6.QtCore import QSettings
        return QSettings("Kapih", "DS3Library").value("db_url", None)
    except Exception:
        return None


def save_url(url: str):
    try:
        from PySide6.QtCore import QSettings
        QSettings("Kapih", "DS3Library").setValue("db_url", url)
    except Exception:
        pass


def resolve_db_url(cli_url=None) -> str:
    return cli_url or os.environ.get("DATABASE_URL") or load_saved_url() or default_db_url()


def default_library_dir() -> Path:
    d = APP_DIR / "libraries"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _get(key, default=None):
    try:
        from PySide6.QtCore import QSettings
        return QSettings("Kapih", "DS3Library").value(key, default)
    except Exception:
        return default


def _set(key, value):
    try:
        from PySide6.QtCore import QSettings
        QSettings("Kapih", "DS3Library").setValue(key, value)
    except Exception:
        pass


def library_dir() -> str:
    """Folder whose files become libraries at every start. Order: DS3_LIBRARY_DIR env > saved > ~/.ds3/libraries."""
    return os.environ.get("DS3_LIBRARY_DIR") or _get("library_dir") or str(default_library_dir())


def save_library_dir(p): _set("library_dir", str(p))


def autoload_enabled() -> bool:
    v = _get("autoload", True)
    return v in (True, "true", "True", 1, "1")


def save_autoload(on: bool): _set("autoload", bool(on))


DOCKER_DB_URL = "postgresql+psycopg://ds3:ds3@localhost:5433/ds3"     # matches docker-compose.yml in the repo root
