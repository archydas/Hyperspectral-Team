from __future__ import annotations
from contextlib import contextmanager
from functools import lru_cache
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from ..config import database_url
from .models import Base


@lru_cache(maxsize=8)
def _engine(url: str):
    kw = {}
    if url.startswith("sqlite"):
        kw["connect_args"] = {"check_same_thread": False}
        if ":memory:" in url or url in ("sqlite://", "sqlite:///"):
            from sqlalchemy.pool import StaticPool
            kw["poolclass"] = StaticPool
    eng = create_engine(url, future=True, **kw)
    if url.startswith("sqlite"):
        @event.listens_for(eng, "connect")
        def _fk(dbapi, _):
            dbapi.execute("PRAGMA foreign_keys=ON")
    return eng


def get_engine(url: str | None = None):
    return _engine(url or database_url())


@contextmanager
def session_scope(url: str | None = None):
    S = sessionmaker(get_engine(url), expire_on_commit=False)
    s = S()
    try:
        yield s
        s.commit()
    except Exception:
        s.rollback(); raise
    finally:
        s.close()


def init_db(url: str | None = None):
    Base.metadata.create_all(get_engine(url))


def reset_db(url: str | None = None):
    Base.metadata.drop_all(get_engine(url)); init_db(url)
