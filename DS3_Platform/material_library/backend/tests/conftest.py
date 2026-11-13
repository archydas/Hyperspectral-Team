import os, sys, tempfile
from pathlib import Path

_tmp = tempfile.mkdtemp()
# Tests never touch your real database. Default: a throw-away SQLite file.
# To run the SAME tests on PostgreSQL use an EMPTY scratch database:  set TEST_DATABASE_URL=postgresql+psycopg://user:pw@localhost:5432/ml_test
# (the tests drop and re-create every table in it).
os.environ["DATABASE_URL"] = os.environ.get("TEST_DATABASE_URL") or f"sqlite:///{Path(_tmp, 'test.db').as_posix()}"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("AUTO_IMPORT", "0")
import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client():
    from app.database import Base, engine, init_db
    from app.main import app
    Base.metadata.drop_all(engine)
    init_db()
    with TestClient(app) as c:
        yield c
