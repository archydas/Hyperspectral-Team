import numpy as np, pytest
from fastapi.testclient import TestClient
from ds3.sampledata import generate, WL
from ds3.spectra import read_spectra
from ds3.pipeline import ingest_path, classify
from ds3.ingest import UnsupportedFormatError
from ds3.ingest.metadata import MissingWavelengthsError


@pytest.fixture(scope="module")
def sample(tmp_path_factory):
    d = tmp_path_factory.mktemp("sample"); generate(d); return d


@pytest.fixture
def dburl(tmp_path, monkeypatch):
    url = f"sqlite:///{tmp_path/'t.db'}"
    monkeypatch.setenv("DATABASE_URL", url); return url


def test_csv_wide_and_rowwise(sample):
    w = read_spectra(sample / "field/wide.csv"); r = read_spectra(sample / "field/rowwise.csv")
    assert [s.material for s in w] == ["vegetation", "bare_soil", "water"] and len(r) == 3
    assert np.allclose(w[0].values, r[0].values, atol=1e-4) and w[0].purity == "ground_measured"


def test_svc_percent_to_fraction(sample):
    s = read_spectra(sample / "field/leaf_001.sig")[0]
    assert s.unit == "reflectance" and s.values.max() < 1.0 and s.material == "leaf_001"


def test_ecostress_micrometres_percent(sample):
    s = read_spectra(sample / "library/oak_leaf.spectrum.txt")[0]
    assert s.wavelengths_nm[0] == pytest.approx(400, abs=0.1) and s.values.max() < 1.0 and s.category == "tree"


def test_usgs_needs_wavelengths_and_deleted_to_nan(sample):
    f = sample / "library/splib07a_Dry_soil_synthetic.txt"
    with pytest.raises(MissingWavelengthsError): read_spectra(f)
    s = read_spectra(f, wavelengths_file=sample / "library/splib07a_Wavelengths_synthetic.txt")[0]
    assert np.isnan(s.values).sum() == 2 and s.wavelengths_nm[0] == pytest.approx(400, abs=0.1)


def test_envi_sli(sample):
    out = read_spectra(sample / "library/synthetic.sli")
    assert [s.material for s in out] == ["vegetation", "bare_soil", "water"] and len(out[0].values) == len(WL)
    assert classify(sample / "library/synthetic.hdr") == "spectra"


def test_mat_as_spectra_points_to_image_import(tmp_path):
    f = tmp_path / "x.mat"; f.write_bytes(b"x")
    with pytest.raises(UnsupportedFormatError, match="images \\(cubes\\)"): read_spectra(f)


def test_ingest_all_formats_and_idempotent(sample, dburl):
    first = ingest_path(sample, source_name="t")
    assert all(r.status == "ingested" for r in first), [r for r in first if r.status != "ingested"]
    assert len(first) == 13
    again = ingest_path(sample, source_name="t")
    assert all(r.status == "duplicate" for r in again)


def test_error_is_reported_not_raised(tmp_path, dburl):
    bad = tmp_path / "bad.hdr"; bad.write_text("not envi")
    r = ingest_path(bad)[0]
    assert r.status == "error"


def test_api_flow(sample, dburl):
    from ds3.api.app import app
    with TestClient(app) as c:
        assert c.get("/health").json()["status"] == "ok"
        assert "csv" in c.get("/formats").json()["spectra"]
        res = c.post("/ingest", json={"path": str(sample), "source": "t"}).json()
        assert sum(r["status"] == "ingested" for r in res) == 13
        cubes = c.get("/cubes").json(); assert len(cubes) == 6
        cid = cubes[0]["id"]
        sp = c.get(f"/cubes/{cid}/spectrum?row=2&col=2").json(); assert len(sp["values"]) == 120
        assert c.get(f"/cubes/{cid}/spectrum?row=999&col=2").status_code == 422
        roi = c.post(f"/cubes/{cid}/roi", json={"bbox_px": [0, 0, 5, 5]}).json(); assert roi["n_pixels"] == 25
        assert c.get("/cubes/9999").status_code == 404
        assert len(c.get("/spectra?material=veg").json()) >= 1
        assert c.get("/libraries").json()[0]["n_spectra"] >= 1
        s1 = c.get("/spectra").json()[0]["id"]; assert len(c.get(f"/spectra/{s1}").json()["values"]) > 0


def test_postgres_ddl_compiles():
    from sqlalchemy.schema import CreateTable
    from sqlalchemy.dialects import postgresql
    from ds3.db.models import Base
    ddl = "\n".join(str(CreateTable(t).compile(dialect=postgresql.dialect())) for t in Base.metadata.sorted_tables)
    assert "FLOAT[]" in ddl and "JSONB" in ddl and "CREATE TABLE spectra" in ddl


def test_cli_smoke(tmp_path, dburl, capsys):
    from ds3.cli import main
    main(["sample-data", "--out", str(tmp_path / "s")])
    with pytest.raises(SystemExit) as e: main(["ingest", str(tmp_path / "s")])
    assert e.value.code == 0
    main(["db", "status"]); assert "cubes      6" in capsys.readouterr().out
