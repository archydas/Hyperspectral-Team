import csv, math
from pathlib import Path
import pytest

DATA = Path(__file__).resolve().parents[2] / "data" / "material_spectra.csv"


def make_csv(path, names=("Grass", "Dry grass"), wl=None, header_row=False):
    wl = wl or [400 + 10 * i for i in range(30)]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        if header_row:
            w.writerow(["sample"] + wl)
            for n in names:
                w.writerow([n] + [0.1 + 0.001 * i for i in range(len(wl))])
        else:
            w.writerow(["Wavelength"] + list(names))
            for i, x in enumerate(wl):
                w.writerow([x] + [0.1 + 0.001 * i + 0.01 * j for j in range(len(names))])
    return path


def test_models_ignore_metadata_kwarg_but_use_right_attribute():
    """The first version passed metadata=... which SQLAlchemy silently drops. The right attribute is *_metadata."""
    from app.models import Library
    assert Library(name="a", slug="a", metadata={"x": 1}).library_metadata is None
    assert Library(name="a", slug="a", library_metadata={"x": 1}).library_metadata == {"x": 1}


def test_import_team_csv_end_to_end(client):
    from app.services.importer import import_path
    rep = import_path(DATA, library="Team materials")
    assert rep.as_dict()["imported"] == 100
    libs = client.get("/api/libraries/").json()
    assert len(libs) == 1 and libs[0]["material_count"] == 100 and libs[0]["spectrum_count"] == 100
    lib = libs[0]
    ds = client.get(f"/api/libraries/{lib['id']}/datasets").json()[0]
    assert ds["metadata"]["wavelength_start"] == 350 and ds["metadata"]["row_count"] == 100     # metadata is stored now
    mats = client.get(f"/api/libraries/{lib['id']}/materials", params={"search": "brick"}).json()
    assert len(mats) == 3 and all(m["category"] == "construction" for m in mats) or all(m["category"] is None for m in mats)
    samples = client.get(f"/api/materials/{mats[0]['id']}/samples").json()
    sp = client.get(f"/api/samples/{samples[0]['id']}/spectra").json()[0]
    assert len(sp["values"]) == 2151 == len(sp["wavelengths"])
    assert sp["wavelengths"][0] == 350 and sp["wavelengths"][-1] == 2500 and sp["wavelength_step"] == 1
    assert sp["measurement_unit"] == "reflectance" and sp["wavelength_unit"] == "nm"


def test_reimport_is_skipped_and_other_library_allowed(client):
    from app.services.importer import import_path
    assert import_path(DATA, library="A").as_dict()["imported"] == 100
    assert import_path(DATA, library="A").as_dict()["duplicates"] == 1
    assert import_path(DATA, library="B").as_dict()["imported"] == 100
    assert len(client.get("/api/libraries/").json()) == 2


def test_non_uniform_grid_keeps_real_wavelengths(client, tmp_path):
    from app.services.importer import import_path
    wl = [400, 410, 425, 445, 470, 500, 540, 590, 650, 720]            # not evenly spaced (like AVIRIS-NG/EnMAP)
    f = make_csv(tmp_path / "uneven.csv", wl=wl)
    import_path(f, library="Uneven")
    lib = client.get("/api/libraries/").json()[0]
    m = client.get(f"/api/libraries/{lib['id']}/materials").json()[0]
    s = client.get(f"/api/materials/{m['id']}/samples").json()[0]
    sp = client.get(f"/api/samples/{s['id']}/spectra").json()[0]
    assert sp["wavelength_step"] is None and sp["wavelengths"] == wl


def test_row_wise_csv_and_micrometres(client, tmp_path):
    from app.services.importer import import_path
    f = make_csv(tmp_path / "rows.csv", wl=[0.4 + 0.01 * i for i in range(30)], header_row=True)
    import_path(f, library="Rows")
    lib = client.get("/api/libraries/").json()[0]
    m = client.get(f"/api/libraries/{lib['id']}/materials").json()
    assert len(m) == 2
    sp = client.get(f"/api/samples/{client.get(f'/api/materials/{m[0]['id']}/samples').json()[0]['id']}/spectra").json()[0]
    assert abs(sp["wavelengths"][0] - 400) < 1e-6                      # micrometres converted to nm


def test_works_without_ds3_package(client, tmp_path, monkeypatch):
    from app.services import readers, importer
    monkeypatch.setattr(readers, "HAVE_DS3", False)
    f = make_csv(tmp_path / "plain.csv")
    assert importer.import_path(f, library="NoDS3").as_dict()["imported"] == 2
    with pytest.raises(ValueError, match="unsupported"):
        (tmp_path / "x.sig").write_text("junk")
        importer.import_path(tmp_path / "x.sig", library="x")


def test_bad_inputs_give_readable_errors(client, tmp_path):
    from app.services.importer import import_path
    with pytest.raises(ValueError, match="not found"):
        import_path(tmp_path / "missing.csv")
    (tmp_path / "empty.csv").write_text("a,b\n")
    with pytest.raises(ValueError, match="too short|at least two|fewer than 3"):
        import_path(tmp_path / "empty.csv", library="e")
    (tmp_path / "nothing").mkdir()
    with pytest.raises(ValueError, match="no importable"):
        import_path(tmp_path / "nothing", library="e")


def test_folder_import_and_category_override(client, tmp_path):
    from app.services.importer import import_path
    make_csv(tmp_path / "a.csv"); make_csv(tmp_path / "b.csv", names=("Sand",))
    (tmp_path / "notes.txt").write_text("ignored")
    rep = import_path(tmp_path, library="Folder", category="custom")
    d = rep.as_dict()
    assert d["imported"] == 3 and len(rep.files) == 3 and d["skipped"] == 1 and d["errors"] == 0   # notes.txt is reported, not an error
    cats = client.get("/api/categories").json()
    assert cats == [{"category": "custom", "materials": 3}]


def test_search_stats_delete_and_upload(client, tmp_path):
    from app.services.importer import import_path
    make_csv(tmp_path / "a.csv", names=("Concrete slab", "Pine tree"))
    import_path(tmp_path / "a.csv", library="L1")
    r = client.get("/api/spectra/search", params={"q": "concrete"}).json()
    assert r["total"] == 1 and r["results"][0]["material"] == "Concrete slab" and r["results"][0]["n_points"] == 30
    assert client.get("/api/spectra/search", params={"category": "nonexistent"}).json()["total"] == 0
    st = client.get("/api/stats").json()
    assert st["libraries"] == 1 and st["spectra"] == 2 and st["database"].split(":")[0].startswith(("sqlite", "postgresql"))
    with open(tmp_path / "a.csv", "rb") as fh:                              # upload through the API
        up = client.post("/api/import", files={"file": ("up.csv", fh, "text/csv")}, data={"library": "Uploaded"})
    assert up.status_code == 200 and up.json()["imported"] == 2
    assert client.post("/api/import", files={"file": ("bad.xyz", b"zz", "text/plain")}).status_code == 400
    lib = [l for l in client.get("/api/libraries/").json() if l["name"] == "L1"][0]
    assert client.delete(f"/api/libraries/{lib['id']}").status_code == 200
    assert client.get(f"/api/libraries/{lib['id']}").status_code == 404
    assert client.get("/api/stats").json()["libraries"] == 1                  # cascade removed the rest of L1


def test_nan_values_become_null_in_json(client, tmp_path):
    from app.services.importer import import_path
    f = tmp_path / "gap.csv"
    f.write_text("Wavelength,X\n" + "\n".join(f"{400+10*i},{'' if i == 3 else 0.2}" for i in range(10)))
    import_path(f, library="Gap")
    lib = client.get("/api/libraries/").json()[0]
    m = client.get(f"/api/libraries/{lib['id']}/materials").json()[0]
    s = client.get(f"/api/materials/{m['id']}/samples").json()[0]
    v = client.get(f"/api/samples/{s['id']}/spectra").json()[0]["values"]
    assert v[3] is None and v[2] == 0.2


def test_postgresql_ddl_compiles():
    """No PostgreSQL server here, but the schema must at least compile for it (arrays, JSONB, UUID)."""
    from sqlalchemy.dialects import postgresql
    from sqlalchemy.schema import CreateTable
    from app.database import Base
    import app.models  # noqa
    ddl = "\n".join(str(CreateTable(t).compile(dialect=postgresql.dialect())) for t in Base.metadata.sorted_tables)
    assert "wavelengths FLOAT[]" in ddl and "JSONB" in ddl and "UUID" in ddl


def test_url_normalisation():
    from app.database import normalize_url
    assert normalize_url("postgresql://u:p@h/db") == "postgresql+psycopg://u:p@h/db"
    assert normalize_url("postgres://u:p@h/db") == "postgresql+psycopg://u:p@h/db"
    assert normalize_url("postgresql+psycopg://u:p@h/db") == "postgresql+psycopg://u:p@h/db"
    assert normalize_url("sqlite:///x.db") == "sqlite:///x.db"


def test_frontend_is_served_and_health(client):
    assert client.get("/health").json() == {"status": "ok"}
    r = client.get("/app/")
    assert r.status_code == 200 and "DS3" in r.text


# ---------------------------------------------------------------- multi-format upload, images, library in image
import io, zipfile
import numpy as np

pytest.importorskip("ds3")
scipy_io = pytest.importorskip("scipy.io")

BASE = np.random.default_rng(0).random((3, 103)) + 0.5
LAB = np.zeros((60, 50), int); LAB[:30] = 1; LAB[30:, :25] = 2; LAB[30:, 25:] = 3


def pavia_bytes():
    cube = (BASE[LAB - 1] * 3000).astype("uint16")
    buf = io.BytesIO(); scipy_io.savemat(buf, {"paviaU": cube}); return buf.getvalue()


def lib_txt(names=("Asphalt-like", "Meadow-like", "Gravel-like")):
    wl = np.linspace(430, 860, 103)
    return ("name\t" + "\t".join(f"{w:g}" for w in wl) + "\n" +
            "".join(n + "\t" + "\t".join(f"{v:.5f}" for v in BASE[i]) + "\n" for i, n in enumerate(names))).encode()


def upload(client, name, data, **form):
    return client.post("/api/import", files=[("files", (name, data))], data=form)


def test_upload_txt_library_and_image_then_use_library_in_image(client):
    r = upload(client, "my_library.txt", lib_txt(), library="My library")
    assert r.status_code == 200 and r.json()["imported"] == 3
    r = upload(client, "PaviaU.mat", pavia_bytes())
    assert r.status_code == 200 and r.json()["images"] == 1, r.text
    img = client.get("/api/images/").json()[0]
    assert (img["rows"], img["cols"], img["bands"]) == (60, 50, 103) and img["assumed_wavelengths"] and img["unit"] == "dn"
    iid = img["id"]
    pv = client.get(f"/api/images/{iid}/preview").json()
    assert pv["image"].startswith("data:image/png;base64,") and (pv["width"], pv["height"]) == (50, 60)
    assert client.get(f"/api/images/{iid}/preview?r=1&g=2").status_code == 200       # partial bands -> true colour
    assert client.get(f"/api/images/{iid}/preview?r=1&g=2&b=999").status_code == 400
    px = client.get(f"/api/images/{iid}/pixel?row=40&col=5").json()
    assert len(px["wavelengths"]) == 103 and px["n_pixels"] == 1
    assert client.get(f"/api/images/{iid}/pixel?row=500&col=5").status_code == 400
    lib = [l for l in client.get("/api/libraries/").json() if l["name"] == "My library"][0]["id"]
    # unit differs (image dn, library reflectance): refused with a reason unless the user allows it
    m = client.post(f"/api/images/{iid}/match", json=dict(row=40, col=5, library_id=lib)).json()
    assert m["results"] == [] and "unit" in m["note"]
    m = client.post(f"/api/images/{iid}/match", json=dict(row=40, col=5, library_id=lib, ignore_unit=True)).json()
    assert m["results"][0]["material"] == "Meadow-like" and m["results"][0]["angle_deg"] < 1
    s = client.post(f"/api/images/{iid}/sam_map", json=dict(library_id=lib)).json()
    assert s["overlay"].startswith("data:image/png") and s["unit_mismatch"] == ["reflectance"]
    assert {l["name"]: l["pixels"] for l in s["legend"]} == {"Asphalt-like": 1500, "Meadow-like": 750, "Gravel-like": 750}
    r = client.post(f"/api/images/{iid}/save_pixel", json=dict(row=40, col=5, size=3, library="Field picks", name="p1"))
    assert r.status_code == 200 and r.json()["bands"] == 103
    assert any(l["name"] == "Field picks" for l in client.get("/api/libraries/").json())
    assert client.get("/api/stats").json()["images"] == 1


def test_upload_wavelength_range_for_unknown_mat_and_errors(client):
    buf = io.BytesIO(); scipy_io.savemat(buf, {"cube": np.ones((8, 8, 20), "float32")})
    r = upload(client, "unk.mat", buf.getvalue())
    assert r.status_code == 400 and "wavelength_range" in r.json()["detail"]
    r = upload(client, "unk.mat", buf.getvalue(), wavelengths="450-950")
    assert r.status_code == 200 and client.get("/api/images/").json()[0]["wavelength_max"] == 950
    assert upload(client, "unk.mat", buf.getvalue(), wavelengths="abc").status_code == 400
    r = upload(client, "junk.txt", b"hello\nnot a spectrum\nat all\n")
    assert r.status_code == 400 and "layout" in r.json()["detail"]


def test_zip_with_library_and_image_and_ancillary_file(client):
    z = io.BytesIO()
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("lib/soil.txt", lib_txt(("a", "b", "c")))
        zf.writestr("scene/PaviaU.mat", pavia_bytes())
        zf.writestr("scene/x_rdn_v2m2_glt", b"1234")
    r = upload(client, "bundle.zip", z.getvalue())
    assert r.status_code == 200, r.text
    rep = r.json()
    assert rep["imported"] == 3 and rep["images"] == 1 and rep["skipped"] == 1 and rep["errors"] == 0


def test_two_files_uploaded_together_enviimage(client):
    """ENVI image = .hdr + data file uploaded together."""
    import ds3.sampledata as sd, tempfile, glob, os
    d = Path(tempfile.mkdtemp()); sd.generate(d)
    hdrs = sorted(glob.glob(str(d / "**/*.hdr"), recursive=True))
    pick = next((h for h in hdrs if os.path.exists(h[:-4])), None)
    if pick is None:
        pytest.skip("no ENVI pair in the sample data")
    files = [("files", (Path(pick).name, open(pick, "rb").read())), ("files", (Path(pick[:-4]).name, open(pick[:-4], "rb").read()))]
    r = client.post("/api/import", files=files)
    assert r.status_code == 200 and r.json()["images"] == 1, r.text


def test_upload_filename_cannot_escape_upload_folder(client):
    r = client.post("/api/import", files=[("files", ("../../evil.csv", b"Wavelength,A\n400,0.1\n410,0.2\n420,0.3\n"))])
    from app.database import PROJECT_ROOT
    assert not (PROJECT_ROOT / "evil.csv").exists() and not (PROJECT_ROOT.parent / "evil.csv").exists()
    assert r.status_code in (200, 400)


def test_label_map_upload_makes_class_library(client):
    upload(client, "PaviaU.mat", pavia_bytes())
    iid = client.get("/api/images/").json()[0]["id"]
    buf = io.BytesIO(); scipy_io.savemat(buf, {"paviaU_gt": LAB.astype("uint8")})
    r = client.post(f"/api/images/{iid}/labels", files={"file": ("PaviaU_gt.mat", buf.getvalue())}, data={"library": "PaviaU classes"})
    assert r.status_code == 200, r.text
    assert r.json()["classes"] == ["Asphalt", "Meadows", "Gravel"]            # published Pavia University class names
    lib = [l for l in client.get("/api/libraries/").json() if l["name"] == "PaviaU classes"][0]
    assert lib["spectrum_count"] == 3
    bad = io.BytesIO(); scipy_io.savemat(bad, {"paviaU_gt": np.ones((5, 5), "uint8")})
    r = client.post(f"/api/images/{iid}/labels", files={"file": ("gt.mat", bad.getvalue())})
    assert r.status_code == 400 and "same pixels" in r.json()["detail"]


def test_startup_folder_loads_each_file_as_a_library_and_is_repeatable(client, tmp_path, monkeypatch):
    import numpy as np, time
    from app import main
    wl = np.arange(350, 2501, 10.0); rng = np.random.default_rng(5)
    with open(tmp_path / "material_spectra.txt", "w") as f:
        f.write("name\tclass\t" + "\t".join(f"{w:g}" for w in wl) + "\n")
        for i in range(4): f.write(f"m{i}\tconstruction\t" + "\t".join(f"{v:.4f}" for v in rng.random(len(wl))) + "\n")
    with open(tmp_path / "soil_spectra.txt", "w") as f:
        f.write("Wavelength s1 s2\n" + "".join(f"{w:g} {rng.random():.4f} {rng.random():.4f}\n" for w in wl))
    (tmp_path / "broken.txt").write_text("this\nis not\na spectrum\n")
    monkeypatch.setenv("AUTO_IMPORT_DIR", str(tmp_path))
    main._auto_import()
    libs = {l["name"]: l["spectrum_count"] for l in client.get("/api/libraries/").json()}
    assert libs == {"Material spectra": 4, "Soil spectra": 2}
    st = {f["file"]: f["status"] for f in client.get("/api/startup_import").json()["files"]}
    assert st["material_spectra.txt"] == "imported" and st["broken.txt"] in ("error", "skipped")
    main._auto_import()                                    # second start: nothing duplicated
    libs2 = {l["name"]: l["spectrum_count"] for l in client.get("/api/libraries/").json()}
    assert libs2 == libs


def test_builder_save_spectra_roundtrip(client):
    wl = [400.0 + 10 * i for i in range(20)]
    body = {"library": "Built", "items": [
        {"name": "a", "wavelengths_nm": wl, "values": [0.1 + 0.01 * i for i in range(20)]},
        {"name": "gap", "wavelengths_nm": wl, "values": [None] * 18 + [0.2, 0.3]},          # < 3 valid points -> rejected
        {"name": "bad", "wavelengths_nm": wl[:5], "values": [0.1] * 4}]}
    r = client.post("/api/spectra/save", json=body).json()
    assert r["saved"] == 1 and {x["name"] for x in r["rejected"]} == {"gap", "bad"}
    lib = [l for l in client.get("/api/libraries/").json() if l["name"] == "Built"][0]
    assert lib["spectrum_count"] == 1
    hit = client.get("/api/spectra/search", params={"q": "a", "library_id": lib["id"]}).json()["results"][0]
    sp = client.get(f"/api/spectra/{hit['spectrum_id']}").json()
    assert len(sp["values"]) == 20 and sp["wavelengths"][0] == 400.0


def test_resample_library_to_new_grid(client):
    wl = [400.0 + 10 * i for i in range(60)]
    client.post("/api/spectra/save", json={"library": "Src", "items": [
        {"name": "s1", "wavelengths_nm": wl, "values": [0.1 + 0.005 * i for i in range(60)]},
        {"name": "short", "wavelengths_nm": wl[:10], "values": [0.2] * 10}]})
    lib = [l for l in client.get("/api/libraries/").json() if l["name"] == "Src"][0]
    r = client.post(f"/api/libraries/{lib['id']}/resample", json={"name": "Src 20nm", "mode": "range", "start": 400, "end": 990, "step": 20})
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["resampled"] == 1 and j["n_rejected"] == 1 and j["bands"] == 31          # 'short' covers < 50 % of the grid
    assert client.post(f"/api/libraries/{lib['id']}/resample", json={"name": "src", "mode": "range", "start": 400, "end": 990, "step": 20}).status_code == 400
    new = [l for l in client.get("/api/libraries/").json() if l["name"] == "Src 20nm"][0]
    hit = client.get("/api/spectra/search", params={"library_id": new["id"]}).json()["results"][0]
    sp = client.get(f"/api/spectra/{hit['spectrum_id']}").json()
    assert sp["wavelengths"][1] - sp["wavelengths"][0] == 20 and abs(sp["values"][0] - 0.1) < 1e-9


def _envi_with_map(tmp, rows=20, cols=30, nb=12):
    import numpy as np
    wl = [450 + 40 * i for i in range(nb)]
    data = np.zeros((nb, rows, cols), np.float32)
    data[:, :, :15] = 0.2 + 0.03 * np.arange(nb)[:, None, None]
    data[:, :, 15:] = 0.6 - 0.02 * np.arange(nb)[:, None, None]
    (tmp / "scene").write_bytes(data.tobytes())
    (tmp / "scene.hdr").write_text(
        "ENVI\nsamples = %d\nlines = %d\nbands = %d\nheader offset = 0\nfile type = ENVI Standard\ndata type = 4\n"
        "interleave = bsq\nbyte order = 0\nreflectance scale factor = 1\nwavelength units = Nanometers\n"
        "map info = {UTM, 1, 1, 500000.0, 2000000.0, 30.0, 30.0, 43, North, WGS-84}\nwavelength = {%s}\n"
        % (cols, rows, nb, ", ".join(str(w) for w in wl)))
    return tmp / "scene", tmp / "scene.hdr"


def test_image_geo_roi_bands_and_alpha(client, tmp_path):
    d, h = _envi_with_map(tmp_path)
    r = client.post("/api/import", files=[("files", ("scene", d.read_bytes())), ("files", ("scene.hdr", h.read_bytes()))])
    assert r.status_code == 200 and r.json()["images"] == 1, r.text
    iid = client.get("/api/images/").json()[0]["id"]
    g = client.get(f"/api/images/{iid}/geo").json()
    assert g["has_geo"] and abs(g["corners"][0]["lat"] - 18.09) < 0.2 and 72 < g["corners"][0]["lon"] < 76      # UTM 43N, 500000 E = 75 deg E
    assert g["corners"][0]["lat"] > g["corners"][3]["lat"]                                                       # top row is further north
    b = client.get(f"/api/images/{iid}/bands").json()
    assert len(b["wavelengths"]) == 12 and b["wavelengths"][0] == 450
    a = client.get(f"/api/images/{iid}/preview", params={"alpha": True}).json()
    assert a["image"].startswith("data:image/png")
    roi = client.post(f"/api/images/{iid}/roi", json={"polygon": [[2, 2], [2, 10], [15, 10], [15, 2]]}).json()
    assert roi["n_pixels"] > 50 and abs(roi["values"][0] - 0.2) < 1e-6                                           # left half is 0.2 at band 1
    assert client.post(f"/api/images/{iid}/roi", json={"polygon": [[0, 0], [1, 1]]}).status_code == 422
    # library + ROI match and ROI save
    client.post("/api/spectra/save", json={"library": "Ref", "items": [
        {"name": "left", "wavelengths_nm": b["wavelengths"], "values": [0.2 + 0.03 * i for i in range(12)]},
        {"name": "right", "wavelengths_nm": b["wavelengths"], "values": [0.6 - 0.02 * i for i in range(12)]}]})
    lib = [l for l in client.get("/api/libraries/").json() if l["name"] == "Ref"][0]
    m = client.post(f"/api/images/{iid}/match", json={"polygon": [[2, 2], [2, 10], [15, 10], [15, 2]], "library_id": lib["id"], "ignore_unit": True}).json()
    assert m["results"][0]["material"] == "left" and m["results"][0]["angle_deg"] < 0.01
    s = client.post(f"/api/images/{iid}/save_pixel", json={"polygon": [[2, 18], [2, 28], [15, 28], [15, 18]], "library": "ROIs", "name": "right area"})
    assert s.status_code == 200 and "ROI" in s.json()["material"]
