"""ROI patches (<class name>.tif), control-point georeferencing, file helpers — material_library backend."""
import pathlib, shutil, sys
import numpy as np, pytest
from fastapi.testclient import TestClient

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = pathlib.Path(__file__).parent / "e2e" / "data"
sys.path.insert(0, str(ROOT / "material_library" / "backend"))
sys.path.insert(0, str(ROOT / "src"))


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path/'t.db'}")
    monkeypatch.setenv("AUTO_IMPORT", "0")
    monkeypatch.setenv("DS3_LOCAL_APP", "1")
    for m in [m for m in sys.modules if m == "app" or m.startswith("app.")]:
        del sys.modules[m]
    from app.main import app
    with TestClient(app) as c:
        yield c


def _scene(tmp_path, name, with_map):
    d = tmp_path / name; d.mkdir()
    shutil.copy(DATA / "sceneA", d / "sceneA")
    hdr = (DATA / "sceneA.hdr").read_text()
    if not with_map:
        hdr = "\n".join(l for l in hdr.splitlines() if not l.startswith("map info"))
    (d / "sceneA.hdr").write_text(hdr)
    return d / "sceneA.hdr"


def _open(client, path):
    r = client.post("/api/import/path", json={"paths": [str(path)]}); assert r.status_code == 200, r.text
    return client.get("/api/images/").json()[0]["id"]


def test_patch_is_named_by_class_georeferenced_and_reloadable(client, tmp_path):
    iid = _open(client, _scene(tmp_path, "geo", True))
    out = tmp_path / "patches"
    poly = [[5, 5], [5, 20], [18, 20], [18, 5]]
    r = client.post(f"/api/images/{iid}/patch", json=dict(polygon=poly, class_name="Dry grass", folder=str(out)))
    assert r.status_code == 200, r.text
    p = r.json(); assert p["file"] == "Dry grass.tif" and p["georeferenced"] and p["lat"] and p["lon"]
    assert (out / "Dry grass.tif").is_file() and p["bands"] == 30
    again = client.post(f"/api/images/{iid}/patch", json=dict(polygon=poly, class_name="Dry grass", folder=str(out))).json()
    assert again["file"] == "Dry grass_2.tif"                                         # same class twice: never overwritten
    import rasterio
    with rasterio.open(out / "Dry grass.tif") as d:
        assert d.count == 30 and d.crs is not None and d.height == 14 and d.width == 16
        assert d.tags(ns="ENVI")["wavelength_units"] == "nm"
    # the patch opens again as an image with the same wavelengths (Library > Load image)
    r = client.post("/api/import/path", json={"paths": [str(out / "Dry grass.tif")]}); assert r.status_code == 200, r.text
    imgs = client.get("/api/images/").json()
    patch = [i for i in imgs if i["path"].endswith("Dry grass.tif")][0]
    assert patch["bands"] == 30 and patch["rows"] == 14 and patch["wavelength_min"] == 450


def test_patch_rejects_bad_input_and_remote_clients(client, tmp_path):
    iid = _open(client, _scene(tmp_path, "g2", True))
    assert client.post(f"/api/images/{iid}/patch", json=dict(polygon=[[0, 0], [1, 1]], class_name="x")).status_code == 422
    assert client.post(f"/api/images/{iid}/patch", json=dict(polygon=[[900, 900], [900, 950], [950, 950]], class_name="x",
                                                              folder=str(tmp_path))).status_code == 400
    d = client.post(f"/api/images/{iid}/patch/download", json=dict(polygon=[[2, 2], [2, 9], [9, 9]], class_name="Soil/1"))
    assert d.status_code == 200 and d.headers["content-type"] == "image/tiff" and "Soil_1.tif" in d.headers["content-disposition"]


def test_gcp_georeferencing_fit_and_use(client, tmp_path):
    iid = _open(client, _scene(tmp_path, "nogeo", False))
    assert client.get(f"/api/images/{iid}/geo").json()["has_geo"] is False
    # truth: lon = 10 + 1e-4*x, lat = 50 - 1e-4*y   (x = col + 0.5, y = row + 0.5)
    def pt(row, col): return dict(row=row, col=col, lon=10 + 1e-4 * (col + .5), lat=50 - 1e-4 * (row + .5))
    assert client.put(f"/api/images/{iid}/georef", json=dict(points=[pt(0, 0), pt(0, 50)])).status_code == 422
    assert client.put(f"/api/images/{iid}/georef", json=dict(points=[pt(0, 0), pt(5, 5), pt(10, 10)])).status_code == 400   # one line
    r = client.put(f"/api/images/{iid}/georef", json=dict(points=[pt(0, 0), pt(0, 59), pt(39, 0), pt(39, 59)]))
    assert r.status_code == 200, r.text
    g = r.json()["georef"]; assert g["n"] == 4 and g["rmse_m"] < 0.01 and abs(g["pixel_size_m"][0] - 7.1) < 1
    assert r.json()["geo"]["has_geo"] is True
    ll = client.post(f"/api/images/{iid}/lonlat", json=dict(points=[[10, 20]])).json()["points"][0]
    assert ll["lon"] == pytest.approx(10 + 1e-4 * 20.5, abs=1e-7) and ll["lat"] == pytest.approx(50 - 1e-4 * 10.5, abs=1e-7)
    # a wrong control point shows up in the RMSE
    bad = [pt(0, 0), pt(0, 59), pt(39, 0), dict(pt(39, 59), lat=50 - 1e-4 * 45)]
    assert client.put(f"/api/images/{iid}/georef", json=dict(points=bad)).json()["georef"]["rmse_m"] > 1
    client.put(f"/api/images/{iid}/georef", json=dict(points=[pt(0, 0), pt(0, 59), pt(39, 0), pt(39, 59)]))
    p = client.post(f"/api/images/{iid}/patch", json=dict(polygon=[[5, 5], [5, 20], [18, 20]], class_name="Road", folder=str(tmp_path / "pp"))).json()
    assert p["georeferenced"] and p["lon"] == pytest.approx(10 + 1e-4 * 15.5, abs=2e-4)
    assert client.delete(f"/api/images/{iid}/georef").status_code == 200
    assert client.get(f"/api/images/{iid}/geo").json()["has_geo"] is False


def test_file_helpers(client, tmp_path):
    f = tmp_path / "a.tif"; f.write_bytes(b"x")
    info = client.post("/api/system/path_info", json=dict(paths=[str(f), str(tmp_path / "gone.tif")])).json()
    assert [i["exists"] for i in info] == [True, False] and info[0]["name"] == "a.tif"
    assert client.post("/api/system/check_folder", json=dict(path=str(tmp_path / "new" / "sub"))).status_code == 200
    assert (tmp_path / "new" / "sub").is_dir()
    assert client.post("/api/system/reveal", json=dict(path=str(tmp_path / "gone.tif"))).status_code == 404
    assert client.get("/api/system/patch_folder").json()["default"].endswith("patches")


def test_roi_statistics_preview_detail_and_dry_run_fit(client, tmp_path):
    iid = _open(client, _scene(tmp_path, "st", False))
    r = client.post(f"/api/images/{iid}/roi", json=dict(polygon=[[3, 3], [3, 30], [25, 30], [25, 3]])).json()
    n = len(r["wavelengths"])
    assert len(r["std"]) == len(r["min"]) == len(r["max"]) == n and r["n_pixels"] > 100
    ok = [i for i, v in enumerate(r["values"]) if v is not None]
    assert all(r["min"][i] <= r["values"][i] <= r["max"][i] and r["std"][i] >= 0 for i in ok)
    # same numbers as numpy on the pixels inside
    import numpy as np
    raw = np.fromfile(DATA / "sceneA", "<f4").reshape(30, 40, 60)           # bsq: bands, lines, samples
    inside = raw[:, 3:26, 3:31].reshape(30, -1)
    i = ok[0]
    assert r["std"][i] == pytest.approx(float(inside[i].std(ddof=1)), rel=0.05)
    small = client.get(f"/api/images/{iid}/preview?max_px=100").json()      # clamped to the 50k minimum
    big = client.get(f"/api/images/{iid}/preview?max_px=4000000").json()
    assert small["step"] >= 1 and big["step"] == 1
    pts = [dict(row=0, col=0, lon=10, lat=50), dict(row=0, col=59, lon=10.006, lat=50), dict(row=39, col=0, lon=10, lat=49.996)]
    f = client.post(f"/api/images/{iid}/georef/fit", json=dict(points=pts)).json()
    assert f["n"] == 3 and f["exact_fit"] and client.get(f"/api/images/{iid}/georef").json()["georef"] is None   # nothing saved


def test_object_library_lists_patches_and_loads_them_in_place(client, tmp_path):
    iid = _open(client, _scene(tmp_path, "ol", True))
    out = tmp_path / "pt"
    poly = [[4, 4], [4, 20], [16, 20], [16, 4]]
    p = client.post(f"/api/images/{iid}/patch", json=dict(polygon=poly, class_name="Water", folder=str(out))).json()
    roi = client.post(f"/api/images/{iid}/roi", json=dict(polygon=poly)).json()
    body = dict(library="Objects A", library_info={"type": "object"}, items=[dict(
        name="Pond 1", wavelengths_nm=roi["wavelengths"], values=roi["values"], unit=roi["unit"], origin="image_roi",
        sample_metadata={"object": dict(kind="roi", class_name="Water", image_name="sceneA", polygon=poly, n_pixels=roi["n_pixels"],
                                        lat=p["lat"], lon=p["lon"], patch_file=p["file"], patch_path=p["path"])})])
    assert client.post("/api/spectra/save", json=body).status_code == 200
    lib = [l for l in client.get("/api/libraries/").json() if l["name"] == "Objects A"][0]
    assert (lib.get("metadata") or {}).get("type") == "object"
    objs = client.get(f"/api/libraries/{lib['id']}/objects").json()["objects"]
    assert len(objs) == 1 and objs[0]["patch_file"] == "Water.tif" and objs[0]["patch_exists"] is True and "path" not in str(objs[0].keys())
    sid = objs[0]["sample_id"]
    r = client.post(f"/api/samples/{sid}/patch/load"); assert r.status_code == 200, r.text
    again = client.post(f"/api/samples/{sid}/patch/load").json()
    assert again["image_id"] == r.json()["image_id"] and again["status"] == "duplicate"        # same file: no second copy in the list
    img = client.get(f"/api/images/{r.json()['image_id']}").json()
    assert img["bands"] == 30 and img["name"] == "Water"
    (out / "Water.tif").rename(out / "moved.tif")
    assert client.get(f"/api/libraries/{lib['id']}/objects").json()["objects"][0]["patch_exists"] is False
    assert client.post(f"/api/samples/{sid}/patch/load").status_code == 404


def test_patch_keeps_all_bands_and_default_display_bands(client, tmp_path):
    iid = _open(client, _scene(tmp_path, "rgb", True))
    out = tmp_path / "p"; poly = [[5, 5], [5, 20], [18, 20], [18, 5]]
    p = client.post(f"/api/images/{iid}/patch", json=dict(polygon=poly, class_name="A", folder=str(out), bands=[20, 12, 4])).json()
    q = client.post(f"/api/images/{iid}/patch", json=dict(polygon=poly, class_name="B", folder=str(out))).json()
    assert p["display_bands"] == [20, 12, 4] and len(q["display_bands"]) == 3          # the page's R, G, B; else bands near 650 / 550 / 470 nm
    import rasterio
    with rasterio.open(out / "A.tif") as d:
        assert d.count == 30 and d.tags()["display_bands_rgb"] == "20,12,4" and d.tags(ns="ENVI")["default_bands"] == "{20, 12, 4}"
