"""v3.4: image viewer helpers, ROI files (GeoJSON / ENVI ASCII), map placement and georeferencing helpers,
Spectral Library Builder / resampling / export, search filters, RMSE matching, images table and the
explicit spectra.wavelengths column.  Database tests run on SQLite, and on PostgreSQL too when
HVVA_TEST_PG_URL is set."""
import csv
import json
import os
import socket
import threading
import time

import numpy as np
import pytest

from hvva.core import basemap as BM
from hvva.core import display as D
from hvva.core import roifiles as RF
from hvva.core import rois as R
from hvva.core.geo import GeoInfo
from hvva.core.scene import Scene, image_info, load_scene, save_scene
from hvva.core.synthetic import make_demo_scenes
from hvva.library import builder as B
from hvva.library import repository as repo
from hvva.library.client import DirectClient
from hvva.library.database import Database


def _scene(h=20, w=30, nb=50, geo=True):
    wl = np.linspace(400, 2400, nb)
    rng = np.random.default_rng(0)
    cube = (0.1 + 0.4 * rng.random((h, w, nb))).astype(np.float32)
    sc = Scene("t", cube, wl, GeoInfo("EPSG:32633", (30.0, 0.0, 500000.0, 0.0, -30.0, 4100000.0)) if geo else None,
               unit="reflectance")
    return sc


# ================================================================== display
def test_display_render_and_persist(tmp_path):
    sc = _scene()
    img, cmap = D.render(sc)
    assert img.shape == (20, 30, 3) and cmap is None and 0 <= img.min() and img.max() <= 1
    assert D.preset_bands(sc, "True colour") == [D.nearest_band(sc, t) for t in (650, 550, 450)]
    sc.display = {"mode": "gray", "bands": [7], "stretch": [5, 95]}
    img, cmap = D.render(sc)
    assert img.ndim == 2 and cmap == "gray"
    assert D.normalize_display(sc, {"mode": "rgb", "bands": [1]})["bands"] == D.preset_bands(sc, "True colour")
    assert D.normalize_display(sc, {"mode": "gray", "bands": [999], "stretch": [-5, 300]}) == \
        {"mode": "gray", "bands": [49], "stretch": [0.0, 100.0]}
    sc.bad_file = np.zeros(sc.n_bands, bool)
    sc.bad_file[3] = True
    assert D.band_label(sc, 3).endswith("(bad)") and "nm" in D.band_label(sc, 3)
    p = tmp_path / "s.npz"
    save_scene(sc, str(p))
    assert load_scene(str(p)).display == sc.display


def test_spectrum_window():
    sc = _scene()
    s3 = sc.spectrum_at(5, 5, 3)
    exp = sc.masked(sc.cube[4:7, 4:7].reshape(-1, sc.n_bands).mean(axis=0))      # bad bands stay gaps
    assert np.allclose(s3, exp, atol=1e-6, equal_nan=True) and np.isnan(s3).sum() == sc.bad_bands.sum()
    s1 = sc.spectrum_at(0, 0, 5)                                  # clipped at the corner
    assert np.allclose(s1, sc.masked(sc.cube[0:3, 0:3].reshape(-1, sc.n_bands).mean(axis=0)), atol=1e-6, equal_nan=True)


# ================================================================== ROI files
def test_pixels_roi_kind():
    sc = _scene()
    r = sc.add_roi({"kind": "pixels", "pixels": [[3, 4], [3, 4], [10, 11], [99, 99]], "class": "pts"})
    assert r["pixels"] == [[3, 4], [10, 11], [99, 99]]            # unique and sorted
    m = R.roi_mask(r, 20, 30)                                     # pixels outside the image are ignored
    assert m.sum() == 2 and m[3, 4] and m[10, 11]
    assert "pixel list" in R.describe_geometry(r)
    back = R.from_ds3_json(R.to_ds3_json([(sc, r)]), 20, 30)[0]
    assert back[0][1]["pixels"] == r["pixels"][:2]
    r2 = sc.add_roi({"kind": "rect", "rect": (0, 0, 2, 2), "class": "a", "name": r["name"]})
    assert r2["name"] == f"{r['name']} (2)"                       # names stay unique per image


def test_geojson_roundtrip_map_and_pixel(tmp_path):
    sc = _scene()
    sc.add_roi({"kind": "rect", "rect": (2, 3, 8, 10), "class": "veg"})
    sc.add_roi({"kind": "polygon", "vertices": [(12, 2), (25, 4), (20, 15)], "class": "soil"})
    sc.add_roi({"kind": "pixels", "pixels": [[1, 1], [15, 20]], "class": "pts"})
    items = [(sc, r) for r in sc.rois]
    p = tmp_path / "r.geojson"
    RF.write_geojson(items, str(p))
    doc = json.load(open(p))
    assert doc["crs"]["properties"]["name"] == "EPSG:32633"
    assert doc["features"][0]["geometry"]["coordinates"][0][0][0] > 400000        # map coordinates
    rois, skipped, space = RF.read_geojson(str(p), sc)
    assert space == "map" and skipped == 0 and len(rois) == 3
    for a, b in zip(sc.rois, rois):
        assert (R.roi_mask(a, 20, 30) == R.roi_mask(b, 20, 30)).all(), a["class"]
        assert a["class"] == b["class"]
    # pixel coordinates (not georeferenced)
    sc2 = _scene(geo=False)
    sc2.rois = [dict(r) for r in sc.rois]
    q = tmp_path / "px.geojson"
    RF.write_geojson([(sc2, r) for r in sc2.rois], str(q))
    rois2, _sk, space2 = RF.read_geojson(str(q), sc2)
    assert space2 == "pixel" and [R.roi_mask(r, 20, 30).sum() for r in rois2] == \
        [R.roi_mask(r, 20, 30).sum() for r in sc.rois]


def test_geojson_lonlat_import():
    sc = _scene()
    corners = [sc.geo.xy(r, c, center=False) for r, c in ((2, 2), (2, 12), (10, 12), (10, 2))]
    lon, lat = BM.to_lonlat(np.array([x for x, _ in corners]), np.array([y for _, y in corners]), "EPSG:32633")
    doc = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {"label": "water"},
         "geometry": {"type": "Polygon", "coordinates": [[[float(a), float(b)] for a, b in zip(lon, lat)]]}},
        {"type": "Feature", "properties": {}, "geometry": {"type": "LineString", "coordinates": [[0, 0], [1, 1]]}}]}
    rois, skipped, space = RF.read_geojson(doc, sc)
    assert space == "lonlat" and skipped == 1 and rois[0]["class"] == "water"
    m = R.roi_mask(rois[0], 20, 30)
    assert m[2:10, 2:12].all() and m.sum() == 80


def test_envi_ascii_roundtrip(tmp_path):
    sc = _scene()
    sc.add_roi({"kind": "rect", "rect": (2, 3, 5, 6), "class": "veg", "name": "field A"})
    sc.add_roi({"kind": "polygon", "vertices": [(12, 2), (25, 4), (20, 15)], "class": "soil"})
    p = tmp_path / "rois.txt"
    RF.write_envi_ascii(sc, sc.rois, str(p), with_values=True)
    text = open(p).read()
    assert "; ROI name: field A [veg]" in text and "Map X" in text and "B50" in text
    rois, skipped = RF.read_envi_ascii(str(p), 20, 30)
    assert skipped == 0 and [r["kind"] for r in rois] == ["pixels", "pixels"]
    assert rois[0]["class"] == "veg" and rois[0]["name"] == "field A"
    for a, b in zip(sc.rois, rois):
        assert (R.roi_mask(a, 20, 30) == R.roi_mask(b, 20, 30)).all()
    small, sk = RF.read_envi_ascii(str(p), 8, 8)                     # smaller image: outside pixels dropped
    assert sk > 0
    with pytest.raises(ValueError):
        (tmp_path / "x.txt").write_text("hello")
        RF.read_envi_ascii(str(tmp_path / "x.txt"), 5, 5)


# ================================================================== map / georeferencing
def test_utm_lonlat_and_mercator():
    lon, lat = BM.to_lonlat(np.array([500000.0]), np.array([4100000.0]), "EPSG:32633")
    assert abs(lon[0] - 15.0) < 1e-9 and abs(lat[0] - 37.0464) < 1e-3
    x, y = BM.from_lonlat(lon, lat, "UTM zone 33N")
    assert abs(x[0] - 500000) < 1e-3 and abs(y[0] - 4100000) < 1e-3
    assert BM.epsg_of("WGS 84 / UTM zone 44S") == 32744 and BM.can_place("EPSG:4326")
    X, Y = BM.lonlat_to_merc(np.array([0.0, 180.0]), np.array([0.0, 0.0]))
    assert abs(X[1] - 20037508.34) < 1 and abs(Y[0]) < 1e-6
    assert np.allclose(BM.merc_to_lonlat(X, Y), ([0, 180], [0, 0]))


def test_scene_affine_and_inverse():
    sc = _scene()
    M = BM.scene_affine(sc)
    for r, c in ((0, 0), (10.0, 20.0), (19, 29)):
        X, Y = M @ np.array([c + 0.5, r + 0.5, 1.0])
        rr, cc = BM.merc_to_pixel(M, X, Y)
        assert abs(rr - r) < 1e-6 and abs(cc - c) < 1e-6
        lon, lat = BM.to_lonlat(*map(np.atleast_1d, sc.geo.xy(r, c)), "EPSG:32633")
        X2, Y2 = BM.lonlat_to_merc(lon, lat)
        assert abs(X2[0] - X) < 1.0 and abs(Y2[0] - Y) < 1.0         # metre-accurate inside a scene
    with pytest.raises(ValueError):
        BM.scene_affine(_scene(geo=False))


def test_fetch_basemap_mocked(monkeypatch):
    calls = []

    def tile(provider, z, x, y, url=None, timeout=10):
        calls.append((z, x, y))
        return np.full((256, 256, 3), 0.5, np.float32)
    monkeypatch.setattr(BM, "fetch_tile", tile)
    sc = _scene()
    fp = BM.footprint(BM.scene_affine(sc), 20, 30)
    ext = (fp[:, 0].min(), fp[:, 1].min(), fp[:, 0].max(), fp[:, 1].max())
    img, (x0, x1, y0, y1), z = BM.fetch_basemap(ext, "OpenStreetMap", 800)
    assert img.shape[0] % 256 == 0 and x0 <= ext[0] and x1 >= ext[2] and y0 <= ext[1] and y1 >= ext[3]
    assert 1 <= len(calls) <= 64 and all(c[0] == z for c in calls)

    def boom(*a, **k):
        raise OSError("offline")
    monkeypatch.setattr(BM, "fetch_tile", boom)
    with pytest.raises(ConnectionError):
        BM.fetch_basemap(ext, "OpenStreetMap", 800)


def test_corners_and_gcps():
    t = BM.affine_from_corners((15.0, 37.05), (15.03, 37.02), 100, 200, "EPSG:4326")
    g = GeoInfo("EPSG:4326", t)
    assert np.allclose(g.xy(0, 0, center=False), (15.0, 37.05)) and np.allclose(g.xy(100, 200, center=False), (15.03, 37.02))
    g0 = GeoInfo("EPSG:32633", (30.0, 0.0, 500000.0, 0.0, -30.0, 4100000.0))
    pts = [(c, r, *g0.xy(r, c)) for r, c in ((0, 0), (0, 90), (80, 5), (70, 80))]
    t2, rmse, res = BM.affine_from_gcps(pts)
    assert np.allclose(t2, g0.transform, atol=1e-6) and rmse < 1e-6 and len(res) == 4
    noisy = [(c, r, x + (5 if k == 0 else 0), y) for k, (c, r, x, y) in enumerate(pts)]
    assert BM.affine_from_gcps(noisy)[1] > 1
    with pytest.raises(ValueError):
        BM.affine_from_gcps(pts[:2])
    with pytest.raises(ValueError):
        BM.affine_from_gcps([(0, 0, 0, 0), (1, 1, 1, 1), (2, 2, 2, 2)])


# ================================================================== builder
def test_grids():
    assert np.allclose(B.grid_from_range(400, 450, 10), [400, 410, 420, 430, 440, 450])
    wl, fw = B.grid_from_text("; header\n0.45 0.01\n0.50 0.01\n0.55 0.012\n")
    assert np.allclose(wl, [450, 500, 550]) and np.allclose(fw, [10, 10, 12])
    wl, fw = B.grid_from_text("500, 450, 600")
    assert list(wl) == [450, 500, 600] and fw is None
    sc = _scene()
    sc.bad_file = np.zeros(sc.n_bands, bool)
    sc.bad_file[[0, 1]] = True
    wl, _ = B.grid_from_scene(sc, 1, 10, skip_bad=True)
    assert len(wl) == 8 and wl[0] == sc.wavelengths[2]
    with pytest.raises(ValueError):
        B.grid_from_range(400, 300, 10)


def test_resampling_no_extrapolation_no_gap_bridging():
    wl = np.r_[np.arange(400, 1300, 10.0), np.arange(1500, 2500, 10.0)]           # masked water gap
    y = wl / 1000.0
    grid = np.arange(350, 2600, 50.0)
    out = B.resample_linear(wl, y, grid)
    assert np.isnan(out[grid < 400]).all() and np.isnan(out[grid > 2490]).all()
    assert np.isnan(out[(grid > 1300) & (grid < 1500)]).all()
    ok = np.isfinite(out)
    assert np.allclose(out[ok], grid[ok] / 1000.0)
    g = B.resample_gaussian(wl, y, np.array([500.0, 1000.0]), np.array([20.0, 20.0]))
    assert np.allclose(g, [0.5, 1.0], atol=1e-3)
    v, cov, why = B.resample_one(wl, y, np.arange(300, 3000, 10.0), min_coverage=0.9)
    assert v is None and "covers only" in why


def test_build_rejections_and_items():
    grid = np.arange(400, 1001, 50.0)
    wl = np.linspace(400, 1000, 61)
    ins = [B.BuildInput("a", wl, np.full(61, 0.2), "x", unit="reflectance"),
           B.BuildInput("b", wl, np.full(61, 50.0), "x", unit="radiance"),
           B.BuildInput("a", wl, np.full(61, 0.2), "x", unit="reflectance"),
           B.BuildInput("c", np.linspace(1500, 2000, 10), np.ones(10), "y", unit="reflectance"),
           B.BuildInput("d", wl, np.full(61, 0.4), "z")]
    res = B.build(ins, grid, "linear", None, 0.5)
    assert [i.name for i, _y, _c in res.added] == ["a", "d"] and res.unit == "reflectance"
    why = {i.name: w for i, w in res.rejected}
    assert "never mixed" in why["b"] and "already" in why["a"] and "covers only" in why["c"]
    items = B.result_items(res, "linear")
    assert items[1]["unit"] == "reflectance" and items[0]["wavelengths"] == grid.tolist()


def test_export_library_formats(tmp_path):
    def e(name, wl, v):
        return {"material": {"name": name}, "spectrum": {"wavelengths": list(wl), "values": list(v),
                                                         "measurement_unit": "reflectance"}}
    wl = np.arange(400, 501, 20.0)
    ents = [e("a", wl, wl / 1000), e("b", wl, wl / 2000)]
    out = B.export_library(ents, str(tmp_path / "l.csv"), "csv")
    rows = list(csv.reader(open(out["files"][0])))
    assert rows[0] == ["wavelength_nm", "a", "b"] and len(rows) == 7 and not out["resampled"]
    out = B.export_library(ents, str(tmp_path / "l.sli"), "sli")
    assert os.path.exists(tmp_path / "l.sli") and "spectra names" in open(tmp_path / "l.hdr").read()
    data = np.fromfile(tmp_path / "l.sli", dtype=np.float32 if "data type = 4" in open(tmp_path / "l.hdr").read()
                       else np.float64)
    assert data.size == 12
    ents.append(e("c", np.arange(400, 501, 25.0), np.full(5, 0.3)))
    with pytest.raises(ValueError, match="different wavelength grids"):
        B.export_library(ents, str(tmp_path / "x.csv"), "csv")
    out = B.export_library(ents, str(tmp_path / "x.json"), "json", B.most_common_grid(ents))
    doc = json.load(open(tmp_path / "x.json"))
    assert out["resampled"] and len(doc["spectra"]) == 3 and doc["wavelengths"] == list(wl)


# ================================================================== database
def _urls():
    yield "sqlite"
    if os.getenv("HVVA_TEST_PG_URL"):
        yield "pg"


@pytest.fixture(params=list(_urls()))
def client(request, tmp_path):
    if request.param == "sqlite":
        c = DirectClient(f"sqlite:///{tmp_path / 'v34.db'}", init_schema=True)
    else:
        url = os.environ["HVVA_TEST_PG_URL"]
        db = Database(url)
        with db.engine.begin() as con:
            con.exec_driver_sql("DROP TABLE IF EXISTS spectrum_rois, rois, images, data_files, spectra, samples, "
                                "materials, datasets, libraries CASCADE")
        db.dispose()
        c = DirectClient(url, init_schema=True)
    yield c
    c.close()


def _it(name, wl, v, unit="reflectance", purity="ground_measured", acq=None):
    return {"name": name, "wavelengths": list(map(float, wl)), "values": list(map(float, v)), "unit": unit,
            "purity": purity, **({"acquisition_type": acq} if acq else {})}


def _seed(c):
    lib = c.create_library("Refl")
    wl = np.linspace(400, 2400, 41)
    wl2 = np.linspace(400, 1000, 13)
    c.save_spectra(lib["id"], "ds", [_it("grass", wl, 0.1 + wl / 5000), _it("soil", wl, 0.2 + wl / 10000),
                                     _it("pixel", wl2, 0.3 + wl2 / 8000, purity="image_pixel")])
    return lib, wl


def test_explicit_wavelengths_column(client):
    lib, wl = _seed(client)
    irregular = [400, 410, 425, 450, 500, 600]
    client.save_spectra(lib["id"], "ds", [_it("irr", irregular, np.full(6, 0.3))])
    for e in client.library_spectra(lib["id"]):
        s = e["spectrum"]
        assert s["wavelengths"] is not None and len(s["wavelengths"]) == len(s["values"])
        assert "wavelengths" not in s["metadata"]
    irr = next(e for e in client.library_spectra(lib["id"]) if e["material"]["name"] == "irr")["spectrum"]
    assert irr["wavelengths"] == irregular and irr["wavelength_step"] is None


def test_filters_unit_purity_covers(client):
    lib, _ = _seed(client)
    lid = lib["id"]
    names = lambda r: sorted(m["name"] for m in r["items"])
    assert names(client.list_materials(lid, None, None, 50, 0, None, None, "reflectance")) == ["grass", "pixel", "soil"]
    assert names(client.list_materials(lid, None, None, 50, 0, None, None, "radiance")) == []
    assert names(client.list_materials(lid, None, None, 50, 0, None, None, None, "image_pixel")) == ["pixel"]
    assert names(client.list_materials(lid, None, None, 50, 0, None, None, None, None, (500, 2000))) == ["grass", "soil"]
    assert [m["name"] for m in client.search_materials("pix", 10, None, None, None, (450, 900))["items"]] == ["pixel"]
    assert client.search_materials("pix", 10, None, None, None, (450, 1900))["items"] == []
    assert len(client.library_spectra(lid, None, None, None, "ground_measured")) == 2
    assert len(client.library_spectra(lid, None, None, None, None, (1000, 2400))) == 2


def test_match_rank_sam_and_rmse(client):
    lib, wl = _seed(client)
    ents = client.library_spectra(lib["id"])
    q = 0.1 + wl / 5000
    sam = repo.match_rank(q * 3, wl, ents, 3, "sam")               # scaled: SAM still finds grass
    assert sam[0][1]["name"] == "grass" and sam[0][0] < 1e-6
    rm = repo.match_rank(q, wl, ents, 3, "rmse")
    assert rm[0][1]["name"] == "grass" and rm[0][0] < 1e-6 and rm[1][0] > 0.01
    rm3 = repo.match_rank(q * 3, wl, ents, 3, "rmse")
    assert rm3[0][0] > 0.1                                          # RMSE sees the level difference
    with pytest.raises(ValueError):
        repo.match_rank(q, wl, ents, 3, "xyz")


def test_resample_library(client):
    lib, _ = _seed(client)
    res = client.resample_library(lib["id"], np.arange(500, 2001, 100.0), "Refl 100nm", "linear", None, 0.8)
    assert res["spectra"] == 2 and res["skipped"] == 1 and "pixel" in res["reasons"][0]
    new = res["library"]
    assert new["unit"] == "reflectance" and new["name"] == "Refl 100nm"
    ents = client.library_spectra(new["id"])
    assert all(len(e["spectrum"]["values"]) == 16 and e["spectrum"]["purity"] == "ground_measured" for e in ents)
    assert len(client.library_spectra(lib["id"])) == 3                 # original unchanged
    with pytest.raises(ValueError):
        client.resample_library(lib["id"], [500.0], "bad")


def test_images_table_and_roi_link(client):
    lib, _ = _seed(client)
    sc = make_demo_scenes(3)[2]
    info = image_info(sc)
    assert len(info["fingerprint"]) == 40 and info["rows"] == 100 and info["crs"] == "EPSG:32633"
    im = client.register_image(info)
    again = client.register_image(dict(info, name="renamed"))
    assert again["id"] == im["id"] and again["name"] == "renamed"
    assert len(client.list_images()) == 1 and client.get_image(im["id"])["bands"] == sc.n_bands
    r = sc.add_roi({"kind": "pixels", "pixels": [[1, 1], [5, 6]], "class": "pts"})
    out = client.save_roi_spectra(lib["id"], "rois", [R.roi_record(sc, x) for x in sc.rois], [])
    rois = client.list_rois(lib["id"])
    assert {x["geometry_type"] for x in rois} == {"rect", "pixels"}
    assert {x["image_id"] for x in rois} == {im["id"]}
    assert client.list_images()[0]["n_rois"] == len(sc.rois)
    client.delete_image(im["id"])
    assert client.list_images() == [] and all(x["image_id"] is None for x in client.list_rois(lib["id"]))
    with pytest.raises(Exception):
        client.get_image(im["id"])


def test_sqlite_migration_backfills_wavelengths(tmp_path):
    url = f"sqlite:///{tmp_path / 'old.db'}"
    c = DirectClient(url, init_schema=True)
    lib, _ = _seed(c)
    c.close()
    db = Database(url)
    with db.engine.begin() as con:                       # make it look like a v3.3 database
        rows = con.exec_driver_sql("SELECT id, json_array_length(\"values\") FROM spectra").fetchall()
        for sid, n in rows:                              # v3.3 kept an explicit grid in the metadata
            con.exec_driver_sql("UPDATE spectra SET metadata = json_set(metadata, '$.wavelengths', json(?)) "
                                "WHERE id = ?", (json.dumps([float(v) for v in np.linspace(400, 2400, 41)]), sid))
        con.exec_driver_sql("ALTER TABLE spectra DROP COLUMN wavelengths")
        con.exec_driver_sql("DROP TABLE images")
    db.dispose()
    db = Database(url)
    changed = db.migrate()
    assert {"spectra.wavelengths", "images"} <= set(changed), changed
    assert db.migrate() == []
    db.dispose()
    c = DirectClient(url)
    ents = c.library_spectra(lib["id"])
    assert all(e["spectrum"]["wavelengths"] is not None and len(e["spectrum"]["wavelengths"]) == len(e["spectrum"]["values"])
               for e in ents)
    pix = next(e for e in ents if e["material"]["name"] == "pixel")["spectrum"]   # 13 values: metadata copy
    assert np.allclose(pix["wavelengths"], np.linspace(400, 1000, 13))           # of another length is ignored
    assert c.list_images() == []
    c.close()


# ================================================================== HTTP API
def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def test_api_roundtrip(tmp_path):
    pytest.importorskip("fastapi")
    uvicorn = pytest.importorskip("uvicorn")
    from hvva.library.api import main as api
    from hvva.library.client import ApiClient
    url = f"sqlite:///{tmp_path / 'api.db'}"
    DirectClient(url, init_schema=True).close()
    api._db = Database(url)
    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(api.app, host="127.0.0.1", port=port, log_level="error"))
    th = threading.Thread(target=server.run, daemon=True)
    th.start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.05)
    try:
        c = ApiClient(f"http://127.0.0.1:{port}")
        lib, wl = _seed(c)
        assert [m["name"] for m in c.list_materials(lib["id"], None, None, 50, 0, None, None, None, "image_pixel")["items"]] == ["pixel"]
        assert len(c.library_spectra(lib["id"], None, None, None, None, (500, 2000))) == 2
        res = c.resample_library(lib["id"], np.arange(500, 1001, 50.0), "Api resampled")
        assert res["spectra"] == 3
        sc = make_demo_scenes(3)[2]
        im = c.register_image(image_info(sc))
        assert c.get_image(im["id"])["name"] == "scene_C" and len(c.list_images("scene")) == 1
        c.delete_image(im["id"])
        assert c.list_images() == []
    finally:
        server.should_exit = True
        th.join(5)
        api._db = None
