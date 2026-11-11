"""ROI analysis: geometry / masks / pooling (hvva.core.rois) and ROIs in the library database (migration 002).

Database tests run on SQLite; set HVVA_TEST_PG_URL (see test_library_db.py) to run them on PostgreSQL too.
"""
import json
import os

import numpy as np
import pytest

from hvva.core import rois as R
from hvva.core.geo import GeoInfo
from hvva.core.scene import Scene, load_scene, save_scene
from hvva.core.synthetic import make_demo_scenes
from hvva.library.client import DirectClient
from hvva.library.database import Database


def _scene(name="A", h=20, w=30, b=5, geo=None, value=None):
    rng = np.random.default_rng(1)
    cube = rng.random((h, w, b)).astype(np.float32) if value is None else np.full((h, w, b), value, np.float32)
    return Scene(name, cube, np.linspace(400, 800, b), geo)


# ------------------------------------------------------------------ core
def test_masks_match_viewer_conventions():
    rect = R.normalize_roi({"class": "a", "rect": (2, 3, 5, 7)})
    assert rect["kind"] == "rect" and rect["id"] and rect["name"]
    m = R.roi_mask(rect, 10, 10)
    assert m.sum() == 3 * 4 and m[2, 3] and m[4, 6] and not m[5, 6]
    tri = R.normalize_roi({"class": "b", "vertices": [(0, 0), (9, 0), (0, 9)]})
    assert tri["kind"] == "polygon"
    pm = R.roi_mask(tri, 10, 10)
    assert pm[1, 1] and not pm[8, 8] and 30 < pm.sum() < 60
    with pytest.raises(ValueError):
        R.normalize_roi({"class": "c", "vertices": [(0, 0), (1, 1)]})


def test_scene_rois_backward_compatible_and_persisted(tmp_path):
    sc = _scene()
    sc.rois = []
    old = Scene("old", sc.cube, sc.wavelengths, rois=[{"class": "soil", "rect": (0, 0, 4, 4)}])
    assert old.rois[0]["kind"] == "rect" and old.rois[0]["id"]
    r1 = sc.add_roi({"class": "veg", "rect": (0, 0, 5, 5)})
    r2 = sc.add_roi({"class": "veg", "kind": "polygon", "vertices": [(10, 10), (20, 10), (20, 18), (10, 18)]})
    sc.add_roi({"class": "soil", "rect": (15, 0, 20, 5)})
    assert r1["name"] == "ROI 1" and r2["name"] == "ROI 2"
    masks = sc.class_masks()
    assert set(masks) == {"veg", "soil"}
    assert masks["veg"].sum() == 25 + R.roi_mask(r2, 20, 30).sum()
    p = str(tmp_path / "s.npz")
    save_scene(sc, p)
    back = load_scene(p)
    assert [(r["id"], r["kind"], r["class"]) for r in back.rois] == [(r["id"], r["kind"], r["class"]) for r in sc.rois]
    assert np.array_equal(back.class_masks()["veg"], masks["veg"])
    demo = make_demo_scenes()[2]
    assert len(demo.rois) == 3 and all(r["id"] for r in demo.rois)


def test_pool_classes_across_images_and_grids():
    a, b = _scene("A", value=1.0), _scene("B", value=3.0)
    b.wavelengths = np.linspace(420, 820, 5)                        # different grid -> resampled
    ra = a.add_roi({"class": "veg", "rect": (0, 0, 2, 2)})           # 4 px
    rb = b.add_roi({"class": "veg", "rect": (0, 0, 2, 6)})           # 12 px
    a.add_roi({"class": "veg", "rect": (0, 0, 1, 1)})                # overlaps ra: counted once
    p = R.pool_classes([(a, r) for r in a.rois] + [(b, rb)])
    veg = p["veg"]
    assert veg["n_pixels"] == 16 and veg["n_scenes"] == 2 and veg["n_rois"] == 3
    assert np.allclose(veg["mean"], (4 * 1 + 12 * 3) / 16)
    assert veg["wavelengths"][0] >= 420 and veg["wavelengths"][-1] <= 800
    assert ra["id"] in veg["roi_ids"]


def test_ds3_json_roundtrip_viewer_format():
    sc = _scene()
    sc.add_roi({"class": "veg", "rect": (2, 3, 5, 7)})
    sc.add_roi({"class": "soil", "vertices": [(1, 1), (8, 1), (8, 8)]})
    data = json.loads(json.dumps(R.to_ds3_json([(sc, r) for r in sc.rois])))
    assert data["format"] == "DS3-ROIs" and data["image_shape"] == [20, 30]
    assert data["rois"][0]["rect"] == [2, 4, 3, 6]                  # viewer: ymin, ymax, xmin, xmax inclusive
    back, skipped = R.from_ds3_json(data, 20, 30)
    assert skipped == 0 and [s for s, _ in back] == ["A", "A"]
    assert back[0][1]["rect"] == (2, 3, 5, 7) and back[1][1]["kind"] == "polygon"
    # a file written by the viewer (version 1, no ids / scene)
    v1 = {"format": "DS3-ROIs", "version": 1, "image_shape": [20, 30],
          "rois": [{"name": "ROI 1", "class": "Class 1", "kind": "rect", "rect": [0, 100, 0, 3]}]}
    (scene_name, roi), = R.from_ds3_json(v1, 20, 30)[0]
    assert scene_name is None and roi["rect"] == (0, 0, 20, 4)       # clipped to the image


def test_map_geometry_transfers_between_images():
    g1 = GeoInfo("EPSG:32633", (10.0, 0.0, 500000.0, 0.0, -10.0, 4000000.0))
    g2 = GeoInfo("EPSG:32633", (10.0, 0.0, 500050.0, 0.0, -10.0, 3999950.0))   # shifted by 5 px / 5 rows
    a, b = _scene("A", geo=g1), _scene("B", geo=g2)
    r = a.add_roi({"class": "veg", "rect": (10, 10, 14, 16)})
    rec = R.roi_record(a, r)
    rec_db = {"name": r["name"], "class_name": "veg", "geometry_type": "rect", "geometry": rec["geometry"],
              "geometry_map": rec["geometry_map"], "image_rows": 20, "image_cols": 30, "scene_name": "A",
              "roi_uid": r["id"]}
    on_b = R.roi_from_record(rec_db, b)
    assert on_b["rect"] == (5, 5, 9, 11) and on_b["id"] != r["id"]
    same = R.roi_from_record(rec_db, a)
    assert same["id"] == r["id"] and same["rect"] == r["rect"]
    poly = a.add_roi({"class": "veg", "vertices": [(10, 10), (15, 10), (15, 14)]})
    pr = R.roi_from_record(dict(rec_db, geometry_type="polygon", geometry=R.roi_record(a, poly)["geometry"],
                                geometry_map=R.roi_record(a, poly)["geometry_map"]), b)
    assert np.allclose(pr["vertices"][0], (5, 5))


def test_anomaly_map():
    sc = _scene(value=1.0)
    sc.cube[3, 4] = 5.0
    sc.cube[0, 0] = np.nan
    a = R.anomaly_map(sc.cube)
    assert np.isnan(a[0, 0]) and np.nanargmax(a) == 3 * 30 + 4


# ------------------------------------------------------------------ database
def _urls():
    yield "sqlite"
    if os.getenv("HVVA_TEST_PG_URL"):
        yield "pg"


@pytest.fixture(params=list(_urls()))
def client(request, tmp_path):
    if request.param == "sqlite":
        c = DirectClient(f"sqlite:///{tmp_path / 'lib.db'}", init_schema=True)
    else:
        url = os.environ["HVVA_TEST_PG_URL"]
        db = Database(url)
        with db.engine.begin() as con:
            con.exec_driver_sql("DROP TABLE IF EXISTS spectrum_rois, rois, data_files, spectra, samples, materials, "
                                "datasets, libraries CASCADE")
        db.dispose()
        c = DirectClient(url, init_schema=True)
    yield c
    c.close()


def _two_scenes():
    a, b = _scene("scene_A"), _scene("scene_B")
    a.add_roi({"class": "vegetation", "rect": (0, 0, 4, 4)})
    a.add_roi({"class": "vegetation", "vertices": [(10, 10), (16, 10), (16, 16), (10, 16)]})
    a.add_roi({"class": "soil", "rect": (15, 20, 19, 29)})
    b.add_roi({"class": "vegetation", "rect": (5, 5, 9, 9)})
    return a, b, [(a, r) for r in a.rois] + [(b, r) for r in b.rois]


@pytest.mark.parametrize("mode", ["class", "roi", "pixels"])
def test_save_rois_class_by_class(client, mode):
    a, b, items = _two_scenes()
    lib = client.create_library("ROI lib")
    recs, specs = R.library_items(items, mode, max_pixels=5)
    res = client.save_roi_spectra(lib["id"], "Field ROIs", recs, specs, category="Field", acquisition="airborne")
    n_expected = {"class": 2, "roi": 4, "pixels": 20}[mode]
    assert res["rois"] == 4 and res["spectra"] == n_expected and res["classes"] == ["soil", "vegetation"]
    again = client.save_roi_spectra(lib["id"], "Field ROIs", recs, specs)
    assert again["spectra"] == 0 and again["skipped"] == n_expected       # idempotent

    rois = client.list_rois(lib["id"])
    assert len(rois) == 4 and {r["scene_name"] for r in rois} == {"scene_A", "scene_B"}
    assert {r["geometry_type"] for r in rois} == {"rect", "polygon"}
    assert all(r["n_spectra"] >= 1 for r in rois)
    assert len(client.list_rois(lib["id"], scene_name="scene_B")) == 1
    classes = {c["class_name"]: c for c in client.roi_classes(lib["id"])}
    assert classes["vegetation"]["n_rois"] == 3 and classes["vegetation"]["n_scenes"] == 2
    assert classes["soil"]["n_spectra"] == {"class": 1, "roi": 1, "pixels": 5}[mode]

    veg = client.roi_spectra(library_id=lib["id"], class_name="vegetation")
    assert all(e["spectrum"]["roi_class"] == "vegetation" for e in veg)
    assert all(e["spectrum"]["acquisition_type"] == "airborne" for e in veg)
    assert all(e["material"]["category"] == "Field" for e in veg)
    if mode == "class":
        (e,) = veg
        assert len(e["roi_ids"]) == 3 and e["spectrum"]["roi_id"] is None
        assert e["spectrum"]["metadata"]["n_scenes"] == 2
    else:
        assert all(e["spectrum"]["roi_id"] == e["roi_ids"][0] for e in veg)
    mats = client.list_materials(lib["id"])["items"]
    assert sorted(m["name"] for m in mats) == ["soil", "vegetation"]       # class = material, ROIs = samples

    # provenance of one spectrum
    info = client.spectrum_roi_info(veg[0]["spectrum"]["id"])
    assert info and all(r["class_name"] == "vegetation" for r in info)

    # load back onto the image
    back = [R.roi_from_record(r, a) for r in client.list_rois(lib["id"], scene_name="scene_A")]
    assert sorted(x["id"] for x in back) == sorted(r["id"] for r in a.rois)


def test_roi_update_and_delete(client):
    a, b, items = _two_scenes()
    lib = client.create_library("ROI lib")
    recs, specs = R.library_items(items, "roi")
    client.save_roi_spectra(lib["id"], "ROIs", recs, specs)
    # move one ROI and save again: same rois row, new spectrum
    a.rois[0]["rect"] = (1, 1, 5, 5)
    recs2, specs2 = R.library_items([(a, a.rois[0])], "roi")
    res = client.save_roi_spectra(lib["id"], "ROIs", recs2, specs2)
    assert res["spectra"] == 1
    rows = client.list_rois(lib["id"])
    assert len(rows) == 4
    moved = next(r for r in rows if r["roi_uid"] == a.rois[0]["id"])
    assert moved["geometry"]["rect"] == [1, 1, 5, 5] and moved["n_spectra"] == 2
    out = client.delete_roi(moved["id"], delete_spectra=True)
    assert out["deleted_spectra"] == 2
    assert len(client.list_rois(lib["id"])) == 3
    other = next(r for r in client.list_rois(lib["id"]) if r["class_name"] == "soil")
    client.delete_roi(other["id"])                                    # spectra kept, roi_id -> NULL
    soil = client.roi_spectra(library_id=lib["id"], class_name="soil")
    assert len(soil) == 1 and soil[0]["spectrum"]["roi_id"] is None
    client.delete_library(lib["id"])
    assert client.list_rois() == []


def test_class_pooling_across_libraries_is_separate(client):
    a, b, items = _two_scenes()
    l1, l2 = client.create_library("L1"), client.create_library("L2")
    recs, specs = R.library_items(items, "class")
    client.save_roi_spectra(l1["id"], "ROIs", recs, specs)
    client.save_roi_spectra(l2["id"], "ROIs", recs, specs)                 # same ROI ids, other library: allowed
    assert len(client.list_rois()) == 8


def test_migration_002_on_sqlite(tmp_path):
    """A v3.0 database (no ROI tables) gets them on connect, data intact."""
    url = f"sqlite:///{tmp_path / 'v3.db'}"
    c = DirectClient(url, init_schema=True)
    lib = c.create_library("Old")
    c.close()
    from sqlalchemy import create_engine
    eng = create_engine(url)
    with eng.begin() as con:
        con.exec_driver_sql("DROP TABLE spectrum_rois")
        con.exec_driver_sql("DROP TABLE rois")
        con.exec_driver_sql("CREATE TABLE spectra_v3 AS SELECT id, sample_id, wavelength_start, wavelength_end, "
                            "wavelength_step, measurement_unit, \"values\", metadata, acquisition_type, created_at "
                            "FROM spectra")
        con.exec_driver_sql("DROP TABLE spectra")
        con.exec_driver_sql("ALTER TABLE spectra_v3 RENAME TO spectra")
    eng.dispose()
    db = Database(url)
    assert set(db.migrate()) == {"spectra.roi_id", "spectra.roi_class", "rois", "spectrum_rois"}
    assert db.migrate() == []
    db.dispose()
    c = DirectClient(url)
    a, b, items = _two_scenes()
    recs, specs = R.library_items(items, "class")
    assert c.save_roi_spectra(lib["id"], "ROIs", recs, specs)["spectra"] == 2
    c.close()
