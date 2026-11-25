"""Point ground truth (ds3.gt_points), ground truth / model storage (migration 005) and the helpers of
ds3.analysis_ext that need no window: band matching and the 'unlike the training data' check."""
import numpy as np
import pandas as pd
import pytest

from ds3 import gt_points as G
from hvva.core.geo import GeoInfo

GEO = GeoInfo("EPSG:32644", (30.0, 0.0, 433590.0, 0.0, -30.0, 1842990.0))      # an EnMAP-like UTM 44N grid
ROWS, COLS = 40, 50


def _latlon(rows, cols):
    """lat/lon of pixel centres of GEO."""
    from rasterio.warp import transform
    x = 433590.0 + (np.asarray(cols) + 0.5) * 30.0
    y = 1842990.0 - (np.asarray(rows) + 0.5) * 30.0
    lon, lat = transform("EPSG:32644", "EPSG:4326", x.tolist(), y.tolist())
    return np.asarray(lat), np.asarray(lon)


def _tables(tmp_path):
    rng = np.random.default_rng(0)
    r = rng.integers(0, ROWS, 300)
    c = rng.integers(0, COLS, 300)
    lat, lon = _latlon(r, c)
    paths = []
    for name, scale in (("Cadmium", 0.4), ("Lead", 0.1)):
        p = tmp_path / f"abundance_{name}.xlsx"
        pd.DataFrame({"lat": lat, "lon": lon, "val": rng.random(300) * scale}).to_excel(p, index=False)
        paths.append(p)
    p = tmp_path / "outside.csv"                                         # far away: every point outside
    pd.DataFrame({"latitude": [10.0, 10.1], "longitude": [70.0, 70.1], "value": [0.2, 0.3]}).to_csv(p, index=False)
    return paths, p, (r, c)


def test_place_points_and_rules(tmp_path):
    paths, outside, (r, c) = _tables(tmp_path)
    assert G.layer_names(paths) == ["Cadmium", "Lead"]
    frames = []
    for p, n in zip(paths, G.layer_names(paths)):
        df = G.read_table(p)
        m = G.guess_columns(df.columns)
        assert (m["lat"], m["lon"], m["value"]) == ("lat", "lon", "val")
        fr, rep = G.place_points(df, m, GEO, ROWS, COLS, n)
        assert rep["placed"] == 300 and rep["outside"] == 0 and rep["max_offset_px"] < 1e-6
        assert (fr["row"].to_numpy() == r).all() and (fr["col"].to_numpy() == c).all()
        frames.append(fr)
    df = G.read_table(outside)
    fr, rep = G.place_points(df, G.guess_columns(df.columns), GEO, ROWS, COLS, "far")
    assert rep["placed"] == 0 and rep["outside"] == 2
    values, classes = G.stack_layers(frames, ROWS, COLS)
    assert set(values) == {"Cadmium", "Lead"} and not classes
    lab, names, counts = G.make_labels(values, classes, {"kind": "dominant", "threshold": 0.05})
    assert names == {1: "Cadmium", 2: "Lead", 3: "none"}
    covered = np.isfinite(values["Cadmium"])
    assert ((lab > 0) == covered).all() and sum(counts.values()) == covered.sum()
    lab, names, _ = G.make_labels(values, classes, {"kind": "levels", "layer": "Cadmium", "edges": [0.1, 0.2]})
    assert len(names) == 3 and (lab[covered] > 0).all()
    lab, names, _ = G.make_labels(values, classes, {"kind": "presence", "layer": "Lead", "threshold": 0.05})
    assert names == {1: "Lead", 2: "not Lead"}
    # stored points come back on the same pixels, by lat/lon
    recs = G.points_to_records(pd.concat(frames))
    back, _ = G.records_to_frame(recs, GEO, ROWS, COLS, same_image=False)
    assert len(back) == 600 and back["inside"].all()


def test_class_column_and_rowcol(tmp_path):
    p = tmp_path / "gt.csv"
    pd.DataFrame({"row": [1, 2, 3, 3], "col": [4, 5, 6, 6], "class": ["water", "soil", "water", "water"]}).to_csv(p)
    df = G.read_table(p)
    m = G.guess_columns(df.columns)
    fr, rep = G.place_points(df, m, None, ROWS, COLS, "gt")
    assert rep["placed"] == 4 and rep["duplicates"] == 1
    values, classes = G.stack_layers([fr], ROWS, COLS)
    lab, names, counts = G.make_labels(values, classes, {"kind": "class"})
    assert sorted(names.values()) == ["soil", "water"] and counts == {1: 1, 2: 2}


def test_lat_lon_needs_georeference(tmp_path):
    paths, _, _ = _tables(tmp_path)
    df = G.read_table(paths[0])
    with pytest.raises(ValueError, match="map projection"):
        G.place_points(df, G.guess_columns(df.columns), None, ROWS, COLS, "x")


def test_ground_truth_and_model_storage(tmp_path):
    from hvva.library.client import DirectClient
    c = DirectClient(f"sqlite:///{tmp_path / 'lib.db'}", init_schema=True)
    lib = c.create_library("GT")
    img = {"fingerprint": "f1", "name": "scene", "rows": ROWS, "cols": COLS, "bands": 4,
           "wavelengths": [500.0, 600.0, 700.0, 800.0], "crs": GEO.crs, "transform": list(GEO.transform)}
    pts = [{"layer": "Cd", "lat": 16.6, "lon": 80.4, "row": 1, "col": 2, "value": 0.3, "cls": None}] * 5
    g = c.save_ground_truth({"name": "metals", "library_id": lib["id"], "image": img, "layers": ["Cd"],
                             "rule": {"kind": "dominant"}, "class_names": {1: "Cd"}}, pts)
    assert c.list_ground_truths()[0]["image_name"] == "scene"
    assert len(c.get_ground_truth(g["id"])["points"]) == 5
    m = c.save_model({"name": "rf", "model_type": "Random Forest", "class_names": ["Cd", "none"], "n_bands": 4,
                      "wavelengths": img["wavelengths"], "image": img, "ground_truth_id": g["id"], "window": 1},
                     b"model-bytes")
    assert c.get_model(m["id"])["file"] == b"model-bytes"
    assert c.list_models(ground_truth_id=g["id"])[0]["name"] == "rf"
    c.delete_ground_truth(g["id"])
    assert c.list_models()[0]["ground_truth_id"] is None and c.list_ground_truths() == []


def test_band_matching_and_domain():
    from ds3.analysis_ext import bands_for_model, outside_domain, resample_cube
    cube = np.random.default_rng(1).random((5, 6, 10)).astype(np.float32)
    wl = np.linspace(400, 900, 10)
    same, note = bands_for_model(cube, wl, {"n_bands": 10, "wavelengths": wl.tolist()})
    assert same is cube and "same" in note
    half, note = bands_for_model(cube, wl, {"n_bands": 19, "wavelengths": np.linspace(400, 900, 19).tolist()})
    assert half.shape == (5, 6, 19) and np.allclose(half[:, :, ::2], cube, atol=1e-6)
    with pytest.raises(ValueError, match="not covered"):
        bands_for_model(cube, wl, {"n_bands": 3, "wavelengths": [1500.0, 1600.0, 1700.0]})
    assert np.allclose(resample_cube(cube, wl, wl), cube)
    dom = {"classes": [{"mean": np.zeros(2, np.float32), "inv_cov": np.eye(2, dtype=np.float32), "thr": 9.0}]}
    far = outside_domain(np.array([[0.0, 0.0], [1.0, 2.0], [5.0, 0.0]]), dom)
    assert far.tolist() == [False, False, True]
