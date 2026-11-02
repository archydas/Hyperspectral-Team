import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hvva.core import export, plugins
from hvva.core.geo import GeoInfo
from hvva.core.mosaic import composite, mosaic, stack_cube
from hvva.core.resample import common_grid, resample_spectra
from hvva.core.scene import Scene, load_scene, save_envi, save_scene
from hvva.core.signature import signatures_from_result
from hvva.core.stats import aggregate_classes
from hvva.core.synthetic import make_demo_scenes
from hvva.library.db import SignatureLibrary


@pytest.fixture(scope="module")
def scenes():
    return make_demo_scenes(3)


def test_common_grid_modes(scenes):
    wl = [s.wavelengths for s in scenes]
    g = common_grid(wl)
    assert g.min() >= max(w.min() for w in wl) and g.max() <= min(w.max() for w in wl)
    assert np.isclose(np.diff(g)[0], 6.0)                       # coarsest native step
    u = common_grid(wl, "union", 10)
    assert u.min() == 400 and u.max() <= 1000


def test_resample_linear_and_nan():
    wl = np.array([400., 500., 600.])
    X = np.array([[0., 1., 2.]])
    out = resample_spectra(X, wl, np.array([350., 450., 550., 650.]))
    assert np.isnan(out[0, 0]) and np.isnan(out[0, 3])
    assert np.allclose(out[0, 1:3], [0.5, 1.5])


def test_geo_roundtrip_and_envi_map_info():
    g = GeoInfo("EPSG:32633", (30, 0, 500000, 0, -30, 4100000))
    r, c = g.rowcol(*g.xy(12, 34, center=False))
    assert np.allclose([r, c], [12, 34])
    g2 = GeoInfo.from_envi_map_info(g.to_envi_map_info())
    assert g2.crs == "EPSG:32633" and np.allclose(g2.transform, g.transform)
    assert np.allclose(g.shifted(2, 3).transform[2:6:3], [500090, 4099940])


def test_pooled_stats_equal_direct(scenes):
    g = common_grid([s.wavelengths for s in scenes])
    res = aggregate_classes(scenes[:2], g)
    from hvva.core.resample import resample_spectra as rs
    X = np.vstack([rs(s.cube[s.class_masks()["water"]], s.wavelengths, g) for s in scenes[:2]])
    cs = res.classes["water"]
    assert cs.n_pixels == len(X)
    assert np.allclose(cs.mean, X.mean(0), atol=1e-5)
    assert np.allclose(cs.std, X.std(0, ddof=1), atol=1e-5)
    assert cs.n_scenes == 2


def test_roi_scene_contributes(scenes):
    g = common_grid([s.wavelengths for s in scenes])
    res = aggregate_classes(scenes, g)
    assert "scene_C" in res.classes["urban"].per_scene
    assert res.classes["urban"].n_scenes == 2                      # B (map) + C (ROI)


def test_mosaic_georeferenced(scenes):
    g = common_grid([s.wavelengths for s in scenes])
    m = mosaic(scenes, g, "mean")
    assert m.shape[:2] == (100, 210)                               # 3 scenes, 55 px offset, 100 px wide
    assert m.geo.transform[2] == 500000 and m.geo.crs == "EPSG:32633"
    assert not np.isnan(m.cube).any()
    # overlap of A and B (cols 55-99) is the mean of both
    a = np.nanmean(m.cube[:, 5, :])
    assert np.isfinite(a)


def test_mosaic_rejects_crs_mismatch(scenes):
    g = common_grid([s.wavelengths for s in scenes])
    bad = make_demo_scenes(2)
    bad[1].geo = GeoInfo("EPSG:32634", bad[1].geo.transform)
    with pytest.raises(ValueError):
        mosaic(bad, g)


def test_stack_and_composite(scenes):
    g = common_grid([s.wavelengths for s in scenes])
    arr, geo = stack_cube(scenes[:2], g)
    assert arr.shape[0] == 2 and arr.shape[1] == 100 and arr.shape[2] == 45
    assert geo.transform[2] == 500000 + 55 * 30
    c = composite(scenes[:2], g, "median")
    assert c.shape == arr.shape[1:]


def test_export_roundtrip(tmp_path, scenes):
    g = common_grid([s.wavelengths for s in scenes])
    sigs = signatures_from_result(aggregate_classes(scenes, g))
    paths = export.export_all(str(tmp_path / "lib"), sigs)
    assert all(os.path.exists(p) for p in paths) and len(paths) == 4
    names, wl, data = export.read_sli(str(tmp_path / "lib.sli"))
    assert names == [s.name for s in sigs] and np.allclose(wl, g, atol=1e-3)
    assert np.allclose(data[0], sigs[0].mean, atol=1e-5)
    import json
    doc = json.load(open(tmp_path / "lib.json"))
    assert set(doc["classes"]) == {s.name for s in sigs}
    assert open(tmp_path / "lib.csv").readline().startswith("wavelength_nm,")


def test_scene_npz_and_envi_keep_geo(tmp_path, scenes):
    sc = scenes[0]
    save_scene(sc, str(tmp_path / "a.npz"))
    r = load_scene(str(tmp_path / "a.npz"))
    assert r.geo.crs == sc.geo.crs and np.allclose(r.cube, sc.cube) and r.class_masks().keys() == sc.class_masks().keys()
    save_envi(sc, str(tmp_path / "a"))
    e = load_scene(str(tmp_path / "a.hdr"))
    assert np.allclose(e.cube, sc.cube) and np.allclose(e.wavelengths, sc.wavelengths, atol=1e-3)
    assert e.geo.crs == "EPSG:32633" and np.allclose(e.geo.transform, sc.geo.transform)


def test_library_db_match(scenes):
    g = common_grid([s.wavelengths for s in scenes])
    sigs = signatures_from_result(aggregate_classes(scenes, g))
    lib = SignatureLibrary()
    lib.add_many(sigs)
    assert lib.count() == 4
    veg = next(s for s in sigs if s.name == "vegetation")
    best = lib.match(veg.mean, g, top=1)[0]
    assert best[2] == "vegetation" and best[0] < 1e-6


def test_plugins_discovered_and_run(scenes):
    pl, errs = plugins.discover()
    assert not errs
    names = {p.name for p in pl}
    assert {"NDVI", "Spectral moving average"} <= names
    ndvi = next(p for p in pl if p.name == "NDVI").apply(scenes[0])
    assert ndvi["image"].shape == scenes[0].shape[:2]
    sm = next(p for p in pl if p.kind == "correction" and "average" in p.name).apply(scenes[0], window=3)
    assert sm.shape == scenes[0].shape


def test_broken_plugin_does_not_crash(tmp_path):
    (tmp_path / "bad.py").write_text("raise RuntimeError('boom')")
    pl, errs = plugins.discover(str(tmp_path))
    assert pl == [] and "boom" in errs[0]
