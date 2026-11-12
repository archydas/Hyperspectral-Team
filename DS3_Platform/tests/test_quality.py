"""Data input quality (v3.3): bad-band mask, value units (one per library), purity tag, zip products and
ancillary files.  Database tests run on SQLite, and on PostgreSQL too when HVVA_TEST_PG_URL is set."""
import os
import zipfile

import numpy as np
import pytest

from hvva.core import quality as Q
from hvva.core import rois as R
from hvva.core.scene import Scene, load_envi, load_scene, save_envi, save_scene
from hvva.core.signature import signatures_from_result
from hvva.core.stats import aggregate_classes
from hvva.core.synthetic import make_demo_scenes
from hvva.library import repository as repo
from hvva.library.client import DirectClient
from hvva.library.database import Database
from hvva.library.formats import ReadOptions
from hvva.library.ingest import import_path, preview_path


def _scene(name="s", wl=None, value=0.3, unit="reflectance", h=12, w=10):
    wl = np.linspace(400, 2500, 211) if wl is None else np.asarray(wl, float)
    cube = np.full((h, w, len(wl)), value, np.float32)
    return Scene(name, cube, wl, unit=unit)


# ------------------------------------------------------------------ bad bands
def test_windows_and_spans():
    assert Q.parse_windows("1340-1460, 1790 - 1960") == ((1340, 1460), (1790, 1960))
    assert Q.parse_windows("") == ()
    with pytest.raises(ValueError):
        Q.parse_windows("1340..1460")
    wl = np.arange(1300, 1500, 10.0)
    m = Q.water_mask(wl)
    assert m.sum() == 13 and not m[0]
    assert Q.spans(wl, m) == [(1340.0, 1460.0)]


def test_scene_bad_bands_flag_not_delete():
    sc = _scene()
    sc.cube[:, :, 5] = np.nan                                     # a band without data
    sc.bad_file = np.zeros(sc.n_bands, bool)
    sc.bad_file[0] = True                                         # flagged by the file
    parts = sc.bad_band_parts()
    assert parts["file"].sum() == 1 and parts["empty"].sum() == 1 and parts["water"].sum() > 0
    bad = sc.bad_bands
    assert bad[0] and bad[5] and bad[np.argmin(abs(sc.wavelengths - 1400))]
    assert np.isfinite(sc.cube[0, 0, 0])                          # cube untouched
    spec = sc.spectrum_at(3, 3)
    assert np.isnan(spec[bad]).all() and np.isfinite(spec[~bad]).all()
    assert "bad bands" in sc.bad_band_summary()
    sc.mask_water = False
    assert not sc.bad_bands[np.argmin(abs(sc.wavelengths - 1400))]
    sc.water_windows = ((900, 1000),)
    sc.mask_water = True
    assert sc.bad_bands[np.argmin(abs(sc.wavelengths - 950))]


def test_masks_reach_rois_aggregation_and_files(tmp_path):
    sc = _scene("A")
    sc.add_roi({"class": "veg", "rect": (0, 0, 4, 4)})
    water = sc.bad_bands
    st = R.roi_stats(sc.cube, sc.rois[0], sc.bad_bands)
    assert np.isnan(st["mean"][water]).all() and np.isfinite(st["mean"][~water]).all()
    p = R.pool_classes([(sc, sc.rois[0])])["veg"]
    assert np.isnan(p["mean"][water]).all()
    _, items = R.library_items([(sc, sc.rois[0])], "class")
    assert sum(v is None for v in items[0]["values"]) == int(water.sum())
    res = aggregate_classes([sc], sc.wavelengths)
    assert np.isnan(res.classes["veg"].mean[water]).all()
    # .npz keeps the settings
    sc.mask_empty = False
    sc.water_windows = ((1350, 1450),)
    p = str(tmp_path / "a.npz")
    save_scene(sc, p)
    back = load_scene(p)
    assert back.water_windows == ((1350, 1450),) and back.mask_empty is False and back.unit == "reflectance"
    # ENVI writes bbl + unit and reads them back
    save_envi(sc, str(tmp_path / "e"))
    e = load_envi(str(tmp_path / "e.hdr"))
    assert e.unit == "reflectance" and e.bad_file is not None and e.bad_file.sum() == sc.bad_bands.sum()


def test_envi_header_unit_and_scale(tmp_path):
    cube = (np.full((4, 5, 3), 2500)).astype("<i2")
    cube.transpose(2, 0, 1).tofile(tmp_path / "ang20200101t000000_rfl_v2")
    (tmp_path / "ang20200101t000000_rfl_v2.hdr").write_text(
        "ENVI\nsamples = 5\nlines = 4\nbands = 3\ndata type = 2\ninterleave = bsq\nbyte order = 0\n"
        "reflectance scale factor = 10000\nbbl = {1, 0, 1}\nwavelength = {450, 1400, 2000}\n")
    sc = load_scene(str(tmp_path / "ang20200101t000000_rfl_v2.hdr"))
    assert sc.unit == "reflectance" and np.allclose(np.nanmax(sc.cube), 0.25)
    assert sc.bad_file.tolist() == [False, True, False]


def test_unit_detection_and_mixing():
    assert Q.normalize_unit("Reflectance (%)") == "reflectance"
    assert Q.normalize_unit("uW/cm^2/sr/nm") == "radiance"
    assert Q.normalize_unit("counts") == "dn"
    assert Q.unit_from_name("ang2019_rdn_v2m2_img") == "radiance"
    assert Q.unit_from_values(np.full((5, 5, 3), 0.4, np.float32))[0] == "reflectance"
    assert Q.unit_from_values(np.full((5, 5, 3), 4000, np.float32))[0] == "unknown"
    a, b = _scene("A"), _scene("B", unit="radiance")
    for s in (a, b):
        s.add_roi({"class": "veg", "rect": (0, 0, 4, 4)})
    with pytest.raises(ValueError, match="never"):
        aggregate_classes([a, b], a.wavelengths)
    with pytest.raises(ValueError, match="never"):
        R.pool_classes([(a, a.rois[0]), (b, b.rois[0])])
    c = _scene("C", unit="unknown")
    c.add_roi({"class": "veg", "rect": (0, 0, 4, 4)})
    res = aggregate_classes([a, c], a.wavelengths)                     # unknown mixes (and is reported)
    sig = signatures_from_result(res)[0]
    assert res.unit == "reflectance" and sig.unit == "reflectance" and sig.purity == "image_pixel"
    demo = make_demo_scenes()
    assert all(s.synthetic and s.unit == "reflectance" for s in demo)
    assert aggregate_classes(demo[:2], demo[0].wavelengths).purity == "synthetic"


# ------------------------------------------------------------------ zip / ancillary
def _enmap_files(folder, prefix="ENMAP01-____L2A-DT0000000001_20250101T000000Z_001_V010506_20260101T000000Z-"):
    tifffile = pytest.importorskip("tifffile")
    wl = np.concatenate([np.linspace(418.4, 993.0, 91), np.linspace(902.5, 2445.5, 133)])
    refl = 0.05 + 0.4 / (1 + np.exp(-(wl - 720) / 15))
    cube = np.broadcast_to((refl * 10000).astype(np.int16), (12, 14, 224)).copy()
    cube[:2] = -32768
    xml = "".join(f"<bandID number='{i}'><wavelengthCenterOfBand>{c:.3f}</wavelengthCenterOfBand>"
                  f"<FWHMOfBand>8</FWHMOfBand><GainOfBand>1</GainOfBand><OffsetOfBand>0</OffsetOfBand></bandID>"
                  for i, c in enumerate(wl, 1))
    (folder / (prefix + "METADATA.XML")).write_text(f"<level_X><specific><bandCharacterisation>{xml}"
                                                    "</bandCharacterisation></specific></level_X>")
    tifffile.imwrite(str(folder / (prefix + "SPECTRAL_IMAGE.TIF")), cube.transpose(2, 0, 1),
                     photometric="minisblack", planarconfig="separate")
    tifffile.imwrite(str(folder / (prefix + "QL_VNIR.TIF")), np.zeros((12, 14, 3), np.uint8), photometric="rgb")
    tifffile.imwrite(str(folder / (prefix + "QL_QUALITY_CLOUD.TIF")), np.zeros((12, 14), np.uint8))
    return prefix


@pytest.fixture
def enmap_zip(tmp_path, monkeypatch):
    monkeypatch.setattr(Q, "CACHE_DIR", str(tmp_path / "cache"))
    src = tmp_path / "product"
    src.mkdir()
    prefix = _enmap_files(src)
    zp = tmp_path / "ENMAP_L2A_product.zip"
    with zipfile.ZipFile(zp, "w") as z:
        for f in src.iterdir():
            z.write(f, f"{prefix[:-1]}/{f.name}")
    return zp


def test_enmap_zip_opens_the_spectral_image(enmap_zip, tmp_path):
    sc = load_scene(str(enmap_zip))
    assert sc.n_bands == 209 and sc.unit == "reflectance" and "unpacked" in sc.unit_note
    assert np.isnan(sc.cube[0, 0]).all() and 0 < np.nanmax(sc.cube) < 0.5
    assert sc.bad_bands.sum() > 10                                  # water windows flagged
    again = load_scene(str(enmap_zip))                              # cached copy reused
    assert again.n_bands == 209
    # nested: the delivery zip holds the product zip
    outer = tmp_path / "delivery.zip"
    with zipfile.ZipFile(outer, "w") as z:
        z.write(enmap_zip, "ENMAP01-PRODUCT.ZIP")
    assert load_scene(str(outer)).n_bands == 209
    empty = tmp_path / "nothing.zip"
    with zipfile.ZipFile(empty, "w") as z:
        z.writestr("readme.txt", "hello")
    with pytest.raises(ValueError, match="no spectral image"):
        load_scene(str(empty))


def test_zip_member_names_cannot_escape(tmp_path, monkeypatch):
    monkeypatch.setattr(Q, "CACHE_DIR", str(tmp_path / "cache"))
    cube = np.zeros((3, 4, 5), np.float32)
    np.save(tmp_path / "x.npy", cube)
    zp = tmp_path / "evil.zip"
    with zipfile.ZipFile(zp, "w") as z:
        z.write(tmp_path / "x.npy", "../../outside/x.npy")
    real, _ = Q.resolve_input(str(zp))
    assert os.path.dirname(real).startswith(str(tmp_path / "cache")) and not (tmp_path / "outside").exists()


def test_ancillary_files_refused(tmp_path):
    for name in ("ang20190722t123456_rdn_v2m2_glt", "ang20190722t123456_rdn_v2m2_igm.hdr",
                 "ang20190722t123456_rdn_v2m2_obs_ort", "ang20190722t123456_corr_v2m2_h2o"):
        assert Q.ancillary_reason(name), name
    assert Q.ancillary_reason("ang20190722t123456_rdn_v2m2_img") is None
    assert Q.ancillary_reason("ENMAP01-xx-QL_VNIR.TIF") and Q.ancillary_reason("ENMAP01-xx-QL_QUALITY_CLOUD.TIF")
    assert Q.ancillary_reason("ENMAP01-xx-SPECTRAL_IMAGE.TIF") is None
    hdr = tmp_path / "ang20190722t123456_rdn_v2m2_glt.hdr"
    hdr.write_text("ENVI\nsamples = 2\nlines = 2\nbands = 2\ndata type = 4\ninterleave = bsq\n")
    np.zeros(8, "<f4").tofile(tmp_path / "ang20190722t123456_rdn_v2m2_glt")
    with pytest.raises(ValueError, match="geometry"):
        load_scene(str(hdr))


# ------------------------------------------------------------------ database: units + purity
def _urls():
    yield "sqlite"
    if os.getenv("HVVA_TEST_PG_URL"):
        yield "pg"


@pytest.fixture(params=list(_urls()))
def client(request, tmp_path):
    if request.param == "sqlite":
        c = DirectClient(f"sqlite:///{tmp_path / 'q.db'}", init_schema=True)
    else:
        url = os.environ["HVVA_TEST_PG_URL"]
        db = Database(url)
        with db.engine.begin() as con:
            con.exec_driver_sql("DROP TABLE IF EXISTS spectrum_rois, rois, images, data_files, spectra, samples, materials, "
                                "datasets, libraries CASCADE")
        db.dispose()
        c = DirectClient(url, init_schema=True)
    yield c
    c.close()


def _item(name, unit, purity=None):
    wl = np.linspace(400, 2400, 21)
    return {"name": name, "wavelengths": wl.tolist(), "values": np.full(21, 0.3).tolist(), "unit": unit,
            "purity": purity}


def test_one_unit_per_library(client):
    lib = client.create_library("Units")
    client.save_spectra(lib["id"], "ds", [_item("a", "reflectance", "ground_measured")])
    assert client.library_unit(lib["id"]) == "reflectance"
    assert next(l for l in client.list_libraries() if l["id"] == lib["id"])["unit"] == "reflectance"
    with pytest.raises(ValueError, match="never mixed"):
        client.save_spectra(lib["id"], "ds", [_item("b", "radiance")])
    client.save_spectra(lib["id"], "ds", [_item("c", "unknown")])          # unknown is accepted
    mats = {m["name"]: m for m in client.list_materials(lib["id"])["items"]}
    assert set(mats) == {"a", "c"}                                          # nothing of b was written
    sp = client.list_spectra(client.list_samples(mats["a"]["id"])[0]["id"])[0]
    assert sp["measurement_unit"] == "reflectance" and sp["purity"] == "ground_measured"
    with pytest.raises(ValueError):
        client.save_spectra(lib["id"], "ds", [_item("d", "reflectance", "made-up")])
    r = client.update_materials([mats["a"]["id"], mats["c"]["id"]], purity="synthetic")
    assert r["spectra"] == 2
    sp = client.list_spectra(client.list_samples(mats["c"]["id"])[0]["id"])[0]
    assert sp["purity"] == "synthetic"


def test_roi_saves_carry_unit_and_purity(client):
    demo = make_demo_scenes()[2]
    lib = client.create_library("ROIs")
    recs, items = R.library_items([(demo, r) for r in demo.rois], "class")
    assert {it["unit"] for it in items} == {"reflectance"} and {it["purity"] for it in items} == {"synthetic"}
    client.save_roi_spectra(lib["id"], "rois", recs, items)
    e = client.roi_spectra(library_id=lib["id"], class_name="urban")[0]["spectrum"]
    assert e["measurement_unit"] == "reflectance" and e["purity"] == "synthetic"
    rad = _scene("rad", unit="radiance")
    rad.add_roi({"class": "x", "rect": (0, 0, 3, 3)})
    r2, i2 = R.library_items([(rad, rad.rois[0])], "roi")
    with pytest.raises(ValueError, match="never mixed"):
        client.save_roi_spectra(lib["id"], "rois", r2, i2)
    assert len(client.list_rois(lib["id"])) == 3                            # the radiance ROI was not stored


def test_import_units_purity_and_rejections(client, tmp_path):
    csv = tmp_path / "leaves.csv"
    wl = np.arange(400, 2401, 100)
    csv.write_text("wavelength,leaf_a,leaf_b\n" + "\n".join(f"{w},{0.1 + w / 1e4:.4f},{0.2:.4f}" for w in wl))
    db = client.db
    with db.session() as s:
        res = import_path(s, csv, library_name="Leaves")
    assert res["spectra_added"] == 2 and res["library_unit"] == "reflectance"
    pv = preview_path(csv)
    assert pv["units"] == ["reflectance"] and pv["purity"] == ["ground_measured"]
    rad = tmp_path / "radiance.csv"
    rad.write_text("wavelength,lamp\n" + "\n".join(f"{w},{50 + w / 10:.2f}" for w in wl))
    with db.session() as s:
        res2 = import_path(s, rad, library_id=res["library_id"],
                           options=ReadOptions(value_scale="none", unit="radiance"))
    assert res2["spectra_added"] == 0 and res2["unit_rejected"] == 1 and "never mixed" in res2["unit_reason"]
    with db.session() as s:
        res3 = import_path(s, rad, library_name="Lamps", options=ReadOptions(value_scale="none", unit="radiance",
                                                                             purity="synthetic"))
    assert res3["spectra_added"] == 1 and res3["library_unit"] == "radiance"


def test_image_import_masks_bad_bands_and_tags_image_pixel(client, tmp_path):
    sc = _scene("img", value=0.25)
    sc.labels = np.ones(sc.cube.shape[:2], np.int32)
    sc.class_names = {1: "grass"}
    p = tmp_path / "img.npz"
    save_scene(sc, str(p))
    with client.db.session() as s:
        res = import_path(s, p, library_name="From image")
    assert res["spectra_added"] == 1
    m = client.list_materials(res["library_id"])["items"][0]
    sp = client.list_spectra(client.list_samples(m["id"])[0]["id"])[0]
    vals = np.array([np.nan if v is None else v for v in sp["values"]])
    assert sp["purity"] == "image_pixel" and sp["measurement_unit"] == "reflectance"
    assert np.isnan(vals).sum() == int(sc.bad_bands.sum()) and sp["metadata"]["bad_bands_masked"] > 0


def test_folder_import_skips_quicklooks(client, tmp_path, monkeypatch):
    monkeypatch.setattr(Q, "CACHE_DIR", str(tmp_path / "cache"))
    folder = tmp_path / "enmap"
    folder.mkdir()
    _enmap_files(folder)
    with client.db.session() as s:
        res = import_path(s, folder, library_name="EnMAP")
    assert res["spectra_added"] == 1                                       # only the SPECTRAL_IMAGE mean
    assert any("quicklook" in w for w in res["warnings"]) and any("quality" in w for w in res["warnings"])
    m = client.list_materials(res["library_id"])["items"][0]
    sp = client.list_spectra(client.list_samples(m["id"])[0]["id"])[0]
    assert sp["purity"] == "image_pixel" and sp["measurement_unit"] == "reflectance"
