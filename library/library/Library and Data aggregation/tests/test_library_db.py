"""Material library database (hvva.library: models / repository / importer / client).

Runs on a temporary SQLite file by default.  To run the same tests against PostgreSQL:
    HVVA_TEST_PG_URL=postgresql://postgres:postgres@localhost:5432/material_library_test pytest tests/test_library_db.py
(the database is emptied first - do not point this at real data).
"""
import os
from pathlib import Path

import numpy as np
import pytest

from hvva.core.resample import common_grid as grid_fn
from hvva.core.signature import signatures_from_result
from hvva.core.stats import aggregate_classes
from hvva.core.synthetic import make_demo_scenes
from hvva.library import repository as repo
from hvva.library.client import DirectClient
from hvva.library.database import Database, resolve_driver
from hvva.library.importer import read_wide_csv

DATA = Path(__file__).resolve().parents[1] / "data"


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
            con.exec_driver_sql("DROP TABLE IF EXISTS data_files, spectra, samples, materials, datasets, libraries CASCADE")
        db.dispose()
        c = DirectClient(url, init_schema=True)
    yield c
    c.close()


def _csv(tmp_path, wl=(400, 410, 420, 430), cols=None):
    cols = cols or {"Copper Metal": [.1, .2, .3, .4], "Red, Brick": [.4, .3, "", .1], "Grass": [.05, .3, .6, .5]}
    p = tmp_path / "spectra.csv"
    lines = ["Wavelength," + ",".join(f'"{c}"' for c in cols)]
    for i, w in enumerate(wl):
        lines.append(f"{w}," + ",".join(str(v[i]) for v in cols.values()))
    p.write_text("\n".join(lines))
    return p


def test_driver_resolution():
    assert resolve_driver("sqlite:///x.db") == "sqlite:///x.db"
    assert resolve_driver("postgresql://u:p@h/db").startswith("postgresql+psycopg")


def test_read_wide_csv(tmp_path):
    wl, cols = read_wide_csv(_csv(tmp_path))
    assert list(wl) == [400, 410, 420, 430] and list(cols) == ["Copper Metal", "Red, Brick", "Grass"]
    assert np.isnan(cols["Red, Brick"][2])


def test_import_is_idempotent_and_browsable(client, tmp_path):
    lib = client.create_library("Test Lib", "desc")
    assert lib["slug"] == "test-lib" and client.create_library("Test Lib")["slug"] == "test-lib-2"
    r = client.import_csv(_csv(tmp_path), lib["id"], "Set A", "Mixed")
    assert r["materials"] == 3 and r["spectra_added"] == 3 and r["step"] == 10
    assert client.import_csv(_csv(tmp_path), lib["id"], "Set A")["spectra_added"] == 0
    libs = {l["id"]: l for l in client.list_libraries()}
    assert libs[lib["id"]]["material_count"] == 3 and libs[lib["id"]]["dataset_count"] == 1
    page = client.list_materials(lib["id"], limit=2)
    assert page["total"] == 3 and len(page["items"]) == 2
    assert client.list_materials(lib["id"], limit=2, offset=2)["items"][0]["name"] == "Red, Brick"
    assert [m["name"] for m in client.list_materials(lib["id"], search="copp")["items"]] == ["Copper Metal"]
    assert client.list_materials(lib["id"], category="Mixed")["total"] == 3
    assert client.list_categories(lib["id"]) == ["Mixed"]
    hits = client.search_materials("gra")["items"]
    assert hits[0]["name"] == "Grass" and hits[0]["library_name"] == "Test Lib"
    m = next(x for x in page["items"] if x["name"] == "Copper Metal")
    assert m["metadata"]["source_file"] == "spectra.csv"           # metadata really stored (bug in the original)
    s = client.list_samples(m["id"])
    sp = client.list_spectra(s[0]["id"])[0]
    assert np.allclose(repo.spectrum_wavelengths(sp), [400, 410, 420, 430])
    assert np.allclose(repo.spectrum_values(sp), [.1, .2, .3, .4])
    red = client.search_materials("brick")["items"][0]
    v = repo.spectrum_values(client.list_spectra(client.list_samples(red["id"])[0]["id"])[0])
    assert np.isnan(v[2])                                          # empty cell -> NULL -> NaN


def test_save_signatures_and_match(client):
    scenes = make_demo_scenes(3)
    g = grid_fn([s.wavelengths for s in scenes])
    sigs = signatures_from_result(aggregate_classes(scenes, g))
    lib = client.create_library("Signatures")
    saved = client.save_spectra(lib["id"], "Run 1", sigs, category="Demo")
    assert len(saved) == 4
    again = client.save_spectra(lib["id"], "Run 2", sigs[:1])      # same name -> second sample of that material
    assert again[0]["sample"]["sample_number"] == 2 and client.list_materials(lib["id"])["total"] == 4
    assert {d["name"] for d in client.list_datasets(lib["id"])} == {"Run 1", "Run 2"}
    ents = client.library_spectra(lib["id"])
    assert len(ents) == 4
    veg = next(s for s in sigs if s.name == "vegetation")
    best = repo.spectral_angle_rank(veg.mean, g, ents, top=1)[0]
    assert best[1]["name"] == "vegetation" and best[0] < 1e-5
    sp = next(e for e in ents if e["material"]["name"] == "vegetation")["spectrum"]
    assert len(sp["metadata"]["std"]) == len(g) and sp["metadata"]["n_pixels"] == veg.n_pixels
    # irregular wavelengths keep an explicit axis
    client.save_spectra(lib["id"], "Pixels", [{"name": "px", "wavelengths": [400, 405, 420], "values": [1, 2, 3]}])
    px = next(e for e in client.library_spectra(lib["id"]) if e["material"]["name"] == "px")["spectrum"]
    assert px["wavelength_step"] is None and list(repo.spectrum_wavelengths(px)) == [400, 405, 420]


def test_delete_cascades(client, tmp_path):
    lib = client.create_library("Gone")
    client.import_csv(_csv(tmp_path), lib["id"])
    m = client.list_materials(lib["id"])["items"][0]
    client.delete_material(m["id"])
    assert client.list_materials(lib["id"])["total"] == 2
    with pytest.raises(repo.NotFound):
        client.get_material(m["id"])
    client.delete_library(lib["id"])
    assert all(l["name"] != "Gone" for l in client.list_libraries())
    assert client.search_materials("grass")["items"] == []
    with pytest.raises(repo.NotFound):
        client.get_material("not-a-uuid")


@pytest.mark.skipif(not (DATA / "soil_spectra.csv").exists(), reason="sample data not present")
def test_import_bundled_csv(client):
    lib = client.create_library("Soils")
    r = client.import_csv(DATA / "soil_spectra.csv", lib["id"], "Soil Spectra", "Soil")
    assert r["materials"] == 20 and r["points"] == 2151 and r["range"] == (350.0, 2500.0)


# ---------------------------------------------------------------- category tree + acquisition type
def test_category_paths_normalised():
    assert repo.normalize_category(" Vegetation > Tree -> Tsuga / Canadensis ") == "Vegetation/Tree/Tsuga/Canadensis"
    assert repo.normalize_category("a//b|c") == "a/b/c"
    assert repo.normalize_category("  ") is None
    assert repo.normalize_acquisition("Lab") == "lab" and repo.normalize_acquisition("untagged") is None
    with pytest.raises(ValueError):
        repo.normalize_acquisition("drone")


def test_category_tree_and_acquisition(client, tmp_path):
    lib = client.create_library("Tree")
    p = _csv(tmp_path, cols={"Tsuga canadensis": [.1, .2, .3, .4], "Tsuga heterophylla": [.2, .2, .3, .4],
                             "Pinus strobus": [.3, .2, .3, .4], "Grass": [.1, .1, .1, .1], "Basalt": [.05] * 4,
                             "Pinus_%x": [.3] * 4})
    client.import_csv(p, lib["id"], "Set", category="Vegetation > Tree", acquisition="lab")
    ids = {m["name"]: m["id"] for m in client.list_materials(lib["id"], limit=50)["items"]}
    client.update_materials([ids["Tsuga canadensis"]], category="Vegetation/Tree/Tsuga/Canadensis", acquisition="field")
    client.update_materials([ids["Tsuga heterophylla"]], category="Vegetation/Tree/Tsuga/Heterophylla")
    client.update_materials([ids["Grass"]], category="Vegetation/Grass", acquisition="airborne")
    client.update_materials([ids["Basalt"]], category=None, acquisition=None)
    client.update_materials([ids["Pinus_%x"]], category="Vegetation/Tree_%")
    f = client.library_facets(lib["id"])
    assert f["total"] == 6 and f["uncategorised"] == 1
    assert {c["path"]: c["count"] for c in f["categories"]} == {
        "Vegetation/Grass": 1, "Vegetation/Tree": 1, "Vegetation/Tree/Tsuga/Canadensis": 1,
        "Vegetation/Tree/Tsuga/Heterophylla": 1, "Vegetation/Tree_%": 1}
    assert f["acquisition"] == {"lab": 3, "field": 1, "airborne": 1, "satellite": 0, "untagged": 1}
    names = lambda **kw: sorted(m["name"] for m in client.list_materials(lib["id"], limit=50, **kw)["items"])
    assert names(category_path="Vegetation/Tree") == ["Pinus strobus", "Tsuga canadensis", "Tsuga heterophylla"]
    assert names(category_path="Vegetation/Tree/Tsuga") == ["Tsuga canadensis", "Tsuga heterophylla"]
    assert names(category_path="Vegetation/Tre") == []                 # whole levels only
    assert names(category_path="Vegetation/Tree_%") == ["Pinus_%x"]   # LIKE wildcards escaped
    assert names(category_path=repo.UNCATEGORISED) == ["Basalt"]
    assert names(acquisition="field") == ["Tsuga canadensis"]
    assert names(acquisition=repo.UNTAGGED) == ["Basalt"]
    assert names(category_path="Vegetation", acquisition="lab") == ["Pinus strobus", "Pinus_%x", "Tsuga heterophylla"]
    row = client.list_materials(lib["id"], search="grass")["items"][0]
    assert row["acquisition_types"] == ["airborne"] and row["category"] == "Vegetation/Grass"
    # rename / move a node with everything below it
    assert client.rename_category(lib["id"], "Vegetation/Tree", "Plants/Conifers")["materials"] == 3
    assert names(category_path="Plants/Conifers/Tsuga") == ["Tsuga canadensis", "Tsuga heterophylla"]
    with pytest.raises(Exception):
        client.rename_category(lib["id"], "Plants", "Plants/Inner")
    client.rename_category(lib["id"], "Plants/Conifers/Tsuga", None)  # -> uncategorised
    assert names(category_path=repo.UNCATEGORISED) == ["Basalt", "Tsuga canadensis", "Tsuga heterophylla"]
    # per-spectrum tag, matching pool and global search respect the tag
    sp = client.list_spectra(client.list_samples(ids["Pinus strobus"])[0]["id"])[0]
    assert sp["acquisition_type"] == "lab"
    assert client.set_spectrum_acquisition(sp["id"], "satellite")["acquisition_type"] == "satellite"
    assert [e["material"]["name"] for e in client.library_spectra(lib["id"], acquisition="satellite")] == ["Pinus strobus"]
    assert len(client.library_spectra(lib["id"], acquisition="lab")) == 2
    assert [m["name"] for m in client.search_materials("pinus", acquisition="satellite")["items"]] == ["Pinus strobus"]
    saved = client.save_spectra(lib["id"], "HV", [{"name": "px", "wavelengths": [400, 500], "values": [1, 2]}],
                                category="Urban > Roof", acquisition="airborne")
    assert saved[0]["material"]["category"] == "Urban/Roof"
    assert client.list_spectra(saved[0]["sample"]["id"])[0]["acquisition_type"] == "airborne"


def test_migration_adds_acquisition_column(tmp_path):
    """A database made by the original backend schema (no acquisition_type) is upgraded on connect."""
    from sqlalchemy import create_engine, inspect, text
    url = f"sqlite:///{tmp_path / 'old.db'}"
    eng = create_engine(url)
    with eng.begin() as c:
        c.execute(text("CREATE TABLE libraries (id CHAR(32) PRIMARY KEY, name VARCHAR(255) NOT NULL, slug VARCHAR(255) "
                       "NOT NULL UNIQUE, description TEXT, metadata JSON NOT NULL, created_at DATETIME, updated_at DATETIME)"))
        c.execute(text("CREATE TABLE spectra (id CHAR(32) PRIMARY KEY, sample_id CHAR(32) NOT NULL, wavelength_start "
                       "FLOAT, wavelength_end FLOAT, wavelength_step FLOAT, measurement_unit VARCHAR(100), "
                       "\"values\" JSON, metadata JSON, created_at DATETIME)"))
    eng.dispose()
    db = Database(url)
    assert db.migrate() == ["spectra.acquisition_type"]
    assert "acquisition_type" in {c["name"] for c in inspect(db.engine).get_columns("spectra")}
    assert db.migrate() == []                                       # idempotent


# ---------------------------------------------------------------- USGS splib07 / ECOSTRESS importers
def _usgs_zip(tmp_path):
    import zipfile
    wl = 0.35 + 0.001 * np.arange(2151)
    root = "ASCIIdata_splib07b_cvASD"
    spec = lambda k: np.where((np.arange(2151) > 1350) & (np.arange(2151) < 1420), -1.23e34, 0.3 + 0.1 * np.sin(wl * k))
    files = {f"{root}/s07_ASD_Wavelengths_ASD_0.35-2.5_microns_2151_ch.txt": wl,
             f"{root}/ChapterM_Minerals/s07_ASD_Kaolinite_KGa-1_BECKb_AREF.txt": spec(3),
             f"{root}/ChapterM_Minerals/s07_ASD_Alunite_GDS84_Na03_BECKb_AREF.txt": spec(4),
             f"{root}/ChapterV_Vegetation/s07_ASD_Lodgepole-Pine_LP-Fir_AVIRIS-1996_AREF.txt": spec(5),
             f"{root}/ChapterA_ArtificialMaterials/s07_ASD_Asphalt_GDS376_Blck_Road_ASDFRa_AREF.txt": spec(6)}
    p = tmp_path / "ASCIIdata_splib07b_cvASD.zip"
    with zipfile.ZipFile(p, "w") as z:
        for n, a in files.items():
            z.writestr(n, "s07_ASD Record=1: title\n" + "\n".join(f"{v:.6g}" for v in a))
    return p


def _ecostress_dir(tmp_path):
    d = tmp_path / "eco"
    d.mkdir()
    x = np.round(np.arange(2.5, 0.3499, -0.001), 4)
    head = ("Name: Acer rubrum\nType: Vegetation\nClass: Tree\nGenus: Acer\nSpecies: rubrum\nSample No.: VH197\n"
            "X Units: Wavelength (micrometers)\nY Units: Reflectance (percent)\n\n")
    (d / "vegetation.tree.acer.rubrum.vswir.vh197.ucsb.asd.spectrum.txt").write_text(
        head + "\n".join(f"{a}\t{50.0}" for a in x))
    xt = np.round(np.arange(15.0, 2.5, -0.01), 3)
    (d / "mineral.silicate.x.tir.spectrum.txt").write_text(
        "Name: Quartz\nType: Mineral\nClass: Silicate\nX Units: Wavelength (micrometers)\nY Units: Reflectance (percent)\n\n"
        + "\n".join(f"{a}\t{10.0}" for a in xt))
    (d / "vegetation.tree.acer.rubrum.vswir.vh197.ucsb.asd.ancillary.txt").write_text("Name: Acer rubrum\n")
    return d


def test_import_usgs_splib07(client, tmp_path):
    lib = client.create_library("USGS")
    r = client.import_speclib("usgs", _usgs_zip(tmp_path), lib["id"], chapters="M,V,A")
    assert r["spectra_added"] == 4 and r["categories"] == 4
    assert client.import_speclib("usgs", _usgs_zip(tmp_path), lib["id"], chapters="M,V,A")["spectra_added"] == 0
    f = client.library_facets(lib["id"])
    assert {c["path"] for c in f["categories"]} == {"Minerals/Kaolinite", "Minerals/Alunite", "Vegetation/Lodgepole Pine",
                                                     "Artificial Materials/Asphalt"}
    assert f["acquisition"]["airborne"] == 1 and f["acquisition"]["lab"] == 3
    m = client.search_materials("kaolinite")["items"][0]
    assert m["name"] == "Kaolinite KGa-1"
    sp = client.list_spectra(client.list_samples(m["id"])[0]["id"])[0]
    assert sp["wavelength_start"] == pytest.approx(350) and sp["wavelength_step"] == pytest.approx(1)
    v = repo.spectrum_values(sp)
    assert np.isnan(v[1400]) and not np.isnan(v[100])               # -1.23e34 bad bands -> NULL
    assert client.import_speclib("usgs", _usgs_zip(tmp_path), client.create_library("U2")["id"],
                                 chapters="V")["spectra_added"] == 1


def test_import_ecostress(client, tmp_path):
    lib = client.create_library("ECO")
    r = client.import_speclib("ecostress", _ecostress_dir(tmp_path), lib["id"])
    assert r["spectra_added"] == 1                                     # TIR-only file and ancillary skipped
    m = client.list_materials(lib["id"])["items"][0]
    assert m["name"] == "Acer rubrum VH197" and m["category"] == "Vegetation/Tree/Acer/Rubrum"
    assert m["acquisition_types"] == ["lab"]
    sp = client.list_spectra(client.list_samples(m["id"])[0]["id"])[0]
    w = repo.spectrum_wavelengths(sp)
    assert w[0] == pytest.approx(350) and w[-1] == pytest.approx(2500)  # µm -> nm, ascending
    assert repo.spectrum_values(sp)[10] == pytest.approx(0.5)          # percent -> fraction


def test_import_openspeclib_parquet(client, tmp_path):
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq
    pq.write_table(pa.table({"grid_id": [1, 2], "source": ["x", "x"], "wavelength_unit": ["um", "um"],
                             "n_points": [3, 3], "wavelengths": [[0.4, 0.5, 0.6], [2.6, 5.0, 10.0]]}),
                   tmp_path / "wavelengths.parquet")
    rows = [("e1", "Acer rubrum", "vegetation", "Tree", "VH1", None, 1, [0.1, 0.2, 0.3]),
            ("e2", "Acer rubrum", "vegetation", "Tree", "VH1", None, 2, [0.1, 0.1, 0.1]),        # TIR -> skipped
            ("e3", "Kaolinite KGa-1", "mineral", "Phyllosilicate", "KGa-1", "BECKb", 1, [0.5, -1.23e34, 0.7]),
            ("e4", "Lodgepole Pine LP-Fir", "vegetation", None, None, "AVIRISb", 1, [0.05, 0.1, 0.2])]
    cols = ["id", "name", "material.category", "material.subcategory", "sample.id", "measurement.instrument",
            "spectral_data.wavelength_grid_id", "spectral_data.values"]
    t = pa.table({c: [r[i] for r in rows] for i, c in enumerate(cols)})
    t = t.append_column("material.name", t.column("name")).append_column(
        "spectral_data.wavelength_unit", pa.array(["um"] * 4)).append_column(
        "spectral_data.reflectance_scale", pa.array(["unit"] * 4))
    pq.write_table(t, tmp_path / "src.parquet")
    lib = client.create_library("OSL")
    r = client.import_speclib("openspeclib", tmp_path / "src.parquet", lib["id"])
    assert r["spectra_added"] == 3
    assert client.import_speclib("openspeclib", tmp_path / "src.parquet", lib["id"])["spectra_added"] == 0
    cats = {m["name"]: (m["category"], m["acquisition_types"]) for m in client.list_materials(lib["id"])["items"]}
    assert cats["Acer rubrum VH1"] == ("Vegetation/Tree/Acer/Rubrum", ["lab"])
    assert cats["Kaolinite KGa-1"] == ("Mineral/Phyllosilicate/Kaolinite", ["lab"])
    assert cats["Lodgepole Pine LP-Fir"][1] == ["airborne"]
    k = client.search_materials("kaolinite")["items"][0]
    sp = client.list_spectra(client.list_samples(k["id"])[0]["id"])[0]
    assert list(repo.spectrum_wavelengths(sp)) == pytest.approx([400, 500, 600]) and np.isnan(repo.spectrum_values(sp)[1])


# ---------------------------------------------------------------- data folder scan + batch import
def test_data_folder_scan_and_import(client, tmp_path):
    from hvva.library import sources
    folder = tmp_path / "speclib"
    (folder / "downloads").mkdir(parents=True)
    import shutil
    shutil.copy(_usgs_zip(tmp_path), folder / "downloads" / "ASCIIdata_splib07b_cvASD.zip")
    eco = _ecostress_dir(tmp_path)
    shutil.copytree(eco, folder / "eco_veg")
    _csv(folder)                                                       # folder/spectra.csv (wide CSV)
    (folder / "notes.txt").write_text("not spectra")
    (folder / "other.csv").write_text("a,b\n1,2\n")
    found = {s["name"]: s for s in client.scan_folder(folder)}
    assert set(found) == {"ASCIIdata_splib07b_cvASD.zip", "eco_veg", "spectra.csv"}
    assert found["ASCIIdata_splib07b_cvASD.zip"]["kind"] == "usgs" and found["ASCIIdata_splib07b_cvASD.zip"]["library"] == "USGS splib07"
    assert found["eco_veg"]["kind"] == "folder" and found["spectra.csv"]["kind"] == "table"
    assert all(s["status"] == "new" for s in found.values())
    res = client.import_sources([{"source": found["ASCIIdata_splib07b_cvASD.zip"], "library": "USGS test"},
                                 {"source": found["eco_veg"], "library": "Eco test"},
                                 {"source": found["spectra.csv"], "library": "CSV test", "acquisition": "field"}])
    assert all(r["ok"] for r in res) and [r["spectra_added"] for r in res] == [4, 1, 3]
    again = {s["name"]: s for s in client.scan_folder(folder)}
    assert all(s["status"] == "imported" for s in again.values())
    assert any("USGS test" in x for x in again["ASCIIdata_splib07b_cvASD.zip"]["imported_into"])
    libs = {l["name"]: l["material_count"] for l in client.list_libraries()}
    assert libs["USGS test"] == 4 and libs["Eco test"] == 1 and libs["CSV test"] == 3
    csvlib = next(l for l in client.list_libraries() if l["name"] == "CSV test")
    assert client.library_facets(csvlib["id"])["acquisition"]["field"] == 3
    # a new file appears -> only that one is imported
    _csv(folder / "downloads", cols={"Roof tile": [.2, .2, .3, .3]})
    res = client.import_new_from_folder(folder)
    assert [(r["source"], r["spectra_added"]) for r in res] == [("spectra.csv", 1)]
    assert client.import_new_from_folder(folder) == []


def test_parquet_without_wavelengths_is_reported(tmp_path):
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq
    from hvva.library import sources
    pq.write_table(pa.table({"id": ["a"], "spectral_data.values": [[0.1, 0.2]],
                             "spectral_data.wavelength_grid_id": [1]}), tmp_path / "ecostress.parquet")
    s = sources.scan(tmp_path)
    assert len(s) == 1 and s[0].status == "error" and "wavelengths.parquet" in s[0].note
