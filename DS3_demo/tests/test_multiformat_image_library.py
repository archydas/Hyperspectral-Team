"""Multi-format import (.mat, .txt tables, .zip, AVIRIS-NG ancillary files) and the image viewer with library overlay."""
import os, zipfile
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import numpy as np, pytest, scipy.io as sio
from ds3.pipeline import ingest_path, classify
from ds3.ingest import open_cube, IngestError
from ds3.ingest.metadata import MissingWavelengthsError, UnsupportedFormatError
from ds3.library.access import list_cubes, list_libraries, db_stats
from ds3.library.sources import read_label_map, spectra_from_labels, PAVIA_U_CLASSES
from ds3.library import image_ops
from ds3.db import session_scope
from ds3.spectra import read_spectra

BASE = np.random.default_rng(0).random((3, 103)) + 0.5
LAB = np.zeros((60, 50), int); LAB[:30] = 1; LAB[30:, :25] = 2; LAB[30:, 25:] = 3


def make_pavia(d, name="PaviaU", var="paviaU"):
    cube = (BASE[LAB - 1] * 3000 + np.random.default_rng(1).normal(0, 30, (60, 50, 103))).astype("uint16")
    sio.savemat(d / f"{name}.mat", {var: cube}); sio.savemat(d / f"{name}_gt.mat", {var + "_gt": LAB.astype("uint8")})
    return d / f"{name}.mat"


def test_mat_known_scene_assumes_and_says_so(tmp_path):
    c = open_cube(make_pavia(tmp_path))
    assert c.shape == (60, 50, 103) and c.unit == "dn" and c.sensor == "ROSIS-03"
    assert c.wavelengths_nm[0] == 430 and c.wavelengths_nm[-1] == 860
    assert any("ASSUMED" in n for n in c.metadata["notes"])


def test_mat_unknown_scene_needs_range_or_list(tmp_path):
    sio.savemat(tmp_path / "x.mat", {"cube": np.ones((5, 5, 20), "float32")})
    with pytest.raises(MissingWavelengthsError, match="wavelength_range"):
        open_cube(tmp_path / "x.mat")
    c = open_cube(tmp_path / "x.mat", wavelength_range=(400, 900))
    assert c.bands == 20 and c.wavelengths_nm[-1] == 900
    c = open_cube(tmp_path / "x.mat", wavelengths=np.linspace(500, 700, 20))
    assert c.wavelengths_nm[0] == 500


def test_mat_label_file_is_not_an_image_but_gives_class_spectra(tmp_path):
    p = make_pavia(tmp_path)
    with pytest.raises(UnsupportedFormatError, match="label"):
        classify(tmp_path / "PaviaU_gt.mat")
    cube = open_cube(p)
    labels, name = read_label_map(tmp_path / "PaviaU_gt.mat")
    sp = spectra_from_labels(cube, labels, PAVIA_U_CLASSES)
    assert [s.material for s in sp] == PAVIA_U_CLASSES[:3] and sp[0].purity == "image_pixel"
    with pytest.raises(Exception, match="same pixels"):
        spectra_from_labels(cube, labels[:10], PAVIA_U_CLASSES)


def test_text_table_layouts_import_through_ingest_path(tmp_path):
    wl = np.arange(350, 2501, 10.0); rng = np.random.default_rng(2)
    with open(tmp_path / "samples_spectra_with_metadata.txt", "w") as f:
        f.write("sample_id\tmaterial\tclass\t" + "\t".join(f"{w:g}" for w in wl) + "\n")
        for i in range(5): f.write(f"S{i}\tbrick{i}\tconstruction\t" + "\t".join(f"{v:.4f}" for v in rng.random(len(wl))) + "\n")
    with open(tmp_path / "soil_spectra.txt", "w") as f:
        f.write("Wavelength soil1 soil2\n")
        for w in wl: f.write(f"{w:g} {rng.random():.4f} {rng.random():.4f}\n")
    with open(tmp_path / "long.csv", "w") as f:
        f.write("name,wavelength,reflectance\n")
        for n in ("a", "b"):
            for w in wl: f.write(f"{n},{w:g},{rng.random():.4f}\n")
    db = f"sqlite:///{tmp_path}/t.db"
    res = {r.path.rsplit(os.sep, 1)[-1]: r for r in ingest_path(tmp_path, db_url=db) if r.kind == "spectra" or r.status == "error"}
    assert res["samples_spectra_with_metadata.txt"].detail == "5 spectra"
    assert res["soil_spectra.txt"].detail == "2 spectra" and res["long.csv"].detail == "2 spectra"
    sp = read_spectra(tmp_path / "samples_spectra_with_metadata.txt")
    assert sp[0].category == "construction" and sp[0].metadata["layout"] == "rows_by_spectrum"


def test_unreadable_text_file_gives_a_diagnosis(tmp_path):
    (tmp_path / "weird.txt").write_text("hello\nthis is not a spectrum\nat all\n")
    r = ingest_path(tmp_path / "weird.txt", db_url=f"sqlite:///{tmp_path}/t.db")[0]
    assert r.status == "error" and "layout" in r.detail


def test_zip_with_enmap_like_names_and_mat(tmp_path):
    p = make_pavia(tmp_path / ".." / tmp_path.name)
    z = tmp_path / "pavia.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.write(p, "scene/PaviaU.mat")
    db = f"sqlite:///{tmp_path}/z.db"
    from ds3.pipeline import extract_archive
    res = ingest_path(z, db_url=db)
    assert [r.status for r in res if r.kind == "cube"] == ["ingested"]


def test_zip_slip_refused(tmp_path):
    z = tmp_path / "evil.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("../../evil.txt", "x")
    r = ingest_path(z, db_url=f"sqlite:///{tmp_path}/e.db")[0]
    assert r.status == "error" and "unsafe" in r.detail


@pytest.mark.parametrize("name", ["a_rdn_v2m2_glt", "a_rdn_v2m2_obs_ort", "a_rdn_v2m2_loc", "a_h2o_v2m2_img", "a_rdn_v2m2_igm.hdr"])
def test_aviris_ng_ancillary_products_are_skipped(tmp_path, name):
    (tmp_path / name).write_text("x")
    r = ingest_path(tmp_path / name, db_url=f"sqlite:///{tmp_path}/a.db")[0]
    assert r.status == "skipped" and "ancillary" in r.detail


# ---------------------------------------------------------------- image viewer + library in the image
pytest.importorskip("PySide6"); pytest.importorskip("matplotlib")
from PySide6.QtWidgets import QApplication


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _url(tmp_path, name):
    """SQLite by default; set DS3_TEST_DB_URL=postgresql+psycopg://... (an EMPTY scratch database) to run on PostgreSQL."""
    import os
    url = os.environ.get("DS3_TEST_DB_URL")
    if url:
        from ds3.db import reset_db
        reset_db(url)
        return url
    return f"sqlite:///{tmp_path}/{name}.db"


@pytest.fixture()
def win(qapp, tmp_path):
    from ds3.gui.main_window import MainWindow
    w = MainWindow(db_url=_url(tmp_path, "v"), sync=True); w.tmp = tmp_path
    yield w
    w.close()


def write_library_txt(path, wl, names=("Asphalt-like", "Meadow-like", "Gravel-like")):
    with open(path, "w") as f:
        f.write("name\t" + "\t".join(f"{w:g}" for w in wl) + "\n")
        for i, n in enumerate(names): f.write(n + "\t" + "\t".join(f"{v:.5f}" for v in BASE[i]) + "\n")


def test_open_image_then_bring_library_into_it(win):
    d = win.tmp; p = make_pavia(d)
    pan = win.image_panel
    pan.open_image_path(str(p))
    assert win.last_error is None and len(win.cubes) == 1 and pan.cube is not None
    assert pan.preview["rgb"].shape == (60, 50, 3) and "ASSUMED" in pan.lbl_img.text()
    # bring a .txt library into the image view (stored in the database)
    lib = d / "my_library.txt"; write_library_txt(lib, np.linspace(430, 860, 103))
    pan.import_library_file(str(lib))
    assert win.last_error is None and pan.lst.count() == 3 and pan.cb_lib.currentText() == "my_library"
    # click a pixel in the 'meadow' block (class 2) -> spectrum + overlay of the selected library spectra
    pan.lst.selectAll()
    pan.pick(40, 5)
    assert pan.pixel is not None and len(pan.spec.ax.lines) >= 4
    # best matches: image is DN, library is unit 'reflectance' -> refused unless the user allows it
    pan.best_matches()
    assert pan.t_match.rowCount() == 0 and "ignore unit" in pan.lbl.text()
    pan.chk_unit.setChecked(True); pan.best_matches()
    assert pan.t_match.rowCount() == 3 and pan.t_match.item(0, 1).text() == "Meadow-like"
    # SAM map of the whole image reproduces the three blocks
    pan.sam_map()
    m = pan.last_map
    assert m["names"] == ["Asphalt-like", "Meadow-like", "Gravel-like"] and (m["index"] == LAB - 1).mean() > 0.99
    assert "unit mismatch" in pan.lbl.text()
    # save the pixel back as a library spectrum
    pan.e_name.setText("my pixel"); pan.e_save_lib.setText("Field picks"); pan.save_pixel()
    assert win.last_error is None
    assert "Field picks" in [win.d_lib.itemText(i) for i in range(win.d_lib.count())]


def test_label_map_to_library_in_app(win):
    p = make_pavia(win.tmp); pan = win.image_panel
    pan.labels_to_library(str(win.tmp / "PaviaU_gt.mat"))
    assert "Open the image first" in win.last_error            # no image open yet -> clear message
    win.last_error = None
    pan.open_image_path(str(p))
    pan.labels_to_library(str(win.tmp / "PaviaU_gt.mat"))
    assert win.last_error is None
    from ds3.library.access import list_libraries
    with session_scope(win.db_url) as s:
        libs = {l["name"]: l["n_spectra"] for l in list_libraries(s)}
    assert libs["PaviaU classes"] == 3


def test_sample_name_headers_with_digits_are_not_mistaken_for_wavelengths(tmp_path):
    """Header 'Wavelength soil1 soil2 ... soil8' (8+ numbered sample columns) must read as wavelength-column layout."""
    rng = np.random.default_rng(4)
    with open(tmp_path / "soil_spectra.txt", "w") as f:
        f.write("Wavelength " + " ".join(f"soil{i}" for i in range(1, 11)) + "\n")
        for w in np.arange(350, 2501, 5.0): f.write(f"{w:g} " + " ".join(f"{rng.random():.4f}" for _ in range(10)) + "\n")
    sp = read_spectra(tmp_path / "soil_spectra.txt")
    assert len(sp) == 10 and sp[0].material == "soil1" and sp[0].metadata["layout"] == "wavelength_column"


def test_library_folder_loaded_at_start_each_file_its_own_library(qapp, tmp_path):
    from ds3.gui.main_window import MainWindow
    wl = np.arange(350, 2501, 10.0); rng = np.random.default_rng(6); d = tmp_path / "libs"; d.mkdir()
    with open(d / "material_spectra.txt", "w") as f:
        f.write("name\tclass\t" + "\t".join(f"{w:g}" for w in wl) + "\n")
        for i in range(4): f.write(f"m{i}\tconstruction\t" + "\t".join(f"{v:.4f}" for v in rng.random(len(wl))) + "\n")
    with open(d / "soil_spectra.txt", "w") as f:
        f.write("Wavelength " + " ".join(f"soil{i}" for i in range(1, 10)) + "\n" + "".join(f"{w:g} " + " ".join(f"{rng.random():.4f}" for _ in range(9)) + "\n" for w in wl))
    (d / "broken.txt").write_text("not\na spectrum\n"); (d / "README.md").write_text("x")
    w = MainWindow(db_url=_url(tmp_path, "f"), sync=True, autoload=False)
    assert w.total == 0
    w.d_dir.setText(str(d)); w.load_library_folder()
    libs = {w.d_lib.itemText(i) for i in range(w.d_lib.count())}
    assert libs == {"Material spectra", "Soil spectra"} and w.total == 13
    log = w.i_log.toPlainText(); assert "broken.txt" in log and "[error" in log and "README" not in log
    w.i_log.clear(); w.load_library_folder()                           # second start: nothing duplicated
    assert w.total == 13 and "[duplicate]" in w.i_log.toPlainText()
    w.close()
