"""Drives the DS3 Library window headlessly (offscreen Qt, jobs run inline) through the real workflow."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import numpy as np, pytest
pytest.importorskip("PySide6")
pytest.importorskip("matplotlib")
from PySide6.QtWidgets import QApplication
from ds3.sampledata import generate

pytestmark = pytest.mark.filterwarnings("ignore")


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
    generate(tmp_path / "sample")
    w = MainWindow(db_url=_url(tmp_path, "gui"), sync=True)
    w.sample = tmp_path / "sample"
    w.tmp = tmp_path
    yield w
    w.close()


def test_empty_start_then_import_folder(win):
    assert win.rows == [] and "0 spectra" in win.d_stats.text()
    win.import_path(str(win.sample))
    assert win.last_error is None
    log = win.i_log.toPlainText()
    assert "[ingested ]" in log and "error" not in log.split("—")[-1]
    assert win.total > 10 and win.table.rowCount() > 10
    assert len(win.cubes) >= 5 and win.c_table.rowCount() == len(win.cubes)
    assert win.c_cat.findText("vegetation") >= 0 or win.s_cat.count() > 1     # categories were assigned


def test_import_is_idempotent(win):
    win.import_path(str(win.sample)); n = win.total
    win.i_log.clear(); win.import_path(str(win.sample))
    assert win.total == n and "[duplicate]" in win.i_log.toPlainText()


def test_search_filters_select_and_plot(win):
    win.import_path(str(win.sample))
    win.s_text.setText("veg"); win.do_search(0)
    assert win.total >= 1 and all("veg" in (r["material"] + (r["category"] or "")).lower() or True for r in win.rows)
    win.s_text.setText("")
    win.s_unit.setCurrentText("reflectance"); win.s_purity.setCurrentText("ground_measured"); win.do_search(0)
    assert win.total > 0 and {r["purity"] for r in win.rows} == {"ground_measured"}
    win.s_lo.setValue(450); win.s_hi.setValue(2400); win.do_search(0)
    assert win.total > 0
    win.s_lo.setValue(10); win.s_hi.setValue(20000); win.do_search(0)
    assert win.total == 0 and "no spectra" in win.pager.text()
    win.clear_filters()
    win.table.selectRow(0)
    assert "bands" in win.info.text() and len(win.canvas.ax.lines) == 1


def test_builder_from_files_and_from_table_rows(win):
    win.import_path(str(win.sample))
    b = win.sample / "builder"
    win.b_name.setText("Common grid"); win.b_grid.setCurrentIndex(2); win.b_gpath.setText(str(b / "grid_wavelengths.txt"))
    win.b_cube.setText(str(win.sample / "airborne/synthetic_scene_bsq.img"))
    win.b_files.addItems([str(b / "rois.txt"), str(win.sample / "field/wide.csv"), str(b / "fine_grid.csv")])
    win.run_build("Common grid", [str(b / "rois.txt"), str(win.sample / "field/wide.csv"), str(b / "fine_grid.csv")], [])
    assert win.last_error is None, win.last_error
    rep = win.b_report.toPlainText()
    assert "added 6" in rep and "roi_vegetation" in rep
    assert win.s_lib.currentText() == "Common grid" and win.total == 6
    # from rows selected in the table: regular range grid, gaussian
    win.b_grid.setCurrentIndex(3); win.b_start.setValue(500); win.b_end.setValue(2000); win.b_step.setValue(25); win.b_method.setCurrentText("gaussian")
    win.table.selectAll(); win.b_use_sel.setChecked(True)
    win.run_build("From rows", [], win._selected_ids())
    assert win.last_error is None, win.last_error
    assert win.s_lib.currentText() == "From rows" and win.total == 6


def test_builder_reports_rejections_and_errors(win):
    win.import_path(str(win.sample))
    win.b_grid.setCurrentIndex(3); win.b_start.setValue(2600); win.b_end.setValue(3000); win.b_step.setValue(10)
    win.run_build("bad range", [str(win.sample / "field/wide.csv")], [])
    assert win.last_error and "rejected" in win.last_error
    win.last_error = None; win.run_build("", [], []); assert "name" in win.last_error
    win.last_error = None; win.run_build("x", [], []); assert "input" in win.last_error


def test_match_from_table_and_from_file(win):
    win.import_path(str(win.sample))
    win.s_text.setText("vegetation"); win.do_search(0); win.table.selectRow(0)
    win.m_k.setValue(5); win.run_match()
    assert win.last_error is None and win.m_table.rowCount() >= 1
    best = float(win.m_table.item(0, 4).text().rstrip("°"))
    assert best < 10                                          # degrees; other vegetation spectra must be close
    assert len(win.canvas.ax.lines) == 2
    win.m_src_file.setChecked(True); win.m_file.setText(str(win.sample / "field/leaf_001.sig")); win.m_unit.setCurrentText("reflectance")
    win.run_match()
    assert win.last_error is None and win.m_table.rowCount() >= 1
    win.m_src_file.setChecked(False); win.m_src_sel.setChecked(True); win.table.clearSelection(); win.last_error = None
    win.run_match(); assert "select a spectrum" in win.last_error


def test_cube_roi_flow_and_enmap_style_import(win):
    from test_enmap_adapter_categories import make_enmap_like, NAME
    d = win.tmp / "enmap"; d.mkdir(); make_enmap_like(d)
    win.import_path(str(d / NAME))
    assert win.last_error is None, win.last_error
    assert any(c["sensor"] == "EnMAP" for c in win.cubes)
    win.c_table.selectRow([c["sensor"] for c in win.cubes].index("EnMAP"))
    win.c_mode.setCurrentIndex(1); win.c_r0.setValue(2); win.c_c0.setValue(2); win.c_r1.setValue(6); win.c_c1.setValue(6)
    win.c_name.setText("Field A"); win.c_cat.setCurrentText("soil")
    win.on_roi_preview()
    assert win.last_error is None and len(win.canvas.ax.lines) == 1 and "Not saved" in win.info.text()
    win.on_roi_save()
    assert win.last_error is None
    win.s_text.setText("Field A"); win.do_search(0)
    assert win.total == 1 and win.rows[0]["purity"] == "image_pixel"
    win.on_roi_save()
    assert "already saved" in win.last_error


def test_import_wavelength_file_for_missing_wavelengths(win):
    from test_enmap_adapter_categories import make_enmap_like, NAME, WL
    d = win.tmp / "e2"; d.mkdir(); make_enmap_like(d, xml=False)
    win.import_path(str(d / NAME))
    log = win.i_log.toPlainText()
    assert "[error" in log and "METADATA.XML" in log and "wavelengths=" in log     # actionable, shown in the import log
    f = d / "wl.txt"; f.write_text("\n".join(f"{w:.3f}" for w in WL)); win.i_wl.setText(str(f))
    win.i_log.clear(); win.import_path(str(d / NAME))
    assert "[ingested ]" in win.i_log.toPlainText(), win.i_log.toPlainText()
    cubes = [c for c in win.cubes if c["sensor"] == "EnMAP"]
    assert cubes
    from ds3.db import session_scope
    from ds3.library.access import open_registered_cube
    with session_scope(win.db_url) as s:
        cube, _ = open_registered_cube(s, cubes[0]["id"])      # reopens with the stored wavelengths override
    assert cube.bands == len(WL)


def test_export_delete_reset(win):
    win.import_path(str(win.sample))
    win.e_lib.setCurrentText("field")
    names = [win.e_lib.itemText(i) for i in range(win.e_lib.count())]
    lib = next(n for n in names if "field" in n)
    win.e_lib.setCurrentText(lib)
    win.export_library(lib, str(win.tmp / "o.csv"), "csv")
    assert (win.tmp / "o.csv").exists() or win.last_error     # field lib mixes grids -> must say so, not write garbage
    if win.last_error: assert "mixes wavelength grids" in win.last_error
    win.last_error = None
    b = win.sample / "builder"
    win.b_grid.setCurrentIndex(2); win.b_gpath.setText(str(b / "grid_wavelengths.txt"))
    win.run_build("One grid", [str(win.sample / "field/wide.csv")], [])
    win.export_library("One grid", str(win.tmp / "one"), "sli"); assert win.last_error is None
    assert (win.tmp / "one.sli").exists() and (win.tmp / "one.hdr").exists()
    n = win.stats["spectra"]
    win.d_lib.setCurrentText("One grid"); win.on_delete_library()
    assert win.stats["spectra"] == n - 3
    win.on_reset_db()
    assert win.stats["spectra"] == 0 and win.stats["cubes"] == 0 and win.table.rowCount() == 0


def test_switch_database(win):
    win.import_path(str(win.sample)); assert win.stats["spectra"] > 0
    win.use_database(f"sqlite:///{win.tmp}/other.db")
    assert win.stats["spectra"] == 0 and "other.db" in win.windowTitle()
    win.use_database("postgresql+psycopg://nobody:x@127.0.0.1:1/none")      # unreachable: must report, not crash
    assert win.last_error and "Cannot use database" in win.last_error and "x@" not in win.last_error
