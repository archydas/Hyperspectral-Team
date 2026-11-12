"""Headless walk through the v3.4 viewer / library features (run by test_gui_v34.py):
python tests/gui_v34_script.py <project dir> <database url> [screenshot dir]

Band selection, zoom, pixel window, collected spectra, SAM / RMSE matching with unit warnings, search filters,
ROI best matches, GeoJSON / ENVI ASCII ROI files, N/S/E/W bounds and control points, map view (map tiles are
faked - no network needed), Library Builder, library resampling and export, images table."""
import os, sys, json, tempfile, time
os.environ["QT_QPA_PLATFORM"] = "offscreen"
sys.path.insert(0, sys.argv[1])
url = sys.argv[2]
out = sys.argv[3] if len(sys.argv) > 3 else "/tmp/v34shots"
os.makedirs(out, exist_ok=True)
import numpy as np
from PyQt5 import QtWidgets, QtCore
from types import SimpleNamespace as NS
app = QtWidgets.QApplication([])
QtCore.QCoreApplication.setOrganizationName("hvva-test-" + str(os.getpid()))
from hvva.gui import main_window as mw, library_dialogs as LD, viewer as V
from hvva.core import basemap
from hvva.library.client import DirectClient
W = QtWidgets

# ---- offline tile server: checkerboard tiles
calls = []
def fake_tile(provider, z, x, y, url=None, timeout=10):
    calls.append((z, x, y))
    t = np.full((256, 256, 3), 0.8, np.float32); t[::32] = 0.3; t[:, ::32] = 0.3
    return t
basemap.fetch_tile = fake_tile

mw.LibraryPanel.connect_db = lambda self, *a, **k: None
w = mw.LibraryBuilderWindow(); w.resize(1500, 980); w.show()
lp, roi, vb, cv = w.libpanel, w.roipanel, w.viewer, w.image_canvas
def wait(cond=lambda: True, t=60):
    end = time.time() + t
    while time.time() < end:
        app.processEvents()
        if not lp._tasks and cond():
            return
        time.sleep(0.02)
    raise SystemExit("timeout")
def say():
    return w.info_label.text()
# no dialogs
W.QMessageBox.question = staticmethod(lambda *a, **k: W.QMessageBox.Yes)
saves, opens = [], []
W.QFileDialog.getSaveFileName = staticmethod(lambda *a, **k: (saves.pop(0), ""))
W.QFileDialog.getOpenFileName = staticmethod(lambda *a, **k: (opens.pop(0), ""))

w.on_demo(); w.tbl_scenes.selectRow(0); wait()
sc = w.current_scene()

# ---------------- A/B: band selection
assert vb.cb_r.count() == sc.n_bands
vb.cb_mode.setCurrentIndex(1); wait()
assert sc.display["mode"] == "gray" and cv.ax.images[0].get_array().ndim == 2, sc.display
vb.cb_mode.setCurrentIndex(0); vb.cb_preset.setCurrentIndex(2); vb._on_preset(2); wait()
assert sc.display["mode"] == "rgb" and len(sc.display["bands"]) == 3
assert "RGB" in cv.ax.get_title()
vb.sp_lo.setValue(5); wait(); assert sc.display["stretch"][0] == 5
# zoom kept on redraw
cv.ax.set_xlim(10, 30); cv.ax.set_ylim(30, 10); w.show_scene(); wait()
assert tuple(np.round(cv.ax.get_xlim())) == (10, 30), cv.ax.get_xlim()
cv.reset_view(); assert cv.ax.get_xlim()[1] > 50
# wheel zoom
x0 = cv.ax.get_xlim(); cv._on_scroll(NS(inaxes=cv.ax, xdata=20, ydata=20, button="up")); assert np.ptp(cv.ax.get_xlim()) < np.ptp(x0)
print("band selection / zoom ok:", sc.display)

# ---------------- pixel window + collect
vb.cb_win.setCurrentIndex(1)                    # 3x3
w.on_image_click(NS(inaxes=cv.ax, xdata=12, ydata=14, button=1))
exp = np.nanmean(sc.cube[13:16, 11:14].reshape(-1, sc.n_bands), axis=0)
assert np.allclose(np.nan_to_num(w.last_spectrum[1]), np.nan_to_num(sc.masked(exp))), "3x3 mean"
vb.chk_collect.setChecked(True)
for xy in ((5, 5), (60, 60), (61, 62), (95, 20)):
    w.on_image_click(NS(inaxes=cv.ax, xdata=xy[0], ydata=xy[1], button=1))
assert len(w.collected) == 4 and vb.btn_collected.text() == "Collected (4)…"
assert len(w._marker_artists) == 8
w.on_show_collected(); wait()
dlg = w._collected_dlg
A = V.angle_matrix(w.collected)
assert np.allclose(A, A.T) and np.all(np.isfinite(A)) and A[1, 2] < A[0, 3], A
assert dlg.ang.rowCount() == 4
csvp = os.path.join(out, "collected.csv"); saves.append(csvp); dlg.on_export()
import csv as _csv
rows = list(_csv.reader(open(csvp))); assert len(rows[0]) == 5 and len(rows) == sc.n_bands + 1
dlg.grab().save(os.path.join(out, "collected_dialog.png"))
print("collect ok:", np.round(A, 2).tolist()[0])

# ---------------- library connection
lp.client = DirectClient(url, init_schema=True); lp._set_connected(True, "test"); lp._sync_buttons()
for l in lp.client.list_libraries():
    if l["name"].startswith(("v34", "Built", "Refl")):
        lp.client.delete_library(l["id"])
lib = lp.client.create_library("v34 refl", "test")
lp.client.save_spectra(lib["id"], "seed", [{"name": f"ref_{i}", "wavelengths": list(map(float, sc.wavelengths)),
                                            "values": list(map(float, np.nan_to_num(sc.cube[i * 20 + 5, i * 20 + 5]))),
                                            "unit": "reflectance", "purity": "image_pixel", "category": "Test/Pixels"}
                                           for i in range(5)])
rad = lp.client.create_library("v34 radiance", "test")
lp.client.save_spectra(rad["id"], "seed", [{"name": "rad_1", "wavelengths": list(map(float, sc.wavelengths)),
                                            "values": list(map(float, 100 * np.nan_to_num(sc.cube[5, 5]))),
                                            "unit": "radiance", "purity": "ground_measured"}])
lp.load_libraries(select_id=lib["id"]); wait(); wait(lambda: lp.active_lib is not None)
assert lp.active_lib["name"] == "v34 refl"

# collected -> library
orig_ask = lp._ask_target
lp._ask_target = lambda *a, **k: {"library_id": lib["id"], "new_library": "", "dataset": "collected", "category": "Collected",
                                  "acquisition": "airborne", "name": ""}
dlg.tbl.selectRow(0); dlg.on_save(); wait()
mats = lp.client.list_materials(lib["id"], None, None, 100)["items"]
assert any("(5, 5)" in m["name"] for m in mats), [m["name"] for m in mats]

# ---------------- match SAM / RMSE / unit warnings
lp.cb_scope.setCurrentIndex(1)                               # all libraries
lp.cb_method.setCurrentIndex(lp.cb_method.findData("sam"))
w.last_spectrum = (sc.wavelengths, sc.masked(sc.cube[25, 25].astype(float))); w.last_pixel = (25, 25); w.last_window = 1
lp.on_match(); wait()
t = lp.txt_match.toPlainText(); print(t[:600])
assert "ref_1" in t.splitlines()[2] or "ref_1" in t, t
assert "v34 radiance is radiance but the query is reflectance" in t
lp.cb_method.setCurrentIndex(lp.cb_method.findData("rmse")); lp.on_match(); wait()
t = lp.txt_match.toPlainText()
assert "v34 radiance skipped" in t and "RMSE" in t and "rad_1" not in t, t
lp.cb_method.setCurrentIndex(0); lp.cb_scope.setCurrentIndex(0)

# ---------------- filters
lp.cb_unitf.setCurrentIndex(lp.cb_unitf.findData("radiance")); wait()
assert lp.mat_total == 0, lp.mat_total
lp.cb_unitf.setCurrentIndex(0); wait(); n_all = lp.mat_total
lp.cb_purf.setCurrentIndex(lp.cb_purf.findData("image_pixel")); wait()
assert lp.mat_total == n_all - 1, (lp.mat_total, n_all)   # the collected one is synthetic (demo scene)
lp.cb_purf.setCurrentIndex(0); wait()
lp.ed_covers.setText("500-900"); lp._on_covers(); wait(); assert lp.mat_total == n_all
lp.ed_covers.setText("300-2400"); lp._on_covers(); wait(); assert lp.mat_total == 0
lp.ed_covers.setText(""); lp._on_covers(); lp.cb_purf.setCurrentIndex(0); wait()
from hvva.gui.library_panel import parse_covers
assert parse_covers("2400 - 450") == (450, 2400) and parse_covers("abc") is None
print("filters ok, materials:", n_all)

# ---------------- ROI best matches
roi.tbl.clearSelection(); w.tbl_scenes.selectRow(2); wait()
assert w.current_scene().rois, "demo C has ROIs"
roi.on_best_matches(); wait()
t = lp.txt_match.toPlainText(); assert t.count("■") == len(w.current_scene().rois), t
dlgs = [d for d in w.findChildren(W.QDialog) if d.windowTitle() == "ROI means vs library"]
assert dlgs; dlgs[-1].grab().save(os.path.join(out, "roi_matches.png"))
print("roi matches ok")

# ---------------- ROI files: GeoJSON + ENVI ASCII via the panel
sc_c = w.current_scene(); n0 = len(sc_c.rois)
sc_c.add_roi({"kind": "pixels", "pixels": [[3, 4], [3, 5], [10, 11]], "class": "pts"})
roi.refresh(); roi.tbl.clearSelection()
gj = os.path.join(out, "rois.geojson"); saves.append(gj); roi.on_save_json()
doc = json.load(open(gj)); assert doc["crs"]["properties"]["name"] == "EPSG:32633" and len(doc["features"]) >= n0 + 1
assert any(f["geometry"]["type"] == "MultiPoint" for f in doc["features"])
ea = os.path.join(out, "rois.txt"); saves.append(ea); roi.on_save_json()
assert "; ROI name:" in open(ea).read()
w.tbl_scenes.selectRow(1); wait(); sc_b = w.current_scene(); sc_b.rois = []; roi.refresh()
opens.append(gj); roi.on_load_json(); wait()
print("geojson onto B:", say())
roi._ask_replace = lambda targets: False          # append (no modal box in the test)
opens.append(ea); roi.on_load_json()
print("envi onto B:", say())
assert any(r["kind"] == "pixels" for r in sc_b.rois)
assert any(r["kind"] == "polygon" for r in sc_b.rois)

# ---------------- georeferencing tab
w.tbl_scenes.selectRow(0); wait(); sc = w.current_scene()
g0 = sc.geo
w.tabs.setCurrentIndex(w.tabs.indexOf(w.tabs.widget(7))); wait()
assert "Corners (lat, lon)" in w.txt_geo.toPlainText(), w.txt_geo.toPlainText()
n, s_, e, ww = w.sp_n.value(), w.sp_s.value(), w.sp_e.value(), w.sp_w.value()
print("bounds:", n, s_, e, ww)
w.cb_bcrs.setCurrentIndex(1); w.on_geo_bounds()
assert sc.geo.crs == "EPSG:32633", sc.geo.crs
assert abs(sc.geo.transform[2] - g0.transform[2]) < 60 and abs(sc.geo.transform[5] - g0.transform[5]) < 60, (sc.geo.transform, g0.transform)
w.cb_bcrs.setCurrentIndex(0); w.on_geo_bounds(); assert sc.geo.crs == "EPSG:4326"
# GCPs: points from the original UTM transform
w.ed_gcrs.setText("EPSG:32633")
w.tbl_gcp.setRowCount(0)
for (r, c) in ((0, 0), (0, 90), (80, 5), (70, 80)):
    k = w._gcp_row(c, r); x, y = g0.xy(r, c)
    w.tbl_gcp.item(k, 2).setText(f"{x:.3f}"); w.tbl_gcp.item(k, 3).setText(f"{y:.3f}")
w.on_gcp_pick(); w.on_image_click(NS(inaxes=cv.ax, xdata=40, ydata=30, button=1))
assert w.tbl_gcp.rowCount() == 5 and w.tbl_gcp.item(4, 0).text() == "40"
x, y = g0.xy(30, 40); w.tbl_gcp.item(4, 2).setText(str(x)); w.tbl_gcp.item(4, 3).setText(str(y))
w.on_gcp_fit(True)
assert np.allclose(sc.geo.transform, g0.transform, atol=1e-3), sc.geo.transform
assert "north-up" in w.lbl_gcp.text() and sc.geo.is_north_up
assert "RMSE 0" in w.lbl_gcp.text() or "RMSE 1e" in w.lbl_gcp.text() or "e-" in w.lbl_gcp.text(), w.lbl_gcp.text()
print("gcp:", w.lbl_gcp.text())

# ---------------- map view
w.tbl_scenes.selectRow(2); wait(); sc = w.current_scene()
roi.tbl.clearSelection()
w.on_geo_show_map(); wait(); wait(lambda: cv._basemap is not None, 20)
assert cv.map_mode and cv._basemap is not None and calls, "map shown"
assert "OpenStreetMap" in cv.attribution.get_text()
M = basemap.scene_affine(sc)
X, Y = (M @ np.array([30.5, 20.5, 1.0]))                      # centre of pixel (row 20, col 30)
rc = cv.event_pixel(NS(inaxes=cv.ax, xdata=X, ydata=Y)); assert np.allclose(rc, (20, 30), atol=1e-6), rc
vb.chk_collect.setChecked(False)
w.on_image_click(NS(inaxes=cv.ax, xdata=X, ydata=Y, button=1)); assert w.last_pixel == (20, 30)
assert len(roi._artists) >= len(sc.rois) and roi._selector is None
print("format_coord:", cv.ax.format_coord(X, Y))
assert "°" in cv.ax.format_coord(X, Y)
cv.ax.set_xlim(cv.ax.get_xlim()[0] + 100, cv.ax.get_xlim()[1] - 100); wait(); time.sleep(0.6); wait(); wait()
vb.cb_provider.setCurrentIndex(vb.cb_provider.count() - 1); vb.cb_provider.activated.emit(vb.cb_provider.count() - 1); wait()
assert cv._basemap is None
vb.cb_provider.setCurrentIndex(1); vb.cb_provider.activated.emit(1); wait(); wait(lambda: cv._basemap is not None, 20)
w.show_scene(); wait(); wait(lambda: cv._basemap is not None, 20)
for _ in range(5): app.processEvents()
w.grab().save(os.path.join(out, "map_view.png"))
# offline -> message, image still placed
def boom(*a, **k): raise OSError("network unreachable")
basemap.fetch_tile = boom
vb.cb_provider.setCurrentIndex(0); vb.cb_provider.activated.emit(0); wait(); time.sleep(0.2); wait()
assert "Map tiles" in say(), say()
basemap.fetch_tile = fake_tile
vb.chk_map.setChecked(False); wait(); assert not cv.map_mode
print("map ok, tiles fetched:", len(calls))

# ---------------- builder
b = LD.BuilderDialog(lp)
b.add_collected(); b.add_rois(); b.cb_lib.setCurrentIndex(b.cb_lib.findData(lib["id"])); b.add_library(); wait()
n_in = len(b.inputs); assert n_in >= 10, n_in
b.grid_ch.cb.setCurrentIndex(2); b.grid_ch.sp_r0.setValue(450); b.grid_ch.sp_r1.setValue(950); b.preview(); print("\n".join(l for l in b.report.toPlainText().splitlines() if "✗" in l))
b.grid_ch.cb.setCurrentIndex(2); b.grid_ch.sp_r0.setValue(450); b.grid_ch.sp_r1.setValue(950); b.grid_ch.sp_rs.setValue(10)
b.preview(); rep = b.report.toPlainText(); print(rep.splitlines()[0]); print(rep.splitlines()[1])
assert b.result and len(b.result.added) + len(b.result.rejected) == n_in and b.result.grid.size == 51
assert all("already in the list" in why for _i, why in b.result.rejected)
n_ok = len(b.result.added)
b.grid_ch.sp_r0.setValue(300); b.grid_ch.sp_r1.setValue(2400); b.sp_cov.setValue(50); b.preview()
assert len(b.result.added) == 0 and "covers only" in b.report.toPlainText()
b.grid_ch.cb.setCurrentIndex(0); b.grid_ch.sp_b0.setValue(5); b.grid_ch.sp_b1.setValue(40); b.preview()
assert b.result.grid.size <= 36 and len(b.result.added) == n_ok
b.grid_ch.cb.setCurrentIndex(1); b.grid_ch.txt.setPlainText("0.5 0.01\n0.6 0.01\n0.7 0.012\n0.8 0.012")
b.cb_method.setCurrentIndex(1); b.preview(); assert list(b.result.grid) == [500, 600, 700, 800]
b.show(); app.processEvents(); b.grab().save(os.path.join(out, "builder.png"))
lp._ask_target = lambda *a, **k: {"library_id": "__new__", "new_library": "Built v34", "dataset": "builder",
                                  "category": None, "acquisition": None, "name": ""}
from hvva.gui.library_panel import SaveDialog
lp._ask_target = (lambda f: (lambda *a, **k: dict(f(), library_id=SaveDialog.NEW)))(lambda: {"new_library": "Built v34",
                  "dataset": "builder", "category": None, "acquisition": None, "name": ""})
b.save(); wait()
built = next(l for l in lp.client.list_libraries() if l["name"] == "Built v34")
ents = lp.client.library_spectra(built["id"], all_samples=True)
assert len(ents) >= n_ok - 2 and all(len(e["spectrum"]["values"]) == 4 for e in ents), len(ents)
assert ents[0]["spectrum"]["wavelengths"] == [500, 600, 700, 800]
assert built["unit"] == "reflectance", built
print("builder ok:", len(ents))

# ---------------- resample + export
lp.load_libraries(select_id=lib["id"]); wait(); wait(lambda: lp.active_lib and lp.active_lib["id"] == lib["id"])
r = LD.ResampleDialog(lp, lp.active_lib)
r.grid_ch.cb.setCurrentIndex(2); r.grid_ch.sp_r0.setValue(450); r.grid_ch.sp_r1.setValue(950); r.grid_ch.sp_rs.setValue(25)
r.ed_name.setText("v34 refl 25nm"); r.go(); wait()
print(r.lbl.text())
rl = next(l for l in lp.client.list_libraries() if l["name"] == "v34 refl 25nm")
assert len(lp.client.library_spectra(rl["id"])[0]["spectrum"]["values"]) == 21
lp.load_libraries(select_id=lib["id"]); wait(); wait(lambda: lp.active_lib and lp.active_lib["id"] == lib["id"])
x = LD.ExportLibraryDialog(lp, lp.active_lib)
sli = os.path.join(out, "lib.sli"); saves.append(sli); x.go(); wait()
print(x.lbl.text())
assert os.path.exists(sli) and os.path.exists(os.path.join(out, "lib.hdr"))
x.cb_fmt.setCurrentIndex(1); x.cb_grid.setCurrentIndex(2); x.grid_ch.cb.setCurrentIndex(2)
x.grid_ch.sp_r0.setValue(500); x.grid_ch.sp_r1.setValue(900); x.grid_ch.sp_rs.setValue(100)
c2 = os.path.join(out, "lib.csv"); saves.append(c2); x.go(); wait()
assert open(c2).read().splitlines()[0].startswith("wavelength_nm,") and len(open(c2).read().splitlines()) == 6

# ---------------- images
im = LD.ImagesDialog(w)
wait()
before = im.tbl.rowCount()
w.tbl_scenes.selectRow(0); wait(); im.register(); wait(); wait()
assert im.tbl.rowCount() == before + 1, (before, im.tbl.rowCount())
im.register(); wait(); wait(); assert im.tbl.rowCount() == before + 1      # same fingerprint -> updated, not added
im.show(); app.processEvents(); im.grab().save(os.path.join(out, "images.png"))
im.tbl.selectRow(0); im.open_selected()
im.tbl.selectRow(0); im.remove(); wait(); wait()
print("images ok:", before, "->", im.tbl.rowCount())

# ---------------- screenshots
w.tbl_scenes.selectRow(0); vb.chk_collect.setChecked(True); wait()
w.tabs.setCurrentIndex(w.tabs.indexOf(lp)); wait()
for _ in range(5): app.processEvents()
w.grab().save(os.path.join(out, "library_tab.png"))
w.tabs.setCurrentIndex(7); w.geo_tabs.setCurrentIndex(2); wait()
w.grab().save(os.path.join(out, "geo_tab.png"))
print("GUI v3.4 OK")
