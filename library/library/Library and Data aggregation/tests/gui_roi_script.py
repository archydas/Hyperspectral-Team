"""Headless walk through the ROI Analysis tab (run by test_gui_rois.py):
python tests/gui_roi_script.py <project dir> <database url>"""
import os, sys, json, tempfile, time
os.environ["QT_QPA_PLATFORM"] = "offscreen"
sys.path.insert(0, sys.argv[1])
url = sys.argv[2]
from PyQt5 import QtWidgets, QtCore
from types import SimpleNamespace as NS
app = QtWidgets.QApplication([])
QtCore.QCoreApplication.setOrganizationName("hvva-test-" + str(os.getpid()))
from hvva.gui import main_window as mw
from hvva.gui import roi_panel as rp
from hvva.library.client import DirectClient
mw.LibraryPanel.connect_db = lambda self, *a, **k: None     # no autoconnect to the default DB
w = mw.LibraryBuilderWindow()
lp, roi = w.libpanel, w.roipanel
def wait(cond=lambda: True, t=30):
    end = time.time() + t
    while time.time() < end:
        app.processEvents()
        if not lp._tasks and cond():
            return
        time.sleep(0.02)
    raise SystemExit("timeout")
wait()
w.on_demo()
assert w.tabs.tabText(1) == "ROI Analysis"
print("rois after demo:", roi.tbl.rowCount(), roi.lbl_summary.text())
assert roi.tbl.rowCount() == 3
# draw on scene A (row 0)
w.tbl_scenes.selectRow(0); app.processEvents()
roi.chk_enable.setChecked(True)
assert isinstance(roi._selector, rp.RectangleSelector)
roi.cb_class.setEditText("vegetation")
roi._on_rect(NS(xdata=2.2, ydata=3.0), NS(xdata=8.7, ydata=9.4))
assert roi.pending["rect"] == (3, 2, 10, 10), roi.pending
roi.on_add_selection()
roi.cb_shape.setCurrentIndex(1); app.processEvents()
assert isinstance(roi._selector, rp.PolygonSelector)
roi.cb_class.setEditText("soil")
roi._on_polygon([(20, 20), (40, 22), (35, 40)])
roi.on_add_selection()
# scene B
w.tbl_scenes.selectRow(1); app.processEvents()
roi.cb_class.setEditText("vegetation"); roi.chk_auto.setChecked(True)
roi.cb_shape.setCurrentIndex(0)
roi._on_rect(NS(xdata=50, ydata=50), NS(xdata=60, ydata=58))
print(roi.lbl_summary.text())
assert roi.tbl.rowCount() == 6 and len(w.scenes[1].rois) == 1
assert len(roi._artists) >= 2
# click on view must not plot while selecting
w.on_image_click(NS(inaxes=w.image_canvas.ax, xdata=5, ydata=5)); assert w.last_pixel is None
roi.cb_view.setCurrentIndex(1); assert roi.tbl.rowCount() == 1; roi.cb_view.setCurrentIndex(0)
# plots
roi.tbl.selectRow(3); roi.on_plot_rois()
roi.tbl.clearSelection(); roi.on_plot_classes()
labels = [l.get_label() for l in w.plot_canvas.ax.get_lines()]
print("class plot:", labels)
assert any(l.startswith("vegetation") and "3 img" in l for l in labels)
# anomaly view
roi.btn_anomaly.setChecked(True); roi.on_anomaly()
assert "anomaly" in w.image_canvas.ax.get_title()
roi.btn_anomaly.setChecked(False); roi.on_anomaly()
# set class / rename / delete
row = next(i for i, (s, r) in enumerate(roi.rows) if r["class"] == "soil" and s is w.scenes[0])
roi.tbl.selectRow(row)
QtWidgets.QInputDialog.getItem = staticmethod(lambda *a, **k: ("bare soil", True))
roi.on_set_class()
assert "bare soil" in roi.class_names()
QtWidgets.QInputDialog.getText = staticmethod(lambda *a, **k: ("my polygon", True))
roi.tbl.selectRow(next(i for i, (s, r) in enumerate(roi.rows) if r["class"] == "bare soil")); roi.on_rename()
assert any(r["name"] == "my polygon" for r in w.scenes[0].rois)
# Aggregation uses polygons too
assert "bare soil" in w.scenes[0].class_masks()
# JSON save / load
tmp = tempfile.mkdtemp(); jp = os.path.join(tmp, "r.json")
roi.tbl.clearSelection()
QtWidgets.QFileDialog.getSaveFileName = staticmethod(lambda *a, **k: (jp, ""))
roi.on_save_json()
d = json.load(open(jp)); assert d["format"] == "DS3-ROIs" and len(d["rois"]) == 6
QtWidgets.QFileDialog.getOpenFileName = staticmethod(lambda *a, **k: (jp, ""))
class MB(QtWidgets.QMessageBox):
    def exec_(self):
        for b in self.buttons():
            if b.text() == "Append": self._clicked = b
        return 0
    def clickedButton(self): return getattr(self, "_clicked", None)
_QMB = QtWidgets.QMessageBox
rp.W.QMessageBox = MB
QtWidgets.QMessageBox.question = staticmethod(lambda *a, **k: QtWidgets.QMessageBox.Yes)
n_before = len(roi.all_items())
roi.on_load_json()
assert len(roi.all_items()) == n_before, "same ids replace, not duplicate"
rp.W.QMessageBox = _QMB
# ---- database
lp.client = DirectClient(url, init_schema=True); lp._set_connected(True, lp.client.describe())
lp.load_libraries(); wait()
dialog_choice = {}
class SD(rp.RoiSaveDialog):
    def exec_(self):
        self.rb_scope["all"].setChecked(True)
        self.rb_mode[dialog_choice["mode"]].setChecked(True)
        self.cb_lib.setCurrentIndex(self.cb_lib.findData(rp.SaveDialog.NEW)) if dialog_choice.get("new") else None
        self.ed_newlib.setText("ROI test lib")
        self.cb_cat.setEditText("Field ROIs")
        self.sp_px.setValue(3)
        return True
rp.RoiSaveDialog = SD
dialog_choice.update(mode="class", new=True)
roi.on_save_db(); wait()
print(w.info_label.text())
libs = lp.client.list_libraries(); lib = next(l for l in libs if l["name"] == "ROI test lib")
dialog_choice.update(mode="roi", new=False)
lp.active_lib = lib
roi.on_save_db(); wait()
print(w.info_label.text())
dialog_choice.update(mode="pixels")
roi.on_save_db(); wait()
print(w.info_label.text())
rows = lp.client.list_rois(lib["id"])
assert len(rows) == 6, len(rows)
cls = {c["class_name"]: c for c in lp.client.roi_classes(lib["id"])}
print({k: (v["n_rois"], v["n_scenes"], v["n_spectra"]) for k, v in cls.items()})
assert cls["vegetation"]["n_scenes"] == 3
mats = lp.client.list_materials(lib["id"])["items"]
print(sorted((m["name"], m["category"]) for m in mats))
assert all(m["category"] == "Field ROIs/" + m["name"] for m in mats)
assert all(r.get("saved_to") == ["ROI test lib"] for _s, r in roi.all_items())
# material info shows ROI provenance
veg = next(m for m in mats if m["name"] == "vegetation")
lp.open_material(veg["id"]); wait(lambda: lp.spectrum is not None)
print("info:", lp.lbl_mat.text()[-200:])
assert "ROI class" in lp.lbl_mat.text()
# load ROIs back from db onto a fresh copy of scene A
from hvva.core.scene import Scene
sA = w.scenes[0]
copy = Scene(sA.name + "_copy", sA.cube.copy(), sA.wavelengths, sA.geo)
w.add_scene(copy); app.processEvents()
class LD(rp.RoiLoadDialog):
    def exec_(self):
        print("load dialog rows:", self.tbl.rowCount(), self.lbl.text()[:120])
        self._check(True); self._add(); return True
rp.RoiLoadDialog = LD
roi.on_load_db(); wait()
print("copy rois:", [(r["class"], r["kind"]) for r in copy.rois])
assert len(copy.rois) == 2   # demo scenes are georeferenced on the same grid -> placed through map coordinates
# delete one ROI from db with spectra
class LD2(rp.RoiLoadDialog):
    def exec_(self):
        self.chk_fit.setChecked(False)
        self.tbl.selectRow(0)
        rec = self.rows[0][0]
        self.panel.delete_db_rois([rec], True, self)
        return True
rp.RoiLoadDialog = LD2
roi.on_load_db(); wait(); wait()
print(w.info_label.text())
assert len(lp.client.list_rois(lib["id"])) == 5
# scene save keeps ROIs
from hvva.core.scene import save_scene, load_scene
p = os.path.join(tmp, "a.npz"); save_scene(w.scenes[0], p); back = load_scene(p)
assert [r["id"] for r in back.rois] == [r["id"] for r in w.scenes[0].rois]
assert back.rois[0].get("saved_to")
roi.chk_enable.setChecked(False)
w.on_image_click(NS(inaxes=w.image_canvas.ax, xdata=5, ydata=5)); assert w.last_pixel == (5, 5)
w.close()
print("GUI OK")
