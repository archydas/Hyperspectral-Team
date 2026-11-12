"""DS3 shell end to end (offscreen): home counts, Library cards + material search, Image module with the Library
tab (collect, angles, best matches, library SAM map, spectral index), hand-over to Analysis, every menu entry.
Run:  QT_QPA_PLATFORM=offscreen python -m pytest -q tests/test_ds3_gui.py"""
import csv
import os
import time
import types

import numpy as np
import pytest

pytest.importorskip("PyQt5")
pytest.importorskip("sklearn")
envi = pytest.importorskip("spectral.io.envi")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _cube(path):
    rows = list(csv.reader(open(os.path.join(ROOT, "data", "material_spectra.csv"))))
    data = np.array([[float(x) for x in r] for r in rows[1:]])
    sel = np.arange(0, len(data), max(1, len(data) // 120))[:120]
    wl, S = data[sel, 0], data[sel, 1:]
    cube = np.zeros((40, 80, len(wl)), np.float32)
    for i, k in enumerate((0, 1, 6, 5)):                 # concrete, brick, marble, glass
        cube[:, i * 20:(i + 1) * 20] = S[:, k]
    cube += np.random.default_rng(0).normal(0, 0.003, cube.shape).astype(np.float32)
    envi.save_image(str(path), cube, metadata={"wavelength": [f"{w:g}" for w in wl]}, force=True, interleave="bsq")
    return [rows[0][1:][k] for k in (0, 1, 6, 5)]


def test_ds3_end_to_end(tmp_path, monkeypatch):
    from PyQt5 import QtCore, QtWidgets
    W = QtWidgets
    QtCore.QSettings.setDefaultFormat(QtCore.QSettings.IniFormat)
    QtCore.QSettings.setPath(QtCore.QSettings.IniFormat, QtCore.QSettings.UserScope, str(tmp_path / "settings"))
    app = W.QApplication.instance() or W.QApplication([])
    for n in ("information", "warning", "critical", "about"):
        monkeypatch.setattr(W.QMessageBox, n, staticmethod(lambda *a, **k: W.QMessageBox.Ok))
    monkeypatch.setattr(W.QMessageBox, "question", staticmethod(lambda *a, **k: W.QMessageBox.Yes))
    monkeypatch.setattr(W.QDialog, "exec_", lambda self: 0)
    monkeypatch.setattr(W.QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: (str(tmp_path / "out.csv"), "")))
    names = _cube(tmp_path / "synth.hdr")
    from ds3 import style
    style.apply(app)
    from ds3.shell import DS3Window

    def pump(sec):
        t = time.time()
        while time.time() - t < sec:
            app.processEvents()
            time.sleep(0.02)

    def wait(cond, sec=20):
        t = time.time()
        while time.time() - t < sec and not cond():
            pump(0.1)
        return cond()

    w = DS3Window()
    w.show()
    lp = w.libpanel
    lp.cfg.update(mode="direct", db_url=f"sqlite:///{tmp_path / 'lib.db'}", data_folder="")
    lp.connect_db(init=True)
    # main libraries are imported from data/libraries automatically
    assert wait(lambda: len(lp.libraries or []) == 2 and all(l.get("spectrum_count") for l in lp.libraries), 60)
    w.show_module("home")
    pump(0.3)
    assert wait(lambda: w.library.main_names >= {"Material Spectra", "Soil Spectra"})
    # material search across libraries -> open in its library
    w.show_module("library")
    w.library.ed_mats.setText("brick")
    w.library.on_material_search()
    assert wait(lambda: w.library.lst_res.count() == 3)
    w.library.on_result_open(w.library.lst_res.item(0))
    assert wait(lambda: lp.active_material is not None and "Brick" in lp.active_material["name"])
    assert wait(lambda: len(w.library.plot.ax.lines) >= 1)
    # Image module
    w.open_image_path(str(tmp_path / "synth.hdr"))
    v = w.image.viewer
    assert wait(lambda: v.cube is not None) and v.cube.shape == (40, 80, 120)
    L = w.image.link
    ax = v.image_canvas.ax

    def click(r, c):
        L.on_click(types.SimpleNamespace(inaxes=ax, xdata=c, ydata=r, button=1, dblclick=False))
    L.btn_collect.setChecked(True)
    for c in (5, 25, 45, 65):
        click(10, c)
    assert L.tbl.rowCount() == 4
    L.on_angles()
    L.on_csv()
    assert (tmp_path / "out.csv").exists()
    L.btn_collect.setChecked(False)
    click(20, 45)
    L.tbl.clearSelection()
    L.on_best_matches()
    assert wait(lambda: "White Marble" in w.statusBar().currentMessage())
    L.on_classify_library()
    idx, cls = L._class_map, L._class_names
    assert idx is not None
    for i, k in enumerate(names):                        # every column block is classified as its material
        assert cls[int(np.bincount(idx[:, i * 20:(i + 1) * 20].ravel()).argmax())] == " ".join(k.split())
    L.on_back()
    w.on_index("NDVI", "(w800 - w670) / (w800 + w670)")
    assert "B" in v.bandmath_edit.text()
    # DS3 'Class spectra from map': ground-truth map -> one mean spectrum per class -> new library
    from hvva.gui.library_panel import SaveDialog
    lab = np.zeros((40, 80), np.int32)
    lab[:, :20], lab[:, 40:60] = 1, 2
    np.save(tmp_path / "gt.npy", lab)
    monkeypatch.setattr(lp, "_ask_target", lambda *a, **k: {"library_id": SaveDialog.NEW, "new_library": "GT classes",
                                                             "dataset": "gt", "category": None, "acquisition": None})
    L.on_class_map_means(str(tmp_path / "gt.npy"))
    assert wait(lambda: any(l["name"] == "GT classes" and l.get("material_count") == 2 for l in lp.libraries or []))
    L.on_send()
    an = w.analysis.win
    assert len(an.scenes) == 1 and not an.scenes[0].rois
    # DS3 layout: no ROI Analysis tab in the Viewer, no Georeferencing tab in Data Aggregation
    assert "ROI Analysis" not in [v.tabs.tabText(i) for i in range(v.tabs.count())]
    assert "Georeferencing" not in [an.tabs.tabText(i) for i in range(an.tabs.count())]
    assert "ROI Analysis" in [an.tabs.tabText(i) for i in range(an.tabs.count())]
    # ROIs drawn in Data Aggregation -> Viewer (rectangle + polygon), then used by the viewer's ROI picker
    sc = an.scenes[0]
    sc.add_roi({"kind": "rect", "rect": (5, 2, 15, 12), "class": "concrete", "name": "r1"})
    sc.add_roi({"kind": "polygon", "vertices": [(42, 5), (58, 5), (58, 20), (42, 20)], "class": "marble", "name": "p1"})
    an.roipanel.refresh()                                # what every ROI edit in Data Aggregation calls
    RL = w.image.roilink
    assert "ROIs (Data Aggregation)" in [v.tabs.tabText(i) for i in range(v.tabs.count())]
    assert [r["class"] for r in v.rois] == ["concrete", "marble"]          # synced automatically
    assert int(v.rois[0]["mask"].sum()) == 100 and v.rois[0]["rect"] == (5, 14, 2, 11)
    assert v.roi_list.count() == 2 and "Linked to" in RL.lbl_link.text()
    # a pixel-list ROI (e.g. from a label map) comes over as a mask ROI; a delete in DA is mirrored too
    sc.add_roi({"kind": "pixels", "pixels": [[30, 70], [30, 71], [31, 70]], "class": "glass", "name": "px"})
    an.roipanel.refresh()
    assert [r["kind"] for r in v.rois] == ["rect", "polygon", "mask"] and int(v.rois[2]["mask"].sum()) == 3
    sc.rois.pop(2)
    an.roipanel.refresh()
    assert len(v.rois) == 2
    # tools that need 'selected ROIs': list selection first, otherwise a picker
    v.roi_list.item(0).setSelected(True)
    m = v._selected_roi_union("dark")
    assert m is not None and m.sum() == 100
    v.roi_list.clearSelection()
    monkeypatch.setattr(W.QInputDialog, "getItem", staticmethod(lambda *a, **k: (a[3][1], True)))
    m = v._selected_roi_union("dark")
    assert m is not None and m.sum() == v.rois[1]["mask"].sum()
    v.on_show_class_spectra()                            # viewer's own ROI plots work on the synced ROIs
    assert len(v.plot_canvas.ax.lines) >= 2
    RL.chk_sync.setChecked(False)
    sc.add_roi({"kind": "rect", "rect": (30, 62, 38, 78), "class": "glass", "name": "r3"})
    an.roipanel.refresh()
    assert len(v.rois) == 2                              # sync off: unchanged
    RL.chk_sync.setChecked(True)                         # turning it on pulls the change
    assert len(v.rois) == 3
    L.select_or_send()                                   # already there: selected, not sent again
    assert len(an.scenes) == 1
    # every menu entry that does not open a file dialog
    skip = ("…", "Exit", "Processing History")
    for top in w.menuBar().actions():
        stack = [top.menu()]
        while stack:
            menu = stack.pop()
            menu.aboutToShow.emit()
            for a in menu.actions():
                if a.menu():
                    stack.append(a.menu())
                elif a.text() and a.isEnabled() and not any(s in a.text() for s in skip):
                    a.trigger()
                    pump(0.1)
    w.close()
    pump(0.3)
