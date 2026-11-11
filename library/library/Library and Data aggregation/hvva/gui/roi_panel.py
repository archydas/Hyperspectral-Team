"""ROI Analysis tab - the DS3 viewer's "ROI Tools" + "ROI Manager", moved into the library tool and made
multi-image and database-aware.

    ROI tools     enable selection, Rectangle (drag) / Polygon (click points, click the first point to close),
                  class (editable), add the pending selection to the ROI list, outlines on/off, anomaly map
    ROI Manager   one table for the ROIs of every loaded image (or just the current one): image, ROI, class,
                  shape, pixels, saved-to; delete / clear / set class / rename; plot ROI spectra; plot class means
                  pooled over every image; save / load ROIs as JSON (the viewer's DS3-ROIs format)
    Library       save ROIs class by class to the library database (rois table + spectra.roi_id / roi_class +
                  spectrum_rois) as pooled class means, one mean per ROI, or pixel spectra; load ROIs back from
                  the database onto an image (same image, or any georeferenced image of the same area)

The ROI geometry lives in Scene.rois (hvva.core.rois), so the Aggregation, Mosaic and export tabs and the .npz
scene files use the same ROIs.  No database code here - everything goes through the Library tab's client.
"""
from __future__ import annotations

import json
import os

import numpy as np
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.widgets import PolygonSelector, RectangleSelector
from PyQt5 import QtCore, QtGui, QtWidgets

from ..core import rois as R
from ..library import repository as repo
from .library_panel import CATEGORY_HINT, SaveDialog, _acq_combo, _category_combo, _hint

W = QtWidgets
Qt = QtCore.Qt
PENDING_COLOR = "#ffd400"
COLS = ["Image", "ROI", "Class", "Shape", "Pixels", "In library"]


def class_colors(n):
    import matplotlib
    cmap = matplotlib.colormaps["tab10" if n <= 10 else "tab20"]
    return [tuple(cmap(i % cmap.N)[:3]) for i in range(max(n, 1))]


def _group(title, *items):
    g = W.QGroupBox(title)
    g.setSizePolicy(W.QSizePolicy.Ignored, W.QSizePolicy.Preferred)
    lay = W.QVBoxLayout(g)
    for w in items:
        if isinstance(w, W.QWidget):
            lay.addWidget(w)
        else:
            lay.addLayout(w)
    return g


def _row(*widgets):
    r = W.QHBoxLayout()
    for w in widgets:
        r.addWidget(w)
    return r


def _help(text):
    """Collapsible note (same look as the other tabs' 'About this step')."""
    box = W.QToolButton()
    box.setText("ⓘ  How to draw ROIs")
    box.setCheckable(True)
    box.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
    box.setStyleSheet("QToolButton { border: none; text-align: left; }")
    lab = W.QLabel(text)
    lab.setWordWrap(True)
    lab.setVisible(False)
    box.toggled.connect(lab.setVisible)
    w = W.QWidget()
    lay = W.QVBoxLayout(w)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.addWidget(box)
    lay.addWidget(lab)
    return w


def _btn(text, fn, tip=None):
    b = W.QPushButton(text)
    b.clicked.connect(fn)
    if tip:
        b.setToolTip(tip)
    return b


# ============================================================ dialogs
class RoiSaveDialog(W.QDialog):
    """Target library / dataset / mode / category / acquisition for 'Save ROIs to library'."""

    def __init__(self, parent, libraries, current_id, counts, paths=(), category=None):
        super().__init__(parent)
        self.setWindowTitle("Save ROIs to the library database")
        self.counts = counts
        self.rb_scope = {}
        self.grp_scope = W.QButtonGroup(self)            # separate groups: all radios share this dialog as parent
        scope_box = W.QVBoxLayout()
        for key, text in (("selected", "Selected ROIs"), ("image", "All ROIs of the current image"),
                          ("all", "All ROIs of all images")):
            n = counts.get(key, (0, 0, 0))
            rb = W.QRadioButton(f"{text}  -  {n[0]} ROI(s), {n[1]} class(es), {n[2]} image(s)")
            rb.setEnabled(n[0] > 0)
            self.rb_scope[key] = rb
            self.grp_scope.addButton(rb)
            scope_box.addWidget(rb)
        first = next((k for k in ("selected", "image", "all") if counts.get(k, (0,))[0] > 0), "all")
        self.rb_scope[first].setChecked(True)

        self.cb_lib = W.QComboBox()
        for l in libraries:
            self.cb_lib.addItem(l["name"], l["id"])
        self.cb_lib.addItem("New library...", SaveDialog.NEW)
        i = self.cb_lib.findData(current_id) if current_id else -1
        self.cb_lib.setCurrentIndex(i if i >= 0 else 0)
        self.ed_newlib = W.QLineEdit()
        self.ed_newlib.setPlaceholderText("name of the new library")
        self.ed_ds = W.QLineEdit("ROI analysis")
        self.rb_mode = {}
        self.grp_mode = W.QButtonGroup(self)
        mode_box = W.QVBoxLayout()
        for key, text in R.LIBRARY_MODES.items():
            rb = W.QRadioButton(text)
            self.rb_mode[key] = rb
            self.grp_mode.addButton(rb)
            mode_box.addWidget(rb)
        self.rb_mode["class"].setChecked(True)
        self.sp_px = W.QSpinBox()
        self.sp_px.setRange(1, 100000)
        self.sp_px.setValue(200)
        self.sp_px.setSuffix(" pixels per ROI")
        self.cb_cat = _category_combo(list(paths), category)
        self.chk_cls = W.QCheckBox("Add the class name as the last category level (e.g. Field ROIs/vegetation)")
        self.chk_cls.setChecked(True)
        self.cb_acq = _acq_combo()
        self.cb_acq.setCurrentIndex(max(0, self.cb_acq.findData("airborne")))

        form = W.QFormLayout(self)
        form.addRow(W.QLabel("<b>Which ROIs</b>"))
        form.addRow(scope_box)
        form.addRow("Library", self.cb_lib)
        form.addRow("New library name", self.ed_newlib)
        form.addRow("Dataset", self.ed_ds)
        form.addRow(W.QLabel("<b>What to store per class</b>"))
        form.addRow(mode_box)
        form.addRow("Pixel cap", self.sp_px)
        form.addRow("Category", self.cb_cat)
        form.addRow("", self.chk_cls)
        form.addRow("", _hint(CATEGORY_HINT))
        form.addRow("Acquisition", self.cb_acq)
        form.addRow(_hint("Each ROI is stored in the rois table; each class becomes a material whose spectra "
                          "record the ROI(s) they came from. Saving the same ROIs again adds nothing."))
        bb = W.QDialogButtonBox(W.QDialogButtonBox.Save | W.QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        form.addRow(bb)
        self.cb_lib.currentIndexChanged.connect(self._sync)
        for rb in self.rb_mode.values():
            rb.toggled.connect(self._sync)
        self._sync()
        self.setMinimumWidth(620)
        self.adjustSize()

    def _sync(self, *_):
        self.ed_newlib.setEnabled(self.cb_lib.currentData() == SaveDialog.NEW)
        self.sp_px.setEnabled(self.rb_mode["pixels"].isChecked())

    def choices(self):
        a = self.cb_acq.currentData()
        return {"scope": next(k for k, rb in self.rb_scope.items() if rb.isChecked()),
                "mode": next(k for k, rb in self.rb_mode.items() if rb.isChecked()),
                "max_pixels": self.sp_px.value(),
                "library_id": self.cb_lib.currentData(), "new_library": self.ed_newlib.text().strip(),
                "dataset": self.ed_ds.text().strip() or "ROI analysis",
                "category": repo.normalize_category(self.cb_cat.currentText()),
                "class_level": self.chk_cls.isChecked(),
                "acquisition": None if a == repo.UNTAGGED else a}


class RoiLoadDialog(W.QDialog):
    """ROIs stored in the database: filter by library, see whether each fits the current image, add / delete."""

    def __init__(self, panel, records, scene):
        super().__init__(panel)
        self.panel, self.scene, self.records = panel, scene, []
        self.setWindowTitle("ROIs in the library database")
        self.cb_lib = W.QComboBox()
        self.chk_fit = W.QCheckBox("Only ROIs that fit the current image")
        self.chk_fit.setChecked(True)
        self.tbl = W.QTableWidget(0, 8)
        self.tbl.setHorizontalHeaderLabels(["", "Library", "Class", "ROI", "Image", "Shape", "Pixels / spectra",
                                            "Fits current image"])
        self.tbl.horizontalHeader().setStretchLastSection(True)
        self.tbl.setSelectionBehavior(W.QAbstractItemView.SelectRows)
        self.tbl.setEditTriggers(W.QAbstractItemView.NoEditTriggers)
        self.lbl = _hint("")
        b_add = _btn("Add checked ROIs to the current image", self._add)
        b_all = _btn("Check all", lambda: self._check(True))
        b_none = _btn("Check none", lambda: self._check(False))
        b_del = _btn("Delete selected from database...", self._delete)
        b_close = _btn("Close", self.reject)
        lay = W.QVBoxLayout(self)
        lay.addLayout(_row(W.QLabel("Library"), self.cb_lib, self.chk_fit))
        lay.addWidget(self.tbl, 1)
        lay.addWidget(self.lbl)
        lay.addLayout(_row(b_all, b_none, b_del))
        lay.addLayout(_row(b_add, b_close))
        self.cb_lib.currentIndexChanged.connect(self._fill)
        self.chk_fit.toggled.connect(self._fill)
        self.set_records(records)
        self.resize(980, 520)

    def set_records(self, records):
        keep = self.cb_lib.currentData()
        self.cb_lib.blockSignals(True)
        self.cb_lib.clear()
        self.cb_lib.addItem("All libraries", None)
        libs = {}
        for r in records:
            libs.setdefault(r["library_id"], r.get("library_name") or r["library_id"])
        for lid, name in sorted(libs.items(), key=lambda kv: kv[1].lower()):
            self.cb_lib.addItem(name, lid)
        i = self.cb_lib.findData(keep)
        self.cb_lib.setCurrentIndex(i if i >= 0 else 0)
        self.cb_lib.blockSignals(False)
        self.records = []
        for r in records:
            roi, how = None, "no"
            if self.scene is not None:
                try:
                    roi = R.roi_from_record(r, self.scene)
                except (ValueError, KeyError, TypeError):
                    roi = None
                if roi is not None:
                    how = "yes - same image" if roi["id"] == r["roi_uid"] else "yes - via map coordinates"
            self.records.append((r, roi, how))
        self._fill()

    def _fill(self, *_):
        lid = self.cb_lib.currentData()
        rows = [x for x in self.records if (lid is None or x[0]["library_id"] == lid)
                and (not self.chk_fit.isChecked() or x[1] is not None)]
        self.rows = rows
        self.tbl.setRowCount(len(rows))
        for i, (r, roi, how) in enumerate(rows):
            chk = W.QTableWidgetItem()
            chk.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            chk.setCheckState(Qt.Checked if roi is not None and how.endswith("same image") else Qt.Unchecked)
            if roi is None:
                chk.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            self.tbl.setItem(i, 0, chk)
            vals = [r.get("library_name") or "", r["class_name"], r.get("name") or "", r.get("scene_name") or "",
                    r["geometry_type"], f"{r.get('n_pixels') or 0} / {r.get('n_spectra', 0)}", how]
            for c, v in enumerate(vals, 1):
                self.tbl.setItem(i, c, W.QTableWidgetItem(str(v)))
        self.tbl.resizeColumnsToContents()
        n_fit = sum(1 for _r, roi, _h in rows if roi is not None)
        where = self.scene.name if self.scene is not None else "no image loaded"
        self.lbl.setText(f"{len(rows)} ROI(s) listed, {n_fit} can be placed on the current image ({where}). "
                         "Same image = same name and size; otherwise both images must be georeferenced in the "
                         "same CRS.")

    def _check(self, on):
        for i in range(self.tbl.rowCount()):
            it = self.tbl.item(i, 0)
            if it.flags() & Qt.ItemIsUserCheckable:
                it.setCheckState(Qt.Checked if on else Qt.Unchecked)

    def _add(self):
        picked = [self.rows[i][1] for i in range(self.tbl.rowCount())
                  if self.tbl.item(i, 0).checkState() == Qt.Checked and self.rows[i][1] is not None]
        if not picked:
            return self.panel.host._err("check the ROIs to add (only ROIs that fit the current image can be added)")
        self.panel.add_rois_to_scene(self.scene, picked, [self.rows[i][0] for i in range(self.tbl.rowCount())
                                                          if self.tbl.item(i, 0).checkState() == Qt.Checked])
        self.accept()

    def _delete(self):
        sel = sorted({i.row() for i in self.tbl.selectedIndexes()})
        if not sel:
            return self.panel.host._err("select the ROI rows to delete")
        recs = [self.rows[i][0] for i in sel]
        mb = W.QMessageBox(self)
        mb.setWindowTitle("Delete ROIs from the database")
        mb.setText(f"Delete {len(recs)} ROI(s) from the library database?")
        mb.setInformativeText("Keep spectra: spectra computed from them stay in the library (their ROI link is "
                              "removed).\nDelete spectra: spectra computed only from these ROIs are deleted too "
                              "(a class mean that also used other ROIs is kept).")
        b_keep = mb.addButton("Delete ROIs, keep spectra", W.QMessageBox.AcceptRole)
        b_all = mb.addButton("Delete ROIs and their spectra", W.QMessageBox.DestructiveRole)
        mb.addButton(W.QMessageBox.Cancel)
        mb.exec_()
        if mb.clickedButton() not in (b_keep, b_all):
            return
        self.panel.delete_db_rois(recs, mb.clickedButton() is b_all, self)


# ============================================================ the panel
class RoiPanel(W.QWidget):
    """ROI Analysis tab. *host* = main window (scenes, current_scene(), image_canvas, plot_canvas, libpanel...)."""

    def __init__(self, host):
        super().__init__()
        self.host = host
        self.pending = None                 # selection drawn but not yet added: {"kind", "rect"|"vertices"}
        self.show_anomaly = False
        self._anomaly = {}                  # id(scene) -> map
        self._selector = None
        self._artists = []
        self.rows = []                      # [(scene, roi)] shown in the table
        self._build()

    # ------------------------------------------------------------ UI
    def _build(self):
        self.chk_enable = W.QCheckBox("Enable ROI selection on the scene view")
        self.chk_enable.toggled.connect(self._on_enable)
        self.cb_shape = W.QComboBox()
        self.cb_shape.addItems(["Rectangle (drag)", "Polygon (click points)"])
        self.cb_shape.currentIndexChanged.connect(lambda *_: self.attach_selector())
        self.lbl_sel = W.QLabel("No selection.")
        self.lbl_sel.setWordWrap(True)
        self.cb_class = W.QComboBox()
        self.cb_class.setEditable(True)
        self.cb_class.setInsertPolicy(W.QComboBox.NoInsert)
        self.cb_class.setEditText("Class 1")
        self.cb_class.setToolTip("Type a new class name or pick one already used in any image. Classes are "
                                 "matched across images by name.")
        self.chk_auto = W.QCheckBox("Add to the ROI list as soon as it is drawn")
        self.btn_add = _btn("Add selection to ROI list", self.on_add_selection)
        self.chk_outlines = W.QCheckBox("Show ROI outlines")
        self.chk_outlines.setChecked(True)
        self.chk_outlines.toggled.connect(lambda *_: self.draw_overlays())
        self.btn_anomaly = _btn("Show anomaly map", self.on_anomaly,
                                "Distance of every pixel from the image's mean spectrum (bright = unusual). "
                                "Draw ROIs on the anomaly map to capture rare materials.")
        self.btn_anomaly.setCheckable(True)
        form = W.QFormLayout()
        form.setFieldGrowthPolicy(W.QFormLayout.AllNonFixedFieldsGrow)
        form.addRow(self.chk_enable)
        form.addRow("Shape", self.cb_shape)
        form.addRow("Class", self.cb_class)
        form.addRow(self.lbl_sel)
        form.addRow(self.btn_add)
        form.addRow(self.chk_auto)
        form.addRow(self.chk_outlines)
        form.addRow(self.btn_anomaly)
        tools = _group("ROI tools", form, _help(
            "Rectangle: drag on the scene view.  Polygon: click the corners, click the first corner again to "
            "close (Esc restarts).  The selection is shown dashed yellow until it is added.  ROIs belong to the "
            "image they are drawn on; switch images in the Scenes tab and keep drawing - one class can have ROIs "
            "in many images."))

        self.lbl_summary = W.QLabel("No ROIs.")
        self.cb_view = W.QComboBox()
        self.cb_view.addItems(["All images", "Current image"])
        self.cb_view.currentIndexChanged.connect(lambda *_: self.refresh())
        self.tbl = W.QTableWidget(0, len(COLS))
        self.tbl.setHorizontalHeaderLabels(COLS)
        self.tbl.horizontalHeader().setStretchLastSection(True)
        self.tbl.setSelectionBehavior(W.QAbstractItemView.SelectRows)
        self.tbl.setSelectionMode(W.QAbstractItemView.ExtendedSelection)
        self.tbl.setEditTriggers(W.QAbstractItemView.NoEditTriggers)
        self.tbl.itemSelectionChanged.connect(self.draw_overlays)
        self.tbl.itemDoubleClicked.connect(self.on_row_double_clicked)
        manage = _row(_btn("Delete selected", self.on_delete), _btn("Clear all...", self.on_clear),
                      _btn("Set class...", self.on_set_class), _btn("Rename...", self.on_rename))
        plots = _row(_btn("Plot ROI spectra", self.on_plot_rois, "Mean ± std of each selected ROI "
                          "(all ROIs of the current image if none is selected)"),
                     _btn("Plot class means", self.on_plot_classes, "Pixels of every ROI of a class pooled over "
                          "all images (selected ROIs only if any are selected)"),
                     _btn("Save ROIs...", self.on_save_json, "DS3-ROIs JSON - opens in the viewer too"),
                     _btn("Load ROIs...", self.on_load_json))
        db = _row(_btn("Save ROIs to library database...", self.on_save_db),
                  _btn("Load ROIs from library database...", self.on_load_db))
        manager = _group("ROI Manager", _row(self.lbl_summary, W.QLabel("Show"), self.cb_view), self.tbl,
                         manage, plots, db)

        lay = W.QHBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.addWidget(tools, 2)
        lay.addWidget(manager, 5)

    # ------------------------------------------------------------ helpers
    def selecting(self):
        return self.chk_enable.isChecked()

    def all_items(self):
        return [(sc, r) for sc in self.host.scenes for r in sc.rois]

    def class_names(self):
        return R.class_order(r for _sc, r in self.all_items())

    def color_of(self, cname):
        names = self.class_names()
        cols = class_colors(len(names))
        return cols[names.index(cname) % len(cols)] if cname in names else (1, 1, 1)

    def selected_items(self):
        rows = sorted({i.row() for i in self.tbl.selectedIndexes()})
        return [self.rows[r] for r in rows if r < len(self.rows)]

    def _class_text(self):
        return self.cb_class.currentText().strip() or "Class 1"

    # ------------------------------------------------------------ table
    def refresh(self):
        """Rebuild the table + class list from the scenes (call after any ROI change)."""
        keep = {r["id"] for _s, r in self.selected_items()}
        sc = self.host.current_scene()
        items = self.all_items() if self.cb_view.currentIndex() == 0 else [(sc, r) for r in (sc.rois if sc else [])]
        self.rows = items
        self.tbl.blockSignals(True)
        self.tbl.setRowCount(len(items))
        for i, (s, r) in enumerate(items):
            h, w, _ = s.cube.shape
            n = int(R.roi_mask(r, h, w).sum())
            shape = "Rectangle" if r["kind"] == "rect" else f"Polygon ({len(r['vertices'])} pts)"
            vals = [s.name, r["name"], r["class"], shape, n, ", ".join(r.get("saved_to") or [])]
            for c, v in enumerate(vals):
                it = W.QTableWidgetItem(str(v))
                if c == 2:
                    col = self.color_of(r["class"])
                    it.setForeground(QtGui.QBrush(QtGui.QColor.fromRgbF(*col)))
                if c == 3:
                    it.setToolTip(R.describe_geometry(r))
                self.tbl.setItem(i, c, it)
            if r["id"] in keep:
                self.tbl.selectRow(i)
        self.tbl.resizeColumnsToContents()
        self.tbl.blockSignals(False)
        all_items = self.all_items()
        n_cls = len(self.class_names())
        n_img = len({id(s) for s, _ in all_items})
        self.lbl_summary.setText(f"{len(all_items)} ROI(s) in {n_cls} class(es) across {n_img} image(s)")
        cur = self.cb_class.currentText()
        self.cb_class.blockSignals(True)
        self.cb_class.clear()
        self.cb_class.addItems(self.class_names())
        self.cb_class.setEditText(cur)
        self.cb_class.blockSignals(False)
        self.draw_overlays()

    def on_row_double_clicked(self, item):
        s, _r = self.rows[item.row()]
        if s is not self.host.current_scene():
            self.host.select_scene(s)

    # ------------------------------------------------------------ selection on the canvas
    def _on_enable(self, on):
        if on and self.host.current_scene() is None:
            self.host.say("Load a scene first (Scenes tab).")
        self.attach_selector()
        self.host.say("ROI selection on: drag a rectangle or click polygon corners on the scene view."
                      if on else "ROI selection off: clicking the scene view shows pixel spectra again.")

    def on_canvas_cleared(self):
        """ImageCanvas.after_clear hook: the figure was rebuilt -> new selector, redraw outlines."""
        self._artists = []
        self.attach_selector(draw=False)
        self.draw_overlays(draw=False)

    def attach_selector(self, draw=True):
        old, self._selector = self._selector, None
        if old is not None:
            try:
                old.set_active(False)
                old.disconnect_events()
            except Exception:                                                   # noqa: BLE001
                pass
        ax = self.host.image_canvas.ax
        if not self.selecting() or self.host.current_scene() is None:
            if draw:
                self.host.image_canvas.draw_idle()
            return
        if self.cb_shape.currentIndex() == 0:
            self._selector = RectangleSelector(ax, self._on_rect, useblit=False, button=[1], interactive=False,
                                               minspanx=1, minspany=1, spancoords="data",
                                               props=dict(edgecolor=PENDING_COLOR, fill=False, linestyle="--"))
        else:
            self._selector = PolygonSelector(ax, self._on_polygon, useblit=False,
                                             props=dict(color=PENDING_COLOR, linestyle="--", linewidth=1.5),
                                             handle_props=dict(markersize=4, markerfacecolor=PENDING_COLOR))
        if draw:
            self.host.image_canvas.draw_idle()

    def _on_rect(self, press, release):
        sc = self.host.current_scene()
        if sc is None or press.xdata is None or release.xdata is None:
            return
        h, w, _ = sc.cube.shape
        c0, c1 = sorted((int(round(press.xdata)), int(round(release.xdata))))
        r0, r1 = sorted((int(round(press.ydata)), int(round(release.ydata))))
        c0, r0 = max(c0, 0), max(r0, 0)
        c1, r1 = min(c1, w - 1), min(r1, h - 1)
        if c1 < c0 or r1 < r0:
            return
        self._set_pending({"kind": "rect", "rect": (r0, c0, r1 + 1, c1 + 1)})

    def _on_polygon(self, verts):
        if len(verts) < 3:
            return
        self._set_pending({"kind": "polygon", "vertices": [(float(x), float(y)) for x, y in verts]})

    def _set_pending(self, geom):
        sc = self.host.current_scene()
        h, w, _ = sc.cube.shape
        n = int(R.roi_mask(R.normalize_roi(dict(geom, **{"class": "x"})), h, w).sum())
        if n == 0:
            self.lbl_sel.setText("The selection contains no pixel centre - draw it a little larger.")
            return
        self.pending = geom
        what = R.describe_geometry(dict(geom))
        self.lbl_sel.setText(f"Selection: {what}, {n} pixels - press 'Add selection to ROI list'.")
        if self.chk_auto.isChecked():
            self.on_add_selection()
        else:
            self.draw_overlays()

    def on_add_selection(self):
        sc = self.host.current_scene()
        if sc is None or self.pending is None:
            return self.host._err("draw a rectangle or polygon on the scene view first (enable ROI selection)")
        roi = sc.add_roi(dict(self.pending, **{"class": self._class_text()}))
        self.pending = None
        self.lbl_sel.setText("No selection.")
        h, w, _ = sc.cube.shape
        self.host.say(f"{sc.name}: added {roi['name']} ({roi['class']}, {int(R.roi_mask(roi, h, w).sum())} px).")
        self.host.refresh_scenes()
        self.refresh()
        if self.cb_shape.currentIndex() == 1:            # PolygonSelector keeps the finished polygon: restart it
            self.attach_selector()

    # ------------------------------------------------------------ drawing
    def draw_overlays(self, draw=True):
        canvas = self.host.image_canvas
        for a in self._artists:
            try:
                a.remove()
            except (ValueError, NotImplementedError, AttributeError):
                pass
        self._artists = []
        sc = self.host.current_scene()
        if sc is None or not hasattr(canvas, "ax") or not canvas.ax.images:
            return
        ax = canvas.ax
        sel = {r["id"] for s, r in self.selected_items() if s is sc}
        if self.chk_outlines.isChecked():
            for r in sc.rois:
                col = self.color_of(r["class"])
                lw = 3.0 if r["id"] in sel else 1.6
                p = MplPolygon(R.outline(r), closed=True, fill=False, ec=col, lw=lw)
                ax.add_patch(p)
                x0, y0 = R.outline(r).min(axis=0)
                t = ax.text(x0 + .5, y0 + .5, r["name"].replace("ROI ", ""), color=col, fontsize=7,
                            va="top", ha="left", fontweight="bold" if r["id"] in sel else "normal")
                self._artists += [p, t]
        if self.pending is not None:
            p = MplPolygon(R.outline(self.pending), closed=True, fill=False, ec=PENDING_COLOR, lw=1.5, ls="--")
            ax.add_patch(p)
            self._artists.append(p)
        if draw:
            canvas.draw_idle()

    def on_anomaly(self):
        sc = self.host.current_scene()
        if sc is None:
            self.btn_anomaly.setChecked(False)
            return self.host._err("load a scene first")
        self.show_anomaly = self.btn_anomaly.isChecked()
        self.btn_anomaly.setText("Back to RGB view" if self.show_anomaly else "Show anomaly map")
        self.host.show_scene()

    def anomaly_image(self, sc):
        """2-98 % stretched anomaly map of *sc* (cached)."""
        key = (id(sc), sc.cube.shape)
        if key not in self._anomaly:
            a = R.anomaly_map(sc.cube)
            lo, hi = np.nanpercentile(a, (2, 98))
            self._anomaly[key] = np.nan_to_num(np.clip((a - lo) / (hi - lo + 1e-12), 0, 1))
        return self._anomaly[key]

    # ------------------------------------------------------------ manage
    def _need_selection(self):
        items = self.selected_items()
        if not items:
            self.host._err("select one or more ROIs in the ROI Manager")
        return items

    def on_delete(self):
        items = self._need_selection()
        if not items:
            return
        for s, r in items:
            s.rois = [x for x in s.rois if x["id"] != r["id"]]
        self.tbl.clearSelection()
        self.host.say(f"Deleted {len(items)} ROI(s) (the library database is not changed).")
        self._changed()

    def on_clear(self):
        sc = self.host.current_scene()
        if not self.all_items():
            return
        mb = W.QMessageBox(self)
        mb.setWindowTitle("Clear ROIs")
        mb.setText("Remove which ROIs from the ROI list? (The library database is not changed.)")
        b_img = mb.addButton("This image", W.QMessageBox.AcceptRole) if sc is not None and sc.rois else None
        b_all = mb.addButton("All images", W.QMessageBox.DestructiveRole)
        mb.addButton(W.QMessageBox.Cancel)
        mb.exec_()
        if b_img is not None and mb.clickedButton() is b_img:
            sc.rois = []
        elif mb.clickedButton() is b_all:
            for s in self.host.scenes:
                s.rois = []
        else:
            return
        self.pending = None
        self._changed()

    def on_set_class(self):
        items = self._need_selection()
        if not items:
            return
        names = self.class_names()
        cur = items[0][1]["class"]
        name, ok = W.QInputDialog.getItem(self, "Set class", f"Class for {len(items)} ROI(s):", names,
                                          names.index(cur) if cur in names else 0, True)
        name = (name or "").strip()
        if ok and name:
            for _s, r in items:
                if r["class"] != name:
                    r["class"] = name
                    r.pop("saved_to", None)          # the stored row now differs - save again to update it
            self._changed()

    def on_rename(self):
        items = self._need_selection()
        if not items:
            return
        s, r = items[0]
        name, ok = W.QInputDialog.getText(self, "Rename ROI", f"New name for {r['name']} ({s.name}):", text=r["name"])
        name = (name or "").strip()
        if ok and name:
            r["name"] = name
            r.pop("saved_to", None)
            self._changed()

    def _changed(self):
        self.host.refresh_scenes()                       # also refreshes this table

    # ------------------------------------------------------------ plots
    def on_plot_rois(self):
        items = self.selected_items()
        if not items:
            sc = self.host.current_scene()
            items = [(sc, r) for r in (sc.rois if sc else [])]
        if not items:
            return self.host._err("no ROIs to plot")
        ax = self.host.plot_canvas.reset()
        for s, r in items:
            st = R.roi_stats(s.cube, r)
            x = s.wavelengths if s.has_wavelengths else np.arange(s.n_bands)
            col = self.color_of(r["class"])
            ax.plot(x, st["mean"], color=col, lw=1.3, label=f"{s.name} · {r['name']} ({r['class']}, {st['n_pixels']} px)")
            if st["n_pixels"] > 1:
                ax.fill_between(x, st["mean"] - st["std"], st["mean"] + st["std"], color=col, alpha=0.12)
        any_wl = all(s.has_wavelengths for s, _ in items)
        self.host.plot_canvas.finish("ROI spectra (mean ± std)", "Wavelength (nm)" if any_wl else "Band", "Value",
                                     len(items) <= 12)

    def on_plot_classes(self):
        items = self.selected_items() or self.all_items()
        if not items:
            return self.host._err("no ROIs - draw some first")
        try:
            pooled = R.pool_classes(items)
        except ValueError as e:
            return self.host._err(e)
        ax = self.host.plot_canvas.reset()
        for cname, p in pooled.items():
            col = self.color_of(cname)
            ax.plot(p["wavelengths"], p["mean"], color=col, lw=1.6,
                    label=f"{cname} ({p['n_pixels']} px, {p['n_rois']} ROI, {p['n_scenes']} img)")
            if p["n_pixels"] > 1:
                ax.fill_between(p["wavelengths"], p["mean"] - p["std"], p["mean"] + p["std"], color=col, alpha=0.15)
        wl_ok = all(s.has_wavelengths for s, _ in items)
        self.host.plot_canvas.finish("Class means (pooled over images, ± std)",
                                     "Wavelength (nm)" if wl_ok else "Band", "Value", True)

    # ------------------------------------------------------------ JSON files
    def on_save_json(self):
        items = self.selected_items() or self.all_items()
        if not items:
            return self.host._err("no ROIs to save")
        p, _ = W.QFileDialog.getSaveFileName(self, "Save ROIs", "rois.json", "ROI files (*.json)")
        if not p:
            return
        try:
            with open(p, "w", encoding="utf-8") as f:
                json.dump(R.to_ds3_json(items), f, indent=1)
        except OSError as e:
            return self.host._err(e)
        self.host.say(f"Saved {len(items)} ROI(s) to {p} (DS3-ROIs v2 - the viewer reads it too).")

    def on_load_json(self):
        sc = self.host.current_scene()
        if sc is None:
            return self.host._err("load a scene first")
        p, _ = W.QFileDialog.getOpenFileName(self, "Load ROIs", "", "ROI files (*.json);;All files (*)")
        if not p:
            return
        try:
            with open(p, encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict) or "rois" not in data:
                raise ValueError("not a DS3-ROIs file")
        except (OSError, ValueError) as e:
            return self.host._err(f"{os.path.basename(p)}: {e}")
        by_name = {s.name: s for s in self.host.scenes}
        groups = {}
        for it in data["rois"]:
            target = by_name.get(it.get("scene")) or sc
            groups.setdefault(id(target), (target, []))[1].append(it)
        shape = data.get("image_shape")
        if shape and tuple(shape) != tuple(sc.cube.shape[:2]) and id(sc) in groups:
            ans = W.QMessageBox.question(self, "Image size differs",
                                         f"The ROIs were drawn on a {shape[0]} x {shape[1]} image; {sc.name} is "
                                         f"{sc.cube.shape[0]} x {sc.cube.shape[1]}. Load them anyway (clipped)?")
            if ans != W.QMessageBox.Yes:
                return
        if any(t.rois for t, _ in groups.values()):
            mb = W.QMessageBox(self)
            mb.setWindowTitle("Load ROIs")
            mb.setText("Replace the existing ROIs of the image(s) or append to them?")
            b_rep = mb.addButton("Replace", W.QMessageBox.DestructiveRole)
            b_app = mb.addButton("Append", W.QMessageBox.AcceptRole)
            mb.addButton(W.QMessageBox.Cancel)
            mb.exec_()
            if mb.clickedButton() not in (b_rep, b_app):
                return
            replace = mb.clickedButton() is b_rep
        else:
            replace = False
        n, skipped = 0, 0
        for target, its in groups.values():
            h, w, _ = target.cube.shape
            loaded, sk = R.from_ds3_json({"rois": its}, h, w)
            skipped += sk
            if replace:
                target.rois = []
            n += self._merge(target, [roi for _name, roi in loaded])
        self._changed()
        self.host.say(f"Loaded {n} ROI(s) from {os.path.basename(p)}" + (f", {skipped} outside the image skipped."
                                                                          if skipped else "."))

    @staticmethod
    def _merge(scene, rois):
        """Add ROIs to a scene; a ROI with an id the scene already has replaces it. Returns the number added."""
        for roi in rois:
            old = scene.roi_by_id(roi["id"])
            if old is not None:
                old.clear()
                old.update(roi)
            else:
                scene.add_roi(roi)
        return len(rois)

    # ------------------------------------------------------------ library database
    def _scope_items(self, scope):
        sc = self.host.current_scene()
        if scope == "selected":
            return self.selected_items()
        if scope == "image":
            return [(sc, r) for r in (sc.rois if sc else [])]
        return self.all_items()

    @staticmethod
    def _count(items):
        return (len(items), len({r["class"] for _s, r in items}), len({id(s) for s, _ in items}))

    def on_save_db(self):
        lp = self.host.libpanel
        if not lp._need_client():
            return
        if not self.all_items():
            return self.host._err("no ROIs - draw some on the scene view first")
        counts = {k: self._count(self._scope_items(k)) for k in ("selected", "image", "all")}
        start = lp.cat_path if lp.cat_path not in (None, repo.UNCATEGORISED) else None
        d = RoiSaveDialog(self, lp.libraries, (lp.active_lib or {}).get("id"), counts, lp.category_paths(), start)
        if not d.exec_():
            return
        ch = d.choices()
        if ch["library_id"] == SaveDialog.NEW and not ch["new_library"]:
            return self.host._err("enter a name for the new library")
        items = self._scope_items(ch["scope"])
        try:
            recs, specs = R.library_items(items, ch["mode"], ch["max_pixels"])
        except ValueError as e:
            return self.host._err(e)
        for it in specs:
            if ch["class_level"]:
                it["category"] = repo.normalize_category(f"{ch['category'] or ''}/{it['class']}")
        c = lp.client
        self.host.say(f"Saving {len(recs)} ROI(s) and {len(specs)} spectra to the library database...")

        def go():
            lid = ch["library_id"]
            if lid == SaveDialog.NEW:
                lid = c.create_library(ch["new_library"], "ROIs and their spectra saved from HyperVista")["id"]
            res = c.save_roi_spectra(lid, ch["dataset"], recs, specs, ch["category"], ch["acquisition"])
            name = next((l["name"] for l in c.list_libraries() if l["id"] == lid), ch["new_library"] or "")
            return lid, name, res

        def ok(out):
            lid, name, res = out
            for _s, r in items:
                saved = r.setdefault("saved_to", [])
                if name not in saved:
                    saved.append(name)
            self.refresh()
            what = {"class": "class mean(s)", "roi": "ROI mean(s)", "pixels": "pixel spectra"}[ch["mode"]]
            extra = f"; {res['skipped']} already stored, skipped" if res.get("skipped") else ""
            self.host.say(f"Library '{name}': {res['rois']} ROI(s) and {res['spectra']} {what} saved for "
                          f"class(es) {', '.join(res['classes'])} (dataset {res['dataset']['name']}{extra}).")
            if lp.active_lib and lp.active_lib["id"] == lid:
                lp.load_facets()
                lp.fetch_materials(False)
            lp.load_libraries(select_id=lid)
        lp.run(go, done=ok)

    def on_load_db(self):
        lp = self.host.libpanel
        if not lp._need_client():
            return
        c = lp.client

        def ok(recs):
            if not recs:
                return self.host.say("The library database has no ROIs yet - save some with "
                                     "'Save ROIs to library database...'.")
            self._load_dialog = RoiLoadDialog(self, recs, self.host.current_scene())
            self._load_dialog.exec_()
            self._load_dialog = None
        lp.run(c.list_rois, done=ok)

    def add_rois_to_scene(self, scene, rois, records):
        libs = {}
        for rec in records:
            libs[rec["roi_uid"]] = rec.get("library_name")
        for roi in rois:
            name = libs.get(roi["id"])
            if name:
                roi["saved_to"] = [name]
        n = self._merge(scene, rois)
        self._changed()
        self.host.say(f"{scene.name}: {n} ROI(s) loaded from the library database.")

    def delete_db_rois(self, records, delete_spectra, dialog):
        lp = self.host.libpanel
        c = lp.client

        def go():
            n = 0
            for r in records:
                n += c.delete_roi(r["id"], delete_spectra)["deleted_spectra"]
            return n, c.list_rois()

        def ok(out):
            n, recs = out
            gone = {(r["library_name"], r["roi_uid"]) for r in records}
            for s in self.host.scenes:
                for roi in s.rois:
                    if roi.get("saved_to"):
                        roi["saved_to"] = [l for l in roi["saved_to"] if (l, roi["id"]) not in gone]
            self.refresh()
            self.host.say(f"Deleted {len(records)} ROI(s) from the database" +
                          (f" and {n} spectra." if delete_spectra else " (spectra kept)."))
            if lp.active_lib:
                lp.load_facets()
                lp.fetch_materials(False)
            lp.load_libraries()
            if getattr(self, "_load_dialog", None) is dialog and dialog is not None:
                dialog.set_records(recs)
        lp.run(go, done=ok)

