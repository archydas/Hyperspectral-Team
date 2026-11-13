"""Native PyQt5 'Image' page: open hyperspectral images through the backend, show a colour composite, click pixels for
spectra, draw ROIs (object samples with their image patch), georeference, map view, Spectral menu (analyst, SAM,
unmixing, k-means, PCA, band math / indices, RX anomaly, SNR) and ROI aggregation. Replaces the web Image page."""
from __future__ import annotations

import json
import math

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

from ..widgets import label
from .api import pixmap_from_data_url, run_bg
from .common import (DS_COLORS, LIB_GREY, OBJ_COLORS, SaveDialog, SpectrumChart, angle_deg, btn, card, hrow,
                     save_bytes, save_csv, small, xy)
from .image_dialogs import (INDICES, AggregateDialog, AnalysisDialog, AnalystDialog, BandsDialog, GeorefDialog, MapPanel,
                            SamMapDialog, default_rgb, legend_widget)
from .library_tools import Hub, ImportDialog, cell, get_spectrum, table

W = QtWidgets
Qt = QtCore.Qt
SETTINGS = ("DS3", "native_image")


def dist_seg(p, a, b):
    ax, ay, bx, by = a.x(), a.y(), b.x(), b.y()
    dx, dy = bx - ax, by - ay
    L = dx * dx + dy * dy
    t = 0 if L == 0 else max(0, min(1, ((p.x() - ax) * dx + (p.y() - ay) * dy) / L))
    return math.hypot(p.x() - ax - t * dx, p.y() - ay - t * dy)


def same_poly(a, b):
    return len(a) == len(b) and all(abs(x[0] - y[0]) < 1e-6 and abs(x[1] - y[1]) < 1e-6 for x, y in zip(a, b))


# ------------------------------------------------------------------ the image view
class ImageView(W.QGraphicsView):
    """Scene units = full-resolution pixels (x = col, y = row). Wheel zoom, drag pan, click / double / right click."""
    clicked = QtCore.pyqtSignal(int, int)
    finish = QtCore.pyqtSignal()
    obj_clicked = QtCore.pyqtSignal(int, QtCore.QPoint)
    hover = QtCore.pyqtSignal(int, int)
    zoomed = QtCore.pyqtSignal()

    def __init__(self):
        super().__init__()
        self.sc = W.QGraphicsScene(self)
        self.setScene(self.sc)
        self.setBackgroundBrush(QtGui.QColor("#1b2430"))
        self.setTransformationAnchor(W.QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(W.QGraphicsView.AnchorViewCenter)
        self.setRenderHint(QtGui.QPainter.Antialiasing)
        self.setMouseTracking(True)
        self.setCursor(Qt.CrossCursor)
        self.base = self.sc.addPixmap(QtGui.QPixmap())
        self.over = self.sc.addPixmap(QtGui.QPixmap())
        self.over.setZValue(1)
        self.over.setOpacity(0.85)
        self.ann = []
        self.rows = self.cols = 0
        self.objs_poly = []                     # [(index, [QPointF scene])] for hit tests
        self._p = None
        self._drag = False
        self.drawing = False
        self.placeholder = self.sc.addText("No image yet.\nUse “Open image…” — AVIRIS-NG (.hdr + data file), EnMAP (.zip / "
                                           ".TIF), PaviaU.mat, GeoTIFF, HDF5 / NetCDF.")
        self.placeholder.setDefaultTextColor(QtGui.QColor("#cbd5e1"))

    def set_base(self, pixmap, rows, cols, fit=False):
        self.placeholder.setVisible(False)
        self.rows, self.cols = rows, cols
        self.base.setPixmap(pixmap)
        self.base.setTransform(QtGui.QTransform.fromScale(cols / max(1, pixmap.width()), rows / max(1, pixmap.height())))
        self._mode()
        self.sc.setSceneRect(-cols * 0.05, -rows * 0.05, cols * 1.1, rows * 1.1)
        if fit:
            self.fit()

    def clear(self):
        self.base.setPixmap(QtGui.QPixmap())
        self.over.setPixmap(QtGui.QPixmap())
        self.set_annotations([])
        self.placeholder.setVisible(True)
        self.rows = self.cols = 0

    def set_overlay(self, pixmap):
        if pixmap is None or pixmap.isNull():
            self.over.setPixmap(QtGui.QPixmap())
            return
        self.over.setPixmap(pixmap)
        self.over.setTransform(QtGui.QTransform.fromScale(self.cols / pixmap.width(), self.rows / pixmap.height()))

    def scale_px(self):
        return self.transform().m11()

    def _mode(self):
        m = Qt.FastTransformation if self.scale_px() * self.cols / max(1, self.base.pixmap().width()) > 2 else Qt.SmoothTransformation
        self.base.setTransformationMode(m)

    def fit(self):
        if self.cols:
            self.fitInView(QtCore.QRectF(0, 0, self.cols, self.rows), Qt.KeepAspectRatio)
            self._mode()
            self.zoomed.emit()

    def zoom(self, f):
        self.setTransformationAnchor(W.QGraphicsView.AnchorViewCenter)
        self.scale(f, f)
        self.setTransformationAnchor(W.QGraphicsView.AnchorUnderMouse)
        self._mode()
        self.zoomed.emit()

    def wheelEvent(self, e):
        if not self.cols:
            return
        f = 1.25 if e.angleDelta().y() > 0 else 0.8
        s = self.scale_px() * f
        if s < 0.02 or s > 80:
            return
        self.scale(f, f)
        self._mode()
        self.zoomed.emit()

    def set_annotations(self, items):
        for it in self.ann:
            self.sc.removeItem(it)
        self.ann = []
        for it in items:
            self.sc.addItem(it)
            self.ann.append(it)

    def _hit_obj(self, pos):
        for idx, pts in self.objs_poly:
            sp = [self.mapFromScene(p) for p in pts]
            for k in range(len(sp)):
                if dist_seg(QtCore.QPointF(pos), QtCore.QPointF(sp[k]), QtCore.QPointF(sp[(k + 1) % len(sp)])) < 5:
                    return idx
        return None

    def _cell(self, pos):
        p = self.mapToScene(pos)
        if 0 <= p.x() < self.cols and 0 <= p.y() < self.rows:
            return int(p.y()), int(p.x())
        return None

    def mousePressEvent(self, e):
        if e.button() == Qt.RightButton:
            if self.drawing:
                self.finish.emit()
            return
        if e.button() == Qt.LeftButton:
            self._p = e.pos()
            self._drag = False

    def mouseMoveEvent(self, e):
        c = self._cell(e.pos())
        if c:
            self.hover.emit(*c)
        if self._p is not None and e.buttons() & Qt.LeftButton:
            d = e.pos() - self._p
            if self._drag or abs(d.x()) + abs(d.y()) > 4:
                self._drag = True
                self.setCursor(Qt.ClosedHandCursor)
                self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - d.x())
                self.verticalScrollBar().setValue(self.verticalScrollBar().value() - d.y())
                self._p = e.pos()

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.LeftButton and self._p is not None:
            if not self._drag and self.cols:
                hit = None if self.drawing else self._hit_obj(e.pos())
                if hit is not None:
                    self.obj_clicked.emit(hit, self.mapToGlobal(e.pos()))
                else:
                    c = self._cell(e.pos())
                    if c:
                        self.clicked.emit(*c)
            self._p = None
            self._drag = False
            self.setCursor(Qt.CrossCursor)

    def mouseDoubleClickEvent(self, e):
        if self.drawing:
            self.finish.emit()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self.zoomed.emit()


def marker(x, y, color, text=None, r=7, width=2):
    el = W.QGraphicsEllipseItem(-r, -r, 2 * r, 2 * r)
    el.setPen(QtGui.QPen(QtGui.QColor(color), width))
    el.setFlag(W.QGraphicsItem.ItemIgnoresTransformations)
    el.setPos(x, y)
    el.setZValue(5)
    if text:
        bg = W.QGraphicsRectItem(r + 1, -r - 9, 7 * len(text) + 6, 14, el)
        bg.setBrush(QtGui.QColor(0, 0, 0, 160))
        bg.setPen(QtGui.QPen(Qt.NoPen))
        t = W.QGraphicsSimpleTextItem(text, el)
        t.setBrush(QtGui.QColor(color))
        f = t.font()
        f.setPointSizeF(8)
        f.setBold(True)
        t.setFont(f)
        t.setPos(r + 4, -r - 9)
    return el


def poly_item(pts, color, closed=True, width=2, dash=False):
    """pts = [[row, col], ...] full-resolution."""
    path = QtGui.QPainterPath()
    for k, (r, c) in enumerate(pts):
        (path.moveTo if k == 0 else path.lineTo)(c, r)
    if closed and len(pts) > 2:
        path.closeSubpath()
    it = W.QGraphicsPathItem(path)
    pen = QtGui.QPen(QtGui.QColor(color), width, Qt.DashLine if dash else Qt.SolidLine)
    pen.setCosmetic(True)
    it.setPen(pen)
    if closed:
        col = QtGui.QColor(color)
        col.setAlpha(40)
        it.setBrush(col)
    it.setZValue(4)
    return it


# ------------------------------------------------------------------ the page
class ImageMainPage(W.QWidget):
    def __init__(self, shell, api, store):
        super().__init__()
        self.shell, self.api, self.store = shell, api, store
        self.setObjectName("ds3Page")
        Hub.image_page = self
        self.images, self.active, self.wl, self.bad = [], None, [], []
        self.bands = [1, 1, 1]
        self.preview = None
        self.base_pixmap = None
        self.geo = None
        self.pixel = None
        self.spec = None
        self.roi = dict(pts=[], closed=False, drawing=False)
        self.coll, self.coll_n = [], 0
        self.lib_curves = []                      # [(spectrum id, name)]
        self.overlay = None
        self.legend = None
        self.map_on = False
        self._detail_seq = 0
        self.settings = QtCore.QSettings(*SETTINGS)
        try:
            self.objs = json.loads(self.settings.value("objects", "[]") or "[]")
        except ValueError:
            self.objs = []
        self.patch_folder = self.settings.value("patch_folder", "") or ""
        self._build()
        self._detail_t = QtCore.QTimer(self)
        self._detail_t.setSingleShot(True)
        self._detail_t.setInterval(350)
        self._detail_t.timeout.connect(self._detail)
        self.view.zoomed.connect(lambda: self.active and self._detail_t.start())
        self.refresh_objects()
        self.refresh_side()

    # ================================================================== layout
    def _build(self):
        lay = W.QVBoxLayout(self)
        lay.setContentsMargins(16, 10, 16, 10)
        lay.setSpacing(6)
        t = label("Image", "pagetitle")
        self.busy_lab = small("", wrap=False)
        lay.addLayout(hrow(t, self.busy_lab, None))
        # tool row 1
        self.b_open = btn("Open image…", self.open_dialog, kind="primary", accent="image")
        self.cb_img = W.QComboBox()
        self.cb_img.setMinimumWidth(240)
        self.cb_img.setMaximumWidth(360)
        self.cb_img.activated.connect(lambda i: self.open_image(self.cb_img.itemData(i)))
        self.b_remove = btn("Remove", self.remove_image, tip="Remove the image from the list (the file is not deleted)")
        self.cb_b = [W.QComboBox() for _ in range(3)]
        for cb in self.cb_b:
            cb.setMaximumWidth(150)
        self.b_bands = btn("Apply bands", self.apply_bands)
        self.b_bandlist = btn("Bands list…", self.bands_dialog, tip="Available Bands List (Gray Scale / RGB)")
        self.cb_win = W.QComboBox()
        for t_, v in (("1 px", 1), ("3×3", 3), ("5×5", 5)):
            self.cb_win.addItem(t_, v)
        self.cb_win.setToolTip("pixel window: average of 1, 3×3 or 5×5 pixels per click")
        self.b_clear_ov = btn("Clear class overlay", self.clear_overlay)
        self.b_spectral = W.QToolButton()
        self.b_spectral.setText("Spectral ▾")
        self.b_spectral.setPopupMode(W.QToolButton.InstantPopup)
        self.b_spectral.setMenu(self._spectral_menu())
        self.b_spectral.setMinimumHeight(30)
        self.row1_img = [self.b_remove, W.QLabel(" R"), self.cb_b[0], W.QLabel("G"), self.cb_b[1], W.QLabel("B"),
                         self.cb_b[2], self.b_bands, self.b_bandlist, self.cb_win, self.b_clear_ov]
        lay.addLayout(hrow(self.b_open, self.cb_img, *self.row1_img, None, self.b_spectral))
        # tool row 2
        self.b_geo = btn("📍 Georeference", self.georef, tip="Match points of the image with the map")
        self.b_map = btn("🗺 View map", self.toggle_map)
        self.b_roi = btn("✏ Draw ROI polygon", self.toggle_roi, tip="Click points on the image, double-click or right-click to finish")
        self.b_roi.setCheckable(True)
        self.b_clear_roi = btn("Clear ROI", self.clear_roi)
        self.b_dl = btn("Download image", self.download, tip="Save the colour composite as PNG")
        self.ck_px = W.QCheckBox("Pixel spectra")
        self.ck_px.setChecked(True)
        self.ck_px.setToolTip("the spectra of the clicked pixels: chart and numbered markers on the image")
        self.ck_obj = W.QCheckBox("Object samples (ROI)")
        self.ck_obj.setChecked(True)
        self.ck_obj.setToolTip("the ROI object samples: chart, table and outlines on the image")
        self.ck_px.toggled.connect(lambda _: self._show_parts())
        self.ck_obj.toggled.connect(lambda _: self._show_parts())
        self.roi_hint = small("")
        self.roi_hint.setStyleSheet("color:#047857;")
        zoom = [btn("−", lambda: self.view.zoom(0.8)), btn("Fit", lambda: self.view.fit()), btn("+", lambda: self.view.zoom(1.25))]
        self.row2 = [self.b_geo, self.b_map, self.b_roi, self.b_clear_roi, self.b_dl, W.QLabel("  <b>Show</b>"), self.ck_px,
                     self.ck_obj] + zoom
        lay.addLayout(hrow(*self.row2, self.roi_hint, stretch_at=len(self.row2)))
        # body
        self.split = W.QSplitter(Qt.Horizontal)
        left = W.QSplitter(Qt.Vertical)
        vbox = W.QWidget()
        vl = W.QVBoxLayout(vbox)
        vl.setContentsMargins(0, 0, 0, 0)
        vl.setSpacing(4)
        self.vstack = W.QStackedWidget()
        self.view = ImageView()
        self.view.clicked.connect(self.on_click)
        self.view.finish.connect(self.finish_roi)
        self.view.obj_clicked.connect(self.obj_menu)
        self.view.hover.connect(lambda r, c: self.info.setText(self._info_line(r, c)))
        self.mappanel = MapPanel()
        self.mappanel.pixel.connect(self.pick_pixel)
        self.vstack.addWidget(self.view)
        self.vstack.addWidget(self.mappanel)
        vl.addWidget(self.vstack, 1)
        self.info = small("")
        vl.addWidget(self.info)
        self.legend_box = W.QWidget()
        self.legend_lay = W.QVBoxLayout(self.legend_box)
        self.legend_lay.setContentsMargins(0, 0, 0, 0)
        vl.addWidget(self.legend_box)
        left.addWidget(vbox)
        self.obj_panel = self._objects_panel()
        left.addWidget(self.obj_panel)
        left.setSizes([620, 260])
        self.split.addWidget(left)
        self.side = self._side_panel()
        self.split.addWidget(self.side)
        self.split.setSizes([900, 470])
        self.split.setStretchFactor(0, 1)
        lay.addWidget(self.split, 1)
        self._enable()

    def _spectral_menu(self):
        m = W.QMenu(self)
        lm = m.addMenu("Spectral Libraries")
        for t_, tab in (("Spectral Library Builder", "builder"), ("Spectral Library Viewer", "viewer"),
                        ("Spectral Library Resampling", "resample")):
            lm.addAction(t_, lambda tab=tab: self.open_library_functions(tab))
        m.addAction("Spectral Analyst…", self.analyst)
        mm = m.addMenu("Mapping Methods")
        mm.addAction("Spectral Angle Mapper…", lambda: self.analysis("sam"))
        mm.addAction("Classify with a library (SAM map)…", self.sam_map)
        mm.addAction("Linear Spectral Unmixing…", lambda: self.analysis("unmix"))
        mm.addAction("K-Means Clustering…", lambda: self.analysis("kmeans"))
        tm = m.addMenu("Transforms")
        tm.addAction("Principal Components (PCA)…", lambda: self.analysis("pca"))
        tm.addAction("Band Math…", lambda: self.analysis("bandmath"))
        im = m.addMenu("Spectral Indices")
        for t_, expr in INDICES:
            im.addAction(t_, lambda expr=expr: self.analysis("bandmath", expression=expr, run=True))
        m.addAction("Anomaly Detection (RX)…", lambda: self.analysis("anomaly"))
        m.addAction("Signal-to-Noise Ratio…", lambda: self.analysis("snr"))
        m.addSeparator()
        m.addAction("Data Aggregation (ROI)…", self.aggregate)
        m.addSeparator()
        m.addAction("Open Spectral Library File…", lambda: ImportDialog(self, self.api, self.store).exec_())
        m.addAction("Available Bands List…", self.bands_dialog)
        m.addAction("Using the image page", self.help)
        return m

    def _side_panel(self):
        w = W.QWidget()
        w.setMinimumWidth(400)
        lay = W.QVBoxLayout(w)
        lay.setContentsMargins(6, 0, 0, 0)
        lay.setSpacing(6)
        f, l = card()
        self.spec_title = W.QLabel("Spectrum")
        self.spec_title.setWordWrap(True)
        l.addWidget(self.spec_title)
        self.chart = SpectrumChart(250)
        l.addWidget(self.chart, 1)
        self.chart_note = small("")
        l.addWidget(self.chart_note)
        self.ed_label = W.QLineEdit()
        self.ed_label.setPlaceholderText("label")
        self.ed_class = W.QLineEdit()
        self.ed_class.setPlaceholderText("class (e.g. Dry grass)")
        self.b_add = btn("Add", self.add_current, kind="primary", accent="library")
        self.b_csv_cur = btn("Export CSV", self.export_current)
        self.b_libclr = btn("Clear library curves", lambda: self.set_library_curves([]))
        l.addLayout(hrow(self.ed_label, self.ed_class, self.b_add, self.b_csv_cur, stretch_at=0))
        l.addLayout(hrow(None, self.b_libclr))
        lay.addWidget(f, 3)
        self.coll_box, cl = card()
        self.coll_head = label("Collected spectra", "section")
        cl.addWidget(self.coll_head)
        self.tbl_coll = W.QTableWidget(0, 3)
        self.tbl_coll.setHorizontalHeaderLabels(["", "Name", "Where"])
        self.tbl_coll.verticalHeader().setVisible(False)
        self.tbl_coll.horizontalHeader().setSectionResizeMode(1, W.QHeaderView.Stretch)
        self.tbl_coll.horizontalHeader().setSectionResizeMode(0, W.QHeaderView.ResizeToContents)
        self.tbl_coll.horizontalHeader().setSectionResizeMode(2, W.QHeaderView.ResizeToContents)
        self.tbl_coll.setSelectionBehavior(W.QAbstractItemView.SelectRows)
        self.tbl_coll.itemChanged.connect(self._coll_changed)
        cl.addWidget(self.tbl_coll, 1)
        cl.addLayout(hrow(btn("All", lambda: self._coll_all(True)), btn("None", lambda: self._coll_all(False)),
                          btn("Solo", self.coll_solo, tip="Show only the selected spectrum"),
                          btn("Remove", self.coll_remove), btn("Angles…", self.coll_angles), None))
        cl.addLayout(hrow(btn("Save to library…", self.coll_save, kind="primary", accent="library"),
                          btn("Export CSV", self.coll_csv), None))
        cl.addLayout(hrow(btn("Classify (SAM)…", lambda: self.analysis("sam"), tip="Use the collected spectra as endmembers"),
                          None, btn("Delete all", self.coll_clear)))
        lay.addWidget(self.coll_box, 2)
        return w

    def _objects_panel(self):
        f, l = card()
        f.setMinimumHeight(170)
        self.obj_head = label("Collected object samples (ROI)", "section")
        self.lab_folder = small("")
        l.addLayout(hrow(self.obj_head, None, self.lab_folder, btn("Change…", self.choose_folder,
                                                                    tip="Folder for the ROI image patches (<class>.tif)")))
        self.tbl_obj = table(["", "#", "Label", "Class", "Latitude", "Longitude", "Pixels", "Patch", "Source scene"], 2)
        self.tbl_obj.setEditTriggers(W.QAbstractItemView.DoubleClicked)
        self.tbl_obj.itemChanged.connect(self._obj_changed)
        self.tbl_obj.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tbl_obj.customContextMenuRequested.connect(
            lambda p: self.tbl_obj.rowAt(p.y()) >= 0 and self.obj_menu(self.tbl_obj.rowAt(p.y()), self.tbl_obj.viewport().mapToGlobal(p)))
        l.addWidget(self.tbl_obj, 1)
        l.addLayout(hrow(small("Right-click a row (or an ROI outline on the image) for Spectrum / Export patch / Locate / "
                               "Remove; double-click a label or class to edit it."), None,
                         btn("Save to library…", self.obj_save, kind="primary", accent="library"),
                         btn("Aggregate ticked ROIs…", self.aggregate, kind="primary", accent="image"),
                         btn("Delete all", self.obj_clear)))
        return f

    def _enable(self):
        has = self.active is not None and self.base_pixmap is not None
        for w in self.row1_img + self.row2[:5]:
            w.setEnabled(has)
        self.b_clear_ov.setVisible(self.overlay is not None)
        self.b_clear_roi.setEnabled(has and bool(self.roi["pts"]))
        self.b_map.setText("🗺 Hide map" if self.map_on else "🗺 View map")

    def _show_parts(self):
        self.coll_box.setVisible(self.ck_px.isChecked())
        self.obj_panel.setVisible(self.ck_obj.isChecked())
        self.refresh_view()
        self.refresh_side()

    def flash(self, text, ms=6000):
        try:
            self.shell.flash(text, ms)
        except Exception:                                           # noqa: BLE001
            pass
        self.busy_lab.setText(text)
        QtCore.QTimer.singleShot(ms, lambda: self.busy_lab.text() == text and self.busy_lab.setText(""))

    def help(self):
        W.QMessageBox.information(self, "Using the image page", (
            "1. Open image…: choose the .hdr (or .tif, .mat, .zip …). In the Available Bands List pick Gray Scale or RGB, "
            "click a band for each colour, then Load.\n\n"
            "2. Click pixels one after another: the spectrum is shown on the right. Give it a label and press Add to list it "
            "under Collected spectra.\n\n"
            "3. Draw ROI polygon (double-click or right-click to finish) for a mean spectrum; give it a label and class and "
            "press Add: it goes to Collected object samples, with its image patch saved as <class>.tif.\n\n"
            "4. Georeference / View map: match control points with the map, or see the image on the map.\n\n"
            "5. Spectral menu: Spectral Analyst, SAM, unmixing, k-means, PCA, band math / indices, anomaly, SNR, aggregation."))

    # ================================================================== images
    def refresh(self):
        """Reload the image list (the page became visible / data changed elsewhere)."""
        run_bg(self.api.get, "/api/images/", done=self._images_loaded, failed=lambda m: self.busy_lab.setText(m))

    def _images_loaded(self, lst):
        self.images = lst if isinstance(lst, list) else lst.get("items", [])
        cur = self.active["id"] if self.active else None
        self.cb_img.blockSignals(True)
        self.cb_img.clear()
        for im in self.images:
            self.cb_img.addItem(f"{im['name']} ({im['rows']}×{im['cols']}×{im['bands']})", im["id"])
        if not self.images:
            self.cb_img.addItem("(no images yet)", None)
        i = self.cb_img.findData(cur) if cur else -1
        self.cb_img.setCurrentIndex(max(0, i))
        self.cb_img.blockSignals(False)
        if self.active and not any(im["id"] == self.active["id"] for im in self.images):
            self._close_image()

    def open_dialog(self, wl=None):
        paths, _ = W.QFileDialog.getOpenFileNames(
            self, "Open image file", "", "Images (*.hdr *.tif *.tiff *.mat *.zip *.h5 *.hdf *.he5 *.nc *.img *.dat);;All files (*.*)")
        if paths:
            self._open_paths(paths, wl)

    def _open_paths(self, paths, wl=None):
        self.busy_lab.setText("⌛ Reading the file(s)…")
        run_bg(self.api.post, "/api/import/path", dict(paths=paths, wavelengths=wl or None),
               done=lambda rep: self._after_open(rep, paths), failed=lambda m: (self.busy_lab.setText(""),
                                                                               W.QMessageBox.warning(self, "Open image", m)))

    def _after_open(self, rep, paths):
        self.busy_lab.setText("")
        self.store.refresh()
        try:
            lst = self.api.get("/api/images/")
        except Exception:                                           # noqa: BLE001
            lst = self.images
        self._images_loaded(lst)
        got = next((f for f in rep.get("files", []) if f.get("kind") == "image" and f.get("id")), None)
        bad = [f for f in rep.get("files", []) if f.get("status") == "error"]
        if not got and rep.get("duplicates"):
            stem = next((f.get("file") or "" for f in rep.get("files", []) if f.get("status") == "duplicate"), "")
            base = stem.rsplit(".", 1)[0]
            m = next((i for i in self.images if stem.startswith(i["name"]) or i["name"].startswith(base)), None)
            if m:
                got = dict(id=m["id"])
        if got:
            return self.bands_dialog(got["id"])
        if rep.get("imported"):
            return self.flash(f"{rep['imported']} spectra were added to the library “{rep.get('library')}” (no image in these files).")
        txt = "\n".join(f"{f['file']}: {f.get('detail')}" for f in bad) or \
            "No hyperspectral image was found in the chosen file(s). For ENVI, choose the .hdr file."
        wl, ok = W.QInputDialog.getText(self, "Could not open", txt + "\n\nIf the file has no wavelengths, give the range of the "
                                        "sensor (for example 430-860) and open it again:")
        if ok and wl.strip():
            self._open_paths(paths, wl.strip())

    def bands_dialog(self, image_id=None):
        image_id = image_id or (self.active["id"] if self.active else (self.images[0]["id"] if self.images else None))
        if not image_id:
            return self.open_dialog()
        info = next((i for i in self.images if i["id"] == image_id), None)

        def load():
            inf = info or self.api.get(f"/api/images/{image_id}")
            return inf, self.api.get(f"/api/images/{image_id}/bands")

        def done(r):
            inf, bd = r
            cur = self.bands if self.active and self.active["id"] == image_id else None
            d = BandsDialog(self, inf, bd.get("wavelengths") or [], cur)
            if d.exec_():
                self.open_image(image_id, d.bands())
        run_bg(load, done=done, failed=lambda m: W.QMessageBox.warning(self, "Bands", m))

    def open_image(self, image_id, bands=None):
        if not image_id:
            return
        info = next((i for i in self.images if i["id"] == image_id), None)
        self.busy_lab.setText("⌛ Loading the image…")

        def load():
            inf = info or self.api.get(f"/api/images/{image_id}")
            bd = self.api.get(f"/api/images/{image_id}/bands")
            wl = bd.get("wavelengths") or []
            b = bands or [x + 1 for x in default_rgb(wl)]
            pv = self.api.get(f"/api/images/{image_id}/preview", r=b[0], g=b[1], b=b[2], max_px=self._want_px(inf))
            try:
                geo = self.api.get(f"/api/images/{image_id}/geo")
            except Exception:                                       # noqa: BLE001
                geo = None
            return inf, bd, b, pv, geo

        def done(r):
            inf, bd, b, pv, geo = r
            new = not self.active or self.active["id"] != inf["id"]
            self.active, self.wl, self.bad, self.bands, self.geo = inf, bd.get("wavelengths") or [], bd.get("bad") or [], b, geo
            self.preview = pv
            self.base_pixmap = pixmap_from_data_url(pv["image"])
            if new:
                self.pixel = self.spec = None
                self.roi = dict(pts=[], closed=False, drawing=False)
                self.b_roi.setChecked(False)
                self.view.drawing = False
                self.overlay, self.legend = None, None
                self.view.set_overlay(None)
                self.lib_curves = []
            self.view.set_base(self.base_pixmap, inf["rows"], inf["cols"], fit=new)
            i = self.cb_img.findData(inf["id"])
            if i >= 0:
                self.cb_img.setCurrentIndex(i)
            for k, cb in enumerate(self.cb_b):
                cb.blockSignals(True)
                cb.clear()
                for j in range(int(inf["bands"])):
                    w = self.wl[j] if j < len(self.wl) else None
                    cb.addItem(f"Band {j + 1}" + (f" · {round(w)} nm" if w is not None else ""), j + 1)
                cb.setCurrentIndex(max(0, b[k] - 1))
                cb.blockSignals(False)
            self.busy_lab.setText("")
            if self.map_on:
                self._map_show()
            self._enable()
            self.refresh_view()
            self.refresh_side()
            self.refresh_objects()

        def fail(m):
            self.busy_lab.setText("")
            W.QMessageBox.warning(self, "Image", m)
        run_bg(load, done=done, failed=fail)

    def _want_px(self, inf=None):
        inf = inf or self.active
        vw = max(400, self.view.viewport().width()) * max(300, self.view.viewport().height())
        return int(min(6_000_000, max(300_000, vw * 1.3), int(inf["rows"]) * int(inf["cols"])))

    def _detail(self):
        """Zoomed in beyond the preview resolution: ask for a sharper preview."""
        if not self.active or not self.preview:
            return
        a = self.active
        s = self.view.scale_px()
        vp = self.view.viewport()
        vis_cols = min(a["cols"], vp.width() / max(s, 1e-6))
        vis_rows = min(a["rows"], vp.height() / max(s, 1e-6))
        need_step = max(1.0, min(vis_cols / max(1, vp.width()), vis_rows / max(1, vp.height())))
        cur_step = a["cols"] / max(1, self.base_pixmap.width())
        if cur_step <= need_step * 1.3 or cur_step <= 1.0:
            return
        want = int(min(6_000_000, a["rows"] * a["cols"] / (need_step * need_step)))
        if want <= self.base_pixmap.width() * self.base_pixmap.height() * 1.3:
            return
        self._detail_seq += 1
        seq, iid, b = self._detail_seq, a["id"], list(self.bands)
        self.busy_lab.setText("⌛ Loading detail…")

        def done(pv):
            if seq != self._detail_seq or not self.active or self.active["id"] != iid:
                return
            self.busy_lab.setText("")
            self.preview = pv
            self.base_pixmap = pixmap_from_data_url(pv["image"])
            self.view.set_base(self.base_pixmap, a["rows"], a["cols"])
        run_bg(self.api.get, f"/api/images/{iid}/preview", r=b[0], g=b[1], b=b[2], max_px=want, done=done,
               failed=lambda m: self.busy_lab.setText(""))

    def apply_bands(self):
        if self.active:
            self.open_image(self.active["id"], [cb.currentData() or 1 for cb in self.cb_b])

    def remove_image(self):
        a = self.active
        if not a or W.QMessageBox.question(self, "Remove image", f"Remove “{a['name']}” from the list?\n(The file itself is "
                                           "not deleted.)") != W.QMessageBox.Yes:
            return

        def done(_):
            self._close_image()
            self.refresh()
            self.store.refresh()
        run_bg(self.api.delete, f"/api/images/{a['id']}", done=done, failed=lambda m: W.QMessageBox.warning(self, "Remove", m))

    def _close_image(self):
        self.active = self.preview = self.base_pixmap = self.geo = None
        self.pixel = self.spec = None
        self.roi = dict(pts=[], closed=False, drawing=False)
        self.overlay = self.legend = None
        self.view.clear()
        if self.map_on:
            self.toggle_map()
        self._enable()
        self.refresh_side()
        self.refresh_view()

    def download(self):
        if self.base_pixmap is not None:
            p, _ = W.QFileDialog.getSaveFileName(self, "Save image", f"{self.active['name']}_preview.png", "PNG (*.png)")
            if p:
                self.base_pixmap.save(p, "PNG")

    # ================================================================== geo
    def geo_corners(self):
        g = self.geo or {}
        cs = g.get("corners")
        if g.get("has_geo") and cs and len(cs) == 4 and all(c and c.get("lat") is not None for c in cs):
            return [(c["lon"], c["lat"]) for c in cs]
        return None

    def georef(self):
        if self.active and self.base_pixmap is not None:
            GeorefDialog(self).exec_()

    def after_georef(self, geo):
        def done(g):
            self.geo = g
            self._fill_coords(self.objs)
            if self.map_on:
                if self.geo_corners():
                    self._map_show()
                else:
                    self.toggle_map()
            self.refresh_side()
        run_bg(self.api.get, f"/api/images/{self.active['id']}/geo", done=done)

    def toggle_map(self):
        if not self.map_on:
            if not self.active:
                return
            if not self.geo_corners():
                return W.QMessageBox.information(self, "View map", "This image has no map coordinates. Use Georeference to "
                                                 "match a few points of the image with the map first.")
            self.map_on = True
            self._map_show()
            self.vstack.setCurrentWidget(self.mappanel)
        else:
            self.map_on = False
            self.vstack.setCurrentWidget(self.view)
        self._enable()

    def _map_show(self):
        a, b = self.active, list(self.bands)

        def done(pv):
            pm = pixmap_from_data_url(pv["image"])
            self.mappanel.show_image(pm, self.geo_corners(), a["rows"], a["cols"])
            self._map_markers()
        self.mappanel.show_image(self.base_pixmap, self.geo_corners(), a["rows"], a["cols"])
        run_bg(self.api.get, f"/api/images/{a['id']}/preview", r=b[0], g=b[1], b=b[2], alpha=True, max_px=1_500_000,
               done=done)

    def _lonlat_local(self, row, col):
        """lon/lat of a pixel from the 4 corners (bilinear) — for markers; the backend is used for stored values."""
        cs = self.geo_corners()
        if not cs or not self.active:
            return None
        u, v = (col + 0.5) / self.active["cols"], (row + 0.5) / self.active["rows"]
        (x0, y0), (x1, y1), (x2, y2), (x3, y3) = cs
        lon = (1 - u) * (1 - v) * x0 + u * (1 - v) * x1 + u * v * x2 + (1 - u) * v * x3
        lat = (1 - u) * (1 - v) * y0 + u * (1 - v) * y1 + u * v * y2 + (1 - u) * v * y3
        return lon, lat

    def _map_markers(self):
        if not self.map_on:
            return
        mk = []
        if self.pixel:
            ll = self._lonlat_local(*self.pixel)
            if ll:
                mk.append((ll[0], ll[1], None, "#fbbf24"))
        if self.ck_px.isChecked():
            for i, c in enumerate(self.coll):
                if c.get("show") and c.get("kind") == "px" and (c.get("img") or {}).get("id") == self.active["id"]:
                    ll = self._lonlat_local(c["row"], c["col"])
                    if ll:
                        mk.append((ll[0], ll[1], str(i + 1), c["color"]))
        self.mappanel.map.set_markers(mk)

    # ================================================================== clicking / ROI
    def on_click(self, row, col):
        if self.roi["drawing"]:
            self.roi["pts"].append([row + 0.5, col + 0.5])
            self.roi_hint.setText(self._roi_hint())
            self._enable()
            return self.refresh_view()
        self.pick_pixel(row, col)

    def pick_pixel(self, row, col):
        if not self.active:
            return
        size = self.cb_win.currentData() or 1
        self.pixel = (row, col)
        iid = self.active["id"]

        def done(s):
            if not self.active or self.active["id"] != iid:
                return
            self.spec = dict(s, kind="pixel")
            if not self.roi["closed"]:
                self.ed_label.setText(f"pixel {row},{col}")
            self.refresh_side()
        run_bg(self.api.get, f"/api/images/{iid}/pixel", row=row, col=col, size=size, done=done,
               failed=lambda m: self.flash(m))
        self.refresh_view()
        self._map_markers()

    def toggle_roi(self):
        if not self.active:
            return
        if self.roi["drawing"]:
            return self.clear_roi()
        self.roi = dict(pts=[], closed=False, drawing=True)
        self.view.drawing = True
        self.b_roi.setChecked(True)
        if self.map_on:
            self.toggle_map()
        self.roi_hint.setText(self._roi_hint())
        self.refresh_view()

    def _roi_hint(self):
        n = len(self.roi["pts"])
        return f"Drawing ROI — {n} point{'s' if n != 1 else ''}. Click to add points · double-click or right-click to finish"

    def finish_roi(self):
        if not self.roi["drawing"]:
            return
        if len(self.roi["pts"]) < 3:
            self.roi_hint.setText("An ROI needs at least 3 points.")
            return
        self.roi.update(closed=True, drawing=False)
        self.view.drawing = False
        self.b_roi.setChecked(False)
        self.roi_hint.setText("")
        self.refresh_view()
        poly, iid = [list(p) for p in self.roi["pts"]], self.active["id"]

        def done(s):
            if not self.active or self.active["id"] != iid:
                return
            self.spec = dict(s, kind="roi")
            n = len(self.objs) + 1
            self.ed_label.setText(f"ROI {n}")
            self.ed_class.setText("")
            self.ed_class.setFocus()
            self.refresh_side()
        run_bg(self.api.post, f"/api/images/{iid}/roi", dict(polygon=poly), done=done, failed=lambda m: self.flash(m))

    def clear_roi(self):
        self.roi = dict(pts=[], closed=False, drawing=False)
        self.view.drawing = False
        self.b_roi.setChecked(False)
        self.roi_hint.setText("")
        if self.spec and self.spec.get("kind") == "roi":
            self.spec = None
        self._enable()
        self.refresh_view()
        self.refresh_side()

    def current_selection(self):
        """For the Spectral Library Builder: the current pixel / ROI spectrum as a raw spectrum."""
        s, a = self.spec, self.active
        if not s or not a:
            return None
        nm = f"{a['name']} ROI" if s["kind"] == "roi" else f"{a['name']} pixel {s.get('row')},{s.get('col')}"
        return dict(name=nm, wavelengths=s["wavelengths"], values=s["values"], unit=s.get("unit"), source=a["name"],
                    origin="image_roi" if s["kind"] == "roi" else "image_pixel")

    def _target(self):
        if self.roi["closed"] and len(self.roi["pts"]) >= 3:
            return dict(polygon=self.roi["pts"]), "ROI mean"
        if self.pixel:
            return dict(row=self.pixel[0], col=self.pixel[1], size=self.cb_win.currentData() or 1), \
                f"pixel ({self.pixel[0]}, {self.pixel[1]})"
        c = next((c for c in reversed(self.coll) if c.get("kind") == "px" and c.get("row", -1) >= 0), None)
        if c:
            return dict(row=c["row"], col=c["col"], size=1), "collected spectrum"
        return None, None

    # ================================================================== current spectrum
    def add_current(self):
        s = self.spec
        if not s:
            return self.flash("Click a pixel (or finish an ROI) first.")
        if s["kind"] == "roi":
            return self.obj_add(s, self.ed_label.text(), self.ed_class.text())
        if len(self.coll) >= 24:
            return self.flash("At most 24 collected spectra: remove some first.")
        a = self.active
        if any(c.get("row") == s["row"] and c.get("col") == s["col"] and (c.get("img") or {}).get("id") == a["id"]
               for c in self.coll):
            return self.flash("This pixel is already in the list.")
        self.coll_n += 1
        self.coll.append(dict(name=self.ed_label.text().strip() or f"pixel {s['row']},{s['col']}",
                              color=DS_COLORS[(self.coll_n - 1) % len(DS_COLORS)], show=True, kind="px", row=s["row"],
                              col=s["col"], n=s.get("n_pixels") or 1, unit=s.get("unit"), wavelengths=s["wavelengths"],
                              values=s["values"], img=dict(id=a["id"], name=a["name"], path=a.get("path") or "")))
        self.refresh_side()
        self.refresh_view()
        self._map_markers()

    def export_current(self):
        s = self.spec
        if not s:
            return
        rows = [[w, v] for w, v in zip(s["wavelengths"], s["values"]) if v is not None]
        save_csv(self, ("roi_mean" if s["kind"] == "roi" else f"pixel_{s['row']}_{s['col']}") + ".csv",
                 ["wavelength_nm", s.get("unit") or "value"], rows)

    def set_library_curves(self, items):
        self.lib_curves = list(items)[:8]
        self.refresh_side()

    def add_spectra_to_collected(self, spectra):
        for s in spectra:
            if len(self.coll) >= 24:
                break
            self.coll_n += 1
            self.coll.append(dict(name=s["name"], color=DS_COLORS[(self.coll_n - 1) % len(DS_COLORS)], show=True, kind="px",
                                  row=-1, col=-1, n=0, unit=s.get("unit"), wavelengths=self.wl, values=s["values"], img=None))
        self.refresh_side()

    def set_overlay(self, pixmap, legend, unmatched=None):
        self.overlay = pixmap
        self.legend = (legend, unmatched)
        self.view.set_overlay(pixmap)
        if self.map_on:
            self.toggle_map()
        self._enable()
        self.refresh_view()

    def clear_overlay(self):
        self.overlay = self.legend = None
        self.view.set_overlay(None)
        self._enable()
        self.refresh_view()

    # ================================================================== drawing
    def _info_line(self, row=None, col=None):
        a = self.active
        if not a:
            return ""
        t = (f"{a['name']} · {a['cols']}×{a['rows']} px · {a['bands']} bands · {a.get('unit') or ''} · "
             f"bands {'/'.join(str(b) for b in self.bands)} · zoom {self.view.scale_px():.2f}×")
        if row is not None:
            t += f" · pixel ({row}, {col})"
            ll = self._lonlat_local(row, col)
            if ll:
                t += f" · {ll[1]:.5f}, {ll[0]:.5f}"
        if self.preview and self.preview.get("step", 1) > 1:
            t += f" · preview step {self.preview['step']}"
        return t

    def refresh_view(self):
        items = []
        objs_poly = []
        a = self.active
        if a:
            if self.ck_obj.isChecked():
                for i, o in enumerate(self.objs):
                    if o.get("image_id") == a["id"] and o.get("show", True):
                        items.append(poly_item(o["polygon"], o["color"]))
                        r = sum(p[0] for p in o["polygon"]) / len(o["polygon"])
                        c = sum(p[1] for p in o["polygon"]) / len(o["polygon"])
                        t = W.QGraphicsSimpleTextItem(o["name"])
                        t.setBrush(QtGui.QColor(o["color"]))
                        t.setFlag(W.QGraphicsItem.ItemIgnoresTransformations)
                        t.setPos(c, r)
                        t.setZValue(6)
                        items.append(t)
                        objs_poly.append((i, [QtCore.QPointF(p[1], p[0]) for p in o["polygon"]]))
            pts = self.roi["pts"]
            if pts:
                items.append(poly_item(pts, "#22d3ee", closed=self.roi["closed"], dash=not self.roi["closed"]))
                if not self.roi["closed"]:
                    for p in pts:
                        items.append(marker(p[1], p[0], "#22d3ee", r=3, width=2))
            if self.ck_px.isChecked():
                for i, c in enumerate(self.coll):
                    if c.get("show") and c.get("kind") == "px" and (c.get("img") or {}).get("id") == a["id"] and c["row"] >= 0:
                        items.append(marker(c["col"] + 0.5, c["row"] + 0.5, c["color"], str(i + 1)))
            if self.pixel and not self.roi["drawing"]:
                items.append(marker(self.pixel[1] + 0.5, self.pixel[0] + 0.5, "#fbbf24", r=7, width=2))
        self.view.set_annotations(items)
        self.view.objs_poly = objs_poly
        self.info.setText(self._info_line())
        while self.legend_lay.count():
            it = self.legend_lay.takeAt(0)
            if it.widget():
                it.widget().deleteLater()
        if self.legend:
            lg, un = self.legend
            self.legend_lay.addWidget(legend_widget([l for l in lg if l.get("pixels", 1) > 0], un))
        self._enable()

    def refresh_side(self):
        s = self.spec
        if s:
            head = (f"<b>ROI mean</b> · {s.get('n_pixels')} pixels" if s["kind"] == "roi" else
                    f"<b>Pixel ({s.get('row')}, {s.get('col')})</b>" + (f" · mean of {s['n_pixels']} px" if (s.get("n_pixels") or 1) > 1 else ""))
            head += f" · {s.get('unit') or ''}"
            if s["kind"] == "pixel":
                ll = self._lonlat_local(s.get("row"), s.get("col"))
                if ll:
                    head += f" · {ll[1]:.5f}, {ll[0]:.5f}"
            self.spec_title.setText(head)
        else:
            self.spec_title.setText("<b>Spectrum</b> · click a pixel on the image")
        roi = bool(s and s["kind"] == "roi")
        self.ed_class.setVisible(roi)
        self.b_add.setText("Add to object samples" if roi else "Add")
        self.b_add.setEnabled(bool(s))
        self.b_csv_cur.setEnabled(bool(s))
        self.b_libclr.setVisible(bool(self.lib_curves))
        self._draw_chart()
        self._fill_coll()

    def _draw_chart(self):
        s, a = self.spec, self.active
        ser, unit = [], None
        if self.ck_px.isChecked():
            for c in self.coll:
                if c.get("show"):
                    x, y = xy(c["wavelengths"], c["values"])
                    ser.append(dict(name=c["name"], x=x, y=y, color=c["color"]))
                    unit = unit or c.get("unit")
        if self.ck_obj.isChecked():
            for o in self.objs:
                if o.get("show", True) and o.get("values"):
                    x, y = xy(o["wavelengths"], o["values"])
                    ser.append(dict(name=f"{o['name']} (ROI)", x=x, y=y, color=o["color"], ls="--"))
                    unit = unit or o.get("unit")
        if s:
            in_coll = (any(same_poly(o["polygon"], self.roi["pts"]) for o in self.objs if a and o.get("image_id") == a["id"])
                       if s["kind"] == "roi" else
                       any(c.get("show") and c.get("row") == s.get("row") and c.get("col") == s.get("col") and
                           (c.get("img") or {}).get("id") == (a or {}).get("id") for c in self.coll))
            if not in_coll:
                x, y = xy(s["wavelengths"], s["values"])
                nm = (f"ROI mean ({s.get('n_pixels')} px)" if s["kind"] == "roi" else f"pixel ({s['row']}, {s['col']})") + " (not added)"
                ser.insert(0, dict(name=nm, x=x, y=y, color="#1b2430", lw=1.8))
                unit = unit or s.get("unit")
        if not ser and not self.lib_curves:
            self.chart_note.setText("")
            return self.chart.message("Tick Pixel spectra and/or Object samples above to plot them."
                                      if not self.ck_px.isChecked() and not self.ck_obj.isChecked() else
                                      "Click a pixel on the image to see its spectrum here.")
        ref = ser[0] if ser else None
        curves = list(self.lib_curves)

        def load():
            return [(sid, nm, get_spectrum(self.api, sid)) for sid, nm in curves]

        def done(rows):
            rescaled = False
            out = list(ser)
            for k, (sid, nm, sp) in enumerate(rows):
                x, y = xy(sp["wavelengths"], sp["values"])
                if unit and sp.get("measurement_unit") != unit and ref is not None:
                    m, pm = np.nanmax(y) if np.isfinite(y).any() else 0, np.nanmax(ref["y"]) if np.isfinite(ref["y"]).any() else 0
                    if m > 0:
                        y = y / m * pm
                        rescaled = True
                out.append(dict(name=f"{nm} (library)", x=x, y=y, color=LIB_GREY[k % len(LIB_GREY)], ls=":", lw=1.6))
            self.chart.plot(out, unit=unit)
            self.chart_note.setText("Library curves are scaled to the image spectrum (units differ), so only the shape is "
                                    "comparable." if rescaled else "")
        if curves:
            run_bg(load, done=done, failed=lambda m: done([]))
        else:
            done([])

    # ================================================================== collected spectra
    def _fill_coll(self):
        self.coll_head.setText(f"Collected spectra ({len(self.coll)})")
        self.tbl_coll.blockSignals(True)
        self.tbl_coll.setRowCount(len(self.coll))
        for i, c in enumerate(self.coll):
            it = W.QTableWidgetItem()
            it.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            it.setCheckState(Qt.Checked if c.get("show") else Qt.Unchecked)
            pm = QtGui.QPixmap(12, 12)
            pm.fill(QtGui.QColor(c["color"]))
            it.setIcon(QtGui.QIcon(pm))
            it.setText(str(i + 1))
            self.tbl_coll.setItem(i, 0, it)
            nm = W.QTableWidgetItem(c["name"])
            self.tbl_coll.setItem(i, 1, nm)
            where = (f"{c['row']}, {c['col']}" if c.get("row", -1) >= 0 else "analysis") + \
                (f" · {c['img']['name']}" if c.get("img") and self.active and c["img"]["id"] != self.active["id"] else "")
            w = W.QTableWidgetItem(where)
            w.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            self.tbl_coll.setItem(i, 2, w)
        self.tbl_coll.blockSignals(False)

    def _coll_changed(self, it):
        i = it.row()
        if not 0 <= i < len(self.coll):
            return
        if it.column() == 0:
            self.coll[i]["show"] = it.checkState() == Qt.Checked
        elif it.column() == 1:
            self.coll[i]["name"] = it.text().strip() or self.coll[i]["name"]
        self._draw_chart()
        self.refresh_view()
        self._map_markers()

    def _sel_rows(self, tbl):
        return sorted({i.row() for i in tbl.selectedIndexes()})

    def _coll_all(self, on):
        for c in self.coll:
            c["show"] = on
        self.refresh_side()
        self.refresh_view()

    def coll_solo(self):
        rows = self._sel_rows(self.tbl_coll)
        if not rows:
            return self.flash("Select a row of the list first.")
        for i, c in enumerate(self.coll):
            c["show"] = i in rows
        self.refresh_side()
        self.refresh_view()

    def coll_remove(self):
        for i in reversed(self._sel_rows(self.tbl_coll)):
            self.coll.pop(i)
        self.refresh_side()
        self.refresh_view()

    def coll_clear(self):
        if self.coll and W.QMessageBox.question(self, "Collected spectra", "Delete all collected spectra?") == W.QMessageBox.Yes:
            self.coll, self.coll_n = [], 0
            self.refresh_side()
            self.refresh_view()

    def coll_angles(self):
        c = [x for x in self.coll if x.get("show")] or self.coll
        if len(c) < 2:
            return self.flash("Collect at least 2 spectra.")
        d = W.QDialog(self)
        d.setWindowTitle("Spectral angle (°) between the collected spectra")
        d.resize(720, 460)
        l = W.QVBoxLayout(d)
        t = W.QTableWidget(len(c), len(c))
        t.setHorizontalHeaderLabels([str(i + 1) for i in range(len(c))])
        t.setVerticalHeaderLabels([f"{i + 1} {x['name'][:24]}" for i, x in enumerate(c)])
        t.setEditTriggers(W.QAbstractItemView.NoEditTriggers)
        for i in range(len(c)):
            for j in range(len(c)):
                a = 0.0 if i == j else angle_deg(c[i]["values"], c[j]["values"])
                t.setItem(i, j, cell("–" if a is None else f"{a:.2f}", align=Qt.AlignCenter))
        t.resizeColumnsToContents()
        l.addWidget(t)
        l.addWidget(small("Smaller angle = more similar shape (independent of brightness). Spectra of different images are "
                          "compared band by band."))
        l.addWidget(btn("Close", d.accept))
        d.exec_()

    def coll_csv(self):
        c = self.coll
        if not c:
            return
        wl = c[0]["wavelengths"]
        rows = [[w] + ["" if k >= len(s["values"]) or s["values"][k] is None else s["values"][k] for s in c]
                for k, w in enumerate(wl)]
        save_csv(self, f"{(self.active or {}).get('name') or 'image'}_collected_spectra.csv",
                 ["wavelength_nm"] + [s["name"] for s in c], rows)

    def coll_save(self):
        if not self.coll:
            return self.flash("Nothing to save.")
        nm = f"{(self.active or {}).get('name') or 'Image'} spectra"
        d = SaveDialog(self, "Save collected spectra to a library", f"{len(self.coll)} spectra will be stored in the database.",
                       self.store, nm, "pixel")
        if not d.exec_():
            return
        v = d.values()
        items = []
        for s in self.coll:
            im = s.get("img") or ({"id": self.active["id"], "name": self.active["name"], "path": self.active.get("path") or ""}
                                  if self.active else {})
            items.append(dict(name=s["name"], wavelengths_nm=s["wavelengths"], values=s["values"], unit=s.get("unit"),
                              category=v["category"], origin="image_pixel", acquisition_type=v["acquisition"],
                              sample_metadata=dict(pixel=dict(image_id=im.get("id"), image_name=im.get("name"),
                                                              image_path=im.get("path"), row=s.get("row"), col=s.get("col")))))
        run_bg(self.api.post, "/api/spectra/save", dict(library=v["library"], items=items),
               done=lambda r: (self.store.refresh(), self.flash(
                   f"Saved {r['saved']} spectra to library “{r['library']}”" +
                   (f" ({len(r['rejected'])} rejected)" if r.get("rejected") else ""))),
               failed=lambda m: W.QMessageBox.warning(self, "Save", m))

    # ================================================================== object samples (ROI)
    def _store_objs(self):
        keep = [{k: v for k, v in o.items()} for o in self.objs]
        self.settings.setValue("objects", json.dumps(keep))
        self.settings.setValue("patch_folder", self.patch_folder)

    def obj_add(self, s, lab, cls):
        a = self.active
        if not a or not self.roi["closed"]:
            return
        poly = [list(p) for p in self.roi["pts"]]
        if any(o.get("image_id") == a["id"] and same_poly(o["polygon"], poly) for o in self.objs):
            return self.flash("This ROI is already in the list.")
        name = lab.strip() or f"ROI {len(self.objs) + 1}"
        o = dict(image_id=a["id"], image_name=a["name"], image_path=a.get("path") or "", polygon=poly,
                 class_name=cls.strip() or name, name=name, color=OBJ_COLORS[len(self.objs) % len(OBJ_COLORS)], show=True,
                 n=s.get("n_pixels") or 0, unit=s.get("unit"), wavelengths=s["wavelengths"], values=s["values"],
                 std=s.get("std"), lat=None, lon=None, patch=None, patchErr="")
        self.objs.append(o)
        self.roi = dict(pts=[], closed=False, drawing=False)
        self.spec = dict(self.spec, kind="pixel") if self.pixel and self.spec and self.spec.get("row") is not None else None
        self._store_objs()
        self.refresh_view()
        self.refresh_side()
        self.refresh_objects()
        self.flash(f"“{name}” added to the object samples")
        self._fill_coords([o])
        if not self.patch_folder:
            self.choose_folder()
        else:
            self.save_patch(o)

    def _fill_coords(self, lst):
        by = {}
        for o in lst:
            by.setdefault(o["image_id"], []).append(o)
        for iid, os_ in by.items():
            pts = [[sum(p[0] for p in o["polygon"]) / len(o["polygon"]), sum(p[1] for p in o["polygon"]) / len(o["polygon"])]
                   for o in os_]

            def done(r, os_=os_):
                for k, o in enumerate(os_):
                    p = (r.get("points") or [None] * len(os_))[k]
                    o["lat"], o["lon"] = (p["lat"], p["lon"]) if p else (None, None)
                self._store_objs()
                self.refresh_objects()
            run_bg(self.api.post, f"/api/images/{iid}/lonlat", dict(points=pts), done=done)

    def choose_folder(self):
        def go(default):
            path = W.QFileDialog.getExistingDirectory(
                self, "Where should the ROI image patches be saved? (one GeoTIFF per ROI, named after its class)",
                self.patch_folder or default or "")
            if not path:
                return

            def ok(r):
                self.patch_folder = r.get("path") or path
                self._store_objs()
                self.refresh_objects()
                for o in self.objs:
                    if not o.get("patch"):
                        self.save_patch(o)
            run_bg(self.api.post, "/api/system/check_folder", dict(path=path), done=ok,
                   failed=lambda m: W.QMessageBox.warning(self, "Patch folder", m))
        if self.patch_folder:
            go(self.patch_folder)
        else:
            run_bg(self.api.get, "/api/system/patch_folder", done=lambda r: go(r.get("default") or ""),
                   failed=lambda m: go(""))

    def _obj_bands(self, o):
        return list(self.bands) if self.active and self.active["id"] == o["image_id"] else None

    def save_patch(self, o):
        if not self.patch_folder:
            return self.choose_folder()

        def done(r):
            o["patch"] = dict(file=r.get("file"), path=r.get("path"), folder=r.get("folder"))
            o["patchErr"] = ""
            if r.get("lat") is not None:
                o["lat"], o["lon"] = r["lat"], r["lon"]
            self._store_objs()
            self.refresh_objects()

        def fail(m):
            o["patch"], o["patchErr"] = None, m
            self._store_objs()
            self.refresh_objects()
            self.flash("Patch not saved: " + m)
        run_bg(self.api.post, f"/api/images/{o['image_id']}/patch",
               dict(polygon=o["polygon"], class_name=o["class_name"], folder=self.patch_folder, bands=self._obj_bands(o)),
               done=done, failed=fail)

    def refresh_objects(self):
        n = len(self.objs)
        self.obj_head.setText(f"Collected object samples (ROI) · {n}")
        self.lab_folder.setText(f"Patch folder: {self.patch_folder}" if self.patch_folder else "Patch folder: not chosen yet")
        self.tbl_obj.blockSignals(True)
        self.tbl_obj.setRowCount(n)
        f5 = lambda v: "—" if v is None else f"{float(v):.5f}"                     # noqa: E731
        for i, o in enumerate(self.objs):
            it = W.QTableWidgetItem()
            it.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            it.setCheckState(Qt.Checked if o.get("show", True) else Qt.Unchecked)
            pm = QtGui.QPixmap(12, 12)
            pm.fill(QtGui.QColor(o["color"]))
            it.setIcon(QtGui.QIcon(pm))
            self.tbl_obj.setItem(i, 0, it)
            patch = (f"📷 {o['patch']['file']}" if o.get("patch") else ("✗ " + o["patchErr"][:40] if o.get("patchErr") else "—"))
            vals = (i + 1, o["name"], o["class_name"], f5(o.get("lat")), f5(o.get("lon")), o.get("n") or "—", patch,
                    o.get("image_name") or "")
            for k, v in enumerate(vals):
                c = W.QTableWidgetItem(str(v))
                if k not in (1, 2):
                    c.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                self.tbl_obj.setItem(i, k + 1, c)
        self.tbl_obj.blockSignals(False)

    def _obj_changed(self, it):
        i = it.row()
        if not 0 <= i < len(self.objs):
            return
        o = self.objs[i]
        if it.column() == 0:
            o["show"] = it.checkState() == Qt.Checked
        elif it.column() == 2:
            o["name"] = it.text().strip() or o["name"]
        elif it.column() == 3:
            new = it.text().strip() or o["class_name"]
            if new != o["class_name"]:
                o["class_name"] = new
                if o.get("patch"):
                    self.save_patch(o)                                  # the patch file is named after the class
        self._store_objs()
        self.refresh_view()
        self._draw_chart()

    def obj_menu(self, i, gpos):
        if not 0 <= i < len(self.objs):
            return
        o = self.objs[i]
        m = W.QMenu(self)
        m.addAction(f"{o['name']} · class {o['class_name']}").setEnabled(False)
        m.addAction("Spectrum (mean ± std)…", lambda: self.obj_spectrum(i))
        m.addAction("Export patch (GeoTIFF)…", lambda: self.obj_export_patch(i))
        a = m.addAction("Locate patch file", lambda: run_bg(self.api.post, "/api/system/reveal", dict(path=o["patch"]["path"]),
                                                            failed=lambda e: self.flash(e)))
        a.setEnabled(bool(o.get("patch")))
        m.addAction("Save patch again" if o.get("patch") else "Save patch", lambda: self.save_patch(o))
        if o.get("image_id"):
            m.addAction("Locate image file", lambda: run_bg(self.api.post, f"/api/images/{o['image_id']}/reveal", {},
                                                            failed=lambda e: self.flash(e)))
        m.addSeparator()
        m.addAction("Remove", lambda: self.obj_remove(i))
        m.exec_(gpos)

    def obj_spectrum(self, i):
        o = self.objs[i]
        d = W.QDialog(self)
        d.setWindowTitle(f"{o['name']} — ROI spectrum")
        d.resize(780, 520)
        l = W.QVBoxLayout(d)
        ch = SpectrumChart(330, toolbar=True)
        x, y = xy(o["wavelengths"], o["values"])
        band = None
        if o.get("std"):
            _, sd = xy(o["wavelengths"], o["std"])
            band = (y - sd, y + sd)
        ch.plot([dict(name=f"{o['name']} mean", x=x, y=y, color=o["color"], band=band, lw=2)], unit=o.get("unit"))
        l.addWidget(ch, 1)
        ok = np.isfinite(y)
        l.addWidget(small(f"class {o['class_name']} · {o.get('n')} pixels · scene {o.get('image_name')} · "
                          f"{len(x)} bands · mean {np.nanmean(y[ok]) if ok.any() else float('nan'):.4f} · "
                          f"min {np.nanmin(y[ok]) if ok.any() else float('nan'):.4f} · "
                          f"max {np.nanmax(y[ok]) if ok.any() else float('nan'):.4f}" + (" · shaded: ± 1 std" if band else "")))

        def csv():
            rows = [[w, "" if v is None else v] + ([("" if s is None else s)] if o.get("std") else [])
                    for w, v, s in zip(o["wavelengths"], o["values"], o.get("std") or [None] * len(o["values"]))]
            save_csv(d, f"{o['name']}_roi.csv".replace(" ", "_"),
                     ["wavelength_nm", "mean"] + (["std"] if o.get("std") else []), rows)
        l.addLayout(hrow(btn("Export CSV", csv), None, btn("Close", d.accept)))
        d.exec_()

    def obj_export_patch(self, i):
        o = self.objs[i]

        def done(r):
            data, name = r
            save_bytes(self, name or f"{o['class_name']}.tif", data, "GeoTIFF (*.tif)")
        run_bg(self.api.download, f"/api/images/{o['image_id']}/patch/download",
               dict(polygon=o["polygon"], class_name=o["class_name"], bands=self._obj_bands(o)), done=done,
               failed=lambda m: self.flash(m))

    def obj_remove(self, i):
        if 0 <= i < len(self.objs):
            self.objs.pop(i)
            self._store_objs()
            self.refresh_objects()
            self.refresh_view()
            self._draw_chart()

    def obj_clear(self):
        if self.objs and W.QMessageBox.question(self, "Object samples", "Delete all object samples from the list?\n(The patch "
                                                "files stay in their folder.)") == W.QMessageBox.Yes:
            self.objs = []
            self._store_objs()
            self.refresh_objects()
            self.refresh_view()
            self._draw_chart()

    def obj_save(self):
        lst = [o for o in self.objs if o.get("values")]
        if not lst:
            return self.flash("No object samples to save.")
        nm = f"{(self.active or {}).get('name') or 'Image'} objects"
        d = SaveDialog(self, "Save object samples to a library",
                       f"{len(lst)} object samples (ROI mean spectra) will be stored in the database. Each sample keeps the name "
                       "of its patch file (&lt;class name&gt;.tif), the folder where that file was saved, the ROI polygon and "
                       "the source image; the patch file itself stays in its folder.", self.store, nm, "object")
        if not d.exec_():
            return
        v = d.values()
        items = [dict(name=o["name"], wavelengths_nm=o["wavelengths"], values=o["values"], unit=o.get("unit"),
                      category=v["category"], origin="image_roi", acquisition_type=v["acquisition"],
                      sample_metadata=dict(object=dict(kind="roi", class_name=o["class_name"], image_id=o["image_id"],
                                                       image_name=o.get("image_name"), image_path=o.get("image_path"),
                                                       polygon=o["polygon"], n_pixels=o.get("n"), lat=o.get("lat"),
                                                       lon=o.get("lon"),
                                                       patch_file=(o.get("patch") or {}).get("file"),
                                                       patch_path=(o.get("patch") or {}).get("path"))))
                 for o in lst]
        body = dict(library=v["library"], items=items)
        if not v["existing"]:
            body["library_info"] = dict(type="object", description="Object (ROI) library: each entry is the mean spectrum of "
                                        "an ROI. The ROI image patch is a GeoTIFF file in a folder on disk; the database "
                                        "keeps its file name and address.")
        run_bg(self.api.post, "/api/spectra/save", body,
               done=lambda r: (self.store.refresh(), self.flash(f"Saved {r['saved']} object samples to library “{r['library']}”")),
               failed=lambda m: W.QMessageBox.warning(self, "Save", m))

    def aggregate(self):
        rois = [o for o in self.objs if o.get("show", True)]
        if not rois:
            if not self.ck_obj.isChecked():
                self.ck_obj.setChecked(True)
            return W.QMessageBox.information(self, "Data aggregation", "Tick at least one ROI under Collected object samples. "
                                             "Draw ROIs on the image (✏ Draw ROI polygon), finish with a double-click, "
                                             "give each a class and press Add.")
        AggregateDialog(self, rois).exec_()

    # ================================================================== Spectral menu
    def _need_image(self):
        if not self.active or self.base_pixmap is None:
            W.QMessageBox.information(self, "Image", "Open an image first (Open image…).")
            return False
        return True

    def analysis(self, op, expression=None, run=False):
        if self._need_image():
            AnalysisDialog(self, op, expression, run).exec_()

    def analyst(self):
        if not self._need_image():
            return
        t, lab = self._target()
        if not t:
            return self.flash("Click a pixel (or draw an ROI) first.")
        AnalystDialog(self, t, lab).exec_()

    def sam_map(self):
        if self._need_image():
            SamMapDialog(self).exec_()

    def open_library_functions(self, tab):
        sh = self.shell
        try:
            sh.weblib.ensure()
            sh.show_module("weblib")
        except Exception:                                           # noqa: BLE001
            pass
        lp = Hub.library_page
        if lp is not None:
            lp.open_functions(tab)

    def showEvent(self, e):
        super().showEvent(e)
        if not self.images:
            self.refresh()
