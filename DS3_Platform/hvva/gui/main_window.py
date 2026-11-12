"""Main window of the Library Builder and Aggregation Tool - same layout language as the DS3 viewer:
image + spectra panes on top, tabs underneath, status/info bar at the bottom."""
from __future__ import annotations

import os

import matplotlib
import numpy as np
from matplotlib.patches import Rectangle
from PyQt5 import QtCore, QtGui, QtWidgets

from .. import APP_NAME, __version__
from ..core import display, export, plugins as plugmod
from ..core import quality as Q
from ..core.classify import kmeans_labels
from ..core.geo import GeoInfo
from ..core.mosaic import composite, mosaic, stack_cube
from ..core.resample import common_grid
from ..core.scene import assign_wavelengths, load_scene, save_envi, save_scene
from ..core.signature import signatures_from_result
from ..core.stats import aggregate_classes
from ..core.synthetic import make_demo_scenes
from . import theme
from .canvases import ImageCanvas, PlotCanvas
from .library_panel import LibraryPanel
from .roi_panel import RoiPanel
from .theme import THEMES, restyle_figure, set_matplotlib_theme, apply_app_theme
from .viewer import CollectedSpectraDialog, ViewerBar

W = QtWidgets
PALETTE = [tuple(matplotlib.colormaps["tab10"](i)[:3]) for i in range(10)]
SETTINGS_ORG = "Kapih Team"


def _group(title, *widgets):
    g = W.QGroupBox(title)
    g.setSizePolicy(W.QSizePolicy.Ignored, W.QSizePolicy.Preferred)      # equal-width groups
    lay = W.QVBoxLayout(g)
    for w in widgets:
        lay.addWidget(w) if isinstance(w, W.QWidget) else lay.addLayout(w)
    return g


def _row(*widgets):
    r = W.QHBoxLayout()
    for w in widgets:
        r.addWidget(w)
    return r


def _pair(*groups):
    page = W.QWidget()
    lay = W.QHBoxLayout(page)
    lay.setContentsMargins(6, 6, 6, 6)
    for g in groups:
        lay.addWidget(g, 1)
    return page


def _help(text):
    """Collapsible description (keeps tabs uncluttered, like the DS3 Data Correction tab)."""
    box = W.QToolButton()
    box.setText("ⓘ  About this step")
    box.setCheckable(True)
    box.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
    box.setStyleSheet("QToolButton { border: none; text-align: left; }")
    lab = W.QLabel(text)
    lab.setWordWrap(True)
    lab.setVisible(False)
    box.toggled.connect(lab.setVisible)
    w = W.QWidget()
    l = W.QVBoxLayout(w)
    l.setContentsMargins(0, 0, 0, 0)
    l.addWidget(box)
    l.addWidget(lab)
    return w


class ScrollTabs(W.QTabWidget):
    """Tab widget whose pages scroll when the window is short, so the scene view above keeps its height.
    indexOf / setCurrentWidget / widget accept and return the page itself (not the scroll area)."""

    def addTab(self, page, title):
        sa = W.QScrollArea()
        sa.setWidget(page)
        sa.setWidgetResizable(True)
        sa.setFrameShape(W.QFrame.NoFrame)
        sa.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        sa._page = page
        return super().addTab(sa, title)

    def widget(self, i):
        w = super().widget(i)
        return getattr(w, "_page", w)

    def indexOf(self, page):
        for i in range(self.count()):
            if self.widget(i) is page:
                return i
        return super().indexOf(page)

    def setCurrentWidget(self, page):
        i = self.indexOf(page)
        if i >= 0:
            self.setCurrentIndex(i)

    def currentWidget(self):
        return self.widget(self.currentIndex())


class LibraryBuilderWindow(W.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} {__version__}")
        ico = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "assets", "lbat.ico")
        if os.path.exists(ico):
            self.setWindowIcon(QtGui.QIcon(ico))
        self.resize(1360, 900)
        self.scenes = []
        self.grid = None
        self.result = None
        self.plugins, self.plugin_errors = plugmod.discover()
        self.plugin_dir = plugmod.DEFAULT_DIR
        self.last_spectrum = None                      # (wavelengths, spectrum) of last clicked pixel
        self.last_pixel = None                         # (row, col) of last clicked pixel
        self.collected = []                            # spectra collected by clicking (viewer "Collect spectra")
        self._collected_dlg = None
        self._map_gen = 0                              # drops stale map-tile downloads
        self._map_timer = QtCore.QTimer(self)
        self._map_timer.setSingleShot(True)
        self._map_timer.setInterval(450)
        self._map_timer.timeout.connect(self.refresh_basemap)
        self._gcp_pick = False                         # next click on the view fills a GCP row
        self.theme = str(QtCore.QSettings(SETTINGS_ORG, APP_NAME).value("theme", "dark"))
        if self.theme not in THEMES:
            self.theme = "dark"
        self._build_menu()
        self._build_ui()
        self._apply_window_qss(self.theme)
        self.statusBar().showMessage("Ready")
        self.say("Add scenes (File > Add scenes, or 'Load demo scenes') to begin.")

    # ------------------------------------------------------------------ UI
    def _build_menu(self):
        mb = self.menuBar()
        f = mb.addMenu("&File")
        for text, fn in (("Add scenes...", self.on_add_scenes), ("Add raw binary cube...", self.on_add_raw),
                         ("Load demo scenes", self.on_demo)):
            a = f.addAction(text)
            a.triggered.connect(fn)
        f.addSeparator()
        f.addAction("Connect library database...").triggered.connect(lambda: self.libpanel.on_connect_dialog())
        f.addAction("Import spectral data into library...").triggered.connect(self._menu_import_csv)
        f.addSeparator()
        f.addAction("Exit").triggered.connect(self.close)
        v = mb.addMenu("&View")
        th = v.addMenu("Theme")
        self.theme_actions = {}
        for key in THEMES:
            a = th.addAction(THEMES[key]["label"])
            a.setCheckable(True)
            a.setChecked(key == self.theme)
            a.triggered.connect(lambda _c, k=key: self.on_theme(k))
            self.theme_actions[key] = a
        t = mb.addMenu("&Tools")
        t.addAction("Spectral Library Builder...").triggered.connect(lambda: self.libpanel.on_builder())
        t.addAction("Resample selected library...").triggered.connect(lambda: self.libpanel.on_resample_library())
        t.addAction("Export selected library...").triggered.connect(lambda: self.libpanel.on_export_library())
        t.addSeparator()
        t.addAction("Collected spectra...").triggered.connect(self.on_show_collected)
        t.addAction("Library images...").triggered.connect(self.on_library_images)
        t.addAction("Save scene view as PNG...").triggered.connect(lambda: self.viewer.toolbar.save_figure())
        h = mb.addMenu("&Help")
        h.addAction("About").triggered.connect(self.on_about)

    def _build_ui(self):
        central = W.QWidget()
        self.setCentralWidget(central)
        outer = W.QVBoxLayout(central)
        self.image_canvas = ImageCanvas()
        self.plot_canvas = PlotCanvas()
        self.viewer = ViewerBar(self, self.image_canvas)
        top = W.QSplitter(QtCore.Qt.Horizontal)
        top.addWidget(_group("Scene view  (click a pixel for its spectrum, wheel = zoom)", self.image_canvas,
                             self.viewer))
        top.addWidget(_group("Spectra / plots", self.plot_canvas))
        top.setSizes([680, 640])
        self.tabs = ScrollTabs()
        self.tabs.addTab(self._tab_scenes(), "Scenes")
        self.roipanel = RoiPanel(self)                     # ROI Tools + ROI Manager (moved from the DS3 viewer)
        self.tabs.addTab(self.roipanel, "ROI Analysis")
        self.tabs.addTab(self._tab_grid(), "Wavelength Grid")
        self.tabs.addTab(self._tab_aggregate(), "Aggregation")
        self.tabs.addTab(self._tab_mosaic(), "Mosaic / Stack")
        self.tabs.addTab(self._tab_export(), "Signature Export")
        self.tabs.addTab(self._tab_library(), "Library")
        self.tabs.addTab(self._tab_geo(), "Georeferencing")
        self.tabs.addTab(self._tab_plugins(), "Plugins")
        self.tabs.currentChanged.connect(self.on_tab_changed)
        split = W.QSplitter(QtCore.Qt.Vertical)
        split.addWidget(top)
        split.addWidget(self.tabs)
        split.setSizes([560, 320])
        outer.addWidget(split, 1)
        self.info_label = W.QLabel()
        self.info_label.setWordWrap(True)
        self.info_label.setMinimumHeight(44)
        outer.addWidget(self.info_label)
        self.progress = W.QProgressBar()
        self.progress.setMaximumWidth(220)
        self.progress.setVisible(False)
        self.statusBar().addPermanentWidget(self.progress)
        self.image_canvas.mpl_connect("button_press_event", self.on_image_click)
        self.image_canvas.mpl_connect("motion_notify_event", self.on_image_motion)
        self.image_canvas.after_clear = self._after_canvas_clear
        self.image_canvas.message("No scene loaded")
        self.plot_canvas.message("Nothing to plot yet")

    # --- Scenes tab ---------------------------------------------------
    def _tab_scenes(self):
        self.tbl_scenes = W.QTableWidget(0, 8)
        self.tbl_scenes.setHorizontalHeaderLabels(["Scene", "Rows x Cols", "Bands", "Wavelengths", "Unit", "Bad bands",
                                                   "CRS", "Classes"])
        self.tbl_scenes.horizontalHeader().setStretchLastSection(True)
        self.tbl_scenes.setSelectionBehavior(W.QAbstractItemView.SelectRows)
        self.tbl_scenes.setEditTriggers(W.QAbstractItemView.NoEditTriggers)
        self.tbl_scenes.itemSelectionChanged.connect(self.on_scene_selected)
        b = [W.QPushButton(t) for t in ("Add scenes...", "Load demo scenes", "Remove", "Save .npz", "Save ENVI")]
        for btn, fn in zip(b, (self.on_add_scenes, self.on_demo, self.on_remove, self.on_save_npz, self.on_save_envi)):
            btn.clicked.connect(fn)
        b_img = W.QPushButton("Library images..."); b_img.clicked.connect(self.on_library_images)
        b_img.setToolTip("Images registered in the library database (images table): register this scene, "
                         "open a registered image, remove entries")
        left = _group("Scene list", self.tbl_scenes, _row(*b), _row(b_img))

        self.sp_wl0 = W.QDoubleSpinBox(); self.sp_wl0.setRange(0, 20000); self.sp_wl0.setValue(400)
        self.sp_wl1 = W.QDoubleSpinBox(); self.sp_wl1.setRange(0, 20000); self.sp_wl1.setValue(1000)
        btn_wl = W.QPushButton("Assign wavelengths"); btn_wl.clicked.connect(self.on_assign_wl)
        self.sp_k = W.QSpinBox(); self.sp_k.setRange(2, 20); self.sp_k.setValue(4)
        btn_k = W.QPushButton("Auto-classes (k-means)"); btn_k.clicked.connect(self.on_kmeans)
        btn_lab = W.QPushButton("Import label map (any format)..."); btn_lab.clicked.connect(self.on_import_labels)
        btn_roi = W.QPushButton("Draw ROIs (ROI Analysis tab)...")
        btn_roi.clicked.connect(lambda: self.tabs.setCurrentWidget(self.roipanel))
        btn_clr = W.QPushButton("Clear label map"); btn_clr.clicked.connect(self.on_clear_labels)
        # data quality: unit + bad bands
        self.cb_unit = W.QComboBox()
        for u in Q.UNITS:
            self.cb_unit.addItem({"dn": "DN / counts"}.get(u, u.capitalize()), u)
        self.cb_unit.activated.connect(self.on_set_unit)
        self.lbl_unit = W.QLabel(""); self.lbl_unit.setWordWrap(True)
        self.chk_water = W.QCheckBox("Mask water-absorption bands")
        self.chk_water.setChecked(True)
        self.ed_water = W.QLineEdit(Q.format_windows(Q.WATER_WINDOWS))
        self.ed_water.setToolTip("Wavelength ranges in nm, e.g. 1340-1460, 1790-1960 (add 930-960 or 2400-2500 if needed)")
        self.chk_empty = W.QCheckBox("Mask bands with no data")
        self.chk_empty.setChecked(True)
        self.chk_bbl = W.QCheckBox("Use the file's bad-band list")
        self.chk_bbl.setChecked(True)
        for w in (self.chk_water, self.chk_empty, self.chk_bbl):
            w.toggled.connect(lambda *_: self.on_bad_bands())
        self.ed_water.editingFinished.connect(self.on_bad_bands)
        btn_bad_all = W.QPushButton("Apply bad-band settings to all scenes"); btn_bad_all.clicked.connect(lambda: self.on_bad_bands(True))
        self.lbl_bad = W.QLabel("No scene."); self.lbl_bad.setWordWrap(True)
        form = W.QFormLayout(); form.setFieldGrowthPolicy(W.QFormLayout.AllNonFixedFieldsGrow)
        form.addRow("Wavelength start (nm)", self.sp_wl0); form.addRow("Wavelength stop (nm)", self.sp_wl1)
        form.addRow(btn_wl)
        form.addRow("Value unit", self.cb_unit); form.addRow(self.lbl_unit)
        form.addRow(self.chk_water, self.ed_water)
        form.addRow(self.chk_empty, self.chk_bbl)
        form.addRow(self.lbl_bad); form.addRow(btn_bad_all)
        form.addRow("Classes (k)", self.sp_k); form.addRow(btn_k)
        form.addRow(btn_lab); form.addRow(btn_clr); form.addRow(btn_roi)
        right = _group("Prepare selected scene", _help(
            "Scenes without wavelengths (e.g. .npy or plain GeoTIFF) need them assigned before aggregation. "
            "Value unit: reflectance, radiance or DN - read from the file where possible; set it if it says unknown. "
            "Scenes with different units are never pooled, and a library keeps one unit. "
            "Bad bands (water absorption, the file's bad-band list, bands with no data) are kept in the cube but "
            "left out (gaps) of every spectrum taken from the scene: pixels, ROIs, aggregation and library saves. "
            "Classes come from a label map (.npy, 0 = unlabelled), from a quick k-means, or from rectangle / "
            "polygon ROIs drawn in the ROI Analysis tab (which also saves them class by class to the library "
            "database). Classes are matched across scenes by name."), form)
        return _pair(left, right)

    # --- Grid tab -------------------------------------------------------
    def _tab_grid(self):
        self.cb_grid = W.QComboBox(); self.cb_grid.addItems(["intersection", "union"])
        self.sp_step = W.QDoubleSpinBox(); self.sp_step.setRange(0, 1000); self.sp_step.setDecimals(2)
        self.sp_step.setSpecialValueText("auto (coarsest scene)")
        btn = W.QPushButton("Compute common grid"); btn.clicked.connect(self.on_grid)
        self.lbl_grid = W.QLabel("No grid yet."); self.lbl_grid.setWordWrap(True)
        form = W.QFormLayout(); form.addRow("Range", self.cb_grid); form.addRow("Step (nm)", self.sp_step); form.addRow(btn)
        left = _group("Common wavelength grid", _help(
            "Intersection keeps only the range every scene covers, so nothing is extrapolated. Union keeps the "
            "full range; scenes then contribute NaN outside their own range and statistics are computed per band "
            "from the scenes that have data. Resampling is linear."), form)
        right = _group("Grid summary", self.lbl_grid)
        return _pair(left, right)

    # --- Aggregation tab --------------------------------------------------
    def _tab_aggregate(self):
        self.sp_maxpx = W.QSpinBox(); self.sp_maxpx.setRange(0, 10_000_000); self.sp_maxpx.setSpecialValueText("all pixels")
        btn = W.QPushButton("Run aggregation"); btn.clicked.connect(self.on_aggregate)
        self.chk_scene_curves = W.QCheckBox("Show each scene's mean"); self.chk_scene_curves.setChecked(True)
        self.chk_scene_curves.toggled.connect(self.plot_aggregate)
        form = W.QFormLayout(); form.addRow("Max pixels per scene", self.sp_maxpx); form.addRow(btn); form.addRow(self.chk_scene_curves)
        left = _group("Per-class / per-ROI statistics", _help(
            "Every scene is resampled to the common grid, the pixels of each class (label map and ROIs) are "
            "pooled across scenes, and the mean and standard deviation (n-1) are computed per band. "
            "Scene statistics are merged exactly (parallel variance update), so no pixel is double counted."), form)
        self.tbl_classes = W.QTableWidget(0, 4)
        self.tbl_classes.setHorizontalHeaderLabels(["Plot", "Class", "Pixels", "Scenes"])
        self.tbl_classes.horizontalHeader().setStretchLastSection(True)
        self.tbl_classes.itemChanged.connect(lambda *_: self.plot_aggregate())
        return _pair(left, _group("Classes found", self.tbl_classes))

    # --- Mosaic tab -------------------------------------------------------
    def _tab_mosaic(self):
        self.cb_mode = W.QComboBox(); self.cb_mode.addItems(["Mosaic (side by side)", "Composite (same area)", "Stack cube (same area)"])
        self.cb_blend = W.QComboBox(); self.cb_blend.addItems(["mean", "first", "last", "max"])
        self.cb_comp = W.QComboBox(); self.cb_comp.addItems(["mean", "median", "max", "min"])
        self.ed_mname = W.QLineEdit("mosaic")
        btn = W.QPushButton("Build"); btn.clicked.connect(self.on_build)
        form = W.QFormLayout(); form.addRow("Operation", self.cb_mode); form.addRow("Overlap blend (mosaic)", self.cb_blend)
        form.addRow("Composite method", self.cb_comp); form.addRow("Result name", self.ed_mname); form.addRow(btn)
        left = _group("Mosaic / stack", _help(
            "Uses the scenes selected in the Scenes tab (all scenes if fewer than two are selected). All scenes are "
            "put on the common grid first. Mosaic places georeferenced scenes by map coordinates (same CRS and "
            "pixel size required) and blends overlaps; scenes without georeferencing are laid out left to right. "
            "Composite collapses scenes of the same area to one cube. Stack cube keeps them as a 4-D array "
            "(scenes, rows, cols, bands) and saves it as .npy."), form)
        self.lbl_mosaic = W.QLabel("Nothing built yet."); self.lbl_mosaic.setWordWrap(True)
        return _pair(left, _group("Result", self.lbl_mosaic))

    # --- Export tab -------------------------------------------------------
    def _tab_export(self):
        self.lst_sigs = W.QListWidget()
        self.ed_base = W.QLineEdit(os.path.join(os.path.expanduser("~"), "signatures"))
        btn_b = W.QPushButton("Browse..."); btn_b.clicked.connect(self.on_browse_base)
        self.chk_fmt = {k: W.QCheckBox(t) for k, t in (("csv", "CSV (wide: wavelength, <class>_mean, <class>_std)"),
                                                       ("json", "JSON"), ("sli", "ENVI spectral library (.sli + .hdr; mean and *_std)"))}
        for c in self.chk_fmt.values():
            c.setChecked(True)
        btn_e = W.QPushButton("Export"); btn_e.clicked.connect(self.on_export)
        btn_l = W.QPushButton("Save to library database..."); btn_l.clicked.connect(self.on_add_library)
        left = _group("Classes to export (run Aggregation first)", self.lst_sigs)
        right = _group("Output", _row(self.ed_base, btn_b), *self.chk_fmt.values(), _row(btn_e, btn_l))
        return _pair(left, right)

    # --- Library tab ------------------------------------------------------
    def _tab_library(self):
        """PostgreSQL material spectral library (see gui/library_panel.py and hvva/library/)."""
        self.libpanel = LibraryPanel(self)
        return self.libpanel

    # --- Georeferencing tab -----------------------------------------------
    def _tab_geo(self):
        self.txt_geo = W.QPlainTextEdit(); self.txt_geo.setReadOnly(True)
        b_map = W.QPushButton("Show on map"); b_map.clicked.connect(self.on_geo_show_map)
        b_map.setToolTip("Turn on the map view of the scene view (web map background)")
        b_foot = W.QPushButton("Plot footprints"); b_foot.clicked.connect(self.plot_footprints)
        b_clear = W.QPushButton("Remove georeferencing"); b_clear.clicked.connect(self.on_geo_clear)
        left = _group("Selected scene", self.txt_geo, _row(b_map, b_foot, b_clear))

        # (1) origin + pixel size (north-up)
        self.ed_crs = W.QLineEdit()
        self.ed_crs.setPlaceholderText("EPSG:32633, EPSG:4326, UTM zone 33N …")
        self.sp_x0 = W.QDoubleSpinBox(); self.sp_y0 = W.QDoubleSpinBox()
        self.sp_px = W.QDoubleSpinBox(); self.sp_py = W.QDoubleSpinBox()
        for s in (self.sp_x0, self.sp_y0):
            s.setRange(-1e12, 1e12); s.setDecimals(6)
        for s in (self.sp_px, self.sp_py):
            s.setRange(1e-12, 1e9); s.setDecimals(9); s.setValue(1)
        btn = W.QPushButton("Apply to selected scene"); btn.clicked.connect(self.on_geo_apply)
        form = W.QFormLayout()
        form.addRow("CRS", self.ed_crs); form.addRow("Origin X (upper-left corner)", self.sp_x0)
        form.addRow("Origin Y (upper-left corner)", self.sp_y0); form.addRow("Pixel size X", self.sp_px)
        form.addRow("Pixel size Y", self.sp_py); form.addRow(btn)
        p1 = W.QWidget(); p1.setLayout(form)

        # (2) N / S / E / W bounds in degrees
        self.sp_n, self.sp_s, self.sp_e, self.sp_w = (W.QDoubleSpinBox() for _ in range(4))
        for sp, lim in ((self.sp_n, 85), (self.sp_s, 85), (self.sp_e, 180), (self.sp_w, 180)):
            sp.setRange(-lim, lim); sp.setDecimals(7)
        self.cb_bcrs = W.QComboBox()
        self.cb_bcrs.addItem("Longitude / latitude (EPSG:4326)", "EPSG:4326")
        self.cb_bcrs.addItem("UTM, zone from the centre (WGS84)", "utm")
        b_bounds = W.QPushButton("Apply bounds to selected scene"); b_bounds.clicked.connect(self.on_geo_bounds)
        f2 = W.QFormLayout()
        f2.addRow("North (latitude of the top edge)", self.sp_n); f2.addRow("South (latitude of the bottom edge)", self.sp_s)
        f2.addRow("West (longitude of the left edge)", self.sp_w); f2.addRow("East (longitude of the right edge)", self.sp_e)
        f2.addRow("Store as", self.cb_bcrs); f2.addRow(b_bounds)
        p2 = W.QWidget(); p2.setLayout(f2)

        # (3) ground control points
        self.tbl_gcp = W.QTableWidget(0, 5)
        self.tbl_gcp.setHorizontalHeaderLabels(["Col", "Row", "Map X / lon", "Map Y / lat", "Residual"])
        self.tbl_gcp.horizontalHeader().setStretchLastSection(True)
        self.tbl_gcp.setSelectionBehavior(W.QAbstractItemView.SelectRows)
        self.ed_gcrs = W.QLineEdit("EPSG:4326")
        self.ed_gcrs.setToolTip("CRS of the map coordinates typed in the table (EPSG:4326 = longitude, latitude)")
        b_pick = W.QPushButton("Pick pixel on the view"); b_pick.clicked.connect(self.on_gcp_pick)
        b_pick.setToolTip("Click the scene view: the pixel goes into the selected row (or a new row); then type "
                          "its map coordinates")
        b_add = W.QPushButton("Add row"); b_add.clicked.connect(lambda: self._gcp_row())
        b_del = W.QPushButton("Remove row"); b_del.clicked.connect(self.on_gcp_remove)
        b_fit = W.QPushButton("Fit"); b_fit.clicked.connect(lambda: self.on_gcp_fit(False))
        b_fapp = W.QPushButton("Fit && apply"); b_fapp.clicked.connect(lambda: self.on_gcp_fit(True))
        self.lbl_gcp = W.QLabel("At least 3 points spread over the image (affine fit, RMSE shown).")
        self.lbl_gcp.setWordWrap(True)
        f3 = W.QVBoxLayout()
        f3.addWidget(self.tbl_gcp, 1)
        f3.addLayout(_row(W.QLabel("Coordinates CRS"), self.ed_gcrs))
        f3.addLayout(_row(b_pick, b_add, b_del, b_fit, b_fapp))
        f3.addWidget(self.lbl_gcp)
        p3 = W.QWidget(); p3.setLayout(f3)

        self.geo_tabs = W.QTabWidget()
        self.geo_tabs.addTab(p1, "Origin && pixel size")
        self.geo_tabs.addTab(p2, "N / S / E / W bounds")
        self.geo_tabs.addTab(p3, "Control points")
        right = _group("Set / correct georeferencing", _help(
            "ENVI (map info), GeoTIFF and the native .npz keep CRS and transform with the cube; EnMAP / other "
            "GeoTIFFs are read with their CRS. For files without it: enter the upper-left corner and pixel size; "
            "or the N/S/E/W edges in degrees (from a product's metadata or a map); or ground control points "
            "(pixel <-> map coordinate pairs, least-squares affine with RMSE, rotation allowed). 'Show on map' "
            "places the scene on OpenStreetMap / satellite tiles: ROI outlines, clicks and the coordinate "
            "read-out work there too. UTM and lon/lat work without extra packages; other CRSs need pyproj."),
            self.geo_tabs)
        return _pair(left, right)

    # --- Plugins tab ------------------------------------------------------
    def _tab_plugins(self):
        self.lst_plugins = W.QListWidget()
        self.lst_plugins.currentRowChanged.connect(self.on_plugin_selected)
        b_reload = W.QPushButton("Reload plugin folder"); b_reload.clicked.connect(self.reload_plugins)
        self.lbl_plugdir = W.QLabel(); self.lbl_plugdir.setWordWrap(True)
        left = _group("Available methods (dropped into the plugins folder)", self.lst_plugins, self.lbl_plugdir, b_reload)
        self.lbl_plugdesc = W.QLabel(""); self.lbl_plugdesc.setWordWrap(True)
        self.plug_form = W.QFormLayout()
        pw = W.QWidget(); pw.setLayout(self.plug_form)
        btn = W.QPushButton("Run on selected scene"); btn.clicked.connect(self.on_plugin_run)
        right = _group("Method and parameters", self.lbl_plugdesc, pw, btn)
        self.plug_widgets = {}
        self._fill_plugins()
        return _pair(left, right)

    # ------------------------------------------------------------ helpers
    def say(self, text):
        self.info_label.setText(text)

    def current_scene(self):
        r = self.tbl_scenes.currentRow()
        return self.scenes[r] if 0 <= r < len(self.scenes) else None

    def _selected_or_all(self):
        rows = sorted({i.row() for i in self.tbl_scenes.selectedIndexes()})
        return [self.scenes[r] for r in rows] if len(rows) >= 2 else list(self.scenes)

    def _need_scenes(self, n=1):
        if len(self.scenes) < n:
            self.say(f"Load at least {n} scene{'s' if n > 1 else ''} first.")
            return False
        return True

    def _err(self, e):
        self.say(f"⚠ {e}")
        self.statusBar().showMessage(str(e), 8000)

    def add_scene(self, sc):
        base, k = sc.name, 2
        while any(s.name == sc.name for s in self.scenes):
            sc.name, k = f"{base}_{k}", k + 1
        self.scenes.append(sc)
        self.refresh_scenes()
        self.tbl_scenes.selectRow(len(self.scenes) - 1)

    def refresh_scenes(self):
        keep = self.tbl_scenes.currentRow()
        self.tbl_scenes.blockSignals(True)
        self.tbl_scenes.setRowCount(len(self.scenes))
        for r, sc in enumerate(self.scenes):
            s = sc.summary()
            for c, v in enumerate((s["name"], f"{s['rows']} x {s['cols']}", s["bands"], s["wavelengths"], s["unit"],
                                   s["bad"] or "-", s["crs"], s["classes"])):
                it = W.QTableWidgetItem(str(v))
                if c == 4:
                    it.setToolTip(sc.unit_note or "")
                if c == 5:
                    it.setToolTip(sc.bad_band_summary())
                self.tbl_scenes.setItem(r, c, it)
        self.tbl_scenes.resizeColumnsToContents()
        self.tbl_scenes.blockSignals(False)
        if 0 <= keep < len(self.scenes):
            self.tbl_scenes.selectRow(keep)
        if hasattr(self, "roipanel"):
            self.roipanel.refresh()

    # ------------------------------------------------------------ scenes
    def on_add_scenes(self):
        files, _ = W.QFileDialog.getOpenFileNames(
            self, "Add scenes", "",
            "Hyperspectral data (*.npz *.npy *.hdr *.mat *.h5 *.hdf5 *.he5 *.hdf *.nc *.nc4 *.tif *.tiff *.img *.dat *.raw *.bin);;All files (*)")
        self.add_files(files)

    def on_add_raw(self):
        f, _ = W.QFileDialog.getOpenFileName(self, "Raw binary cube (no header)", "", "All files (*)")
        if f:
            self.add_files([f], force_raw=True)

    def add_files(self, files, force_raw=False):
        """Load files of any supported format; asks what is inside when a file is not self-describing."""
        from ..core.loaders import UnsupportedFormat, open_source
        from ..core.scene import _is_envi, _is_native_npz
        from .open_dialog import OpenOptionsDialog, RawDialog
        self.progress.setVisible(bool(files)); self.progress.setRange(0, len(files))
        remembered = None
        from ..core.quality import resolve_input
        for i, f in enumerate(files):
            try:
                if not force_raw:
                    f, cnote = resolve_input(f)             # .zip product -> image inside; ancillary -> refused
                    if cnote:
                        self.say(cnote)
                if force_raw:
                    d = RawDialog(f, self)
                    if not d.exec_():
                        continue
                    self.add_scene(load_scene(f, **d.choices()))
                elif _is_envi(f) or _is_native_npz(f):
                    self.add_scene(load_scene(f))
                else:
                    try:
                        src = open_source(f)
                    except UnsupportedFormat as e:
                        if "raw" not in str(e):
                            raise
                        d = RawDialog(f, self)
                        if d.exec_():
                            self.add_scene(load_scene(f, **d.choices()))
                        continue
                    if remembered and remembered["cube_key"] in src.arrays:
                        ch = remembered
                    else:
                        d = OpenOptionsDialog(src, self, len(files) - i - 1)
                        if not d.exec_():
                            continue
                        ch = d.choices()
                        remembered = ch if d.chk_all.isChecked() else None
                    self.add_scene(src.build(**ch))
            except Exception as e:                                       # noqa: BLE001
                self._err(f"{os.path.basename(f)}: {e}")
            self.progress.setValue(i + 1)
            QtWidgets.QApplication.processEvents()
        self.progress.setVisible(False)

    def on_demo(self):
        for sc in make_demo_scenes(3):
            self.add_scene(sc)
        self.say("Loaded 3 synthetic overlapping scenes (different band sampling; A, B labelled by map, C by ROI rectangles - see the ROI Analysis tab).")

    def on_remove(self):
        r = self.tbl_scenes.currentRow()
        if r >= 0:
            del self.scenes[r]
            self.refresh_scenes()
            self.show_scene()

    def on_save_npz(self):
        sc = self.current_scene()
        if sc:
            p, _ = W.QFileDialog.getSaveFileName(self, "Save scene", sc.name + ".npz", "Scene (*.npz)")
            if p:
                save_scene(sc, p); self.say(f"Saved {p} (cube, wavelengths, georeferencing, classes).")

    def on_save_envi(self):
        sc = self.current_scene()
        if sc:
            p, _ = W.QFileDialog.getSaveFileName(self, "Save ENVI (base name)", sc.name, "ENVI (*)")
            if p:
                save_envi(sc, os.path.splitext(p)[0] if p.endswith((".img", ".hdr")) else p)
                self.say(f"Saved ENVI image + header with map info next to {p}.")

    def on_library_images(self):
        if not self.libpanel._need_client():
            return
        from .library_dialogs import ImagesDialog
        ImagesDialog(self).exec_()

    def on_scene_selected(self):
        self.roipanel.pending = None
        self.viewer.set_scene(self.current_scene())
        self.show_scene()
        self.roipanel.refresh()
        sc = self.current_scene()
        if sc and sc.has_wavelengths:
            self.sp_wl0.setValue(float(sc.wavelengths.min())); self.sp_wl1.setValue(float(sc.wavelengths.max()))
        self.fill_quality()
        self.fill_geo()
        self.libpanel.update_unit_hint()

    # --- data quality -----------------------------------------------------
    def fill_quality(self):
        sc = self.current_scene()
        widgets = (self.chk_water, self.chk_empty, self.chk_bbl, self.ed_water, self.cb_unit)
        for w in widgets:
            w.blockSignals(True)
        if sc is None:
            self.lbl_bad.setText("No scene.")
            self.lbl_unit.setText("")
        else:
            self.cb_unit.setCurrentIndex(max(0, self.cb_unit.findData(sc.unit)))
            self.lbl_unit.setText(sc.unit_note or ("set the unit" if sc.unit == "unknown" else ""))
            self.chk_water.setChecked(sc.mask_water)
            self.ed_water.setText(Q.format_windows(sc.water_windows))
            self.chk_empty.setChecked(sc.mask_empty)
            self.chk_bbl.setEnabled(sc.bad_file is not None)
            self.chk_bbl.setChecked(sc.bad_file is not None and not getattr(sc, "_ignore_bbl", False))
            self.lbl_bad.setText(sc.bad_band_summary() + ("" if sc.has_wavelengths else
                                                           " (water bands need wavelengths)"))
        for w in widgets:
            w.blockSignals(False)

    def on_set_unit(self):
        sc = self.current_scene()
        if sc is None:
            return
        u = self.cb_unit.currentData()
        if u != sc.unit:
            sc.unit, sc.unit_note = u, "set by hand"
            self.refresh_scenes()
            self.libpanel.update_unit_hint()
            self.fill_quality()
            self.say(f"{sc.name}: values are {u}.")

    def on_bad_bands(self, all_scenes=False):
        scenes = list(self.scenes) if all_scenes else [s for s in [self.current_scene()] if s]
        if not scenes:
            return
        try:
            windows = Q.parse_windows(self.ed_water.text())
        except ValueError as e:
            return self._err(e)
        for sc in scenes:
            sc.mask_water = self.chk_water.isChecked()
            sc.water_windows = windows
            sc.mask_empty = self.chk_empty.isChecked()
            if sc.bad_file is not None and not self.chk_bbl.isChecked():
                sc._file_bbl, sc.bad_file, sc._ignore_bbl = sc.bad_file, None, True
            elif getattr(sc, "_ignore_bbl", False) and self.chk_bbl.isChecked():
                sc.bad_file, sc._ignore_bbl = getattr(sc, "_file_bbl", None), False
        self.refresh_scenes()
        self.fill_quality()
        self.say("Bad bands: " + "; ".join(f"{s.name}: {s.bad_band_summary()}" for s in scenes[:6]))

    def show_scene(self):
        """Draw the current scene: band composite (viewer bar), anomaly map, or on the web map."""
        sc = self.current_scene()
        if sc is None:
            self.image_canvas.message("No scene loaded")
            self.roipanel.refresh()
            return
        key = (id(sc), sc.cube.shape)
        if self.roipanel.show_anomaly:
            try:
                img, cmap, title = self.roipanel.anomaly_image(sc), "inferno", f"{sc.name} - anomaly map"
            except ValueError as e:
                self._err(e)
                img, cmap = display.render(sc)
                title = sc.name
        else:
            img, cmap = display.render(sc)
            title = f"{sc.name}  ·  {display.describe(sc)}"
        if self.viewer.map_on:
            try:
                from ..core.basemap import PROVIDERS, scene_affine
                M = scene_affine(sc)
            except Exception as e:                                               # noqa: BLE001
                self.viewer.chk_map.blockSignals(True)
                self.viewer.chk_map.setChecked(False)
                self.viewer.chk_map.blockSignals(False)
                self._err(f"Map view: {e}")
            else:
                prov = self.viewer.provider
                attr = PROVIDERS[prov]["attribution"] if prov else "no background"
                self.image_canvas.show_on_map(img, M, title=title, cmap=cmap or "gray", alpha=self.viewer.alpha,
                                              view_key=key, attribution=attr)
                self.image_canvas.ax.format_coord = self._format_coord
                self.image_canvas.ax.callbacks.connect("xlim_changed", lambda *_: self._map_timer.start())
                self.refresh_basemap(force=True)
                if self.roipanel.selecting():
                    self.say("ROI drawing is off in map view - switch the map off to draw ROIs.")
                return
        self.image_canvas.show_image(img, title=title, cmap=cmap or "gray", colorbar=self.roipanel.show_anomaly,
                                     view_key=key)
        self.image_canvas.ax.format_coord = self._format_coord

    def _format_coord(self, x, y):
        """Toolbar read-out: row / col, value of the shown band, map and lon/lat coordinates."""
        sc = self.current_scene()
        if sc is None:
            return ""
        class _E:                                                                # a fake event for event_pixel
            inaxes, xdata, ydata = self.image_canvas.ax, x, y
        rc = self.image_canvas.event_pixel(_E)
        if rc is None:
            return ""
        r, c = int(round(rc[0])), int(round(rc[1]))
        if not (0 <= r < sc.shape[0] and 0 <= c < sc.shape[1]):
            return f"outside the image (row {r}, col {c})"
        txt = f"row {r}, col {c}"
        d = display.normalize_display(sc, sc.display)
        txt += f"  ·  b{d['bands'][0] + 1} = {float(sc.cube[r, c, d['bands'][0]]):.4g}"
        if sc.geo is not None:
            gx, gy = sc.geo.xy(r, c)
            txt += f"  ·  x {gx:.2f}, y {gy:.2f}"
            if self.image_canvas.map_mode:
                from ..core.basemap import merc_to_lonlat
                lon, lat = merc_to_lonlat(x, y)
                txt += f"  ·  {float(lat):.5f}°, {float(lon):.5f}°"
        return txt

    def on_image_motion(self, ev):
        """Pixel / value / coordinate read-out in the status bar while the mouse moves over the scene view."""
        if ev.inaxes is not self.image_canvas.ax or ev.xdata is None or self.current_scene() is None:
            return
        try:
            txt = self._format_coord(ev.xdata, ev.ydata)
        except Exception:                                                        # noqa: BLE001
            return
        if txt:
            self.statusBar().showMessage(txt, 4000)

    def refresh_basemap(self, force=False):
        """Download (in the background) the map tiles for the visible part of the map view."""
        cv = self.image_canvas
        if not cv.map_mode:
            return
        prov = self.viewer.provider
        if prov is None:
            if cv._basemap is not None:
                try:
                    cv._basemap.remove()
                except (ValueError, NotImplementedError):
                    pass
                cv._basemap = None
            cv.attribution and cv.attribution.set_text("no background")
            cv.draw_idle()
            return
        from ..core.basemap import PROVIDERS, fetch_basemap
        x0, y0, x1, y1 = cv.visible_extent()
        ext = (x0, y0, x1, y1)
        if not force and getattr(self, "_map_extent", None) == (prov, ext):
            return
        self._map_extent = (prov, ext)
        self._map_gen += 1
        gen, width = self._map_gen, max(300, cv.width())
        self.statusBar().showMessage(f"Loading {prov} map tiles…")

        def ok(res):
            if gen != self._map_gen or not cv.map_mode:
                return
            tiles, extent, z = res
            cv.set_basemap(tiles, extent, PROVIDERS[prov]["attribution"])
            self.statusBar().showMessage(f"{prov} map, zoom level {z}", 4000)

        def bad(msg):
            if gen != self._map_gen:
                return
            self.statusBar().showMessage("Map background unavailable (offline?) - the image is still placed "
                                         "correctly in map coordinates.", 10000)
            self.say(f"⚠ Map tiles: {msg.splitlines()[-1] if msg else 'download failed'}")
        self.libpanel.run(fetch_basemap, ext, prov, width, done=ok, failed=bad)

    def _after_canvas_clear(self):
        self.roipanel.on_canvas_cleared()
        self.draw_collected_markers(draw=False)

    def draw_collected_markers(self, draw=True):
        cv = self.image_canvas
        for a in getattr(self, "_marker_artists", []):
            try:
                a.remove()
            except (ValueError, NotImplementedError, AttributeError):
                pass
        self._marker_artists = []
        sc = self.current_scene()
        if sc is None or not cv.ax.images:
            return
        tr = cv.pix_transform
        for k, it in enumerate(self.collected):
            if it.get("scene_id") != id(sc):
                continue
            r, c = it["pixel"]
            h = it["size"] / 2
            p = Rectangle((c - h, r - h), it["size"], it["size"], fill=False, ec=it["color"], lw=1.6, transform=tr,
                          zorder=6)
            cv.ax.add_patch(p)
            t = cv.ax.text(c + h + 0.5, r - h, str(k + 1), color=it["color"], fontsize=7, fontweight="bold",
                           transform=tr, zorder=6)
            self._marker_artists += [p, t]
        if draw:
            cv.draw_idle()

    def on_show_collected(self):
        if self._collected_dlg is None:
            self._collected_dlg = CollectedSpectraDialog(self)
        self._collected_dlg.refresh()
        self._collected_dlg.show()
        self._collected_dlg.raise_()
        self._collected_dlg.plot()

    def collected_changed(self):
        self.viewer.set_collected_count(len(self.collected))
        if self._collected_dlg is not None:
            self._collected_dlg.refresh()
        self.draw_collected_markers()

    def select_scene(self, sc):
        """Make *sc* the current scene (ROI Manager double-click)."""
        if sc in self.scenes:
            self.tbl_scenes.selectRow(self.scenes.index(sc))

    def on_assign_wl(self):
        sc = self.current_scene()
        if sc:
            assign_wavelengths(sc, self.sp_wl0.value(), self.sp_wl1.value())
            self.refresh_scenes(); self.fill_quality()
            self.say(f"{sc.name}: {sc.n_bands} bands spread over {self.sp_wl0.value():g}-{self.sp_wl1.value():g} nm. "
                     + sc.bad_band_summary().capitalize() + ".")

    def on_kmeans(self):
        sc = self.current_scene()
        if sc:
            k = self.sp_k.value()
            self.setCursor(QtCore.Qt.WaitCursor)
            try:
                sc.labels = kmeans_labels(sc.cube, k)
                sc.class_names = {i: f"class_{i}" for i in range(1, k + 1)}
            finally:
                self.unsetCursor()
            self.refresh_scenes(); self.show_scene(); self.say(f"{sc.name}: {k} k-means classes (class_1 ... class_{k}).")

    def on_import_labels(self):
        """Ground truth often comes as a separate file (e.g. PaviaU_gt.mat): any supported format works."""
        sc = self.current_scene()
        if not sc:
            return
        p, _ = W.QFileDialog.getOpenFileName(self, "Label map (any format)", "", "Label maps (*.npy *.npz *.mat *.h5 *.hdf5 *.tif *.tiff *.nc);;All files (*)")
        if not p:
            return
        try:
            from ..core.loaders import open_source
            src = open_source(p)
            cands = [k for k, (s_, _) in src.arrays.items() if len(s_) == 2 and sorted(s_) == sorted(sc.shape[:2])]
            if not cands and len(src.arrays) == 1:
                cands = list(src.arrays)
            if not cands:
                return self._err(f"no 2-D array of size {sc.shape[:2]} in the file")
            key = cands[0] if len(cands) == 1 else W.QInputDialog.getItem(self, "Label array", "Array:", cands, 0, False)[0]
            lab = np.squeeze(np.asarray(src.reader(key)))
            if lab.shape == sc.shape[:2][::-1] and lab.shape != sc.shape[:2]:
                lab = lab.T
            if lab.shape != sc.shape[:2]:
                return self._err(f"label map is {lab.shape}, scene is {sc.shape[:2]}")
        except Exception as e:                                               # noqa: BLE001
            return self._err(e)
        sc.labels = np.nan_to_num(lab).astype(np.int32)
        sc.class_names = {int(i): f"class_{int(i)}" for i in np.unique(sc.labels) if i}
        self.refresh_scenes(); self.show_scene()
        self.say(f"{sc.name}: {len(sc.class_names)} classes imported (class_1 ...). Classes are matched across scenes by this name.")

    def on_clear_labels(self):
        sc = self.current_scene()
        if sc:
            sc.labels, sc.class_names = None, {}
            self.refresh_scenes(); self.show_scene()
            self.say(f"{sc.name}: label map cleared (ROIs are managed in the ROI Analysis tab).")

    def on_image_click(self, ev):
        sc = self.current_scene()
        cv = self.image_canvas
        if sc is None or cv.navigating() or getattr(ev, "button", 1) not in (1, None):
            return
        if self.roipanel.selecting() and not cv.map_mode:
            return
        rc = cv.event_pixel(ev)
        if rc is None:
            return
        r, c = int(round(rc[0])), int(round(rc[1]))
        if not (0 <= r < sc.shape[0] and 0 <= c < sc.shape[1]):
            return
        if self._gcp_pick:
            return self.gcp_picked(r, c)
        size = self.viewer.window_size
        x = sc.wavelengths if sc.has_wavelengths else np.arange(sc.n_bands)
        spec = sc.spectrum_at(r, c, size)                    # bad bands -> gaps; mean of the window
        self.last_spectrum = (x if sc.has_wavelengths else None, spec)
        self.last_pixel = (r, c)
        self.last_window = size
        win = f" {size}×{size}" if size > 1 else ""
        unit = "" if sc.unit == "unknown" else f" ({sc.unit})"
        if self.viewer.collecting:
            col = PALETTE[len(self.collected) % 10]
            self.collected.append({"label": f"{sc.name} ({r}, {c}){win}", "scene": sc.name, "scene_id": id(sc),
                                   "pixel": (r, c), "size": size,
                                   "wavelengths": np.asarray(sc.wavelengths, float) if sc.has_wavelengths else None,
                                   "values": np.asarray(spec, float), "unit": sc.unit, "purity": sc.purity,
                                   "color": col})
            self.collected_changed()
            ax = self.plot_canvas.reset()
            for it in self.collected[-10:]:
                xx = it["wavelengths"] if it["wavelengths"] is not None else np.arange(len(it["values"]))
                ax.plot(xx, it["values"], color=it["color"], lw=1.2, label=it["label"])
            if sc.has_wavelengths:
                self.plot_canvas.shade_bad(Q.spans(sc.wavelengths, sc.bad_bands))
            self.plot_canvas.finish(f"Collected spectra ({len(self.collected)}; last 10 shown)",
                                    "Wavelength (nm)" if sc.has_wavelengths else "Band", "Value" + unit, True)
            if len(self.collected) >= 2:
                from .viewer import angle_matrix
                a = angle_matrix(self.collected[-2:])[0, 1]
                self.say(f"Collected #{len(self.collected)}: pixel ({r}, {c}){win}" +
                         (f" - angle to the previous one {a:.2f}°" if np.isfinite(a) else "") +
                         ". 'Collected…' compares them all.")
            else:
                self.say(f"Collected #1: pixel ({r}, {c}){win}. Keep clicking; 'Collected…' compares them.")
            return
        ax = self.plot_canvas.reset()
        ax.plot(x, spec, color=PALETTE[0], label=f"pixel ({r}, {c})" + (f" mean of {size}×{size}" if size > 1 else ""))
        if sc.has_wavelengths:
            self.plot_canvas.shade_bad(Q.spans(sc.wavelengths, sc.bad_bands))
        self.plot_canvas.finish(sc.name, "Wavelength (nm)" if sc.has_wavelengths else "Band", "Value" + unit, True)
        self.say(f"{sc.name}: pixel row {r}, col {c}{win}" +
                 (f" -> x {sc.geo.xy(r, c)[0]:.2f}, y {sc.geo.xy(r, c)[1]:.2f}" if sc.geo else ""))

    # ------------------------------------------------------------ grid
    def on_grid(self):
        if not self._need_scenes():
            return
        try:
            wls = [s.wavelengths for s in self.scenes if s.has_wavelengths]
            if len(wls) < len(self.scenes):
                raise ValueError("some scenes have no wavelengths - assign them in the Scenes tab")
            self.grid = common_grid(wls, self.cb_grid.currentText(), self.sp_step.value() or None)
        except Exception as e:                                               # noqa: BLE001
            return self._err(e)
        g = self.grid
        self.lbl_grid.setText(f"{len(g)} bands, {g[0]:.1f} - {g[-1]:.1f} nm, step {np.median(np.diff(g)):.2f} nm "
                              f"({self.cb_grid.currentText()}).")
        self.plot_coverage()
        self.say("Common grid ready. Run the Aggregation tab next.")

    def plot_coverage(self):
        ax = self.plot_canvas.reset()
        for i, s in enumerate(self.scenes):
            if s.has_wavelengths:
                ax.barh(i, s.wavelengths.max() - s.wavelengths.min(), left=s.wavelengths.min(), color=PALETTE[i % 10], alpha=0.8, label=s.name)
        if self.grid is not None:
            ax.axvspan(self.grid[0], self.grid[-1], color="#ffaa00", alpha=0.18, label="common grid")
        ax.set_yticks(range(len(self.scenes)), [s.name for s in self.scenes])
        self.plot_canvas.finish("Spectral coverage per scene", "Wavelength (nm)", "", True)

    # ------------------------------------------------------------ aggregation
    def on_aggregate(self):
        if not self._need_scenes():
            return
        if self.grid is None:
            self.on_grid()
            if self.grid is None:
                return
        self.setCursor(QtCore.Qt.WaitCursor)
        try:
            self.result = aggregate_classes(self.scenes, self.grid, self.sp_maxpx.value() or None)
        except Exception as e:                                               # noqa: BLE001
            self.unsetCursor()
            return self._err(e)
        self.unsetCursor()
        if not self.result.classes:
            return self._err("no classes found - add a label map, ROIs or run k-means in the Scenes tab")
        self.tbl_classes.blockSignals(True)
        self.tbl_classes.setRowCount(len(self.result.classes))
        for r, (n, cs) in enumerate(self.result.classes.items()):
            chk = W.QTableWidgetItem(); chk.setFlags(QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled)
            chk.setCheckState(QtCore.Qt.Checked)
            self.tbl_classes.setItem(r, 0, chk)
            for c, v in enumerate((n, cs.n_pixels, cs.n_scenes), 1):
                self.tbl_classes.setItem(r, c, W.QTableWidgetItem(str(v)))
        self.tbl_classes.blockSignals(False)
        self.lst_sigs.clear()
        for n in self.result.classes:
            it = W.QListWidgetItem(n); it.setFlags(it.flags() | QtCore.Qt.ItemIsUserCheckable); it.setCheckState(QtCore.Qt.Checked)
            self.lst_sigs.addItem(it)
        self.plot_aggregate()
        self.say(f"{len(self.result.classes)} classes pooled over {len(self.scenes)} scenes on {len(self.grid)} bands.")

    def plot_aggregate(self):
        if self.result is None:
            return
        ax = self.plot_canvas.reset()
        names = list(self.result.classes)
        for r, n in enumerate(names):
            it = self.tbl_classes.item(r, 0)
            if it is None or it.checkState() != QtCore.Qt.Checked:
                continue
            cs, col, wl = self.result.classes[n], PALETTE[r % 10], self.result.wavelengths
            ax.plot(wl, cs.mean, color=col, lw=1.8, label=f"{n} (n={cs.n_pixels})")
            ax.fill_between(wl, cs.mean - cs.std, cs.mean + cs.std, color=col, alpha=0.22)
            if self.chk_scene_curves.isChecked():
                for sd in cs.per_scene.values():
                    ax.plot(wl, sd["mean"], color=col, lw=0.7, ls="--", alpha=0.7)
        self.plot_canvas.finish("Class mean ± 1 std (dashed = per-scene means)", "Wavelength (nm)", "Value", True)

    # ------------------------------------------------------------ mosaic / stack
    def on_build(self):
        if not self._need_scenes(2):
            return
        use = self._selected_or_all()
        try:
            wls = [s.wavelengths for s in use]
            if any(w is None for w in wls):
                raise ValueError("assign wavelengths to all scenes first")
            grid = common_grid(wls, self.cb_grid.currentText(), self.sp_step.value() or None)
            mode = self.cb_mode.currentIndex()
            name = self.ed_mname.text().strip() or "result"
            if mode == 0:
                res = mosaic(use, grid, self.cb_blend.currentText(), name)
            elif mode == 1:
                res = composite(use, grid, self.cb_comp.currentText(), name)
            else:
                arr, geo = stack_cube(use, grid)
                p, _ = W.QFileDialog.getSaveFileName(self, "Save stack cube", name + ".npy", "NumPy (*.npy)")
                if p:
                    np.save(p, arr)
                self.lbl_mosaic.setText(f"Stack cube {arr.shape} (scenes, rows, cols, bands)" + (f" saved to {p}." if p else " (not saved)."))
                self.say(self.lbl_mosaic.text())
                return
        except Exception as e:                                               # noqa: BLE001
            return self._err(e)
        self.add_scene(res)
        self.lbl_mosaic.setText(f"{res.name}: {res.shape[0]} x {res.shape[1]} px, {res.n_bands} bands, "
                                f"{'georeferenced (' + str(res.geo.crs) + ')' if res.geo else 'not georeferenced'}, "
                                f"{len(res.class_masks())} classes carried over.")
        self.say(self.lbl_mosaic.text() + " Added to the scene list.")
        self.plot_footprints()

    # ------------------------------------------------------------ export / library
    def _checked_sigs(self):
        if self.result is None:
            self._err("run Aggregation first")
            return []
        chosen = {self.lst_sigs.item(i).text() for i in range(self.lst_sigs.count()) if self.lst_sigs.item(i).checkState() == QtCore.Qt.Checked}
        return signatures_from_result(self.result, chosen)

    def on_browse_base(self):
        p, _ = W.QFileDialog.getSaveFileName(self, "Output base name", self.ed_base.text())
        if p:
            self.ed_base.setText(os.path.splitext(p)[0])

    def on_export(self):
        sigs = self._checked_sigs()
        if not sigs:
            return
        fm = tuple(k for k, c in self.chk_fmt.items() if c.isChecked())
        try:
            paths = export.export_all(self.ed_base.text(), sigs, fm)
        except Exception as e:                                               # noqa: BLE001
            return self._err(e)
        self.say("Wrote:\n" + "\n".join(paths))

    def on_add_library(self):
        """Save the checked aggregated signatures to the PostgreSQL library (Material -> Sample -> Spectrum)."""
        sigs = self._checked_sigs()
        if sigs:
            self.libpanel.save_signatures(sigs)

    def _menu_import_csv(self):
        self.tabs.setCurrentWidget(self.libpanel)
        self.libpanel.on_import_files("files")

    # ------------------------------------------------------------ georeferencing
    def fill_geo(self):
        sc = self.current_scene()
        if sc is None:
            self.txt_geo.setPlainText("")
            return
        if sc.geo:
            from ..core.basemap import can_place, to_lonlat
            txt = sc.geo.describe(*sc.shape[:2]) + (f"\nENVI map info: {sc.geo.map_info}" if sc.geo.map_info else "")
            try:
                h, w = sc.shape[:2]
                xs, ys = zip(*[sc.geo.xy(r, c, center=False) for r, c in ((0, 0), (0, w), (h, w), (h, 0))])
                lon, lat = to_lonlat(np.array(xs), np.array(ys), sc.geo.crs)
                txt += (f"\nCorners (lat, lon): UL {lat[0]:.5f}, {lon[0]:.5f}   UR {lat[1]:.5f}, {lon[1]:.5f}"
                        f"\n                    LR {lat[2]:.5f}, {lon[2]:.5f}   LL {lat[3]:.5f}, {lon[3]:.5f}")
                self.sp_n.setValue(float(np.max(lat))); self.sp_s.setValue(float(np.min(lat)))
                self.sp_w.setValue(float(np.min(lon))); self.sp_e.setValue(float(np.max(lon)))
            except Exception:                                                    # noqa: BLE001
                txt += "\n(no lon/lat conversion for this CRS - install pyproj)"
            txt += "\nMap view: " + ("available" if sc.geo.crs and can_place(sc.geo.crs) else "not available (CRS unknown)")
            self.txt_geo.setPlainText(txt)
            self.ed_crs.setText(sc.geo.crs or "")
            self.sp_x0.setValue(sc.geo.transform[2]); self.sp_y0.setValue(sc.geo.transform[5])
            self.sp_px.setValue(sc.geo.pixel_size[0]); self.sp_py.setValue(sc.geo.pixel_size[1])
        else:
            self.txt_geo.setPlainText("No georeferencing stored with this scene.\nSet it on the right.")

    def _geo_changed(self, sc, how):
        self.refresh_scenes(); self.fill_geo(); self.plot_footprints()
        if self.viewer.map_on:
            self.show_scene()
        self.say(f"{sc.name}: georeferencing set ({how}). 'Show on map' to check it.")

    def on_geo_apply(self):
        sc = self.current_scene()
        if sc:
            sc.geo = GeoInfo(self.ed_crs.text().strip() or None,
                             (self.sp_px.value(), 0.0, self.sp_x0.value(), 0.0, -self.sp_py.value(), self.sp_y0.value()))
            self._geo_changed(sc, "origin and pixel size")

    def on_geo_bounds(self):
        sc = self.current_scene()
        if sc is None:
            return self._err("select a scene")
        n, s_, e, w = self.sp_n.value(), self.sp_s.value(), self.sp_e.value(), self.sp_w.value()
        if n <= s_ or e <= w:
            return self._err("North must be above South and East right of West")
        crs = self.cb_bcrs.currentData()
        if crs == "utm":
            lon_c, lat_c = (e + w) / 2, (n + s_) / 2
            zone = int((lon_c + 180) // 6) + 1
            crs = f"EPSG:{32600 + zone if lat_c >= 0 else 32700 + zone}"
        from ..core.basemap import affine_from_corners
        try:
            t = affine_from_corners((w, n), (e, s_), sc.shape[0], sc.shape[1], crs)
        except ValueError as ex:
            return self._err(ex)
        sc.geo = GeoInfo(crs, t)
        self._geo_changed(sc, f"N/S/E/W bounds, stored in {crs}")

    def on_geo_clear(self):
        sc = self.current_scene()
        if sc and sc.geo is not None:
            sc.geo = None
            if self.viewer.map_on:
                self.viewer.chk_map.setChecked(False)
            self.refresh_scenes(); self.fill_geo(); self.plot_footprints()
            self.say(f"{sc.name}: georeferencing removed.")

    def on_geo_show_map(self):
        sc = self.current_scene()
        if sc is None or sc.geo is None:
            return self._err("the selected scene is not georeferenced - set it on the right first")
        if self.viewer.map_on:
            self.show_scene()
        else:
            self.viewer.chk_map.setChecked(True)

    # --- ground control points
    def _gcp_row(self, col=None, row=None):
        r = self.tbl_gcp.rowCount()
        self.tbl_gcp.insertRow(r)
        for c, v in enumerate((col, row, None, None)):
            self.tbl_gcp.setItem(r, c, W.QTableWidgetItem("" if v is None else f"{v:g}"))
        res = W.QTableWidgetItem(""); res.setFlags(QtCore.Qt.ItemIsEnabled)
        self.tbl_gcp.setItem(r, 4, res)
        self.tbl_gcp.selectRow(r)
        return r

    def on_gcp_pick(self):
        if self.current_scene() is None:
            return self._err("select a scene")
        if self.viewer.map_on:
            self.viewer.chk_map.setChecked(False)
        self._gcp_pick = True
        self.say("Click the control point on the scene view (zoom with the wheel first for precision).")

    def gcp_picked(self, r, c):
        self._gcp_pick = False
        rows = sorted({i.row() for i in self.tbl_gcp.selectedIndexes()})
        k = rows[0] if rows and not (self.tbl_gcp.item(rows[0], 0) or W.QTableWidgetItem()).text() else None
        if k is None:
            k = self._gcp_row(c, r)
        else:
            self.tbl_gcp.item(k, 0).setText(str(c)); self.tbl_gcp.item(k, 1).setText(str(r))
        self.say(f"Control point {k + 1}: pixel col {c}, row {r} - now type its map coordinates in the table.")

    def on_gcp_remove(self):
        for r in sorted({i.row() for i in self.tbl_gcp.selectedIndexes()}, reverse=True):
            self.tbl_gcp.removeRow(r)

    def gcp_points(self):
        pts = []
        for r in range(self.tbl_gcp.rowCount()):
            try:
                vals = [float(self.tbl_gcp.item(r, c).text().replace(",", ".")) for c in range(4)]
            except (AttributeError, ValueError):
                continue
            pts.append(vals)
        return pts

    def on_gcp_fit(self, apply):
        sc = self.current_scene()
        if sc is None:
            return self._err("select a scene")
        from ..core.basemap import affine_from_gcps
        pts = self.gcp_points()
        crs = self.ed_gcrs.text().strip() or None
        try:
            t, rmse, res = affine_from_gcps(pts)
        except ValueError as e:
            return self._err(e)
        k = 0
        for r in range(self.tbl_gcp.rowCount()):
            try:
                [float(self.tbl_gcp.item(r, c).text().replace(",", ".")) for c in range(4)]
            except (AttributeError, ValueError):
                continue
            self.tbl_gcp.item(r, 4).setText(f"{res[k]:.4g}")
            k += 1
        a_, b_, c_, d_, e_, f_ = t
        scale = max(abs(a_), abs(e_), 1e-300)
        if abs(b_) < 1e-7 * scale and abs(d_) < 1e-7 * scale:      # numerically north-up -> exactly north-up
            t = (a_, 0.0, c_, 0.0, e_, f_)
        g = GeoInfo(crs, t)
        px, py = g.pixel_size
        unit = "°" if crs and "4326" in crs else " map units"
        self.lbl_gcp.setText(f"{len(pts)} points: RMSE {rmse:.4g}{unit}; pixel size {px:.4g} x {py:.4g}{unit}, "
                             f"{'north-up' if g.is_north_up else 'rotated / sheared'}." +
                             ("" if len(pts) > 3 else " (3 points fit exactly - add more to see the error.)"))
        if apply:
            sc.geo = g
            self._geo_changed(sc, f"{len(pts)} control points, RMSE {rmse:.3g}{unit}")

    def plot_footprints(self):
        ax = self.plot_canvas.reset()
        any_geo = False
        for i, s in enumerate(self.scenes):
            if s.geo is None:
                continue
            x0, y0, x1, y1 = s.geo.bounds(*s.shape[:2])
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=True, alpha=0.25, color=PALETTE[i % 10]))
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, lw=1.5, ec=PALETTE[i % 10], label=s.name))
            any_geo = True
        if not any_geo:
            return self.plot_canvas.message("No georeferenced scenes")
        ax.autoscale_view(); ax.set_aspect("equal", adjustable="datalim")
        self.plot_canvas.finish("Scene footprints (map units)", "X", "Y", True)

    def on_tab_changed(self, i):
        name = self.tabs.tabText(i)
        if name == "Georeferencing":
            self.fill_geo(); self.plot_footprints()
        elif name == "Wavelength Grid" and self.scenes:
            self.plot_coverage()
        elif name == "Aggregation":
            self.plot_aggregate()
        elif name == "Library":
            self.libpanel.on_shown()

    # ------------------------------------------------------------ plugins
    def _fill_plugins(self):
        self.lst_plugins.clear()
        for p in self.plugins:
            self.lst_plugins.addItem(f"[{p.kind}] {p.name}")
        self.lbl_plugdir.setText(f"Folder: {self.plugin_dir}" + (("\n⚠ " + "; ".join(self.plugin_errors)) if self.plugin_errors else ""))
        if self.plugins:
            self.lst_plugins.setCurrentRow(0)

    def reload_plugins(self):
        self.plugins, self.plugin_errors = plugmod.discover(self.plugin_dir)
        self._fill_plugins()
        self.say(f"{len(self.plugins)} plugin(s) loaded" + (f", {len(self.plugin_errors)} with errors." if self.plugin_errors else "."))

    def on_plugin_selected(self, row):
        while self.plug_form.rowCount():
            self.plug_form.removeRow(0)
        self.plug_widgets = {}
        if not 0 <= row < len(self.plugins):
            return
        p = self.plugins[row]
        self.lbl_plugdesc.setText(f"{p.description}\n(kind: {p.kind}; file: {os.path.basename(p.file)})")
        for k, spec in p.params.items():
            t = spec.get("type", "float")
            if t == "int":
                w = W.QSpinBox(); w.setRange(int(spec.get("min", -10**9)), int(spec.get("max", 10**9))); w.setValue(int(spec["default"]))
            elif t == "bool":
                w = W.QCheckBox(); w.setChecked(bool(spec["default"]))
            elif t == "str":
                w = W.QLineEdit(str(spec["default"]))
            else:
                w = W.QDoubleSpinBox(); w.setDecimals(4); w.setRange(float(spec.get("min", -1e12)), float(spec.get("max", 1e12))); w.setValue(float(spec["default"]))
            self.plug_widgets[k] = w
            self.plug_form.addRow(k, w)

    def on_plugin_run(self):
        sc = self.current_scene()
        row = self.lst_plugins.currentRow()
        if sc is None or row < 0:
            return self._err("select a scene and a method")
        p = self.plugins[row]
        params = {}
        for k, w in self.plug_widgets.items():
            params[k] = w.value() if hasattr(w, "value") else w.isChecked() if hasattr(w, "isChecked") else w.text()
        self.setCursor(QtCore.Qt.WaitCursor)
        try:
            out = p.apply(sc, **params)
        except Exception as e:                                               # noqa: BLE001
            self.unsetCursor()
            return self._err(f"{p.name}: {e}")
        self.unsetCursor()
        if p.kind == "correction":
            self.add_scene(out)
            self.say(f"'{p.name}' applied; result added as scene '{out.name}'.")
        else:
            self.image_canvas.show_image(out["image"], p.name, "viridis", colorbar=True)
            if "plot" in out:
                ax = self.plot_canvas.reset(); ax.plot(*out["plot"]); self.plot_canvas.finish(p.name)
            tbl = out.get("table") or {}
            self.say(f"'{p.name}' on {sc.name}: " + ", ".join(f"{k} = {v:.4g}" if isinstance(v, float) else f"{k} = {v}" for k, v in tbl.items()))

    # ------------------------------------------------------------ theme / about
    def _apply_window_qss(self, name):
        t = THEMES[name]
        self.setStyleSheet(
            "QGroupBox { font-weight: bold; margin-top: 8px; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px 0 4px; }"
            "QPushButton, QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox { min-height: 24px; }"
            f"QTabWidget::pane {{ border: 1px solid {t['pane_border']}; }}"
            "QTabBar::tab { min-width: 110px; min-height: 24px; padding: 3px 8px; }")
        self.info_label.setStyleSheet("QLabel {" + t["info_style"] + " border-radius:6px; padding:10px;}")

    def on_theme(self, name):
        if name == self.theme:
            return
        self.theme = name
        set_matplotlib_theme(name)
        apply_app_theme(QtWidgets.QApplication.instance(), name)
        self._apply_window_qss(name)
        for c in (self.image_canvas, self.plot_canvas):
            restyle_figure(c.fig); c.draw()
        for k, a in self.theme_actions.items():
            a.setChecked(k == name)
        QtCore.QSettings(SETTINGS_ORG, APP_NAME).setValue("theme", name)

    def closeEvent(self, ev):
        self.libpanel.shutdown()
        super().closeEvent(ev)

    def on_about(self):
        W.QMessageBox.about(self, f"About {APP_NAME}",
                            f"<b>{APP_NAME} {__version__}</b><br>Prototype of the HyperVista_va add-on: multi-scene aggregation, "
                            "signature export, PostgreSQL material spectral library, georeferencing and plugins.<br><br><b>Kapih Team</b><br>Open source (MIT).")


def main():
    import sys
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setOrganizationName(SETTINGS_ORG)
    theme._ORIG_STYLE = app.style().objectName()
    th = str(QtCore.QSettings(SETTINGS_ORG, APP_NAME).value("theme", "dark"))
    th = th if th in THEMES else "dark"
    set_matplotlib_theme(th)
    apply_app_theme(app, th)
    win = LibraryBuilderWindow()
    win.show()
    sys.exit(app.exec_())
