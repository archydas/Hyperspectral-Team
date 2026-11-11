"""Main window of the Library Builder and Aggregation Tool - same layout language as the DS3 viewer:
image + spectra panes on top, tabs underneath, status/info bar at the bottom."""
from __future__ import annotations

import os

import matplotlib
import numpy as np
from matplotlib.patches import Rectangle
from PyQt5 import QtCore, QtGui, QtWidgets

from .. import APP_NAME, __version__
from ..core import export, plugins as plugmod
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
        h = mb.addMenu("&Help")
        h.addAction("About").triggered.connect(self.on_about)

    def _build_ui(self):
        central = W.QWidget()
        self.setCentralWidget(central)
        outer = W.QVBoxLayout(central)
        self.image_canvas = ImageCanvas()
        self.plot_canvas = PlotCanvas()
        top = W.QSplitter(QtCore.Qt.Horizontal)
        top.addWidget(_group("Scene view  (click a pixel for its spectrum)", self.image_canvas))
        top.addWidget(_group("Spectra / plots", self.plot_canvas))
        top.setSizes([600, 700])
        self.tabs = W.QTabWidget()
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
        split.setSizes([470, 330])
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
        self.image_canvas.after_clear = self.roipanel.on_canvas_cleared
        self.image_canvas.message("No scene loaded")
        self.plot_canvas.message("Nothing to plot yet")

    # --- Scenes tab ---------------------------------------------------
    def _tab_scenes(self):
        self.tbl_scenes = W.QTableWidget(0, 6)
        self.tbl_scenes.setHorizontalHeaderLabels(["Scene", "Rows x Cols", "Bands", "Wavelengths", "CRS", "Classes"])
        self.tbl_scenes.horizontalHeader().setStretchLastSection(True)
        self.tbl_scenes.setSelectionBehavior(W.QAbstractItemView.SelectRows)
        self.tbl_scenes.setEditTriggers(W.QAbstractItemView.NoEditTriggers)
        self.tbl_scenes.itemSelectionChanged.connect(self.on_scene_selected)
        b = [W.QPushButton(t) for t in ("Add scenes...", "Load demo scenes", "Remove", "Save .npz", "Save ENVI")]
        for btn, fn in zip(b, (self.on_add_scenes, self.on_demo, self.on_remove, self.on_save_npz, self.on_save_envi)):
            btn.clicked.connect(fn)
        left = _group("Scene list", self.tbl_scenes, _row(*b))

        self.sp_wl0 = W.QDoubleSpinBox(); self.sp_wl0.setRange(0, 20000); self.sp_wl0.setValue(400)
        self.sp_wl1 = W.QDoubleSpinBox(); self.sp_wl1.setRange(0, 20000); self.sp_wl1.setValue(1000)
        btn_wl = W.QPushButton("Assign wavelengths"); btn_wl.clicked.connect(self.on_assign_wl)
        self.sp_k = W.QSpinBox(); self.sp_k.setRange(2, 20); self.sp_k.setValue(4)
        btn_k = W.QPushButton("Auto-classes (k-means)"); btn_k.clicked.connect(self.on_kmeans)
        btn_lab = W.QPushButton("Import label map (any format)..."); btn_lab.clicked.connect(self.on_import_labels)
        btn_roi = W.QPushButton("Draw ROIs (ROI Analysis tab)...")
        btn_roi.clicked.connect(lambda: self.tabs.setCurrentWidget(self.roipanel))
        btn_clr = W.QPushButton("Clear label map"); btn_clr.clicked.connect(self.on_clear_labels)
        form = W.QFormLayout(); form.setFieldGrowthPolicy(W.QFormLayout.AllNonFixedFieldsGrow)
        form.addRow("Wavelength start (nm)", self.sp_wl0); form.addRow("Wavelength stop (nm)", self.sp_wl1)
        form.addRow(btn_wl)
        form.addRow("Classes (k)", self.sp_k); form.addRow(btn_k)
        form.addRow(btn_lab); form.addRow(btn_clr); form.addRow(btn_roi)
        right = _group("Prepare selected scene", _help(
            "Scenes without wavelengths (e.g. .npy or plain GeoTIFF) need them assigned before aggregation. "
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
        self.ed_crs = W.QLineEdit()
        self.sp_x0 = W.QDoubleSpinBox(); self.sp_y0 = W.QDoubleSpinBox()
        self.sp_px = W.QDoubleSpinBox(); self.sp_py = W.QDoubleSpinBox()
        for s in (self.sp_x0, self.sp_y0):
            s.setRange(-1e12, 1e12); s.setDecimals(4)
        for s in (self.sp_px, self.sp_py):
            s.setRange(1e-9, 1e9); s.setDecimals(6); s.setValue(1)
        btn = W.QPushButton("Apply to selected scene"); btn.clicked.connect(self.on_geo_apply)
        form = W.QFormLayout()
        form.addRow("CRS (e.g. EPSG:32633)", self.ed_crs); form.addRow("Origin X (upper-left)", self.sp_x0)
        form.addRow("Origin Y (upper-left)", self.sp_y0); form.addRow("Pixel size X", self.sp_px)
        form.addRow("Pixel size Y", self.sp_py); form.addRow(btn)
        left = _group("Selected scene", self.txt_geo)
        right = _group("Set / correct georeferencing", _help(
            "ENVI (map info), GeoTIFF (with rasterio) and the native .npz keep CRS and transform with the cube. "
            "For files without it, enter the upper-left corner and pixel size here; the footprint plot shows "
            "how all scenes lie relative to each other, which is what mosaicking uses."), form)
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
            for c, v in enumerate((s["name"], f"{s['rows']} x {s['cols']}", s["bands"], s["wavelengths"], s["crs"], s["classes"])):
                self.tbl_scenes.setItem(r, c, W.QTableWidgetItem(str(v)))
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
        for i, f in enumerate(files):
            try:
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

    def on_scene_selected(self):
        self.roipanel.pending = None
        self.show_scene()
        self.roipanel.refresh()
        sc = self.current_scene()
        if sc and sc.has_wavelengths:
            self.sp_wl0.setValue(float(sc.wavelengths.min())); self.sp_wl1.setValue(float(sc.wavelengths.max()))
        self.fill_geo()

    def show_scene(self):
        sc = self.current_scene()
        if sc is None:
            self.image_canvas.message("No scene loaded")
            self.roipanel.refresh()
            return
        if self.roipanel.show_anomaly:
            try:
                img = self.roipanel.anomaly_image(sc)
                self.image_canvas.show_image(img, title=f"{sc.name} - anomaly map", cmap="inferno", colorbar=True)
                return
            except ValueError as e:
                self._err(e)
        self.image_canvas.show_image(sc.rgb(), title=sc.name)

    def select_scene(self, sc):
        """Make *sc* the current scene (ROI Manager double-click)."""
        if sc in self.scenes:
            self.tbl_scenes.selectRow(self.scenes.index(sc))

    def on_assign_wl(self):
        sc = self.current_scene()
        if sc:
            assign_wavelengths(sc, self.sp_wl0.value(), self.sp_wl1.value())
            self.refresh_scenes(); self.say(f"{sc.name}: {sc.n_bands} bands spread over {self.sp_wl0.value():g}-{self.sp_wl1.value():g} nm.")

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
        if sc is None or ev.inaxes is not self.image_canvas.ax or ev.xdata is None or self.roipanel.selecting():
            return
        r, c = int(round(ev.ydata)), int(round(ev.xdata))
        if not (0 <= r < sc.shape[0] and 0 <= c < sc.shape[1]):
            return
        x = sc.wavelengths if sc.has_wavelengths else np.arange(sc.n_bands)
        spec = sc.cube[r, c, :]
        self.last_spectrum = (x if sc.has_wavelengths else None, spec)
        self.last_pixel = (r, c)
        ax = self.plot_canvas.reset()
        ax.plot(x, spec, color=PALETTE[0], label=f"pixel ({r}, {c})")
        self.plot_canvas.finish(sc.name, "Wavelength (nm)" if sc.has_wavelengths else "Band", "Value", True)
        self.say(f"{sc.name}: pixel row {r}, col {c}" + (f" -> x {sc.geo.xy(r, c)[0]:.2f}, y {sc.geo.xy(r, c)[1]:.2f}" if sc.geo else ""))

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
            self.txt_geo.setPlainText(sc.geo.describe(*sc.shape[:2]) + (f"\nENVI map info: {sc.geo.map_info}" if sc.geo.map_info else ""))
            self.ed_crs.setText(sc.geo.crs or "")
            self.sp_x0.setValue(sc.geo.transform[2]); self.sp_y0.setValue(sc.geo.transform[5])
            self.sp_px.setValue(sc.geo.pixel_size[0]); self.sp_py.setValue(sc.geo.pixel_size[1])
        else:
            self.txt_geo.setPlainText("No georeferencing stored with this scene.\nSet it on the right.")

    def on_geo_apply(self):
        sc = self.current_scene()
        if sc:
            sc.geo = GeoInfo(self.ed_crs.text().strip() or None,
                             (self.sp_px.value(), 0.0, self.sp_x0.value(), 0.0, -self.sp_py.value(), self.sp_y0.value()))
            self.refresh_scenes(); self.fill_geo(); self.plot_footprints()

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
