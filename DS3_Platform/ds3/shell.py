"""The DS3 main window: ENVI-style menu bar, module sidebar (Library · Data Aggregation · Viewer and Analysis), pages, status bar."""
from __future__ import annotations

import os
import re

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

from . import APP_NAME, APP_TITLE, __version__, style
from .analysis_page import AnalysisPage
from .home import HomePage
from .image_page import BandsListDialog, ImagePage
from .library_page import LibraryPage
from .style import C
from .widgets import Chip, icon_pixmap, label

W = QtWidgets
Qt = QtCore.Qt

# Spectral ▸ Spectral Indices (same list as the DS3 prototype); wNNN = band nearest to NNN nm
INDICES = [
    ("NDVI — vegetation greenness", "(w800 - w670) / (w800 + w670)"),
    ("NDRE — red edge", "(w790 - w720) / (w790 + w720)"),
    ("SAVI — soil-adjusted vegetation", "1.5 * (w800 - w670) / (w800 + w670 + 0.5)"),
    ("EVI — enhanced vegetation", "2.5 * (w800 - w670) / (w800 + 6 * w670 - 7.5 * w475 + 1)"),
    ("NDWI — canopy water", "(w860 - w1240) / (w860 + w1240)"),
    ("NDBI — built-up", "(w1650 - w830) / (w1650 + w830)"),
    ("PRI — photochemical reflectance", "(w531 - w570) / (w531 + w570)"),
]


class HostProxy:
    """What the library panel sees as its 'host': the Analysis window, except that plots go to the plot of the
    module on screen and messages to the DS3 status bar."""

    def __init__(self, shell, win):
        self._shell, self._win = shell, win

    @property
    def plot_canvas(self):
        if self._shell.current_module() == "library":
            return self._shell.library.plot
        return self._win.plot_canvas

    def statusBar(self):
        return self._shell.statusBar()

    def __getattr__(self, name):
        return getattr(self._win, name)


class DS3Window(W.QMainWindow):
    libpanel_libraries_changed = QtCore.pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.setWindowIcon(QtGui.QIcon(self._logo_icon()))
        self.resize(1440, 900)
        self.setMinimumSize(1100, 680)
        # modules: Analysis first (it owns the library panel and its database connection)
        self.analysis = AnalysisPage(self)
        self.libpanel = self.analysis.win.libpanel
        self.libpanel.host = HostProxy(self, self.analysis.win)
        self.library = LibraryPage(self, self.libpanel)
        self.home = HomePage(self)
        self.image = ImagePage(self)
        self._hook_libpanel()

        central = W.QWidget()
        central.setObjectName("ds3Main")
        row = W.QHBoxLayout(central)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(0)
        row.addWidget(self._build_sidebar())
        self.stack = W.QStackedWidget()
        for p in (self.home, self.library, self.image, self.analysis):
            self.stack.addWidget(p)
        row.addWidget(self.stack, 1)
        self.setCentralWidget(central)
        self._build_menus()
        self.msg = W.QLabel("")
        self.statusBar().addWidget(self.msg, 1)
        self.statusBar().setSizeGripEnabled(False)
        self.show_module("home")
        self.refresh_counts()
        self._update_db_status(self.libpanel.client is not None)
        if self.libpanel.client is None and not self.libpanel.cfg.get("autoconnect"):
            self.flash("Library database not connected: File ▸ Connect Library Database… "
                       "(PostgreSQL in Docker, or a local SQLite file).")

    # ------------------------------------------------------------ chrome
    @staticmethod
    def _tile(kind, col, fill, border=None, size=52):
        """Rounded module tile with its icon, drawn at 2x for HiDPI."""
        n = size * 2
        pm = QtGui.QPixmap(n, n)
        pm.fill(Qt.transparent)
        p = QtGui.QPainter(pm)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        p.setBrush(QtGui.QColor(fill))
        p.setPen(QtGui.QPen(QtGui.QColor(border), 3) if border else Qt.NoPen)
        p.drawRoundedRect(QtCore.QRectF(2, 2, n - 4, n - 4), n * 0.26, n * 0.26)
        ip = icon_pixmap(kind, col, size)
        m = int(n * 0.2)
        p.drawPixmap(QtCore.QRect(m, m, n - 2 * m, n - 2 * m), ip)
        p.end()
        pm.setDevicePixelRatio(2)
        return pm

    def _logo_icon(self):
        return self._tile("logo", "white", C["accent"])

    def _build_sidebar(self):
        side = W.QWidget()
        side.setObjectName("ds3Sidebar")
        side.setFixedWidth(236)
        lay = W.QVBoxLayout(side)
        lay.setContentsMargins(10, 12, 10, 10)
        lay.setSpacing(2)
        logo = W.QPushButton("  DS3")
        style.prop(logo, ds3="nav")
        logo.setIcon(QtGui.QIcon(self._logo_icon()))
        logo.setIconSize(QtCore.QSize(30, 30))
        f = logo.font()
        f.setPointSizeF(13)
        f.setBold(True)
        logo.setFont(f)
        logo.setToolTip("Home: the three modules")
        logo.clicked.connect(lambda: self.show_module("home"))
        lay.addWidget(logo)
        lay.addSpacing(10)
        lay.addWidget(label("MAIN MODULES", "seclabel"))
        lay.addSpacing(4)
        self.nav = {}
        grp = self.nav_grp = W.QButtonGroup(self)
        grp.setExclusive(True)
        for key, text in (("library", "Library"), ("analysis", "Data Aggregation"), ("image", "Viewer and Analysis")):
            b = W.QPushButton("  " + text)
            style.prop(b, ds3="nav")
            b.setCheckable(True)
            col, fill, border, _ = style.MODULE[key]
            pm = self._tile(key, col, fill, border)
            b.setIcon(QtGui.QIcon(pm))
            b.setIconSize(QtCore.QSize(26, 26))
            b.clicked.connect(lambda _=False, k=key: self.show_module(k))
            grp.addButton(b)
            self.nav[key] = b
            lay.addWidget(b)
        lay.addSpacing(14)
        self.lbl_recent = label("RECENT IMAGES", "seclabel")
        lay.addWidget(self.lbl_recent)
        self.lst_recent = W.QListWidget()
        self.lst_recent.setStyleSheet("QListWidget { background: transparent; border: none; font-size: 8.5pt; }"
                                      f"QListWidget::item {{ padding: 3px 4px; color: {C['muted2']}; }}"
                                      f"QListWidget::item:hover {{ background: {C['hover']}; }}")
        self.lst_recent.itemClicked.connect(lambda it: self.open_image_path(it.data(Qt.UserRole)))
        self.lst_recent.setMaximumHeight(150)
        lay.addWidget(self.lst_recent)
        self.lbl_recent.setVisible(False)
        self.lst_recent.setVisible(False)
        lay.addStretch(1)
        self.lbl_db = W.QLabel("")
        self.lbl_db.setWordWrap(True)
        self.lbl_db.setCursor(Qt.PointingHandCursor)
        self.lbl_db.mousePressEvent = lambda e: self.libpanel.on_connect_dialog()
        lay.addWidget(self.lbl_db)
        return side

    def current_module(self):
        w = self.stack.currentWidget()
        return {self.home: "home", self.library: "library", self.image: "image", self.analysis: "analysis"}.get(w)

    def show_module(self, key, keep_view=False):
        if key == "image":
            self.stack.setCurrentWidget(self.image)
            QtCore.QTimer.singleShot(0, self.image.ensure)
        else:
            page = {"home": self.home, "library": self.library, "analysis": self.analysis}[key]
            self.stack.setCurrentWidget(page)
            if key == "library" and not keep_view:
                self.library.show_cards()
            if key == "home":
                self.refresh_counts()
        for k, b in self.nav.items():
            b.setChecked(k == key)
        if key == "home":
            self.nav_grp.setExclusive(False)
            for b in self.nav.values():
                b.setChecked(False)
            self.nav_grp.setExclusive(True)

    def flash(self, text, ms=8000):
        self.statusBar().showMessage(str(text).replace("\n", "  "), ms)

    def update_title(self):
        v = self.image.viewer
        p = getattr(v, "_ds3_path", None) if v else None
        self.setWindowTitle(APP_TITLE + (f" — {os.path.basename(p)}" if p else ""))
        rec = self.image.recent
        self.lst_recent.clear()
        for path in rec[:6]:
            it = W.QListWidgetItem(os.path.basename(path))
            it.setToolTip(path)
            it.setData(Qt.UserRole, path)
            self.lst_recent.addItem(it)
        self.lbl_recent.setVisible(bool(rec))
        self.lst_recent.setVisible(bool(rec))

    # ------------------------------------------------------------ library panel hooks
    def _hook_libpanel(self):
        lp = self.libpanel
        orig_conn, orig_fill = lp._set_connected, lp._fill_libraries

        def set_connected(ok, text=None, _o=orig_conn):
            _o(ok, text)
            self._update_db_status(ok, text)

        def fill(*a, _o=orig_fill, **k):
            _o(*a, **k)
            self.library.fill_cards()
            self.refresh_counts()
            self.libpanel_libraries_changed.emit()
            if not getattr(self, "_scanned", False) and lp.client is not None:
                self._scanned = True
                self._ensure_main_folder()
                self.library.scan()
        lp._set_connected = set_connected
        lp._fill_libraries = fill

    def _ensure_main_folder(self):
        """Default libraries folder: data/libraries next to the program (as in the DS3 prototype)."""
        lp = self.libpanel
        if not lp.cfg.get("data_folder"):
            d = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "libraries")
            if os.path.isdir(d):
                lp.cfg["data_folder"] = d
                lp._save_folder_cfg()

    def _update_db_status(self, ok, text=None):
        lp = self.libpanel
        if ok and lp.client is not None:
            desc = lp.client.describe()
            kind = "PostgreSQL" if "postgres" in desc.lower() else "SQLite" if "sqlite" in desc.lower() else "API"
            self.lbl_db.setText(f"<span style='color:{C['green_dot']}'>●</span> Database connected<br>"
                                f"<span style='color:{C['muted']}; font-size:8pt'>{kind}</span>")
            self.lbl_db.setToolTip(desc + "\nClick to change the connection")
            if hasattr(self, "db_info"):
                self.db_info.setText(f"database: {kind}")
        else:
            t = text or "Database not connected"
            self.lbl_db.setText(f"<span style='color:{C['red']}'>●</span> {t if t != 'Not connected' else 'Database not connected'}")
            self.lbl_db.setToolTip("Click to connect the spectral library database")
            if hasattr(self, "db_info"):
                self.db_info.setText("database: not connected")

    def refresh_counts(self):
        libs = self.libpanel.libraries or []
        con = self.libpanel.client is not None
        n_img = self.image.n_opened + len(self.analysis.win.scenes)
        self.home.set_counts(len(libs) if con else None,
                             sum(int(l.get("material_count") or 0) for l in libs) if con else None,
                             n_img,
                             sum(int(l.get("spectrum_count") or 0) for l in libs) if con else None,
                             "" if con else "Library database not connected - counts of libraries, materials and "
                                            "spectra appear once it is (File ▸ Connect Library Database…).")

    # ------------------------------------------------------------ menus
    def _act(self, menu, text, fn, key=None, tip=None, enabled=True):
        a = menu.addAction(text)
        a.triggered.connect(lambda _=False: fn())
        if key:
            a.setShortcut(QtGui.QKeySequence(key))
        if tip:
            a.setStatusTip(tip)
        a.setEnabled(enabled)
        return a

    def _build_menus(self):
        mb = self.menuBar()
        v = lambda: self.image.ensure()                                          # noqa: E731
        an = self.analysis.win
        lp = self.libpanel
        # File
        m = mb.addMenu("File")
        self._act(m, "Open Image File…", lambda: (self.show_module("image"), v().open_envi()), "Ctrl+O")
        self.recent_menu = m.addMenu("Open Previously Opened Image")
        self.recent_menu.aboutToShow.connect(self._fill_recent)
        self._act(m, "Open Zarr Folder…", lambda: (self.show_module("image"), v().open_zarr_folder()))
        self._act(m, "Open Spectral Library File…", lambda: (self.show_module("library"), lp.on_import_files("files")))
        self._act(m, "Import ASD Field Spectra…", lambda: (self.show_module("image"), v().import_asd_dialog()))
        m.addSeparator()
        self._act(m, "Available Bands List…", self.on_bands_list)
        self._act(m, "Save Image View (PNG)…", self.on_save_view)
        self._act(m, "Export Active Cube…", lambda: self._viewer_call("on_prep_export_cube"))
        m.addSeparator()
        self._act(m, "Add Scenes to Data Aggregation…", lambda: (self.show_module("analysis"), an.on_add_scenes()))
        self._act(m, "Add Raw Binary Cube to Data Aggregation…", lambda: (self.show_module("analysis"), an.on_add_raw()))
        self._act(m, "Load Demo Scenes (Data Aggregation)", lambda: (self.show_module("analysis"), an.on_demo()))
        m.addSeparator()
        self._act(m, "Connect Library Database…", lp.on_connect_dialog)
        self._act(m, "Home", lambda: self.show_module("home"), "Ctrl+H")
        m.addSeparator()
        self._act(m, "Exit", self.close, "Ctrl+Q")
        # View
        m = mb.addMenu("View")
        self._act(m, "Home (3 modules)", lambda: self.show_module("home"))
        self._act(m, "Library", lambda: self.show_module("library"), "Ctrl+1")
        self._act(m, "Data Aggregation", lambda: self.show_module("analysis"), "Ctrl+2")
        self._act(m, "Viewer and Analysis", lambda: self.show_module("image"), "Ctrl+3")
        m.addSeparator()
        self._act(m, "Zoom In", lambda: self.on_zoom(1 / 1.25), "Ctrl++")
        self._act(m, "Zoom Out", lambda: self.on_zoom(1.25), "Ctrl+-")
        self._act(m, "Fit Image to Window", lambda: self.on_zoom(None), "Ctrl+0")
        self._act(m, "Map Background On/Off", self.on_map)
        m.addSeparator()
        self._act(m, "Processing History…", lambda: self._viewer_call("on_show_processing_history"))
        self._act(m, "Revert to Original Cube", lambda: self._viewer_call("on_revert_to_original"))
        # Tools
        m = mb.addMenu("Tools")
        self._act(m, "Collect Spectra by Clicking Pixels", self.on_collect)
        self._act(m, "Draw ROIs (Data Aggregation)", self.on_draw_roi)
        self._act(m, "Clear Collected Spectra", lambda: self.image.link and self.image.link.on_clear())
        self._act(m, "ROIs Linked to Data Aggregation (Viewer and Analysis)", lambda: self.image.goto_tab("ROIs"))
        self._act(m, "Get ROIs from Data Aggregation (into Viewer and Analysis)", self.on_rois_to_viewer)
        self._act(m, "Load ROI File into Viewer and Analysis…", lambda: self._viewer_call("on_load_rois"))
        m.addSeparator()
        self._act(m, "Send Image to Data Aggregation", lambda: self._link_call("on_send"))
        m.addSeparator()
        sub = m.addMenu("Preprocessing")
        for t in ("Normalisation", "Units", "Common Grid", "Indices & Features", "Statistics"):
            self._act(sub, t, lambda t=t: self.image.goto_tab("Preprocessing", t))
        self._act(m, "Band Management", lambda: self.image.goto_tab("Band Management"))
        self._act(m, "Data Correction", lambda: self.image.goto_tab("Data Correction"))
        m.addSeparator()
        self._act(m, "Collected Spectra (Data Aggregation)…", lambda: (self.show_module("analysis"), an.on_show_collected()))
        self._act(m, "Library Images…", lambda: an.on_library_images())
        self._act(m, "Plugins", lambda: self.analysis.goto_tab("Plugins"))
        # Spectral
        m = mb.addMenu("Spectral")
        sub = m.addMenu("Spectral Libraries")
        self._act(sub, "Spectral Library Builder", lp.on_builder)
        self._act(sub, "Spectral Library Viewer", self.on_lib_viewer)
        self._act(sub, "Spectral Library Resampling", self.on_lib_resample)
        self._act(sub, "Export Spectral Library", self.on_lib_export)
        self._act(m, "Spectral Analyst…", lambda: self._link_call("on_best_matches"))
        sub = m.addMenu("Mapping Methods")
        self._act(sub, "Spectral Angle Mapper (library)…", lambda: self._link_call("on_classify_library"))
        self._act(sub, "Spectral Angle Mapper (ROI classes)…", lambda: self.image.goto_tab("Spectral Matching"))
        self._act(sub, "Linear Spectral Unmixing…", lambda: self.image.goto_tab("Unmixing"))
        self._act(sub, "K-Means Clustering…", lambda: self.image.goto_tab("Spectral Analysis"))
        sub = m.addMenu("Transforms")
        self._act(sub, "Principal Components (PCA)…", lambda: self.image.goto_tab("Dimensionality Reduction"))
        self._act(sub, "Minimum Noise Fraction (MNF)…", lambda: self.image.goto_tab("Dimensionality Reduction"))
        self._act(sub, "Band Math…", self.on_bandmath)
        sub = m.addMenu("Spectral Indices")
        for t, expr in INDICES:
            self._act(sub, t, lambda t=t, e=expr: self.on_index(t, e))
        sub.addSeparator()
        self._act(sub, "More indices and band features…", lambda: self.image.goto_tab("Preprocessing", "Indices"))
        self._act(m, "Anomaly Detection (RX)…", lambda: self._viewer_call("on_compute_anomaly"))
        self._act(m, "Signal-to-Noise Ratio…", lambda: self._viewer_run("Spectral Analysis", "on_compute_snr"))
        sub = m.addMenu("Classification")
        self._act(sub, "Supervised Classification (RF, SVM, …)…", lambda: self.image.goto_tab("Classification"))
        self._act(sub, "Deep Learning Segmentation…", lambda: self.image.goto_tab("Segmentation"))
        self._act(sub, "Spectral Matching (SAM)…", lambda: self.image.goto_tab("Spectral Matching"))
        m.addSeparator()
        self._act(m, "Data Aggregation…", lambda: self.analysis.goto_tab("Aggregation"))
        # Help
        m = mb.addMenu("Help")
        self._act(m, "Supported Formats", lambda: self._viewer_call("show_formats_dialog", need_cube=False))
        self._act(m, "Using Viewer and Analysis", self.on_help_image)
        self._act(m, "Using Library and Data Aggregation", self.on_help_library)
        m.addSeparator()
        self._act(m, "About DS3", self.on_about)
        self.db_info = W.QLabel("")
        self.db_info.setStyleSheet(f"color: {C['muted']}; padding-right: 10px;")
        mb.setCornerWidget(self.db_info, Qt.TopRightCorner)

    def _fill_recent(self):
        self.recent_menu.clear()
        if not self.image.recent:
            a = self.recent_menu.addAction("(no images opened yet)")
            a.setEnabled(False)
        for p in self.image.recent:
            self.recent_menu.addAction(os.path.basename(p)).triggered.connect(lambda _=False, p=p: self.open_image_path(p))

    def open_image_path(self, path):
        if path and os.path.exists(path):
            self.show_module("image")
            self.image.ensure().load_cube_file(path)
        else:
            self.flash(f"⚠ Not found: {path}")

    # ------------------------------------------------------------ actions on the image module
    def _viewer(self, need_cube=True):
        v = self.image.ensure()
        if need_cube and v.cube is None:
            self.show_module("image")
            W.QMessageBox.information(self, "No image", "Open an image first: File ▸ Open Image File…")
            return None
        return v

    def _viewer_call(self, name, need_cube=True):
        v = self._viewer(need_cube)
        if v is not None:
            self.show_module("image")
            getattr(v, name)()

    def _viewer_run(self, tab, name):
        v = self._viewer()
        if v is not None:
            self.image.goto_tab(tab)
            getattr(v, name)()

    def _link_call(self, name):
        v = self._viewer(need_cube=False)
        self.image.goto_tab("Library")
        if v is not None and self.image.link is not None:
            getattr(self.image.link, name)()

    def on_bands_list(self):
        v = self._viewer()
        if v is not None:
            self.show_module("image")
            BandsListDialog(v, self).exec_()

    def on_save_view(self):
        if self.current_module() == "analysis":
            return self.analysis.win.viewer.toolbar.save_figure()
        v = self._viewer()
        if v is None:
            return
        p, _ = W.QFileDialog.getSaveFileName(self, "Save image view", "image_view.png",
                                             "PNG (*.png);;JPEG (*.jpg);;PDF (*.pdf);;SVG (*.svg)")
        if p:
            v.image_canvas.fig.savefig(p, dpi=200, bbox_inches="tight")
            self.flash(f"Saved {p}")

    def on_zoom(self, factor):
        if self.current_module() == "analysis":
            cv = self.analysis.win.image_canvas
            if factor is None:
                return cv.reset_view()
        else:
            v = self._viewer()
            if v is None:
                return
            cv = v.image_canvas
            if factor is None:
                if cv.ax.images:
                    h, w = cv.ax.images[0].get_array().shape[:2]
                    cv.ax.set_xlim(-0.5, w - 0.5)
                    cv.ax.set_ylim(h - 0.5, -0.5)
                    cv.draw_idle()
                return
        x0, x1 = cv.ax.get_xlim()
        y0, y1 = cv.ax.get_ylim()
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        cv.ax.set_xlim(cx + (x0 - cx) * factor, cx + (x1 - cx) * factor)
        cv.ax.set_ylim(cy + (y0 - cy) * factor, cy + (y1 - cy) * factor)
        cv.draw_idle()

    def on_map(self):
        if self.current_module() == "image" and self.image.viewer is not None and self.image.viewer.cube is not None:
            return self.image.link.on_open_analysis()
        self.show_module("analysis")
        vb = self.analysis.win.viewer
        vb.chk_map.setChecked(not vb.chk_map.isChecked())

    def on_collect(self):
        v = self._viewer()
        if v is not None:
            self.image.goto_tab("Library")
            self.image.link.btn_collect.setChecked(True)

    def on_rois_to_viewer(self):
        v = self._viewer()
        if v is not None:
            self.show_module("image")
            self.image.rois_from_analysis()

    def on_draw_roi(self):
        """ROIs are drawn in Data Aggregation (ROI Analysis tab); the Viewer has no ROI tab in DS3."""
        if self.current_module() == "image" and self.image.viewer is not None and self.image.viewer.cube is not None:
            self.image.link.select_or_send()
        self.analysis.goto_tab("ROI Analysis")
        rp = self.analysis.win.roipanel
        if hasattr(rp, "chk_enable"):
            rp.chk_enable.setChecked(True)
        self.flash("ROI Analysis: tick 'Enable ROI selection', drag a rectangle or click a polygon, then "
                   "'Add selection to ROI list'.")

    def on_bandmath(self):
        v = self._viewer(need_cube=False)
        self.image.goto_tab("Spectral Analysis")
        v.bandmath_edit.setFocus()

    def on_index(self, title, expr):
        v = self._viewer()
        if v is None:
            return
        if v.wavelengths is None:
            return W.QMessageBox.information(self, "No wavelengths", "Spectral indices need band wavelengths "
                                             "(Band Management ▸ Assign wavelengths).")
        wl = np.asarray(v.wavelengths, float)
        lo, hi = float(np.nanmin(wl)), float(np.nanmax(wl))
        missing = []

        def band(m):
            nm = float(m.group(1))
            if nm < lo - 25 or nm > hi + 25:
                missing.append(int(nm))
            return f"B{int(np.nanargmin(np.abs(wl - nm)))}"
        b_expr = re.sub(r"w(\d+(?:\.\d+)?)", band, expr)
        if missing:
            return W.QMessageBox.information(self, title, f"The image ({lo:.0f}–{hi:.0f} nm) has no bands near "
                                             f"{', '.join(map(str, missing))} nm.")
        self.image.goto_tab("Spectral Analysis")
        v.bandmath_edit.setText(b_expr)
        v.on_apply_bandmath()
        v.band_label.setText(f"{title.split(' — ')[0]}: {expr}")
        v.image_canvas.ax.set_title(title, fontsize=9)
        v.image_canvas.draw_idle()
        self.flash(f"{title}: {expr}  →  {b_expr}")

    def on_lib_viewer(self):
        self.show_module("library")
        if self.libpanel.active_lib:
            self.library.open(self.libpanel.active_lib)
            self.libpanel.right_tabs.setCurrentIndex(1) if hasattr(self.libpanel, "right_tabs") else None

    def on_lib_resample(self):
        if self.libpanel.active_lib is None:
            self.show_module("library")
            return self.flash("Open a library first (click its card), then Spectral ▸ Spectral Libraries ▸ Resampling.")
        self.libpanel.on_resample_library()

    def on_lib_export(self):
        if self.libpanel.active_lib is None:
            self.show_module("library")
            return self.flash("Open a library first (click its card), then export it.")
        self.libpanel.on_export_library()

    # ------------------------------------------------------------ help
    def on_help_image(self):
        W.QMessageBox.information(self, "Using Viewer and Analysis", (
            "1. File ▸ Open Image File… (.hdr of an ENVI image, GeoTIFF / EnMAP SPECTRAL_IMAGE, .mat, HDF5, "
            "NetCDF, Zarr, …).\n"
            "2. File ▸ Available Bands List: Gray Scale or RGB, click a band per colour, Load.\n"
            "3. Click a pixel for its spectrum. In the Library tab: pixel window, 'Collect on' to gather "
            "several spectra, Compare angles, Export CSV, Build library.\n"
            "4. Best matches / Classification map compare the image with the spectral library database.\n"
            "5. ROIs: the 'ROIs (Data Aggregation)' tab links this image to its Data Aggregation scene. Click "
            "'Draw ROIs in Data Aggregation', draw them in its ROI Analysis tab, and they are copied here "
            "automatically (Keep ROIs in sync). Classification, Spectral Matching, Segmentation, Unmixing and Data "
            "Correction use them; select ROIs in that tab's ROI Manager for tools that ask for selected ROIs.\n"
            "6. Preprocessing, Band Management, Data Correction, Spectral Analysis, Unmixing, Dimensionality "
            "Reduction (PCA / MNF), Classification, Spectral Matching and Segmentation tabs, or the Spectral menu, "
            "process and analyse the image."))

    def on_help_library(self):
        W.QMessageBox.information(self, "Library and Data Aggregation", (
            "Library: every library as a card. Main libraries are loaded from the libraries folder "
            "(Reload main libraries); built libraries come from the Builder, images or imports. Click a card for "
            "the category tree, materials, spectrum, compare, match and the filters (acquisition, unit, purity, "
            "covers). Spectral ▸ Spectral Libraries: Builder, Viewer, Resampling, Export.\n\n"
            "Data Aggregation: add scenes (or send the image from Viewer and Analysis), draw ROIs per class, Wavelength "
            "Grid ▸ Aggregation ▸ Signature Export (CSV / JSON / ENVI .sli or the library). Mosaic / Stack, the "
            "map background (for georeferenced files), Plugins."))

    def on_about(self):
        W.QMessageBox.about(self, f"About {APP_NAME}", (
            f"<b>{APP_TITLE}</b><br>version {__version__}<br><br>"
            "Spectral library, hyperspectral image module and data aggregation in one program, backed by a "
            "real database (PostgreSQL or SQLite).<br><br>"
            "Viewer and Analysis: DS3 Hyperspectral Viewer & Analysis Tool. Library and Data Aggregation: HyperVista Library v3.4.<br>"
            f"Database: {self.libpanel.client.describe() if self.libpanel.client else 'not connected'}"
            "<br><br><b>Kapih Team</b>"))

    def closeEvent(self, ev):
        try:
            self.libpanel.shutdown()
        except Exception:                                                        # noqa: BLE001
            pass
        super().closeEvent(ev)
