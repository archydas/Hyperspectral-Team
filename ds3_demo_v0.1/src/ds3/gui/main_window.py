"""DS3 Library main window.

Layout follows the DS3 viewer: results table + spectrum plot on top, work tabs below.
All real work is done by the Qt-free `ds3` functions (ingest, library, search); this file only collects inputs,
runs them off the UI thread, and shows results. Every action has a method that takes plain arguments
(import_path, do_search, run_build, ...), which is what the automated tests call.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QSplitter, QTabWidget, QTableWidget,
    QTableWidgetItem, QAbstractItemView, QGroupBox, QLabel, QLineEdit, QPushButton, QComboBox, QSpinBox,
    QDoubleSpinBox, QCheckBox, QPlainTextEdit, QListWidget, QFileDialog, QMessageBox, QInputDialog, QProgressBar,
    QRadioButton, QButtonGroup, QHeaderView, QApplication)

from ..db import session_scope, init_db, reset_db
from ..ingest import IngestError
from ..ingest.readers.wavelength_files import read_wavelengths_file
from ..library import cube_roi
from ..library.access import (list_libraries, list_categories, list_cubes, db_stats, get_spectra, delete_library)
from ..library.builder import build_library
from ..library.export import export_envi_sli, export_csv
from ..library.grid import Grid
from ..library.search import find_spectra, match_spectrum
from ..pipeline import ingest_path, ingest_library_folder
from ..spectra import read_spectra
from . import settings
from .plot import SpectrumCanvas
from .image_panel import ImagePanel
from .workers import Worker

VERSION = "0.3.0"
KIND_ITEMS = ["auto (from folder name)", "satellite", "airborne", "drone", "field", "library"]
UNITS = ["reflectance", "radiance", "dn"]
ANY = "(any)"


def _item(text, right=False):
    it = QTableWidgetItem("" if text is None else str(text))
    if right:
        it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
    return it


class MainWindow(QMainWindow):
    PAGE = 100
    MAX_PLOT = 12

    def __init__(self, db_url=None, sync=False, autoload=None):
        super().__init__()
        self.sync = sync                       # True = run jobs inline (automated tests, headless use)
        self.db_url = settings.resolve_db_url(db_url)
        self.rows, self.total, self.offset, self._filters = [], 0, 0, {}
        self._workers, self._jobs = set(), 0
        self.last_error = None
        self._measured = None                  # (spectra, cube_id) from the last ROI preview
        self._match_rows, self._match_query = [], None
        self._build_ui()
        self._set_title()
        try:
            init_db(self.db_url)
        except Exception as e:
            self.show_error(f"Cannot open database {settings.pretty(self.db_url)}:\n{e}")
        self.refresh_all()
        # libraries folder: loaded at every start so the libraries are simply there when the app opens
        if (not sync if autoload is None else autoload) and settings.autoload_enabled():
            self.load_library_folder()

    # ------------------------------------------------------------------ infrastructure
    def _set_title(self):
        name = self.db_url.rsplit("/", 1)[-1] if self.db_url.startswith("sqlite") else "PostgreSQL"
        self.setWindowTitle(f"DS3 Library — {name} | Created by Kapih Team")

    def run(self, fn, on_done, label="Working…"):
        """Run fn() off the UI thread (inline when sync) and hand its result to on_done on the UI thread."""
        self.status.showMessage(label)
        if self.sync:
            try:
                res = fn()
            except Exception as e:
                self.show_error(f"{type(e).__name__}: {e}")
                return
            self.status.showMessage("Done", 3000)
            on_done(res)
            return
        self._jobs += 1
        self.busy.show()
        w = Worker(fn)
        self._workers.add(w)

        def finish():
            self._workers.discard(w)
            self._jobs -= 1
            if self._jobs <= 0:
                self.busy.hide()

        def ok(res):
            finish(); self.status.showMessage("Done", 3000); on_done(res)

        def bad(msg):
            finish(); self.show_error(msg)
        w.done.connect(ok); w.failed.connect(bad)
        w.start()

    def show_error(self, msg):
        self.last_error = msg
        self.status.showMessage(msg.splitlines()[0][:160], 8000)
        if not self.sync:
            QMessageBox.warning(self, "DS3 Library", msg)

    def _confirm(self, text):
        return True if self.sync else QMessageBox.question(self, "DS3 Library", text) == QMessageBox.Yes

    # ------------------------------------------------------------------ UI construction
    def _build_ui(self):
        self.resize(1320, 820)
        self.status = self.statusBar()
        self.busy = QProgressBar(); self.busy.setRange(0, 0); self.busy.setMaximumWidth(140); self.busy.hide()
        self.db_label = QLabel(); self.status.addPermanentWidget(self.db_label); self.status.addPermanentWidget(self.busy)
        mb = self.menuBar()
        fm = mb.addMenu("File")
        fm.addAction("Open image…", lambda: (self.modes.setCurrentIndex(1), self.image_panel._open_dialog()))
        fm.addAction("Bring library file into image…", lambda: (self.modes.setCurrentIndex(1), self.image_panel._library_dialog()))
        fm.addSeparator()
        fm.addAction("Import file…", lambda: (self.tabs.setCurrentIndex(0), self._browse_file(self.i_path)))
        fm.addAction("Import folder…", lambda: (self.tabs.setCurrentIndex(0), self._browse_dir(self.i_path)))
        fm.addSeparator(); fm.addAction("Exit", self.close)
        vm = mb.addMenu("View"); vm.addAction("Refresh", self.refresh_all)
        hm = mb.addMenu("Help"); hm.addAction("About", self._about)

        central = QWidget(); self.setCentralWidget(central)
        root = QVBoxLayout(central)
        top = QSplitter(Qt.Horizontal)
        # left: results table
        gb = QGroupBox("Library"); lay = QVBoxLayout(gb)
        self.table = QTableWidget(0, 8)
        self.table.setHorizontalHeaderLabels(["ID", "Material", "Category", "Library", "Unit", "Purity", "Bands", "Range (nm)"])
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.verticalHeader().hide()
        hh = self.table.horizontalHeader(); hh.setSectionResizeMode(1, QHeaderView.Stretch)
        for i in (0, 2, 3, 4, 5, 6, 7): hh.setSectionResizeMode(i, QHeaderView.ResizeToContents)
        hh.setMinimumSectionSize(40)
        self.table.itemSelectionChanged.connect(self.on_selection)
        lay.addWidget(self.table)
        pg = QHBoxLayout()
        self.b_prev = QPushButton("◀ Prev"); self.b_next = QPushButton("Next ▶"); self.pager = QLabel("")
        self.b_prev.clicked.connect(lambda: self.do_search(max(0, self.offset - self.PAGE)))
        self.b_next.clicked.connect(lambda: self.do_search(self.offset + self.PAGE))
        pg.addWidget(self.b_prev); pg.addWidget(self.b_next); pg.addWidget(self.pager, 1)
        lay.addLayout(pg)
        top.addWidget(gb)
        # right: plot + info
        right = QWidget(); rl = QVBoxLayout(right); rl.setContentsMargins(0, 0, 0, 0)
        self.canvas = SpectrumCanvas(); rl.addWidget(self.canvas, 1)
        self.info = QLabel("No spectrum selected"); self.info.setWordWrap(True)
        self.info.setStyleSheet("QLabel{border:1px solid #bbb;border-radius:4px;padding:6px;background:#f7f7f7}")
        rl.addWidget(self.info)
        top.addWidget(right); top.setStretchFactor(0, 3); top.setStretchFactor(1, 2); top.setSizes([800, 520])

        self.tabs = QTabWidget()
        for title, builder in (("Import", self._tab_import), ("Search", self._tab_search),
                               ("Cubes → Library", self._tab_cubes), ("Library Builder", self._tab_builder),
                               ("Match", self._tab_match), ("Export", self._tab_export), ("Database", self._tab_db)):
            self.tabs.addTab(builder(), title)
        split = QSplitter(Qt.Vertical); split.addWidget(top); split.addWidget(self.tabs)
        split.setStretchFactor(0, 3); split.setStretchFactor(1, 2)
        self.cubes = []
        self.image_panel = ImagePanel(self)
        self.modes = QTabWidget(); self.modes.setDocumentMode(True)
        self.modes.addTab(split, "Library manager"); self.modes.addTab(self.image_panel, "Image viewer")
        root.addWidget(self.modes)

    def _form(self, rows, widget=None):
        w = widget or QWidget(); g = QGridLayout(w)
        for r, (label, ctrl) in enumerate(rows):
            if label:
                g.addWidget(QLabel(label), r, 0)
            if isinstance(ctrl, (list, tuple)):
                h = QHBoxLayout()
                for c in ctrl: h.addWidget(c)
                g.addLayout(h, r, 1)
            else:
                g.addWidget(ctrl, r, 1)
        g.setColumnStretch(1, 1)
        return w

    @staticmethod
    def _btn(text, slot):
        b = QPushButton(text); b.clicked.connect(lambda _=False: slot()); return b

    def _combo(self, items, editable=False):
        c = QComboBox(); c.addItems(items); c.setEditable(editable); return c

    # ---- Import
    def _tab_import(self):
        w = QWidget(); outer = QHBoxLayout(w)
        left = QWidget()
        self.i_path = QLineEdit(); self.i_path.setPlaceholderText("A file (.hdr, .tif, .h5, .nc, .csv, .sig, .sli, …) or a folder")
        self.i_kind = self._combo(KIND_ITEMS); self.i_source = QLineEdit(); self.i_source.setPlaceholderText("optional, e.g. AVIRIS-NG India 2018")
        self.i_wl = QLineEdit(); self.i_wl.setPlaceholderText("optional: EnMAP METADATA.XML, a .txt/.csv list of wavelengths, or a range like 430-860")
        self.i_unit = self._combo(["auto"] + UNITS)
        self.i_div = QLineEdit(); self.i_div.setPlaceholderText("auto  (e.g. 10000 for int16 reflectance ×10000)")
        self._form([
            ("Path", [self.i_path, self._btn("File…", lambda: self._browse_file(self.i_path)),
                      self._btn("Folder…", lambda: self._browse_dir(self.i_path))]),
            ("Source kind", self.i_kind), ("Source name", self.i_source),
            ("Wavelengths file", [self.i_wl, self._btn("Browse…", lambda: self._browse_file(self.i_wl))]),
            ("Unit", self.i_unit), ("Reflectance divisor", self.i_div),
            ("", self._btn("Import into database", self.on_import_clicked))], left)
        self.i_log = QPlainTextEdit(); self.i_log.setReadOnly(True)
        self.i_log.setPlaceholderText("Import log: one line per file (ingested / duplicate / skipped / error)")
        outer.addWidget(left, 3); outer.addWidget(self.i_log, 2)
        return w

    # ---- Search
    def _tab_search(self):
        self.s_text = QLineEdit(); self.s_text.setPlaceholderText("material, category, library or label contains…")
        self.s_cat = self._combo([ANY]); self.s_unit = self._combo([ANY] + UNITS)
        self.s_purity = self._combo([ANY, "ground_measured", "image_pixel", "synthetic"])
        self.s_lib = self._combo([ANY])
        self.s_lo = QDoubleSpinBox(); self.s_hi = QDoubleSpinBox()
        for sp in (self.s_lo, self.s_hi):
            sp.setRange(0, 20000); sp.setDecimals(0); sp.setSpecialValueText("off"); sp.setSuffix(" nm")
        self.s_text.returnPressed.connect(lambda: self.do_search(0))
        w = self._form([("Text", self.s_text), ("Category", self.s_cat), ("Unit", self.s_unit), ("Purity", self.s_purity),
                        ("Library", self.s_lib), ("Must cover", [self.s_lo, QLabel("to"), self.s_hi]),
                        ("", [self._btn("Search", lambda: self.do_search(0)), self._btn("Clear filters", self.clear_filters)])])
        return w

    # ---- Cubes -> library
    def _tab_cubes(self):
        w = QWidget(); outer = QHBoxLayout(w)
        self.c_table = QTableWidget(0, 7)
        self.c_table.setHorizontalHeaderLabels(["ID", "Name", "Sensor", "Rows×Cols×Bands", "Range (nm)", "Unit", "QC"])
        self.c_table.setSelectionBehavior(QAbstractItemView.SelectRows); self.c_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.c_table.setEditTriggers(QAbstractItemView.NoEditTriggers); self.c_table.verticalHeader().hide()
        ch = self.c_table.horizontalHeader(); ch.setSectionResizeMode(1, QHeaderView.Stretch)
        for i in (0, 2, 3, 4, 5, 6): ch.setSectionResizeMode(i, QHeaderView.ResizeToContents)
        outer.addWidget(self.c_table, 3)
        self.c_mode = self._combo(["Single pixel (row, col)", "Box (row0, col0, row1, col1)", "ROI file (ENVI ASCII / GeoJSON)"])
        self.c_r0, self.c_c0, self.c_r1, self.c_c1 = (QSpinBox() for _ in range(4))
        for sp in (self.c_r0, self.c_c0, self.c_r1, self.c_c1): sp.setRange(0, 10_000_000)
        self.c_r1.setValue(1); self.c_c1.setValue(1)
        self.c_file = QLineEdit(); self.c_onebased = QCheckBox("ROI file pixels are 1-based (ENVI default)"); self.c_onebased.setChecked(True)
        self.c_name = QLineEdit("ROI 1"); self.c_cat = self._combo([""], editable=True); self.c_lib = QLineEdit("ROI spectra")
        self._form_c = self._form([
            ("Mode", self.c_mode),
            ("Pixel / box start", [QLabel("row"), self.c_r0, QLabel("col"), self.c_c0]),
            ("Box end (exclusive)", [QLabel("row"), self.c_r1, QLabel("col"), self.c_c1]),
            ("ROI file", [self.c_file, self._btn("Browse…", lambda: self._browse_file(self.c_file))]),
            ("", self.c_onebased), ("Name", self.c_name), ("Category", self.c_cat), ("Save into library", self.c_lib),
            ("", [self._btn("Preview", self.on_roi_preview), self._btn("Save to library", self.on_roi_save),
                  self._btn("Open in image viewer", self.open_selected_in_viewer),
                  self._btn("Refresh cubes", self.refresh_all)])])
        outer.addWidget(self._form_c, 2)
        self.c_table.itemSelectionChanged.connect(self.on_cube_selected)
        return w

    # ---- Builder
    def _tab_builder(self):
        w = QWidget(); outer = QHBoxLayout(w)
        left = QWidget()
        self.b_name = QLineEdit("My library")
        self.b_grid = self._combo(["First input spectrum", "From an image / cube file", "From an ASCII wavelength file", "Regular range"])
        self.b_gpath = QLineEdit(); self.b_start = QDoubleSpinBox(); self.b_end = QDoubleSpinBox(); self.b_step = QDoubleSpinBox()
        for sp, v in ((self.b_start, 400), (self.b_end, 2500), (self.b_step, 10)):
            sp.setRange(1, 20000); sp.setDecimals(2); sp.setValue(v); sp.setSuffix(" nm")
        self.b_method = self._combo(["linear", "gaussian"]); self.b_unit = self._combo(["auto"] + UNITS[:2])
        self.b_cov = QDoubleSpinBox(); self.b_cov.setRange(0.05, 1.0); self.b_cov.setSingleStep(0.05); self.b_cov.setValue(0.5)
        self.b_cube = QLineEdit(); self.b_cube.setPlaceholderText("image the ROI / GeoJSON inputs are measured from")
        self.b_onebased = QCheckBox("ROI file pixels are 1-based (ENVI default)"); self.b_onebased.setChecked(True)
        self._form([
            ("Library name", self.b_name), ("Wavelength grid", self.b_grid),
            ("  from file", [self.b_gpath, self._btn("Browse…", lambda: self._browse_file(self.b_gpath))]),
            ("  or range", [self.b_start, self.b_end, self.b_step]),
            ("Resampling", self.b_method), ("Unit", self.b_unit), ("Min. coverage", self.b_cov),
            ("Cube for ROI inputs", [self.b_cube, self._btn("Browse…", lambda: self._browse_file(self.b_cube))]),
            ("", self.b_onebased)], left)
        right = QWidget(); rl = QVBoxLayout(right)
        rl.addWidget(QLabel("Inputs (files: spectra, ENVI ASCII ROI, GeoJSON)  — plus any rows selected in the table:"))
        self.b_files = QListWidget(); rl.addWidget(self.b_files, 1)
        h = QHBoxLayout()
        h.addWidget(self._btn("Add files…", self._add_build_files))
        h.addWidget(self._btn("Remove", lambda: [self.b_files.takeItem(self.b_files.row(i)) for i in self.b_files.selectedItems()]))
        self.b_use_sel = QCheckBox("Include rows selected in the table"); self.b_use_sel.setChecked(False); h.addWidget(self.b_use_sel)
        h.addStretch(1); h.addWidget(self._btn("Build library", self.on_build_clicked)); rl.addLayout(h)
        self.b_report = QPlainTextEdit(); self.b_report.setReadOnly(True); self.b_report.setMaximumHeight(110)
        self.b_report.setPlaceholderText("Build report: spectra added, spectra rejected and why")
        rl.addWidget(self.b_report)
        outer.addWidget(left, 2); outer.addWidget(right, 3)
        return w

    # ---- Match
    def _tab_match(self):
        w = QWidget(); outer = QHBoxLayout(w)
        left = QWidget()
        self.m_src_sel = QRadioButton("First selected row in the table"); self.m_src_file = QRadioButton("Spectrum file (first spectrum):")
        self.m_src_sel.setChecked(True); grp = QButtonGroup(w); grp.addButton(self.m_src_sel); grp.addButton(self.m_src_file)
        self.m_file = QLineEdit()
        self.m_metric = self._combo(["SAM (spectral angle)", "RMSE"]); self.m_k = QSpinBox(); self.m_k.setRange(1, 200); self.m_k.setValue(10)
        self.m_method = self._combo(["linear", "gaussian"]); self.m_unit = self._combo(["same as query"] + UNITS)
        self.m_lib = self._combo([ANY]); self.m_ov = QDoubleSpinBox(); self.m_ov.setRange(0.05, 1.0); self.m_ov.setSingleStep(0.05); self.m_ov.setValue(0.5)
        self._form([("Query", self.m_src_sel), ("", [self.m_src_file, self.m_file, self._btn("Browse…", lambda: self._browse_file(self.m_file))]),
                    ("Metric", self.m_metric), ("Top N", self.m_k), ("Resampling", self.m_method), ("Unit filter", self.m_unit),
                    ("Library", self.m_lib), ("Min. overlap", self.m_ov), ("", self._btn("Find matches", self.on_match_clicked))], left)
        self.m_table = QTableWidget(0, 6)
        self.m_table.setHorizontalHeaderLabels(["#", "Material", "Library", "Category", "Score", "Overlap"])
        self.m_table.setSelectionBehavior(QAbstractItemView.SelectRows); self.m_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.m_table.setEditTriggers(QAbstractItemView.NoEditTriggers); self.m_table.verticalHeader().hide()
        self.m_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.m_table.itemSelectionChanged.connect(self.on_match_selected)
        outer.addWidget(left, 2); outer.addWidget(self.m_table, 3)
        return w

    # ---- Export
    def _tab_export(self):
        self.e_lib = self._combo([])
        self.e_msg = QLabel("")
        return self._form([("Library", self.e_lib),
                           ("", [self._btn("Export as ENVI .sli + .hdr…", lambda: self._export_dialog("sli")),
                                 self._btn("Export as CSV…", lambda: self._export_dialog("csv"))]),
                           ("", self.e_msg)])

    # ---- Database
    def _tab_db(self):
        self.d_url = QLabel(); self.d_url.setTextInteractionFlags(Qt.TextSelectableByMouse); self.d_stats = QLabel()
        self.d_lib = self._combo([])
        self.d_dir = QLineEdit(settings.library_dir())
        self.d_auto = QCheckBox("Load this folder every time the app starts (each file = one library)")
        self.d_auto.setChecked(settings.autoload_enabled())
        self.d_auto.toggled.connect(lambda on: settings.save_autoload(on) if not self.sync else None)
        return self._form([("Database", self.d_url), ("Contents", self.d_stats),
                           ("", [self._btn("Use / create SQLite file…", self._choose_sqlite),
                                 self._btn("Connect to PostgreSQL…", self._choose_postgres),
                                 self._btn("Docker PostgreSQL (default)", self._choose_docker)]),
                           ("Libraries folder", [self.d_dir, self._btn("Browse…", lambda: self._browse_dir(self.d_dir)),
                                                 self._btn("Load now", self.load_library_folder)]),
                           ("", self.d_auto),
                           ("Delete library", [self.d_lib, self._btn("Delete", self.on_delete_library)]),
                           ("", self._btn("Reset database (delete everything)…", self.on_reset_db))])

    # ------------------------------------------------------------------ small helpers
    def _browse_file(self, line):
        p, _ = QFileDialog.getOpenFileName(self, "Choose file", line.text() or str(Path.home()))
        if p: line.setText(p)

    def _browse_dir(self, line):
        p = QFileDialog.getExistingDirectory(self, "Choose folder", line.text() or str(Path.home()))
        if p: line.setText(p)

    def _add_build_files(self):
        ps, _ = QFileDialog.getOpenFileNames(self, "Add input files", str(Path.home()))
        self.b_files.addItems(ps)

    def _about(self):
        QMessageBox.information(self, "About", f"DS3 Library {VERSION}\nCreated by Kapih Team\nDatabase: {settings.pretty(self.db_url)}")

    @staticmethod
    def _fill_combo(c, items, first=None):
        cur = c.currentText()
        c.blockSignals(True); c.clear()
        if first: c.addItem(first)
        c.addItems(items)
        i = c.findText(cur)
        if i >= 0: c.setCurrentIndex(i)
        c.blockSignals(False)

    def _selected_ids(self):
        return [self.rows[r.row()]["id"] for r in self.table.selectionModel().selectedRows() if r.row() < len(self.rows)]

    # ------------------------------------------------------------------ refresh / search / plot
    def refresh_all(self, search=True):
        def job():
            with session_scope(self.db_url) as s:
                return dict(libs=list_libraries(s), cats=list_categories(s), cubes=list_cubes(s), stats=db_stats(s))
        self.run(job, lambda d: self._after_refresh(d, search), "Loading…")

    def _after_refresh(self, d, search):
        names = [x["name"] for x in d["libs"]]
        self._fill_combo(self.s_lib, names, ANY); self._fill_combo(self.m_lib, names, ANY)
        self._fill_combo(self.e_lib, names); self._fill_combo(self.d_lib, names)
        self._fill_combo(self.s_cat, d["cats"], ANY)
        cur = self.c_cat.currentText(); self.c_cat.clear(); self.c_cat.addItems([""] + d["cats"]); self.c_cat.setCurrentText(cur)
        self.cubes = d["cubes"]
        self.image_panel.refresh_images()
        t = self.c_table; t.setRowCount(len(self.cubes))
        for i, c in enumerate(self.cubes):
            vals = [c["id"], c["name"], c["sensor"] or "", f'{c["rows"]}×{c["cols"]}×{c["bands"]}',
                    f'{c["wl_min_nm"]:.0f}–{c["wl_max_nm"]:.0f}', c["unit"],
                    "—" if c["qc_passed"] is None else ("passed" if c["qc_passed"] else "FAILED")]
            for j, v in enumerate(vals): t.setItem(i, j, _item(v, j in (0, 3, 4)))
        st = d["stats"]
        txt = f'{st["spectra"]} spectra · {st["libraries"]} libraries · {st["materials"]} materials · {st["cubes"]} cubes'
        self.d_stats.setText(txt); self.d_url.setText(settings.pretty(self.db_url)); self.db_label.setText(f"  {txt}  ")
        self.stats = st
        if search:
            self.do_search(self.offset if self.offset < max(1, st["spectra"]) else 0)

    def clear_filters(self):
        self.s_text.clear(); self.s_cat.setCurrentIndex(0); self.s_unit.setCurrentIndex(0)
        self.s_purity.setCurrentIndex(0); self.s_lib.setCurrentIndex(0); self.s_lo.setValue(0); self.s_hi.setValue(0)
        self.do_search(0)

    def read_filters(self):
        f = {}
        if self.s_text.text().strip(): f["text"] = self.s_text.text().strip()
        for key, c in (("category", self.s_cat), ("unit", self.s_unit), ("purity", self.s_purity), ("library", self.s_lib)):
            if c.currentText() and c.currentText() != ANY: f[key] = c.currentText()
        lo, hi = self.s_lo.value(), self.s_hi.value()
        if lo > 0 and hi > lo: f["covers_nm"] = (lo, hi)
        return f

    def do_search(self, offset=0):
        f = self.read_filters(); self._filters = f

        def job():
            with session_scope(self.db_url) as s:
                return find_spectra(s, limit=self.PAGE, offset=offset, **f)
        self.run(job, self._show_results, "Searching…")

    def _show_results(self, res):
        self.rows, self.total, self.offset = res["results"], res["total"], res["offset"]
        t = self.table; t.blockSignals(True); t.clearSelection(); t.setRowCount(len(self.rows))
        for i, r in enumerate(self.rows):
            vals = [r["id"], r["material"], r["category"] or "—", r["library"], r["unit"], r["purity"], r["n_bands"],
                    f'{r["wl_min_nm"]:.0f}–{r["wl_max_nm"]:.0f}']
            for j, v in enumerate(vals): t.setItem(i, j, _item(v, j in (0, 6, 7)))
        t.blockSignals(False)
        n = len(self.rows)
        self.pager.setText(f"{self.offset + 1}–{self.offset + n} of {self.total}" if n else "no spectra match")
        self.b_prev.setEnabled(self.offset > 0); self.b_next.setEnabled(self.offset + n < self.total)
        self.canvas.clear("Select spectra in the table to plot them"); self.info.setText("No spectrum selected")

    def on_selection(self):
        ids = self._selected_ids()[: self.MAX_PLOT]
        if not ids: return
        try:
            with session_scope(self.db_url) as s:
                recs = get_spectra(s, ids)
        except Exception as e:
            self.show_error(str(e)); return
        self.canvas.show_spectra([dict(label=r["material"], wavelengths_nm=r["wavelengths_nm"], values=r["values"]) for r in recs],
                                 title=f"{len(recs)} spectrum/spectra", unit=recs[0]["unit"] if recs else None)
        if len(recs) == 1:
            r = recs[0]; wl = r["wavelengths_nm"]
            self.info.setText(f'<b>{r["material"]}</b> · {r["category"] or "no category"} · library: {r["library"]}<br>'
                              f'{r["unit"]} · {r["purity"]} · {len(wl)} bands, {wl[0]:.0f}–{wl[-1]:.0f} nm')
        else:
            n = len(self._selected_ids())
            self.info.setText(f"{n} spectra selected" + (f" (first {self.MAX_PLOT} plotted)" if n > self.MAX_PLOT else ""))

    # ------------------------------------------------------------------ import
    def on_import_clicked(self):
        self.import_path(self.i_path.text().strip())

    def import_path(self, path):
        if not path or not Path(path).exists():
            self.show_error(f"Path not found: {path or '(empty)'}"); return
        kw = {}
        if self.i_kind.currentIndex() > 0: kw["kind"] = self.i_kind.currentText()
        if self.i_source.text().strip(): kw["source_name"] = self.i_source.text().strip()
        if self.i_unit.currentText() != "auto": kw["unit"] = self.i_unit.currentText()
        wlfile, div = self.i_wl.text().strip(), self.i_div.text().strip()
        try:
            if div: kw["reflectance_divisor"] = float(div)
        except ValueError:
            self.show_error("Reflectance divisor must be a number (e.g. 10000)"); return
        db = self.db_url

        def job():
            k = dict(kw)
            notes = []
            if wlfile:
                import re
                m = re.fullmatch(r"\s*([0-9.]+)\s*[-–]\s*([0-9.]+)\s*(nm)?\s*", wlfile)
                if m:                                   # "430-860": evenly spaced range (for files with no wavelengths, e.g. .mat)
                    k["wavelength_range"] = (float(m.group(1)), float(m.group(2))); notes.append(f"wavelength range {wlfile} nm")
                else:
                    wl, fw, n = read_wavelengths_file(wlfile); k["wavelengths"] = wl.tolist(); notes += n
            return notes, ingest_path(path, db_url=db, **k)

        def done(res):
            notes, results = res
            self.i_log.appendPlainText("\n".join(f"note: {n}" for n in notes)) if notes else None
            counts = {}
            for r in results:
                counts[r.status] = counts.get(r.status, 0) + 1
                self.i_log.appendPlainText(f"[{r.status:9s}] {Path(r.path).name}  {r.detail}")
            self.i_log.appendPlainText("— " + ", ".join(f"{v} {k}" for k, v in counts.items()) + "\n")
            self.refresh_all()
        self.run(job, done, "Importing…")

    # ------------------------------------------------------------------ cubes -> library
    def on_cube_selected(self):
        r = self.c_table.currentRow()
        if 0 <= r < len(self.cubes):
            c = self.cubes[r]
            self.info.setText(f'<b>{c["name"]}</b> · {c["format"]} · {c["rows"]}×{c["cols"]}×{c["bands"]} · {c["unit"]}<br>'
                              f'{c["wl_min_nm"]:.0f}–{c["wl_max_nm"]:.0f} nm · sensor {c["sensor"] or "unknown"} · CRS {c["crs"] or "none"} · '
                              f'QC {"passed" if c["qc_passed"] else ("FAILED" if c["qc_passed"] is False else "n/a")}')

    def open_selected_in_viewer(self):
        try:
            cid = self._selected_cube_id()
        except IngestError as e:
            self.show_error(str(e)); return
        self.modes.setCurrentIndex(1); self.image_panel.load_image(cid)

    def _roi_args(self):
        m = self.c_mode.currentIndex()
        if m == 0: return dict(pixel=(self.c_r0.value(), self.c_c0.value()))
        if m == 1: return dict(bbox_px=(self.c_r0.value(), self.c_c0.value(), self.c_r1.value(), self.c_c1.value()))
        if not self.c_file.text().strip(): raise IngestError("choose an ROI file")
        return dict(roi_file=self.c_file.text().strip())

    def _selected_cube_id(self):
        r = self.c_table.currentRow()
        if r < 0 or r >= len(self.cubes): raise IngestError("select a cube in the table first")
        return self.cubes[r]["id"]

    def on_roi_preview(self, then=None):
        try:
            cid, args = self._selected_cube_id(), self._roi_args()
        except IngestError as e:
            self.show_error(str(e)); return
        name, cat, db, ob = self.c_name.text().strip() or "ROI", self.c_cat.currentText().strip() or None, self.db_url, self.c_onebased.isChecked()

        def job():
            return cube_roi.measure(cid, name=name, category=cat, one_based=ob, db_url=db, **args)

        def done(res):
            spectra, desc = res
            self._measured = (spectra, cid)
            self.canvas.show_spectra([dict(label=sp.material, wavelengths_nm=sp.wavelengths_nm, values=sp.values,
                                           std=sp.uncertainty) for sp in spectra[:8]],
                                     title=f"ROI preview — {spectra[0].metadata.get('n_pixels', '')} px" if len(spectra) == 1 else f"{len(spectra)} ROIs",
                                     unit=spectra[0].unit)
            n = spectra[0].metadata.get("n_pixels")
            self.info.setText(f"Preview of {len(spectra)} spectrum/spectra from cube {cid} ({desc})" + (f" · {n} pixels averaged" if n else "")
                              + "<br>Not saved yet — press <b>Save to library</b>.")
            if then: then()
        self.run(job, done, "Measuring ROI…")

    def on_roi_save(self):
        def save():
            spectra, cid = self._measured; lib = self.c_lib.text().strip() or "ROI spectra"; db = self.db_url

            def job(): return cube_roi.save(spectra, cid, library=lib, db_url=db)
            self.run(job, lambda ids: (self.info.setText(f"Saved {len(ids)} spectrum/spectra to library '{lib}'."),
                                       self.refresh_all()), "Saving…")
        self.on_roi_preview(then=save)

    # ------------------------------------------------------------------ builder
    def _make_grid(self):
        i = self.b_grid.currentIndex()
        if i == 0: return "first"
        if i == 1: return Grid.from_cube(self.b_gpath.text().strip())
        if i == 2: return Grid.from_ascii(self.b_gpath.text().strip())
        a, b, s = self.b_start.value(), self.b_end.value(), self.b_step.value()
        if b <= a: raise IngestError("range end must be greater than start")
        return Grid(np.arange(a, b + s / 2, s), origin=f"range:{a:g}-{b:g}/{s:g}")

    def on_build_clicked(self):
        files = [self.b_files.item(i).text() for i in range(self.b_files.count())]
        self.run_build(self.b_name.text().strip(), files, self._selected_ids() if self.b_use_sel.isChecked() else [])

    def run_build(self, name, files, ids):
        if not name:
            self.show_error("Give the library a name"); return
        if not files and not ids:
            self.show_error("Add input files, or tick 'Include rows selected in the table' and select rows"); return
        m, u, cov, cube, ob, db = (self.b_method.currentText(), self.b_unit.currentText(), self.b_cov.value(),
                                   self.b_cube.text().strip() or None, self.b_onebased.isChecked(), self.db_url)

        def job():
            return build_library(name, files, self._make_grid(), cube=cube, method=m, unit=None if u == "auto" else u,
                                 one_based=ob, min_coverage=cov, db_url=db, spectrum_ids=ids or None)

        def done(rep):
            self.b_report.setPlainText(rep.summary())
            self.refresh_all(search=False)
            self.s_lib.setCurrentText(name) if self.s_lib.findText(name) >= 0 else None
            self.do_search(0)
        self.run(job, done, "Building library…")

    # ------------------------------------------------------------------ match
    def on_match_clicked(self):
        self.run_match()

    def run_match(self):
        metric = "sam" if self.m_metric.currentIndex() == 0 else "rmse"
        k, meth, ov, db = self.m_k.value(), self.m_method.currentText(), self.m_ov.value(), self.db_url
        unit_sel, lib = self.m_unit.currentText(), self.m_lib.currentText()
        exclude = None
        try:
            if self.m_src_file.isChecked():
                sp = read_spectra(self.m_file.text().strip())[0]
                q = dict(material=sp.material, wl=np.asarray(sp.wavelengths_nm, float), v=np.asarray(sp.values, float), unit=sp.unit)
            else:
                ids = self._selected_ids()
                if not ids: raise IngestError("select a spectrum in the table (or choose a file)")
                with session_scope(db) as s:
                    r = get_spectra(s, ids[:1])[0]
                q = dict(material=r["material"], wl=r["wavelengths_nm"], v=r["values"], unit=r["unit"]); exclude = r["id"]
        except Exception as e:
            self.show_error(f"{type(e).__name__}: {e}"); return
        filt = dict(unit=q["unit"] if unit_sel == "same as query" else unit_sel)
        if lib and lib != ANY: filt["library"] = lib

        def job():
            with session_scope(db) as s:
                return match_spectrum(s, q["wl"], q["v"], top_k=k, metric=metric, method=meth, min_overlap=ov,
                                      exclude_id=exclude, **filt)

        def done(rows):
            self._match_rows, self._match_query, self._match_metric = rows, q, metric
            t = self.m_table; t.blockSignals(True); t.setRowCount(len(rows))
            for i, r in enumerate(rows):
                score = f'{np.degrees(r["score"]):.2f}°' if metric == "sam" else f'{r["score"]:.4f}'
                for j, v in enumerate([i + 1, r["material"], r["library"], r["category"] or "—", score, f'{r["overlap"]*100:.0f}%']):
                    t.setItem(i, j, _item(v, j in (0, 4, 5)))
            t.blockSignals(False)
            self.info.setText(f'{len(rows)} match(es) for <b>{q["material"]}</b> ({q["unit"]}).' if rows else
                              "No stored spectrum overlaps the query enough (try a lower overlap or another unit filter).")
            if rows: t.selectRow(0)
        self.run(job, done, "Matching…")

    def on_match_selected(self):
        r = self.m_table.currentRow()
        if r < 0 or r >= len(self._match_rows) or not self._match_query: return
        q, m = self._match_query, self._match_rows[r]
        try:
            with session_scope(self.db_url) as s:
                rec = get_spectra(s, [m["id"]])[0]
        except Exception as e:
            self.show_error(str(e)); return
        self.canvas.show_spectra([dict(label=f'query: {q["material"]}', wavelengths_nm=q["wl"], values=q["v"], style="k--"),
                                  dict(label=f'match: {rec["material"]}', wavelengths_nm=rec["wavelengths_nm"], values=rec["values"])],
                                 title="Query vs match", unit=q["unit"])

    # ------------------------------------------------------------------ export
    def _export_dialog(self, kind):
        lib = self.e_lib.currentText()
        if not lib:
            self.show_error("No library to export"); return
        ext = "ENVI library (*.sli)" if kind == "sli" else "CSV (*.csv)"
        p, _ = QFileDialog.getSaveFileName(self, "Export", str(Path.home() / lib.replace(" ", "_")), ext)
        if p: self.export_library(lib, p, kind)

    def export_library(self, lib, path, kind):
        db = self.db_url
        fn = export_envi_sli if kind == "sli" else export_csv
        self.run(lambda: fn(lib, path, db_url=db), lambda out: self.e_msg.setText(f"Wrote {out}"), "Exporting…")

    # ------------------------------------------------------------------ database
    def use_database(self, url):
        try:
            init_db(url)
        except Exception as e:
            self.show_error(f"Cannot use database {settings.pretty(url)}:\n{e}"); return
        self.db_url = url; settings.save_url(url) if not self.sync else None
        self.offset = 0; self._set_title(); self.refresh_all()

    def _choose_sqlite(self):
        p, _ = QFileDialog.getSaveFileName(self, "SQLite database file", str(settings.APP_DIR / "ds3_library.db"),
                                           "SQLite (*.db *.sqlite)", options=QFileDialog.DontConfirmOverwrite)
        if p: self.use_database(settings.sqlite_url(p))

    def _choose_postgres(self):
        url, ok = QInputDialog.getText(self, "PostgreSQL", "SQLAlchemy URL:", text="postgresql+psycopg://user:password@localhost:5432/ds3")
        if ok and url.strip(): self.use_database(url.strip())

    def _choose_docker(self):
        """The PostgreSQL started by `docker compose up -d` in the repository root (user/password/db = ds3)."""
        self.use_database(settings.DOCKER_DB_URL)
        if self.last_error and "Cannot use database" in self.last_error:
            return

    def load_library_folder(self, folder=None):
        """Every file in the folder becomes its own library (named after the file). Safe to repeat."""
        folder = str(folder or self.d_dir.text().strip() or settings.library_dir())
        if not Path(folder).is_dir():
            self.show_error(f"Libraries folder not found: {folder}"); return
        if not self.sync: settings.save_library_dir(folder)
        db = self.db_url
        self.status.showMessage(f"Loading libraries from {folder} …")
        self.i_log.appendPlainText(f"— libraries folder: {folder}")

        def job():
            return ingest_library_folder(folder, db_url=db)

        def done(results):
            counts = {}
            for r in results:
                counts[r.status] = counts.get(r.status, 0) + 1
                self.i_log.appendPlainText(f"[{r.status:9s}] {Path(r.path).name}  {r.detail}")
            if not results:
                self.i_log.appendPlainText(f"(folder is empty — copy your library files into {folder}, then press Load now)")
            else:
                self.i_log.appendPlainText("— " + ", ".join(f"{v} {k}" for k, v in counts.items()) + "\n")
            bad = [r for r in results if r.status == "error"]
            if bad:
                self.status.showMessage(f"{len(bad)} library file(s) could not be read — see the Import tab log", 12000)
            self.refresh_all()
        self.run(job, done, "Loading libraries…")

    def on_delete_library(self):
        name = self.d_lib.currentText()
        if not name: return
        if not self._confirm(f"Delete library '{name}' and all its spectra? This cannot be undone."): return
        db = self.db_url

        def job():
            with session_scope(db) as s:
                return delete_library(s, name)
        self.run(job, lambda ok: self.refresh_all(), "Deleting…")

    def on_reset_db(self):
        if not self._confirm("Delete EVERYTHING in this database (all libraries, spectra and registered cubes)?"): return
        db = self.db_url
        self.run(lambda: reset_db(db), lambda _: (setattr(self, "offset", 0), self.refresh_all()), "Resetting…")
