"""Library tab: the PostgreSQL material spectral library inside HyperVista.

Feature set follows the DS3 library module (new_hyperspectral/modified_ds3_prototype.html + its FastAPI
backend), rebuilt in Qt for this window's layout:

    search bar     "Search libraries by name" | "Search all materials by name" (pop-up results) | acquisition filter
    libraries      list with material counts, add / delete, import a wide CSV
    category tree  materials.category holds a path (Vegetation/Tree/Tsuga/Canadensis); the tree shows the
                   hierarchy with counts, clicking a node filters to it and everything below; rename / move nodes
    materials      server-side search, "See more" paging, acquisition tags as coloured chips, add / edit
                   (category + acquisition for many at once) / delete
    spectrum       sample -> spectrum (+/- std) in the Spectra pane, acquisition tag per spectrum, export CSV
    compare/match  overlay and export spectra; save aggregated signatures / clicked pixels; match a pixel by
                   spectral angle, optionally limited to the acquisition type and category
    ROIs           spectra saved from the ROI Analysis tab show their ROI class and source ROI(s) / image(s)

Acquisition types: lab, field, airborne, satellite (spectra.acquisition_type, NULL = untagged).
All database / HTTP work runs on a QThreadPool so the window never freezes; superseded requests are dropped.
"""
from __future__ import annotations

import csv
import os
import re
import traceback
from pathlib import Path

import matplotlib
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from sqlalchemy.exc import DBAPIError, SQLAlchemyError

from .. import APP_NAME
from ..library import repository as repo
from ..library.client import UNSET, make_client
from ..library.database import default_url
from ..library.formats import file_dialog_filter

W = QtWidgets
Qt = QtCore.Qt
PALETTE = [tuple(matplotlib.colormaps["tab10"](i)[:3]) for i in range(10)]
PAGE_SIZE = 25
MAX_COMPARE = 8
SETTINGS_ORG = "Kapih Team"
SETTINGS_APP = APP_NAME
ID_ROLE = Qt.UserRole
TAGS_ROLE = Qt.UserRole + 1
PATH_ROLE = Qt.UserRole + 2

ACQ = repo.ACQUISITION_TYPES                         # ('lab', 'field', 'airborne', 'satellite')
ACQ_LABEL = {"lab": "Lab", "field": "Field", "airborne": "Airborne", "satellite": "Satellite", "untagged": "Untagged"}
ACQ_COLOR = {"lab": "#a78bfa", "field": "#34d399", "airborne": "#60a5fa", "satellite": "#fbbf24",
             "untagged": "#8b949e"}
ACQ_HELP = {"lab": "laboratory spectrometer (controlled illumination)", "field": "field spectroradiometer",
            "airborne": "airborne / UAV imaging sensor", "satellite": "spaceborne sensor"}

PANEL_QSS = """
QLabel[role="fieldlabel"] { color: #8b9bb4; font-size: 8.5pt; padding-left: 2px; }
QLineEdit[role="search"], QComboBox[role="search"] {
    border: 1px solid #3a4a63; border-radius: 7px; padding: 5px 10px; min-height: 22px; }
QLineEdit[role="search"]:focus, QComboBox[role="search"]:focus { border: 1px solid #5b8def; }
QLabel[role="crumb"] { color: #8b9bb4; }
"""


def _dot(color, size=10):
    pm = QtGui.QPixmap(size, size)
    pm.fill(Qt.transparent)
    p = QtGui.QPainter(pm)
    p.setRenderHint(QtGui.QPainter.Antialiasing)
    p.setPen(Qt.NoPen)
    p.setBrush(QtGui.QColor(color))
    p.drawEllipse(0, 0, size - 1, size - 1)
    p.end()
    return QtGui.QIcon(pm)


def _crumb(path):
    return " › ".join(repo.category_levels(path)) if path else ""


# ============================================================ background work
class _Signals(QtCore.QObject):
    done = QtCore.pyqtSignal(object)
    failed = QtCore.pyqtSignal(str)
    progress = QtCore.pyqtSignal(int, int, str)


class _Task(QtCore.QRunnable):
    def __init__(self, fn, args, kwargs, with_progress):
        super().__init__()
        self.fn, self.args, self.kwargs = fn, args, kwargs
        self.sig = _Signals()
        if with_progress:
            self.kwargs["progress"] = lambda i, n, t: self.sig.progress.emit(i, n, t)

    def run(self):
        try:
            res = self.fn(*self.args, **self.kwargs)
        except Exception as e:                                                   # noqa: BLE001
            if isinstance(e, DBAPIError) and e.orig is not None:
                msg = str(e.orig).strip()                    # driver message, without SQLAlchemy's wrapper text
            else:
                msg = str(e) or e.__class__.__name__
            if not isinstance(e, (repo.NotFound, ValueError, ConnectionError, RuntimeError, OSError, SQLAlchemyError)):
                traceback.print_exc()
            self.sig.failed.emit(msg.splitlines()[0][:400])
            return
        self.sig.done.emit(res)


# ============================================================ acquisition chips
class TagDelegate(W.QStyledItemDelegate):
    """Paints a cell's acquisition tags (list in TAGS_ROLE) as coloured rounded chips."""

    def paint(self, painter, opt, idx):
        tags = idx.data(TAGS_ROLE) or []
        o = W.QStyleOptionViewItem(opt)
        self.initStyleOption(o, idx)
        o.text = ""
        style = o.widget.style() if o.widget else W.QApplication.style()
        style.drawControl(W.QStyle.CE_ItemViewItem, o, painter, o.widget)
        if not tags:
            return
        light = o.palette.color(QtGui.QPalette.Base).lightness() > 128
        fm = o.fontMetrics
        h = fm.height() + 2
        x, y = o.rect.x() + 4, o.rect.y() + (o.rect.height() - h) // 2
        painter.save()
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        for t in tags:
            label = ACQ_LABEL.get(t, t)
            w = fm.horizontalAdvance(label) + 14
            if x + w > o.rect.right() - 2:
                painter.setPen(o.palette.color(QtGui.QPalette.Text))
                painter.drawText(QtCore.QRectF(x, y, 16, h), Qt.AlignCenter, "…")
                break
            col = QtGui.QColor(ACQ_COLOR.get(t, "#8b949e"))
            bg = QtGui.QColor(col)
            bg.setAlpha(150 if o.state & W.QStyle.State_Selected else (60 if not light else 45))
            painter.setPen(Qt.NoPen)
            painter.setBrush(bg)
            painter.drawRoundedRect(QtCore.QRectF(x, y, w, h), h / 2, h / 2)
            if o.state & W.QStyle.State_Selected:
                painter.setPen(o.palette.color(QtGui.QPalette.HighlightedText))
            else:
                painter.setPen(col.darker(160) if light else col.lighter(115))
            painter.drawText(QtCore.QRectF(x, y, w, h), Qt.AlignCenter, label)
            x += w + 4
        painter.restore()

    def sizeHint(self, opt, idx):
        tags = idx.data(TAGS_ROLE) or []
        fm = opt.fontMetrics
        return QtCore.QSize(8 + sum(fm.horizontalAdvance(ACQ_LABEL.get(t, t)) + 18 for t in tags), fm.height() + 8)


def _acq_combo(include_all=None, include_unchanged=False, counts=None):
    """Combo of acquisition types. include_all: label of a 'no filter' entry (data None)."""
    cb = W.QComboBox()
    if include_unchanged:
        cb.addItem("(leave unchanged)", UNSET)
    if include_all:
        cb.addItem(include_all, None)
    for t in ACQ:
        n = f"  ({counts.get(t, 0)})" if counts else ""
        cb.addItem(_dot(ACQ_COLOR[t]), ACQ_LABEL[t] + n, t)
        cb.setItemData(cb.count() - 1, ACQ_HELP[t], Qt.ToolTipRole)
    n = f"  ({counts.get('untagged', 0)})" if counts else ""
    cb.addItem(_dot(ACQ_COLOR["untagged"]), ACQ_LABEL["untagged"] + n, repo.UNTAGGED)
    return cb


def _category_combo(paths, current=None):
    """Editable combo offering the library's existing paths (with completion) - keeps spellings consistent."""
    cb = W.QComboBox()
    cb.setEditable(True)
    cb.setInsertPolicy(W.QComboBox.NoInsert)
    cb.addItem("")
    for p in paths:
        cb.addItem(p)
    cb.setEditText(current or "")
    comp = W.QCompleter(paths, cb)
    comp.setCaseSensitivity(Qt.CaseInsensitive)
    comp.setFilterMode(Qt.MatchContains)
    cb.setCompleter(comp)
    cb.lineEdit().setPlaceholderText("e.g. Vegetation/Tree/Tsuga/Canadensis  (empty = uncategorised)")
    cb.setMinimumWidth(340)
    return cb


CATEGORY_HINT = ("Levels are separated by  /  (also accepted:  >  ->  ›  |).  "
                 "Example: Vegetation/Tree/Tsuga/Canadensis")


# ============================================================ dialogs
def _form_dialog(parent, title, fields, ok_text="Create"):
    """Small modal form. fields = [(key, label, default, placeholder)] -> dict or None."""
    d = W.QDialog(parent)
    d.setWindowTitle(title)
    form = W.QFormLayout(d)
    eds = {}
    for key, label, default, ph in fields:
        e = W.QLineEdit(default or "")
        e.setPlaceholderText(ph or "")
        e.setMinimumWidth(320)
        form.addRow(label, e)
        eds[key] = e
    bb = W.QDialogButtonBox(W.QDialogButtonBox.Ok | W.QDialogButtonBox.Cancel)
    bb.button(W.QDialogButtonBox.Ok).setText(ok_text)
    bb.accepted.connect(d.accept)
    bb.rejected.connect(d.reject)
    form.addRow(bb)
    if not d.exec_():
        return None
    return {k: e.text().strip() for k, e in eds.items()}


def _hint(text):
    lab = W.QLabel(text)
    lab.setWordWrap(True)
    lab.setProperty("role", "crumb")
    return lab


class MaterialDialog(W.QDialog):
    """Add a material: name, category path, description."""

    def __init__(self, parent, lib_name, paths, category=None):
        super().__init__(parent)
        self.setWindowTitle(f"Add material to \"{lib_name}\"")
        self.ed_name = W.QLineEdit()
        self.ed_name.setPlaceholderText("e.g. Eastern hemlock needles")
        self.cb_cat = _category_combo(paths, category)
        self.ed_desc = W.QLineEdit()
        self.ed_desc.setPlaceholderText("optional")
        f = W.QFormLayout(self)
        f.addRow("Name", self.ed_name)
        f.addRow("Category", self.cb_cat)
        f.addRow("", _hint(CATEGORY_HINT))
        f.addRow("Description", self.ed_desc)
        f.addRow(_hint("Acquisition type is set per spectrum - save or import a spectrum for this material."))
        bb = W.QDialogButtonBox(W.QDialogButtonBox.Ok | W.QDialogButtonBox.Cancel)
        bb.button(W.QDialogButtonBox.Ok).setText("Create")
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        f.addRow(bb)

    def choices(self):
        return {"name": self.ed_name.text().strip(), "category": self.cb_cat.currentText().strip() or None,
                "description": self.ed_desc.text().strip() or None}


class EditMaterialsDialog(W.QDialog):
    """Bulk edit of the selected materials: category path and/or the acquisition tag of all their spectra."""

    def __init__(self, parent, materials, paths):
        super().__init__(parent)
        n = len(materials)
        self.setWindowTitle("Edit material" if n == 1 else f"Edit {n} materials")
        cats = {m.get("category") for m in materials}
        self.chk_cat = W.QCheckBox("Set category")
        self.cb_cat = _category_combo(paths, cats.pop() if len(cats) == 1 else "")
        self.chk_cat.setChecked(n == 1)
        self.cb_cat.setEnabled(n == 1)
        self.chk_cat.toggled.connect(self.cb_cat.setEnabled)
        self.cb_acq = _acq_combo(include_unchanged=True)
        names = ", ".join(m["name"] for m in materials[:6]) + (f" … (+{n - 6})" if n > 6 else "")
        f = W.QFormLayout(self)
        f.addRow(_hint(names))
        f.addRow(self.chk_cat, self.cb_cat)
        f.addRow("", _hint(CATEGORY_HINT))
        f.addRow("Acquisition (all spectra)", self.cb_acq)
        bb = W.QDialogButtonBox(W.QDialogButtonBox.Ok | W.QDialogButtonBox.Cancel)
        bb.button(W.QDialogButtonBox.Ok).setText("Apply")
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        f.addRow(bb)

    def choices(self):
        cat = (self.cb_cat.currentText().strip() or None) if self.chk_cat.isChecked() else UNSET
        return {"category": cat, "acquisition": self.cb_acq.currentData()}


class ImportWizard(W.QDialog):
    """Import any supported spectral data: detect, preview, override the interpretation, import."""
    UNITS = [("Auto-detect", "auto"), ("nm", "nm"), ("µm", "um"), ("wavenumber cm⁻¹", "cm-1")]
    SCALES = [("Auto-detect", "auto"), ("Reflectance 0–1", "unit"), ("Percent (0–100)", "percent"),
              ("Scaled ×10000", "x10000"), ("As measured (radiance, counts…)", "none")]
    LAYOUTS = [("Auto-detect", "auto"), ("Spectra in columns (wavelength column)", "columns"),
               ("Spectra in rows (wavelengths in header)", "rows"), ("Long: name, wavelength, value", "long")]
    RANGES = [("Auto (VSWIR for big reference libraries)", "auto"), ("Keep everything", "all"),
              ("VSWIR only (350–2500 nm)", "vswir")]
    IMAGES = [("Auto (class means, else image mean)", "auto"), ("Class means (label map / ROIs)", "classes"),
              ("Image mean", "mean"), ("Random pixels", "sample"), ("All pixels (capped)", "all")]

    def __init__(self, panel, paths):
        super().__init__(panel)
        self.panel = panel
        self.setWindowTitle("Import spectral data")
        self._gen = 0
        self.previews = {}
        # sources
        self.lst = W.QListWidget()
        self.lst.setSelectionMode(W.QAbstractItemView.SingleSelection)
        self.lst.currentRowChanged.connect(lambda *_: self.preview())
        b_add = W.QPushButton("Add files...")
        b_add.clicked.connect(self._add_files)
        b_addd = W.QPushButton("Add folder...")
        b_addd.clicked.connect(self._add_folder)
        b_rm = W.QPushButton("Remove")
        b_rm.clicked.connect(self._remove)
        left = W.QVBoxLayout()
        left.addWidget(W.QLabel("Sources"))
        left.addWidget(self.lst, 1)
        left.addLayout(_row(b_add, b_addd, b_rm))
        # target
        self.rb_exist = W.QRadioButton("Existing library")
        self.rb_new = W.QRadioButton("New library")
        self.rb_auto = W.QRadioButton("One new library per source (named after it)")
        self.cb_lib = W.QComboBox()
        for l in panel.libraries:
            self.cb_lib.addItem(l["name"], l["id"])
        self.ed_newlib = W.QLineEdit()
        self.ed_ds = W.QLineEdit()
        self.ed_ds.setPlaceholderText("automatic (from each source)")
        if panel.active_lib is not None:
            self.rb_exist.setChecked(True)
            self.cb_lib.setCurrentIndex(max(0, self.cb_lib.findData(panel.active_lib["id"])))
        elif len(paths) > 1:
            self.rb_auto.setChecked(True)
        else:
            self.rb_new.setChecked(True)
        if not panel.libraries:
            self.rb_exist.setEnabled(False)
        tgt = W.QFormLayout()
        tgt.addRow(self.rb_exist, self.cb_lib)
        tgt.addRow(self.rb_new, self.ed_newlib)
        tgt.addRow(self.rb_auto)
        tgt.addRow("Dataset", self.ed_ds)
        g_tgt = W.QGroupBox("Import into")
        g_tgt.setLayout(tgt)
        # interpretation
        self.cb_unit = self._combo(self.UNITS)
        self.cb_scale = self._combo(self.SCALES)
        self.cb_layout = self._combo(self.LAYOUTS)
        self.ed_wlcol = W.QLineEdit()
        self.ed_wlcol.setPlaceholderText("auto")
        self.ed_namecol = W.QLineEdit()
        self.ed_namecol.setPlaceholderText("auto")
        self.cb_range = self._combo(self.RANGES)
        self.cb_cat = _category_combo(panel.category_paths())
        self.chk_folders = W.QCheckBox("Subfolder names become category levels")
        self.chk_folders.setChecked(True)
        self.cb_acq = W.QComboBox()
        self.cb_acq.addItem("Auto-detect (from the file)", None)
        for t in ACQ:
            self.cb_acq.addItem(_dot(ACQ_COLOR[t]), ACQ_LABEL[t], t)
        self.chk_force = W.QCheckBox("Use for every spectrum")
        self.cb_img = self._combo(self.IMAGES)
        self.sp_px = W.QSpinBox()
        self.sp_px.setRange(1, 1000000)
        self.sp_px.setValue(100)
        self.chk_wlr = W.QCheckBox("Files without wavelengths: spread bands over")
        self.sp_wl0 = W.QDoubleSpinBox()
        self.sp_wl1 = W.QDoubleSpinBox()
        for sp, v in ((self.sp_wl0, 400), (self.sp_wl1, 2500)):
            sp.setRange(0, 100000)
            sp.setValue(v)
            sp.setSuffix(" nm")
        self.sp_lim = W.QSpinBox()
        self.sp_lim.setRange(0, 10000000)
        self.sp_lim.setSpecialValueText("all")
        f = W.QFormLayout()
        f.addRow("Wavelength unit", self.cb_unit)
        f.addRow("Values", self.cb_scale)
        f.addRow("Table layout", self.cb_layout)
        f.addRow("Wavelength / name column", _row(self.ed_wlcol, self.ed_namecol))
        f.addRow("Spectral range", self.cb_range)
        f.addRow("Category if none", self.cb_cat)
        f.addRow("", self.chk_folders)
        f.addRow("Acquisition", _row(self.cb_acq, self.chk_force))
        f.addRow("Images", _row(self.cb_img, W.QLabel("pixels"), self.sp_px))
        f.addRow(self.chk_wlr, _row(self.sp_wl0, self.sp_wl1))
        f.addRow("At most per source", self.sp_lim)
        g_opt = W.QGroupBox("Interpretation (everything is auto-detected; override only if the preview is wrong)")
        g_opt.setLayout(f)
        for w in (self.cb_unit, self.cb_scale, self.cb_layout, self.cb_range, self.cb_acq, self.cb_img):
            w.currentIndexChanged.connect(self._later)
        for w in (self.ed_wlcol, self.ed_namecol):
            w.editingFinished.connect(self._later)
        for w in (self.chk_folders, self.chk_force, self.chk_wlr):
            w.toggled.connect(self._later)
        self.cb_cat.lineEdit().editingFinished.connect(self._later)
        for w in (self.sp_px, self.sp_wl0, self.sp_wl1, self.sp_lim):
            w.valueChanged.connect(self._later)
        # preview
        from .canvases import PlotCanvas
        self.canvas = PlotCanvas()
        self.canvas.setMinimumHeight(220)
        self.lbl = W.QLabel("Select a source to preview it.")
        self.lbl.setWordWrap(True)
        self.lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.tbl = W.QTableWidget(0, 4)
        self.tbl.setHorizontalHeaderLabels(["Spectrum", "Category", "Acquisition", "Points"])
        _table_defaults(self.tbl, stretch_first=True)
        self.tbl.setMaximumHeight(150)
        right = W.QVBoxLayout()
        right.addWidget(g_tgt)
        right.addWidget(g_opt)
        prev = W.QVBoxLayout()
        prev.addWidget(self.lbl)
        prev.addWidget(self.canvas, 1)
        prev.addWidget(self.tbl)
        for g in (g_tgt, g_opt):
            g.setMaximumWidth(560)
        self.tbl.horizontalHeader().setSectionResizeMode(0, W.QHeaderView.Stretch)
        top = W.QHBoxLayout()
        top.addLayout(left, 2)
        top.addLayout(right, 0)
        top.addLayout(prev, 6)
        self.bb = W.QDialogButtonBox(W.QDialogButtonBox.Ok | W.QDialogButtonBox.Cancel)
        self.bb.button(W.QDialogButtonBox.Ok).setText("Import")
        self.bb.accepted.connect(self._accept)
        self.bb.rejected.connect(self.reject)
        lay = W.QVBoxLayout(self)
        lay.addLayout(top, 1)
        lay.addWidget(_hint("Supported: CSV/TSV/TXT (spectra in columns, rows or long format, any delimiter, "
                            "decimal comma), Excel, JSON, Parquet, ENVI spectral libraries, hyperspectral images "
                            "(ENVI, GeoTIFF, MAT, HDF5, NetCDF, NumPy), spectra matrices in MAT/HDF5/NPZ, ASD / SVC / "
                            "Spectral Evolution files, USGS splib07, ECOSTRESS, OpenSpecLib, and folders or zips of "
                            "any of these."))
        lay.addWidget(self.bb)
        self._timer = QtCore.QTimer(self, singleShot=True, interval=400, timeout=self.preview)
        self.resize(1500, 780)
        for p in paths:
            self._add(p)
        if self.lst.count():
            self.lst.setCurrentRow(0)

    def _combo(self, items):
        cb = W.QComboBox()
        for label, val in items:
            cb.addItem(label, val)
        return cb

    def _add(self, p):
        p = str(p)
        for i in range(self.lst.count()):
            if self.lst.item(i).data(ID_ROLE) == p:
                return
        it = W.QListWidgetItem(os.path.basename(p.rstrip("/\\")) or p)
        it.setData(ID_ROLE, p)
        it.setToolTip(p)
        self.lst.addItem(it)
        if not self.ed_newlib.text().strip():
            self.ed_newlib.setText(re.sub(r"[_\s]+", " ", Path(p).stem if Path(p).is_file() else Path(p).name).strip().title())

    def _add_files(self):
        ps, _ = W.QFileDialog.getOpenFileNames(self, "Spectral data files", "", file_dialog_filter())
        for p in ps:
            self._add(p)
        if ps and self.lst.currentRow() < 0:
            self.lst.setCurrentRow(0)

    def _add_folder(self):
        d = W.QFileDialog.getExistingDirectory(self, "Folder of spectral files")
        if d:
            self._add(d)
            self.lst.setCurrentRow(self.lst.count() - 1)

    def _remove(self):
        r = self.lst.currentRow()
        if r >= 0:
            self.lst.takeItem(r)

    def _later(self, *_):
        self._timer.start()

    def options(self) -> dict:
        o = {"wavelength_unit": self.cb_unit.currentData(), "value_scale": self.cb_scale.currentData(),
             "orientation": self.cb_layout.currentData(), "spectral_range": self.cb_range.currentData(),
             "category": self.cb_cat.currentText().strip() or None, "category_from_folders": self.chk_folders.isChecked(),
             "acquisition": self.cb_acq.currentData(), "force_acquisition": self.chk_force.isChecked(),
             "image_mode": self.cb_img.currentData(), "image_samples": self.sp_px.value(),
             "wavelength_column": self.ed_wlcol.text().strip() or None,
             "name_column": self.ed_namecol.text().strip() or None, "limit": self.sp_lim.value() or None}
        if self.chk_wlr.isChecked():
            o["wl_range"] = (self.sp_wl0.value(), self.sp_wl1.value())
        return o

    def preview(self):
        it = self.lst.currentItem()
        if it is None:
            return
        path = it.data(ID_ROLE)
        self._gen += 1
        gen = self._gen
        self.lbl.setText(f"Reading {os.path.basename(path)}…")

        def ok(pv):
            if gen != self._gen:
                return
            self.previews[path] = pv
            it.setText(f"{_base(path)}  —  {pv['reader_label']}, {pv['n']} spectra")
            acq = ", ".join(f"{ACQ_LABEL.get(k, k)} {v}" for k, v in pv["acquisitions"].items())
            cats = pv["categories"]
            txt = [f"<b>{_esc(pv['reader_label'])}</b>: {pv['n']} spectra" +
                   (f" (of {pv['found']} found)" if pv["found"] != pv["n"] else ""),
                   f"Wavelengths {pv['wl_min']:g}–{pv['wl_max']:g} nm, read as <b>{pv['wl_unit']}</b>"
                   if pv["wl_min"] is not None else "No spectra",
                   f"Values: <b>{_esc(pv['scale'])}</b>",
                   f"Categories: {len(cats)}" + (f" (e.g. {_esc(', '.join(_crumb(c) for c in cats[:3]))})" if cats else
                                                 " — none in the file; use 'Category if none'"),
                   f"Acquisition: {_esc(acq)}", f"Dataset: {_esc(pv['dataset'])}"]
            txt += [f"<span style='color:#d29922'>⚠ {_esc(w)}</span>" for w in pv["warnings"][:5]]
            self.lbl.setText("<br>".join(txt))
            ax = self.canvas.reset()
            for k, sp in enumerate(pv["sample"][:10]):
                y = np.array([np.nan if v is None else v for v in sp["values"]], float)
                ax.plot(sp["wavelengths"], y, color=PALETTE[k % 10], lw=1.1, label=sp["name"][:30])
            self.canvas.finish(f"Preview ({min(10, len(pv['sample']))} of {pv['n']})", "Wavelength (nm)",
                               "Value", len(pv["sample"]) <= 10)
            self.tbl.setRowCount(len(pv["sample"]))
            for r, sp in enumerate(pv["sample"]):
                self.tbl.setItem(r, 0, W.QTableWidgetItem(sp["name"]))
                self.tbl.setItem(r, 1, W.QTableWidgetItem(_crumb(sp["category"]) or "—"))
                self.tbl.setItem(r, 2, W.QTableWidgetItem(ACQ_LABEL.get(sp["acquisition"] or "untagged", "")))
                self.tbl.setItem(r, 3, _num_item(sp["points"]))
            if not self.ed_ds.text().strip() and self.lst.count() == 1:
                self.ed_ds.setPlaceholderText(pv["dataset"])

        def bad(msg):
            if gen != self._gen:
                return
            self.previews[path] = None
            it.setText(f"{_base(path)}  —  ⚠ cannot read")
            self.lbl.setText(f"<span style='color:#d29922'>⚠ {_esc(msg)}</span><br>Try another table layout, "
                             "wavelength unit or a wavelength column name, or remove this source.")
            self.canvas.message("Nothing to preview")
            self.tbl.setRowCount(0)
        self.panel.run(self.panel.client.preview_file, path, self.options(), 12, done=ok, failed=bad)

    def _accept(self):
        if self.lst.count() == 0:
            return
        if self.rb_new.isChecked() and not self.ed_newlib.text().strip():
            self.lbl.setText("Enter a name for the new library.")
            return
        self.accept()

    def jobs(self):
        out, opts = [], self.options()
        for i in range(self.lst.count()):
            p = self.lst.item(i).data(ID_ROLE)
            j = {"path": p, "options": opts, "dataset": self.ed_ds.text().strip() or None}
            if self.rb_exist.isChecked():
                j["library_id"] = self.cb_lib.currentData()
            elif self.rb_new.isChecked():
                j["library_name"] = self.ed_newlib.text().strip()
            else:
                j["library_name"] = re.sub(r"[_\s]+", " ", Path(p).stem if Path(p).is_file() else Path(p).name).strip().title()
            out.append(j)
        return out


class DataFolderDialog(W.QDialog):
    """Scan a local folder for spectral data, show what is new, pick libraries, import in one go."""

    def __init__(self, panel, folder, auto, library_names):
        super().__init__(panel)
        self.panel = panel
        self.lib_names = library_names
        self.sources = []
        self.setWindowTitle("Import from data folder")
        self.ed_folder = W.QLineEdit(folder)
        self.ed_folder.setPlaceholderText(r"e.g. D:\speclib  - downloaded .parquet / .zip / .csv / ECOSTRESS folders")
        b_browse = W.QPushButton("Browse...")
        b_browse.clicked.connect(self._browse)
        self.b_scan = W.QPushButton("Scan")
        self.b_scan.clicked.connect(self.scan)
        self.ed_folder.returnPressed.connect(self.scan)
        self.chk_auto = W.QCheckBox("Check this folder for new files when HyperVista starts (asks before importing)")
        self.chk_auto.setChecked(auto)
        self.tbl = W.QTableWidget(0, 5)
        self.tbl.setHorizontalHeaderLabels(["Source", "Type", "Import into library", "Status", "Notes"])
        h = self.tbl.horizontalHeader()
        h.setSectionResizeMode(0, W.QHeaderView.ResizeToContents)
        h.setSectionResizeMode(1, W.QHeaderView.ResizeToContents)
        h.setSectionResizeMode(2, W.QHeaderView.Interactive)
        h.setSectionResizeMode(3, W.QHeaderView.Stretch)
        h.setSectionResizeMode(4, W.QHeaderView.Interactive)
        self.tbl.setColumnWidth(2, 210)
        self.tbl.setColumnWidth(4, 150)
        self.tbl.verticalHeader().setVisible(False)
        self.tbl.setSelectionMode(W.QAbstractItemView.NoSelection)
        self.lbl = W.QLabel("Choose a folder and press Scan.")
        self.lbl.setWordWrap(True)
        self.cb_acq = W.QComboBox()
        self.cb_acq.addItem("Automatic (from each source)", None)
        for t in ACQ:
            self.cb_acq.addItem(_dot(ACQ_COLOR[t]), f"All {ACQ_LABEL[t].lower()}", t)
        self.sp_lim = W.QSpinBox()
        self.sp_lim.setRange(0, 1000000)
        self.sp_lim.setSpecialValueText("all")
        b_all = W.QPushButton("Select new")
        b_all.clicked.connect(lambda: self._check(lambda s: s["status"] == "new"))
        b_none = W.QPushButton("Select none")
        b_none.clicked.connect(lambda: self._check(lambda s: False))
        opts = W.QHBoxLayout()
        opts.addWidget(W.QLabel("Acquisition:"))
        opts.addWidget(self.cb_acq)
        opts.addSpacing(12)
        opts.addWidget(W.QLabel("At most per source:"))
        opts.addWidget(self.sp_lim)
        opts.addStretch(1)
        opts.addWidget(b_all)
        opts.addWidget(b_none)
        self.bb = W.QDialogButtonBox(W.QDialogButtonBox.Ok | W.QDialogButtonBox.Close)
        self.bb.button(W.QDialogButtonBox.Ok).setText("Import selected")
        self.bb.button(W.QDialogButtonBox.Ok).setEnabled(False)
        self.bb.accepted.connect(self.accept)
        self.bb.rejected.connect(self.reject)
        top = W.QHBoxLayout()
        top.addWidget(W.QLabel("Folder:"))
        top.addWidget(self.ed_folder, 1)
        top.addWidget(b_browse)
        top.addWidget(self.b_scan)
        lay = W.QVBoxLayout(self)
        lay.addLayout(top)
        lay.addWidget(self.chk_auto)
        lay.addWidget(self.tbl, 1)
        lay.addWidget(self.lbl)
        lay.addLayout(opts)
        lay.addWidget(_hint("Every supported format is recognised: CSV/TSV/TXT, Excel, JSON, Parquet, ENVI spectral "
                            "libraries and images, MAT/HDF5/NetCDF/NumPy, ASD/SVC/Spectral Evolution files (a folder of "
                            "them is one source, subfolders → categories), USGS splib07, ECOSTRESS, OpenSpecLib, zips. "
                            "Libraries that do not exist yet are created; already-imported sources are recognised by "
                            "checksum. To change units, layout, category or images mode for one file, use "
                            "Import ▸ Spectral data files instead."))
        lay.addWidget(self.bb)
        self.resize(1100, 520)
        if folder and os.path.isdir(folder):
            QtCore.QTimer.singleShot(0, self.scan)

    def folder(self):
        return self.ed_folder.text().strip()

    def auto_import(self):
        return self.chk_auto.isChecked()

    def _browse(self):
        d = W.QFileDialog.getExistingDirectory(self, "Data folder", self.folder() or "")
        if d:
            self.ed_folder.setText(d)
            self.scan()

    def scan(self):
        folder = self.folder()
        if not folder or not os.path.isdir(folder):
            self.lbl.setText("That folder does not exist.")
            return
        self.lbl.setText("Scanning and checking against the database… (large files are checksummed once)")
        self.b_scan.setEnabled(False)
        self.tbl.setRowCount(0)

        def ok(srcs):
            self.b_scan.setEnabled(True)
            self.sources = srcs
            self._fill()

        def bad(msg):
            self.b_scan.setEnabled(True)
            self.lbl.setText(f"Scan failed: {msg}")
        self.panel.run(self.panel.client.scan_folder, folder, done=ok, failed=bad)

    def _fill(self):
        self.tbl.setRowCount(len(self.sources))
        for r, s in enumerate(self.sources):
            it = W.QTableWidgetItem(s["name"])
            it.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            it.setCheckState(Qt.Checked if s["status"] == "new" else Qt.Unchecked)
            it.setToolTip(s["path"])
            if s["status"] == "error":
                it.setFlags(Qt.ItemIsEnabled)
            self.tbl.setItem(r, 0, it)
            self.tbl.setItem(r, 1, W.QTableWidgetItem(s["kind_label"]))
            cb = W.QComboBox()
            cb.setEditable(True)
            names = sorted(set(self.lib_names) | {s["library"]}, key=str.lower)
            cb.addItems(names)
            cb.setCurrentText(s["library"])
            self.tbl.setCellWidget(r, 2, cb)
            if s["status"] == "imported":
                st = W.QTableWidgetItem("✓ imported → " + "; ".join(s["imported_into"]))
                st.setForeground(QtGui.QBrush(QtGui.QColor("#3fb950")))
            elif s["status"] == "error":
                st = W.QTableWidgetItem("⚠ cannot import")
                st.setForeground(QtGui.QBrush(QtGui.QColor("#d29922")))
            else:
                st = W.QTableWidgetItem("● new")
                st.setForeground(QtGui.QBrush(QtGui.QColor("#5b8def")))
            st.setToolTip(st.text())
            self.tbl.setItem(r, 3, st)
            size = f"{s['size'] / 1e6:.1f} MB" if s.get("size") else ""
            self.tbl.setItem(r, 4, W.QTableWidgetItem("  ".join(x for x in (size, s.get("note") or "") if x)))
        n_new = sum(s["status"] == "new" for s in self.sources)
        self.lbl.setText(f"{len(self.sources)} source(s) found, {n_new} new." if self.sources else
                         "No supported spectral data found in this folder.")
        self.bb.button(W.QDialogButtonBox.Ok).setEnabled(bool(self.sources))

    def _check(self, pred):
        for r, s in enumerate(self.sources):
            it = self.tbl.item(r, 0)
            if it.flags() & Qt.ItemIsUserCheckable:
                it.setCheckState(Qt.Checked if pred(s) else Qt.Unchecked)

    def items(self):
        out = []
        acq, lim = self.cb_acq.currentData(), self.sp_lim.value() or None
        for r, s in enumerate(self.sources):
            if self.tbl.item(r, 0).checkState() != Qt.Checked:
                continue
            lib = self.tbl.cellWidget(r, 2).currentText().strip() or s["library"]
            it = {"source": s, "library": lib}
            if acq:
                it["acquisition"] = acq
            if lim:
                it["limit"] = lim
            out.append(it)
        return out


class ConnectDialog(W.QDialog):
    def __init__(self, parent, mode, db_url, api_url, autoconnect):
        super().__init__(parent)
        self.setWindowTitle("Connect to the material library database")
        self.rb_direct = W.QRadioButton("Direct database connection (PostgreSQL, no server needed)")
        self.rb_api = W.QRadioButton("Through the library API server (FastAPI)")
        (self.rb_api if mode == "api" else self.rb_direct).setChecked(True)
        self.ed_db = W.QLineEdit(db_url)
        self.ed_db.setPlaceholderText("postgresql://user:password@host:5432/material_library")
        self.ed_api = W.QLineEdit(api_url)
        self.ed_api.setPlaceholderText("http://127.0.0.1:8000")
        self.chk_init = W.QCheckBox("Create the library tables if the database is empty (runs schema.sql)")
        self.chk_auto = W.QCheckBox("Connect automatically when HyperVista starts")
        self.chk_auto.setChecked(autoconnect)
        hint = W.QLabel("Start the database with  <code>docker compose -f database/docker-compose.yml up -d</code>, "
                        "load data with <code>python -m hvva.library.importer data/material_spectra.csv</code>; "
                        "API server: <code>python -m hvva.library.api</code>. sqlite:///file.db also works offline. "
                        "Older databases get the acquisition column added automatically on connect.")
        hint.setWordWrap(True)
        hint.setTextFormat(Qt.RichText)
        form = W.QFormLayout()
        form.addRow(self.rb_direct)
        form.addRow("Database URL", self.ed_db)
        form.addRow(self.rb_api)
        form.addRow("API URL", self.ed_api)
        form.addRow(self.chk_init)
        form.addRow(self.chk_auto)
        lay = W.QVBoxLayout(self)
        lay.addLayout(form)
        lay.addWidget(hint)
        bb = W.QDialogButtonBox(W.QDialogButtonBox.Ok | W.QDialogButtonBox.Cancel)
        bb.button(W.QDialogButtonBox.Ok).setText("Connect")
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)
        self.rb_direct.toggled.connect(self._sync)
        self._sync()
        self.resize(640, 0)

    def _sync(self):
        d = self.rb_direct.isChecked()
        self.ed_db.setEnabled(d)
        self.chk_init.setEnabled(d)
        self.ed_api.setEnabled(not d)

    def choices(self):
        mode = "direct" if self.rb_direct.isChecked() else "api"
        return {"mode": mode, "db_url": self.ed_db.text().strip(), "api_url": self.ed_api.text().strip(),
                "init": self.chk_init.isChecked(), "autoconnect": self.chk_auto.isChecked()}


class SaveDialog(W.QDialog):
    """Pick target library (or a new one), dataset, category path and acquisition for spectra saved from HyperVista."""
    NEW = "__new__"

    def __init__(self, parent, libraries, current_id, what, default_name=None, paths=(), category=None,
                 acquisition=None):
        super().__init__(parent)
        self.setWindowTitle("Save to library database")
        self.cb_lib = W.QComboBox()
        for l in libraries:
            self.cb_lib.addItem(l["name"], l["id"])
        self.cb_lib.addItem("New library...", self.NEW)
        i = self.cb_lib.findData(current_id) if current_id else -1
        self.cb_lib.setCurrentIndex(i if i >= 0 else 0)
        self.ed_newlib = W.QLineEdit()
        self.ed_newlib.setPlaceholderText("name of the new library")
        self.ed_ds = W.QLineEdit("HyperVista signatures")
        self.cb_cat = _category_combo(list(paths), category)
        self.cb_acq = _acq_combo()
        self.cb_acq.setCurrentIndex(max(0, self.cb_acq.findData(acquisition or repo.UNTAGGED)))
        self.ed_name = W.QLineEdit(default_name or "")
        form = W.QFormLayout(self)
        form.addRow(W.QLabel(what))
        form.addRow("Library", self.cb_lib)
        form.addRow("New library name", self.ed_newlib)
        form.addRow("Dataset", self.ed_ds)
        if default_name is not None:
            form.addRow("Material name", self.ed_name)
        form.addRow("Category", self.cb_cat)
        form.addRow("", _hint(CATEGORY_HINT))
        form.addRow("Acquisition", self.cb_acq)
        form.addRow(_hint("Scene spectra are usually airborne or satellite. A material name that already exists in "
                          "the library gets the spectrum as an additional sample."))
        bb = W.QDialogButtonBox(W.QDialogButtonBox.Save | W.QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        form.addRow(bb)
        self.cb_lib.currentIndexChanged.connect(self._sync)
        self._sync()
        self.resize(520, 0)

    def _sync(self):
        self.ed_newlib.setEnabled(self.cb_lib.currentData() == self.NEW)

    def choices(self):
        a = self.cb_acq.currentData()
        return {"library_id": self.cb_lib.currentData(), "new_library": self.ed_newlib.text().strip(),
                "dataset": self.ed_ds.text().strip() or "HyperVista signatures",
                "category": self.cb_cat.currentText().strip() or None, "name": self.ed_name.text().strip(),
                "acquisition": None if a == repo.UNTAGGED else a}


# ============================================================ the panel
class LibraryPanel(W.QWidget):
    """Library tab. *host* is the main window (plot_canvas, say(), _err(), last_spectrum, progress)."""

    def __init__(self, host):
        super().__init__()
        self.host = host
        self.client = None
        self.pool = QtCore.QThreadPool()
        self.pool.setMaxThreadCount(4)
        self._tasks = set()
        self.libraries = []
        self.active_lib = None              # library dict
        self.facets = None                  # library_facets() of the active library
        self.cat_path = None                # selected tree node: None = all, path, or repo.UNCATEGORISED
        self.materials = []                 # loaded page(s) of the active library
        self.mat_total = 0
        self._mat_gen = 0                   # request generations -> drop superseded results
        self._glob_gen = 0
        self._facet_gen = 0
        self.active_material = None
        self.samples = []
        self.spectrum = None
        self.compare = []                   # [{'id','name','library','spectrum'}]
        s = QtCore.QSettings(SETTINGS_ORG, SETTINGS_APP)
        self.cfg = {"mode": str(s.value("library/mode", "direct")),
                    "db_url": str(s.value("library/db_url", default_url())),
                    "api_url": str(s.value("library/api_url", "http://127.0.0.1:8000")),
                    "autoconnect": str(s.value("library/autoconnect", "true")).lower() == "true",
                    "data_folder": str(s.value("library/data_folder", "")),
                    "auto_import": str(s.value("library/auto_import", "false")).lower() == "true"}
        self._build()
        self._set_connected(False)
        if self.cfg["autoconnect"]:
            QtCore.QTimer.singleShot(300, lambda: self.connect_db(quiet=True))

    # ------------------------------------------------------------ async helper
    def run(self, fn, *args, done=None, failed=None, progress=None, **kwargs):
        t = _Task(fn, args, kwargs, progress is not None)
        self._tasks.add(t)

        def fin(*_):
            self._tasks.discard(t)
        t.sig.done.connect(fin)
        t.sig.failed.connect(fin)
        if done:
            t.sig.done.connect(done)
        t.sig.failed.connect(failed or self._fail)
        if progress:
            t.sig.progress.connect(progress)
        self.pool.start(t)

    def _fail(self, msg):
        self.host._err(msg)

    def _need_client(self):
        if self.client is None:
            self.host._err("not connected to the library database - press Connect...")
            return False
        return True

    @property
    def acq_filter(self):
        return self.cb_acqf.currentData()

    def category_paths(self):
        return [c["path"] for c in (self.facets or {}).get("categories", [])]

    # ------------------------------------------------------------ UI
    def _build(self):
        self.setStyleSheet(PANEL_QSS)
        # --- search bar (labels above the fields, like the DS3 web module) -------------------
        self.ed_libsearch = W.QLineEdit()
        self.ed_libsearch.setPlaceholderText("e.g. USGS")
        self.ed_libsearch.setProperty("role", "search")
        self.ed_libsearch.setClearButtonEnabled(True)
        self.ed_libsearch.textChanged.connect(self._fill_libraries)
        self.ed_global = W.QLineEdit()
        self.ed_global.setPlaceholderText("e.g. Basalt")
        self.ed_global.setProperty("role", "search")
        self.ed_global.setClearButtonEnabled(True)
        self.ed_global.setToolTip("Searches every library (min 2 characters); pick a result to open it")
        self.ed_global.textEdited.connect(self.on_global_text)
        self.ed_global.returnPressed.connect(self._global_search)
        self._glob_model = QtGui.QStandardItemModel(self)
        self._glob_comp = W.QCompleter(self._glob_model, self)
        self._glob_comp.setCompletionMode(W.QCompleter.UnfilteredPopupCompletion)
        self._glob_comp.setCompletionRole(Qt.UserRole + 9)          # inserting a result keeps the typed text
        self._glob_comp.setMaxVisibleItems(12)
        self._glob_comp.activated[QtCore.QModelIndex].connect(self.on_global_open)
        self.ed_global.setCompleter(self._glob_comp)
        self._glob_timer = QtCore.QTimer(self, singleShot=True, interval=300, timeout=self._global_search)
        self.cb_acqf = _acq_combo(include_all="All acquisition types")
        self.cb_acqf.setProperty("role", "search")
        self.cb_acqf.setToolTip("Show only materials with a spectrum of this acquisition type "
                                "(also applies to the all-materials search and to matching)")
        self.cb_acqf.currentIndexChanged.connect(self.on_acq_filter)
        self.lbl_conn = W.QLabel()
        self.lbl_conn.setSizePolicy(W.QSizePolicy.Ignored, W.QSizePolicy.Preferred)
        self.lbl_conn.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        b_conn = W.QPushButton("Connect...")
        b_conn.clicked.connect(self.on_connect_dialog)
        self.b_refresh = W.QPushButton("Refresh")
        self.b_refresh.clicked.connect(self.refresh_all)
        for b in (b_conn, self.b_refresh):
            b.setSizePolicy(W.QSizePolicy.Maximum, W.QSizePolicy.Fixed)
        bar = W.QHBoxLayout()
        bar.setSpacing(14)
        bar.addLayout(_labeled("Search libraries by name", self.ed_libsearch), 3)
        bar.addLayout(_labeled("Search all materials by name", self.ed_global), 3)
        bar.addLayout(_labeled("Acquisition", self.cb_acqf), 2)
        right = W.QVBoxLayout()
        right.setSpacing(2)
        right.addWidget(self.lbl_conn)
        rb = W.QHBoxLayout()
        rb.addStretch(1)
        rb.addWidget(b_conn)
        rb.addWidget(self.b_refresh)
        right.addLayout(rb)
        bar.addLayout(right, 2)

        # --- pane 1: libraries ----------------------------------------------------------------
        self.tbl_libs = W.QTableWidget(0, 2)
        self.tbl_libs.setHorizontalHeaderLabels(["Library", "Materials"])
        _table_defaults(self.tbl_libs, stretch_first=True)
        self.tbl_libs.itemSelectionChanged.connect(self.on_library_selected)
        self.b_addlib = W.QPushButton("+ Add")
        self.b_addlib.clicked.connect(self.on_add_library)
        self.b_dellib = W.QPushButton("Delete")
        self.b_dellib.clicked.connect(self.on_delete_library)
        self.b_import = W.QPushButton("Import")
        self.b_import.setToolTip("Import spectral data in any supported format")
        imp = W.QMenu(self.b_import)
        a_files = imp.addAction("Spectral data files (any format)...")
        f_ = a_files.font()
        f_.setBold(True)
        a_files.setFont(f_)
        a_files.triggered.connect(lambda: self.on_import_files("files"))
        imp.addAction("A folder or zip of spectral files...").triggered.connect(lambda: self.on_import_files("folder"))
        imp.addSeparator()
        imp.addAction("From data folder (scan && import everything new)...").triggered.connect(self.on_data_folder)
        self.b_import.setMenu(imp)
        g_libs = _group("Libraries", self.tbl_libs, _row(self.b_addlib, self.b_dellib), self.b_import)

        # --- pane 2: category tree ------------------------------------------------------------
        self.tree = W.QTreeWidget()
        self.tree.setHeaderLabels(["Category", "#"])
        self.tree.setRootIsDecorated(True)
        self.tree.setUniformRowHeights(True)
        self.tree.header().setStretchLastSection(False)
        self.tree.header().setSectionResizeMode(0, W.QHeaderView.Stretch)
        self.tree.header().setSectionResizeMode(1, W.QHeaderView.ResizeToContents)
        self.tree.itemSelectionChanged.connect(self.on_category_selected)
        self.tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._tree_menu)
        self.b_rencat = W.QPushButton("Rename / move...")
        self.b_rencat.setToolTip("Rename the selected category; every category below it moves along")
        self.b_rencat.clicked.connect(self.on_rename_category)
        self.g_tree = _group("Category tree", self.tree, self.b_rencat)

        # --- pane 3: materials ----------------------------------------------------------------
        self.lbl_crumb = W.QLabel("")
        self.lbl_crumb.setProperty("role", "crumb")
        self.lbl_crumb.setSizePolicy(W.QSizePolicy.Ignored, W.QSizePolicy.Preferred)
        self.ed_matsearch = W.QLineEdit()
        self.ed_matsearch.setPlaceholderText("Search in this library (min 2 characters)")
        self.ed_matsearch.setClearButtonEnabled(True)
        self.ed_matsearch.textChanged.connect(self.on_material_text)
        self._mat_timer = QtCore.QTimer(self, singleShot=True, interval=300, timeout=lambda: self.fetch_materials(False))
        self.lbl_count = W.QLabel("Select a library.")
        self.tbl_mats = W.QTableWidget(0, 3)
        self.tbl_mats.setHorizontalHeaderLabels(["Material", "Category", "Acquisition"])
        _table_defaults(self.tbl_mats, stretch_first=True)
        self.tbl_mats.setSelectionMode(W.QAbstractItemView.ExtendedSelection)
        self.tbl_mats.setItemDelegateForColumn(2, TagDelegate(self.tbl_mats))
        self.tbl_mats.itemSelectionChanged.connect(self.on_material_selected)
        self.tbl_mats.doubleClicked.connect(lambda *_: self.on_edit_materials())
        self.b_more = W.QPushButton("See more")
        self.b_more.clicked.connect(lambda: self.fetch_materials(True))
        self.b_addmat = W.QPushButton("+ Add")
        self.b_addmat.clicked.connect(self.on_add_material)
        self.b_editmat = W.QPushButton("Edit...")
        self.b_editmat.setToolTip("Category and acquisition type of the selected material(s) - double-click a row too")
        self.b_editmat.clicked.connect(self.on_edit_materials)
        self.b_delmat = W.QPushButton("Delete")
        self.b_delmat.clicked.connect(self.on_delete_material)
        self.grp_mats = _group("Materials", self.lbl_crumb, self.ed_matsearch, self.lbl_count, self.tbl_mats,
                               _row(self.b_more, self.b_addmat, self.b_editmat, self.b_delmat))

        # --- pane 4: spectrum / compare / match --------------------------------------------------
        inner = W.QTabWidget()
        self.lbl_mat = W.QLabel("Open a material to see its spectrum in the Spectra pane.")
        self.lbl_mat.setWordWrap(True)
        self.cb_sample = W.QComboBox()
        self.cb_sample.currentIndexChanged.connect(self.on_sample_changed)
        self.cb_acq = _acq_combo()
        self.cb_acq.setToolTip("How this spectrum was measured - saved to the database immediately")
        self.cb_acq.currentIndexChanged.connect(self.on_spectrum_acq_changed)
        self.b_csv = W.QPushButton("Export CSV...")
        self.b_csv.clicked.connect(self.on_export_spectrum)
        self.b_cmp = W.QPushButton("Add to comparison")
        self.b_cmp.clicked.connect(self.on_toggle_compare)
        f = W.QFormLayout()
        f.addRow(self.lbl_mat)
        f.addRow("Sample", self.cb_sample)
        f.addRow("Acquisition", self.cb_acq)
        f.addRow(_row(self.b_csv, self.b_cmp))
        inner.addTab(_page(f), "Spectrum")
        self.lst_cmp = W.QListWidget()
        self.lst_cmp.setSelectionMode(W.QAbstractItemView.ExtendedSelection)
        b_plot = W.QPushButton("Plot")
        b_plot.clicked.connect(self.plot_comparison)
        b_rm = W.QPushButton("Remove")
        b_rm.clicked.connect(self.on_compare_remove)
        b_clr = W.QPushButton("Clear")
        b_clr.clicked.connect(self.on_compare_clear)
        b_cexp = W.QPushButton("Export...")
        b_cexp.setToolTip("CSV, JSON and ENVI spectral library (.sli) of the compared spectra")
        b_cexp.clicked.connect(self.on_compare_export)
        v = W.QVBoxLayout()
        v.addWidget(self.lst_cmp)
        v.addLayout(_row(b_plot, b_rm, b_clr, b_cexp))
        inner.addTab(_page(v), "Compare")
        self.cb_scope = W.QComboBox()
        self.cb_scope.addItems(["Selected library", "All libraries"])
        self.chk_matchcat = W.QCheckBox("Only the selected category")
        self.sp_top = W.QSpinBox()
        self.sp_top.setRange(1, 50)
        self.sp_top.setValue(5)
        b_match = W.QPushButton("Match last clicked pixel")
        b_match.clicked.connect(self.on_match)
        b_savepx = W.QPushButton("Save clicked pixel...")
        b_savepx.clicked.connect(self.on_save_pixel)
        self.txt_match = W.QPlainTextEdit("Click a pixel in the scene view, then match it against the library "
                                          "(spectral angle on the overlapping wavelengths). The Acquisition filter "
                                          "above limits which reference spectra are used.")
        self.txt_match.setReadOnly(True)
        f2 = W.QFormLayout()
        f2.addRow("Match against", _row(self.cb_scope, self.sp_top))
        f2.addRow(self.chk_matchcat)
        f2.addRow(_row(b_match, b_savepx))
        f2.addRow(self.txt_match)
        inner.addTab(_page(f2), "Match / Save pixel")
        self.inner = inner
        g_spec = _group("Spectrum, comparison and matching", inner)

        split = W.QSplitter(Qt.Horizontal)
        for w_, stretch in ((g_libs, 2), (self.g_tree, 2), (self.grp_mats, 3), (g_spec, 3)):
            split.addWidget(w_)
            split.setStretchFactor(split.count() - 1, stretch)
        split.setChildrenCollapsible(False)
        split.setSizes([230, 240, 400, 400])
        lay = W.QVBoxLayout(self)
        lay.setContentsMargins(6, 4, 6, 4)
        lay.setSpacing(4)
        lay.addLayout(bar)
        lay.addWidget(split, 1)
        self._sync_buttons()

    def _sync_buttons(self):
        on = self.client is not None
        lib = self.active_lib is not None
        for b in (self.b_refresh, self.b_addlib, self.ed_libsearch, self.ed_global, self.cb_acqf, self.b_import):
            b.setEnabled(on)
        for b in (self.b_dellib, self.ed_matsearch, self.b_addmat, self.tree):
            b.setEnabled(on and lib)
        sel = self._selected_materials()
        self.b_delmat.setEnabled(on and self.active_material is not None)
        self.b_editmat.setEnabled(on and bool(sel))
        self.b_editmat.setText(f"Edit {len(sel)}..." if len(sel) > 1 else "Edit...")
        node = self._tree_node_path()
        self.b_rencat.setEnabled(on and lib and node not in (None, repo.UNCATEGORISED))
        self.b_more.setVisible(len(self.materials) < self.mat_total)
        has = self.spectrum is not None
        for b in (self.b_csv, self.b_cmp, self.cb_acq):
            b.setEnabled(has)
        if self.active_material is not None:
            inside = any(c["id"] == self.active_material["id"] for c in self.compare)
            self.b_cmp.setText("Remove from comparison" if inside else "Add to comparison")

    def _set_connected(self, ok, text=None):
        dot = "#3fb950" if ok else "#d29922"
        self.lbl_conn.setText(f"<span style='color:{dot}'>●</span> " + (text or ("Connected" if ok else "Not connected")))
        self._sync_buttons()

    # ------------------------------------------------------------ connection
    def on_connect_dialog(self):
        d = ConnectDialog(self, self.cfg["mode"], self.cfg["db_url"], self.cfg["api_url"], self.cfg["autoconnect"])
        if not d.exec_():
            return
        ch = d.choices()
        self.cfg.update({k: ch[k] for k in ("mode", "db_url", "api_url", "autoconnect")})
        s = QtCore.QSettings(SETTINGS_ORG, SETTINGS_APP)
        for k in ("mode", "db_url", "api_url"):
            s.setValue(f"library/{k}", self.cfg[k])
        s.setValue("library/autoconnect", "true" if self.cfg["autoconnect"] else "false")
        self.connect_db(init=ch["init"])

    def connect_db(self, init=False, quiet=False):
        mode = self.cfg["mode"]
        url = self.cfg["api_url"] if mode == "api" else self.cfg["db_url"]
        self._set_connected(False, "Connecting...")

        def ok(client):
            if self.client is not None:
                try:
                    self.client.close()
                except Exception:                                               # noqa: BLE001
                    pass
            self.client = client
            self._set_connected(True, _short(client))
            self.lbl_conn.setToolTip(client.describe())
            self.host.say(f"Library database connected ({client.describe()}).")
            self.load_libraries()
            if self.cfg["auto_import"] and self.cfg["data_folder"] and client.mode == "direct":
                self.auto_import()

        def bad(msg):
            self.client = None
            self._set_connected(False, "Not connected")
            self.lbl_conn.setToolTip(msg)
            if quiet:
                self.host.statusBar().showMessage(f"Library database not connected: {msg}", 8000)
            else:
                self.host._err(f"library database: {msg}")
        self.run(make_client, mode, url, init, done=ok, failed=bad)

    def shutdown(self):
        self.pool.waitForDone(3000)
        if self.client is not None:
            try:
                self.client.close()
            except Exception:                                                   # noqa: BLE001
                pass

    def refresh_all(self):
        self.load_libraries()
        if self.active_lib:
            self.load_facets()
            self.fetch_materials(False)

    # ------------------------------------------------------------ libraries
    def load_libraries(self, select_id=None):
        if not self._need_client():
            return
        keep = select_id or (self.active_lib or {}).get("id")

        def ok(libs):
            self.libraries = libs or []
            self._fill_libraries(select_id=keep)
            self.host.statusBar().showMessage(f"{len(self.libraries)} libraries in the database", 4000)
        self.run(self.client.list_libraries, done=ok)

    def _fill_libraries(self, *_, select_id=None):
        q = self.ed_libsearch.text().strip().lower()
        rows = [l for l in self.libraries if q in str(l.get("name", "")).lower()]
        select_id = select_id or (self.active_lib or {}).get("id")
        self.tbl_libs.blockSignals(True)
        self.tbl_libs.clearSelection()
        self.tbl_libs.setRowCount(len(rows))
        sel = -1
        for r, l in enumerate(rows):
            it = W.QTableWidgetItem(l["name"])
            it.setData(ID_ROLE, l["id"])
            it.setToolTip(f"{l['name']}\n{l.get('description') or 'Spectral data library'}\n"
                          f"{l.get('dataset_count', '?')} dataset(s), slug: {l.get('slug', '')}")
            self.tbl_libs.setItem(r, 0, it)
            n = l.get("material_count")
            self.tbl_libs.setItem(r, 1, _num_item("" if n is None else n))
            if l["id"] == select_id:
                sel = r
            if self.active_lib is not None and l["id"] == self.active_lib["id"]:
                self.active_lib = l
        self.tbl_libs.blockSignals(False)
        if sel >= 0:
            self.tbl_libs.selectRow(sel)
        elif select_id is None:
            self._clear_library()

    def _is_active(self, lib):
        return self.active_lib is not None and self.active_lib["id"] == lib["id"]

    def _lib_by_id(self, lid):
        return next((l for l in self.libraries if l["id"] == lid), None)

    def on_library_selected(self):
        r = self.tbl_libs.currentRow()
        it = self.tbl_libs.item(r, 0) if r >= 0 else None
        lib = self._lib_by_id(it.data(ID_ROLE)) if it else None
        if lib is None or (self._is_active(lib) and self.facets is not None):
            return
        self.open_library(lib)

    def open_library(self, lib, then=None):
        self.active_lib = lib
        self.cat_path = None
        self.facets = None
        self.tree.clear()
        self.g_tree.setTitle(f"Category tree — {lib['name']}")
        self.grp_mats.setTitle(f"Materials — {lib['name']}")
        self.ed_matsearch.blockSignals(True)
        self.ed_matsearch.clear()
        self.ed_matsearch.blockSignals(False)
        self.materials, self.mat_total = [], 0
        self.tbl_mats.setRowCount(0)
        self.active_material, self.samples, self.spectrum = None, [], None
        self.cb_sample.blockSignals(True)
        self.cb_sample.clear()
        self.cb_sample.blockSignals(False)
        self._show_material_info()
        self._update_crumb()
        self.load_facets()
        self.fetch_materials(False, then=then)
        self._sync_buttons()

    def _clear_library(self):
        self.active_lib = None
        self.facets = None
        self.cat_path = None
        self.tree.clear()
        self.materials, self.mat_total = [], 0
        self.tbl_mats.setRowCount(0)
        self.g_tree.setTitle("Category tree")
        self.grp_mats.setTitle("Materials")
        self.lbl_count.setText("Select a library.")
        self._update_crumb()
        self._sync_buttons()

    def on_add_library(self):
        if not self._need_client():
            return
        v = _form_dialog(self, "Add library", [("name", "Name", "", "e.g. USGS Splib07"),
                                               ("description", "Description (optional)", "", "e.g. Custom rooftop samples")])
        if not v:
            return
        if not v["name"]:
            return self.host._err("library name is required")

        def ok(lib):
            self.host.say(f"Library \"{lib['name']}\" created.")
            self.load_libraries(select_id=lib["id"])
        self.run(self.client.create_library, v["name"], v["description"] or None, done=ok)

    def on_delete_library(self):
        lib = self.active_lib
        if not lib or not self._need_client():
            return
        n = lib.get("material_count", "all")
        if W.QMessageBox.question(self, "Delete library",
                                  f"Delete library \"{lib['name']}\" with its {n} materials, samples, spectra and "
                                  f"datasets from the database?\nThis cannot be undone.") != W.QMessageBox.Yes:
            return

        def ok(_):
            self.host.say(f"Library \"{lib['name']}\" deleted.")
            self._clear_library()
            self.compare = [c for c in self.compare if c.get("library_id") != lib["id"]]
            self._fill_compare()
            self.load_libraries()
        self.run(self.client.delete_library, lib["id"], done=ok)

    # ------------------------------------------------------------ universal import
    def on_import_files(self, what="files", paths=None):
        if not self._need_client():
            return
        if self.client.mode != "direct":
            return self.host._err("importing local files needs a direct database connection (Connect... -> Direct)")
        if paths is None:
            start = QtCore.QSettings(SETTINGS_ORG, SETTINGS_APP).value("library/last_import_dir", "") or ""
            if what == "folder":
                d = W.QFileDialog.getExistingDirectory(self, "Folder of spectral files", str(start))
                paths = [d] if d else []
                if not paths:
                    z, _ = W.QFileDialog.getOpenFileName(self, "...or a zip of spectral files", str(start), "Zip (*.zip)")
                    paths = [z] if z else []
            else:
                paths, _ = W.QFileDialog.getOpenFileNames(self, "Spectral data files", str(start),
                                                          file_dialog_filter())
            if not paths:
                return
            QtCore.QSettings(SETTINGS_ORG, SETTINGS_APP).setValue("library/last_import_dir",
                                                                  os.path.dirname(paths[0].rstrip("/\\")))
        d = ImportWizard(self, paths)
        if d.exec_():
            jobs = d.jobs()
            if jobs:
                self._run_file_imports(jobs)

    def _run_file_imports(self, jobs):
        bar = self.host.progress
        bar.setVisible(True)
        bar.setRange(0, 0)
        c = self.client

        def go(progress=None):
            out = []
            for k, j in enumerate(jobs, 1):
                name = os.path.basename(j["path"].rstrip("/\\"))
                if progress:
                    progress(0, 0, f"[{k}/{len(jobs)}] {name}")
                try:
                    r = c.import_file(j["path"], library_id=j.get("library_id"), library_name=j.get("library_name"),
                                      dataset_name=j.get("dataset"), options=j["options"],
                                      progress=(lambda i, n, t, _k=k: progress(i, n, f"[{_k}/{len(jobs)}] {name}: {t}"))
                                      if progress else None)
                    out.append({"ok": True, **r})
                except Exception as e:                                      # noqa: BLE001
                    out.append({"ok": False, "source": name, "error": str(e).splitlines()[0][:300]})
            return out

        def prog(i, n, text):
            if n:
                bar.setRange(0, n)
                bar.setValue(i)
            self.host.statusBar().showMessage(f"Importing {text}")

        def ok(results):
            bar.setVisible(False)
            added = sum(r.get("spectra_added", 0) for r in results if r["ok"])
            lines = []
            for r in results:
                if r["ok"]:
                    lines.append(f"• {r['source']} → {r['library_name']} / {r['dataset_name']}: {r['spectra_added']} added"
                                 + (f", {r['skipped']} already there" if r.get("skipped") else "")
                                 + f"  [{r['detected']}]")
                    lines += [f"    ! {w}" for w in r.get("warnings", [])[:3]]
                else:
                    lines.append(f"• {r['source']}: FAILED – {r['error']}")
            self.host.say(f"Imported {added} spectra from {sum(r['ok'] for r in results)} of {len(results)} source(s).\n"
                          + "\n".join(lines))
            libs = [r for r in results if r["ok"]]
            if self.active_lib:
                self.load_facets()
                self.fetch_materials(False)
            self.load_libraries(select_id=libs[-1]["library_id"] if libs else None)

        def bad(msg):
            bar.setVisible(False)
            self.host._err(f"import failed: {msg}")
        self.run(go, done=ok, failed=bad, progress=prog)

    # ------------------------------------------------------------ local data folder
    def _save_folder_cfg(self):
        s = QtCore.QSettings(SETTINGS_ORG, SETTINGS_APP)
        s.setValue("library/data_folder", self.cfg["data_folder"])
        s.setValue("library/auto_import", "true" if self.cfg["auto_import"] else "false")

    def on_data_folder(self):
        if not self._need_client():
            return
        if self.client.mode != "direct":
            return self.host._err("importing local files needs a direct database connection (Connect... -> Direct)")
        d = DataFolderDialog(self, self.cfg["data_folder"], self.cfg["auto_import"], [l["name"] for l in self.libraries])
        ok = d.exec_()
        self.cfg["data_folder"], self.cfg["auto_import"] = d.folder(), d.auto_import()
        self._save_folder_cfg()
        if ok:
            items = d.items()
            if items:
                self._run_imports(items)

    def _run_imports(self, items, quiet=False):
        bar = self.host.progress
        bar.setVisible(True)
        bar.setRange(0, 0)

        def prog(i, n, text):
            if n > 1:
                bar.setRange(0, n)
                bar.setValue(i)
            if i <= 1 or i % 20 == 0 or i == n:
                self.host.statusBar().showMessage(f"Importing {text}")

        def ok(results):
            bar.setVisible(False)
            good = [r for r in results if r["ok"]]
            bad = [r for r in results if not r["ok"]]
            added = sum(r.get("spectra_added", 0) for r in good)
            lines = [f"• {r['source']} → {r['library_name']}: {r['spectra_added']} added, {r.get('skipped', 0)} already there"
                     for r in good] + [f"• {r['source']}: FAILED – {r['error']}" for r in bad]
            if quiet and not added and not bad:
                return
            self.host.say(f"Data folder import: {added} spectra added from {len(good)} source(s)"
                          + (f", {len(bad)} failed" if bad else "") + ".\n" + "\n".join(lines))
            if self.active_lib:
                self.load_facets()
                self.fetch_materials(False)
            self.load_libraries()

        def bad(msg):
            bar.setVisible(False)
            self.host._err(f"import failed: {msg}")
        self.run(self.client.import_sources, items, done=ok, failed=bad, progress=prog)

    def auto_import(self):
        """Startup: look for new files in the data folder and ASK before importing anything."""
        folder = self.cfg["data_folder"]
        if not folder or not os.path.isdir(folder):
            return

        def scanned(srcs):
            new = [s for s in srcs if s["status"] == "new"]
            if not new:
                return
            box = W.QMessageBox(self)
            box.setIcon(W.QMessageBox.Question)
            box.setWindowTitle("New spectral data found")
            listing = "\n".join(f"•  {s['name']}  →  library \"{s['library']}\"" for s in new[:12])
            more = f"\n… and {len(new) - 12} more" if len(new) > 12 else ""
            box.setText(f"{len(new)} new source(s) in {folder}:\n\n{listing}{more}\n\nImport them now?")
            box.setInformativeText("Spectra already in a library are skipped, so nothing is duplicated.")
            b_imp = box.addButton("Import", W.QMessageBox.AcceptRole)
            b_rev = box.addButton("Review...", W.QMessageBox.ActionRole)
            box.addButton("Not now", W.QMessageBox.RejectRole)
            b_off = box.addButton("Don't check at startup", W.QMessageBox.DestructiveRole)
            box.exec_()
            clicked = box.clickedButton()
            if clicked is b_imp:
                self._run_imports([{"source": s, "library": s["library"]} for s in new])
            elif clicked is b_rev:
                self.on_data_folder()
            elif clicked is b_off:
                self.cfg["auto_import"] = False
                self._save_folder_cfg()
                self.host.statusBar().showMessage("Startup check of the data folder turned off "
                                                  "(Import ▸ From data folder... to turn it back on)", 8000)
        self.run(self.client.scan_folder, folder, done=scanned,
                 failed=lambda m: self.host.statusBar().showMessage(f"Auto-import skipped: {m}", 8000))

    # ------------------------------------------------------------ category tree
    def load_facets(self):
        lib = self.active_lib
        if lib is None or not self.client:
            return
        self._facet_gen += 1
        gen = self._facet_gen

        def ok(f):
            if gen != self._facet_gen or not self._is_active(lib):
                return
            self.facets = f
            self._fill_tree()
            self._update_acq_counts()
        self.run(self.client.library_facets, lib["id"], done=ok)

    def _fill_tree(self):
        f = self.facets or {"total": 0, "categories": [], "uncategorised": 0}
        expanded = set()
        it = W.QTreeWidgetItemIterator(self.tree)
        while it.value():
            node = it.value()
            if node.isExpanded():
                expanded.add(node.data(0, PATH_ROLE))
            it += 1
        self.tree.blockSignals(True)
        self.tree.clear()
        root = W.QTreeWidgetItem([f"All materials", str(f["total"])])
        root.setData(0, PATH_ROLE, None)
        bold = root.font(0)
        bold.setBold(True)
        root.setFont(0, bold)
        self.tree.addTopLevelItem(root)
        nodes = {}
        counts = {}
        for c in f["categories"]:                          # roll counts up to every ancestor
            lv = repo.category_levels(c["path"])
            for k in range(1, len(lv) + 1):
                key = repo.SEP.join(lv[:k])
                counts[key] = counts.get(key, 0) + c["count"]
        for c in f["categories"]:
            lv = repo.category_levels(c["path"])
            parent = root
            for k in range(1, len(lv) + 1):
                key = repo.SEP.join(lv[:k])
                if key not in nodes:
                    n = W.QTreeWidgetItem([lv[k - 1], str(counts[key])])
                    n.setData(0, PATH_ROLE, key)
                    n.setToolTip(0, _crumb(key))
                    n.setTextAlignment(1, Qt.AlignRight | Qt.AlignVCenter)
                    parent.addChild(n)
                    nodes[key] = n
                parent = nodes[key]
        if f.get("uncategorised"):
            u = W.QTreeWidgetItem(["(uncategorised)", str(f["uncategorised"])])
            u.setData(0, PATH_ROLE, repo.UNCATEGORISED)
            ital = u.font(0)
            ital.setItalic(True)
            u.setFont(0, ital)
            u.setTextAlignment(1, Qt.AlignRight | Qt.AlignVCenter)
            root.addChild(u)
        root.setTextAlignment(1, Qt.AlignRight | Qt.AlignVCenter)
        root.setExpanded(True)
        for key, n in nodes.items():
            if key in expanded:
                n.setExpanded(True)
        target = nodes.get(self.cat_path) if self.cat_path not in (None, repo.UNCATEGORISED) else None
        if self.cat_path == repo.UNCATEGORISED:
            target = next((root.child(i) for i in range(root.childCount())
                           if root.child(i).data(0, PATH_ROLE) == repo.UNCATEGORISED), None)
        if self.cat_path is not None and target is None:       # node vanished (renamed / emptied)
            self.cat_path = None
            self.fetch_materials(False)
        target = target or root
        p = target.parent()
        while p is not None:
            p.setExpanded(True)
            p = p.parent()
        target.setSelected(True)
        self.tree.setCurrentItem(target)
        self.tree.blockSignals(False)
        self._update_crumb()
        self._sync_buttons()

    def _tree_node_path(self):
        items = self.tree.selectedItems()
        return items[0].data(0, PATH_ROLE) if items else None

    def on_category_selected(self):
        path = self._tree_node_path()
        if path == self.cat_path:
            return
        self.cat_path = path
        self._update_crumb()
        self.fetch_materials(False)
        self._sync_buttons()

    def _update_crumb(self):
        if self.active_lib is None:
            self.lbl_crumb.setText("")
            return
        bits = [self.active_lib["name"]]
        if self.cat_path == repo.UNCATEGORISED:
            bits.append("(uncategorised)")
        elif self.cat_path:
            bits += repo.category_levels(self.cat_path)
        a = self.acq_filter
        acq = f"   ·   {ACQ_LABEL['untagged' if a == repo.UNTAGGED else a]} spectra only" if a else ""
        self.lbl_crumb.setText(" › ".join(bits) + acq)
        self.lbl_crumb.setToolTip(self.lbl_crumb.text())

    def _tree_menu(self, pos):
        item = self.tree.itemAt(pos)
        if item is None:
            return
        self.tree.setCurrentItem(item)
        path = item.data(0, PATH_ROLE)
        m = W.QMenu(self)
        a_ren = m.addAction("Rename / move...")
        a_unc = m.addAction("Make uncategorised")
        m.addSeparator()
        a_exp = m.addAction("Expand all")
        a_col = m.addAction("Collapse all")
        real = path not in (None, repo.UNCATEGORISED)
        a_ren.setEnabled(real)
        a_unc.setEnabled(real)
        act = m.exec_(self.tree.viewport().mapToGlobal(pos))
        if act is a_ren:
            self.on_rename_category()
        elif act is a_unc:
            self._rename_category(path, None)
        elif act is a_exp:
            self.tree.expandAll()
        elif act is a_col:
            self.tree.collapseAll()
            self.tree.topLevelItem(0).setExpanded(True)

    def on_rename_category(self):
        path = self._tree_node_path()
        if path in (None, repo.UNCATEGORISED) or not self._need_client():
            return
        new, ok = W.QInputDialog.getText(
            self, "Rename / move category",
            f"New path for  {_crumb(path)}\n(everything below it moves along; empty = uncategorised)\n{CATEGORY_HINT}",
            W.QLineEdit.Normal, path)
        if ok and repo.normalize_category(new) != path:
            self._rename_category(path, new)

    def _rename_category(self, old, new):
        if old is None or not self._need_client():
            return
        lib = self.active_lib

        def ok(r):
            nn = repo.normalize_category(new)
            self.host.say(f"{r['materials']} material(s) moved from {_crumb(old)} to "
                          f"{_crumb(nn) if nn else '(uncategorised)'}.")
            if self.cat_path and (self.cat_path == old or self.cat_path.startswith(old + repo.SEP)):
                self.cat_path = (nn + self.cat_path[len(old):]) if nn else repo.UNCATEGORISED
            self.load_facets()
            self.fetch_materials(False)
        self.run(self.client.rename_category, lib["id"], old, new, done=ok)

    # ------------------------------------------------------------ acquisition filter
    def on_acq_filter(self, *_):
        self._update_crumb()
        if self.active_lib:
            self.fetch_materials(False)
        if self.ed_global.text().strip():
            self._global_search(show=False)

    def _update_acq_counts(self):
        acq = (self.facets or {}).get("acquisition") or {}
        for i in range(self.cb_acqf.count()):
            t = self.cb_acqf.itemData(i)
            if t is None:
                continue
            key = "untagged" if t == repo.UNTAGGED else t
            self.cb_acqf.setItemText(i, f"{ACQ_LABEL[key]}  ({acq.get(key, 0)})")

    # ------------------------------------------------------------ global material search (pop-up)
    def on_global_text(self, text):
        self._glob_timer.stop()
        t = text.strip()
        if not t:
            self._glob_gen += 1
            self._glob_model.clear()
            self._glob_comp.popup().hide()
            return
        if len(t) >= 2:
            self._glob_timer.start()

    def _global_search(self, show=True):
        if not self.client:
            return
        q = self.ed_global.text().strip()
        if len(q) < 2:
            return
        self._glob_gen += 1
        gen = self._glob_gen

        def ok(payload):
            if gen != self._glob_gen or self.ed_global.text().strip() != q:
                return
            items = payload.get("items", []) if isinstance(payload, dict) else payload
            self._glob_model.clear()
            if not items:
                it = QtGui.QStandardItem(f"No materials match \"{q}\"")
                it.setEnabled(False)
                it.setData(q, Qt.UserRole + 9)
                self._glob_model.appendRow(it)
            for m in items:
                tags = " · ".join(ACQ_LABEL.get(t, t) for t in m.get("acquisition_types") or [])
                extra = "   ".join(x for x in (_crumb(m.get("category")), tags) if x)
                it = QtGui.QStandardItem(f"{m['name']}   —   {m['library_name']}" + (f"   [{extra}]" if extra else ""))
                it.setData((m["library_id"], m["id"]), ID_ROLE)
                it.setData(q, Qt.UserRole + 9)
                tl = m.get("acquisition_types") or []
                if tl:
                    it.setIcon(_dot(ACQ_COLOR.get(tl[0], "#8b949e")))
                self._glob_model.appendRow(it)
            if show and self.ed_global.hasFocus():
                self._glob_comp.complete()
        self.run(self.client.search_materials, q, 50, self.acq_filter, done=ok)

    def on_global_open(self, index):
        ids = index.data(ID_ROLE)
        if not ids:
            return
        lid, mid = ids
        lib = self._lib_by_id(lid)
        if lib is None:
            return
        if not self._is_active(lib):
            self._select_library_row(lid)
            self.open_library(lib, then=lambda: self.open_material(mid))
        else:
            self.open_material(mid)

    def _select_library_row(self, lid):
        for r in range(self.tbl_libs.rowCount()):
            if self.tbl_libs.item(r, 0).data(ID_ROLE) == lid:
                self.tbl_libs.blockSignals(True)
                self.tbl_libs.selectRow(r)
                self.tbl_libs.blockSignals(False)
                return

    # ------------------------------------------------------------ materials (server-side search + paging)
    def on_material_text(self, text):
        t = text.strip()
        self._mat_timer.stop()
        if not t:                                    # cleared -> full list right away
            self.fetch_materials(False)
        elif len(t) >= 2:                            # one character is too broad; wait for more
            self._mat_timer.start()

    def fetch_materials(self, append=False, then=None):
        lib = self.active_lib
        if lib is None or not self.client:
            return
        self._mat_gen += 1
        gen = self._mat_gen
        offset = len(self.materials) if append else 0
        search = self.ed_matsearch.text().strip()
        search = search if len(search) >= 2 else None
        if append:
            self.b_more.setText("Loading…")
        else:
            self.lbl_count.setText("Searching…")

        def ok(p):
            if gen != self._mat_gen or not self._is_active(lib):
                return
            items = p.get("items", [])
            self.materials = (self.materials + items) if append else items
            self.mat_total = int(p.get("total", len(self.materials)))
            self._fill_materials()
            if then:
                then()

        def bad(msg):
            self.b_more.setText("See more")
            self.host._err(msg)
        self.run(self.client.list_materials, lib["id"], search, None, PAGE_SIZE, offset,
                 self.cat_path, self.acq_filter, done=ok, failed=bad)

    def _fill_materials(self):
        sel = (self.active_material or {}).get("id")
        self.tbl_mats.blockSignals(True)
        self.tbl_mats.clearSelection()
        self.tbl_mats.setRowCount(len(self.materials))
        for r, m in enumerate(self.materials):
            it = W.QTableWidgetItem(m["name"])
            it.setData(ID_ROLE, m["id"])
            it.setToolTip(m["name"] + (f"\n{m['description']}" if m.get("description") else ""))
            self.tbl_mats.setItem(r, 0, it)
            lv = repo.category_levels(m.get("category"))
            c = W.QTableWidgetItem(lv[-1] if lv else "—")
            c.setToolTip(_crumb(m.get("category")) or "uncategorised")
            self.tbl_mats.setItem(r, 1, c)
            a = W.QTableWidgetItem()
            tags = m.get("acquisition_types") or []
            a.setData(TAGS_ROLE, tags)
            a.setToolTip("\n".join(f"{ACQ_LABEL.get(t, t)}: {ACQ_HELP.get(t, 'no acquisition tag')}" for t in tags)
                         or "no spectrum")
            self.tbl_mats.setItem(r, 2, a)
            if m["id"] == sel:
                self.tbl_mats.selectRow(r)
        self.tbl_mats.blockSignals(False)
        q = self.ed_matsearch.text().strip()
        filt = len(q) >= 2 or self.cat_path or self.acq_filter
        what = f" match \"{q}\"" if len(q) >= 2 else ""
        if self.cat_path:
            what += " in this category"
        if self.acq_filter:
            what += " with that acquisition type"
        self.lbl_count.setText(f"{self.mat_total} material{'s' if self.mat_total != 1 else ''}"
                               f"{what or ' in this library'} — showing {len(self.materials)}.")
        if not self.materials:
            self.lbl_count.setText("No materials match these filters." if filt else "This library has no materials yet.")
        self.b_more.setText("See more")
        self._sync_buttons()

    def _selected_materials(self):
        rows = sorted({i.row() for i in self.tbl_mats.selectedIndexes()})
        return [self.materials[r] for r in rows if r < len(self.materials)]

    def on_material_selected(self):
        self._sync_buttons()
        r = self.tbl_mats.currentRow()
        it = self.tbl_mats.item(r, 0) if r >= 0 else None
        if it is not None and len(self._selected_materials()) == 1 and \
                (self.active_material or {}).get("id") != it.data(ID_ROLE):
            self.open_material(it.data(ID_ROLE))

    def on_add_material(self):
        lib = self.active_lib
        if not lib or not self._need_client():
            return
        start = self.cat_path if self.cat_path not in (None, repo.UNCATEGORISED) else None
        d = MaterialDialog(self, lib["name"], self.category_paths(), start)
        if not d.exec_():
            return
        v = d.choices()
        if not v["name"]:
            return self.host._err("material name is required")

        def ok(m):
            lib["material_count"] = (lib.get("material_count") or 0) + 1
            self._fill_libraries()
            self.load_facets()
            self.fetch_materials(False)
            self.host.say(f"\"{m['name']}\" added to {lib['name']}" +
                          (f" under {_crumb(m['category'])}" if m.get("category") else "") +
                          ". It has no spectrum yet - save one from the Signature Export tab or with "
                          "'Save clicked pixel' using the same name.")
        self.run(self.client.create_material, lib["id"], v["name"], v["category"], v["description"], done=ok)

    def on_edit_materials(self):
        mats = self._selected_materials()
        if not mats or not self._need_client():
            return
        d = EditMaterialsDialog(self, mats, self.category_paths())
        if not d.exec_():
            return
        ch = d.choices()
        acq = ch["acquisition"]
        kw = {}
        if ch["category"] is not UNSET:
            kw["category"] = ch["category"]
        if acq is not UNSET:
            kw["acquisition"] = None if acq == repo.UNTAGGED else acq
        if not kw:
            return

        def ok(r):
            parts = []
            if "category" in kw:
                c = repo.normalize_category(kw["category"])
                parts.append(f"category {_crumb(c) if c else '(uncategorised)'}")
            if "acquisition" in kw:
                parts.append(f"{r['spectra']} spectra tagged {kw['acquisition'] or 'untagged'}")
            self.host.say(f"{r['materials']} material(s) updated: " + ", ".join(parts) + ".")
            self.load_facets()
            self.fetch_materials(False)
            if self.active_material and any(m["id"] == self.active_material["id"] for m in mats):
                self.open_material(self.active_material["id"])
        self.run(self.client.update_materials, [m["id"] for m in mats], done=ok, **kw)

    def on_delete_material(self):
        m = self.active_material
        if not m or not self._need_client():
            return
        if W.QMessageBox.question(self, "Delete material", f"Delete \"{m['name']}\" with all its samples and spectra?"
                                  ) != W.QMessageBox.Yes:
            return

        def ok(_):
            self.materials = [x for x in self.materials if x["id"] != m["id"]]
            self.mat_total = max(0, self.mat_total - 1)
            if self.active_lib:
                self.active_lib["material_count"] = max(0, (self.active_lib.get("material_count") or 1) - 1)
            self.compare = [c for c in self.compare if c["id"] != m["id"]]
            self._fill_compare()
            self.active_material, self.samples, self.spectrum = None, [], None
            self._fill_libraries()
            self._fill_materials()
            self._show_material_info()
            self.load_facets()
            self.host.say(f"\"{m['name']}\" deleted.")
        self.run(self.client.delete_material, m["id"], done=ok)

    # ------------------------------------------------------------ one material -> samples -> spectrum
    def open_material(self, material_id):
        if not self.client:
            return
        c = self.client

        def load():
            mat = c.get_material(material_id)
            samples = c.list_samples(material_id)
            spec = None
            for s in samples:                           # first sample that actually has a spectrum
                sp = c.list_spectra(s["id"])
                if sp:
                    spec = sp[0]
                    break
            return mat, samples, spec

        def ok(res):
            mat, samples, spec = res
            self.active_material, self.samples, self.spectrum = mat, samples, spec
            self.cb_sample.blockSignals(True)
            self.cb_sample.clear()
            for s in samples:
                self.cb_sample.addItem(s.get("name") or f"Sample {s.get('sample_number')}", s["id"])
            if spec is not None:
                self.cb_sample.setCurrentIndex(max(0, self.cb_sample.findData(spec["sample_id"])))
            self.cb_sample.blockSignals(False)
            self._show_material_info()
            self.plot_spectrum()
            self.inner.setCurrentIndex(0)
            r = next((i for i, m in enumerate(self.materials) if m["id"] == mat["id"]), -1)
            selected = {i.row() for i in self.tbl_mats.selectedIndexes()}
            if r >= 0 and r not in selected:            # keep a multi-selection the user made meanwhile
                self.tbl_mats.blockSignals(True)
                self.tbl_mats.clearSelection()
                self.tbl_mats.selectRow(r)
                self.tbl_mats.blockSignals(False)
            self._sync_buttons()
        self.run(load, done=ok)

    def on_sample_changed(self, i):
        sid = self.cb_sample.itemData(i)
        if not sid or not self.client:
            return

        def ok(spectra):
            self.spectrum = spectra[0] if spectra else None
            self._show_material_info()
            self.plot_spectrum()
        self.run(self.client.list_spectra, sid, done=ok)

    def on_spectrum_acq_changed(self, *_):
        sp = self.spectrum
        if sp is None or not self.client:
            return
        t = self.cb_acq.currentData()
        new = None if t == repo.UNTAGGED else t
        if new == sp.get("acquisition_type"):
            return

        def ok(updated):
            self.spectrum = updated
            self._show_material_info()
            self.load_facets()
            self.fetch_materials(False)                  # refresh the row's chips (selection is kept)
            self.host.statusBar().showMessage(
                f"{(self.active_material or {}).get('name', 'Spectrum')}: acquisition set to "
                f"{ACQ_LABEL[new or 'untagged']}", 4000)
        self.run(self.client.set_spectrum_acquisition, sp["id"], new, done=ok)

    def _show_material_info(self):
        m, sp = self.active_material, self.spectrum
        if m is None:
            self.lbl_mat.setText("Open a material to see its spectrum in the Spectra pane.")
        else:
            cat = _crumb(m.get("category")) or "uncategorised"
            lines = [f"<b>{_esc(m['name'])}</b> · {len(self.samples)} sample{'s' if len(self.samples) != 1 else ''}",
                     f"<span style='color:#8b9bb4'>{_esc(cat)}</span>"]
            if m.get("description"):
                lines.append(_esc(m["description"]))
            if sp is None:
                lines.append("<i>No spectrum stored for this material/sample.</i>")
            else:
                unit = (sp.get("metadata") or {}).get("wavelength_unit", "nm")
                step = sp.get("wavelength_step")
                a = sp.get("acquisition_type") or "untagged"
                lines.append(f"<span style='color:{ACQ_COLOR[a]}'>●</span> {ACQ_LABEL[a]} · "
                             f"{len(sp['values'])} points · {sp['wavelength_start']:g}–{sp['wavelength_end']:g} {unit}"
                             + (f" · step {step:g}" if step else " · irregular grid"))
                meta = sp.get("metadata") or {}
                if sp.get("roi_class"):
                    mode = meta.get("roi_mode")
                    if mode == "class_mean":
                        src = (f"class mean of {meta.get('n_rois', '?')} ROI(s), {meta.get('n_pixels', '?')} px, "
                               f"{meta.get('n_scenes', '?')} image(s): {_esc(', '.join(meta.get('scenes') or []))}")
                    elif mode == "roi_mean":
                        src = f"mean of {_esc(meta.get('roi_name'))} in {_esc(meta.get('scene'))}, {meta.get('n_pixels', '?')} px"
                    elif mode == "pixel":
                        px = meta.get("pixel") or ["?", "?"]
                        src = f"pixel (r{px[0]}, c{px[1]}) of {_esc(meta.get('roi_name'))} in {_esc(meta.get('scene'))}"
                    else:
                        src = "ROI spectrum"
                    lines.append(f"ROI class <b>{_esc(sp['roi_class'])}</b> · {src}")
                elif meta.get("n_pixels"):
                    lines.append(f"Aggregated from {meta['n_pixels']} pixels" +
                                 (f" ({_esc(meta.get('source_scenes'))})" if meta.get("source_scenes") else ""))
            self.lbl_mat.setText("<br>".join(lines))
        self.cb_acq.blockSignals(True)
        a = (sp or {}).get("acquisition_type") or repo.UNTAGGED
        self.cb_acq.setCurrentIndex(max(0, self.cb_acq.findData(a)))
        self.cb_acq.blockSignals(False)
        self._sync_buttons()

    def _xy(self, spec):
        return repo.spectrum_wavelengths(spec), repo.spectrum_values(spec)

    def plot_spectrum(self):
        pc = self.host.plot_canvas
        m, sp = self.active_material, self.spectrum
        if m is None:
            return
        if sp is None:
            return pc.message(f"{m['name']}: no spectrum stored")
        x, y = self._xy(sp)
        meta = sp.get("metadata") or {}
        a = sp.get("acquisition_type") or "untagged"
        ax = pc.reset()
        ax.plot(x, y, color=ACQ_COLOR[a] if a != "untagged" else PALETTE[2], lw=1.6,
                label=f"{m['name']} ({ACQ_LABEL[a].lower()})")
        std = meta.get("std")
        if std is not None and len(std) == len(y):
            s = np.array([np.nan if v is None else v for v in std], float)
            ax.fill_between(x, y - s, y + s, color=ACQ_COLOR[a] if a != "untagged" else PALETTE[2], alpha=0.22,
                            label="± 1 std")
        unit = meta.get("wavelength_unit", "nm")
        cat = _crumb(m.get("category"))
        pc.finish(f"{m['name']}  —  {(self.active_lib or {}).get('name', '')}" + (f"  ·  {cat}" if cat else ""),
                  f"Wavelength ({unit})", sp.get("measurement_unit") or "Reflectance", True)

    def on_export_spectrum(self):
        m, sp = self.active_material, self.spectrum
        if sp is None:
            return
        safe = "".join(ch if ch.isalnum() else "_" for ch in m["name"]).strip("_") or "spectrum"
        p, _ = W.QFileDialog.getSaveFileName(self, "Export spectrum", safe + ".csv", "CSV (*.csv)")
        if not p:
            return
        x, y = self._xy(sp)
        meta = sp.get("metadata") or {}
        std = meta.get("std")
        unit = meta.get("wavelength_unit", "nm")
        try:
            with open(p, "w", newline="") as f:
                f.write(f"# material: {m['name']}\n# category: {m.get('category') or ''}\n"
                        f"# acquisition: {sp.get('acquisition_type') or 'untagged'}\n")
                w = csv.writer(f)
                w.writerow([f"wavelength_{unit}", sp.get("measurement_unit") or "reflectance"] + (["std"] if std else []))
                for i in range(len(x)):
                    row = [f"{x[i]:.6g}", "" if np.isnan(y[i]) else f"{y[i]:.8g}"]
                    if std:
                        row.append("" if std[i] is None else f"{std[i]:.8g}")
                    w.writerow(row)
        except OSError as e:
            return self.host._err(e)
        self.host.say(f"Wrote {p}")

    # ------------------------------------------------------------ comparison
    def on_toggle_compare(self):
        m = self.active_material
        if m is None or self.spectrum is None:
            return
        i = next((k for k, c in enumerate(self.compare) if c["id"] == m["id"]), -1)
        if i >= 0:
            del self.compare[i]
        else:
            if len(self.compare) >= MAX_COMPARE:
                return self.host._err(f"you can compare up to {MAX_COMPARE} spectra")
            self.compare.append({"id": m["id"], "name": m["name"], "library_id": m["library_id"],
                                 "library": (self.active_lib or {}).get("name", ""), "spectrum": self.spectrum})
        self._fill_compare()
        self._sync_buttons()
        if self.compare:
            self.plot_comparison()

    def _fill_compare(self):
        self.lst_cmp.clear()
        for k, c in enumerate(self.compare):
            a = c["spectrum"].get("acquisition_type") or "untagged"
            it = W.QListWidgetItem(f"{c['name']}  —  {c['library']}  [{ACQ_LABEL[a]}]")
            pm = QtGui.QPixmap(10, 10)
            pm.fill(QtGui.QColor.fromRgbF(*PALETTE[k % 10]))
            it.setIcon(QtGui.QIcon(pm))
            self.lst_cmp.addItem(it)
        self.inner.setTabText(1, f"Compare ({len(self.compare)})" if self.compare else "Compare")

    def on_compare_remove(self):
        rows = sorted({self.lst_cmp.row(i) for i in self.lst_cmp.selectedItems()}, reverse=True)
        for r in rows:
            del self.compare[r]
        self._fill_compare()
        self._sync_buttons()
        self.plot_comparison()

    def on_compare_clear(self):
        self.compare = []
        self._fill_compare()
        self._sync_buttons()
        self.host.plot_canvas.message("Comparison cleared")

    def plot_comparison(self):
        if not self.compare:
            return self.host.plot_canvas.message("Add materials to the comparison from the Spectrum tab")
        ax = self.host.plot_canvas.reset()
        styles = {"lab": "-", "field": "--", "airborne": "-.", "satellite": ":", "untagged": "-"}
        for k, c in enumerate(self.compare):
            x, y = self._xy(c["spectrum"])
            a = c["spectrum"].get("acquisition_type") or "untagged"
            ax.plot(x, y, color=PALETTE[k % 10], lw=1.4, ls=styles[a],
                    label=f"{c['name']} ({c['library']}, {ACQ_LABEL[a].lower()})")
        self.host.plot_canvas.finish(f"Comparison of {len(self.compare)} spectra  (line style = acquisition)",
                                     "Wavelength (nm)", "Reflectance", True)

    def on_compare_export(self):
        if not self.compare:
            return self.host._err("the comparison is empty")
        p, _ = W.QFileDialog.getSaveFileName(self, "Export comparison (base name)", "library_comparison")
        if not p:
            return
        from ..core import export
        from ..core.resample import resample_spectra
        from ..core.signature import Signature
        x0, _ = self._xy(self.compare[0]["spectrum"])
        sigs, resampled = [], []
        for c in self.compare:
            x, y = self._xy(c["spectrum"])
            std = (c["spectrum"].get("metadata") or {}).get("std")
            s = np.array([np.nan if v is None else v for v in std], float) if std and len(std) == len(y) else np.full(len(y), np.nan)
            if len(x) != len(x0) or not np.allclose(x, x0):
                y = resample_spectra(y[None], x, x0)[0]
                s = resample_spectra(s[None], x, x0)[0]
                resampled.append(c["name"])
            sigs.append(Signature(c["name"], x0, y, s, 0, c["library"],
                                  {"material_id": c["id"], "acquisition": c["spectrum"].get("acquisition_type")}))
        try:
            paths = export.export_all(os.path.splitext(p)[0], sigs)
        except Exception as e:                                                   # noqa: BLE001
            return self.host._err(e)
        self.host.say("Wrote:\n" + "\n".join(paths) +
                      (f"\n(resampled onto {self.compare[0]['name']}'s grid: {', '.join(resampled)})" if resampled else ""))

    # ------------------------------------------------------------ saving from HyperVista
    def _ask_target(self, what, default_name=None, acquisition=None):
        """SaveDialog -> choices dict or None."""
        if not self._need_client():
            return None
        start = self.cat_path if self.cat_path not in (None, repo.UNCATEGORISED) else None
        d = SaveDialog(self, self.libraries, (self.active_lib or {}).get("id"), what, default_name,
                       self.category_paths(), start, acquisition)
        if not d.exec_():
            return None
        ch = d.choices()
        if ch["library_id"] == SaveDialog.NEW and not ch["new_library"]:
            self.host._err("enter a name for the new library")
            return None
        return ch

    def _save(self, ch, items, label):
        c = self.client

        def go():
            lid = ch["library_id"]
            if lid == SaveDialog.NEW:
                lid = c.create_library(ch["new_library"], "Signatures saved from HyperVista")["id"]
            return lid, c.save_spectra(lid, ch["dataset"], items, ch["category"], ch.get("acquisition"))

        def ok(res):
            lid, saved = res
            names = ", ".join(s["material"]["name"] for s in saved)
            tag = f", {ch['acquisition']}" if ch.get("acquisition") else ""
            self.host.say(f"Saved {len(saved)} {label} to the library database ({ch['dataset']}{tag}): {names}.")
            if self.active_lib and self.active_lib["id"] == lid:
                self.load_facets()
                self.fetch_materials(False)
            self.load_libraries(select_id=lid)          # selecting another library's row opens it
        self.run(go, done=ok)

    def save_signatures(self, sigs):
        """Called by the Signature Export tab's 'Save to library database' button."""
        if not sigs:
            return
        ch = self._ask_target(f"{len(sigs)} aggregated class signature(s): " + ", ".join(s.name for s in sigs))
        if ch:
            self._save(ch, sigs, "signatures")

    def on_save_pixel(self):
        ls = self.host.last_spectrum
        if ls is None or ls[0] is None:
            return self.host._err("click a pixel of a scene that has wavelengths first")
        sc = self.host.current_scene()
        px = getattr(self.host, "last_pixel", None)
        default = f"{sc.name if sc else 'scene'} pixel" + (f" ({px[0]}, {px[1]})" if px else "")
        ch = self._ask_target("Spectrum of the last clicked pixel", default)
        if not ch:
            return
        item = {"name": ch["name"] or default, "wavelengths": np.asarray(ls[0], float).tolist(),
                "values": np.asarray(ls[1], float).tolist(),
                "material_metadata": {"created_by": "HyperVista", "scene": sc.name if sc else None,
                                      "pixel": list(px) if px else None},
                "spectrum_metadata": {"n_pixels": 1, "source_scenes": sc.name if sc else ""}}
        self._save(ch, [item], "pixel spectrum")

    # ------------------------------------------------------------ matching
    def on_match(self):
        ls = self.host.last_spectrum
        if ls is None or ls[0] is None:
            return self.host._err("click a pixel of a scene that has wavelengths first")
        if not self._need_client():
            return
        everywhere = self.cb_scope.currentIndex() == 1
        if not everywhere and self.active_lib is None:
            return self.host._err("select a library (or choose 'All libraries')")
        libs = list(self.libraries) if everywhere else [self.active_lib]
        wl, spec = np.asarray(ls[0], float), np.asarray(ls[1], float)
        top = self.sp_top.value()
        acq = self.acq_filter
        cat = self.cat_path if self.chk_matchcat.isChecked() else None
        c = self.client
        self.txt_match.setPlainText("Matching…")

        def go():
            entries = []
            for l in libs:
                for e in c.library_spectra(l["id"], acq, cat):
                    e["library"] = l["name"]
                    entries.append(e)
            ranked = repo.spectral_angle_rank(spec, wl, entries, top)
            lib_of = {e["material"]["id"]: (e["library"], e["spectrum"]) for e in entries}
            return len(entries), [(a, m, n, *lib_of[m["id"]]) for a, m, n in ranked]

        def ok(res):
            n_ref, ranked = res
            scope = []
            if acq:
                scope.append(ACQ_LABEL["untagged" if acq == repo.UNTAGGED else acq].lower() + " spectra")
            if cat:
                scope.append(_crumb(cat) if cat != repo.UNCATEGORISED else "uncategorised")
            scope = f"; {', '.join(scope)}" if scope else ""
            if not ranked:
                if n_ref == 0:
                    self.txt_match.setPlainText(f"No reference spectra pass the filters{scope.replace(';', ':')} in "
                                                f"{'any library' if everywhere else libs[0]['name']}.")
                else:
                    self.txt_match.setPlainText(f"None of the {n_ref} reference spectra{scope.replace(';', ' -')} "
                                                f"overlap the pixel's {wl.min():g}–{wl.max():g} nm.")
                return
            self.txt_match.setPlainText(
                "\n".join(f"{k + 1}. {m['name']}  ({lib}, {ACQ_LABEL[s.get('acquisition_type') or 'untagged'].lower()})"
                          f"  angle {np.degrees(a):.2f}°  [{n} bands]"
                          for k, (a, m, n, lib, s) in enumerate(ranked)) +
                f"\n({n_ref} reference spectra searched{scope})")
            ax = self.host.plot_canvas.reset()
            ax.plot(wl, spec / (np.nanmax(np.abs(spec)) or 1), color=PALETTE[3], lw=2.2, label="clicked pixel (scaled)")
            for k, (a, m, n, lib, s) in enumerate(ranked[:5]):
                x, y = self._xy(s)
                sel = (x >= wl.min()) & (x <= wl.max())
                y = y / (np.nanmax(np.abs(y[sel])) if sel.any() and np.nanmax(np.abs(y[sel])) else 1)
                ax.plot(x[sel] if sel.any() else x, y[sel] if sel.any() else y, color=PALETTE[[0, 1, 2, 4, 5][k]],
                        lw=1.1, ls="--", label=f"{m['name']} ({np.degrees(a):.1f}°)")
            self.host.plot_canvas.finish("Pixel vs best library matches (each scaled to max 1)", "Wavelength (nm)",
                                         "Scaled value", True)
        self.run(go, done=ok, failed=lambda msg: (self.txt_match.setPlainText("Matching failed."), self._fail(msg)))

    # ------------------------------------------------------------ tab shown
    def on_shown(self):
        if self.client is None:
            return
        if self.spectrum is not None:
            self.plot_spectrum()
        elif not self.libraries:
            self.load_libraries()


# ============================================================ small helpers
def _labeled(text, widget):
    v = W.QVBoxLayout()
    v.setSpacing(3)
    lab = W.QLabel(text)
    lab.setProperty("role", "fieldlabel")
    v.addWidget(lab)
    v.addWidget(widget)
    return v


def _group(title, *items):
    g = W.QGroupBox(title)
    g.setSizePolicy(W.QSizePolicy.Ignored, W.QSizePolicy.Preferred)
    lay = W.QVBoxLayout(g)
    for w in items:
        lay.addWidget(w) if isinstance(w, W.QWidget) else lay.addLayout(w)
    return g


def _row(*widgets):
    r = W.QHBoxLayout()
    for w in widgets:
        r.addWidget(w)
    return r


def _page(layout):
    w = W.QWidget()
    w.setLayout(layout)
    return w


def _short(client):
    """Compact one-line connection text (full text goes in the tooltip)."""
    if client.mode == "api":
        return f"API · {client.base.split('://', 1)[-1]}"
    u = client.db.engine.url
    if client.db.is_postgres:
        return f"PostgreSQL · {u.database}"
    return f"SQLite · {os.path.basename(u.database or 'memory')}"


def _table_defaults(t, stretch_first=False):
    h = t.horizontalHeader()
    if stretch_first:
        h.setSectionResizeMode(0, W.QHeaderView.Stretch)
        for c in range(1, t.columnCount()):
            h.setSectionResizeMode(c, W.QHeaderView.ResizeToContents)
    else:
        h.setStretchLastSection(True)
    t.verticalHeader().setVisible(False)
    t.setSelectionBehavior(W.QAbstractItemView.SelectRows)
    t.setSelectionMode(W.QAbstractItemView.SingleSelection)
    t.setEditTriggers(W.QAbstractItemView.NoEditTriggers)
    t.setAlternatingRowColors(True)
    t.setWordWrap(False)


def _num_item(v):
    it = W.QTableWidgetItem(str(v))
    it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
    return it


def _base(path) -> str:
    return os.path.basename(str(path).rstrip("/\\")) or str(path)


def _esc(s):
    return str(s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
