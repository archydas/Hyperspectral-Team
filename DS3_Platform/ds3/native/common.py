"""Shared parts of the native Library (main) and Image pages: library store, spectrum chart, pickers, dialogs."""
from __future__ import annotations

import csv
import math

import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PyQt5 import QtCore, QtWidgets

from .. import style
from ..style import C, MPL
from .api import run_bg

W = QtWidgets
Qt = QtCore.Qt

DS_COLORS = ["#059669", "#2563eb", "#db2777", "#d97706", "#7c3aed", "#ea580c", "#0891b2", "#dc2626", "#4d7c0f",
             "#be185d", "#0f766e", "#6d28d9", "#92400e", "#1d4ed8", "#9d174d", "#166534"]
OBJ_COLORS = ["#d97706", "#7c3aed", "#0891b2", "#dc2626", "#4d7c0f", "#be185d", "#2563eb", "#0f766e", "#ea580c",
              "#6d28d9"]
LIB_GREY = ["#6b7280", "#a16207", "#0e7490", "#9d174d", "#3f6212", "#4338ca", "#b91c1c", "#9ca3af"]
ACQ = ["", "lab", "field", "airborne", "satellite"]
NONE_TOKEN = "__none__"


def fmt_num(v):
    if v is None:
        return "–"
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, int) or float(v).is_integer() and abs(v) < 1e12:
        return f"{int(v):,}"
    a = abs(v)
    return f"{v:.2e}" if a >= 1000 or (0 < a < 0.01) else f"{v:.3f}"


def angle_deg(a, b):
    ab = aa = bb = 0.0
    n = 0
    for x, y in zip(a, b):
        if x is None or y is None:
            continue
        ab += x * y
        aa += x * x
        bb += y * y
        n += 1
    if n < 3 or not aa or not bb:
        return None
    return math.degrees(math.acos(max(-1.0, min(1.0, ab / math.sqrt(aa * bb)))))


def xy(wl, vals):
    """wavelengths + values (with None gaps) -> x, y float arrays (NaN gaps, so the line breaks there)."""
    x = np.asarray([float(w) for w in wl], float)
    y = np.asarray([np.nan if v is None else float(v) for v in vals], float)
    return x, y


def save_csv(parent, default_name, header, rows):
    path, _ = W.QFileDialog.getSaveFileName(parent, "Export CSV", default_name, "CSV (*.csv)")
    if not path:
        return None
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    return path


def save_bytes(parent, default_name, data, filt="All files (*.*)"):
    path, _ = W.QFileDialog.getSaveFileName(parent, "Save file", default_name, filt)
    if not path:
        return None
    with open(path, "wb") as f:
        f.write(data)
    return path


def btn(text, fn=None, kind=None, tip=None, accent=None):
    b = W.QPushButton(text)
    if kind or accent:
        style.prop(b, **({"ds3": kind} if kind else {}), **({"accent": accent} if accent else {}))
    if tip:
        b.setToolTip(tip)
    if fn:
        b.clicked.connect(lambda _=False: fn())
    return b


def small(text, wrap=True, rich=False):
    lab = W.QLabel(text)
    style.prop(lab, ds3="small")
    lab.setWordWrap(wrap)
    if rich:
        lab.setTextFormat(Qt.RichText)
        lab.setOpenExternalLinks(False)
    return lab


def note_err(lab, text):
    lab.setText(text or "")
    lab.setStyleSheet(f"color:{C['red']};")
    lab.setVisible(bool(text))


def hrow(*widgets, stretch_at=None, spacing=6):
    h = W.QHBoxLayout()
    h.setSpacing(spacing)
    h.setContentsMargins(0, 0, 0, 0)
    for i, w in enumerate(widgets):
        if w is None:
            h.addStretch(1)
        elif isinstance(w, W.QLayout):
            h.addLayout(w)
        else:
            h.addWidget(w, 1 if i == stretch_at else 0)
    return h


def card(layout_cls=W.QVBoxLayout, margins=(14, 12, 14, 12)):
    f = W.QFrame()
    style.prop(f, ds3="card")
    lay = layout_cls(f)
    lay.setContentsMargins(*margins)
    lay.setSpacing(8)
    return f, lay


# ------------------------------------------------------------------ the libraries known to the backend
class Store(QtCore.QObject):
    """One per backend: libraries, statistics, start-up import state; both pages listen to `changed`."""
    changed = QtCore.pyqtSignal()
    failed = QtCore.pyqtSignal(str)
    _inst = None

    def __init__(self, api):
        super().__init__()
        self.api = api
        self.libraries = []
        self.stats = None
        self.startup = None
        self.error = ""
        self._busy = False
        self._again = False
        self._timer = QtCore.QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self.refresh)

    @classmethod
    def get(cls, api=None):
        if cls._inst is None and api is not None:
            cls._inst = Store(api)
        return cls._inst

    def refresh(self):
        if self._busy:
            self._again = True
            return
        self._busy = True

        def load():
            libs = self.api.get("/api/libraries/")
            if isinstance(libs, dict):
                libs = libs.get("items") or libs.get("results") or []
            stats = self.api.get("/api/stats")
            try:
                su = self.api.get("/api/startup_import")
            except Exception:                                       # noqa: BLE001
                su = None
            return libs, stats, su

        def done(r):
            self._busy = False
            libs, self.stats, self.startup = r
            out = []
            for i, l in enumerate(libs):
                md = l.get("metadata") or {}
                main = bool(md.get("main"))
                typ = md.get("type") or "spectral"
                out.append(dict(id=str(l["id"]), name=l["name"], main=main, type=typ, metadata=md,
                                description=l.get("description") or "",
                                source="Main library" if main else ("Object library" if typ == "object" else "Pixel library"),
                                count=l.get("material_count") or 0, spectra=l.get("spectrum_count") or 0))
            self.libraries = out
            self.error = ""
            self.changed.emit()
            if self.startup and self.startup.get("running"):
                self._timer.start(2500)                             # main libraries are still being read
            if self._again:
                self._again = False
                self.refresh()

        def fail(m):
            self._busy = False
            self.error = m
            self.failed.emit(m)
            self.changed.emit()
        run_bg(load, done=done, failed=fail)

    def lib(self, lid):
        return next((l for l in self.libraries if l["id"] == str(lid)), None)

    def db_kind(self):
        d = (self.stats or {}).get("database") or ""
        return "PostgreSQL" if d.startswith("postgres") else ("SQLite (local file)" if d.startswith("sqlite") else
                                                               (d.split(":")[0] or "database"))


# ------------------------------------------------------------------ spectrum chart (matplotlib)
class SpectrumChart(W.QWidget):
    """A matplotlib line chart of spectra. series = [dict(name, x, y, color, lw=1.6, ls='-', alpha=1, band=None)]."""
    clicked = QtCore.pyqtSignal(float, float)

    def __init__(self, height=260, toolbar=False, parent=None):
        super().__init__(parent)
        import matplotlib as mpl
        with mpl.rc_context(MPL):
            self.fig = Figure(figsize=(5, height / 100), dpi=100)
            self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.canvas.setMinimumHeight(int(height * 0.7))
        lay = W.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        if toolbar:
            self.toolbar = NavigationToolbar2QT(self.canvas, self)
            self.toolbar.setIconSize(QtCore.QSize(16, 16))
            lay.addWidget(self.toolbar)
        lay.addWidget(self.canvas, 1)
        self.canvas.mpl_connect("button_press_event",
                                lambda e: e.inaxes and e.xdata is not None and self.clicked.emit(e.xdata, e.ydata))
        self.series = []
        self.message("")

    def message(self, text):
        self.series = []
        self.ax.clear()
        self.ax.set_axis_off()
        if text:
            self.ax.text(0.5, 0.5, text, ha="center", va="center", color=C["muted"], fontsize=9, wrap=True,
                         transform=self.ax.transAxes)
        self.canvas.draw_idle()

    def plot(self, series, unit=None, xlabel="Wavelength (nm)", ylabel=None, title=None, legend=True, max_legend=12):
        self.series = list(series)
        ax = self.ax
        ax.clear()
        ax.set_axis_on()
        if not self.series:
            return self.message("Nothing to plot")
        for s in self.series:
            x = np.asarray(s["x"], float)
            y = np.asarray(s["y"], float)
            if s.get("band") is not None:
                lo, hi = (np.asarray(v, float) for v in s["band"])
                ax.fill_between(x, lo, hi, color=s.get("color"), alpha=0.18, linewidth=0)
            ax.plot(x, y, color=s.get("color"), lw=s.get("lw", 1.5), ls=s.get("ls", "-"), alpha=s.get("alpha", 1.0),
                    label=s.get("name"))
        ax.set_xlabel(xlabel, fontsize=8.5)
        yl = ylabel or ((unit[:1].upper() + unit[1:]) if unit else "Value")
        ax.set_ylabel(yl, fontsize=8.5)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.6)
        if title:
            ax.set_title(title, fontsize=9)
        if legend and 1 <= len(self.series) <= max_legend:
            ax.legend(fontsize=7.5, loc="best", framealpha=0.85)
        try:
            self.fig.tight_layout()
        except Exception:                                           # noqa: BLE001
            pass
        self.canvas.draw_idle()

    def save_png(self, parent, name="chart.png"):
        p, _ = W.QFileDialog.getSaveFileName(parent, "Save chart", name, "PNG (*.png);;PDF (*.pdf);;SVG (*.svg)")
        if p:
            self.fig.savefig(p, dpi=200, bbox_inches="tight")
        return p


# ------------------------------------------------------------------ "Save to library" chooser
class LibraryPicker(W.QWidget):
    """'＋ New library…' + existing libraries of one kind (object / pixel); never the main libraries."""

    def __init__(self, store, default_name="", kind="pixel", parent=None):
        super().__init__(parent)
        self.store, self.kind = store, kind
        self.cb = W.QComboBox()
        self.cb.setMinimumWidth(220)
        self.ed = W.QLineEdit(default_name)
        self.ed.setPlaceholderText("name of the new library")
        self.ed.setMinimumWidth(200)
        h = W.QHBoxLayout(self)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(self.cb)
        h.addWidget(self.ed, 1)
        self.cb.currentIndexChanged.connect(lambda _: self.ed.setVisible(not self.cb.currentData()))
        self.fill()

    def fill(self):
        cur = self.cb.currentData()
        self.cb.blockSignals(True)
        self.cb.clear()
        self.cb.addItem("＋ New library…", "")
        for l in self.store.libraries:
            if l["main"] or ((self.kind == "object") != (l["type"] == "object")):
                continue
            self.cb.addItem(f"{l['name']} ({l['spectra']})", l["name"])
        i = self.cb.findData(cur) if cur else 0
        self.cb.setCurrentIndex(max(0, i))
        self.cb.blockSignals(False)
        self.ed.setVisible(not self.cb.currentData())

    def value(self):
        return (self.cb.currentData() or self.ed.text()).strip()

    def existing(self):
        return bool(self.cb.currentData())


class SaveDialog(W.QDialog):
    """Library (new / existing) + category + acquisition."""

    def __init__(self, parent, title, note, store, default_name, kind="pixel", with_category=True):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumWidth(560)
        lay = W.QVBoxLayout(self)
        lay.addWidget(small(note, rich=True))
        lay.addWidget(W.QLabel("Library"))
        self.picker = LibraryPicker(store, default_name, kind)
        lay.addWidget(self.picker)
        g = W.QGridLayout()
        self.ed_cat = W.QLineEdit()
        self.ed_cat.setPlaceholderText("optional, e.g. Vegetation/Trees")
        self.cb_acq = W.QComboBox()
        self.cb_acq.addItems(["—"] + ACQ[1:])
        if with_category:
            g.addWidget(W.QLabel("Category"), 0, 0)
            g.addWidget(self.ed_cat, 0, 1)
        g.addWidget(W.QLabel("Acquisition"), 1, 0)
        g.addWidget(self.cb_acq, 1, 1)
        lay.addLayout(g)
        self.err = small("")
        note_err(self.err, "")
        lay.addWidget(self.err)
        bb = W.QDialogButtonBox()
        self.ok = bb.addButton("Save to library", W.QDialogButtonBox.AcceptRole)
        style.prop(self.ok, ds3="primary", accent="library")
        bb.addButton(W.QDialogButtonBox.Cancel)
        bb.accepted.connect(self._ok)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

    def _ok(self):
        if not self.picker.value():
            return note_err(self.err, "Give the library a name or choose an existing one.")
        self.accept()

    def values(self):
        acq = self.cb_acq.currentText()
        return dict(library=self.picker.value(), existing=self.picker.existing(),
                    category=self.ed_cat.text().strip() or None, acquisition=None if acq == "—" else acq)


class ImageWindow(W.QDialog):
    """Show a picture (PNG data) large, with Save PNG."""

    def __init__(self, parent, title, pixmap, data=None, caption=""):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(760, 640)
        lay = W.QVBoxLayout(self)
        sc = W.QScrollArea()
        sc.setWidgetResizable(True)
        lab = W.QLabel()
        lab.setAlignment(Qt.AlignCenter)
        lab.setStyleSheet("background:#222;")
        pm = pixmap
        if pm.width() < 500:
            f = max(1, 500 // max(1, pm.width()))
            pm = pm.scaled(pm.width() * f, pm.height() * f, Qt.KeepAspectRatio, Qt.FastTransformation)
        lab.setPixmap(pm)
        sc.setWidget(lab)
        lay.addWidget(sc, 1)
        if caption:
            lay.addWidget(small(caption))
        row = W.QHBoxLayout()
        row.addStretch(1)
        if data is not None:
            row.addWidget(btn("Save PNG", lambda: save_bytes(self, title.replace(" ", "_") + ".png", data, "PNG (*.png)")))
        row.addWidget(btn("Close", self.accept))
        lay.addLayout(row)
