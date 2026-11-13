"""Dialogs of the native Image page: Available Bands List, analysis windows (Spectral menu) and their results,
Spectral Analyst, SAM classification map, georeferencing with control points, map view, ROI aggregation."""
from __future__ import annotations

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

from .. import style
from ..style import C
from .api import data_url_bytes, pixmap_from_data_url, run_bg
from .common import (DS_COLORS, ImageWindow, SaveDialog, SpectrumChart, btn, fmt_num, hrow, note_err, save_bytes, small,
                     xy)
from .library_tools import cell, section, table
from .mapview import ESRI, PROVIDERS, MapView

W = QtWidgets
Qt = QtCore.Qt

TITLES = {"pca": "Principal Components (PCA)", "kmeans": "K-Means Clustering", "sam": "Spectral Angle Mapper",
          "unmix": "Linear Spectral Unmixing", "bandmath": "Band Math / Spectral Index", "anomaly": "Anomaly Detection (RX)",
          "snr": "Signal-to-Noise Ratio", "analyst": "Spectral Analyst"}
INDICES = [("NDVI — vegetation greenness", "(w800 - w670) / (w800 + w670)"),
           ("NDRE — red edge", "(w790 - w720) / (w790 + w720)"),
           ("SAVI — soil-adjusted vegetation", "1.5 * (w800 - w670) / (w800 + w670 + 0.5)"),
           ("EVI — enhanced vegetation", "2.5 * (w800 - w670) / (w800 + 6 * w670 - 7.5 * w475 + 1)"),
           ("NDWI — canopy water", "(w860 - w1240) / (w860 + w1240)"),
           ("NDBI — built-up", "(w1650 - w830) / (w1650 + w830)"),
           ("PRI — photochemical reflectance", "(w531 - w570) / (w531 + w570)")]
PRESETS = [("NDVI", "(w800 - w670) / (w800 + w670)"), ("NDRE", "(w790 - w720) / (w790 + w720)"),
           ("NDWI", "(w860 - w1240) / (w860 + w1240)"), ("Clay ratio", "w2200 / w2100"),
           ("Visible mean", "mean(w450, w550, w650)")]


def default_rgb(wl):
    n = len(wl)
    if n and min(wl) <= 480 and max(wl) >= 660:
        near = lambda t: int(np.argmin(np.abs(np.asarray(wl) - t)))          # noqa: E731
        r, g, b = near(650), near(550), near(470)
        if len({r, g, b}) == 3:
            return [r, g, b]
    return [min(n - 1, max(0, int(n * f))) for f in (0.75, 0.5, 0.25)]


def lib_combo(store, empty="— none —"):
    cb = W.QComboBox()
    cb.addItem(empty, "")
    for l in store.libraries:
        cb.addItem(f"{l['name']} ({l['spectra']})", l["id"])
    return cb


# ------------------------------------------------------------------ Available Bands List
class BandsDialog(W.QDialog):
    """ENVI-style band list: Gray Scale / RGB, click a band for each colour, Load."""

    def __init__(self, parent, info, wl, current=None):
        super().__init__(parent)
        self.setWindowTitle("Available Bands List")
        self.resize(460, 620)
        self.wl = wl
        self.sel = {"R": 0, "G": 0, "B": 0}
        d = [b - 1 for b in current] if current else default_rgb(wl)
        self.sel.update(R=d[0], G=d[1], B=d[2])
        self.cur = "R"
        lay = W.QVBoxLayout(self)
        lay.addWidget(W.QLabel(f"▾ {info['name']}"))
        self.lst = W.QListWidget()
        for i, w in enumerate(wl):
            self.lst.addItem(self._name(i))
        self.lst.itemClicked.connect(lambda it: self._pick(self.lst.row(it)))
        lay.addWidget(self.lst, 1)
        mb = W.QHBoxLayout()
        self.rb_gray = W.QRadioButton("Gray Scale")
        self.rb_rgb = W.QRadioButton("RGB Color")
        self.rb_rgb.setChecked(True)
        mb.addWidget(self.rb_gray)
        mb.addWidget(self.rb_rgb)
        mb.addStretch(1)
        lay.addLayout(mb)
        self.rbs, self.fields = {}, {}
        grp = W.QButtonGroup(self)
        for c in "RGB":
            rb = W.QRadioButton(c)
            grp.addButton(rb)
            rb.toggled.connect(lambda on, c=c: on and setattr(self, "cur", c))
            f = W.QLineEdit()
            f.setReadOnly(True)
            self.rbs[c], self.fields[c] = rb, f
            lay.addLayout(hrow(rb, f, stretch_at=1))
        self.rbs["R"].setChecked(True)
        self.rb_gray.toggled.connect(lambda _: self._mode())
        lay.addWidget(small(f"Dims {info.get('cols')} x {info.get('rows')} x {len(wl)} · {info.get('unit') or ''}"
                            + (f" · {info['format']}" if info.get("format") else "")
                            + (" · wavelengths assumed" if info.get("assumed_wavelengths") else "")))
        bb = W.QDialogButtonBox()
        self.ok = bb.addButton("Load RGB", W.QDialogButtonBox.AcceptRole)
        style.prop(self.ok, ds3="primary", accent="image")
        bb.addButton(W.QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)
        self._draw()
        self.lst.scrollToItem(self.lst.item(self.sel["R"]), W.QAbstractItemView.PositionAtCenter)

    def _name(self, i):
        return f"Band {i + 1}" + (f" ({self.wl[i]:.2f} nm)" if i < len(self.wl) else "")

    def _mode(self):
        gray = self.rb_gray.isChecked()
        for c in "GB":
            self.rbs[c].setEnabled(not gray)
            self.fields[c].setEnabled(not gray)
        self.rbs["R"].setText("Gray" if gray else "R")
        self.rbs["R"].setChecked(True)
        self.ok.setText("Load Gray" if gray else "Load RGB")
        self._draw()

    def _pick(self, i):
        self.sel[self.cur] = i
        if self.rb_rgb.isChecked():
            nxt = {"R": "G", "G": "B", "B": "B"}[self.cur]
            self.rbs[nxt].setChecked(True)
        self._draw()

    def _draw(self):
        gray = self.rb_gray.isChecked()
        for i in range(self.lst.count()):
            tags = "".join(c for c in "RGB" if (not gray or c == "R") and self.sel[c] == i)
            it = self.lst.item(i)
            it.setText(self._name(i) + (f"   [{tags}]" if tags else ""))
            f = it.font()
            f.setBold(bool(tags))
            it.setFont(f)
        for c in "RGB":
            self.fields[c].setText(self._name(self.sel[c]))

    def bands(self):
        if self.rb_gray.isChecked():
            return [self.sel["R"] + 1] * 3
        return [self.sel[c] + 1 for c in "RGB"]


# ------------------------------------------------------------------ analysis windows
class EndmemberBox(W.QWidget):
    """Collected spectra (ticks) or all spectra of a library."""

    def __init__(self, page):
        super().__init__()
        lay = W.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(section("Spectra to use"))
        self.lst = W.QListWidget()
        self.lst.setMaximumHeight(150)
        self.coll = page.coll
        for i, s in enumerate(page.coll):
            it = W.QListWidgetItem(f"{s['name']}  ·  " + (f"{s.get('n')} px" if s.get("kind") == "roi" else f"{s.get('row')}, {s.get('col')}"))
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.Checked)
            pm = QtGui.QPixmap(10, 10)
            pm.fill(QtGui.QColor(s["color"]))
            it.setIcon(QtGui.QIcon(pm))
            self.lst.addItem(it)
        if not page.coll:
            self.lst.addItem("(no collected spectra — click pixels and press Add, or choose a library)")
            self.lst.setEnabled(False)
        lay.addWidget(self.lst)
        self.cb_lib = lib_combo(page.store)
        lay.addLayout(hrow(W.QLabel("or all spectra (max 60) of library"), self.cb_lib, stretch_at=1))

    def endmembers(self):
        out = []
        for i in range(self.lst.count()):
            if i < len(self.coll) and self.lst.item(i).checkState() == Qt.Checked:
                s = self.coll[i]
                out.append(dict(name=s["name"], values=s["values"]))
        return out


class AnalysisDialog(W.QDialog):
    def __init__(self, page, op, expression=None, run=False):
        super().__init__(page)
        self.page, self.op = page, op
        a = page.active
        self.setWindowTitle(TITLES[op])
        self.setMinimumWidth(560 if op == "bandmath" else 500)
        lay = W.QVBoxLayout(self)
        lay.addWidget(small(f"Image: <b>{a['name']}</b> · {a['cols']}×{a['rows']}×{a['bands']}", rich=True))
        self.w = {}
        if op == "pca":
            cb = W.QComboBox()
            cb.addItems(["3", "4", "5", "6", "8", "10"])
            self.w["n"] = cb
            lay.addLayout(hrow(W.QLabel("Number of components"), cb, None))
            lay.addWidget(small("Shows PC1/PC2/PC3 as an RGB image, the scree plot and the first components."))
        elif op == "kmeans":
            k, it = W.QSpinBox(), W.QSpinBox()
            k.setRange(2, 20)
            k.setValue(6)
            it.setRange(3, 100)
            it.setValue(20)
            self.w.update(k=k, it=it)
            lay.addLayout(hrow(W.QLabel("Number of classes"), k, W.QLabel("Iterations"), it, None))
            lay.addWidget(small("Unsupervised clustering of all good bands. The class mean spectra can be saved to a library."))
        elif op in ("sam", "unmix"):
            self.em = EndmemberBox(page)
            lay.addWidget(self.em)
            if op == "sam":
                ang = W.QDoubleSpinBox()
                ang.setRange(0, 90)
                ang.setValue(0)
                self.w["ang"] = ang
                lay.addLayout(hrow(W.QLabel("Maximum angle (°, 0 = none)"), ang, None))
            else:
                s1 = W.QCheckBox("sum-to-one constraint")
                s1.setChecked(True)
                self.w["s1"] = s1
                lay.addWidget(s1)
                lay.addWidget(small("Non-negative least squares per pixel (decimated image). One abundance map per endmember, "
                                    "plus the dominant endmember and the fit error."))
        elif op == "bandmath":
            rng = f"{a.get('wavelength_min') or 0:.0f}–{a.get('wavelength_max') or 0:.0f} nm, {a['bands']} bands"
            lay.addWidget(small(f"Expression — <b>b</b>N = band number (1–{a['bands']}), <b>w</b>NNN = band nearest to that "
                                f"wavelength in nm ({rng})", rich=True))
            ed = W.QPlainTextEdit(expression or "(w800 - w670) / (w800 + w670)")
            ed.setMaximumHeight(80)
            self.w["expr"] = ed
            lay.addWidget(ed)
            pre = W.QComboBox()
            pre.addItem("—", "")
            for n, e in PRESETS:
                pre.addItem(n, e)
            pre.currentIndexChanged.connect(lambda _: pre.currentData() and ed.setPlainText(pre.currentData()))
            lay.addLayout(hrow(W.QLabel("Preset"), pre, None))
            lay.addWidget(small("Operators + - * / **, functions sqrt, log, log10, exp, abs, min, max, mean, where. "
                                "The bands actually used are listed with the result."))
        elif op == "anomaly":
            c = W.QSpinBox()
            c.setRange(2, 50)
            c.setValue(15)
            self.w["comp"] = c
            lay.addLayout(hrow(W.QLabel("PCA components used"), c, None))
            lay.addWidget(small("Global RX: Mahalanobis distance of every pixel from the scene background (in the leading "
                                "principal components)."))
        elif op == "snr":
            lay.addWidget(small("Nearest-neighbour estimate: the signal is the mean spectrum, the noise is the standard deviation "
                                "of the difference between neighbouring pixels. Uses the first rows of the scene."))
        self.err = small("")
        lay.addWidget(self.err)
        bb = W.QDialogButtonBox()
        self.go = bb.addButton("Run", W.QDialogButtonBox.AcceptRole)
        style.prop(self.go, ds3="primary", accent="image")
        bb.addButton(W.QDialogButtonBox.Cancel)
        bb.accepted.connect(self.run)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)
        if run:
            QtCore.QTimer.singleShot(0, self.run)

    def run(self):
        op, g = self.op, self.w
        params, body = {}, {}
        if op == "pca":
            params = dict(n_components=int(g["n"].currentText()))
        elif op == "kmeans":
            params = dict(k=g["k"].value(), iterations=g["it"].value())
        elif op in ("sam", "unmix"):
            params["endmembers"] = self.em.endmembers()
            lib = self.em.cb_lib.currentData()
            if lib:
                body["library_id"] = lib
            if not params["endmembers"] and not lib:
                return note_err(self.err, "Choose at least one spectrum or a library.")
            if op == "sam":
                params["max_angle_deg"] = g["ang"].value() or None
            else:
                params["sum_to_one"] = g["s1"].isChecked()
        elif op == "bandmath":
            params = dict(expression=g["expr"].toPlainText().strip())
        elif op == "anomaly":
            params = dict(components=g["comp"].value())
        body["params"] = params
        self.go.setEnabled(False)
        self.go.setText("Running…")
        note_err(self.err, "")

        def done(r):
            self.accept()
            ResultDialog(self.page, r).show()

        def fail(m):
            self.go.setEnabled(True)
            self.go.setText("Run")
            note_err(self.err, m)
        run_bg(self.page.api.post, f"/api/images/{self.page.active['id']}/analyze/{op}", body, done=done, failed=fail)


def colorbar(cmap, lo, hi, width=420):
    import matplotlib
    try:
        cm = matplotlib.colormaps[cmap or "viridis"]
    except Exception:                                               # noqa: BLE001
        cm = matplotlib.colormaps["viridis"]
    img = QtGui.QImage(256, 1, QtGui.QImage.Format_RGB32)
    for i in range(256):
        r, g, b, _ = cm(i / 255)
        img.setPixel(i, 0, QtGui.qRgb(int(r * 255), int(g * 255), int(b * 255)))
    w = W.QWidget()
    lay = W.QVBoxLayout(w)
    lay.setContentsMargins(0, 4, 0, 0)
    lay.setSpacing(1)
    lab = W.QLabel()
    lab.setPixmap(QtGui.QPixmap.fromImage(img.scaled(width, 12)))
    lay.addWidget(lab)
    lay.addLayout(hrow(small(fmt_num(lo)), None, small(fmt_num(hi))))
    w.setMaximumWidth(width)
    return w


def legend_widget(legend, unmatched=None):
    f = W.QWidget()
    fl = W.QGridLayout(f)
    fl.setContentsMargins(0, 4, 0, 0)
    fl.setHorizontalSpacing(14)
    for i, l in enumerate(legend):
        lab = W.QLabel(f"<span style='color:{l['color']}'>■</span> {l['name']} "
                       f"<span style='color:{C['muted']}'>{int(l.get('pixels') or 0):,} px</span>")
        lab.setTextFormat(Qt.RichText)
        fl.addWidget(lab, i // 3, i % 3)
    if unmatched:
        fl.addWidget(small(f"unclassified: {int(unmatched):,} px"), (len(legend) + 2) // 3, 0)
    return f


class ResultDialog(W.QDialog):
    """Result of an analysis: image, colour bar or legend, plot, statistics, further images, actions."""

    def __init__(self, page, r):
        super().__init__(page)
        self.page, self.r = page, r
        self.setWindowTitle(r.get("title") or "Result")
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.resize(min(980, max(640, (r.get("width") or 0) + 120)), 820)
        outer = W.QVBoxLayout(self)
        sc = W.QScrollArea()
        sc.setWidgetResizable(True)
        body = W.QWidget()
        lay = W.QVBoxLayout(body)
        sc.setWidget(body)
        outer.addWidget(sc, 1)
        self.pm = pixmap_from_data_url(r["image"]) if r.get("image") else None
        if self.pm is not None and not self.pm.isNull():
            lab = W.QLabel()
            lab.setAlignment(Qt.AlignCenter)
            lab.setStyleSheet("background:#222; border:1px solid #6d7b8e;")
            pm = self.pm
            if pm.width() < 420:
                f = max(1, 420 // max(1, pm.width()))
                pm = pm.scaled(pm.width() * f, pm.height() * f, Qt.KeepAspectRatio, Qt.FastTransformation)
            elif pm.width() > 760:
                pm = pm.scaledToWidth(760, Qt.SmoothTransformation)
            lab.setPixmap(pm)
            lay.addWidget(lab)
        if r.get("kind") == "scalar" and r.get("range"):
            lay.addWidget(colorbar(r.get("cmap"), r["range"][0], r["range"][1]))
        if r.get("kind") == "classes" and r.get("legend"):
            lay.addWidget(legend_widget(r["legend"], r.get("unmatched")))
        pl = r.get("plot") or {}
        if pl.get("series"):
            lay.addWidget(section(pl.get("title") or "Plot"))
            ch = SpectrumChart(240)
            pal = ["#2563eb", "#dc2626", "#059669", "#d97706", "#7c3aed"]
            ch.plot([dict(name=s["name"], x=pl["x"], y=[np.nan if v is None else v for v in s["y"]], color=pal[i % len(pal)])
                     for i, s in enumerate(pl["series"])], xlabel=pl.get("xlabel") or "Wavelength (nm)",
                    ylabel=pl.get("ylabel") or "Value")
            lay.addWidget(ch)
        if r.get("table"):
            lay.addWidget(section("Statistics"))
            t = table(["Quantity", "Value"], 0)
            items = list(r["table"].items())
            t.setRowCount(len(items))
            for i, (k, v) in enumerate(items):
                t.setItem(i, 0, cell(k))
                t.setItem(i, 1, cell(fmt_num(v)))
            t.setMinimumHeight(min(320, 30 * len(items) + 30))
            lay.addWidget(t)
        if r.get("extra"):
            lay.addWidget(section("Further images (click to enlarge)"))
            g = W.QGridLayout()
            for i, e in enumerate(r["extra"]):
                epm = pixmap_from_data_url(e["image"])
                b = W.QToolButton()
                b.setIcon(QtGui.QIcon(epm))
                b.setIconSize(QtCore.QSize(150, 150))
                b.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
                rng = e.get("range")
                b.setText(e["name"] + (f"\n{fmt_num(rng[0])} … {fmt_num(rng[1])}" if rng else ""))
                b.clicked.connect(lambda _=False, e=e, epm=epm: ImageWindow(self, e["name"], epm, data_url_bytes(e["image"])).exec_())
                g.addWidget(b, i // 4, i % 4)
            lay.addLayout(g)
        lay.addWidget(small(f"Computed on a decimated copy of the scene (step {r.get('step')}); the full image is not modified."))
        lay.addStretch(1)
        row = []
        if r.get("kind") == "classes" and self.pm is not None:
            row.append(btn("Overlay on the image", self.overlay, kind="primary", accent="library"))
        if r.get("spectra"):
            row.append(btn(f"Add {len(r['spectra'])} spectra to the collected list", self.to_coll))
            row.append(btn("Save as library…", self.save_lib))
        if r.get("image"):
            row.append(btn("Save PNG", lambda: save_bytes(self, (r.get("title") or "result").replace(" ", "_") + ".png",
                                                           data_url_bytes(r["image"]), "PNG (*.png)")))
        row += [None, btn("Close", self.close)]
        outer.addLayout(hrow(*row))

    def overlay(self):
        self.page.set_overlay(self.pm, self.r.get("legend") or [], self.r.get("unmatched"))
        self.close()

    def to_coll(self):
        self.page.add_spectra_to_collected(self.r["spectra"])
        self.close()

    def save_lib(self):
        a = self.page.active
        d = SaveDialog(self, "Save spectra as library", f"{len(self.r['spectra'])} spectra", self.page.store,
                       f"{a['name']} {self.r.get('title') or ''}".strip(), "pixel")
        if not d.exec_():
            return
        v = d.values()
        wl = self.page.wl
        body = dict(library=v["library"], items=[dict(name=s["name"], wavelengths_nm=wl, values=s["values"], unit=s.get("unit"),
                                                      category=v["category"], origin="image_pixel",
                                                      acquisition_type=v["acquisition"]) for s in self.r["spectra"]])
        run_bg(self.page.api.post, "/api/spectra/save", body,
               done=lambda res: (self.page.store.refresh(),
                                 self.page.flash(f"Saved {res['saved']} spectra to library “{res['library']}”")),
               failed=lambda m: W.QMessageBox.warning(self, "Save", m))


# ------------------------------------------------------------------ Spectral Analyst / SAM map
class AnalystDialog(W.QDialog):
    def __init__(self, page, target, label):
        super().__init__(page)
        self.page, self.target = page, target
        self.setWindowTitle("Spectral Analyst")
        self.resize(620, 520)
        lay = W.QVBoxLayout(self)
        lay.addWidget(small(f"Unknown spectrum: {label} — compared with every spectrum of the library by spectral angle."))
        self.cb_lib = lib_combo(page.store, "— choose —")
        lay.addLayout(hrow(W.QLabel("Library"), self.cb_lib, stretch_at=1))
        self.ck = W.QCheckBox("ignore unit differences (compare shape only)")
        self.ck.setChecked(True)
        lay.addWidget(self.ck)
        self.tbl = table(["#", "Material", "Category", "Angle"], 1)
        lay.addWidget(self.tbl, 1)
        self.err = small("")
        lay.addWidget(self.err)
        self.results = []
        self.b_run = btn("Run", self.run, kind="primary", accent="image")
        self.b_show = btn("Show best 3 in the plot", self.show_best, kind="primary", accent="library")
        self.b_show.setEnabled(False)
        lay.addLayout(hrow(self.b_run, self.b_show, None, btn("Close", self.close)))

    def run(self):
        lib = self.cb_lib.currentData()
        if not lib:
            return note_err(self.err, "Choose a library.")
        body = dict(self.target, library_id=lib, ignore_unit=self.ck.isChecked(), top_k=15)
        self.b_run.setEnabled(False)
        note_err(self.err, "")

        def done(r):
            self.b_run.setEnabled(True)
            self.results = r.get("results", [])
            self.tbl.setRowCount(len(self.results))
            for i, m in enumerate(self.results):
                for c, v in enumerate((i + 1, m["material"], m.get("category") or "", f"{m['angle_deg']:.2f}°")):
                    self.tbl.setItem(i, c, cell(v))
            self.b_show.setEnabled(bool(self.results))
            self.err.setStyleSheet("")
            self.err.setText(r.get("note") or ("" if self.results else "No match."))

        def fail(m):
            self.b_run.setEnabled(True)
            note_err(self.err, m)
        run_bg(self.page.api.post, f"/api/images/{self.page.active['id']}/match", body, done=done, failed=fail)

    def show_best(self):
        self.page.set_library_curves([(m["spectrum_id"], m["material"]) for m in self.results[:3]])
        self.close()


class SamMapDialog(W.QDialog):
    """Classify the whole image with the spectra of a library (spectral angle) -> class overlay."""

    def __init__(self, page):
        super().__init__(page)
        self.page = page
        self.setWindowTitle("Classify with a library (SAM map)")
        self.setMinimumWidth(480)
        lay = W.QVBoxLayout(self)
        self.cb_lib = lib_combo(page.store, "— choose —")
        self.ang = W.QDoubleSpinBox()
        self.ang.setRange(0, 90)
        self.ang.setValue(0)
        self.lim = W.QSpinBox()
        self.lim.setRange(1, 200)
        self.lim.setValue(60)
        lay.addLayout(hrow(W.QLabel("Library"), self.cb_lib, stretch_at=1))
        lay.addLayout(hrow(W.QLabel("Maximum angle (°, 0 = none)"), self.ang, W.QLabel("Max. spectra"), self.lim, None))
        self.err = small("")
        lay.addWidget(self.err)
        self.b = btn("Run", self.run, kind="primary", accent="image")
        lay.addLayout(hrow(self.b, None, btn("Cancel", self.reject)))

    def run(self):
        lib = self.cb_lib.currentData()
        if not lib:
            return note_err(self.err, "Choose a library.")
        self.b.setEnabled(False)
        body = dict(library_id=lib, max_angle_deg=self.ang.value() or None, limit=self.lim.value())

        def done(r):
            self.accept()
            r.setdefault("title", "SAM classification with a library")
            r.setdefault("kind", "classes")
            r.setdefault("image", r.get("overlay"))
            if r.get("unit_mismatch"):
                r["table"] = {"library units different from the image": ", ".join(r["unit_mismatch"])}
            ResultDialog(self.page, r).show()

        def fail(m):
            self.b.setEnabled(True)
            note_err(self.err, m)
        run_bg(self.page.api.post, f"/api/images/{self.page.active['id']}/sam_map", body, done=done, failed=fail)


# ------------------------------------------------------------------ georeferencing with control points
class PickView(W.QGraphicsView):
    """The image preview (zoom / pan); a click gives (row, col)."""
    picked = QtCore.pyqtSignal(int, int)

    def __init__(self, pixmap, rows, cols):
        super().__init__()
        self.rows, self.cols = rows, cols
        self.sc = W.QGraphicsScene(self)
        self.setScene(self.sc)
        it = self.sc.addPixmap(pixmap)
        it.setTransform(QtGui.QTransform.fromScale(cols / pixmap.width(), rows / pixmap.height()))
        self.sc.setSceneRect(0, 0, cols, rows)
        self.setTransformationAnchor(W.QGraphicsView.AnchorUnderMouse)
        self.setBackgroundBrush(QtGui.QColor("#111"))
        self.setCursor(Qt.CrossCursor)
        self.marks = []
        self._p = None
        self._drag = False

    def fit(self):
        self.fitInView(self.sc.sceneRect(), Qt.KeepAspectRatio)

    def wheelEvent(self, e):
        f = 1.25 if e.angleDelta().y() > 0 else 0.8
        self.scale(f, f)

    def mousePressEvent(self, e):
        self._p = e.pos()
        self._drag = False

    def mouseMoveEvent(self, e):
        if self._p is not None and e.buttons() & Qt.LeftButton:
            d = e.pos() - self._p
            if self._drag or abs(d.x()) + abs(d.y()) > 4:
                self._drag = True
                self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - d.x())
                self.verticalScrollBar().setValue(self.verticalScrollBar().value() - d.y())
                self._p = e.pos()

    def mouseReleaseEvent(self, e):
        if self._p is not None and not self._drag and e.button() == Qt.LeftButton:
            p = self.mapToScene(e.pos())
            if 0 <= p.x() < self.cols and 0 <= p.y() < self.rows:
                self.picked.emit(int(p.y()), int(p.x()))
        self._p = None

    def set_marks(self, pts, pending=None):
        for m in self.marks:
            self.sc.removeItem(m)
        self.marks = []
        allp = [(p["row"], p["col"], str(i + 1), "#fbbf24") for i, p in enumerate(pts)]
        if pending:
            allp.append((pending[0], pending[1], "?", "#ef4444"))
        for r, c, t, col in allp:
            el = self.sc.addEllipse(-6, -6, 12, 12, QtGui.QPen(QtGui.QColor(col), 2))
            el.setFlag(W.QGraphicsItem.ItemIgnoresTransformations)
            el.setPos(c + 0.5, r + 0.5)
            tx = W.QGraphicsSimpleTextItem(t, el)
            tx.setBrush(QtGui.QColor(col))
            tx.setPos(7, -16)
            self.marks.append(el)


class GeorefDialog(W.QDialog):
    def __init__(self, page):
        super().__init__(page)
        self.page, self.api = page, page.api
        a = page.active
        self.setWindowTitle("Georeference — match points of the image with the map")
        self.resize(1240, 860)
        self.pts, self.pending, self.saved, self.fit_res = [], None, None, None
        lay = W.QVBoxLayout(self)
        self.top = small("<b>1.</b> Click a point on the <b>image</b> that you can also find on the map (a road crossing, a "
                         "building corner…). <b>2.</b> Click the <b>same point on the map</b>. Repeat for at least <b>3 points</b> "
                         "spread over the image (4–6 are better: the error of each point is shown). Wheel = zoom, drag = move.",
                         rich=True)
        lay.addWidget(self.top)
        sp = W.QSplitter()
        left = W.QWidget()
        ll = W.QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.addWidget(W.QLabel(f"<b>Image</b> · {a['name']}"))
        self.iv = PickView(page.base_pixmap, a["rows"], a["cols"])
        self.iv.picked.connect(self._img_pick)
        ll.addWidget(self.iv, 1)
        right = W.QWidget()
        rl = W.QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        self.ed_go = W.QLineEdit()
        self.ed_go.setPlaceholderText("go to: lat, lon")
        self.ed_go.returnPressed.connect(self.go_to)
        self.cb_prov = W.QComboBox()
        self.cb_prov.addItems(list(PROVIDERS))
        self.cb_prov.setCurrentText(ESRI)
        rl.addLayout(hrow(W.QLabel("<b>Map</b>"), self.cb_prov, None, self.ed_go, btn("Go", self.go_to)))
        self.map = MapView(ESRI)
        self.map.clicked.connect(self._map_pick)
        self.cb_prov.currentTextChanged.connect(self.map.set_provider)
        rl.addWidget(self.map, 1)
        sp.addWidget(left)
        sp.addWidget(right)
        sp.setSizes([600, 600])
        lay.addWidget(sp, 3)
        self.tbl = table(["#", "Row", "Col", "Latitude", "Longitude", "Error (m)", ""], 3)
        self.tbl.setMaximumHeight(200)
        lay.addWidget(self.tbl, 1)
        self.msg = small("")
        lay.addWidget(self.msg)
        self.b_apply = btn("Apply georeference", self.apply, kind="primary", accent="library")
        self.b_del = btn("Remove the saved georeference", self.remove_saved)
        lay.addLayout(hrow(self.b_apply, btn("Clear points", self.clear), self.b_del, None, btn("Close", self.accept)))
        self.b_del.setVisible(False)
        QtCore.QTimer.singleShot(0, self.iv.fit)
        run_bg(self.api.get, f"/api/images/{a['id']}/georef", done=self._loaded, failed=lambda m: self._loaded({}))

    def _loaded(self, g):
        self.saved = g.get("georef")
        if g.get("file_has_geo"):
            self.top.setText("<b style='color:#047857'>This image already carries map coordinates in its file</b> — use "
                             "<b>View map</b>. Control points are only needed for images without them.<br>" + self.top.text())
        if self.saved and self.saved.get("points"):
            self.pts = [dict(p) for p in self.saved["points"]]
        self.b_del.setVisible(bool(self.saved))
        corners = self.page.geo_corners()
        if corners:
            QtCore.QTimer.singleShot(50, lambda: self.map.fit(corners))
        elif self.pts:
            QtCore.QTimer.singleShot(50, lambda: self.map.fit([(p["lon"], p["lat"]) for p in self.pts]))
        self._refresh()

    def go_to(self):
        try:
            lat, lon = [float(x) for x in self.ed_go.text().replace(";", ",").split(",")[:2]]
        except ValueError:
            return note_err(self.msg, "Type latitude, longitude — e.g. 48.137, 11.575")
        self.map.center_on(lon, lat, 16)

    def _img_pick(self, row, col):
        self.pending = (row, col)
        self.msg.setStyleSheet("")
        self.msg.setText(f"Image point ({row}, {col}) — now click the same point on the map.")
        self._marks()

    def _map_pick(self, lon, lat):
        if not self.pending:
            return note_err(self.msg, "Click the point on the image first, then on the map.")
        self.pts.append(dict(row=self.pending[0], col=self.pending[1], lat=lat, lon=lon))
        self.pending = None
        self._refresh()

    def _marks(self):
        self.iv.set_marks(self.pts, self.pending)
        self.map.set_markers([(p["lon"], p["lat"], str(i + 1), "#fbbf24") for i, p in enumerate(self.pts)])

    def _refresh(self):
        self._marks()
        res = (self.fit_res or {}).get("residuals_m") or []
        self.tbl.setRowCount(len(self.pts))
        for i, p in enumerate(self.pts):
            for c, v in enumerate((i + 1, p["row"], p["col"], f"{p['lat']:.6f}", f"{p['lon']:.6f}",
                                   f"{res[i]:.1f}" if i < len(res) else "")):
                self.tbl.setItem(i, c, cell(v))
            self.tbl.setCellWidget(i, 6, btn("Remove", lambda i=i: self._remove(i)))
        if len(self.pts) >= 3:
            run_bg(self.api.post, f"/api/images/{self.page.active['id']}/georef/fit", dict(points=self.pts),
                   done=self._fitted, failed=lambda m: (setattr(self, "fit_res", None), note_err(self.msg, m)))
        else:
            self.fit_res = None
            self.msg.setStyleSheet("")
            self.msg.setText(f"{len(self.pts)} control point(s) — at least 3 are needed.")

    def _fitted(self, f):
        changed = (self.fit_res or {}).get("residuals_m") != f.get("residuals_m")
        self.fit_res = f
        self.msg.setStyleSheet(f"color:{C['emerald']};")
        px = f.get("pixel_size_m") or [0, 0]
        self.msg.setText(f"{f['n']} points · RMSE {f['rmse_m']:.1f} m · pixel ≈ {px[0]:.1f} × {px[1]:.1f} m"
                         + (" · exact fit with 3 points (add a 4th to check it)" if f.get("exact_fit") else ""))
        if changed:
            res = f.get("residuals_m") or []
            for i in range(min(len(res), self.tbl.rowCount())):
                self.tbl.setItem(i, 5, cell(f"{res[i]:.1f}"))

    def _remove(self, i):
        if 0 <= i < len(self.pts):
            self.pts.pop(i)
        self._refresh()

    def clear(self):
        self.pts, self.pending = [], None
        self._refresh()

    def apply(self):
        if len(self.pts) < 3:
            return note_err(self.msg, "At least 3 control points are needed.")

        def done(r):
            self.saved = r.get("georef")
            self.b_del.setVisible(True)
            self.page.after_georef(r.get("geo"))
            self.msg.setStyleSheet(f"color:{C['emerald']};")
            self.msg.setText(f"Georeference saved with the image (RMSE {r['georef']['rmse_m']:.1f} m). View map, ROI "
                             "latitude / longitude and the patch GeoTIFFs use it now.")
        run_bg(self.api.put, f"/api/images/{self.page.active['id']}/georef", dict(points=self.pts), done=done,
               failed=lambda m: note_err(self.msg, m))

    def remove_saved(self):
        if W.QMessageBox.question(self, "Georeference", "Remove the saved georeference of this image?") != W.QMessageBox.Yes:
            return
        run_bg(self.api.delete, f"/api/images/{self.page.active['id']}/georef",
               done=lambda _: (self.b_del.setVisible(False), setattr(self, "saved", None), self.page.after_georef(None),
                               self.msg.setText("The saved georeference was removed.")),
               failed=lambda m: note_err(self.msg, m))


# ------------------------------------------------------------------ map view of the image
class MapPanel(W.QWidget):
    """The image on the web map (by its corner coordinates); click the map to read a pixel."""
    pixel = QtCore.pyqtSignal(int, int)

    def __init__(self):
        super().__init__()
        lay = W.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.cb = W.QComboBox()
        self.cb.addItems(list(PROVIDERS))
        self.cb.setCurrentText(ESRI)
        self.sl = W.QSlider(Qt.Horizontal)
        self.sl.setRange(10, 100)
        self.sl.setValue(80)
        self.sl.setMaximumWidth(160)
        self.info = small("click the map to read a pixel")
        lay.addLayout(hrow(W.QLabel("Map"), self.cb, W.QLabel("  image opacity"), self.sl, self.info, stretch_at=4))
        self.map = MapView(ESRI)
        lay.addWidget(self.map, 1)
        self.cb.currentTextChanged.connect(self.map.set_provider)
        self.sl.valueChanged.connect(lambda v: (setattr(self.map, "opacity", v / 100), self.map.update()))
        self.map.clicked.connect(self._click)
        self.rows = self.cols = 0

    def show_image(self, pixmap, corners, rows, cols):
        self.rows, self.cols = rows, cols
        self.map.set_image(pixmap, corners, self.sl.value() / 100)
        QtCore.QTimer.singleShot(30, lambda: self.map.fit(corners))

    def _click(self, lon, lat):
        t = self.map.image_transform()
        if t is None:
            return
        inv, ok = t.inverted()
        if not ok:
            return
        p = inv.map(self.map.to_screen(lon, lat))
        w, h = self.map.image.width(), self.map.image.height()
        if 0 <= p.x() < w and 0 <= p.y() < h:
            row, col = int(p.y() / h * self.rows), int(p.x() / w * self.cols)
            self.info.setText(f"pixel ({row}, {col}) · {lat:.5f}, {lon:.5f}")
            self.pixel.emit(row, col)
        else:
            self.info.setText(f"{lat:.5f}, {lon:.5f} — outside the image")


# ------------------------------------------------------------------ ROI aggregation
class AggregateDialog(W.QDialog):
    """Pool the pixels of the ticked ROIs per class: mean / std / variance / min–max per band."""

    def __init__(self, page, rois):
        super().__init__(page)
        self.page, self.api, self.rois = page, page.api, rois
        self.res = None
        self.setWindowTitle("Aggregate ROIs (data aggregation)")
        self.resize(1000, 800)
        lay = W.QVBoxLayout(self)
        self.cb_mode = W.QComboBox()
        self.cb_mode.addItem("intersection — only the range every scene covers", "intersection")
        self.cb_mode.addItem("union — the full range (gaps where a scene does not reach)", "union")
        self.ed_step = W.QLineEdit()
        self.ed_step.setPlaceholderText("native")
        self.ed_step.setMaximumWidth(90)
        self.ck_mask = W.QCheckBox("mask bad bands")
        self.ck_mask.setChecked(True)
        lay.addLayout(hrow(W.QLabel("Wavelength grid"), self.cb_mode, W.QLabel("step (nm)"), self.ed_step, self.ck_mask,
                           btn("Aggregate", self.run, kind="primary", accent="image"), None))
        self.head = small(f"{len(rois)} ticked ROI(s) · pixels of the same class are pooled")
        lay.addWidget(self.head)
        self.stat = {}
        sr = [W.QLabel("<b>Show</b>")]
        g = W.QButtonGroup(self)
        for k, t in (("mean", "Mean"), ("std", "Std. deviation"), ("variance", "Variance"), ("minmax", "Min–Max")):
            rb = W.QRadioButton(t)
            g.addButton(rb)
            rb.toggled.connect(lambda on: on and self._plot())
            self.stat[k] = rb
            sr.append(rb)
        self.stat["mean"].setChecked(True)
        sr += [None, btn("Export CSV", lambda: self.export("csv")), btn("Export JSON", lambda: self.export("json")),
               btn("Export ENVI .sli", lambda: self.export("sli")),
               btn("Save class means to library…", self.save, kind="primary", accent="image")]
        lay.addLayout(hrow(*sr))
        self.chart = SpectrumChart(300, toolbar=True)
        lay.addWidget(self.chart, 2)
        self.note = small("")
        lay.addWidget(self.note)
        self.tbl = table(["Class", "ROIs", "Pixels", "Mean", "Std. deviation", "Variance", "Min", "Max"], 0)
        lay.addWidget(self.tbl, 1)
        self.foot = small("")
        lay.addWidget(self.foot)
        self.chart.message("Press Aggregate")
        QtCore.QTimer.singleShot(0, self.run)

    def body(self, save=None):
        b = dict(rois=[dict(image_id=o["image_id"], class_name=o["class_name"], polygon=o["polygon"]) for o in self.rois],
                 mode=self.cb_mode.currentData(), mask_bad_bands=self.ck_mask.isChecked())
        try:
            st = float(self.ed_step.text())
            if st > 0:
                b["step"] = st
        except ValueError:
            pass
        if save:
            b["save"] = save
        return b

    def run(self):
        self.chart.message("Aggregating…")

        def done(r):
            self.res = r
            self._plot()
        run_bg(self.api.post, "/api/aggregate", self.body(), done=done,
               failed=lambda m: (self.chart.message(m), note_err(self.foot, m)))

    def _plot(self):
        r = self.res
        if not r:
            return
        wl = r["wavelengths"]
        cls = r["classes"]
        names = list(dict.fromkeys(o["class_name"] for o in self.rois))
        col = lambda n: DS_COLORS[max(0, names.index(n) if n in names else 0) % len(DS_COLORS)]   # noqa: E731
        f = next(k for k, b in self.stat.items() if b.isChecked())
        nan = lambda a: [np.nan if v is None else v for v in a]                                  # noqa: E731
        ser = []
        if f == "mean":
            for o in self.rois:
                if o.get("values"):
                    x, y = xy(o["wavelengths"], o["values"])
                    ser.append(dict(name=None, x=x, y=y, color=col(o["class_name"]), lw=0.8, alpha=0.5))
            for c in cls:
                ser.append(dict(name=f"{c['name']} mean", x=wl, y=nan(c["mean"]), color=col(c["name"]), lw=2.4))
            yl, note = None, "Thin lines: every ticked ROI. Thick lines: the mean of each class."
        elif f == "minmax":
            for c in cls:
                ser.append(dict(name=f"{c['name']} min", x=wl, y=nan(c["min"]), color=col(c["name"]), ls="--"))
                ser.append(dict(name=f"{c['name']} max", x=wl, y=nan(c["max"]), color=col(c["name"])))
            yl, note = "Minimum and maximum", "Lowest and highest value of any pixel of the class, per band."
        else:
            for c in cls:
                y = np.asarray(nan(c["std"]), float)
                ser.append(dict(name=c["name"], x=wl, y=y * y if f == "variance" else y, color=col(c["name"])))
            yl = "Variance" if f == "variance" else "Standard deviation"
            note = "Per band, over all pixels of the class (sample statistics, n − 1)."
        named = [s for s in ser if s["name"]]
        self.chart.plot(ser, unit=(r.get("units") or [None])[0], ylabel=yl, legend=False)
        if named and len(named) <= 14:
            self.chart.ax.legend([l for l in self.chart.ax.get_lines() if not l.get_label().startswith("_")],
                                 [s["name"] for s in named], fontsize=7.5)
            self.chart.canvas.draw_idle()
        self.note.setText(note)

        def avg(a):
            v = [x for x in a if x is not None and np.isfinite(x)]
            return sum(v) / len(v) if v else None
        f4 = lambda v: "—" if v is None else f"{v:.4f}"                                           # noqa: E731
        self.tbl.setRowCount(len(cls))
        for i, c in enumerate(cls):
            var = [None if v is None else v * v for v in c["std"]]
            vals = (f"■ {c['name']}", sum(1 for o in self.rois if o["class_name"] == c["name"]), c["n_pixels"],
                    f4(avg(c["mean"])), f4(avg(c["std"])), f4(avg(var)), f4(avg(c["min"])), f4(avg(c["max"])))
            for k, v in enumerate(vals):
                it = cell(v)
                if k == 0:
                    it.setForeground(QtGui.QColor(col(c["name"])))
                self.tbl.setItem(i, k, it)
        mixed = (f" <b style='color:#b45309'>The scenes have different units ({', '.join(r.get('units') or [])}): the "
                 f"pooled values mix them.</b>" if r.get("mixed_units") else "")
        self.foot.setStyleSheet("")
        self.foot.setTextFormat(Qt.RichText)
        self.foot.setText(f"Each value is averaged over all {len(wl)} bands ({wl[0]:.0f}–{wl[-1]:.0f} nm).{mixed}"
                          if wl else "")

    def export(self, fmt):
        def done(res):
            data, name = res
            save_bytes(self, name or f"aggregated_signatures.{fmt}", data)
        run_bg(self.api.download, f"/api/aggregate/export/{fmt}", self.body(), done=done,
               failed=lambda m: note_err(self.foot, m))

    def save(self):
        d = SaveDialog(self, "Save class mean spectra to a library", f"{len((self.res or {}).get('classes') or [])} class mean "
                       "spectra will be stored in the database.", self.page.store, "Aggregated signatures", "pixel")
        if not d.exec_():
            return
        v = d.values()

        def done(r):
            self.res = r
            self._plot()
            sv = r.get("saved") or {}
            self.page.store.refresh()
            self.foot.setStyleSheet(f"color:{C['emerald']};")
            self.foot.setText(f"Saved {len(sv.get('saved') or [])} class signature(s) to “{sv.get('library')}”.")
        run_bg(self.api.post, "/api/aggregate", self.body(dict(library=v["library"], category=v["category"],
                                                               acquisition_type=v["acquisition"])),
               done=done, failed=lambda m: note_err(self.foot, m))
