"""Library tools of the native Library (main) page: import dialog, spectra picker, image-patch viewer,
Spectral Library Builder, Library Viewer and Resampling (all through the backend API)."""
from __future__ import annotations

from PyQt5 import QtCore, QtWidgets

from .. import style
from ..style import C
from .api import pixmap_from_data_url, run_bg
from .common import (ACQ, DS_COLORS, NONE_TOKEN, LibraryPicker, SpectrumChart, angle_deg, btn, card, fmt_num, hrow,
                     note_err, save_csv, small, xy)

W = QtWidgets
Qt = QtCore.Qt

SPEC_CACHE = {}          # spectrum id -> spectrum dict (wavelengths, values, measurement_unit ...)


class Hub:
    """Links the two native pages (the Builder takes the pixel / ROI selected on the Image page)."""
    image_page = None
    library_page = None


def get_spectrum(api, sid):
    sid = str(sid)
    if sid not in SPEC_CACHE:
        SPEC_CACHE[sid] = api.get(f"/api/spectra/{sid}")
    return SPEC_CACHE[sid]


def section(text):
    lab = W.QLabel(text)
    style.prop(lab, ds3="section")
    return lab


def table(headers, stretch=0):
    t = W.QTableWidget(0, len(headers))
    t.setHorizontalHeaderLabels(headers)
    t.verticalHeader().setVisible(False)
    t.setEditTriggers(W.QAbstractItemView.NoEditTriggers)
    t.setSelectionBehavior(W.QAbstractItemView.SelectRows)
    t.setAlternatingRowColors(True)
    hh = t.horizontalHeader()
    hh.setSectionResizeMode(W.QHeaderView.ResizeToContents)
    hh.setSectionResizeMode(stretch, W.QHeaderView.Stretch)
    return t


def cell(text, data=None, align=None):
    it = W.QTableWidgetItem("" if text is None else str(text))
    if data is not None:
        it.setData(Qt.UserRole, data)
    if align:
        it.setTextAlignment(align)
    return it


def busy(widget, on=True):
    if on:
        W.QApplication.setOverrideCursor(Qt.WaitCursor)
    else:
        W.QApplication.restoreOverrideCursor()


# ------------------------------------------------------------------ import files into a library
class ImportDialog(W.QDialog):
    """Files on this computer -> POST /api/import/path (the backend reads them in place)."""

    UNITS = ["(read from the file)", "reflectance", "radiance", "transmittance", "emissivity", "absorbance", "DN"]

    def __init__(self, parent, api, store, library_name=None, title="Add library (import file)"):
        super().__init__(parent)
        self.api, self.store, self.report = api, store, None
        self.setWindowTitle(title)
        self.setMinimumWidth(620)
        lay = W.QVBoxLayout(self)
        lay.addWidget(small("Spectra files of any supported format (CSV, TXT, ENVI .sli/.hdr, .sig, .asd, .xlsx, JSON, "
                            "USGS, ECOSTRESS …). An image file (ENVI .hdr, GeoTIFF …) is registered as an image and is "
                            "then available on the Image page."))
        self.files = W.QListWidget()
        self.files.setMaximumHeight(110)
        lay.addLayout(hrow(btn("Choose files…", self._choose, accent="library", kind="primary"),
                           btn("Clear", self.files.clear), None))
        lay.addWidget(self.files)
        g = W.QGridLayout()
        self.ed_lib = W.QLineEdit(library_name or "")
        self.ed_lib.setPlaceholderText("default: the name of the first file")
        if library_name:
            self.ed_lib.setReadOnly(True)
        self.ed_cat = W.QLineEdit()
        self.ed_cat.setPlaceholderText("optional, e.g. Minerals/Clays")
        self.cb_unit = W.QComboBox()
        self.cb_unit.setEditable(True)
        self.cb_unit.addItems(self.UNITS)
        self.ed_wl = W.QLineEdit()
        self.ed_wl.setPlaceholderText("only if the file has none, e.g. 430-860")
        self.ed_div = W.QLineEdit()
        self.ed_div.setPlaceholderText("e.g. 10000 when reflectance is stored × 10000")
        self.cb_acq = W.QComboBox()
        self.cb_acq.addItems(["—"] + ACQ[1:])
        for r, (t, w) in enumerate((("Library", self.ed_lib), ("Category", self.ed_cat), ("Unit", self.cb_unit),
                                    ("Wavelengths", self.ed_wl), ("Reflectance divisor", self.ed_div),
                                    ("Acquisition", self.cb_acq))):
            g.addWidget(W.QLabel(t), r, 0)
            g.addWidget(w, r, 1)
        lay.addLayout(g)
        self.msg = small("")
        lay.addWidget(self.msg)
        bb = W.QDialogButtonBox()
        self.ok = bb.addButton("Import", W.QDialogButtonBox.AcceptRole)
        style.prop(self.ok, ds3="primary", accent="library")
        bb.addButton(W.QDialogButtonBox.Close)
        bb.accepted.connect(self._go)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

    def _choose(self):
        paths, _ = W.QFileDialog.getOpenFileNames(self, "Choose spectra / library files", "",
                                                  "All files (*.*);;Spectra (*.csv *.txt *.sli *.hdr *.sig *.asd *.xlsx *.json)")
        for p in paths:
            if not self.files.findItems(p, Qt.MatchExactly):
                self.files.addItem(p)

    def paths(self):
        return [self.files.item(i).text() for i in range(self.files.count())]

    def _go(self):
        paths = self.paths()
        if not paths:
            return note_err(self.msg, "Choose at least one file.")
        unit = self.cb_unit.currentText().strip()
        acq = self.cb_acq.currentText()
        body = dict(paths=paths, library=self.ed_lib.text().strip() or None, category=self.ed_cat.text().strip() or None,
                    unit=None if unit.startswith("(") or not unit else unit,
                    wavelengths=self.ed_wl.text().strip() or None,
                    reflectance_divisor=self.ed_div.text().strip() or None, acquisition=None if acq == "—" else acq)
        self.ok.setEnabled(False)
        self.msg.setStyleSheet("")
        self.msg.setText(f"Reading {len(paths)} file(s)… large libraries can take a minute.")

        def done(rep):
            self.ok.setEnabled(True)
            self.report = rep
            bad = [f for f in rep.get("files", []) if f.get("status") == "error"]
            txt = (f"<b>{rep.get('imported', 0)}</b> spectra imported into <b>{rep.get('library')}</b>"
                   f" · {rep.get('images', 0)} image(s) · {rep.get('duplicates', 0)} duplicate(s) · "
                   f"{rep.get('skipped', 0)} skipped · {rep.get('errors', 0)} error(s)")
            if bad:
                txt += "<br>" + "<br>".join(f"{f['file']}: {f.get('detail') or ''}" for f in bad[:8])
            self.msg.setTextFormat(Qt.RichText)
            self.msg.setText(txt)
            self.store.refresh()

        def fail(m):
            self.ok.setEnabled(True)
            note_err(self.msg, m)
        run_bg(self.api.post, "/api/import/path", body, done=done, failed=fail)


# ------------------------------------------------------------------ browse / pick database spectra
class SpectraBrowser(W.QWidget):
    """Library + search + category + acquisition filters over /api/spectra/search, with ticks (max_pick)."""
    picked_changed = QtCore.pyqtSignal()

    def __init__(self, api, store, max_pick=12, default_lib="", limit=200, parent=None):
        super().__init__(parent)
        self.api, self.store, self.max_pick, self.limit = api, store, max_pick, limit
        self.picked = {}             # spectrum id -> hit
        self.hits = []
        self._seq = 0
        lay = W.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.cb_lib = W.QComboBox()
        self.ed_q = W.QLineEdit()
        self.ed_q.setPlaceholderText("search material name…")
        style.prop(self.ed_q, ds3="search")
        self.cb_cat = W.QComboBox()
        self.cb_acq = W.QComboBox()
        self.cb_acq.addItem("Any acquisition", "")
        for a in ACQ[1:]:
            self.cb_acq.addItem(a, a)
        self.cb_acq.addItem("not set", NONE_TOKEN)
        lay.addLayout(hrow(self.cb_lib, self.ed_q, stretch_at=1))
        lay.addLayout(hrow(self.cb_cat, self.cb_acq, stretch_at=0))
        self.list = W.QListWidget()
        self.list.setMinimumHeight(180)
        lay.addWidget(self.list, 1)
        self.info = small("")
        lay.addWidget(self.info)
        self._t = QtCore.QTimer(self)
        self._t.setSingleShot(True)
        self._t.setInterval(280)
        self._t.timeout.connect(self.search)
        self.fill_libs(default_lib)
        self.cb_lib.currentIndexChanged.connect(lambda _: (self._facets(), self.search()))
        self.ed_q.textChanged.connect(lambda _: self._t.start())
        self.cb_cat.currentIndexChanged.connect(lambda _: self.search())
        self.cb_acq.currentIndexChanged.connect(lambda _: self.search())
        self.list.itemChanged.connect(self._toggled)
        self._facets()
        self.search()

    def fill_libs(self, select=""):
        self.cb_lib.blockSignals(True)
        self.cb_lib.clear()
        self.cb_lib.addItem("All libraries", "")
        for l in self.store.libraries:
            self.cb_lib.addItem(f"{l['name']} ({l['spectra']})", l["id"])
        i = self.cb_lib.findData(select) if select else 0
        self.cb_lib.setCurrentIndex(max(0, i))
        self.cb_lib.blockSignals(False)

    def _facets(self):
        lid = self.cb_lib.currentData() or None
        self.cb_cat.blockSignals(True)
        self.cb_cat.clear()
        self.cb_cat.addItem("All categories", "")
        self.cb_cat.blockSignals(False)

        def done(f):
            self.cb_cat.blockSignals(True)
            for c in f.get("categories", []):
                p = c["path"]
                self.cb_cat.addItem(f"{'(no category)' if p == NONE_TOKEN else p} ({c['count']})", p)
            self.cb_cat.blockSignals(False)
        run_bg(self.api.get, "/api/facets", **({"library_id": lid} if lid else {}), done=done)

    def search(self):
        self._seq += 1
        seq = self._seq
        params = dict(limit=self.limit)
        if self.cb_lib.currentData():
            params["library_id"] = self.cb_lib.currentData()
        if self.ed_q.text().strip():
            params["q"] = self.ed_q.text().strip()
        if self.cb_cat.currentData():
            params["category_path"] = self.cb_cat.currentData()
        if self.cb_acq.currentData():
            params["acquisition"] = self.cb_acq.currentData()
        self.info.setText("Searching…")

        def done(r):
            if seq != self._seq:
                return
            self.hits = r.get("results", [])
            self._draw(r.get("total", len(self.hits)))

        def fail(m):
            if seq == self._seq:
                self.info.setText(m)
        run_bg(self.api.get, "/api/spectra/search", **params, done=done, failed=fail)

    def _draw(self, total):
        self.list.blockSignals(True)
        self.list.clear()
        for h in self.hits:
            rng = f"{fmt_num(h.get('wavelength_start'))}–{fmt_num(h.get('wavelength_end'))} nm"
            it = W.QListWidgetItem(f"{h['material']}   ·  {h.get('library', '')} · {h.get('unit') or ''} · {rng}"
                                   + (f" · {h['category']}" if h.get("category") else ""))
            it.setData(Qt.UserRole, h)
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.Checked if h["spectrum_id"] in self.picked else Qt.Unchecked)
            self.list.addItem(it)
        self.list.blockSignals(False)
        self._info(total)

    def _info(self, total=None):
        t = f"{len(self.hits)} of {total} shown · " if total is not None else ""
        self.info.setText(f"{t}{len(self.picked)} ticked (max {self.max_pick})")

    def _toggled(self, it):
        h = it.data(Qt.UserRole)
        sid = h["spectrum_id"]
        if it.checkState() == Qt.Checked:
            if len(self.picked) >= self.max_pick and sid not in self.picked:
                self.list.blockSignals(True)
                it.setCheckState(Qt.Unchecked)
                self.list.blockSignals(False)
                self.info.setText(f"At most {self.max_pick} spectra can be ticked.")
                return
            self.picked[sid] = h
        else:
            self.picked.pop(sid, None)
        self._info()
        self.picked_changed.emit()

    def clear_picks(self):
        self.picked.clear()
        self._draw(None)
        self.picked_changed.emit()

    def tick_all_shown(self):
        for h in self.hits:
            if len(self.picked) >= self.max_pick:
                break
            self.picked[h["spectrum_id"]] = h
        self._draw(None)
        self.picked_changed.emit()


class SpectraPickerDialog(W.QDialog):
    def __init__(self, parent, api, store, title="Choose spectra", max_pick=200, default_lib="", ok_text="Add"):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(760, 560)
        lay = W.QVBoxLayout(self)
        self.browser = SpectraBrowser(api, store, max_pick=max_pick, default_lib=default_lib, limit=500)
        lay.addWidget(self.browser, 1)
        bb = W.QDialogButtonBox()
        all_b = bb.addButton("Tick all shown", W.QDialogButtonBox.ActionRole)
        all_b.clicked.connect(self.browser.tick_all_shown)
        ok = bb.addButton(ok_text, W.QDialogButtonBox.AcceptRole)
        style.prop(ok, ds3="primary", accent="library")
        bb.addButton(W.QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

    def hits(self):
        return list(self.browser.picked.values())


# ------------------------------------------------------------------ the saved image patch of an object sample
class PatchDialog(W.QDialog):
    """View the image patch (<class>.tif) of an object sample: bands, click a pixel for its spectrum (view only)."""

    def __init__(self, parent, api, sample_id):
        super().__init__(parent)
        self.api, self.sid, self.d = api, sample_id, None
        self.setWindowTitle("Image patch")
        self.resize(860, 760)
        lay = W.QVBoxLayout(self)
        self.head = small("Loading…")
        lay.addWidget(self.head)
        self.cbs = [W.QComboBox() for _ in range(3)]
        row = [W.QLabel("R"), self.cbs[0], W.QLabel("G"), self.cbs[1], W.QLabel("B"), self.cbs[2],
               btn("Apply bands", self.apply), None, btn("Locate file", self.locate)]
        lay.addLayout(hrow(*row))
        self.view = W.QLabel()
        self.view.setAlignment(Qt.AlignCenter)
        self.view.setStyleSheet("background:#000; border-radius:8px;")
        self.view.setMinimumHeight(320)
        self.view.setCursor(Qt.CrossCursor)
        self.view.mousePressEvent = self._click
        lay.addWidget(self.view, 1)
        self.chart = SpectrumChart(210)
        self.chart.message("Click a pixel to see its spectrum")
        lay.addWidget(self.chart)
        self.load()

    def load(self, bands=None):
        q = dict(r=bands[0], g=bands[1], b=bands[2]) if bands else {}

        def done(d):
            self.d = d
            wl = d.get("wavelengths") or []
            self.setWindowTitle(f"Image patch — {d['name']}")
            self.head.setText(f"{d['name']} · {d['rows']}×{d['cols']} pixels · {d['bands']} bands · {d.get('unit') or ''}"
                              + (f" · {round(wl[0])}–{round(wl[-1])} nm" if wl else "") + " · for viewing only")
            sel = (d.get("preview") or {}).get("bands") or [1, 1, 1]
            for k, cb in enumerate(self.cbs):
                cb.blockSignals(True)
                cb.clear()
                for i, w in enumerate(wl):
                    cb.addItem(f"Band {i + 1} · {round(w)} nm", i + 1)
                cb.setCurrentIndex(max(0, int(sel[k]) - 1))
                cb.blockSignals(False)
            self.pm = pixmap_from_data_url(d["preview"]["image"])
            self._show()

        run_bg(self.api.get, f"/api/samples/{self.sid}/patch/view", **q, done=done,
               failed=lambda m: self.head.setText(m))

    def _show(self):
        if not self.d:
            return
        f = max(1, min(12, 500 // max(self.d["rows"], self.d["cols"], 1)))
        self.scaled = self.pm.scaled(self.pm.width() * f, self.pm.height() * f, Qt.KeepAspectRatio, Qt.FastTransformation)
        self.view.setPixmap(self.scaled)

    def _click(self, e):
        if not self.d or self.view.pixmap() is None:
            return
        pm = self.view.pixmap()
        x0 = (self.view.width() - pm.width()) / 2
        y0 = (self.view.height() - pm.height()) / 2
        fx, fy = (e.x() - x0) / pm.width(), (e.y() - y0) / pm.height()
        if not (0 <= fx < 1 and 0 <= fy < 1):
            return
        row, col = int(fy * self.d["rows"]), int(fx * self.d["cols"])

        def done(s):
            x, y = xy(s["wavelengths"], s["values"])
            self.chart.plot([dict(name=f"pixel ({row}, {col})", x=x, y=y, color="#2563eb")], unit=s.get("unit"))
        run_bg(self.api.get, f"/api/samples/{self.sid}/patch/pixel", row=row, col=col, done=done,
               failed=self.chart.message)

    def apply(self):
        self.load([cb.currentData() or 1 for cb in self.cbs])

    def locate(self):
        run_bg(self.api.post, f"/api/samples/{self.sid}/patch/reveal", {}, failed=lambda m: self.head.setText(m))


# ------------------------------------------------------------------ Spectral Library Builder
class SourceDialog(W.QDialog):
    """Output wavelengths from a data file (image or library) with an optional band subset."""

    def __init__(self, parent, api):
        super().__init__(parent)
        self.api, self.result = api, None
        self.setWindowTitle("Input Data File — output wavelengths")
        self.resize(640, 460)
        lay = W.QVBoxLayout(self)
        self.list = W.QListWidget()
        lay.addWidget(self.list, 1)
        self.info = small("Loading…")
        lay.addWidget(self.info)
        self.sp_from, self.sp_to = W.QSpinBox(), W.QSpinBox()
        for s in (self.sp_from, self.sp_to):
            s.setRange(1, 1)
        lay.addLayout(hrow(W.QLabel("Spectral subset: bands"), self.sp_from, W.QLabel("to"), self.sp_to, None))
        self.err = small("")
        lay.addWidget(self.err)
        bb = W.QDialogButtonBox(W.QDialogButtonBox.Ok | W.QDialogButtonBox.Cancel)
        bb.accepted.connect(self._ok)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)
        self.list.currentRowChanged.connect(self._sel)
        run_bg(api.get, "/api/builder/sources", done=self._fill, failed=lambda m: note_err(self.err, m))

    def _fill(self, src):
        self.src = src
        for s in src:
            it = W.QListWidgetItem(f"{s['name']}  —  {s['type']} · {s['bands']} bands · "
                                   f"{fmt_num(s.get('wl_min'))}–{fmt_num(s.get('wl_max'))} nm · {s.get('info') or ''}")
            it.setData(Qt.UserRole, s)
            self.list.addItem(it)
        self.info.setText(f"{len(src)} data files (images and libraries in the database)" if src else
                          "No image or library yet: open an image on the Image page or import a library first.")

    def _sel(self, i):
        if i < 0:
            return
        s = self.list.item(i).data(Qt.UserRole)
        n = int(s.get("bands") or 1)
        for sp in (self.sp_from, self.sp_to):
            sp.setRange(1, n)
        self.sp_from.setValue(1)
        self.sp_to.setValue(n)

    def _ok(self):
        it = self.list.currentItem()
        if not it:
            return note_err(self.err, "Choose a data file.")
        s = it.data(Qt.UserRole)
        try:
            w = self.api.get(f"/api/builder/sources/{s['kind']}/{s['id']}/wavelengths",
                             band_from=self.sp_from.value(), band_to=self.sp_to.value())
        except Exception as e:                                       # noqa: BLE001
            return note_err(self.err, str(e))
        sub = self.sp_to.value() - self.sp_from.value() + 1 < int(s.get("bands") or 0)
        self.result = (w["wavelengths"], f"{s['kind']} {s['name']}" +
                       (f" (bands {self.sp_from.value()}–{self.sp_to.value()})" if sub else ""))
        self.accept()


class BuilderView(W.QWidget):
    def __init__(self, page):
        super().__init__()
        self.page, self.api, self.store = page, page.api, page.store
        self.grid, self.grid_src, self.items, self.rejected = None, "", [], []
        outer = W.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        sc = W.QScrollArea()
        sc.setWidgetResizable(True)
        sc.setFrameShape(W.QFrame.NoFrame)
        outer.addWidget(sc)
        body = W.QWidget()
        sc.setWidget(body)
        lay = W.QVBoxLayout(body)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.setSpacing(10)

        f1, l1 = card()
        l1.addWidget(section("1 · Output wavelengths"))
        self.lab_grid = small("")
        l1.addLayout(hrow(btn("Data file…", self.grid_data, tip="Wavelengths of an image or library (band subset)"),
                          btn("ASCII file…", self.grid_ascii, tip="A text file with one wavelength per line"),
                          btn("First input spectrum", self.grid_first), None))
        l1.addWidget(self.lab_grid)
        self.ck_water = W.QCheckBox("Mask water-absorption bands (1350–1450, 1800–1950, > 2450 nm)")
        self.sp_cov = W.QDoubleSpinBox()
        self.sp_cov.setRange(0.05, 1.0)
        self.sp_cov.setSingleStep(0.05)
        self.sp_cov.setValue(0.5)
        l1.addLayout(hrow(self.ck_water, None, W.QLabel("Minimum coverage"), self.sp_cov))
        lay.addWidget(f1)

        f2, l2 = card()
        l2.addWidget(section("2 · Add spectra"))
        l2.addLayout(hrow(btn("Spectra file(s)…", self.add_files, kind="primary", accent="library",
                              tip="CSV, TXT, SLI, SIG, ASD, XLSX, JSON … — read, not stored"),
                          btn("Library spectra…", self.add_library),
                          btn("From image (pixel / ROI)", self.add_image,
                              tip="The pixel clicked or the ROI drawn on the Image page"),
                          None, btn("Remove selected", self.remove_sel), btn("Reset", self.reset)))
        self.lab_items = small("")
        l2.addWidget(self.lab_items)
        self.tbl = table(["Name", "Unit", "Coverage", "Origin", "Source"], 0)
        self.tbl.setMinimumHeight(160)
        self.tbl.setSelectionMode(W.QAbstractItemView.ExtendedSelection)
        l2.addWidget(self.tbl)
        self.chart = SpectrumChart(260, toolbar=True)
        l2.addWidget(self.chart)
        self.lab_rej = small("")
        l2.addWidget(self.lab_rej)
        lay.addWidget(f2)

        f3, l3 = card()
        l3.addWidget(section("Match (spectral angle to a reference)"))
        self.cb_ref = W.QComboBox()
        self.cb_ref.currentIndexChanged.connect(lambda _: self._match())
        l3.addLayout(hrow(W.QLabel("Reference"), self.cb_ref, stretch_at=1))
        self.tbl_match = table(["Spectrum", "Angle (°)", "Bands used"], 0)
        self.tbl_match.setMaximumHeight(180)
        l3.addWidget(self.tbl_match)
        lay.addWidget(f3)

        f4, l4 = card()
        l4.addWidget(section("3 · Save"))
        self.picker = LibraryPicker(self.store, "", "pixel")
        self.ed_cat = W.QLineEdit()
        self.ed_cat.setPlaceholderText("category (optional)")
        self.cb_acq = W.QComboBox()
        self.cb_acq.addItems(["—"] + ACQ[1:])
        self.ck_raw = W.QCheckBox("Also keep the original file spectra in a “<name> (raw)” library")
        self.ck_raw.setChecked(True)
        l4.addWidget(self.picker)
        l4.addLayout(hrow(self.ed_cat, self.cb_acq, stretch_at=0))
        l4.addWidget(self.ck_raw)
        self.msg = small("")
        l4.addWidget(self.msg)
        l4.addLayout(hrow(btn("Save to library", self.save, kind="primary", accent="library"),
                          btn("Export CSV", self.export), None))
        lay.addWidget(f4)
        lay.addStretch(1)
        self.store.changed.connect(self.picker.fill)
        self._redraw()

    # ---- grid
    def grid_data(self):
        d = SourceDialog(self, self.api)
        if d.exec_() and d.result:
            self.grid, self.grid_src = d.result
            self._regrid()

    def grid_ascii(self):
        p, _ = W.QFileDialog.getOpenFileName(self, "ASCII wavelength file", "", "Text (*.txt *.csv *.asc *.dat);;All (*.*)")
        if not p:
            return

        def done(r):
            self.grid = r["wavelengths"]
            self.grid_src = f"ASCII file {r['name']}" + (" (µm converted to nm)" if r.get("converted_from_um") else "")
            self._regrid()
        run_bg(self.api.upload, "/api/builder/wavelengths/ascii", [("file", p)], done=done,
               failed=lambda m: note_err(self.msg, m))

    def grid_first(self):
        self.grid, self.grid_src = None, "first input spectrum"
        self._regrid()

    def _regrid(self):
        """A new grid: resample the spectra already in the list again (from their originals)."""
        raws = [dict(it["raw"], name=it["name"], origin=it["origin"], source=it["source"]) for it in self.items if it.get("raw")]
        self.items = []
        if raws:
            self._add(raws)
        else:
            self._redraw()

    # ---- adding spectra
    def _add(self, raw, origin=None):
        if not raw:
            return
        body = dict(spectra=[dict(name=s["name"], wavelengths=s["wavelengths"], values=s["values"],
                                  unit=s.get("unit") or "reflectance", origin=origin or s.get("origin") or "file",
                                  source=s.get("source") or "") for s in raw],
                    grid=self.grid, mask_water=self.ck_water.isChecked(), min_coverage=self.sp_cov.value())
        self.lab_items.setText("Resampling…")

        def done(r):
            if not self.grid and r.get("grid"):
                self.grid, self.grid_src = r["grid"], "first input spectrum"
            q = {}
            for s in raw:
                q.setdefault(s["name"], []).append(s)
            for s in r.get("spectra", []):
                src = (q.get(s["name"]) or [None]).pop(0) if q.get(s["name"]) else None
                s["raw"] = dict(wavelengths=src["wavelengths"], values=src["values"], unit=src.get("unit") or "reflectance") if src else None
                self.items.append(s)
            self.rejected += r.get("rejected", [])
            self._redraw()

        def fail(m):
            note_err(self.msg, m)
            self._redraw()
        run_bg(self.api.post, "/api/builder/resample", body, done=done, failed=fail)

    def add_files(self):
        paths, _ = W.QFileDialog.getOpenFileNames(self, "Spectra files", "", "All files (*.*)")
        if not paths:
            return

        def done(r):
            self.rejected += [dict(name=e["file"], reason=e["reason"]) for e in r.get("errors", [])]
            if r.get("spectra"):
                self._add(r["spectra"], "file")
            else:
                self._redraw()
        run_bg(self.api.upload, "/api/builder/parse", [("files", p) for p in paths], done=done,
               failed=lambda m: note_err(self.msg, m))

    def add_library(self):
        d = SpectraPickerDialog(self, self.api, self.store, "Library spectra for the builder", max_pick=500)
        if not d.exec_():
            return
        hits = d.hits()
        if not hits:
            return

        def load():
            out = []
            for h in hits:
                sp = get_spectrum(self.api, h["spectrum_id"])
                out.append(dict(name=h["material"], wavelengths=sp["wavelengths"], values=sp["values"],
                                unit=sp.get("measurement_unit"), source=h.get("library") or "", origin="library"))
            return out
        self.lab_items.setText(f"Reading {len(hits)} spectra…")
        run_bg(load, done=lambda raw: self._add(raw, "library"), failed=lambda m: note_err(self.msg, m))

    def add_image(self):
        ip = Hub.image_page
        sel = ip.current_selection() if ip is not None and hasattr(ip, "current_selection") else None
        if not sel:
            return note_err(self.msg, "On the Image page open an image and click a pixel or finish an ROI first.")
        self._add([sel])

    def remove_sel(self):
        rows = sorted({i.row() for i in self.tbl.selectedIndexes()}, reverse=True)
        for r in rows:
            if 0 <= r < len(self.items):
                self.items.pop(r)
        self._redraw()

    def reset(self):
        if self.items and W.QMessageBox.question(self, "Builder", "Remove all spectra and start again?") != W.QMessageBox.Yes:
            return
        self.items, self.rejected, self.grid, self.grid_src = [], [], None, ""
        self._redraw()

    # ---- drawing
    def _redraw(self):
        g = self.grid
        self.lab_grid.setText(f"<b>{len(g)}</b> wavelengths, {fmt_num(min(g))}–{fmt_num(max(g))} nm · from {self.grid_src}"
                              if g else "The wavelengths are taken from the first spectrum added (First Input Spectrum)."
                              if not self.grid_src or self.grid_src.startswith("first") else self.grid_src)
        self.lab_grid.setTextFormat(Qt.RichText)
        self.tbl.setRowCount(len(self.items))
        for r, it in enumerate(self.items):
            for c, v in enumerate((it["name"], it.get("unit"), f"{100 * (it.get('coverage') or 0):.0f} %",
                                   it.get("origin"), it.get("source"))):
                self.tbl.setItem(r, c, cell(v))
        self.lab_items.setText(f"{len(self.items)} spectra in the list")
        if g and self.items:
            ser = []
            for k, it in enumerate(self.items[:40]):
                x, y = xy(g, it["values"])
                ser.append(dict(name=it["name"], x=x, y=y, color=DS_COLORS[k % len(DS_COLORS)], lw=1.2))
            units = {it.get("unit") for it in self.items}
            self.chart.plot(ser, unit=units.pop() if len(units) == 1 else None)
        else:
            self.chart.message("Add spectra: files, library spectra or a pixel / ROI of an image")
        self.lab_rej.setText(("<b>Rejected</b><br>" + "<br>".join(f"{x['name']}: {x['reason']}" for x in self.rejected[-30:]))
                             if self.rejected else "")
        self.lab_rej.setTextFormat(Qt.RichText)
        cur = self.cb_ref.currentText()
        self.cb_ref.blockSignals(True)
        self.cb_ref.clear()
        self.cb_ref.addItem("—", -1)
        for k, it in enumerate(self.items):
            self.cb_ref.addItem(it["name"], k)
        self.cb_ref.setCurrentIndex(max(0, self.cb_ref.findText(cur)))
        self.cb_ref.blockSignals(False)
        self._match()

    def _match(self):
        k = self.cb_ref.currentData()
        self.tbl_match.setRowCount(0)
        if k is None or k < 0 or k >= len(self.items):
            return
        ref = self.items[k]
        rows = []
        for j, it in enumerate(self.items):
            if j == k:
                continue
            a = angle_deg(ref["values"], it["values"])
            n = sum(1 for x, y in zip(ref["values"], it["values"]) if x is not None and y is not None)
            if a is not None:
                rows.append((a, it["name"], n))
        rows.sort()
        self.tbl_match.setRowCount(len(rows))
        for r, (a, nm, n) in enumerate(rows):
            self.tbl_match.setItem(r, 0, cell(nm))
            self.tbl_match.setItem(r, 1, cell(f"{a:.2f}"))
            self.tbl_match.setItem(r, 2, cell(n))

    # ---- output
    def export(self):
        if not self.grid or not self.items:
            return note_err(self.msg, "Nothing to export.")
        rows = [[w] + ["" if it["values"][k] is None else it["values"][k] for it in self.items] for k, w in enumerate(self.grid)]
        save_csv(self, "spectral_library_builder.csv", ["wavelength_nm"] + [it["name"] for it in self.items], rows)

    def save(self):
        if not self.items or not self.grid:
            return note_err(self.msg, "Nothing to save.")
        name = self.picker.value()
        if not name:
            return note_err(self.msg, "Give the library a name or choose an existing one.")
        cat = self.ed_cat.text().strip() or None
        acq = None if self.cb_acq.currentText() == "—" else self.cb_acq.currentText()
        items, grid, src, keep = list(self.items), list(self.grid), self.grid_src, self.ck_raw.isChecked()

        def work():
            files = [i for i in items if i.get("origin") == "file" and i.get("raw")]
            libs = sorted({i.get("source") for i in items if i.get("origin") == "library" and i.get("source")})
            raw_name, rr = None, None
            if keep and files:
                raw_name = f"{name} (raw)"
                rr = self.api.post("/api/spectra/save", dict(
                    library=raw_name,
                    library_info=dict(type="spectral", raw=True, modified_library=name,
                                      source_files=sorted({i.get("source") for i in files if i.get("source")}),
                                      note="Original spectra exactly as read from the source file(s): no resampling."),
                    items=[dict(name=i["name"], wavelengths_nm=i["raw"]["wavelengths"], values=i["raw"]["values"],
                                unit=i["raw"].get("unit") or i.get("unit"), category=cat, origin="raw_file",
                                acquisition_type=acq) for i in files]))
            info = dict(type="spectral", modified=True, grid=src or "first input spectrum",
                        built_with="Spectral Library Builder")
            if libs:
                info["derived_from_libraries"] = libs
            if raw_name:
                info["raw_library"] = raw_name
            r = self.api.post("/api/spectra/save", dict(library=name, library_info=info, items=[
                dict(name=i["name"], wavelengths_nm=grid, values=i["values"], unit=i.get("unit"), category=cat,
                     origin="builder", acquisition_type=acq) for i in items]))
            return r, rr

        def done(res):
            r, rr = res
            self.rejected += r.get("rejected") or []
            extra = f" and {rr['saved']} original spectra to “{rr['library']}”" if rr else ""
            self.msg.setStyleSheet(f"color:{C['emerald']};")
            self.msg.setText(f"Saved {r['saved']} spectra to library “{r['library']}”{extra}.")
            self.store.refresh()
        self.msg.setStyleSheet("")
        self.msg.setText("Saving…")
        run_bg(work, done=done, failed=lambda m: note_err(self.msg, m))


# ------------------------------------------------------------------ Library Viewer
class ViewerView(W.QWidget):
    def __init__(self, page):
        super().__init__()
        self.page, self.api = page, page.api
        lay = W.QHBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        left = W.QVBoxLayout()
        left.addWidget(section("Library Viewer"))
        left.addWidget(small("Tick up to 12 spectra to plot them together."))
        self.browser = SpectraBrowser(self.api, page.store, max_pick=12)
        left.addWidget(self.browser, 1)
        left.addLayout(hrow(btn("Clear ticks", self.browser.clear_picks), None, btn("Export CSV", self.export)))
        lw = W.QWidget()
        lw.setLayout(left)
        lw.setMinimumWidth(380)
        self.chart = SpectrumChart(420, toolbar=True)
        sp = W.QSplitter()
        sp.addWidget(lw)
        sp.addWidget(self.chart)
        sp.setStretchFactor(1, 1)
        lay.addWidget(sp)
        self.browser.picked_changed.connect(self.redraw)
        self.chart.message("Tick spectra on the left")
        self.data = []

    def redraw(self):
        hits = list(self.browser.picked.values())
        if not hits:
            self.data = []
            return self.chart.message("Tick spectra on the left")

        def load():
            return [(h, get_spectrum(self.api, h["spectrum_id"])) for h in hits]

        def done(rows):
            self.data = rows
            ser = []
            for k, (h, sp) in enumerate(rows):
                x, y = xy(sp["wavelengths"], sp["values"])
                ser.append(dict(name=f"{h['material']} ({h.get('library', '')})", x=x, y=y, color=DS_COLORS[k % len(DS_COLORS)]))
            units = {sp.get("measurement_unit") for _, sp in rows}
            self.chart.plot(ser, unit=units.pop() if len(units) == 1 else None,
                            title=None if len(units) <= 1 else "different units — scales are not comparable")
        run_bg(load, done=done, failed=self.chart.message)

    def export(self):
        if not self.data:
            return
        rows = []
        for h, sp in self.data:
            for w, v in zip(sp["wavelengths"], sp["values"]):
                rows.append([h["material"], h.get("library", ""), sp.get("measurement_unit") or "", w, "" if v is None else v])
        save_csv(self, "library_viewer_spectra.csv", ["material", "library", "unit", "wavelength_nm", "value"], rows)


# ------------------------------------------------------------------ Resampling
class ResampleView(W.QWidget):
    def __init__(self, page):
        super().__init__()
        self.page, self.api, self.store = page, page.api, page.store
        lay = W.QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        f, l = card()
        l.addWidget(section("Resample a library to a new wavelength grid"))
        l.addWidget(small("Every spectrum of the library is linearly resampled into a NEW library (the original is never "
                          "changed). No extrapolation; gaps are not bridged."))
        g = W.QGridLayout()
        self.cb_lib = W.QComboBox()
        self.rb = {k: W.QRadioButton(t) for k, t in (("range", "Regular range"), ("image", "Bands of an image"),
                                                      ("list", "List of wavelengths"))}
        self.rb["range"].setChecked(True)
        self.sp_a, self.sp_b, self.sp_s = W.QDoubleSpinBox(), W.QDoubleSpinBox(), W.QDoubleSpinBox()
        for s, v in ((self.sp_a, 400), (self.sp_b, 2500), (self.sp_s, 10)):
            s.setRange(0.01, 100000)
            s.setDecimals(2)
            s.setValue(v)
        self.cb_img = W.QComboBox()
        self.ed_list = W.QLineEdit()
        self.ed_list.setPlaceholderText("e.g. 450, 550, 650, 860 (nm, comma or space separated)")
        self.ck_water = W.QCheckBox("Mask water-absorption bands")
        self.sp_cov = W.QDoubleSpinBox()
        self.sp_cov.setRange(0.05, 1.0)
        self.sp_cov.setSingleStep(0.05)
        self.sp_cov.setValue(0.5)
        self.ed_name = W.QLineEdit()
        self.ed_name.setPlaceholderText("name of the new library")
        g.addWidget(W.QLabel("Library"), 0, 0)
        g.addWidget(self.cb_lib, 0, 1, 1, 5)
        g.addWidget(self.rb["range"], 1, 0)
        g.addLayout(hrow(W.QLabel("from"), self.sp_a, W.QLabel("to"), self.sp_b, W.QLabel("step"), self.sp_s, None), 1, 1, 1, 5)
        g.addWidget(self.rb["image"], 2, 0)
        g.addWidget(self.cb_img, 2, 1, 1, 5)
        g.addWidget(self.rb["list"], 3, 0)
        g.addWidget(self.ed_list, 3, 1, 1, 5)
        g.addWidget(W.QLabel("Minimum coverage"), 4, 0)
        g.addLayout(hrow(self.sp_cov, self.ck_water, None), 4, 1, 1, 5)
        g.addWidget(W.QLabel("New library"), 5, 0)
        g.addWidget(self.ed_name, 5, 1, 1, 5)
        l.addLayout(g)
        self.msg = small("")
        l.addWidget(self.msg)
        l.addLayout(hrow(btn("Resample", self.run, kind="primary", accent="library"), None))
        lay.addWidget(f)
        lay.addStretch(1)
        self.cb_lib.currentIndexChanged.connect(self._default_name)
        self.store.changed.connect(self.fill)
        self.fill()

    def fill(self, select=None):
        cur = select or self.cb_lib.currentData()
        self.cb_lib.blockSignals(True)
        self.cb_lib.clear()
        for l in self.store.libraries:
            self.cb_lib.addItem(f"{l['name']} ({l['spectra']} spectra)", l["id"])
        i = self.cb_lib.findData(cur) if cur else 0
        self.cb_lib.setCurrentIndex(max(0, i))
        self.cb_lib.blockSignals(False)
        self._default_name()

        def imgs(lst):
            c = self.cb_img.currentData()
            self.cb_img.clear()
            for im in lst:
                self.cb_img.addItem(f"{im['name']} ({im.get('bands')} bands)", im["id"])
            j = self.cb_img.findData(c)
            if j >= 0:
                self.cb_img.setCurrentIndex(j)
        run_bg(self.api.get, "/api/images/", done=imgs)

    def _default_name(self, *_):
        l = self.store.lib(self.cb_lib.currentData())
        if l and (not self.ed_name.text() or self.ed_name.text().endswith("(resampled)")):
            self.ed_name.setText(f"{l['name']} (resampled)")

    def run(self):
        lid = self.cb_lib.currentData()
        if not lid:
            return note_err(self.msg, "Choose a library.")
        mode = next(k for k, r in self.rb.items() if r.isChecked())
        body = dict(name=self.ed_name.text().strip(), mode=mode, min_coverage=self.sp_cov.value(),
                    mask_water=self.ck_water.isChecked())
        if not body["name"]:
            return note_err(self.msg, "Give the new library a name.")
        if mode == "range":
            body.update(start=self.sp_a.value(), end=self.sp_b.value(), step=self.sp_s.value())
        elif mode == "image":
            if not self.cb_img.currentData():
                return note_err(self.msg, "No image: open one on the Image page first.")
            body["image_id"] = self.cb_img.currentData()
        else:
            try:
                body["wavelengths"] = [float(x) for x in self.ed_list.text().replace(",", " ").split()]
            except ValueError:
                return note_err(self.msg, "The list must contain numbers only.")
        self.msg.setStyleSheet("")
        self.msg.setText("Resampling…")

        def done(r):
            self.msg.setStyleSheet(f"color:{C['emerald']};")
            rej = r.get("n_rejected") or 0
            self.msg.setText(f"“{r['library']}”: {r['resampled']} spectra resampled to {r['bands']} bands "
                             f"({r['range_nm'][0]:.1f}–{r['range_nm'][1]:.1f} nm, grid from {r['grid_source']})"
                             + (f" · {rej} rejected (e.g. {r['rejected'][0]['name']}: {r['rejected'][0]['reason']})" if rej else ""))
            self.store.refresh()
        run_bg(self.api.post, f"/api/libraries/{lid}/resample", body, done=done, failed=lambda m: note_err(self.msg, m))


class FunctionsView(W.QWidget):
    """Library functions: Spectral Library Builder, Library Viewer, Resampling (tabs)."""

    def __init__(self, page):
        super().__init__()
        lay = W.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.tabs = W.QTabWidget()
        self.builder = BuilderView(page)
        self.viewer = ViewerView(page)
        self.resample = ResampleView(page)
        self.tabs.addTab(self.builder, "Spectral Library Builder")
        self.tabs.addTab(self.viewer, "Library Viewer")
        self.tabs.addTab(self.resample, "Resampling")
        lay.addWidget(self.tabs)

    def show_tab(self, name):
        self.tabs.setCurrentWidget({"builder": self.builder, "viewer": self.viewer, "resample": self.resample}[name])
