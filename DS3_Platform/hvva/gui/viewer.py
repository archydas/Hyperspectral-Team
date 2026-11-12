"""Scene-view controls (under the image): band selection (gray / RGB, presets, stretch), zoom / pan / save PNG
toolbar, pixel window (1x1 ... 9x9), collecting spectra from several clicks, and the map background.

    ViewerBar               the control strip; talks to the main window (host) through show_scene() & co.
    CollectedSpectraDialog  the spectra collected by clicking: plot, pairwise spectral angles, export CSV,
                            match against the library, save to the library
No database code here: matching / saving goes through the Library tab (host.libpanel).
"""
from __future__ import annotations

import csv

import matplotlib
import numpy as np
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
from PyQt5 import QtCore, QtGui, QtWidgets

from ..core import display as D

W = QtWidgets
Qt = QtCore.Qt
PALETTE = [tuple(matplotlib.colormaps["tab10"](i)[:3]) for i in range(10)]
WINDOWS = (1, 3, 5, 9)


class NavBar(NavigationToolbar2QT):
    """Matplotlib toolbar limited to Home / Back / Forward / Pan / Zoom / Save."""
    toolitems = [t for t in NavigationToolbar2QT.toolitems if t[0] in ("Home", "Back", "Forward", "Pan", "Zoom", "Save")]

    def __init__(self, canvas, parent):
        super().__init__(canvas, parent, coordinates=False)       # the read-out goes to the status bar
        self.setIconSize(QtCore.QSize(18, 18))
        self.setMinimumWidth(len(self.toolitems) * 30)

    def save_figure(self, *args):
        """Save the current view (zoom kept) as PNG / JPG / PDF / SVG."""
        p, _ = W.QFileDialog.getSaveFileName(self.parent(), "Save view", "scene_view.png",
                                             "PNG image (*.png);;JPEG image (*.jpg);;PDF (*.pdf);;SVG (*.svg)")
        if p:
            try:
                self.canvas.fig.savefig(p, dpi=200, bbox_inches="tight", facecolor=self.canvas.fig.get_facecolor())
            except Exception as e:                                               # noqa: BLE001
                W.QMessageBox.warning(self.parent(), "Save view", str(e))


def _combo(items=(), tip=None, min_chars=None, expand=False):
    c = W.QComboBox()
    c.addItems(list(items))
    if tip:
        c.setToolTip(tip)
    if min_chars:
        c.setSizeAdjustPolicy(W.QComboBox.AdjustToMinimumContentsLengthWithIcon)
        c.setMinimumContentsLength(min_chars)
    if expand:
        c.setSizePolicy(W.QSizePolicy.Expanding, W.QSizePolicy.Fixed)
    return c


def _lab(text):
    l = W.QLabel(text)
    l.setStyleSheet("QLabel { color: #8b9bb4; }")
    return l


class ViewerBar(W.QWidget):
    """Controls under the scene view.  host = main window."""

    def __init__(self, host, canvas):
        super().__init__()
        self.host = host
        self.canvas = canvas
        self._busy = False
        self.toolbar = NavBar(canvas, self)
        # -- band selection
        self.cb_mode = _combo(["RGB", "Gray"], "RGB composite of three bands or one band in gray")
        self.cb_r = _combo(tip="Red band (or the band shown in gray)", min_chars=9, expand=True)
        self.cb_g = _combo(tip="Green band", min_chars=9, expand=True)
        self.cb_b = _combo(tip="Blue band", min_chars=9, expand=True)
        self.cb_preset = _combo(["Preset…"] + list(D.PRESETS), "Pick the bands nearest to standard wavelengths",
                                min_chars=8)
        self.sp_lo = W.QDoubleSpinBox(); self.sp_lo.setRange(0, 49); self.sp_lo.setValue(2); self.sp_lo.setSuffix(" %")
        self.sp_hi = W.QDoubleSpinBox(); self.sp_hi.setRange(51, 100); self.sp_hi.setValue(98); self.sp_hi.setSuffix(" %")
        for s in (self.sp_lo, self.sp_hi):
            s.setDecimals(1); s.setSingleStep(0.5); s.setKeyboardTracking(False)
            s.setToolTip("Percentile stretch (low / high) of every displayed band")
        self.cb_mode.currentIndexChanged.connect(self._on_mode)
        for c in (self.cb_r, self.cb_g, self.cb_b):
            c.activated.connect(self._apply)
        self.cb_preset.activated.connect(self._on_preset)
        self.sp_lo.valueChanged.connect(self._apply)
        self.sp_hi.valueChanged.connect(self._apply)
        # -- pixel tools
        self.cb_win = _combo([f"{n}×{n}" for n in WINDOWS], "Pixel window: the clicked spectrum is the mean of "
                                                             "this many pixels around the click", min_chars=4)
        self.chk_collect = W.QCheckBox("Collect spectra")
        self.chk_collect.setToolTip("Every click adds the spectrum to a list (Collected…) to compare angles, "
                                    "export, match or save them")
        self.btn_collected = W.QPushButton("Collected (0)…")
        self.btn_collected.clicked.connect(self.host.on_show_collected)
        # -- map
        self.chk_map = W.QCheckBox("Map")
        self.chk_map.setToolTip("Show a georeferenced scene on a web map (needs internet; tiles are cached)")
        from ..core.basemap import PROVIDERS
        self.cb_provider = _combo(list(PROVIDERS) + ["None (no background)"], "Map background", min_chars=12,
                                  expand=True)
        self.sl_alpha = W.QSlider(Qt.Horizontal); self.sl_alpha.setRange(10, 100); self.sl_alpha.setValue(85)
        self.sl_alpha.setMinimumWidth(70); self.sl_alpha.setMaximumWidth(110)
        self.sl_alpha.setToolTip("Image opacity on the map")
        self.chk_map.toggled.connect(lambda *_: self.host.show_scene())
        self.cb_provider.activated.connect(lambda *_: self.host.refresh_basemap(force=True))
        self.sl_alpha.sliderReleased.connect(lambda: self.host.show_scene())
        self.btn_reset = W.QPushButton("Fit"); self.btn_reset.setToolTip("Show the whole scene")
        self.btn_reset.clicked.connect(self.canvas.reset_view)

        def row(*ws):
            r = W.QHBoxLayout(); r.setContentsMargins(0, 0, 0, 0); r.setSpacing(4)
            for w_ in ws:
                if w_ is None:
                    r.addStretch(1)
                elif isinstance(w_, int):
                    r.addSpacing(w_)
                else:
                    r.addWidget(w_)
            return r
        self._lab_g, self._lab_b = _lab("G"), _lab("B")
        lay = W.QVBoxLayout(self); lay.setContentsMargins(0, 2, 0, 0); lay.setSpacing(3)
        lay.addLayout(row(self.cb_mode, _lab("R"), self.cb_r, self._lab_g, self.cb_g, self._lab_b, self.cb_b,
                          self.cb_preset))
        lay.addLayout(row(self.toolbar, self.btn_reset, 10, _lab("Stretch"), self.sp_lo, self.sp_hi, 10,
                          _lab("Window"), self.cb_win, None))
        lay.addLayout(row(self.chk_collect, self.btn_collected, 10, self.chk_map, self.cb_provider,
                          _lab("Opacity"), self.sl_alpha))
        self.set_scene(None)

    # ------------------------------------------------------------ state
    @property
    def window_size(self) -> int:
        return WINDOWS[self.cb_win.currentIndex()]

    @property
    def collecting(self) -> bool:
        return self.chk_collect.isChecked()

    @property
    def map_on(self) -> bool:
        return self.chk_map.isChecked()

    @property
    def provider(self):
        t = self.cb_provider.currentText()
        return None if t.startswith("None") else t

    @property
    def alpha(self) -> float:
        return self.sl_alpha.value() / 100.0

    def set_collected_count(self, n):
        self.btn_collected.setText(f"Collected ({n})…")

    def set_scene(self, sc):
        """Fill the band combos for *sc* (keeps the scene's own display choice)."""
        self._busy = True
        try:
            for c in (self.cb_r, self.cb_g, self.cb_b):
                c.clear()
            on = sc is not None
            for w in (self.cb_mode, self.cb_r, self.cb_g, self.cb_b, self.cb_preset, self.sp_lo, self.sp_hi):
                w.setEnabled(on)
            if not on:
                return
            labels = [D.band_label(sc, i) for i in range(sc.n_bands)]
            for c in (self.cb_r, self.cb_g, self.cb_b):
                c.addItems(labels)
            d = D.normalize_display(sc, sc.display)
            self.cb_mode.setCurrentIndex(0 if d["mode"] == "rgb" else 1)
            b = d["bands"] + d["bands"][:1] * 3
            for c, i in zip((self.cb_r, self.cb_g, self.cb_b), b[:3]):
                c.setCurrentIndex(i)
            self.sp_lo.setValue(d["stretch"][0]); self.sp_hi.setValue(d["stretch"][1])
            self.cb_preset.setCurrentIndex(0)
            self._sync_mode()
        finally:
            self._busy = False

    def _sync_mode(self):
        gray = self.cb_mode.currentIndex() == 1
        for w in (self.cb_g, self.cb_b, self._lab_g, self._lab_b):
            w.setVisible(not gray)

    def current_display(self):
        gray = self.cb_mode.currentIndex() == 1
        bands = [self.cb_r.currentIndex()] if gray else [self.cb_r.currentIndex(), self.cb_g.currentIndex(),
                                                          self.cb_b.currentIndex()]
        return {"mode": "gray" if gray else "rgb", "bands": bands, "stretch": [self.sp_lo.value(), self.sp_hi.value()]}

    # ------------------------------------------------------------ events
    def _on_mode(self, *_):
        self._sync_mode()
        self._apply()

    def _on_preset(self, i):
        sc = self.host.current_scene()
        if sc is None or i <= 0:
            return
        name = self.cb_preset.itemText(i)
        self._busy = True
        for c, b in zip((self.cb_r, self.cb_g, self.cb_b), D.preset_bands(sc, name)):
            c.setCurrentIndex(b)
        self.cb_mode.setCurrentIndex(0)
        self._busy = False
        self._sync_mode()
        self._apply()
        if not sc.has_wavelengths:
            self.host.say("This scene has no wavelengths: presets pick bands by position only.")

    def _apply(self, *_):
        if self._busy:
            return
        sc = self.host.current_scene()
        if sc is None:
            return
        sc.display = self.current_display()
        self.host.show_scene()
        self.host.say(f"{sc.name}: {D.describe(sc)} (stretch {sc.display['stretch'][0]:g}–{sc.display['stretch'][1]:g} %).")


class CollectedSpectraDialog(W.QDialog):
    """Spectra collected by clicking on the scene view (host.collected: list of dicts)."""

    def __init__(self, host):
        super().__init__(host)
        self.host = host
        self.setWindowTitle("Collected spectra")
        self.resize(760, 520)
        self.tbl = W.QTableWidget(0, 5)
        self.tbl.setHorizontalHeaderLabels(["", "Label", "Scene", "Pixel (row, col)", "Window"])
        self.tbl.horizontalHeader().setStretchLastSection(True)
        self.tbl.setSelectionBehavior(W.QAbstractItemView.SelectRows)
        self.tbl.setEditTriggers(W.QAbstractItemView.DoubleClicked | W.QAbstractItemView.EditKeyPressed)
        self.tbl.itemChanged.connect(self._renamed)
        self.tbl.itemSelectionChanged.connect(self.plot)
        self.ang = W.QTableWidget(0, 0)
        self.ang.setEditTriggers(W.QAbstractItemView.NoEditTriggers)
        self.ang.setToolTip("Spectral angle (degrees) between every pair, on the bands both spectra have. "
                            "0° = same shape; brightness differences are ignored.")
        self.lbl = W.QLabel("")
        self.lbl.setWordWrap(True)
        b = {}
        for key, text, fn in (("plot", "Plot", self.plot), ("remove", "Remove", self.on_remove),
                              ("clear", "Clear all", self.on_clear), ("csv", "Export CSV…", self.on_export),
                              ("match", "Best library matches", self.on_match),
                              ("save", "Save to library…", self.on_save)):
            b[key] = W.QPushButton(text)
            b[key].clicked.connect(fn)
        b["match"].setToolTip("Match the selected spectra (all if none selected) against the library, with the "
                              "Library tab's method and filters")
        top = W.QSplitter(Qt.Vertical)
        top.addWidget(self.tbl)
        g = W.QGroupBox("Spectral angle between collected spectra (degrees)")
        gl = W.QVBoxLayout(g); gl.addWidget(self.ang)
        top.addWidget(g)
        lay = W.QVBoxLayout(self)
        lay.addWidget(top, 1)
        lay.addWidget(self.lbl)
        row = W.QHBoxLayout()
        for k in ("plot", "remove", "clear", "csv", "match", "save"):
            row.addWidget(b[k])
        lay.addLayout(row)
        self.refresh()

    # ------------------------------------------------------------
    def items(self):
        return self.host.collected

    def selected(self):
        rows = sorted({i.row() for i in self.tbl.selectedIndexes()})
        return [self.items()[r] for r in rows if r < len(self.items())] or list(self.items())

    def refresh(self):
        items = self.items()
        self.tbl.blockSignals(True)
        self.tbl.setRowCount(len(items))
        for r, it in enumerate(items):
            pm = QtGui.QPixmap(12, 12); pm.fill(QtGui.QColor.fromRgbF(*it["color"]))
            c0 = W.QTableWidgetItem(); c0.setIcon(QtGui.QIcon(pm)); c0.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            self.tbl.setItem(r, 0, c0)
            lab = W.QTableWidgetItem(it["label"])
            self.tbl.setItem(r, 1, lab)
            for c, v in enumerate((it["scene"], f"{it['pixel'][0]}, {it['pixel'][1]}", f"{it['size']}×{it['size']}"), 2):
                x = W.QTableWidgetItem(str(v)); x.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                self.tbl.setItem(r, c, x)
        self.tbl.resizeColumnsToContents()
        self.tbl.blockSignals(False)
        self.fill_angles()
        units = {it.get("unit") for it in items}
        self.lbl.setText(f"{len(items)} spectra" + ("" if len(units) <= 1 else
                                                   f" ⚠ mixed units ({', '.join(sorted(map(str, units)))})"))

    def fill_angles(self):
        items = self.items()
        n = len(items)
        A = angle_matrix(items)
        self.ang.setRowCount(n); self.ang.setColumnCount(n)
        heads = [it["label"][:14] for it in items]
        self.ang.setHorizontalHeaderLabels(heads); self.ang.setVerticalHeaderLabels(heads)
        finite = A[np.isfinite(A) & ~np.eye(n, dtype=bool)] if n else np.array([])
        vmax = float(finite.max()) if finite.size else 1.0
        for i in range(n):
            for j in range(n):
                v = A[i, j]
                x = W.QTableWidgetItem("—" if not np.isfinite(v) else f"{v:.2f}")
                x.setTextAlignment(Qt.AlignCenter)
                if np.isfinite(v) and i != j:
                    t = min(1.0, v / (vmax or 1))
                    x.setBackground(QtGui.QColor.fromRgbF(0.25 + 0.6 * t, 0.55 - 0.25 * t, 0.85 - 0.6 * t, 0.45))
                self.ang.setItem(i, j, x)
        self.ang.resizeColumnsToContents()

    def _renamed(self, item):
        if item.column() == 1 and item.row() < len(self.items()):
            self.items()[item.row()]["label"] = item.text().strip() or self.items()[item.row()]["label"]
            self.fill_angles()
            self.plot()

    # ------------------------------------------------------------ actions
    def plot(self):
        items = self.selected()
        if not items:
            return self.host.plot_canvas.message("No collected spectra")
        ax = self.host.plot_canvas.reset()
        wl_ok = all(it["wavelengths"] is not None for it in items)
        for it in items:
            x = it["wavelengths"] if wl_ok else np.arange(len(it["values"]))
            ax.plot(x, it["values"], color=it["color"], lw=1.3, label=it["label"])
        units = {it.get("unit") for it in items} - {None, "unknown"}
        self.host.plot_canvas.finish(f"Collected spectra ({len(items)})", "Wavelength (nm)" if wl_ok else "Band",
                                     "Value" + (f" ({units.pop()})" if len(units) == 1 else ""), len(items) <= 14)

    def on_remove(self):
        rows = sorted({i.row() for i in self.tbl.selectedIndexes()}, reverse=True)
        for r in rows:
            del self.host.collected[r]
        self.host.collected_changed()

    def on_clear(self):
        self.host.collected.clear()
        self.host.collected_changed()

    def on_export(self):
        items = self.selected()
        if not items:
            return
        p, _ = W.QFileDialog.getSaveFileName(self, "Export collected spectra", "collected_spectra.csv", "CSV (*.csv)")
        if not p:
            return
        try:
            export_collected_csv(items, p)
        except Exception as e:                                                   # noqa: BLE001
            return self.host._err(e)
        self.host.say(f"Wrote {len(items)} spectra to {p}.")

    def on_match(self):
        items = [it for it in self.selected() if it["wavelengths"] is not None]
        if not items:
            return self.host._err("the collected spectra need wavelengths to be matched")
        queries = [(it["label"], it["wavelengths"], it["values"], it.get("unit")) for it in items]
        self.host.libpanel.match_queries(queries, title="Collected spectra vs library", show_dialog=True)

    def on_save(self):
        items = [it for it in self.selected() if it["wavelengths"] is not None]
        if not items:
            return self.host._err("the collected spectra need wavelengths to be saved")
        self.host.libpanel.save_collected(items)


def angle_matrix(items) -> np.ndarray:
    """Pairwise spectral angles (degrees) of collected spectra, compared on the first one's grid where both exist."""
    from ..core.resample import resample_spectra
    n = len(items)
    A = np.full((n, n), np.nan)
    for i in range(n):
        A[i, i] = 0.0
        for j in range(i + 1, n):
            a, b = items[i], items[j]
            x = np.asarray(a["values"], float)
            y = np.asarray(b["values"], float)
            if a["wavelengths"] is not None and b["wavelengths"] is not None:
                wa, wb = np.asarray(a["wavelengths"], float), np.asarray(b["wavelengths"], float)
                if len(wa) != len(wb) or not np.allclose(wa, wb):
                    y = resample_spectra(y[None].astype(np.float32), wb, wa)[0].astype(float)
            elif len(x) != len(y):
                continue
            ok = np.isfinite(x) & np.isfinite(y)
            if ok.sum() < 3:
                continue
            na, nb = np.linalg.norm(x[ok]), np.linalg.norm(y[ok])
            if na == 0 or nb == 0:
                continue
            A[i, j] = A[j, i] = float(np.degrees(np.arccos(np.clip(np.dot(x[ok], y[ok]) / (na * nb), -1, 1))))
    return A


def export_collected_csv(items, path):
    """Wide CSV: wavelength, <label>... (spectra on different grids are written on the first one's grid)."""
    from ..core.resample import resample_spectra
    base = items[0]
    x0 = np.asarray(base["wavelengths"], float) if base["wavelengths"] is not None else np.arange(len(base["values"]))
    cols = []
    for it in items:
        y = np.asarray(it["values"], float)
        if it["wavelengths"] is not None and base["wavelengths"] is not None:
            w = np.asarray(it["wavelengths"], float)
            if len(w) != len(x0) or not np.allclose(w, x0):
                y = resample_spectra(y[None].astype(np.float32), w, x0)[0].astype(float)
        elif len(y) != len(x0):
            raise ValueError("spectra without wavelengths must have the same number of bands")
        cols.append(y)
    with open(path, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["wavelength_nm" if base["wavelengths"] is not None else "band"] + [it["label"] for it in items])
        for k, x in enumerate(x0):
            wr.writerow([f"{x:g}"] + ["" if not np.isfinite(c[k]) else f"{c[k]:.6g}" for c in cols])
    return path
