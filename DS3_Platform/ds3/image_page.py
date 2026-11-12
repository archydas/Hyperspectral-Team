"""Viewer and Analysis module: the DS3 Hyperspectral Viewer & Analysis Tool (viewer/HyperVista_v1_0.py) embedded in the DS3 shell,
plus a 'Library' tab inside it that links the open image to the spectral library database - the DS3 features
the stand-alone viewer did not have:

    collect spectra by clicking (pixel window 1x1 ... 9x9), markers on the image, spectral-angle matrix, CSV,
    best library matches (SAM / RMSE, unit check), a classification map against a library (SAM, max angle),
    store a pixel spectrum, ROI class means or the collected spectra in a library, register the image,
    hand the image (with its ROIs) to the Data Aggregation module.

The viewer is imported only when Viewer and Analysis is first opened (scikit-learn / scikit-image take a while).
"""
from __future__ import annotations

import os
import time

import matplotlib
import numpy as np
from matplotlib.patches import Rectangle
from PyQt5 import QtCore, QtGui, QtWidgets

from . import style
from .widgets import label

W = QtWidgets
Qt = QtCore.Qt
PALETTE = [tuple(matplotlib.colormaps["tab10"](i)[:3]) for i in range(10)]
CLASS_COLORS = [tuple(matplotlib.colormaps["tab20"](i)[:3]) for i in range(20)]
WINDOWS = (1, 3, 5, 9)


def viewer_unit(v) -> str:
    """Unit of the viewer's active cube from its radiometric stage (or 'unknown')."""
    st = str(getattr(v, "radiometric_stage", "") or "").lower()
    if "reflect" in st:
        return "reflectance"
    if "radian" in st:
        return "radiance"
    if "dn" in st and "unknown" not in st:
        return "dn"
    return "unknown"


def window_spectrum(cube, r, c, size):
    h = size // 2
    blk = np.asarray(cube[max(0, r - h):r + h + 1, max(0, c - h):c + h + 1, :], dtype=float)
    blk = blk.reshape(-1, blk.shape[-1])
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanmean(blk, axis=0)


def sam_classify(cube, wl, refs, max_angle_deg=0.0, chunk_rows=64):
    """Per-pixel spectral angle against reference spectra [(name, wl, values)] on the cube's wavelengths.
    -> (index map, -1 = unclassified / no data; angle map in degrees; names; bands used)"""
    from hvva.core.resample import resample_spectra
    wl = np.asarray(wl, float)
    R = []
    names = []
    for name, rwl, vals in refs:
        y = resample_spectra(np.asarray(vals, np.float32)[None], np.asarray(rwl, float), wl)[0]
        R.append(y)
        names.append(name)
    if not R:
        raise ValueError("no reference spectra")
    R = np.vstack(R).astype(np.float64)
    band_ok = np.isfinite(R).all(axis=0)
    rows, cols, nb = cube.shape
    sample = np.asarray(cube[::max(1, rows // 50), ::max(1, cols // 50), :], float).reshape(-1, nb)
    band_ok &= np.isfinite(sample).any(axis=0)
    if band_ok.sum() < 3:
        raise ValueError("the references and the image share fewer than 3 valid bands - check the wavelength ranges")
    Rn = R[:, band_ok]
    Rn = Rn / np.linalg.norm(Rn, axis=1, keepdims=True)
    idx = np.full((rows, cols), -1, np.int16)
    ang = np.full((rows, cols), np.nan, np.float32)
    for r0 in range(0, rows, chunk_rows):
        X = np.asarray(cube[r0:r0 + chunk_rows, :, :], float)[..., band_ok]
        h = X.shape[0]
        X = X.reshape(-1, band_ok.sum())
        good = np.isfinite(X).all(axis=1)
        nrm = np.linalg.norm(X, axis=1)
        good &= nrm > 0
        cos = np.zeros((X.shape[0], Rn.shape[0]))
        cos[good] = (X[good] / nrm[good, None]) @ Rn.T
        a = np.degrees(np.arccos(np.clip(cos, -1, 1)))
        best = np.argmin(a, axis=1)
        amin = a[np.arange(a.shape[0]), best]
        best = best.astype(np.int16)
        best[~good] = -1
        if max_angle_deg and max_angle_deg > 0:
            best[amin > max_angle_deg] = -1
        idx[r0:r0 + h] = best.reshape(h, cols)
        am = amin.astype(np.float32)
        am[~good] = np.nan
        ang[r0:r0 + h] = am.reshape(h, cols)
    return idx, ang, names, int(band_ok.sum())


class BandsListDialog(W.QDialog):
    """ENVI-style 'Available Bands List': Gray Scale or RGB, click a band for each colour, Load."""

    def __init__(self, viewer, parent=None):
        super().__init__(parent)
        self.v = viewer
        self.setWindowTitle("Available Bands List")
        self.resize(420, 520)
        self.rb_gray = W.QRadioButton("Gray Scale")
        self.rb_rgb = W.QRadioButton("RGB Color")
        grp = W.QButtonGroup(self)
        grp.addButton(self.rb_gray)
        grp.addButton(self.rb_rgb)
        (self.rb_rgb if viewer.radio_rgb.isChecked() else self.rb_gray).setChecked(True)
        self.lst = W.QListWidget()
        wl = viewer.wavelengths
        nb = viewer.cube.shape[2]
        for i in range(nb):
            t = f"Band {i + 1}" + (f"  ({wl[i]:.1f} nm)" if wl is not None and len(wl) == nb else "")
            self.lst.addItem(t)
        self.fields = {}
        frm = W.QFormLayout()
        self.ed_gray = W.QLineEdit()
        self.ed_gray.setReadOnly(True)
        frm.addRow("Gray", self.ed_gray)
        for k in ("R", "G", "B"):
            e = W.QLineEdit()
            e.setReadOnly(True)
            self.fields[k] = e
            frm.addRow(k, e)
        self.sel = {"gray": int(viewer.band_slider.value()), "R": int(viewer.spin_r.value()),
                    "G": int(viewer.spin_g.value()), "B": int(viewer.spin_b.value())}
        self._next = 0
        self.lst.itemClicked.connect(self._pick)
        self.rb_gray.toggled.connect(self._sync)
        b_load = W.QPushButton("Load Band" if self.rb_gray.isChecked() else "Load RGB")
        self.b_load = b_load
        b_load.clicked.connect(self._load)
        b_close = W.QPushButton("Close")
        b_close.clicked.connect(self.close)
        lay = W.QVBoxLayout(self)
        r = W.QHBoxLayout()
        r.addWidget(self.rb_gray)
        r.addWidget(self.rb_rgb)
        r.addStretch(1)
        lay.addLayout(r)
        lay.addWidget(self.lst, 1)
        lay.addLayout(frm)
        lay.addWidget(label(f"Dims: {viewer.cube.shape[1]} x {viewer.cube.shape[0]} x {nb} "
                            "(samples x lines x bands)", "small"))
        r2 = W.QHBoxLayout()
        r2.addStretch(1)
        r2.addWidget(b_load)
        r2.addWidget(b_close)
        lay.addLayout(r2)
        self._sync()

    def _txt(self, i):
        return self.lst.item(i).text() if 0 <= i < self.lst.count() else ""

    def _sync(self, *_):
        gray = self.rb_gray.isChecked()
        self.ed_gray.setEnabled(gray)
        for e in self.fields.values():
            e.setEnabled(not gray)
        self.ed_gray.setText(self._txt(self.sel["gray"]))
        for k, e in self.fields.items():
            e.setText(self._txt(self.sel[k]))
        self.b_load.setText("Load Band" if gray else "Load RGB")

    def _pick(self, item):
        i = self.lst.row(item)
        if self.rb_gray.isChecked():
            self.sel["gray"] = i
        else:
            k = "RGB"[self._next % 3]
            self.sel[k] = i
            self._next += 1
        self._sync()

    def _load(self):
        v = self.v
        if self.rb_gray.isChecked():
            v.radio_single.setChecked(True)
            v.band_slider.setValue(self.sel["gray"])
        else:
            for sp, k in ((v.spin_r, "R"), (v.spin_g, "G"), (v.spin_b, "B")):
                sp.blockSignals(True)
                sp.setValue(self.sel[k])
                sp.blockSignals(False)
            v.radio_rgb.setChecked(True)
        v.on_mode_changed()
        v.update_image()


class LibraryLinkTab(W.QWidget):
    """The DS3 'Library' tab inside the viewer."""

    def __init__(self, page):
        super().__init__()
        self.page = page
        self.v = page.viewer
        self.collected = []                      # same dict format as hvva's collected spectra
        self.last = None                         # (r, c, size, values)
        self._markers = []
        self._class_map = None
        # ---- 1. pick & collect
        self.cb_win = W.QComboBox()
        self.cb_win.addItems([f"{n}×{n}" for n in WINDOWS])
        self.cb_win.setToolTip("A click gives the mean spectrum of this many pixels around it")
        self.btn_collect = W.QPushButton("● Collect off")
        self.btn_collect.setCheckable(True)
        self.btn_collect.toggled.connect(self._collect_toggled)
        self.tbl = W.QTableWidget(0, 3)
        self.tbl.setHorizontalHeaderLabels(["", "Spectrum", "Pixel (row, col)"])
        self.tbl.horizontalHeader().setStretchLastSection(True)
        self.tbl.verticalHeader().setVisible(False)
        self.tbl.setSelectionBehavior(W.QAbstractItemView.SelectRows)
        self.tbl.setEditTriggers(W.QAbstractItemView.DoubleClicked)
        self.tbl.itemChanged.connect(self._renamed)
        self.tbl.itemSelectionChanged.connect(self.plot_collected)
        self.tbl.setMinimumHeight(120)
        g1 = W.QGroupBox("Spectra — click pixels on the image")
        l1 = W.QVBoxLayout(g1)
        r = W.QHBoxLayout()
        r.addWidget(W.QLabel("Window"))
        r.addWidget(self.cb_win)
        r.addWidget(self.btn_collect)
        r.addStretch(1)
        l1.addLayout(r)
        l1.addWidget(label("Collect on: every click adds a spectrum (numbered on the image). Collect off: a "
                           "click shows the pixel and can be matched or stored.", "small", wrap=True))
        l1.addWidget(self.tbl, 1)
        r = W.QHBoxLayout()
        for t, fn in (("Compare angles", self.on_angles), ("Export CSV…", self.on_csv), ("Remove", self.on_remove),
                      ("Clear", self.on_clear)):
            b = W.QPushButton(t)
            b.clicked.connect(fn)
            r.addWidget(b)
        l1.addLayout(r)
        r = W.QHBoxLayout()
        b = W.QPushButton("Build library…")
        style.prop(b, ds3="primary", accent="library")
        b.setToolTip("Save the collected spectra (selected, or all) into a library: category, acquisition, unit")
        b.clicked.connect(self.on_build)
        r.addWidget(b)
        b = W.QPushButton("Classify (SAM) with these")
        b.setToolTip("Classification map using the collected spectra as references")
        b.clicked.connect(self.on_classify_collected)
        r.addWidget(b)
        l1.addLayout(r)
        # ---- 2. library match
        self.cb_lib = W.QComboBox()
        self.cb_lib.setMinimumContentsLength(18)
        self.cb_lib.setSizeAdjustPolicy(W.QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.cb_method = W.QComboBox()
        self.cb_method.addItem("SAM (shape)", "sam")
        self.cb_method.addItem("RMSE (shape + level)", "rmse")
        self.sp_top = W.QSpinBox()
        self.sp_top.setRange(1, 50)
        self.sp_top.setValue(5)
        self.chk_units = W.QCheckBox("ignore unit differences (compare spectral shape only)")
        self.chk_units.setChecked(True)
        self.cb_unit = W.QComboBox()
        for t, d in (("Image unit: auto", None), ("reflectance", "reflectance"), ("radiance", "radiance"),
                     ("DN / counts", "dn"), ("unknown", "unknown")):
            self.cb_unit.addItem(t, d)
        self.cb_unit.setToolTip("Unit of the image values (auto = from the viewer's data stage)")
        self.res = W.QTableWidget(0, 4)
        self.res.setHorizontalHeaderLabels(["#", "Material", "Library", "Score"])
        self.res.horizontalHeader().setStretchLastSection(True)
        self.res.verticalHeader().setVisible(False)
        self.res.setEditTriggers(W.QAbstractItemView.NoEditTriggers)
        self.res.setMinimumHeight(120)
        self.lbl_match = label("Connect the library (Library module) and click a pixel.", "small", wrap=True)
        g2 = W.QGroupBox("Library — compare the image with spectra in the database")
        l2 = W.QVBoxLayout(g2)
        f = W.QFormLayout()
        f.addRow("Library", self.cb_lib)
        r = W.QHBoxLayout()
        r.addWidget(self.cb_method)
        r.addWidget(W.QLabel("top"))
        r.addWidget(self.sp_top)
        r.addWidget(self.cb_unit)
        f.addRow("Method", r)
        l2.addLayout(f)
        l2.addWidget(self.chk_units)
        r = W.QHBoxLayout()
        b = W.QPushButton("Best matches")
        style.prop(b, ds3="primary", accent="library")
        b.setToolTip("Best library matches for the last clicked pixel, or the selected collected spectra")
        b.clicked.connect(self.on_best_matches)
        r.addWidget(b)
        b = W.QPushButton("Classification map")
        style.prop(b, ds3="primary", accent="library")
        b.setToolTip("Every pixel gets the library material with the smallest spectral angle")
        b.clicked.connect(self.on_classify_library)
        r.addWidget(b)
        r.addStretch(1)
        l2.addLayout(r)
        r = W.QHBoxLayout()
        r.addWidget(W.QLabel("max angle"))
        self.sp_angle = W.QDoubleSpinBox()
        self.sp_angle.setRange(0, 90)
        self.sp_angle.setDecimals(1)
        self.sp_angle.setSuffix("°")
        self.sp_angle.setToolTip("Pixels further than this from every reference stay unclassified (0 = no limit)")
        r.addWidget(self.sp_angle)
        r.addWidget(label("(0 = none)", "small"))
        r.addWidget(W.QLabel("max materials"))
        self.sp_nmat = W.QSpinBox()
        self.sp_nmat.setRange(2, 2000)
        self.sp_nmat.setValue(150)
        r.addWidget(self.sp_nmat)
        r.addStretch(1)
        l2.addLayout(r)
        l2.addWidget(self.res, 1)
        l2.addWidget(self.lbl_match)
        # ---- 3. store & hand over
        g3 = W.QGroupBox("Store and hand over")
        l3 = W.QVBoxLayout(g3)
        for t, fn, tip in (
                ("Store pixel spectrum to library…", self.on_store_pixel, "The last clicked pixel (with its window)"),
                ("Class spectra from map…", self.on_class_map_means, "Ground-truth map such as PaviaU_gt.mat "
                                                                     "(.mat .npy .tif .h5 ...): one mean spectrum per "
                                                                     "class, saved to a library"),
                ("Register image in library", self.on_register, "Record the image (path, size, wavelengths, "
                                                                "unit) in the images table"),
                ("Back to the image display", self.on_back, "Leave the classification map"),
                ("Send image to Data Aggregation", self.on_send, "Copy the active cube (after preprocessing) to Data "
                                                      "Aggregation, where ROIs are drawn and aggregated"),
                ("Open this file in Data Aggregation (map view)", self.on_open_analysis,
                 "Load the same file in Data Aggregation, which reads its georeferencing and shows it on a map")):
            b = W.QPushButton(t)
            b.setToolTip(tip)
            b.clicked.connect(fn)
            l3.addWidget(b)
        self.lbl_store = label("", "small", wrap=True)
        l3.addWidget(self.lbl_store)
        l3.addStretch(1)
        lay = W.QHBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.addWidget(g1, 4)
        lay.addWidget(g2, 5)
        lay.addWidget(g3, 3)
        # hooks into the viewer
        self.v.image_canvas.mpl_connect("button_press_event", self.on_click)
        prev = self.v.image_canvas.after_clear
        self.v.image_canvas.after_clear = lambda: (prev() if prev else None, self.draw_markers(draw=False))
        self.refresh_libraries()

    # ------------------------------------------------------------ helpers
    @property
    def libpanel(self):
        return self.page.shell.libpanel

    def say(self, text):
        self.v.info_label.setText(text)
        self.page.shell.flash(text)

    def unit(self):
        u = self.cb_unit.currentData()
        return u if u else viewer_unit(self.v)

    def need_image(self):
        if self.v.cube is None:
            W.QMessageBox.information(self, "No image", "Open an image first (File ▸ Open Image File…).")
            return False
        return True

    def need_db(self):
        lp = self.libpanel
        if lp is None or lp.client is None:
            W.QMessageBox.information(self, "Library not connected",
                                      "Connect the spectral library first: Library module ▸ Connect… "
                                      "(File ▸ Connect Library Database).")
            return False
        return True

    def refresh_libraries(self):
        lp = self.libpanel
        cur = self.cb_lib.currentData()
        self.cb_lib.blockSignals(True)
        self.cb_lib.clear()
        self.cb_lib.addItem("All libraries", "__all__")
        for l in (lp.libraries if lp else []):
            self.cb_lib.addItem(f"{l['name']}" + (f"  [{l['unit']}]" if l.get("unit") else ""), l["id"])
        i = self.cb_lib.findData(cur)
        self.cb_lib.setCurrentIndex(max(0, i))
        self.cb_lib.blockSignals(False)

    def _libs(self):
        lp = self.libpanel
        d = self.cb_lib.currentData()
        return list(lp.libraries) if d in (None, "__all__") else [l for l in lp.libraries if l["id"] == d]

    # ------------------------------------------------------------ clicks / collect
    def _collect_toggled(self, on):
        self.btn_collect.setText("● Collect on" if on else "● Collect off")
        style.prop(self.btn_collect, ds3="primary" if on else "", accent="library")
        if on:
            self.say("Collect on: every click on the image adds a spectrum.")

    def on_click(self, ev):
        v = self.v
        if v.cube is None or v.roi_active or ev.inaxes is not v.image_canvas.ax or ev.xdata is None:
            return
        if getattr(ev, "button", 1) not in (1, None):
            return
        r, c = int(round(ev.ydata)), int(round(ev.xdata))
        rows, cols = v.cube.shape[:2]
        if not (0 <= r < rows and 0 <= c < cols):
            return
        size = WINDOWS[self.cb_win.currentIndex()]
        y = window_spectrum(v.cube, r, c, size)
        self.last = (r, c, size, y)
        if not self.btn_collect.isChecked():
            if size > 1:                                       # the viewer plotted the single pixel; show the mean
                x = np.asarray(v.wavelengths, float) if v.wavelengths is not None else np.arange(len(y))
                ok = np.isfinite(y)
                v.plot_canvas.show_spectrum(x[ok], y[ok], title=f"Mean of {size}×{size} at (row={r}, col={c})",
                                            xlabel="Wavelength (nm)" if v.wavelengths is not None else "Band")
            return
        col = PALETTE[len(self.collected) % 10]
        wl = None if v.wavelengths is None else np.asarray(v.wavelengths, float)
        name = self.page.image_name()
        self.collected.append({"label": f"{name} ({r}, {c})" + (f" {size}×{size}" if size > 1 else ""),
                               "scene": name, "scene_id": id(v.cube), "pixel": (r, c), "size": size,
                               "wavelengths": wl, "values": np.asarray(y, float), "unit": self.unit(),
                               "purity": "image_pixel", "color": col})
        self.refresh_table()
        self.draw_markers()
        self.plot_collected()
        if len(self.collected) >= 2:
            from hvva.gui.viewer import angle_matrix
            a = angle_matrix(self.collected[-2:])[0, 1]
            self.say(f"Collected #{len(self.collected)} at ({r}, {c})" +
                     (f" - angle to the previous one {a:.2f}°" if np.isfinite(a) else ""))

    def refresh_table(self):
        self.tbl.blockSignals(True)
        self.tbl.setRowCount(len(self.collected))
        for i, it in enumerate(self.collected):
            pm = QtGui.QPixmap(12, 12)
            pm.fill(QtGui.QColor.fromRgbF(*it["color"]))
            c0 = W.QTableWidgetItem(str(i + 1))
            c0.setIcon(QtGui.QIcon(pm))
            c0.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            self.tbl.setItem(i, 0, c0)
            self.tbl.setItem(i, 1, W.QTableWidgetItem(it["label"]))
            c2 = W.QTableWidgetItem(f"{it['pixel'][0]}, {it['pixel'][1]}")
            c2.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            self.tbl.setItem(i, 2, c2)
        self.tbl.resizeColumnsToContents()
        self.tbl.blockSignals(False)

    def _renamed(self, item):
        if item.column() == 1 and item.row() < len(self.collected):
            self.collected[item.row()]["label"] = item.text().strip() or self.collected[item.row()]["label"]

    def selected(self):
        rows = sorted({i.row() for i in self.tbl.selectedIndexes()})
        return [self.collected[r] for r in rows if r < len(self.collected)] or list(self.collected)

    def draw_markers(self, draw=True):
        ax = self.v.image_canvas.ax
        for a in self._markers:
            try:
                a.remove()
            except (ValueError, NotImplementedError, AttributeError):
                pass
        self._markers = []
        if self.v.cube is None or self._class_map is not None:
            return
        for k, it in enumerate(self.collected):
            if it["scene_id"] != id(self.v.cube):
                continue
            r, c = it["pixel"]
            h = it["size"] / 2
            p = Rectangle((c - h, r - h), it["size"], it["size"], fill=False, ec=it["color"], lw=1.6, zorder=6)
            ax.add_patch(p)
            t = ax.text(c + h + 0.6, r - h, str(k + 1), color=it["color"], fontsize=7, fontweight="bold", zorder=6)
            self._markers += [p, t]
        if draw:
            self.v.image_canvas.draw_idle()

    def plot_collected(self):
        items = self.selected()
        if not items:
            return
        pc = self.v.plot_canvas
        ax = pc.ax
        ax.clear()
        for it in items[-12:]:
            x = it["wavelengths"] if it["wavelengths"] is not None else np.arange(len(it["values"]))
            y = np.asarray(it["values"], float)
            ax.plot(x, y, color=it["color"], lw=1.3, label=it["label"])
        ax.set_title(f"Collected spectra ({len(items)})" + (" - last 12" if len(items) > 12 else ""))
        ax.set_xlabel("Wavelength (nm)" if items[0]["wavelengths"] is not None else "Band")
        ax.set_ylabel("Value")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
        self._redraw_plot()

    def _redraw_plot(self):
        pc = self.v.plot_canvas
        pc._safe_layout()
        self.page.HV.restyle_figure(pc.fig)
        pc.draw()

    def on_angles(self):
        if len(self.collected) < 2:
            return self.say("Collect at least two spectra first.")
        from hvva.gui.viewer import angle_matrix
        items = self.selected() if len(self.selected()) >= 2 else self.collected
        A = angle_matrix(items)
        d = W.QDialog(self)
        d.setWindowTitle("Spectral angle between collected spectra (degrees)")
        t = W.QTableWidget(len(items), len(items))
        heads = [str(self.collected.index(it) + 1) for it in items]
        t.setHorizontalHeaderLabels(heads)
        t.setVerticalHeaderLabels([f"{h}  {it['label'][:28]}" for h, it in zip(heads, items)])
        finite = A[np.isfinite(A) & ~np.eye(len(items), dtype=bool)]
        vmax = float(finite.max()) if finite.size else 1.0
        for i in range(len(items)):
            for j in range(len(items)):
                x = W.QTableWidgetItem("—" if not np.isfinite(A[i, j]) else f"{A[i, j]:.2f}")
                x.setTextAlignment(Qt.AlignCenter)
                if np.isfinite(A[i, j]) and i != j:
                    tt = min(1.0, A[i, j] / (vmax or 1))
                    x.setBackground(QtGui.QColor.fromRgbF(0.25 + 0.6 * tt, 0.55 - 0.25 * tt, 0.85 - 0.6 * tt, 0.4))
                t.setItem(i, j, x)
        t.resizeColumnsToContents()
        lay = W.QVBoxLayout(d)
        lay.addWidget(label("0° = same shape (brightness ignored). Below about 3-5° is usually the same "
                            "material.", "small", wrap=True))
        lay.addWidget(t)
        d.resize(min(900, 160 + 70 * len(items)), min(700, 140 + 30 * len(items)))
        d.exec_()

    def on_csv(self):
        items = self.selected()
        if not items:
            return self.say("No collected spectra.")
        p, _ = W.QFileDialog.getSaveFileName(self, "Export collected spectra", "collected_spectra.csv", "CSV (*.csv)")
        if p:
            from hvva.gui.viewer import export_collected_csv
            export_collected_csv(items, p)
            self.say(f"Wrote {len(items)} spectra to {p}.")

    def on_remove(self):
        for r in sorted({i.row() for i in self.tbl.selectedIndexes()}, reverse=True):
            del self.collected[r]
        self.refresh_table()
        self.draw_markers()

    def on_clear(self):
        self.collected.clear()
        self.refresh_table()
        self.draw_markers()

    def on_build(self):
        items = [it for it in self.selected() if it["wavelengths"] is not None]
        if not items:
            return self.say("Collect spectra of an image with wavelengths first.")
        if self.need_db():
            self.libpanel.save_collected(items)

    # ------------------------------------------------------------ matching
    def _entries(self, wl_query):
        """Library spectra for the chosen library(s), with the Library module's filters; unit checks."""
        lp = self.libpanel
        method = self.cb_method.currentData()
        qunit = self.unit()
        qunit = None if qunit in (None, "unknown") else qunit
        f = lp.filters()
        notes, entries = [], []
        for l in self._libs():
            lu = l.get("unit")
            if qunit and lu and lu != qunit and not self.chk_units.isChecked():
                if method == "rmse":
                    notes.append(f"{l['name']} skipped (library {lu}, image {qunit})")
                    continue
                notes.append(f"⚠ {l['name']} is {lu}, the image is {qunit}")
            for e in lp.client.library_spectra(l["id"], lp.acq_filter, None, f["unit"], f["purity"], f["covers"]):
                entries.append(dict(e, library=l["name"]))
        return entries, notes

    def on_best_matches(self):
        if not self.need_image() or not self.need_db():
            return
        if self.v.wavelengths is None:
            return self.say("The image has no wavelengths (Band Management ▸ Assign wavelengths).")
        sel = [it for it in self.selected() if it["wavelengths"] is not None] if self.tbl.selectedIndexes() else []
        if sel:
            q = (sel[0]["label"], sel[0]["wavelengths"], sel[0]["values"])
        elif self.last is not None:
            r, c, size, y = self.last
            q = (f"pixel ({r}, {c})" + (f" {size}×{size}" if size > 1 else ""), np.asarray(self.v.wavelengths, float), y)
        else:
            return self.say("Click a pixel first (or select a collected spectrum).")
        from hvva.library import repository as repo
        QtWidgets.QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            entries, notes = self._entries(q[1])
            method = self.cb_method.currentData()
            ranked = repo.match_rank(q[2], q[1], entries, self.sp_top.value(), method)
        except Exception as e:                                                   # noqa: BLE001
            QtWidgets.QApplication.restoreOverrideCursor()
            return self.say(f"⚠ Matching failed: {e}")
        QtWidgets.QApplication.restoreOverrideCursor()
        self.res.setRowCount(len(ranked))
        for i, (score, m, n, e) in enumerate(ranked):
            sc = f"{np.degrees(score):.2f}°" if method == "sam" else f"{score:.4g}"
            for j, v in enumerate((str(i + 1), m["name"], e.get("library", ""), f"{sc}  ({n} bands)")):
                self.res.setItem(i, j, W.QTableWidgetItem(v))
        self.res.resizeColumnsToContents()
        self.lbl_match.setText(f"{q[0]}: {len(entries)} reference spectra searched." +
                               ("  " + "; ".join(notes) if notes else ""))
        # plot query + top 5
        from hvva.library.repository import spectrum_values, spectrum_wavelengths
        ax = self.v.plot_canvas.ax
        ax.clear()
        wl, y = np.asarray(q[1], float), np.asarray(q[2], float)
        scale = method == "sam"
        ax.plot(wl, y / ((np.nanmax(np.abs(y)) or 1) if scale else 1), color="black", lw=2, label=q[0])
        for k, (score, m, n, e) in enumerate(ranked[:5]):
            x = np.asarray(spectrum_wavelengths(e["spectrum"]), float)
            yy = np.asarray(spectrum_values(e["spectrum"]), float)
            sel = (x >= np.nanmin(wl)) & (x <= np.nanmax(wl))
            if sel.any():
                x, yy = x[sel], yy[sel]
            if scale:
                yy = yy / (np.nanmax(np.abs(yy)) or 1)
            s = f"{np.degrees(score):.1f}°" if method == "sam" else f"RMSE {score:.3g}"
            ax.plot(x, yy, ls="--", lw=1.1, color=PALETTE[k], label=f"{m['name'][:30]} ({s})")
        ax.set_title("Best library matches" + (" (each scaled to max 1)" if scale else ""))
        ax.set_xlabel("Wavelength (nm)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
        self._redraw_plot()
        if ranked:
            self.say(f"Best match: {ranked[0][1]['name']} ({ranked[0][3].get('library', '')}).")

    # ------------------------------------------------------------ classification maps
    def _show_class_map(self, idx, names, title):
        colors = [CLASS_COLORS[i % 20] for i in range(len(names))]
        rgb = np.zeros(idx.shape + (3,), np.float32)
        for k, col in enumerate(colors):
            rgb[idx == k] = col
        rgb[idx < 0] = (0.12, 0.12, 0.12)
        n = idx.size
        counts = [(int((idx == k).sum()), k) for k in range(len(names))]
        counts.sort(reverse=True)
        legend = [(f"{names[k][:28]} ({100 * c / n:.1f}%)", colors[k]) for c, k in counts[:12] if c]
        self._class_map = idx
        self._class_names = list(names)
        for a in self._markers:
            try:
                a.remove()
            except (ValueError, NotImplementedError, AttributeError):
                pass
        self._markers = []
        self.v.image_canvas.show_image(rgb, legend=legend)
        self.v.image_canvas.ax.set_title(title, fontsize=9)
        self.v.image_canvas.draw_idle()
        un = int((idx < 0).sum())
        self.lbl_match.setText(f"{title}: {len(names)} classes; unclassified {100 * un / n:.1f}% of pixels. "
                               "'Back to the image display' leaves the map.")

    def _classify(self, refs, title):
        v = self.v
        t0 = time.time()
        QtWidgets.QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            idx, ang, names, nb = sam_classify(v.cube, v.wavelengths, refs, self.sp_angle.value())
        except Exception as e:                                                   # noqa: BLE001
            QtWidgets.QApplication.restoreOverrideCursor()
            return self.say(f"⚠ Classification failed: {e}")
        QtWidgets.QApplication.restoreOverrideCursor()
        self._show_class_map(idx, names, title)
        self.say(f"{title}: {len(names)} references on {nb} bands in {time.time() - t0:.1f} s.")

    def on_classify_library(self):
        if not self.need_image() or not self.need_db():
            return
        if self.v.wavelengths is None:
            return self.say("The image has no wavelengths.")
        from hvva.library.repository import spectrum_values, spectrum_wavelengths
        QtWidgets.QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            entries, notes = self._entries(self.v.wavelengths)
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        lo, hi = float(np.nanmin(self.v.wavelengths)), float(np.nanmax(self.v.wavelengths))
        refs = []
        for e in entries:
            x = np.asarray(spectrum_wavelengths(e["spectrum"]), float)
            if x.size < 3 or np.nanmax(x) < lo or np.nanmin(x) > hi:
                continue
            refs.append((e["material"]["name"], x, spectrum_values(e["spectrum"])))
        if not refs:
            return self.say("No library spectra overlap the image's wavelengths.")
        if len(refs) > self.sp_nmat.value():
            self.say(f"Using the first {self.sp_nmat.value()} of {len(refs)} materials (raise 'max materials' or "
                     "filter the library).")
            refs = refs[:self.sp_nmat.value()]
        lib = self.cb_lib.currentText().split("  [")[0]
        self._classify(refs, f"SAM classification - {lib}")

    def on_classify_collected(self):
        if not self.need_image():
            return
        items = [it for it in self.selected() if it["wavelengths"] is not None]
        if len(items) < 2:
            return self.say("Collect at least two spectra (of an image with wavelengths).")
        self._classify([(it["label"], it["wavelengths"], it["values"]) for it in items],
                       "SAM classification - collected spectra")

    def on_back(self):
        self._class_map = None
        if self.v.cube is not None:
            self.v.update_image()

    # ------------------------------------------------------------ store / hand over
    def on_store_pixel(self):
        if not self.need_image() or not self.need_db():
            return
        if self.last is None:
            return self.say("Click a pixel first.")
        if self.v.wavelengths is None:
            return self.say("The image has no wavelengths.")
        r, c, size, y = self.last
        name = self.page.image_name()
        it = {"label": f"{name} pixel ({r}, {c})" + (f" {size}x{size}" if size > 1 else ""), "scene": name,
              "pixel": (r, c), "size": size, "wavelengths": np.asarray(self.v.wavelengths, float),
              "values": np.asarray(y, float), "unit": self.unit(), "purity": "image_pixel"}
        self.libpanel.save_collected([it])

    def roi_class_means(self):
        v = self.v
        classes = {}
        for roi in v.rois:
            classes.setdefault(roi["class"], []).append(roi)
        out = []
        for cls, rois in classes.items():
            mask = np.zeros(v.cube.shape[:2], bool)
            for roi in rois:
                mask |= np.asarray(roi["mask"], bool)
            X = np.asarray(v.cube[mask], float)
            if X.size == 0:
                continue
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                out.append((cls, np.nanmean(X, 0), np.nanstd(X, 0, ddof=1) if len(X) > 1 else None, len(X),
                            [r["name"] for r in rois]))
        return out

    def on_roi_means(self):
        if not self.need_image() or not self.need_db():
            return
        if not self.v.rois:
            return self.say("Draw ROIs first (ROI Analysis tab: Enable ROI selection, draw, Add selection).")
        if self.v.wavelengths is None:
            return self.say("The image has no wavelengths.")
        means = self.roi_class_means()
        lp = self.libpanel
        ch = lp._ask_target(f"{len(means)} ROI class mean(s) of {self.page.image_name()}: " +
                            ", ".join(m[0] for m in means[:8]), means[0][0] if len(means) == 1 else None)
        if not ch:
            return
        name = self.page.image_name()
        items = []
        for cls, mean, std, n, roi_names in means:
            items.append({"name": cls if not (len(means) == 1 and ch.get("name")) else ch["name"],
                          "wavelengths": [float(x) for x in self.v.wavelengths],
                          "values": [None if not np.isfinite(x) else float(x) for x in mean],
                          "std": None if std is None else [None if not np.isfinite(x) else float(x) for x in std],
                          "material_metadata": {"created_by": "DS3 Viewer", "scene": name, "roi_class": cls},
                          "spectrum_metadata": {"n_pixels": int(n), "source_scenes": name,
                                                "rois": ", ".join(roi_names)},
                          "unit": self.unit(), "purity": "image_pixel"})
        lp._save(ch, items, "ROI class means")

    @staticmethod
    def read_label_map(path, shape):
        """2-D integer label map of the image size from any supported file (0 / NaN = no class)."""
        from hvva.core.loaders import open_source
        src = open_source(path)
        cands = [k for k, (s_, _) in src.arrays.items() if len(s_) == 2 and sorted(s_) == sorted(shape)]
        if not cands and len(src.arrays) == 1:
            cands = list(src.arrays)
        if not cands:
            raise ValueError(f"no 2-D array of size {shape[0]} x {shape[1]} in the file")
        lab = np.squeeze(np.asarray(src.reader(cands[0])))
        if lab.shape == tuple(shape)[::-1] and lab.shape != tuple(shape):
            lab = lab.T
        if lab.shape != tuple(shape):
            raise ValueError(f"the label map is {lab.shape}, the image is {tuple(shape)}")
        return np.nan_to_num(lab).astype(np.int64)

    def class_means_from_labels(self, lab):
        cube = self.v.cube
        out = []
        for k in [int(x) for x in np.unique(lab) if x > 0]:
            px = np.asarray(cube[lab == k], np.float64)
            out.append((f"class_{k}", np.nanmean(px, axis=0), np.nanstd(px, axis=0), px.shape[0]))
        return out

    def on_class_map_means(self, path=None):
        if not self.need_image() or not self.need_db():
            return
        if self.v.wavelengths is None:
            return self.say("The image has no wavelengths.")
        if not path:
            path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, "Ground-truth label map", "", "Label maps (*.mat *.npy *.npz *.tif *.tiff *.h5 *.hdf5 *.nc);;"
                                                   "All files (*)")
        if not path:
            return
        try:
            lab = self.read_label_map(path, self.v.cube.shape[:2])
        except Exception as e:                                                   # noqa: BLE001
            return self.say(f"⚠ Label map: {e}")
        means = self.class_means_from_labels(lab)
        if not means:
            return self.say("The label map has no classes (all values are 0).")
        lp = self.libpanel
        ch = lp._ask_target(f"{len(means)} class mean spectra of {self.page.image_name()} from "
                            f"{os.path.basename(path)}", None)
        if not ch:
            return
        name = self.page.image_name()
        items = [{"name": cls, "wavelengths": [float(x) for x in self.v.wavelengths],
                  "values": [None if not np.isfinite(x) else float(x) for x in mean],
                  "std": [None if not np.isfinite(x) else float(x) for x in std],
                  "material_metadata": {"created_by": "DS3 Viewer", "scene": name,
                                        "label_map": os.path.basename(path)},
                  "spectrum_metadata": {"n_pixels": int(n), "source_scenes": name},
                  "unit": self.unit(), "purity": "image_pixel"} for cls, mean, std, n in means]
        lp._save(ch, items, "Class spectra from map")

    def _image_info(self):
        from hvva.core.scene import Scene, image_info
        sc = self.page.as_scene(with_rois=False)
        info = image_info(sc)
        return info

    def on_register(self):
        if not self.need_image() or not self.need_db():
            return
        try:
            info = self._image_info()
        except Exception as e:                                                   # noqa: BLE001
            return self.say(f"⚠ {e}")

        def ok(im):
            self.lbl_store.setText(f"Registered '{im['name']}' in the images table.")
            self.page.shell.refresh_counts()
        self.libpanel.run(self.libpanel.client.register_image, info, done=ok)

    def on_send(self):
        if not self.need_image():
            return
        sc = self.page.as_scene(with_rois=False)          # ROIs are drawn in Data Aggregation, not in the Viewer
        self.page.shell.analysis.win.add_scene(sc)
        self.page.shell.show_module("analysis")
        self.page.shell.analysis.win.tabs.setCurrentIndex(0)
        self.page.shell.flash(f"'{sc.name}' sent to Data Aggregation: draw its ROIs in the ROI Analysis tab "
                              "(Tools ▸ Get ROIs from Data Aggregation brings them back to Viewer and Analysis).")

    def select_or_send(self):
        """Select this image's scene in Data Aggregation (same name and size), sending it first if it is not there."""
        if not self.need_image():
            return
        win = self.page.shell.analysis.win
        name, shape = self.page.image_name(), tuple(self.v.cube.shape[:2])
        for i, sc in enumerate(win.scenes):
            if sc.name == name and tuple(sc.cube.shape[:2]) == shape:
                win.tbl_scenes.selectRow(i)
                return
        self.on_send()

    def on_open_analysis(self):
        p = getattr(self.v, "_ds3_path", None)
        if not p or not os.path.exists(p):
            return self.say("The image was not opened from a file in this session.")
        an = self.page.shell.analysis
        an.win.add_files([p])
        self.page.shell.show_module("analysis")
        sc = an.win.current_scene()
        if sc is not None and sc.geo is not None:
            an.win.viewer.chk_map.setChecked(True)
        else:
            self.page.shell.flash("Opened in Data Aggregation. The file has no georeferencing, so the map "
                                  "view is not available for it.")


class RoiLinkTab(W.QWidget):
    """'ROIs' tab of Viewer and Analysis: the Data Aggregation ROI Analysis toolbar, connected.

    ROIs are drawn in Data Aggregation (one place, one ROI Manager); this tab links the Viewer's image to its
    Data Aggregation scene (same file / name / size), mirrors that scene's ROIs into the Viewer - automatically
    on every change when 'Keep in sync' is on - and gives the Viewer tools that use ROIs (Classification,
    Spectral Matching, Segmentation, Unmixing, Data Correction) the ROI list and selection they expect."""

    def __init__(self, page):
        super().__init__()
        self.page = page
        self.v = page.viewer
        self._sig = None
        an = page.shell.analysis.win
        self.an = an
        # ---- 1. link to Data Aggregation
        g1 = W.QGroupBox("Link to Data Aggregation")
        l1 = W.QVBoxLayout(g1)
        self.lbl_link = label("", None, wrap=True, rich=True)
        l1.addWidget(self.lbl_link)
        self.chk_sync = W.QCheckBox("Keep ROIs in sync with Data Aggregation (automatic)")
        self.chk_sync.setChecked(True)
        self.chk_sync.setToolTip("Every ROI added, deleted, renamed or re-classed in Data Aggregation is copied here "
                                 "at once")
        self.chk_sync.toggled.connect(lambda on: on and self.sync(force=True))
        l1.addWidget(self.chk_sync)
        for text, fn, tip in (
                ("Send / select this image in Data Aggregation", self.on_link,
                 "Adds the Viewer's image (after preprocessing) to Data Aggregation, or selects it if it is there"),
                ("Draw ROIs in Data Aggregation  ▸", self.on_draw,
                 "Opens Data Aggregation ▸ ROI Analysis on this image with ROI selection switched on"),
                ("Get ROIs from Data Aggregation now", lambda: self.sync(force=True, quiet=False),
                 "Copies the linked scene's ROIs into the Viewer"),
                ("Load ROI file into Data Aggregation…", self.on_load_file,
                 "DS3-ROIs JSON, GeoJSON or ENVI ROI ASCII, loaded onto the linked scene (then synced here)"),
                ("Save ROIs to file…", self.on_save_file, "DS3-ROIs JSON / GeoJSON / ENVI ROI ASCII"),
                ("Save ROIs to library database…", self.on_save_db,
                 "The ROI shapes + their spectra into the spectral library (as in Data Aggregation)"),
                ("Load ROIs from library database…", self.on_load_db,
                 "ROIs stored in the library, onto the linked scene")):
            b = W.QPushButton(text)
            b.setToolTip(tip)
            b.clicked.connect(fn)
            l1.addWidget(b)
        l1.addWidget(label("ROIs are used here by: Classification (training samples), Spectral Matching (SAM "
                           "with ROI classes), Segmentation (label source 'ROIs'), Unmixing (endmember from an "
                           "ROI mean) and Data Correction (empirical line targets, dark pixel, flat field).",
                           "small", wrap=True))
        l1.addStretch(1)
        g1.setMinimumWidth(330)
        g1.setMaximumWidth(460)
        # ---- 2. ROI Manager (the viewer's own list widget, moved here)
        g2 = W.QGroupBox("ROI Manager (from Data Aggregation)")
        l2 = W.QVBoxLayout(g2)
        v = self.v
        top = W.QHBoxLayout()
        top.addWidget(v.roi_summary_label, 1)
        top.addWidget(v.cb_show_outlines)
        l2.addLayout(top)
        v.roi_list.setSelectionMode(W.QAbstractItemView.ExtendedSelection)
        v.roi_list.setMinimumHeight(150)
        l2.addWidget(v.roi_list, 1)
        l2.addWidget(label("Select ROIs here for tools that ask for 'selected ROIs' (Plot ROI spectra, dark pixel, "
                           "flat field, endmember). Nothing selected = all ROIs.", "small", wrap=True))
        r = W.QHBoxLayout()
        for text, fn in (("Plot ROI spectra", v.on_show_roi_spectra), ("Plot class means", v.on_show_class_spectra),
                         ("Selected ROI mean → endmember", self.on_endmember)):
            b = W.QPushButton(text)
            b.clicked.connect(fn)
            r.addWidget(b)
        l2.addLayout(r)
        r = W.QHBoxLayout()
        for text, fn in (("Show in Data Aggregation", self.on_show_in_da), ("Clear ROIs in the Viewer", self.on_clear)):
            b = W.QPushButton(text)
            b.clicked.connect(fn)
            r.addWidget(b)
        r.addStretch(1)
        l2.addLayout(r)
        lay = W.QHBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.addWidget(g1)
        lay.addWidget(g2, 1)
        # every ROI change in Data Aggregation goes through roipanel.refresh() -> sync from there
        rp = an.roipanel
        orig = rp.refresh

        def refresh(*a, _o=orig, **k):
            _o(*a, **k)
            try:
                self.sync()
            except Exception as e:                                               # noqa: BLE001
                self.page.shell.flash(f"⚠ ROI sync: {e}")
        rp.refresh = refresh
        self.update_status()

    # ------------------------------------------------------------ link
    def linked_scene(self):
        """The Data Aggregation scene of this image: same size, preferring the same file, then the same name."""
        v = self.v
        if v.cube is None:
            return None
        h, w = v.cube.shape[:2]
        cands = [sc for sc in self.an.scenes if tuple(sc.cube.shape[:2]) == (h, w)]
        if not cands:
            return None
        path, name = getattr(v, "_ds3_path", None), self.page.image_name()
        same = lambda a, b: bool(a and b) and os.path.normcase(os.path.abspath(str(a))) == \
            os.path.normcase(os.path.abspath(str(b)))                                   # noqa: E731
        cands.sort(key=lambda sc: (not same(sc.path, path), sc.name != name, not sc.rois))
        return cands[0]

    def update_status(self):
        v, sc = self.v, self.linked_scene()
        if v.cube is None:
            t = "No image open. Open one (File ▸ Open Image File…), then link it to Data Aggregation."
        elif sc is None:
            h, w = v.cube.shape[:2]
            t = (f"<b>Not linked.</b> Data Aggregation has no scene of {h} × {w} pixels. Click <i>Send / select "
                 "this image in Data Aggregation</i> (or <i>Draw ROIs in Data Aggregation</i>).")
        else:
            ncls = len({r["class"] for r in sc.rois})
            t = (f"<b>Linked to</b> Data Aggregation scene <b>{sc.name}</b> ({sc.cube.shape[0]} × "
                 f"{sc.cube.shape[1]}): {len(sc.rois)} ROI(s) in {ncls} class(es). "
                 f"Viewer: {len(v.rois)} ROI(s).")
        self.lbl_link.setText(t)

    @staticmethod
    def _signature(sc):
        if sc is None:
            return None
        out = []
        for r in sc.rois:
            geo = r.get("rect") or r.get("vertices") or (len(r.get("pixels") or []), tuple((r.get("pixels") or [])[:3]))
            out.append((r["id"], r["name"], r["class"], r["kind"], repr(geo)))
        return (id(sc), tuple(out))

    def sync(self, force=False, quiet=True):
        """Copy the linked scene's ROIs into the Viewer when they changed (or when forced)."""
        if self.v.cube is None:
            return self.update_status()
        sc = self.linked_scene()
        sig = self._signature(sc)
        if force or (self.chk_sync.isChecked() and sig != self._sig and sc is not None):
            self._sig = sig
            if sc is not None:
                self.page.rois_from_scene(sc, quiet=quiet)
            elif not quiet:
                self.page.rois_from_analysis(quiet=False)
        self.update_status()

    def _ensure_linked(self):
        if self.v.cube is None:
            W.QMessageBox.information(self, "No image", "Open an image first (File ▸ Open Image File…).")
            return None
        sc = self.linked_scene()
        if sc is None:
            self.page.link.on_send()
            sc = self.linked_scene()
        if sc is not None:
            self.an.select_scene(sc)
        return sc

    def on_link(self):
        sc = self._ensure_linked()
        if sc is not None:
            self.page.shell.show_module("image")
            self.sync(force=True)
            self.page.shell.flash(f"Linked to Data Aggregation scene '{sc.name}'.")

    def on_draw(self):
        if self._ensure_linked() is not None:
            self.page.shell.on_draw_roi()

    def _on_scene(self, fn):
        if self._ensure_linked() is not None:
            self.page.shell.show_module("image")
            fn()
            self.sync(force=True)

    def on_load_file(self):
        self._on_scene(self.an.roipanel.on_load_json)

    def on_save_file(self):
        if self._ensure_linked() is not None:
            self.an.roipanel.on_save_json()

    def on_save_db(self):
        if self._ensure_linked() is not None:
            self.an.roipanel.on_save_db()

    def on_load_db(self):
        self._on_scene(self.an.roipanel.on_load_db)

    def on_show_in_da(self):
        if self._ensure_linked() is not None:
            self.page.shell.analysis.goto_tab("ROI Analysis")

    def on_clear(self):
        v = self.v
        v.rois = []
        self._sig = None
        v._refresh_roi_list()
        v._refresh_class_combo()
        v._draw_roi_patch()
        self.update_status()
        self.page.shell.flash("ROIs cleared in Viewer and Analysis (Data Aggregation keeps its ROIs; turn off "
                              "'Keep in sync' to stop them coming back).")

    def on_endmember(self):
        v = self.v
        if v.cube is None or not v.rois:
            return W.QMessageBox.information(self, "No ROIs", "Get ROIs from Data Aggregation first.")
        rows = sorted(v._selected_roi_rows())
        if not rows:
            return W.QMessageBox.information(self, "No ROI selected", "Select the ROI(s) for the endmember first.")
        mask = np.zeros(v.cube.shape[:2], bool)
        for r in rows:
            mask |= np.asarray(v.rois[r]["mask"], bool)
        v.last_roi_mean_full = np.nanmean(np.asarray(v.cube[mask], np.float64), axis=0)
        v.on_add_endmember_from_roi()


class ImagePage(W.QWidget):
    """Hosts the viewer; created lazily on first visit."""

    def __init__(self, shell):
        super().__init__()
        self.shell = shell
        self.setObjectName("ds3Page")
        self.viewer = None
        self.link = None
        self.HV = None
        self.recent = []                      # paths opened in this session (File ▸ Open Previously Opened Image)
        self.n_opened = 0
        self._lay = W.QVBoxLayout(self)
        self._lay.setContentsMargins(10, 8, 10, 6)
        self._wait = label("Loading the image module…", "pagesub")
        self._wait.setAlignment(Qt.AlignCenter)
        self._lay.addWidget(self._wait, 1)

    def ensure(self):
        if self.viewer is not None:
            return self.viewer
        if getattr(self, "_building", False):          # re-entered while the viewer is being created
            return None
        self._building = True
        W.QApplication.setOverrideCursor(Qt.WaitCursor)
        self.shell.flash("Loading Viewer and Analysis (first time only)…")
        self.shell.statusBar().repaint()
        self._wait.repaint()
        try:
            from viewer import HyperVista_v1_0 as HV
            self.HV = HV
            style.register_theme(HV)
            HV.CURRENT_THEME = "ds3"
            v = HV.HyperspectralViewer()
            v.setWindowFlags(Qt.Widget)
            v.setMinimumSize(0, 0)
            v.statusBar().setVisible(False)            # its messages go to the DS3 status bar
            v.statusBar().messageChanged.connect(lambda m: m and self.shell.statusBar().showMessage(m, 8000))
            v.menuBar().setVisible(False)              # its File / View / Help live in the DS3 menus
            v._apply_window_qss("ds3")
            orig = v.load_cube_file

            def load_and_track(path, _orig=orig):
                _orig(path)
                if v.cube is not None:
                    v._ds3_path = str(path)
                    if str(path) in self.recent:
                        self.recent.remove(str(path))
                    self.recent.insert(0, str(path))
                    del self.recent[10:]
                    self.n_opened += 1
                    if self.link is not None:
                        self.link._class_map = None
                        self.link.draw_markers()
                    if getattr(self, "roilink", None) is not None:
                        v.rois = []                      # ROIs of the previous image do not fit this one
                        self.roilink._sig = None
                        v._refresh_roi_list()
                        self.roilink.sync()
                    self.shell.refresh_counts()
                    self.shell.update_title()
            v.load_cube_file = load_and_track
            for i in reversed(range(v.tabs.count())):    # DS3: ROIs are drawn in Data Aggregation, not here
                if v.tabs.tabText(i) == "ROI Analysis":
                    v.tabs.removeTab(i)
            v.cb_roi.setChecked(False)
            self._orig_union = v._selected_roi_union
            v._selected_roi_union = self._pick_roi_union          # selection in the ROIs tab, else ask
            self._relayout(v)
            self.viewer = v
            self._wait.setParent(None)
            hdr = W.QHBoxLayout()
            hdr.addWidget(label("Viewer and Analysis", "pagetitle"))
            sub = label("Open a hyperspectral image, preprocess, reduce, classify, segment and compare it with a "
                        "spectral library.", "pagesub")
            hdr.addSpacing(10)
            hdr.addWidget(sub, 1)
            self._lay.addLayout(hdr)
            self._lay.addWidget(v, 1)
            self.link = LibraryLinkTab(self)
            v.tabs.insertTab(0, self._wrap(self.link), "Library")
            self.roilink = RoiLinkTab(self)
            v.tabs.insertTab(1, self._wrap(self.roilink), "ROIs (Data Aggregation)")
            v.tabs.setCurrentIndex(0)
            self.shell.libpanel_libraries_changed.connect(self.link.refresh_libraries)
        finally:
            self._building = False
            W.QApplication.restoreOverrideCursor()
        self.shell.flash("Viewer and Analysis ready: File ▸ Open Image File… (Ctrl+O).")
        return self.viewer

    @staticmethod
    def _relayout(v):
        """Fit the viewer into the DS3 page: Display controls on one row, and image | tools in a draggable
        vertical splitter so the image keeps most of the height."""
        try:
            grid = v.radio_single.parentWidget().layout()
            items = []
            while grid.count():
                it = grid.takeAt(0)
                if it.widget() is not None:
                    items.append(it.widget())
            lab = [w for w in items if isinstance(w, W.QLabel)]
            order = [v.radio_single, v.radio_rgb] + [x for pair in zip(lab, (v.spin_r, v.spin_g, v.spin_b))
                                                     for x in pair]
            for i, w in enumerate(order):
                grid.addWidget(w, 0, i)
            for w in items:
                if w not in order:
                    grid.addWidget(w, 1, 0)
            for s in (v.spin_r, v.spin_g, v.spin_b):
                s.setMinimumWidth(70)
            grid.setColumnStretch(len(order), 1)
            grid.setContentsMargins(8, 4, 8, 4)
        except Exception:                                                        # noqa: BLE001
            pass
        try:
            main = v.centralWidget().layout()
            widgets = []
            while main.count():
                it = main.takeAt(0)
                if it.widget() is not None:
                    widgets.append(it.widget())
            top, rest = widgets[0], widgets[1:]
            bottom = W.QWidget()
            bl = W.QVBoxLayout(bottom)
            bl.setContentsMargins(0, 0, 0, 0)
            bl.setSpacing(4)
            for w in rest:
                bl.addWidget(w, 1 if w is v.tabs else 0)
            sp = W.QSplitter(Qt.Vertical)
            sp.setChildrenCollapsible(False)
            sp.addWidget(top)
            sp.addWidget(bottom)
            sp.setStretchFactor(0, 3)
            sp.setStretchFactor(1, 2)
            sp.setSizes([520, 330])
            main.addWidget(sp, 1)
            main.setContentsMargins(0, 4, 0, 0)
            v.plot_canvas.setMinimumHeight(180)
            v.info_label.setMinimumHeight(40)
            v.info_label.setMaximumHeight(110)
            v.image_canvas.setMinimumHeight(200)
            v.tabs.setMinimumHeight(170)
            v._ds3_splitter = sp
        except Exception:                                                        # noqa: BLE001
            pass

    def _wrap(self, w):
        sa = W.QScrollArea()
        sa.setWidgetResizable(True)
        sa.setFrameShape(W.QFrame.NoFrame)
        sa.setWidget(w)
        return sa

    def image_name(self):
        p = getattr(self.viewer, "_ds3_path", None) if self.viewer else None
        return os.path.splitext(os.path.basename(p))[0] if p else "image"

    def goto_tab(self, title, sub=None):
        v = self.ensure()
        for i in range(v.tabs.count()):
            if v.tabs.tabText(i).replace("&&", "&").lower().startswith(title.lower()):
                v.tabs.setCurrentIndex(i)
                break
        if sub is not None and hasattr(v, "prep_pages"):
            for i in range(v.prep_pages.count()):
                if v.prep_pages.tabText(i).replace("&&", "&").lower().startswith(sub.lower()):
                    v.prep_pages.setCurrentIndex(i)
        self.shell.show_module("image")

    # ------------------------------------------------------------ ROIs come from Data Aggregation
    def _pick_roi_union(self, what):
        """Replacement for the viewer's 'ROIs selected in the ROI Manager' (that tab is removed in DS3)."""
        v = self.viewer
        if v.rois and v._selected_roi_rows():
            return self._orig_union(what)                 # ROIs selected in the 'ROIs (Data Aggregation)' tab
        if not v.rois:
            W.QMessageBox.information(self, "No ROIs", f"The {what} ROI comes from Data Aggregation: draw it there "
                                      "(ROI Analysis tab); the 'ROIs (Data Aggregation)' tab brings it here.")
            return None
        items = [f"{i + 1}. {r['name']} [{r['class']}] - {int(np.asarray(r['mask']).sum())} px"
                 for i, r in enumerate(v.rois)]
        choice, ok = W.QInputDialog.getItem(self, f"{what.capitalize()} ROI", f"Use which ROI as the {what} ROI?",
                                            items, 0, False)
        if not ok:
            return None
        return np.asarray(v.rois[items.index(choice)]["mask"], bool).copy()

    def rois_from_analysis(self, quiet=False):
        """Copy the ROIs drawn in Data Aggregation onto the Viewer's image (for Classification, Spectral Matching,
        Segmentation, Unmixing, Data Correction). Uses the linked scene (same size; same file / name preferred)."""
        v = self.ensure()
        if v is None or v.cube is None:
            if not quiet:
                W.QMessageBox.information(self, "No image", "Open an image in Viewer and Analysis first.")
            return 0
        sc = self.roilink.linked_scene() if getattr(self, "roilink", None) else None
        if sc is None or not sc.rois:
            if not quiet:
                h, w = v.cube.shape[:2]
                W.QMessageBox.information(self, "No ROIs", f"No Data Aggregation scene of {h} x {w} pixels has ROIs. "
                                          "Use the 'ROIs (Data Aggregation)' tab: Send / select this image in Data "
                                          "Aggregation, then Draw ROIs in Data Aggregation.")
            return 0
        n = self.rois_from_scene(sc, quiet=quiet)
        if getattr(self, "roilink", None):
            self.roilink._sig = self.roilink._signature(sc)
            self.roilink.update_status()
        return n

    def rois_from_scene(self, sc, quiet=True):
        """Mirror the ROIs of Data Aggregation scene *sc* into the Viewer (rectangles, polygons; pixel lists as
        mask ROIs). Keeps the Viewer's ROI selection by name where possible."""
        from hvva.core.rois import roi_mask
        v = self.viewer
        h, w = v.cube.shape[:2]
        keep = {v.rois[i]["name"] for i in v._selected_roi_rows() if i < len(v.rois)}
        loaded = []
        for r in sc.rois:
            mask = roi_mask(r, h, w)
            if not mask.any():
                continue
            if r["kind"] == "rect":
                r0, c0, r1, c1 = (int(x) for x in r["rect"])
                roi = {"kind": "rect", "rect": (r0, r1 - 1, c0, c1 - 1)}
            elif r["kind"] == "polygon":
                roi = {"kind": "polygon", "vertices": [(float(x), float(y)) for x, y in r["vertices"]]}
            else:                                                   # pixel list -> the viewer's 'mask' ROI
                roi = {"kind": "mask", "rect": v._bbox_of(mask)}
            roi.update(name=str(r["name"]), **{"class": str(r["class"])}, mask=mask, bbox=v._bbox_of(mask))
            loaded.append(roi)
        v.rois = loaded
        v.roi_counter = max(getattr(v, "roi_counter", 0), len(loaded))
        for fn in ("_refresh_roi_list", "_refresh_class_combo"):
            try:
                getattr(v, fn)()
            except Exception:                                                    # noqa: BLE001
                pass
        if keep:
            for i, roi in enumerate(loaded):
                if roi["name"] in keep:
                    v.roi_list.item(i).setSelected(True)
        try:
            v._draw_roi_patch()
        except Exception:                                                        # noqa: BLE001
            pass
        classes = sorted({r["class"] for r in loaded})
        if loaded or not quiet:
            self.shell.flash(f"{len(loaded)} ROI(s) from Data Aggregation scene '{sc.name}'"
                             + (f" ({', '.join(classes)})" if classes else "")
                             + " — Classification, Spectral Matching, Segmentation, Unmixing and Data Correction "
                               "use them now.")
        return len(loaded)

    def as_scene(self, with_rois=True):
        """The viewer's active cube (+ ROIs as pixel lists) as a Scene of the Data Aggregation module."""
        from hvva.core.scene import Scene
        from hvva.core.geo import GeoInfo
        v = self.viewer
        cube = np.asarray(v.cube, np.float32)
        geo = None
        md = v.metadata or {}
        mi = md.get("map info") or md.get("map_info")
        if mi:
            try:
                cs = md.get("coordinate system string")
                geo = GeoInfo.from_envi_map_info(mi if isinstance(mi, str) else ", ".join(map(str, mi)),
                                                 cs if isinstance(cs, str) else None)
            except Exception:                                                    # noqa: BLE001
                geo = None
        rois = []
        if with_rois:
            for roi in v.rois:
                px = np.argwhere(np.asarray(roi["mask"], bool))
                if len(px):
                    rois.append({"kind": "pixels", "pixels": px.tolist(), "class": roi["class"], "name": roi["name"]})
        unit = self.link.unit() if self.link else viewer_unit(v)
        sc = Scene(self.image_name(), cube, None if v.wavelengths is None else np.asarray(v.wavelengths, float),
                   geo, rois=rois, path=getattr(v, "_ds3_path", None), unit=unit)
        return sc
