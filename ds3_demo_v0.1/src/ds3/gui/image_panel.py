"""Image viewer with library overlay: open imagery, click a pixel, bring a spectral library into the view.

Workflow (all steps also exist as plain methods used by the automated tests):
  open_image_path(path)      import (file / folder / .zip / .mat ...) and show it            -> load_image(cube_id)
  pick(row, col)             click on the image -> mean spectrum of the pixel (or n x n window)
  use_library(name)          bring a library from the database into the viewer
  import_library_file(path)  read ANY library file (.txt, .csv, .sli, ...) into the database and select it
  overlay()                  plot the selected library spectra over the pixel spectrum
  best_matches()             rank the library by spectral angle to the pixel spectrum
  sam_map()                  classify the whole image with the selected library spectra (spectral angle)
  save_pixel()               store the pixel spectrum as a new library entry (purity = image_pixel)
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib import colormaps
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QSplitter, QLabel, QLineEdit, QPushButton,
                               QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox, QListWidget, QAbstractItemView,
                               QTableWidget, QTableWidgetItem, QHeaderView, QFileDialog, QGroupBox)
from ..db import session_scope
from ..ingest import IngestError
from ..library import cube_roi, image_ops
from ..library.access import get_spectra, list_libraries, open_registered_cube
from ..library.search import find_spectra, match_spectrum
from ..pipeline import ingest_path, ingest_file
from ..ingest.readers.wavelength_files import read_wavelengths_file
from .plot import SpectrumCanvas

IMAGE_FILTER = ("Images (*.hdr *.tif *.tiff *.mat *.zip *.h5 *.he5 *.hdf5 *.nc *.zarr);;All files (*)")
LIB_FILTER = "Spectral libraries (*.txt *.csv *.tsv *.dat *.sli *.hdr *.sig);;All files (*)"
MAX_LIST = 2000


class ImageCanvas(FigureCanvasQTAgg):
    def __init__(self, on_click):
        self.fig = Figure(figsize=(5, 4), dpi=100, layout="constrained")
        super().__init__(self.fig)
        self.ax = self.fig.add_subplot(111)
        self._cb = on_click
        self.mpl_connect("button_press_event", self._press)
        self.message("Open an image to start")

    def message(self, msg):
        self.ax.clear(); self.ax.set_axis_off()
        self.ax.text(0.5, 0.5, msg, ha="center", va="center", transform=self.ax.transAxes, color="#666")
        self.draw_idle()

    def _press(self, ev):
        if ev.inaxes is self.ax and ev.xdata is not None and ev.button == 1:
            self._cb(float(ev.xdata), float(ev.ydata))

    def show(self, rgb, step, label, marker=None, classmap=None, colors=None):
        self.ax.clear(); self.ax.set_axis_on()
        self.ax.imshow(rgb, interpolation="nearest")
        if classmap is not None:
            h, w = classmap.shape
            ov = np.zeros((h, w, 4), np.float32)
            for k, col in enumerate(colors):
                ov[classmap == k] = (*col[:3], 0.6)
            self.ax.imshow(ov, interpolation="nearest")
        if marker is not None:
            r, c = marker
            self.ax.plot([c / step], [r / step], marker="+", color="yellow", ms=14, mew=2)
            self.ax.plot([c / step], [r / step], marker="o", mfc="none", mec="black", ms=9, mew=1)
        self.ax.set_title(label, fontsize=9)
        self.ax.set_xticks([]); self.ax.set_yticks([])
        self.draw_idle()


class ImagePanel(QWidget):
    def __init__(self, win):
        super().__init__()
        self.win = win
        self.cube = self.cube_id = self.preview = None
        self.pixel = None                      # (row, col, values, n_pixels)
        self.lib_rows, self.match_rows = [], []
        self._want = None
        self._classmap = None
        self._build()

    # ------------------------------------------------------------------ UI
    def _build(self):
        root = QVBoxLayout(self)
        bar = QHBoxLayout()
        self.cb_image = QComboBox(); self.cb_image.setMinimumWidth(300)
        self.cb_image.activated.connect(lambda _=0: self.load_image(self.cb_image.currentData()))
        b_open = QPushButton("Open image…"); b_open.clicked.connect(self._open_dialog)
        self.e_wl = QLineEdit(); self.e_wl.setPlaceholderText("wavelengths if the file has none: 430-860  or  a .xml/.txt/.csv file")
        self.e_wl.setMinimumWidth(320)
        self.cb_bands = QComboBox(); self.cb_bands.addItems(["True colour (auto)", "Custom bands (R,G,B) →"])
        self.e_bands = QLineEdit(); self.e_bands.setPlaceholderText("e.g. 60,40,20 (1-based)"); self.e_bands.setMaximumWidth(150)
        b_show = QPushButton("Update view"); b_show.clicked.connect(lambda: self.load_image(self.cube_id))
        for w in (QLabel("Image:"), self.cb_image, b_open, self.e_wl, self.cb_bands, self.e_bands, b_show):
            bar.addWidget(w)
        bar.addStretch(1)
        root.addLayout(bar)

        split = QSplitter(Qt.Horizontal)
        left = QWidget(); ll = QVBoxLayout(left); ll.setContentsMargins(0, 0, 0, 0)
        self.img = ImageCanvas(self._on_click); ll.addWidget(self.img, 1)
        row = QHBoxLayout()
        self.sp_row, self.sp_col = QSpinBox(), QSpinBox()
        for s in (self.sp_row, self.sp_col): s.setRange(0, 10_000_000)
        self.sp_win = QComboBox(); self.sp_win.addItems(["1 px", "3×3", "5×5", "9×9"])
        b_go = QPushButton("Go"); b_go.clicked.connect(lambda: self.pick(self.sp_row.value(), self.sp_col.value()))
        for w in (QLabel("row"), self.sp_row, QLabel("col"), self.sp_col, QLabel("window"), self.sp_win, b_go):
            row.addWidget(w)
        row.addStretch(1); ll.addLayout(row)
        self.lbl_img = QLabel("No image open. Use <b>Open image…</b> (AVIRIS-NG .hdr, EnMAP .zip/.TIF, PaviaU.mat, …) — "
                              "or pick one already in the database above.")
        self.lbl_img.setWordWrap(True); ll.addWidget(self.lbl_img)
        split.addWidget(left)

        right = QWidget(); rl = QVBoxLayout(right); rl.setContentsMargins(0, 0, 0, 0)
        self.spec = SpectrumCanvas(); rl.addWidget(self.spec, 2)
        gb = QGroupBox("Library in this view"); g = QGridLayout(gb)
        self.cb_lib = QComboBox(); self.cb_lib.activated.connect(lambda _=0: self.use_library(self.cb_lib.currentText()))
        b_file = QPushButton("Bring in library file…"); b_file.clicked.connect(self._library_dialog)
        g.addWidget(QLabel("Library"), 0, 0); g.addWidget(self.cb_lib, 0, 1); g.addWidget(b_file, 0, 2)
        self.lst = QListWidget(); self.lst.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.lst.itemSelectionChanged.connect(self.overlay)
        g.addWidget(self.lst, 1, 0, 1, 3)
        self.chk_unit = QCheckBox("ignore unit differences (image DN/radiance vs library reflectance — spectral angle only)")
        g.addWidget(self.chk_unit, 2, 0, 1, 3)
        self.sp_ang = QDoubleSpinBox(); self.sp_ang.setRange(0, 90); self.sp_ang.setValue(0); self.sp_ang.setSuffix("°  (0 = no limit)")
        b_match = QPushButton("Best matches for this pixel"); b_match.clicked.connect(self.best_matches)
        b_map = QPushButton("Classify image (SAM)"); b_map.clicked.connect(self.sam_map)
        b_clr = QPushButton("Clear map"); b_clr.clicked.connect(self.clear_map)
        g.addWidget(b_match, 3, 0); g.addWidget(b_map, 3, 1); g.addWidget(b_clr, 3, 2)
        g.addWidget(QLabel("Max angle"), 4, 0); g.addWidget(self.sp_ang, 4, 1)
        self.t_match = QTableWidget(0, 4); self.t_match.setHorizontalHeaderLabels(["#", "Material", "Library", "Angle"])
        self.t_match.setSelectionBehavior(QAbstractItemView.SelectRows); self.t_match.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.t_match.verticalHeader().hide(); self.t_match.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.t_match.setMaximumHeight(130); self.t_match.itemSelectionChanged.connect(self._match_selected)
        g.addWidget(self.t_match, 5, 0, 1, 3)
        self.e_name = QLineEdit("pixel"); self.e_save_lib = QLineEdit("Image pixels")
        b_save = QPushButton("Save pixel to library"); b_save.clicked.connect(self.save_pixel)
        g.addWidget(QLabel("Save as"), 6, 0); g.addWidget(self.e_name, 6, 1); g.addWidget(self.e_save_lib, 6, 2)
        g.addWidget(b_save, 7, 2)
        b_lab = QPushButton("Class spectra from label map…"); b_lab.clicked.connect(self._labels_dialog)
        b_lab.setToolTip("Ground-truth map (e.g. PaviaU_gt.mat): one mean spectrum per class is stored in a library")
        g.addWidget(b_lab, 8, 0, 1, 3)
        self.lbl = QLabel(""); self.lbl.setWordWrap(True); g.addWidget(self.lbl, 7, 0, 1, 2)
        rl.addWidget(gb, 3)
        split.addWidget(right); split.setSizes([620, 620])
        root.addWidget(split, 1)

    # ------------------------------------------------------------------ database lists
    def refresh_images(self):
        cubes = self.win.cubes
        cur = self.cube_id
        self.cb_image.blockSignals(True); self.cb_image.clear()
        for c in cubes:
            self.cb_image.addItem(f'#{c["id"]}  {c["name"]}  ({c["rows"]}×{c["cols"]}×{c["bands"]}, {c["unit"]})', c["id"])
        want = self._want or cur
        i = self.cb_image.findData(want) if want else -1
        if i >= 0: self.cb_image.setCurrentIndex(i)
        self.cb_image.blockSignals(False)
        if self._want is not None:
            cid, self._want = self._want, None
            self.load_image(cid)
        try:
            with session_scope(self.win.db_url) as s:
                names = [x["name"] for x in list_libraries(s) if x["n_spectra"]]
        except Exception:
            names = []
        keep = self.cb_lib.currentText()
        self.cb_lib.blockSignals(True); self.cb_lib.clear(); self.cb_lib.addItems(names)
        j = self.cb_lib.findText(keep)
        if j >= 0: self.cb_lib.setCurrentIndex(j)
        self.cb_lib.blockSignals(False)
        if names and self.lst.count() == 0:
            self.use_library(self.cb_lib.currentText())

    # ------------------------------------------------------------------ open imagery
    def _open_dialog(self):
        p, _ = QFileDialog.getOpenFileName(self, "Open image", str(Path.home()), IMAGE_FILTER)
        if p: self.open_image_path(p)

    def _reader_kw(self):
        txt = self.e_wl.text().strip()
        if not txt: return {}
        p = Path(txt)
        if p.is_file():
            wl, fw, _ = read_wavelengths_file(p); return dict(wavelengths=wl.tolist())
        import re
        m = re.fullmatch(r"\s*([0-9.]+)\s*[-–to ]+\s*([0-9.]+)\s*(nm)?\s*", txt)
        if not m: raise IngestError("wavelengths: type a range like 430-860, or choose a .xml/.txt/.csv file")
        return dict(wavelength_range=(float(m.group(1)), float(m.group(2))))

    def open_image_path(self, path, **kw):
        try:
            kw = {**self._reader_kw(), **kw}
        except Exception as e:
            self.win.show_error(f"{type(e).__name__}: {e}"); return
        if not path or not Path(path).exists():
            self.win.show_error(f"Path not found: {path or '(empty)'}"); return
        db = self.win.db_url

        def job():
            return ingest_path(path, db_url=db, **kw)

        def done(results):
            ok = [r for r in results if r.kind == "cube" and r.status in ("ingested", "duplicate")]
            for r in results:
                self.win.i_log.appendPlainText(f"[{r.status:9s}] {Path(r.path).name}  {r.detail}")
            if not ok:
                bad = next((r for r in results if r.status == "error"), None)
                self.win.show_error(bad.detail if bad else "No image found in " + str(path)
                                    + ("\n" + "\n".join(f"{Path(r.path).name}: {r.detail}" for r in results[:5]) if results else ""))
                return
            if results and results[0].ids:
                self._want = results[0].ids[0]
            else:
                self._want = None
            self._pending_dup = None
            if self._want is None:                      # duplicate: find its id by path
                self._pending_dup = Path(ok[0].path).stem
            self.win.refresh_all(search=False)
            if self._pending_dup:
                name = self._pending_dup
                for c in self.win.cubes:
                    if c["name"] == name:
                        self.load_image(c["id"]); break
        self.win.run(job, done, "Importing image…")

    def load_image(self, cube_id):
        if cube_id is None: return
        db = self.win.db_url
        txt = self.e_bands.text().strip() if self.cb_bands.currentIndex() == 1 else ""
        try:
            bands = [int(x) - 1 for x in txt.split(",")] if txt else None
            if bands is not None and len(bands) != 3: raise ValueError
        except ValueError:
            self.win.show_error("Custom bands: three band numbers (1-based) separated by commas, e.g. 60,40,20"); return

        def job():
            with session_scope(db) as s:
                cube, rec = open_registered_cube(s, cube_id)
            if bands is not None and not all(0 <= b < cube.bands for b in bands):
                raise IngestError(f"band numbers must be within 1..{cube.bands}")
            return cube, rec, image_ops.preview_rgb(cube, bands)

        def done(res):
            self.cube, rec, self.preview = res
            self.cube_id = rec["id"]; self.pixel = None; self._classmap = None
            i = self.cb_image.findData(self.cube_id)
            if i >= 0: self.cb_image.setCurrentIndex(i)
            self.sp_row.setMaximum(self.cube.rows - 1); self.sp_col.setMaximum(self.cube.cols - 1)
            wl = self.cube.wavelengths_nm
            notes = [n for n in self.cube.metadata.get("notes", []) if "ASSUMED" in n]
            self.lbl_img.setText(f'<b>{rec["name"]}</b> · {self.cube.rows}×{self.cube.cols}×{self.cube.bands} · {self.cube.unit} · '
                                 f'{wl[0]:.0f}–{wl[-1]:.0f} nm · {self.preview["label"]} · showing every {self.preview["step"]}. pixel'
                                 + ("<br><span style='color:#a15c00'>⚠ " + notes[0] + "</span>" if notes else "")
                                 + "<br>Click the image to read a pixel spectrum.")
            self._draw()
            self.spec.clear("Click a pixel in the image")
        self.win.run(job, done, "Opening image…")

    def _draw(self):
        if self.preview is None: return
        marker = (self.pixel[0], self.pixel[1]) if self.pixel else None
        colors = getattr(self, "_colors", None)
        self.img.show(self.preview["rgb"], self.preview["step"], self.lbl_img.text().split("·")[0].replace("<b>", "").replace("</b>", "").strip(),
                      marker, self._classmap, colors)

    # ------------------------------------------------------------------ pixel
    def _on_click(self, x, y):
        if self.preview is None: return
        st = self.preview["step"]
        self.pick(int(round(y * st)), int(round(x * st)))

    def pick(self, row, col):
        if self.cube is None:
            self.win.show_error("Open an image first"); return
        size = {0: 1, 1: 3, 2: 5, 3: 9}[self.sp_win.currentIndex()]
        cube = self.cube
        try:
            vals, n = image_ops.pixel_spectrum(cube, int(row), int(col), size)
        except Exception as e:
            self.win.show_error(f"{type(e).__name__}: {e}"); return
        if n == 0:
            self.win.show_error(f"pixel ({row}, {col}) is nodata in every band"); return
        self.pixel = (int(row), int(col), vals, n)
        self.sp_row.setValue(int(row)); self.sp_col.setValue(int(col))
        self._draw(); self.overlay()

    def _pixel_item(self):
        r, c, v, n = self.pixel
        v = v.copy(); v[np.asarray(self.cube.bad_band_mask, bool)] = np.nan
        return dict(label=f"image pixel ({r}, {c})" + (f" mean of {n}" if n > 1 else ""),
                    wavelengths_nm=self.cube.wavelengths_nm, values=v, style="k-")

    # ------------------------------------------------------------------ library
    def use_library(self, name):
        if not name: return
        db = self.win.db_url

        def job():
            with session_scope(db) as s:
                return find_spectra(s, library=name, limit=MAX_LIST)

        def done(res):
            self.lib_name = name; self.lib_rows = res["results"]
            i = self.cb_lib.findText(name)
            if i >= 0: self.cb_lib.setCurrentIndex(i)
            self.lst.blockSignals(True); self.lst.clear()
            for r in self.lib_rows:
                self.lst.addItem(f'{r["material"]}   [{r["category"] or "—"} · {r["unit"]} · {r["n_bands"]} bands]')
            self.lst.blockSignals(False)
            extra = f" (showing first {MAX_LIST})" if res["total"] > MAX_LIST else ""
            self.lbl.setText(f'Library "{name}": {res["total"]} spectra{extra}. Select spectra to overlay, or use the buttons.')
        self.win.run(job, done, "Loading library…")

    def _library_dialog(self):
        p, _ = QFileDialog.getOpenFileName(self, "Bring in a spectral library file", str(Path.home()), LIB_FILTER)
        if p: self.import_library_file(p)

    def import_library_file(self, path, library=None):
        """Any spectra file (.txt/.csv/.sli/.sig/...) -> database -> shown in this view."""
        if not Path(path).is_file():
            self.win.show_error(f"File not found: {path}"); return
        name = library or Path(path).stem
        db = self.win.db_url

        def job():
            return ingest_file(path, library=name, db_url=db)

        def done(r):
            self.win.i_log.appendPlainText(f"[{r.status:9s}] {Path(r.path).name}  {r.detail}")
            if r.status == "error" or r.status == "skipped":
                self.win.show_error(r.detail); return
            self.win.refresh_all(search=False)
            self.lbl.setText(f'{"Already in the database" if r.status == "duplicate" else "Imported"}: {Path(path).name} → library "{name}"')
            self.cb_lib.blockSignals(True)
            if self.cb_lib.findText(name) < 0: self.cb_lib.addItem(name)
            self.cb_lib.setCurrentText(name); self.cb_lib.blockSignals(False)
            self.use_library(name)
        self.win.run(job, done, "Reading library file…")

    def selected_library_ids(self, default_all=False):
        ids = [self.lib_rows[i.row()]["id"] for i in self.lst.selectedIndexes() if i.row() < len(self.lib_rows)]
        if len(ids) < 2 and default_all:            # a one-class map is meaningless: with fewer than 2 selected use the library
            ids = [r["id"] for r in self.lib_rows[:60]]
        return ids

    def _fetch(self, ids):
        with session_scope(self.win.db_url) as s:
            return get_spectra(s, ids)

    def _unit_note(self, recs):
        if self.cube is None: return ""
        bad = sorted({r["unit"] for r in recs if r["unit"] != self.cube.unit})
        return (f" ⚠ unit mismatch: image is {self.cube.unit}, library has {', '.join(bad)} — shapes are comparable, absolute values are not."
                if bad else "")

    def overlay(self):
        if self.cube is None or self.pixel is None:
            return
        ids = self.selected_library_ids()[:12]
        items = [self._pixel_item()]
        recs = []
        if ids:
            try:
                recs = self._fetch(ids)
            except Exception as e:
                self.win.show_error(str(e)); return
            items += [dict(label=f'{r["material"]}', wavelengths_nm=r["wavelengths_nm"], values=r["values"], style="--") for r in recs]
        self.spec.show_spectra(items, title=f"pixel vs library ({len(recs)} selected)", unit=self.cube.unit)
        self.lbl.setText((f'{len(recs)} library spectra overlaid.' if recs else "Select library spectra to overlay.") + self._unit_note(recs))

    def best_matches(self):
        if self.cube is None or self.pixel is None:
            self.win.show_error("Click a pixel in the image first"); return
        name = self.cb_lib.currentText()
        if not name:
            self.win.show_error("No library in the database yet — bring in a library file first"); return
        vals = self._pixel_item()["values"]; wl = self.cube.wavelengths_nm
        ok = np.isfinite(vals); db = self.win.db_url
        filt = dict(library=name)
        if not self.chk_unit.isChecked(): filt["unit"] = self.cube.unit

        def job():
            with session_scope(db) as s:
                return match_spectrum(s, wl[ok], vals[ok], top_k=10, metric="sam", min_overlap=0.5, **filt)

        def done(rows):
            self.match_rows = rows
            t = self.t_match; t.blockSignals(True); t.setRowCount(len(rows))
            for i, r in enumerate(rows):
                for j, v in enumerate([i + 1, r["material"], r["library"], f'{np.degrees(r["score"]):.2f}°']):
                    t.setItem(i, j, QTableWidgetItem(str(v)))
            t.blockSignals(False)
            if rows: t.selectRow(0)
            else:
                self.lbl.setText(f'No spectrum in "{name}" is {self.cube.unit} and overlaps the image bands. '
                                 'If the library is in another unit, tick "ignore unit differences".')
        self.win.run(job, done, "Matching…")

    def _match_selected(self):
        r = self.t_match.currentRow()
        if r < 0 or r >= len(self.match_rows) or self.pixel is None: return
        rec = self._fetch([self.match_rows[r]["id"]])[0]
        # scale the library spectrum to the pixel (spectral angle ignores gain) so shapes can be compared by eye
        px = self._pixel_item(); v = np.asarray(px["values"], float)
        lib_on = np.interp(px["wavelengths_nm"], rec["wavelengths_nm"], np.nan_to_num(rec["values"]))
        m = np.isfinite(v) & np.isfinite(lib_on) & (lib_on != 0)
        gain = float(np.nanmedian(v[m] / lib_on[m])) if m.any() else 1.0
        self.spec.show_spectra([px, dict(label=f'{rec["material"]} (scaled ×{gain:.3g})', wavelengths_nm=px["wavelengths_nm"],
                                          values=lib_on * gain, style="--")],
                               title=f'best match: {rec["material"]}  ({np.degrees(self.match_rows[r]["score"]):.2f}°)',
                               unit=self.cube.unit)

    def sam_map(self):
        if self.cube is None:
            self.win.show_error("Open an image first"); return
        ids = self.selected_library_ids(default_all=True)
        if not ids:
            self.win.show_error("Choose a library first (select 2 or more spectra to classify with, or none to use the whole library)"); return
        cube, maxang = self.cube, (self.sp_ang.value() or None)
        db = self.win.db_url

        def job():
            with session_scope(db) as s:
                recs = get_spectra(s, ids)
            M = image_ops.to_cube_grid(recs, cube.wavelengths_nm)
            return recs, image_ops.sam_map(cube, M, [r["material"] for r in recs], max_angle_deg=maxang)

        def done(res):
            recs, m = res
            n = len(recs)
            cm = colormaps["tab20" if n > 10 else "tab10"]
            self._colors = [cm(i % cm.N) for i in range(n)]
            # preview and class map share the same decimation only if steps agree; resize class map to preview grid
            cmap = m["index"]
            ph, pw = self.preview["rgb"].shape[:2]
            if cmap.shape != (ph, pw):
                ri = (np.arange(ph) * cmap.shape[0] // ph); ci = (np.arange(pw) * cmap.shape[1] // pw)
                cmap = cmap[np.ix_(ri, ci)]
            self._classmap = cmap
            self._draw()
            cnt = {recs[k]["material"]: int((m["index"] == k).sum()) for k in range(n)}
            top = sorted(cnt.items(), key=lambda t: -t[1])[:5]
            self.lbl.setText(f'SAM map: {n} library spectra, {m["bands_used"]} shared bands'
                             + (f", max angle {maxang:g}°" if maxang else "") + ". Largest classes: "
                             + ", ".join(f"{k} ({v} px)" for k, v in top) + "." + self._unit_note(recs))
            self.last_map = dict(m, names=[r["material"] for r in recs], counts=cnt)
        self.win.run(job, done, "Classifying…")

    def _labels_dialog(self):
        p, _ = QFileDialog.getOpenFileName(self, "Label map (ground truth)", str(Path.home()), "Label maps (*.mat *.npy *.tif *.tiff);;All files (*)")
        if p: self.labels_to_library(p)

    def labels_to_library(self, path, library=None):
        """Ground-truth label map -> one mean spectrum per class, stored in a library (purity = image_pixel)."""
        if self.cube is None:
            self.win.show_error("Open the image first, then its label map"); return
        cid, db = self.cube_id, self.win.db_url
        lib = library or f"{self.cube_name()} classes"

        def job():
            spectra, _ = cube_roi.measure_labels(cid, path, db_url=db)
            return spectra, cube_roi.save(spectra, cid, library=lib, db_url=db)

        def done(res):
            spectra, ids = res
            self.lbl.setText(f'{len(ids)} class spectra saved to library "{lib}": ' + ", ".join(s.material for s in spectra[:6]))
            self.win.refresh_all(search=True)
        self.win.run(job, done, "Measuring classes…")

    def cube_name(self):
        c = next((c for c in self.win.cubes if c["id"] == self.cube_id), None)
        return c["name"] if c else "image"

    def clear_map(self):
        self._classmap = None; self._draw()

    def save_pixel(self):
        if self.cube is None or self.pixel is None:
            self.win.show_error("Click a pixel first"); return
        r, c, _, _ = self.pixel
        size = {0: 1, 1: 3, 2: 5, 3: 9}[self.sp_win.currentIndex()]
        name, lib, cid, db = self.e_name.text().strip() or "pixel", self.e_save_lib.text().strip() or "Image pixels", self.cube_id, self.win.db_url
        if size == 1: args = dict(pixel=(r, c))
        else:
            h = size // 2
            args = dict(bbox_px=(max(0, r - h), max(0, c - h), min(self.cube.rows, r + h + 1), min(self.cube.cols, c + h + 1)))

        def job():
            spectra, _ = cube_roi.measure(cid, name=f"{name} ({r},{c})", db_url=db, **args)
            return cube_roi.save(spectra, cid, library=lib, db_url=db)

        def done(ids):
            self.lbl.setText(f'Saved pixel ({r}, {c}) to library "{lib}".')
            self.win.refresh_all(search=True)
        self.win.run(job, done, "Saving…")
