"""Library dialogs: Spectral Library Builder, library resampling, whole-library export, registered images.

    BuilderDialog          ENVI-style Spectral Library Builder - output grid (scene bands / band subset, ASCII
                           file or typed list with FWHM, first input, regular range), inputs (library spectra,
                           spectra files of any supported format, ROI means, collected spectra, last pixel),
                           linear or Gaussian resampling, minimum coverage, preview + rejection report, save
    ResampleDialog         copy a library onto a new wavelength grid (a new library)
    ExportLibraryDialog    whole library -> ENVI .sli / CSV / JSON (resampled when the grids differ)
    ImagesDialog           the images table: register the current scene, open a registered image, remove

All database work goes through the Library tab's client on its thread pool (panel.run).
"""
from __future__ import annotations

import os

import numpy as np
from PyQt5 import QtCore, QtWidgets

from ..library import builder as B

W = QtWidgets
Qt = QtCore.Qt


def _btn(text, fn, tip=None):
    b = W.QPushButton(text)
    b.clicked.connect(fn)
    if tip:
        b.setToolTip(tip)
    return b


def _row(*ws):
    r = W.QHBoxLayout()
    for w in ws:
        if isinstance(w, W.QWidget):
            r.addWidget(w)
        elif w is None:
            r.addStretch(1)
        else:
            r.addLayout(w)
    return r


def _group(title, *items):
    g = W.QGroupBox(title)
    lay = W.QVBoxLayout(g)
    for w in items:
        lay.addWidget(w) if isinstance(w, W.QWidget) else lay.addLayout(w)
    return g


# ====================================================================== grid chooser (shared)
class GridChooser(W.QWidget):
    """Output wavelength grid: scene bands (subset), typed / ASCII file, first input, regular range."""

    SOURCES = ("Current scene's bands", "Typed list / ASCII file", "Regular range", "First input spectrum")

    def __init__(self, host, first_input=None, allow_first=True):
        super().__init__()
        self.host = host
        self.first_input = first_input              # callable -> BuildInput or None
        self.cb = W.QComboBox()
        self.cb.addItems(self.SOURCES if allow_first else self.SOURCES[:3])
        self.stack = W.QStackedWidget()
        # scene
        self.sp_b0 = W.QSpinBox(); self.sp_b0.setRange(1, 100000); self.sp_b0.setValue(1)
        self.sp_b1 = W.QSpinBox(); self.sp_b1.setRange(1, 100000); self.sp_b1.setValue(100000)
        self.chk_bad = W.QCheckBox("Leave out the scene's bad bands"); self.chk_bad.setChecked(True)
        self.lbl_scene = W.QLabel("")
        f = W.QFormLayout(); f.addRow(self.lbl_scene); f.addRow("First band", self.sp_b0); f.addRow("Last band", self.sp_b1)
        f.addRow(self.chk_bad)
        p = W.QWidget(); p.setLayout(f); self.stack.addWidget(p)
        # text / file
        self.txt = W.QPlainTextEdit()
        self.txt.setPlaceholderText("450, 460, 470 …  or one wavelength per line, optionally a second column "
                                    "with the FWHM (nm or µm)")
        self.txt.setMaximumHeight(90)
        v = W.QVBoxLayout(); v.addWidget(self.txt); v.addLayout(_row(_btn("Load ASCII file…", self.on_file), None))
        p = W.QWidget(); p.setLayout(v); self.stack.addWidget(p)
        # range
        self.sp_r0, self.sp_r1, self.sp_rs = W.QDoubleSpinBox(), W.QDoubleSpinBox(), W.QDoubleSpinBox()
        for s_, v_ in ((self.sp_r0, 400), (self.sp_r1, 2500), (self.sp_rs, 10)):
            s_.setRange(0.001, 100000); s_.setDecimals(3); s_.setValue(v_)
        f = W.QFormLayout(); f.addRow("Start (nm)", self.sp_r0); f.addRow("Stop (nm)", self.sp_r1); f.addRow("Step (nm)", self.sp_rs)
        p = W.QWidget(); p.setLayout(f); self.stack.addWidget(p)
        # first input
        p = W.QLabel("The wavelengths of the first spectrum in the input list."); p.setWordWrap(True)
        self.stack.addWidget(p)
        self.cb.currentIndexChanged.connect(self.stack.setCurrentIndex)
        self.lbl = W.QLabel("")
        self.lbl.setWordWrap(True)
        lay = W.QVBoxLayout(self); lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.cb); lay.addWidget(self.stack); lay.addWidget(self.lbl)
        sc = host.current_scene()
        if sc is not None and sc.has_wavelengths:
            self.lbl_scene.setText(f"{sc.name}: {sc.n_bands} bands, {sc.wavelengths.min():g}–{sc.wavelengths.max():g} nm")
            self.sp_b1.setValue(sc.n_bands)
        else:
            self.lbl_scene.setText("No scene with wavelengths is selected (Scenes tab).")
            self.cb.setCurrentIndex(2)

    def on_file(self):
        p, _ = W.QFileDialog.getOpenFileName(self, "Wavelength file", "", "Text (*.txt *.csv *.dat *.wvl);;All files (*)")
        if p:
            try:
                with open(p, encoding="utf-8", errors="replace") as f:
                    self.txt.setPlainText(f.read())
            except OSError as e:
                W.QMessageBox.warning(self, "Wavelength file", str(e))

    def grid(self):
        """-> (wavelengths, fwhm or None, description); raises ValueError."""
        i = self.cb.currentIndex()
        if i == 0:
            sc = self.host.current_scene()
            if sc is None:
                raise ValueError("select a scene in the Scenes tab")
            wl, fw = B.grid_from_scene(sc, self.sp_b0.value(), self.sp_b1.value(), self.chk_bad.isChecked())
            desc = f"{sc.name} bands {self.sp_b0.value()}–{min(self.sp_b1.value(), sc.n_bands)}"
        elif i == 1:
            wl, fw = B.grid_from_text(self.txt.toPlainText())
            desc = "typed / file grid"
        elif i == 2:
            wl, fw = B.grid_from_range(self.sp_r0.value(), self.sp_r1.value(), self.sp_rs.value()), None
            desc = f"{self.sp_r0.value():g}–{self.sp_r1.value():g} nm step {self.sp_rs.value():g}"
        else:
            it = self.first_input() if self.first_input else None
            if it is None:
                raise ValueError("add an input spectrum first")
            wl = np.asarray(it.wavelengths, float)
            wl = wl[np.isfinite(wl)]
            fw, desc = None, f"grid of {it.name}"
        if wl.size < 2:
            raise ValueError("the grid has fewer than 2 wavelengths")
        self.lbl.setText(f"{wl.size} bands, {wl[0]:g}–{wl[-1]:g} nm" + (", with FWHM" if fw is not None else ""))
        return np.asarray(wl, float), fw, desc


def _method_combo():
    c = W.QComboBox()
    for k, v in B.METHODS.items():
        c.addItem(v, k)
    c.setToolTip("Linear: interpolation between neighbouring bands.  Gaussian: each output band is a Gaussian "
                 "(FWHM from the grid file, or the band spacing) over the input - like a sensor's response.")
    return c


def _coverage_spin():
    s = W.QSpinBox()
    s.setRange(1, 100)
    s.setValue(50)
    s.setSuffix(" %")
    s.setToolTip("Spectra covering less of the output grid are rejected (no extrapolation is ever done)")
    return s


# ====================================================================== builder
class BuilderDialog(W.QDialog):
    def __init__(self, panel):
        super().__init__(panel)
        self.panel = panel
        self.host = panel.host
        self.inputs = []                                     # [BuildInput]
        self.result = None
        self.setWindowTitle("Spectral Library Builder")
        self.resize(1060, 640)
        # inputs
        self.tbl = W.QTableWidget(0, 5)
        self.tbl.setHorizontalHeaderLabels(["Name", "Source", "Bands", "Range (nm)", "Unit"])
        self.tbl.horizontalHeader().setStretchLastSection(True)
        self.tbl.setSelectionBehavior(W.QAbstractItemView.SelectRows)
        self.tbl.setEditTriggers(W.QAbstractItemView.NoEditTriggers)
        self.cb_lib = W.QComboBox()
        for l in panel.libraries:
            self.cb_lib.addItem(l["name"], l["id"])
        if panel.active_lib:
            self.cb_lib.setCurrentIndex(max(0, self.cb_lib.findData(panel.active_lib["id"])))
        self.chk_selmat = W.QCheckBox("only the materials selected in the Library tab")
        add_lib = _row(_btn("Add library spectra", self.add_library,
                            "Spectra of the chosen library (Acquisition / Unit / Purity / Covers filters of the "
                            "Library tab apply)"), self.cb_lib, self.chk_selmat)
        add_other = _row(_btn("Add files…", self.add_files, "Spectra files of any supported format (CSV, ENVI .sli, "
                                                             "ASD, USGS / ECOSTRESS text, …)"),
                         _btn("Add ROI means", self.add_rois, "Mean of each selected ROI (all ROIs if none is selected)"),
                         _btn("Add collected spectra", self.add_collected),
                         _btn("Add last pixel", self.add_pixel), None,
                         _btn("Remove", self.remove), _btn("Clear", self.clear))
        self.lbl_in = W.QLabel("No inputs.")
        left = _group("Inputs", self.tbl, add_lib, add_other, self.lbl_in)
        # grid / options
        self.grid_ch = GridChooser(self.host, lambda: self.inputs[0] if self.inputs else None)
        self.cb_method = _method_combo()
        self.sp_cov = _coverage_spin()
        self.cb_unit = W.QComboBox()
        self.cb_unit.addItem("From the inputs", None)
        from ..core.quality import UNITS
        for u in UNITS:
            if u != "unknown":
                self.cb_unit.addItem(u, u)
        self.cb_unit.setToolTip("The library's value unit; inputs in another unit are rejected (never converted)")
        f = W.QFormLayout()
        f.addRow("Method", self.cb_method); f.addRow("Minimum coverage", self.sp_cov); f.addRow("Library unit", self.cb_unit)
        self.report = W.QPlainTextEdit(); self.report.setReadOnly(True)
        self.report.setPlaceholderText("Preview shows which spectra are added and why others are rejected.")
        b_prev = _btn("Preview", self.preview)
        self.b_save = _btn("Save as library…", self.save)
        self.b_export = _btn("Export file…", self.export_file, "Write the built library to .sli / CSV / JSON")
        self.b_save.setEnabled(False); self.b_export.setEnabled(False)
        right = _group("Output grid and resampling", self.grid_ch, f, self.report,
                       _row(b_prev, self.b_save, self.b_export, None, _btn("Close", self.reject)))
        lay = W.QHBoxLayout(self)
        lay.addWidget(left, 3)
        lay.addWidget(right, 2)

    # ------------------------------------------------------------ inputs
    def _add(self, items, what):
        self.inputs += items
        self.result = None
        self.b_save.setEnabled(False); self.b_export.setEnabled(False)
        self.refresh()
        self.host.say(f"Builder: added {len(items)} {what}.")

    def refresh(self):
        self.tbl.setRowCount(len(self.inputs))
        for r, it in enumerate(self.inputs):
            wl = np.asarray(it.wavelengths, float)
            vals = (it.name, it.source, wl.size, f"{np.nanmin(wl):g}–{np.nanmax(wl):g}", it.unit or "unknown")
            for c, v in enumerate(vals):
                self.tbl.setItem(r, c, W.QTableWidgetItem(str(v)))
        self.tbl.resizeColumnsToContents()
        units = {it.unit for it in self.inputs if it.unit not in (None, "unknown")}
        self.lbl_in.setText(f"{len(self.inputs)} input spectra" + (f" ⚠ units: {', '.join(sorted(units))} - "
                                                                    "only one unit is kept" if len(units) > 1 else ""))

    def add_library(self):
        lid = self.cb_lib.currentData()
        if lid is None:
            return
        lib = next((l for l in self.panel.libraries if l["id"] == lid), {"name": "library"})
        only = None
        if self.chk_selmat.isChecked():
            only = {m["id"] for m in self.panel._selected_materials()}
            if not only:
                return self.host._err("select materials in the Library tab first (or untick 'only the selected')")
        f = self.panel.filters()
        acq = self.panel.acq_filter
        from ..library.repository import spectrum_values, spectrum_wavelengths

        def ok(entries):
            items = []
            for e in entries:
                if only and e["material"]["id"] not in only:
                    continue
                s = e["spectrum"]
                items.append(B.BuildInput(e["material"]["name"], spectrum_wavelengths(s), spectrum_values(s),
                                          f"library {lib['name']}", e["material"].get("category"),
                                          s.get("acquisition_type"), s.get("measurement_unit") or lib.get("unit"),
                                          s.get("purity"), {"from_spectrum": s["id"]}))
            if not items:
                return self.host._err("no spectra in that library pass the filters")
            self._add(items, f"spectra from {lib['name']}")
        self.panel.run(self.panel.client.library_spectra, lid, acq, None, f["unit"], f["purity"], f["covers"], done=ok)

    def add_files(self):
        files, _ = W.QFileDialog.getOpenFileNames(self, "Spectra files", "", "All files (*)")
        if not files:
            return
        from ..library.formats import ReadOptions, read_path

        def go():
            out, errs = [], []
            for p in files:
                try:
                    recs, _info = read_path(p, ReadOptions())
                except Exception as e:                                       # noqa: BLE001
                    errs.append(f"{os.path.basename(p)}: {e}")
                    continue
                for r in recs:
                    out.append(B.BuildInput(r.name, np.asarray(r.wavelengths, float), np.asarray(r.values, float),
                                            os.path.basename(p), r.category, r.acquisition, r.unit, r.purity,
                                            {"file": os.path.basename(p)}))
            return out, errs

        def ok(res):
            items, errs = res
            if errs:
                self.host._err("; ".join(errs[:3]))
            if items:
                self._add(items, f"spectra from {len(files)} file(s)")
        self.panel.run(go, done=ok)

    def add_rois(self):
        from ..core import rois as R
        rp = self.host.roipanel
        items = rp.selected_items() or rp.all_items()
        out = []
        for s, r in items:
            if not s.has_wavelengths:
                continue
            st = R.roi_stats(s.cube, r, s.bad_bands)
            if st["n_pixels"]:
                out.append(B.BuildInput(f"{r['class']} · {r['name']}", np.asarray(s.wavelengths, float), st["mean"],
                                        f"{s.name} / {r['name']}", None, None, s.unit, s.purity,
                                        {"n_pixels": st["n_pixels"], "roi_class": r["class"], "scene": s.name}))
        if not out:
            return self.host._err("no ROIs on images with wavelengths (ROI Analysis tab)")
        self._add(out, "ROI means")

    def add_collected(self):
        out = [B.BuildInput(it["label"], np.asarray(it["wavelengths"], float), np.asarray(it["values"], float),
                            f"{it['scene']} pixel", None, None, it.get("unit"), it.get("purity"),
                            {"pixel": list(it["pixel"]), "window": it["size"]})
               for it in self.host.collected if it["wavelengths"] is not None]
        if not out:
            return self.host._err("no collected spectra (scene view: tick 'Collect spectra' and click pixels)")
        self._add(out, "collected spectra")

    def add_pixel(self):
        ls = self.host.last_spectrum
        if ls is None or ls[0] is None:
            return self.host._err("click a pixel of a scene with wavelengths first")
        sc = self.host.current_scene()
        px = self.host.last_pixel or ("?", "?")
        self._add([B.BuildInput(f"{sc.name if sc else 'scene'} pixel ({px[0]}, {px[1]})", np.asarray(ls[0], float),
                                np.asarray(ls[1], float), sc.name if sc else "scene", None, None,
                                sc.unit if sc else None, sc.purity if sc else "image_pixel")], "pixel spectrum")

    def remove(self):
        for r in sorted({i.row() for i in self.tbl.selectedIndexes()}, reverse=True):
            del self.inputs[r]
        self.refresh()

    def clear(self):
        self.inputs = []
        self.refresh()

    # ------------------------------------------------------------ build
    def preview(self):
        try:
            grid, fwhm, desc = self.grid_ch.grid()
            if not self.inputs:
                raise ValueError("add input spectra first")
            method = self.cb_method.currentData()
            self.result = B.build(self.inputs, grid, method, fwhm, self.sp_cov.value() / 100.0, self.cb_unit.currentData())
        except ValueError as e:
            self.report.setPlainText(f"⚠ {e}")
            return
        res = self.result
        lines = [f"Grid: {desc} - {grid.size} bands, {grid[0]:g}–{grid[-1]:g} nm; method {self.cb_method.currentText()}; "
                 f"unit {res.unit}.", f"Added: {len(res.added)}   Rejected: {len(res.rejected)}", ""]
        lines += [f"  ✓ {it.name}  ({it.source}, coverage {cov * 100:.0f}%)" for it, _y, cov in res.added[:200]]
        if res.rejected:
            lines += ["", "Rejected:"] + [f"  ✗ {it.name} ({it.source}): {why}" for it, why in res.rejected[:200]]
        self.report.setPlainText("\n".join(lines))
        ok = bool(res.added)
        self.b_save.setEnabled(ok); self.b_export.setEnabled(ok)
        if ok:
            import matplotlib
            pal = [matplotlib.colormaps["tab10"](i)[:3] for i in range(10)]
            ax = self.host.plot_canvas.reset()
            for k, (it, y, _c) in enumerate(res.added[:20]):
                ax.plot(res.grid, y, color=pal[k % 10], lw=1.1, label=it.name)
            self.host.plot_canvas.finish(f"Library Builder preview ({len(res.added)} spectra on {grid.size} bands"
                                         + (", first 20 shown)" if len(res.added) > 20 else ")"),
                                         "Wavelength (nm)", f"Value ({res.unit})", len(res.added) <= 12)

    def save(self):
        if self.result is None or not self.result.added:
            return
        items = B.result_items(self.result, self.cb_method.currentData())
        for it in items:
            if not it.get("acquisition_type"):
                it.pop("acquisition_type", None)            # the save dialog's acquisition applies
        ch = self.panel._ask_target(f"{len(items)} resampled spectra ({self.result.grid.size} bands, "
                                    f"{self.result.unit})", items[0]["name"] if len(items) == 1 else None)
        if not ch:
            return
        if len(items) == 1 and ch.get("name"):
            items[0]["name"] = ch["name"]
        self.panel._save(ch, items, "built spectra")
        self.accept()

    def export_file(self):
        if self.result is None or not self.result.added:
            return
        p, chosen = W.QFileDialog.getSaveFileName(self, "Export built library", "built_library.sli",
                                                  "ENVI spectral library (*.sli);;CSV (*.csv);;JSON (*.json)")
        if not p:
            return
        fmt = "csv" if p.lower().endswith(".csv") or chosen.startswith("CSV") else \
            "json" if p.lower().endswith(".json") or chosen.startswith("JSON") else "sli"
        entries = [{"material": {"name": it.name},
                    "spectrum": {"wavelengths": list(map(float, self.result.grid)),
                                 "values": [None if not np.isfinite(v) else float(v) for v in y],
                                 "measurement_unit": self.result.unit}} for it, y, _c in self.result.added]
        try:
            out = B.export_library(entries, p, fmt, library_name="HyperVista builder")
        except (OSError, ValueError) as e:
            return self.host._err(e)
        self.host.say("Wrote " + ", ".join(out["files"]))


# ====================================================================== resample a library
class ResampleDialog(W.QDialog):
    def __init__(self, panel, lib):
        super().__init__(panel)
        self.panel, self.lib, self.host = panel, lib, panel.host
        self.setWindowTitle(f"Resample library - {lib['name']}")
        self.resize(520, 480)
        self.grid_ch = GridChooser(self.host, allow_first=False)
        self.cb_method = _method_combo()
        self.sp_cov = _coverage_spin()
        self.ed_name = W.QLineEdit(f"{lib['name']} resampled")
        f = W.QFormLayout()
        f.addRow("Method", self.cb_method); f.addRow("Minimum coverage", self.sp_cov); f.addRow("New library", self.ed_name)
        self.lbl = W.QLabel("A new library is created; the original is not changed. Materials, categories, "
                            "acquisition, unit and purity are kept.")
        self.lbl.setWordWrap(True)
        self.b_go = _btn("Resample", self.go)
        lay = W.QVBoxLayout(self)
        lay.addWidget(_group("Output wavelength grid", self.grid_ch))
        lay.addLayout(f)
        lay.addWidget(self.lbl)
        lay.addLayout(_row(None, self.b_go, _btn("Close", self.reject)))

    def go(self):
        try:
            grid, fwhm, desc = self.grid_ch.grid()
        except ValueError as e:
            return self.lbl.setText(f"⚠ {e}")
        name = self.ed_name.text().strip()
        if not name:
            return self.lbl.setText("⚠ enter a name for the new library")
        self.b_go.setEnabled(False)
        self.lbl.setText(f"Resampling onto {grid.size} bands ({desc})…")

        def ok(res):
            self.b_go.setEnabled(True)
            txt = (f"Created '{res['library']['name']}': {res['spectra']} spectra on {grid.size} bands"
                   + (f", {res['skipped']} skipped:\n  " + "\n  ".join(res['reasons'][:8]) if res["skipped"] else "."))
            self.lbl.setText(txt)
            self.host.say(txt.splitlines()[0])
            self.panel.load_libraries(select_id=res["library"]["id"])

        def bad(msg):
            self.b_go.setEnabled(True)
            self.lbl.setText(f"⚠ {msg}")
        self.panel.run(self.panel.client.resample_library, self.lib["id"], grid, name, self.cb_method.currentData(),
                       fwhm, self.sp_cov.value() / 100.0, done=ok, failed=bad)


# ====================================================================== export a whole library
class ExportLibraryDialog(W.QDialog):
    def __init__(self, panel, lib):
        super().__init__(panel)
        self.panel, self.lib, self.host = panel, lib, panel.host
        self.entries = None
        self.setWindowTitle(f"Export library - {lib['name']}")
        self.resize(540, 480)
        self.cb_fmt = W.QComboBox()
        self.cb_fmt.addItem("ENVI spectral library (.sli + .hdr)", "sli")
        self.cb_fmt.addItem("CSV (wavelength + one column per spectrum)", "csv")
        self.cb_fmt.addItem("JSON", "json")
        self.chk_filters = W.QCheckBox("Apply the Library tab's filters (acquisition, category, unit, purity, covers)")
        self.chk_filters.setChecked(True)
        self.chk_all = W.QCheckBox("All samples of each material (default: the first spectrum per material)")
        self.cb_grid = W.QComboBox()
        self.cb_grid.addItems(["Keep the library's grid (must be shared)", "Most common grid in the library",
                               "Choose a grid…"])
        self.grid_ch = GridChooser(self.host, allow_first=False)
        self.grid_ch.setVisible(False)
        self.cb_grid.currentIndexChanged.connect(lambda i: self.grid_ch.setVisible(i == 2))
        self.cb_method = _method_combo()
        f = W.QFormLayout()
        f.addRow("Format", self.cb_fmt); f.addRow(self.chk_filters); f.addRow(self.chk_all)
        f.addRow("Wavelength grid", self.cb_grid); f.addRow(self.grid_ch); f.addRow("Resampling", self.cb_method)
        self.lbl = W.QLabel(""); self.lbl.setWordWrap(True)
        self.b_go = _btn("Export…", self.go)
        lay = W.QVBoxLayout(self)
        lay.addLayout(f)
        lay.addWidget(self.lbl)
        lay.addStretch(1)
        lay.addLayout(_row(None, self.b_go, _btn("Close", self.reject)))

    def go(self):
        fmt = self.cb_fmt.currentData()
        ext = {"sli": ".sli", "csv": ".csv", "json": ".json"}[fmt]
        base = "".join(ch if ch.isalnum() or ch in "-_ " else "_" for ch in self.lib["name"]).strip() or "library"
        p, _ = W.QFileDialog.getSaveFileName(self, "Export library", base + ext, f"*{ext}")
        if not p:
            return
        if not p.lower().endswith(ext):
            p += ext
        grid = None
        if self.cb_grid.currentIndex() == 2:
            try:
                grid = self.grid_ch.grid()[0]
            except ValueError as e:
                return self.lbl.setText(f"⚠ {e}")
        most_common = self.cb_grid.currentIndex() == 1
        pn = self.panel
        f = pn.filters() if self.chk_filters.isChecked() else {"unit": None, "purity": None, "covers": None}
        acq = pn.acq_filter if self.chk_filters.isChecked() else None
        cat = pn.cat_path if self.chk_filters.isChecked() else None
        c, lid, name, method = pn.client, self.lib["id"], self.lib["name"], self.cb_method.currentData()
        all_samples = self.chk_all.isChecked()

        def work():
            entries = c.library_spectra(lid, acq, cat, f["unit"], f["purity"], f["covers"], all_samples)
            g = B.most_common_grid(entries) if most_common and entries else grid
            if all_samples:                                   # unique names for repeated samples
                seen = {}
                for e in entries:
                    n = e["material"]["name"]
                    seen[n] = seen.get(n, 0) + 1
                    if seen[n] > 1:
                        e["material"] = dict(e["material"], name=f"{n} ({seen[n]})")
            return B.export_library(entries, p, fmt, g, method, name)

        def ok(out):
            self.b_go.setEnabled(True)
            txt = (f"Wrote {out['spectra']} spectra × {out['bands']} bands" + (" (resampled)" if out["resampled"] else "")
                   + ":\n  " + "\n  ".join(out["files"]))
            if out["skipped"]:
                txt += f"\n{len(out['skipped'])} left out:\n  " + "\n  ".join(out["skipped"][:8])
            self.lbl.setText(txt)
            self.host.say(txt.splitlines()[0])

        def bad(msg):
            self.b_go.setEnabled(True)
            self.lbl.setText(f"⚠ {msg}" + ("\nChoose 'Most common grid' or a grid to resample to."
                                           if "different wavelength grids" in msg else ""))
        self.b_go.setEnabled(False)
        self.lbl.setText("Exporting…")
        pn.run(work, done=ok, failed=bad)


# ====================================================================== images table
class ImagesDialog(W.QDialog):
    COLS = ["Name", "Size (rows × cols × bands)", "Unit", "Sensor", "Range (nm)", "CRS", "ROIs", "Path", "Registered"]

    def __init__(self, host):
        super().__init__(host)
        self.host = host
        self.panel = host.libpanel
        self.images = []
        self.setWindowTitle("Library images")
        self.resize(980, 420)
        self.ed = W.QLineEdit(); self.ed.setPlaceholderText("Filter by name"); self.ed.setClearButtonEnabled(True)
        self.ed.textChanged.connect(lambda *_: self.load())
        self.tbl = W.QTableWidget(0, len(self.COLS))
        self.tbl.setHorizontalHeaderLabels(self.COLS)
        self.tbl.horizontalHeader().setStretchLastSection(True)
        self.tbl.setSelectionBehavior(W.QAbstractItemView.SelectRows)
        self.tbl.setEditTriggers(W.QAbstractItemView.NoEditTriggers)
        self.tbl.doubleClicked.connect(lambda *_: self.open_selected())
        self.lbl = W.QLabel("Images are registered automatically when ROIs are saved to the library; "
                            "the pixels stay in the file (path + fingerprint + metadata are stored).")
        self.lbl.setWordWrap(True)
        lay = W.QVBoxLayout(self)
        lay.addWidget(self.ed)
        lay.addWidget(self.tbl, 1)
        lay.addWidget(self.lbl)
        lay.addLayout(_row(_btn("Register current scene", self.register), _btn("Open selected", self.open_selected),
                           _btn("Remove entry", self.remove), None, _btn("Close", self.reject)))
        self.load()

    def load(self):
        q = self.ed.text().strip() or None

        def ok(res):
            self.images = res.get("items", res) if isinstance(res, dict) else res
            self.tbl.setRowCount(len(self.images))
            for r, im in enumerate(self.images):
                rng = "" if im.get("wl_min_nm") is None else f"{im['wl_min_nm']:g}–{im['wl_max_nm']:g}"
                vals = (im["name"], f"{im['rows']} × {im['cols']} × {im['bands']}", im.get("unit") or "",
                        im.get("sensor") or "", rng, im.get("crs") or "", im.get("n_rois", 0),
                        im.get("path") or "(in memory)",
                        (im.get("created_at") or "")[:19].replace("T", " "))
                for c, v in enumerate(vals):
                    it = W.QTableWidgetItem(str(v))
                    if c == 7:
                        it.setToolTip(str(v))
                    self.tbl.setItem(r, c, it)
            self.tbl.resizeColumnsToContents()
        self.panel.run(self.panel.client.list_images, q, done=ok)

    def _selected(self):
        rows = sorted({i.row() for i in self.tbl.selectedIndexes()})
        return [self.images[r] for r in rows if r < len(self.images)]

    def register(self):
        sc = self.host.current_scene()
        if sc is None:
            return self.host._err("select a scene in the Scenes tab")
        from ..core.scene import image_info
        info = image_info(sc)

        def ok(im):
            self.lbl.setText(f"Registered {im['name']} (fingerprint {im['fingerprint'][:12]}…).")
            self.load()
        self.panel.run(self.panel.client.register_image, info, done=ok)

    def open_selected(self):
        sel = self._selected()
        if not sel:
            return
        im = sel[0]
        p = im.get("path")
        loaded = [s for s in self.host.scenes if s.path and p and os.path.abspath(s.path) == os.path.abspath(p)]
        if loaded:
            self.host.select_scene(loaded[0])
            return self.lbl.setText(f"{im['name']} is already loaded - selected it.")
        if not p or not os.path.exists(p):
            return self.lbl.setText(f"⚠ The file is not at {p or '(no path)'} on this computer. Add it with "
                                    "File > Add scenes; it is recognised by its fingerprint.")
        self.host.add_files([p])
        self.lbl.setText(f"Opened {os.path.basename(p)}.")

    def remove(self):
        sel = self._selected()
        if not sel:
            return
        if W.QMessageBox.question(self, "Remove image entries",
                                  f"Remove {len(sel)} image entr{'y' if len(sel) == 1 else 'ies'} from the library "
                                  "database? ROIs keep their geometry (their image link is cleared); the files on "
                                  "disk are not touched.") != W.QMessageBox.Yes:
            return
        c = self.panel.client

        def go():
            for im in sel:
                c.delete_image(im["id"])
            return len(sel)
        self.panel.run(go, done=lambda n: (self.lbl.setText(f"Removed {n} entr{'y' if n == 1 else 'ies'}."),
                                           self.load()))
