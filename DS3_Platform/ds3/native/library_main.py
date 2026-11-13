"""Native PyQt5 'Library (main)' page: the libraries of the backend database (main libraries from the libraries
folder, pixel-based and object-based libraries), material search, library / material pages, compare, and the
library functions (Builder, Viewer, Resampling). Replaces the web page (index.html + JS) shown in QtWebEngine;
the data still comes from the same FastAPI backend."""
from __future__ import annotations

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

from .. import style
from ..style import C
from ..widgets import Card, CardGrid, Pill, StatCard, label, page_header
from .api import run_bg
from .common import DS_COLORS, NONE_TOKEN, SpectrumChart, angle_deg, btn, card, hrow, note_err, save_csv, small, xy
from .library_tools import (FunctionsView, Hub, ImportDialog, PatchDialog, SpectraBrowser, cell, get_spectrum, section,
                            table)

W = QtWidgets
Qt = QtCore.Qt


def scroll(inner):
    sa = W.QScrollArea()
    sa.setWidgetResizable(True)
    sa.setFrameShape(W.QFrame.NoFrame)
    sa.setWidget(inner)
    return sa


class LibCard(Card):
    def __init__(self, lib, idx):
        super().__init__(clickable=True)
        self.lib = lib
        self.lay.setSpacing(4)
        kind = 0 if lib["main"] else (2 if lib["type"] == "object" else 5)
        self.lay.addWidget(Pill(lib["source"], kind))
        self.lay.addWidget(label(lib["name"], "cardtitle", wrap=True))
        self.lay.addWidget(label(f"{lib['count']:,} materials · {lib['spectra']:,} spectra", "small"))
        if lib.get("description"):
            self.lay.addWidget(label(str(lib["description"])[:120], "small", wrap=True))
        self.setToolTip(f"{lib['name']}\nClick to open: materials, spectra, compare")
        self.setMinimumHeight(92)


# ------------------------------------------------------------------ overview
class OverviewView(W.QWidget):
    def __init__(self, page):
        super().__init__()
        self.page, self.api, self.store = page, page.api, page.store
        body = W.QWidget()
        lay = W.QVBoxLayout(body)
        lay.setContentsMargins(28, 22, 28, 22)
        lay.setSpacing(12)
        top = W.QHBoxLayout()
        top.addLayout(page_header("Library", "Manage and explore spectral libraries."), 1)
        top.addWidget(btn("+ Add library (import file)…", self.page.import_files, kind="primary", accent="library",
                          tip="Read a spectral library file (CSV, ASCII, USGS, ECOSTRESS, ENVI .sli, zip) into a new library"))
        top.addWidget(btn("Build library…", lambda: self.page.open_functions("builder"),
                          tip="Resample, merge and save spectra as a new library"))
        top.addWidget(btn("Viewer", lambda: self.page.open_functions("viewer")))
        top.addWidget(btn("Resampling", lambda: self.page.open_functions("resample")))
        top.addWidget(btn("Compare", lambda: self.page.open_compare()))
        top.addWidget(btn("⟳", self.store.refresh, tip="Reload the list from the database"))
        lay.addLayout(top)

        stats = W.QHBoxLayout()
        self.st = {k: StatCard(t) for k, t in (("libraries", "Libraries"), ("materials", "Materials"),
                                                ("spectra", "Spectra"), ("images", "Images"))}
        for s in self.st.values():
            stats.addWidget(s)
        self.lab_db = small("")
        stats.addWidget(self.lab_db, 1)
        lay.addLayout(stats)

        self.banner, bl = card(W.QHBoxLayout, (14, 10, 14, 10))
        self.lab_main = small("", rich=True)
        self.lab_main.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.b_reload = btn("Reload main libraries", self.reload_main)
        bl.addWidget(self.lab_main, 1)
        bl.addWidget(self.b_reload)
        lay.addWidget(self.banner)
        self.lab_issues = small("", rich=True)
        self.lab_issues.setStyleSheet(f"color:{C['red']};")
        lay.addWidget(self.lab_issues)

        g = W.QGridLayout()
        self.ed_lib = W.QLineEdit()
        self.ed_lib.setPlaceholderText("e.g. ECOSTRESS")
        self.ed_mat = W.QLineEdit()
        self.ed_mat.setPlaceholderText("e.g. basalt, concrete, grass")
        for e in (self.ed_lib, self.ed_mat):
            style.prop(e, ds3="search")
            e.setClearButtonEnabled(True)
        g.addWidget(small("Search libraries by name"), 0, 0)
        g.addWidget(small("Search materials in all libraries"), 0, 1)
        g.addWidget(self.ed_lib, 1, 0)
        g.addWidget(self.ed_mat, 1, 1)
        lay.addLayout(g)
        self.mat_box, ml = card()
        self.lab_hits = small("")
        self.lst_hits = W.QListWidget()
        self.lst_hits.setMaximumHeight(260)
        self.lst_hits.itemActivated.connect(self._open_hit)
        self.lst_hits.itemClicked.connect(self._open_hit)
        ml.addWidget(self.lab_hits)
        ml.addWidget(self.lst_hits)
        self.mat_box.setVisible(False)
        lay.addWidget(self.mat_box)
        self.lay_sections = W.QVBoxLayout()
        self.lay_sections.setSpacing(8)
        lay.addLayout(self.lay_sections)
        lay.addWidget(small("Libraries made from collected pixel spectra or object samples are saved from the Image "
                            "page and listed here."))
        lay.addStretch(1)
        outer = W.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll(body))
        self.grids = []
        self._t = QtCore.QTimer(self)
        self._t.setSingleShot(True)
        self._t.setInterval(300)
        self._t.timeout.connect(self.search_materials)
        self._seq = 0
        self.ed_lib.textChanged.connect(lambda _: self.draw_cards())
        self.ed_mat.textChanged.connect(lambda _: self._t.start())
        self.store.changed.connect(self.redraw)

    def redraw(self):
        s = self.store.stats or {}
        for k, w in self.st.items():
            w.set(s.get(k))
        self.lab_db.setText(f"  Database: {self.store.db_kind()}" if s else
                            (f"  The backend is not reachable: {self.store.error}" if self.store.error else ""))
        su = self.store.startup or {}
        n_main = sum(1 for l in self.store.libraries if l["main"])
        if su.get("folder"):
            if su.get("running"):
                st = ("<span style='color:#b45309'>⌛ Reading the libraries folder… large files (e.g. ECOSTRESS) can "
                      "take a minute.</span>")
            elif n_main:
                st = f"{n_main} main librar{'y' if n_main == 1 else 'ies'} loaded."
            else:
                st = ("<b style='color:#b45309'>No main library files found.</b> Copy your library files (.txt / .csv / …) "
                      "into the folder below, then press <b>Reload main libraries</b>.")
            ex = " (does not exist yet — create it)" if su.get("folder_exists") is False else ""
            self.lab_main.setText(f"{st}<br>Folder: <code>{su['folder']}</code>{ex}")
            self.b_reload.setEnabled(not su.get("running"))
            self.banner.setVisible(True)
        else:
            self.banner.setVisible(False)
        bad = [f for f in su.get("files") or [] if f.get("status") in ("error", "skipped")]
        self.lab_issues.setText((f"{len(bad)} file(s) in the libraries folder could not be read:<br>" +
                                 "<br>".join(f"{f['file']}: {f.get('detail') or ''}" for f in bad[:12])) if bad else "")
        self.lab_issues.setVisible(bool(bad))
        self.draw_cards()

    def draw_cards(self):
        while self.lay_sections.count():
            it = self.lay_sections.takeAt(0)
            if it.widget():
                it.widget().deleteLater()
        q = self.ed_lib.text().strip().lower()
        libs = [l for l in self.store.libraries if q in l["name"].lower()]
        if not self.store.libraries:
            self.lay_sections.addWidget(label("No libraries yet." if self.store.stats else
                                              "Waiting for the database…", "muted"))
            return
        if not libs:
            self.lay_sections.addWidget(label("No libraries match your search.", "muted"))
            return
        groups = (("Main libraries", "loaded from the libraries folder", [l for l in libs if l["main"]]),
                  ("Pixel-based libraries", "one spectrum per pixel or measurement: built with the Spectral Library "
                   "Builder, collected on an image, or imported",
                   [l for l in libs if not l["main"] and l["type"] != "object"]),
                  ("Object-based libraries (ROI)", "one mean spectrum per ROI; each keeps its image patch file (<class>.tif)",
                   [l for l in libs if not l["main"] and l["type"] == "object"]))
        for title, sub, arr in groups:
            if not arr:
                continue
            h = W.QLabel(f"<b>{title}</b> <span style='color:{C['muted']}'>  {len(arr)} · {sub}</span>")
            h.setTextFormat(Qt.RichText)
            h.setWordWrap(True)
            self.lay_sections.addWidget(h)
            grid = CardGrid(min_width=260, max_cols=4, spacing=12)
            cards = []
            for i, l in enumerate(arr):
                c = LibCard(l, i)
                c.clicked.connect(lambda l=l: self.page.open_library(l["id"]))
                cards.append(c)
            grid.set_cards(cards)
            self.lay_sections.addWidget(grid)

    def reload_main(self):
        self.b_reload.setEnabled(False)
        run_bg(self.api.post, "/api/startup_import/reload", {}, done=lambda _: self.store.refresh(),
               failed=lambda m: (self.store.refresh(), W.QMessageBox.warning(self, "Reload", m)))

    def search_materials(self):
        q = self.ed_mat.text().strip()
        self._seq += 1
        seq = self._seq
        if not q:
            self.mat_box.setVisible(False)
            return
        self.mat_box.setVisible(True)
        self.lab_hits.setText("Searching…")

        def done(r):
            if seq != self._seq:
                return
            hits = r.get("results", [])
            self.lst_hits.clear()
            seen = set()
            for h in hits:
                key = (h["library_id"], h["material_id"])
                if key in seen:
                    continue
                seen.add(key)
                it = W.QListWidgetItem(f"{h['material']}   ·  {h.get('library', '')}" +
                                       (f" · {h['category']}" if h.get("category") else "") +
                                       f" · {h.get('unit') or ''}")
                it.setData(Qt.UserRole, h)
                self.lst_hits.addItem(it)
            tot = r.get("total", len(hits))
            self.lab_hits.setText(f"{self.lst_hits.count()} material(s) match “{q}”" +
                                  (f" (first {len(hits)} of {tot} spectra)" if tot > len(hits) else "") +
                                  " — click one to open it" if hits else f"No material matches “{q}”.")
        run_bg(self.api.get, "/api/spectra/search", q=q, limit=60, done=done,
               failed=lambda m: self.lab_hits.setText(m))

    def _open_hit(self, it):
        h = it.data(Qt.UserRole)
        self.page.open_material(h["library_id"], h["material_id"], h["material"])


# ------------------------------------------------------------------ one library
class LibraryView(W.QWidget):
    def __init__(self, page):
        super().__init__()
        self.page, self.api, self.store = page, page.api, page.store
        self.lib = None
        self.objects = []
        self.materials = []
        self._seq = 0
        lay = W.QVBoxLayout(self)
        lay.setContentsMargins(28, 14, 28, 18)
        lay.setSpacing(10)
        self.title = label("", "pagetitle")
        self.sub = label("", "pagesub", wrap=True)
        hd = W.QHBoxLayout()
        tb = W.QVBoxLayout()
        tb.addWidget(self.title)
        tb.addWidget(self.sub)
        hd.addLayout(tb, 1)
        self.b_upload = btn("Upload files into this library", self.upload, kind="primary", accent="library")
        self.b_resample = btn("Resample…", lambda: self.page.open_functions("resample", self.lib and self.lib["id"]))
        self.b_csv = btn("Export CSV", self.export)
        self.b_del = btn("Delete library", self.delete)
        for b in (self.b_upload, self.b_resample, self.b_csv, self.b_del):
            hd.addWidget(b)
        lay.addLayout(hd)
        self.ed_q = W.QLineEdit()
        self.ed_q.setPlaceholderText("search materials…")
        self.ed_q.setClearButtonEnabled(True)
        style.prop(self.ed_q, ds3="search")
        self.cb_cat = W.QComboBox()
        self.cb_cat.setMinimumWidth(200)
        self.cb_acq = W.QComboBox()
        self.cb_acq.addItem("Any acquisition", "")
        for a in ("lab", "field", "airborne", "satellite"):
            self.cb_acq.addItem(a, a)
        self.cb_acq.addItem("not set", NONE_TOKEN)
        self.filters = hrow(self.ed_q, self.cb_cat, self.cb_acq, stretch_at=0)
        lay.addLayout(self.filters)
        self.info = small("")
        lay.addWidget(self.info)
        self.lst = W.QListWidget()
        self.lst.itemActivated.connect(self._open_mat)
        self.lst.itemClicked.connect(self._open_mat)
        lay.addWidget(self.lst, 1)
        self.tbl = table(["Label", "Class", "Latitude", "Longitude", "Pixels", "Image (patch file)", ""], 5)
        lay.addWidget(self.tbl, 1)
        self.note = small("The image patches stay in the folder where they were saved; the database keeps only the file "
                          "name and its address.")
        lay.addWidget(self.note)
        self._t = QtCore.QTimer(self)
        self._t.setSingleShot(True)
        self._t.setInterval(250)
        self._t.timeout.connect(self.load_materials)
        self.ed_q.textChanged.connect(lambda _: self._t.start() if not self._is_obj() else self._draw_objects())
        self.cb_cat.currentIndexChanged.connect(lambda _: self.load_materials())
        self.cb_acq.currentIndexChanged.connect(lambda _: self.load_materials())

    def _is_obj(self):
        return bool(self.lib and self.lib["type"] == "object")

    def open(self, lid, then=None):
        self.lib = self.store.lib(lid) or dict(id=str(lid), name="Library", main=False, type="spectral", count=0,
                                               spectra=0, description="", source="")
        l = self.lib
        self.title.setText(l["name"])
        self.sub.setText(f"{l['source']} · {l['count']:,} materials · {l['spectra']:,} spectra"
                         + (f" · {l['description']}" if l.get("description") else ""))
        self.b_del.setEnabled(not l["main"])
        self.b_del.setToolTip("Main libraries come from the libraries folder; remove the file there instead."
                              if l["main"] else "Delete this library and all its spectra from the database")
        obj = self._is_obj()
        self.lst.setVisible(not obj)
        self.tbl.setVisible(obj)
        self.note.setVisible(obj)
        self.cb_cat.setVisible(not obj)
        self.cb_acq.setVisible(not obj)
        self.ed_q.blockSignals(True)
        self.ed_q.clear()
        self.ed_q.blockSignals(False)
        self.ed_q.setPlaceholderText("search label or class…" if obj else "search materials…")
        self.lst.clear()
        self.tbl.setRowCount(0)
        if obj:
            self.load_objects()
        else:
            self.cb_cat.blockSignals(True)
            self.cb_cat.clear()
            self.cb_cat.addItem("All categories", "")
            self.cb_cat.blockSignals(False)
            self.cb_acq.blockSignals(True)
            self.cb_acq.setCurrentIndex(0)
            self.cb_acq.blockSignals(False)

            def facets(f):
                self.cb_cat.blockSignals(True)
                for c in f.get("categories", []):
                    p = c["path"]
                    self.cb_cat.addItem(f"{'(no category)' if p == NONE_TOKEN else p} ({c['count']})", p)
                self.cb_cat.blockSignals(False)
            run_bg(self.api.get, "/api/facets", library_id=l["id"], done=facets)
            self.load_materials(then)

    def load_materials(self, then=None):
        if not self.lib or self._is_obj():
            return
        self._seq += 1
        seq = self._seq
        params = {}
        if self.ed_q.text().strip():
            params["search"] = self.ed_q.text().strip()
        if self.cb_cat.currentData():
            params["category_path"] = self.cb_cat.currentData()
        if self.cb_acq.currentData():
            params["acquisition"] = self.cb_acq.currentData()
        self.info.setText("Loading materials…")

        def done(r):
            if seq != self._seq:
                return
            mats = r if isinstance(r, list) else (r.get("items") or r.get("results") or [])
            self.materials = mats
            self.lst.clear()
            for m in mats[:300]:
                it = W.QListWidgetItem(m["name"] + (f"   ·  {m['category']}" if m.get("category") else ""))
                it.setData(Qt.UserRole, m)
                self.lst.addItem(it)
            self.info.setText(f"{len(mats):,} material(s)" + (" — showing 300, type in the search box to narrow down"
                                                              if len(mats) > 300 else "") if mats else
                              "No materials match your search.")
            if then:
                then()

        def fail(m):
            if seq == self._seq:
                note_err(self.info, m)
        run_bg(self.api.get, f"/api/libraries/{self.lib['id']}/materials", **params, done=done, failed=fail)

    def load_objects(self):
        self.info.setText("Loading object samples…")

        def done(r):
            self.objects = r.get("objects", [])
            self._draw_objects()
        run_bg(self.api.get, f"/api/libraries/{self.lib['id']}/objects", done=done, failed=lambda m: note_err(self.info, m))

    def _draw_objects(self):
        q = self.ed_q.text().strip().lower()
        lst = [o for o in self.objects if not q or q in (o.get("name") or "").lower() or q in (o.get("class_name") or "").lower()]
        f5 = lambda v: "—" if v is None else f"{float(v):.5f}"                     # noqa: E731
        self.tbl.setRowCount(len(lst))
        for r, o in enumerate(lst):
            ok = bool(o.get("has_patch")) and o.get("patch_exists") is not False
            patch = (f"📷 {o.get('patch_file') or ''}" + (" · not found" if o.get("patch_exists") is False else "")
                     if o.get("has_patch") else "no patch")
            for c, v in enumerate((o.get("name"), o.get("class_name"), f5(o.get("lat")), f5(o.get("lon")),
                                   o.get("n_pixels") if o.get("n_pixels") is not None else "—", patch)):
                self.tbl.setItem(r, c, cell(v))
            w = W.QWidget()
            h = W.QHBoxLayout(w)
            h.setContentsMargins(2, 0, 2, 0)
            b1 = btn("View image", lambda o=o: PatchDialog(self, self.api, o["sample_id"]).exec_(), kind="primary",
                     accent="library", tip="Look at this image patch (view only)")
            b2 = btn("Locate", lambda o=o: run_bg(self.api.post, f"/api/samples/{o['sample_id']}/patch/reveal", {},
                                                  failed=lambda m: note_err(self.info, m)),
                     tip="Show the file in the file browser")
            b1.setEnabled(ok)
            b2.setEnabled(ok)
            b3 = btn("Spectrum", lambda o=o: self.page.open_material(self.lib["id"], o["material_id"], o.get("name")))
            for b in (b1, b2, b3):
                h.addWidget(b)
            self.tbl.setCellWidget(r, 6, w)
        self.tbl.resizeRowsToContents()
        self.info.setText(f"{len(lst)} object sample(s)" if lst else "No object samples match.")

    def _open_mat(self, it):
        m = it.data(Qt.UserRole)
        self.page.open_material(self.lib["id"], m["id"], m["name"])

    def upload(self):
        if not self.lib:
            return
        ImportDialog(self, self.api, self.store, self.lib["name"], f"Upload files into “{self.lib['name']}”").exec_()
        self.open(self.lib["id"])

    def export(self):
        if not self.lib:
            return
        lib = self.lib
        self.info.setText("Reading the spectra for export…")

        def load():
            r = self.api.get("/api/spectra/search", library_id=lib["id"], limit=5000)
            out = []
            for h in r.get("results", []):
                sp = get_spectrum(self.api, h["spectrum_id"])
                out.append((h, sp))
            return out, r.get("total", 0)

        def done(res):
            rows, total = res
            data = []
            for h, sp in rows:
                for w, v in zip(sp["wavelengths"], sp["values"]):
                    data.append([h["material"], h.get("category") or "", sp.get("measurement_unit") or "", w,
                                 "" if v is None else v])
            p = save_csv(self, f"{lib['name']}.csv".replace(" ", "_"), ["material", "category", "unit", "wavelength_nm",
                                                                           "value"], data)
            self.info.setText((f"Exported {len(rows)} spectra" + (f" (of {total})" if total > len(rows) else "") +
                               f" to {p}") if p else "")
        run_bg(load, done=done, failed=lambda m: note_err(self.info, m))

    def delete(self):
        l = self.lib
        if not l or l["main"]:
            return
        if W.QMessageBox.question(self, "Delete library", f"Delete the library “{l['name']}” and its {l['spectra']} "
                                  "spectra from the database?\nThis cannot be undone.") != W.QMessageBox.Yes:
            return

        def done(_):
            self.store.refresh()
            self.page.go_home()
        run_bg(self.api.delete, f"/api/libraries/{l['id']}", done=done, failed=lambda m: note_err(self.info, m))


# ------------------------------------------------------------------ one material (its samples' spectra)
class MaterialView(W.QWidget):
    def __init__(self, page):
        super().__init__()
        self.page, self.api = page, page.api
        self.lid = self.mid = None
        self.spectra = []
        self.n_samples = 0
        body = W.QWidget()
        lay = W.QVBoxLayout(body)
        lay.setContentsMargins(28, 14, 28, 18)
        lay.setSpacing(10)
        self.title = label("", "pagetitle")
        self.sub = label("", "pagesub", wrap=True)
        lay.addWidget(self.title)
        lay.addWidget(self.sub)
        self.warn = small("")
        self.warn.setStyleSheet("color:#b45309;")
        lay.addWidget(self.warn)
        self.chart = SpectrumChart(300, toolbar=True)
        lay.addWidget(self.chart)
        self.tbl = table(["Sample", "Points", "Range (nm)", "Unit", "Acquisition", ""], 0)
        self.tbl.setMinimumHeight(140)
        lay.addWidget(self.tbl)
        self.src_box, self.src_lay = card()
        lay.addWidget(self.src_box)
        self.more = small("")
        lay.addWidget(self.more)
        lay.addLayout(hrow(btn("+ Add more spectra to compare", self.compare, kind="primary", accent="library"),
                           btn("Save chart", lambda: self.chart.save_png(self, "spectra.png")), None))
        lay.addStretch(1)
        outer = W.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll(body))

    def open(self, lid, mid, name=None):
        self.lid, self.mid = str(lid), str(mid)
        lib = self.page.store.lib(lid)
        self.libname = lib["name"] if lib else ""
        self.title.setText(name or "Material")
        self.sub.setText("Loading the spectrum…")
        self.warn.setVisible(False)
        self.tbl.setRowCount(0)
        self.src_box.setVisible(False)
        self.more.setText("")
        self.chart.message("Loading…")

        def load():
            mat = self.api.get(f"/api/materials/{mid}")
            samples = self.api.get(f"/api/materials/{mid}/samples")
            samples = samples if isinstance(samples, list) else samples.get("items", [])
            out = []
            for s in samples[:12]:
                sps = self.api.get(f"/api/samples/{s['id']}/spectra")
                for sp in (sps if isinstance(sps, list) else sps.get("items", [])):
                    sp = dict(sp, sample=s["name"], sample_meta=s.get("metadata") or {}, sample_id=s["id"])
                    SPEC_CACHE_put(sp)
                    out.append(sp)
            return mat, len(samples), out

        def done(r):
            if self.mid != str(mid):
                return
            self.mat, self.n_samples, self.spectra = r
            self._draw()
        run_bg(load, done=done, failed=lambda m: (self.sub.setText(""), note_err(self.warn, m), self.chart.message(m)))

    def _draw(self):
        m, sps = self.mat, self.spectra
        self.title.setText(m.get("name") or self.title.text())
        units = sorted({s.get("measurement_unit") or "" for s in sps})
        self.sub.setText(f"{m.get('category') or 'no category'} · {self.n_samples} sample{'s' if self.n_samples != 1 else ''}"
                         f" · {len(sps)} spectr{'um' if len(sps) == 1 else 'a'} · {self.libname} · {', '.join(units)}"
                         " · wavelength in nm" + (f"\n{m['description']}" if m.get("description") else ""))
        if len(units) > 1:
            self.warn.setText(f"⚠ These spectra have different units ({', '.join(units)}); the scales are not comparable.")
            self.warn.setStyleSheet("color:#b45309;")
            self.warn.setVisible(True)
        if not sps:
            self.chart.message("This material has no stored spectrum.")
            return
        ser = []
        for i, s in enumerate(sps):
            x, y = xy(s["wavelengths"], s["values"])
            ser.append(dict(name=s["sample"], x=x, y=y, color=DS_COLORS[i % len(DS_COLORS)]))
        self.chart.plot(ser, unit=units[0] if len(units) == 1 else None)
        self.tbl.setRowCount(len(sps))
        for r, s in enumerate(sps):
            w = s["wavelengths"]
            for c, v in enumerate((s["sample"], len(s["values"]), f"{min(w):.0f}–{max(w):.0f}" if w else "",
                                   s.get("measurement_unit") or "", s.get("acquisition_type") or "")):
                self.tbl.setItem(r, c, cell(v))
            self.tbl.setCellWidget(r, 5, btn("Export CSV", lambda s=s, r=r: self.csv(s, r)))
        self.tbl.resizeRowsToContents()
        if self.n_samples > 12:
            self.more.setText(f"showing the first 12 of {self.n_samples} samples")
        self._source()

    def _source(self):
        while self.src_lay.count():
            it = self.src_lay.takeAt(0)
            if it.widget():
                it.widget().deleteLater()
            elif it.layout():
                W.QWidget().setLayout(it.layout())
        rows = []
        for s in self.spectra:
            md = s.get("sample_meta") or {}
            if md.get("object"):
                rows.append(("ROI", md["object"], s["sample_id"]))
            elif md.get("pixel"):
                rows.append(("pixel", md["pixel"], s["sample_id"]))
        if not rows:
            self.src_box.setVisible(False)
            return
        self.src_box.setVisible(True)
        self.src_lay.addWidget(label("<b>Source</b> <span style='color:#6b7280'>· the image is not stored in the database, "
                                     "only the name and address of its file</span>", rich=True, wrap=True))
        f5 = lambda v: "" if v is None else f"{float(v):.5f}"                       # noqa: E731
        for kind, o, sid in rows[:6]:
            if kind == "ROI":
                t = (f"ROI of {o.get('n_pixels') or '?'} pixels" + (f" · class <b>{o['class_name']}</b>" if o.get("class_name") else "")
                     + (f" · {f5(o.get('lat'))}, {f5(o.get('lon'))}" if o.get("lat") is not None else "")
                     + f" · scene {o.get('image_name') or ''}" + (f"<br>Image patch: <b>📷 {o['patch_file']}</b>" if o.get("patch_file") else ""))
                bs = [btn("View image", lambda sid=sid: PatchDialog(self, self.api, sid).exec_(), kind="primary", accent="library"),
                      btn("Locate", lambda sid=sid: run_bg(self.api.post, f"/api/samples/{sid}/patch/reveal", {},
                                                           failed=lambda m: note_err(self.warn, m)))] if o.get("patch_file") else []
            else:
                t = f"pixel ({o.get('row')}, {o.get('col')}) · image <b>{o.get('image_name') or ''}</b>"
                bs = [btn("Locate image file", lambda iid=o["image_id"]: run_bg(self.api.post, f"/api/images/{iid}/reveal", {},
                                                                                 failed=lambda m: note_err(self.warn, m)))] \
                    if o.get("image_id") else []
            lab = small(t, rich=True)
            self.src_lay.addLayout(hrow(lab, *bs, stretch_at=0))

    def csv(self, s, i):
        rows = [[w, "" if v is None else v] for w, v in zip(s["wavelengths"], s["values"])]
        save_csv(self, f"{self.title.text()}_{i + 1}.csv".replace(" ", "_"),
                 ["wavelength_nm", s.get("measurement_unit") or "value"], rows)

    def compare(self):
        hits = [dict(spectrum_id=str(s["id"]), material=f"{self.title.text()}" + (f" · {s['sample']}" if len(self.spectra) > 1 else ""),
                     library=self.libname, unit=s.get("measurement_unit")) for s in self.spectra]
        self.page.open_compare(hits)


def SPEC_CACHE_put(sp):
    from .library_tools import SPEC_CACHE
    if sp.get("id"):
        SPEC_CACHE[str(sp["id"])] = sp


# ------------------------------------------------------------------ compare spectra
class CompareView(W.QWidget):
    def __init__(self, page):
        super().__init__()
        self.page, self.api = page, page.api
        self.data = []
        lay = W.QHBoxLayout(self)
        lay.setContentsMargins(20, 10, 20, 14)
        left = W.QVBoxLayout()
        left.addWidget(label("Compare spectra", "pagetitle"))
        left.addWidget(small("Tick up to 16 spectra from any library; filters by library and acquisition."))
        self.browser = SpectraBrowser(self.api, page.store, max_pick=16)
        left.addWidget(self.browser, 1)
        left.addLayout(hrow(btn("Tick all shown", self.browser.tick_all_shown), btn("Clear", self.browser.clear_picks), None))
        left.addWidget(section("Selected"))
        self.sel = W.QListWidget()
        self.sel.setMaximumHeight(150)
        self.sel.setToolTip("Double-click to remove a spectrum from the comparison")
        self.sel.itemDoubleClicked.connect(self._unpick)
        left.addWidget(self.sel)
        lw = W.QWidget()
        lw.setLayout(left)
        lw.setMinimumWidth(400)
        right = W.QVBoxLayout()
        self.ck_norm = W.QCheckBox("Normalise (divide each spectrum by its maximum)")
        self.ck_norm.toggled.connect(lambda _: self._plot())
        right.addLayout(hrow(self.ck_norm, None, btn("Export CSV", self.export),
                             btn("Save chart", lambda: self.chart.save_png(self, "compare.png"))))
        self.chart = SpectrumChart(360, toolbar=True)
        right.addWidget(self.chart, 2)
        right.addWidget(section("Spectral angle (°) between the selected spectra"))
        self.tbl = W.QTableWidget()
        self.tbl.setEditTriggers(W.QAbstractItemView.NoEditTriggers)
        right.addWidget(self.tbl, 1)
        self.note = small("")
        right.addWidget(self.note)
        rw = W.QWidget()
        rw.setLayout(right)
        sp = W.QSplitter()
        sp.addWidget(lw)
        sp.addWidget(rw)
        sp.setStretchFactor(1, 1)
        lay.addWidget(sp)
        self.browser.picked_changed.connect(self.load)
        self.chart.message("Tick spectra on the left")

    def add(self, hits):
        for h in hits:
            if len(self.browser.picked) >= self.browser.max_pick:
                break
            self.browser.picked[h["spectrum_id"]] = h
        self.browser._draw(None)
        self.load()

    def _unpick(self, it):
        self.browser.picked.pop(it.data(Qt.UserRole), None)
        self.browser._draw(None)
        self.load()

    def load(self):
        hits = list(self.browser.picked.values())
        self.sel.clear()
        for k, h in enumerate(hits):
            it = W.QListWidgetItem(f"■ {h['material']}  ·  {h.get('library', '')}")
            it.setForeground(QtGui.QColor(DS_COLORS[k % len(DS_COLORS)]))
            it.setData(Qt.UserRole, h["spectrum_id"])
            self.sel.addItem(it)
        if not hits:
            self.data = []
            self.tbl.clear()
            self.tbl.setRowCount(0)
            self.tbl.setColumnCount(0)
            return self.chart.message("Tick spectra on the left")

        def load():
            return [(h, get_spectrum(self.api, h["spectrum_id"])) for h in hits]

        def done(rows):
            self.data = rows
            self._plot()
        run_bg(load, done=done, failed=self.chart.message)

    def _plot(self):
        if not self.data:
            return
        ser, units = [], set()
        for k, (h, sp) in enumerate(self.data):
            x, y = xy(sp["wavelengths"], sp["values"])
            if self.ck_norm.isChecked():
                mx = np.nanmax(np.abs(y)) if np.isfinite(y).any() else 0
                if mx:
                    y = y / mx
            units.add(sp.get("measurement_unit") or "")
            ser.append(dict(name=f"{h['material']} ({h.get('library', '')})", x=x, y=y, color=DS_COLORS[k % len(DS_COLORS)]))
        self.chart.plot(ser, unit=None if self.ck_norm.isChecked() else (units.pop() if len(units) == 1 else None),
                        ylabel="Normalised value" if self.ck_norm.isChecked() else None)
        self.note.setText("Different units: scales are not comparable (normalise to compare shapes)." if len(units) > 1 else
                          "The angle uses the wavelengths both spectra have (the second is interpolated onto the first).")
        n = len(self.data)
        names = [h["material"][:28] for h, _ in self.data]
        self.tbl.setRowCount(n)
        self.tbl.setColumnCount(n)
        self.tbl.setHorizontalHeaderLabels([str(i + 1) for i in range(n)])
        self.tbl.setVerticalHeaderLabels([f"{i + 1} {nm}" for i, nm in enumerate(names)])
        for i in range(n):
            for j in range(n):
                a = 0.0 if i == j else self._angle(self.data[i][1], self.data[j][1])
                self.tbl.setItem(i, j, cell("–" if a is None else f"{a:.2f}", align=Qt.AlignCenter))
        self.tbl.resizeColumnsToContents()

    @staticmethod
    def _angle(a, b):
        xa, ya = xy(a["wavelengths"], a["values"])
        xb, yb = xy(b["wavelengths"], b["values"])
        ok = np.isfinite(yb)
        if ok.sum() < 3:
            return None
        order = np.argsort(xb[ok])
        yi = np.interp(xa, xb[ok][order], yb[ok][order], left=np.nan, right=np.nan)
        return angle_deg([None if not np.isfinite(v) else v for v in ya], [None if not np.isfinite(v) else v for v in yi])

    def export(self):
        if not self.data:
            return
        rows = []
        for h, sp in self.data:
            for w, v in zip(sp["wavelengths"], sp["values"]):
                rows.append([h["material"], h.get("library", ""), sp.get("measurement_unit") or "", w, "" if v is None else v])
        save_csv(self, "compare_spectra.csv", ["material", "library", "unit", "wavelength_nm", "value"], rows)


# ------------------------------------------------------------------ the page
class LibraryMainPage(W.QWidget):
    """Library (main): overview / library / material / compare / functions, with a back bar."""

    def __init__(self, shell, api, store):
        super().__init__()
        self.shell, self.api, self.store = shell, api, store
        self.setObjectName("ds3Page")
        Hub.library_page = self
        lay = W.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        self.bar = W.QFrame()
        self.bar.setObjectName("libBar")
        self.bar.setStyleSheet(f"QFrame#libBar {{ background: white; border-bottom: 1px solid {C['border']}; }}")
        bl = W.QHBoxLayout(self.bar)
        bl.setContentsMargins(20, 6, 20, 6)
        self.b_back = btn("← Back", self.back, kind="link")
        self.b_home = btn("Library home", self.go_home, kind="link")
        self.crumb = small("")
        bl.addWidget(self.b_back)
        bl.addWidget(small("  ·  "))
        bl.addWidget(self.b_home)
        bl.addWidget(self.crumb, 1)
        lay.addWidget(self.bar)
        self.stack = W.QStackedWidget()
        lay.addWidget(self.stack, 1)
        self.overview = OverviewView(self)
        self.library = LibraryView(self)
        self.material = MaterialView(self)
        self.compare = CompareView(self)
        self.functions = FunctionsView(self)
        for v in (self.overview, self.library, self.material, self.compare, self.functions):
            self.stack.addWidget(v)
        self.history = []
        self._show(self.overview, "", push=False)
        store.changed.connect(self._store_changed)

    def _store_changed(self):
        for b in (self.compare.browser, self.functions.viewer.browser):
            b.fill_libs(b.cb_lib.currentData() or "")

    def _show(self, view, crumb, push=True):
        cur = self.stack.currentWidget()
        if push and cur is not view:
            self.history.append((cur, self.crumb.text()))
            self.history = self.history[-30:]
        self.stack.setCurrentWidget(view)
        self.crumb.setText(("   " + crumb) if crumb else "")
        self.bar.setVisible(view is not self.overview)

    def back(self):
        if self.history:
            v, c = self.history.pop()
            self.stack.setCurrentWidget(v)
            self.crumb.setText(c)
            self.bar.setVisible(v is not self.overview)
        else:
            self.go_home()

    def go_home(self):
        self.history = []
        self._show(self.overview, "", push=False)

    def open_library(self, lid, then=None):
        self.library.open(lid, then)
        l = self.store.lib(lid)
        self._show(self.library, l["name"] if l else "")

    def open_material(self, lid, mid, name=None):
        self.material.open(lid, mid, name)
        l = self.store.lib(lid)
        self._show(self.material, f"{l['name'] if l else ''}  ›  {name or ''}")

    def open_compare(self, hits=None):
        if hits:
            self.compare.add(hits)
        self._show(self.compare, "Compare")

    def open_functions(self, tab="builder", lib_id=None):
        self.functions.show_tab(tab)
        if tab == "resample" and lib_id:
            self.functions.resample.fill(lib_id)
        self._show(self.functions, "Library functions")

    def import_files(self):
        ImportDialog(self, self.api, self.store).exec_()

    def refresh(self):
        self.store.refresh()
