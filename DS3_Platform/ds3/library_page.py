"""Library module: library cards (main libraries from the libraries folder + built libraries), search, folder
reload - the DS3 overview - and, on a card click, the full library panel (category tree, materials, spectrum,
compare, match, filters, import, Builder, Resample, Export) with its own spectra plot."""
from __future__ import annotations

import os

from PyQt5 import QtCore, QtWidgets

from . import style
from .style import C
from .widgets import Card, CardGrid, Pill, label, page_header, primary, scroll_page

W = QtWidgets
Qt = QtCore.Qt


class LibraryCard(Card):
    def __init__(self, lib, main: bool, idx: int):
        super().__init__(clickable=True)
        self.lib = lib
        self.lay.setSpacing(4)
        self.lay.addWidget(Pill("Main library" if main else "Built library", idx if main else 5))
        self.lay.addWidget(label(lib["name"], "cardtitle", wrap=True))
        m = lib.get("material_count", 0)
        s = lib.get("spectrum_count")
        meta = f"{m:,} material{'s' if m != 1 else ''}" + (f" · {s:,} spectra" if s is not None else "")
        if lib.get("unit"):
            meta += f" · {lib['unit']}"
        self.lay.addWidget(label(meta, "small"))
        if lib.get("description"):
            d = label(str(lib["description"])[:120], "small", wrap=True)
            self.lay.addWidget(d)
        self.setToolTip(f"{lib['name']}\nClick to open: category tree, materials, spectra, compare, match")
        self.setMinimumHeight(96)


class LibraryPage(W.QWidget):
    def __init__(self, shell, libpanel):
        super().__init__()
        self.shell = shell
        self.lp = libpanel
        self.setObjectName("ds3Page")
        self.main_names = set()
        self.stack = W.QStackedWidget()
        outer = W.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(self.stack)

        # ---------------- overview (cards)
        body = W.QWidget()
        lay = W.QVBoxLayout(body)
        lay.setContentsMargins(28, 22, 28, 22)
        lay.setSpacing(14)
        top = W.QHBoxLayout()
        top.addLayout(page_header("Library", "Manage and explore spectral libraries."), 1)
        for t, fn in (("Connect…", lambda: self.lp.on_connect_dialog()),
                      ("+ New library", lambda: self.lp.on_add_library()),
                      ("Import spectra…", lambda: self.lp.on_import_files("files")),
                      ("Builder…", lambda: self.lp.on_builder())):
            b = W.QPushButton(t)
            b.clicked.connect(fn)
            top.addWidget(b, 0, Qt.AlignTop)
        lay.addLayout(top)
        srch = W.QHBoxLayout()
        self.ed_libs = W.QLineEdit()
        self.ed_libs.setPlaceholderText("e.g. ECOSTRESS")
        style.prop(self.ed_libs, ds3="search")
        self.ed_libs.textChanged.connect(lambda *_: self.fill_cards())
        self.ed_mats = W.QLineEdit()
        self.ed_mats.setPlaceholderText("e.g. basalt, concrete, grass  (Enter)")
        style.prop(self.ed_mats, ds3="search")
        self.ed_mats.returnPressed.connect(self.on_material_search)
        for cap, w in (("Search libraries by name", self.ed_libs), ("Search materials by name or location", self.ed_mats)):
            col = W.QVBoxLayout()
            col.setSpacing(3)
            col.addWidget(label(cap, "small"))
            w.setMinimumWidth(280)
            col.addWidget(w)
            srch.addLayout(col)
        srch.addStretch(1)
        lay.addLayout(srch)
        # folder banner
        self.banner = W.QFrame()
        style.prop(self.banner, ds3="banner")
        bl = W.QHBoxLayout(self.banner)
        bl.setContentsMargins(16, 10, 12, 10)
        bcol = W.QVBoxLayout()
        bcol.setSpacing(2)
        self.lbl_main = label("", None)
        self.lbl_folder = label("", "small")
        self.lbl_folder.setTextInteractionFlags(Qt.TextSelectableByMouse)
        bcol.addWidget(self.lbl_main)
        bcol.addWidget(self.lbl_folder)
        bl.addLayout(bcol, 1)
        b = W.QPushButton("Change folder…")
        b.clicked.connect(self.on_change_folder)
        bl.addWidget(b)
        self.b_reload = W.QPushButton("Reload main libraries")
        self.b_reload.setToolTip("Import every new file of the libraries folder (nothing is imported twice)")
        self.b_reload.clicked.connect(self.reload_main)
        bl.addWidget(self.b_reload)
        lay.addWidget(self.banner)
        self.lbl_err = label("", "error", wrap=True)
        lay.addWidget(self.lbl_err)
        # material search results (search over every library, like the DS3 prototype)
        self.res_box = W.QFrame()
        style.prop(self.res_box, ds3="card")
        rl = W.QVBoxLayout(self.res_box)
        rl.setContentsMargins(14, 10, 14, 10)
        rh = W.QHBoxLayout()
        self.lbl_res = label("", None, rich=True)
        rh.addWidget(self.lbl_res, 1)
        b = W.QPushButton("Close")
        style.prop(b, ds3="link")
        b.clicked.connect(lambda: (self.res_box.setVisible(False), self.ed_mats.clear()))
        rh.addWidget(b)
        rl.addLayout(rh)
        self.lst_res = W.QListWidget()
        self.lst_res.setMinimumHeight(140)
        self.lst_res.setMaximumHeight(260)
        self.lst_res.itemActivated.connect(self.on_result_open)
        self.lst_res.itemClicked.connect(self.on_result_open)
        rl.addWidget(self.lst_res)
        rl.addWidget(label("Click a material to open it in its library.", "small"))
        self.res_box.setVisible(False)
        lay.addWidget(self.res_box)
        self.lbl_sec_main = label("", None, rich=True)
        lay.addWidget(self.lbl_sec_main)
        self.grid_main = CardGrid(min_width=290, max_cols=3)
        lay.addWidget(self.grid_main)
        self.lbl_sec_built = label("", None, rich=True)
        lay.addWidget(self.lbl_sec_built)
        self.grid_built = CardGrid(min_width=290, max_cols=3)
        lay.addWidget(self.grid_built)
        lay.addStretch(1)
        self.stack.addWidget(scroll_page(body))

        # ---------------- detail (the library panel)
        det = W.QWidget()
        det.setObjectName("ds3Page")
        dl = W.QVBoxLayout(det)
        dl.setContentsMargins(18, 10, 18, 10)
        dl.setSpacing(6)
        crumb = W.QPushButton("‹  All libraries")
        style.prop(crumb, ds3="link")
        crumb.setCursor(Qt.PointingHandCursor)
        crumb.clicked.connect(self.show_cards)
        hdr = W.QHBoxLayout()
        hdr.addWidget(crumb)
        self.lbl_title = label("", "pagetitle")
        hdr.addSpacing(12)
        hdr.addWidget(self.lbl_title)
        self.lbl_meta = label("", "pagesub")
        hdr.addSpacing(10)
        hdr.addWidget(self.lbl_meta, 1)
        dl.addLayout(hdr)
        from hvva.gui.canvases import PlotCanvas
        self.plot = PlotCanvas()
        self.plot.setMinimumHeight(170)
        split = W.QSplitter(Qt.Vertical)
        split.addWidget(self.lp)                          # reparented from the Analysis window
        pw = W.QWidget()
        pl = W.QVBoxLayout(pw)
        pl.setContentsMargins(0, 0, 0, 0)
        pl.addWidget(label("Spectra", "section"))
        pl.addWidget(self.plot, 1)
        split.addWidget(pw)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)
        split.setSizes([520, 300])
        dl.addWidget(split, 1)
        self.stack.addWidget(det)
        self.plot.message("Select a material: its spectrum appears here")

    # ------------------------------------------------------------ folder of main libraries
    def folder(self):
        f = self.lp.cfg.get("data_folder") or ""
        return f

    def on_change_folder(self):
        d = W.QFileDialog.getExistingDirectory(self, "Folder of main libraries", self.folder() or os.getcwd())
        if d:
            self.lp.cfg["data_folder"] = d
            self.lp._save_folder_cfg()
            self.scan()

    def scan(self):
        """Which libraries come from the libraries folder (main) - the rest are 'built'."""
        f = self.folder()
        self.lbl_folder.setText(f"Folder: {f}" if f else "No libraries folder set - 'Change folder…'")
        if not f or not os.path.isdir(f) or self.lp.client is None or self.lp.client.mode != "direct":
            self.main_names = set()
            self.fill_cards()
            return

        def ok(srcs):
            names = set()
            new = 0
            for s in srcs:
                names.add(s["library"])
                for w in s.get("imported_into") or []:
                    names.add(w.split(" / ")[0])
                new += s["status"] == "new"
            self.main_names = names
            self._new = new
            self.fill_cards()
            if new and not getattr(self, "_auto_loaded", False):     # like DS3: main libraries load at start
                self._auto_loaded = True
                self.reload_main()
        self.lp.run(self.lp.client.scan_folder, f, done=ok,
                    failed=lambda m: self.lbl_folder.setText(f"Folder: {f}   (scan failed: {m})"))

    def reload_main(self):
        f = self.folder()
        if not f or not os.path.isdir(f):
            return self.on_change_folder()
        if not self.lp._need_client():
            return
        self.b_reload.setEnabled(False)
        self.b_reload.setText("Loading…")

        def ok(res):
            self.b_reload.setEnabled(True)
            self.b_reload.setText("Reload main libraries")
            n_ok = sum(1 for r in res if r.get("ok"))
            bad = [r for r in res if not r.get("ok")]
            self.shell.flash(f"Main libraries: {n_ok} new file(s) imported" +
                             (f", {len(bad)} failed ({bad[0]['source']}: {bad[0].get('error', '')})" if bad else
                              "" if n_ok else " - everything was already loaded"))
            self.lp.load_libraries()
            self.scan()

        def bad(m):
            self.b_reload.setEnabled(True)
            self.b_reload.setText("Reload main libraries")
            self.shell.flash(f"⚠ Reload failed: {m}")
        self.lp.run(self.lp.client.import_new_from_folder, f, done=ok, failed=bad)

    # ------------------------------------------------------------ cards
    def fill_cards(self):
        libs = list(self.lp.libraries or [])
        connected = self.lp.client is not None
        self.lbl_err.setText("" if connected else
                             "The library database is not connected. Use 'Connect…' (PostgreSQL: start Docker and "
                             "the material_library_db container first; or choose a local SQLite file).")
        q = self.ed_libs.text().strip().lower()
        if q:
            libs = [l for l in libs if q in l["name"].lower()]
        main = [l for l in libs if l["name"] in self.main_names]
        built = [l for l in libs if l["name"] not in self.main_names]
        n_main_all = sum(1 for l in self.lp.libraries if l["name"] in self.main_names)
        new = getattr(self, "_new", 0)
        self.lbl_main.setText(f"{n_main_all} main librar{'y' if n_main_all == 1 else 'ies'} loaded." +
                              (f"  {new} new file(s) in the folder - 'Reload main libraries'." if new else ""))
        self.lbl_sec_main.setText(f"<b style='color:{C['title']}'>Main libraries</b>"
                                  f"<span style='color:{C['muted']}'>&nbsp;&nbsp;{len(main)} · loaded from the "
                                  "libraries folder</span>")
        self.lbl_sec_built.setText(f"<b style='color:{C['title']}'>Built libraries</b>"
                                   f"<span style='color:{C['muted']}'>&nbsp;&nbsp;{len(built)} · made with the "
                                   "Spectral Library Builder, from images or imported</span>")
        cards = []
        for i, l in enumerate(main):
            c = LibraryCard(l, True, i)
            c.clicked.connect(lambda l=l: self.open(l))
            cards.append(c)
        self.grid_main.set_cards(cards)
        cards = []
        for l in built:
            c = LibraryCard(l, False, 0)
            c.clicked.connect(lambda l=l: self.open(l))
            cards.append(c)
        self.grid_built.set_cards(cards)

    def open(self, lib, then=None):
        self.lp._select_library_row(lib["id"])              # sync: no reload that would reopen it
        self.lp.open_library(lib, then=then)
        self.lbl_title.setText(lib["name"])
        m = lib.get("material_count", 0)
        self.lbl_meta.setText(f"{m:,} materials" + (f" · {lib.get('spectrum_count', 0):,} spectra"
                                                    if lib.get("spectrum_count") is not None else "") +
                              (f" · unit {lib['unit']}" if lib.get("unit") else ""))
        self.stack.setCurrentIndex(1)
        self.shell.show_module("library", keep_view=True)

    def show_detail(self):
        self.stack.setCurrentIndex(1)

    def show_cards(self):
        self.stack.setCurrentIndex(0)
        self.fill_cards()

    def on_material_search(self):
        q = self.ed_mats.text().strip()
        if len(q) < 2:
            self.res_box.setVisible(False)
            return
        if not self.lp._need_client():
            return
        f = self.lp.filters()

        def ok(payload):
            if self.ed_mats.text().strip() != q:
                return
            items = payload.get("items", []) if isinstance(payload, dict) else payload
            self.lst_res.clear()
            for m in items:
                cat = m.get("category")
                cat = " / ".join(cat) if isinstance(cat, (list, tuple)) else (cat or "")
                it = W.QListWidgetItem(f"{m['name']}   —   {m['library_name']}" + (f"   ·  {cat}" if cat else ""))
                it.setData(Qt.UserRole, (m["library_id"], m["id"]))
                self.lst_res.addItem(it)
            n = len(items)
            self.lbl_res.setText(f"<b>Materials matching “{q}”</b>  <span style='color:#6b7684'>"
                                 f"{n}{'+' if n >= 100 else ''} found in all libraries</span>" if n else
                                 f"<b>No materials match “{q}”</b>")
            self.res_box.setVisible(True)
        self.lp.run(self.lp.client.search_materials, q, 100, self.lp.acq_filter, f["unit"], f["purity"],
                    f["covers"], done=ok, failed=lambda m: self.shell.flash(f"⚠ Search failed: {m}"))

    def on_result_open(self, item):
        ids = item.data(Qt.UserRole)
        if not ids:
            return
        lid, mid = ids
        lib = next((l for l in self.lp.libraries or [] if l["id"] == lid), None)
        if lib is None:
            return
        self.open(lib, then=lambda: self.lp.open_material(mid))
