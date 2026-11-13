"""Home: welcome text, four counters, the three module cards (as in the DS3 prototype)."""
from __future__ import annotations

from PyQt5 import QtWidgets

from .widgets import CardGrid, ModuleCard, StatCard, label, page_header, scroll_page

W = QtWidgets


try:                                           # the DS3 web modules (Library, Image)
    from . import web_pages as _wp  # noqa: F401
    _WEB_HOME = True
except Exception:
    _WEB_HOME = False

class HomePage(W.QWidget):
    def __init__(self, shell):
        super().__init__()
        self.shell = shell
        self.setObjectName("ds3Page")
        body = W.QWidget()
        lay = W.QVBoxLayout(body)
        lay.setContentsMargins(28, 22, 28, 22)
        lay.setSpacing(18)
        lay.addLayout(page_header(
            "Welcome to DS3",
            "A workspace for managing spectral libraries, exploring and processing hyperspectral images, and "
            "aggregating material signatures across scenes. Your libraries are listed under Library; combine "
            "scenes under Data Aggregation; open and analyse an image under Viewer and Analysis."))
        self.st_libs, self.st_mats, self.st_imgs, self.st_specs = (StatCard(t) for t in
                                                                    ("Libraries", "Materials", "Images opened", "Spectra"))
        stats = CardGrid(min_width=200, max_cols=4, spacing=14)
        stats.set_cards([self.st_libs, self.st_mats, self.st_imgs, self.st_specs])
        lay.addWidget(stats)
        mods = CardGrid(min_width=300, max_cols=3, spacing=14)
        mods.set_cards([
            ModuleCard("library", "Library (main)" if _WEB_HOME else "Library", "Browse and search your spectral libraries; import any spectra format; "
                       "build, view, resample, compare and export them.", "Open Library",
                       lambda: shell.show_module("weblib" if _WEB_HOME else "library")),
            ModuleCard("analysis", "Data Aggregation", "Aggregate scenes into class signatures on a common grid, "
                       "mosaic and stack; export CSV, JSON or ENVI .sli.", "Open Data Aggregation",
                       lambda: shell.show_module("analysis")),
            ModuleCard("image", "Viewer and Analysis", "Open a hyperspectral image, preprocess, reduce (PCA / MNF), "
                       "classify, segment, unmix and compare with a library.", "Open Viewer and Analysis", lambda: shell.show_module("image")),
        ] + ([ModuleCard("library", "Library 2", "The Qt spectral library: import, filter by purity and category, rename "
                         "and move libraries, resample and export.", "Open Library 2", lambda: shell.show_module("library"))]
              if _WEB_HOME else []) + ([ModuleCard("image", "Image", "Open a hyperspectral image, collect pixel spectra, draw ROIs, build pixel and "
                             "object libraries, georeference and aggregate.", "Open Image", lambda: shell.show_module("webimage"))]
              if _WEB_HOME else []))
        lay.addWidget(mods)
        self.lbl_hint = label("", "small", wrap=True)
        lay.addWidget(self.lbl_hint)
        lay.addStretch(1)
        outer = W.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll_page(body))

    def set_counts(self, libs=None, mats=None, imgs=None, specs=None, hint=""):
        self.st_libs.set(libs)
        self.st_mats.set(mats)
        self.st_imgs.set(imgs)
        self.st_specs.set(specs)
        self.lbl_hint.setText(hint)
