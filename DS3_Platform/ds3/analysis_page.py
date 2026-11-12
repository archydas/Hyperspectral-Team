"""Data Aggregation module: the Library and Data Aggregation tool (hvva) - scene view with band selection, map and
collected spectra; tabs Scenes, ROI Analysis, Wavelength Grid, Aggregation, Mosaic / Stack, Signature Export,
Plugins (the Georeferencing tab is left out in DS3). Its Library tab moves to the Library module (same panel, same connection)."""
from __future__ import annotations

from PyQt5 import QtCore, QtWidgets

from .widgets import label

W = QtWidgets
Qt = QtCore.Qt


class AnalysisPage(W.QWidget):
    def __init__(self, shell):
        super().__init__()
        self.shell = shell
        self.setObjectName("ds3Page")
        from hvva.gui import main_window as mw
        from hvva.gui import theme as hv_theme
        from . import style
        style.register_theme(hv_theme)
        hv_theme.CURRENT_THEME = "ds3"
        self.win = win = mw.LibraryBuilderWindow()
        win.setWindowFlags(Qt.Widget)
        win.menuBar().setVisible(False)                 # File / View / Tools / Help live in the DS3 menus
        win.statusBar().setVisible(False)
        win.statusBar().messageChanged.connect(lambda m: m and shell.statusBar().showMessage(m, 6000))
        win.theme = "ds3"
        win._apply_window_qss("ds3")
        idx = win.tabs.indexOf(win.libpanel)              # the Library tab becomes the Library module
        if idx >= 0:
            win.tabs.removeTab(idx)
        for i in reversed(range(win.tabs.count())):       # DS3: no Georeferencing tab in Data Aggregation
            if win.tabs.tabText(i) == "Georeferencing":
                win.tabs.removeTab(i)
        orig_say = win.say

        def say(text, _orig=orig_say):
            _orig(text)
            shell.flash(text)
        win.say = say
        lay = W.QVBoxLayout(self)
        lay.setContentsMargins(10, 8, 10, 6)
        hdr = W.QHBoxLayout()
        hdr.addWidget(label("Data Aggregation", "pagetitle"))
        hdr.addSpacing(10)
        hdr.addWidget(label("Aggregate scenes into class signatures on a common grid; ROIs across images, mosaic "
                            "and stack, map view; export CSV, JSON or ENVI .sli.", "pagesub"), 1)
        lay.addLayout(hdr)
        lay.addWidget(win, 1)

    def goto_tab(self, title):
        t = self.win.tabs
        for i in range(t.count()):
            if t.tabText(i).replace("&&", "&").lower().startswith(title.lower()):
                t.setCurrentIndex(i)
                break
        self.shell.show_module("analysis")
