"""One-time patch: puts the DS3 web modules (Library, Image) into the DS3 window.

    python tools/add_web_modules.py          # patches ds3/shell.py, ds3/app.py, ds3/home.py (copies kept as *.bak)
    python tools/add_web_modules.py --undo   # restores the copies

What it does
  * sidebar:  Library (DS3 Library, web) · Image · Data Aggregation · Viewer and Analysis
  * the original Qt Library page stays available as  View ▸ Library (classic)  (other modules call into it)
  * Home page: Library card opens the new Library, a new Image card is added
Safe to run again (does nothing when already patched), so it can be re-run after `git pull` brings new files."""
import os
import re
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DS3 = os.path.join(ROOT, "ds3")
MARK = "# --- DS3 web modules ---"


def _read(p):
    return open(p, encoding="utf-8").read()


def patch_shell(s):
    if MARK in s:
        return s
    def once(old, new):
        nonlocal s
        if s.count(old) != 1:
            raise SystemExit("ds3/shell.py differs from the expected version; this part was not found exactly once:\n" + old)
        s = s.replace(old, new)

    once("from .style import C\n",
         "from .style import C\n"
         f"{MARK}\n"
         "try:\n"
         "    from .web_pages import WebBackend, WebPage\n"
         "    HAVE_WEB = True\n"
         "except Exception as _e:                      # PyQtWebEngine missing: the DS3 window works without the web modules\n"
         "    print('DS3 web modules not available:', _e)\n"
         "    HAVE_WEB = False\n")
    once("        self.image = ImagePage(self)\n",
         "        self.image = ImagePage(self)\n"
         "        self.webimage = WebPage(self, 'image', 'Image') if HAVE_WEB else None\n"
         "        self.weblib = WebPage(self, 'library', 'Library') if HAVE_WEB else None\n")
    once("        for p in (self.home, self.library, self.image, self.analysis):\n            self.stack.addWidget(p)\n",
         "        for p in (self.home, self.library, self.image, self.analysis, self.webimage, self.weblib):\n"
         "            if p is not None:\n                self.stack.addWidget(p)\n")
    pat = re.compile(r'        for key, text in \(\("library", "Library"\), \("analysis", "Data Aggregation"\), \("image", "([^"]+)"\)\):\n')
    m = pat.search(s)
    if not m or len(pat.findall(s)) != 1:
        raise SystemExit("the sidebar loop of ds3/shell.py was not found exactly once")
    loop = ('        _base = {"weblib": "library", "webimage": "image"}\n'
            '        if HAVE_WEB:\n'
            '            _mods = [("weblib", "Library (main)"), ("library", "Library 2"), ("webimage", "Image"), ("analysis", "Data Aggregation"), ("image", "%s")]\n'
            '        else:\n'
            '            _mods = [("library", "Library"), ("analysis", "Data Aggregation"), ("image", "%s")]\n'
            '        for key, text in _mods:\n') % (m.group(1), m.group(1))
    s = s[:m.start()] + loop + s[m.end():]
    once("            col, fill, border, _ = style.MODULE[key]\n            pm = self._tile(key, col, fill, border)\n",
         "            col, fill, border, _ = style.MODULE[_base.get(key, key)]\n            pm = self._tile(_base.get(key, key), col, fill, border)\n")
    once('        return {self.home: "home", self.library: "library", self.image: "image", self.analysis: "analysis"}.get(w)\n',
         '        return {self.home: "home", self.library: "library", self.image: "image", self.analysis: "analysis",\n'
         '                self.webimage: "webimage", self.weblib: "weblib"}.get(w)\n')
    once('        if key == "image":\n            self.stack.setCurrentWidget(self.image)\n',
         '        if key in ("webimage", "weblib"):\n'
         '            page = self.webimage if key == "webimage" else self.weblib\n'
         '            self.stack.setCurrentWidget(page)\n'
         '            QtCore.QTimer.singleShot(0, page.ensure)\n'
         '            QtCore.QTimer.singleShot(50, page.refresh_data)         # libraries saved elsewhere show up at once\n'
         '        elif key == "image":\n            self.stack.setCurrentWidget(self.image)\n')
    once('        self._act(m, "Library", lambda: self.show_module("library"), "Ctrl+1")\n',
         '        self._act(m, "Library (main)" if HAVE_WEB else "Library", lambda: self.show_module("weblib" if HAVE_WEB else "library"), "Ctrl+1")\n'
         '        if HAVE_WEB:\n'
         '            self._act(m, "Image", lambda: self.show_module("webimage"), "Ctrl+4")\n'
         '            self._act(m, "Library 2", lambda: self.show_module("library"), "Ctrl+5")\n')
    once('            if key == "library" and not keep_view:\n                self.library.show_cards()\n',
         '            if key == "library" and not keep_view:\n                self.library.show_cards()\n'
         '            if key in ("library", "analysis", "home") and self.libpanel.client is not None and HAVE_WEB:\n'
         '                QtCore.QTimer.singleShot(50, self.libpanel.load_libraries)   # libraries saved by Library (main) / Image\n')
    once("    def closeEvent(self, ev):\n        try:\n            self.libpanel.shutdown()\n",
         "    def closeEvent(self, ev):\n        try:\n            if HAVE_WEB:\n                WebBackend.get().stop()\n"
         "        except Exception:                                                        # noqa: BLE001\n            pass\n"
         "        try:\n            self.libpanel.shutdown()\n")
    return s


def patch_app(a):
    if "QtWebEngineWidgets" in a:
        return a
    old = "    from PyQt5 import QtCore, QtWidgets\n"
    if a.count(old) != 1:
        raise SystemExit("ds3/app.py differs from the expected version")
    return a.replace(old, "    try:\n        from PyQt5 import QtWebEngineWidgets  # noqa: F401  (Qt rule: before the QApplication is created)\n"
                          "    except Exception:                                    # DS3 web modules are optional\n        pass\n" + old)


def patch_home(h):
    if "_WEB_HOME" in h:
        return h
    a = '                       lambda: shell.show_module("library")),\n'
    b = '"Open Viewer and Analysis", lambda: shell.show_module("image")),\n        ])\n'
    if h.count(a) != 1 or h.count(b) != 1:
        raise SystemExit("ds3/home.py differs from the expected version")
    h = h.replace(a, '                       lambda: shell.show_module("weblib" if _WEB_HOME else "library")),\n')
    h = h.replace('ModuleCard("library", "Library", "Browse', 'ModuleCard("library", "Library (main)" if _WEB_HOME else "Library", "Browse')
    h = h.replace(b, '"Open Viewer and Analysis", lambda: shell.show_module("image")),\n        ] + ([ModuleCard("library", "Library 2", "The Qt spectral library: import, filter by purity and category, rename "\n'
                     '                         "and move libraries, resample and export.", "Open Library 2", lambda: shell.show_module("library"))]\n'
                     '              if _WEB_HOME else []) + ([ModuleCard("image", "Image", "Open a hyperspectral image, collect pixel spectra, draw ROIs, build pixel and "\n'
                     '                             "object libraries, georeference and aggregate.", "Open Image", lambda: shell.show_module("webimage"))]\n'
                     '              if _WEB_HOME else []))\n')
    i = h.index("\nclass ")
    return h[:i] + "\ntry:                                           # the DS3 web modules (Library, Image)\n    from . import web_pages as _wp  # noqa: F401\n    _WEB_HOME = True\nexcept Exception:\n    _WEB_HOME = False\n" + h[i:]


FILES = (("shell.py", patch_shell), ("app.py", patch_app), ("home.py", patch_home))


def main():
    if "--undo" in sys.argv:
        for name, _ in FILES:
            p = os.path.join(DS3, name)
            if os.path.exists(p + ".bak"):
                shutil.copyfile(p + ".bak", p)
                print("restored", p)
        return 0
    new = {}
    for name, fn in FILES:
        p = os.path.join(DS3, name)
        old = _read(p)
        new[name] = (p, old, fn(old))
    changed = [n for n, (p, o, s) in new.items() if o != s]
    if not changed:
        print("already patched")
        return 0
    for n in changed:
        p, o, s = new[n]
        shutil.copyfile(p, p + ".bak")
        open(p, "w", encoding="utf-8").write(s)
        print("patched", p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
