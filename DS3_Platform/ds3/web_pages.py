"""DS3 web modules inside the Qt window.

The Image module (and the web Library) of the DS3 Library app (FastAPI + HTML/JS, folder ``material_library/``) are
shown here as pages of the PyQt5 DS3 window: one separate process runs the API server on 127.0.0.1, and each page is
a QWebEngineView on ``/app/?embed=image`` or ``/app/?embed=library``.

* Database: the connection of the Qt Library panel is reused (``libpanel.cfg['db_url']``), so the Qt modules and the
  web modules use the same PostgreSQL / SQLite database.  Without a connection the web modules use the SQLite file in
  ``material_library/data``.  Override with the environment variable DS3_WEB_DB.
* File dialogs: the web page asks for ``window.pywebview.api.pick_files``; a small QWebChannel shim answers it with the
  Qt file dialog, so image files are opened in place (no upload copy).
"""
from __future__ import annotations

import os
import atexit
import socket
import subprocess
import sys
import time
import urllib.request

from PyQt5 import QtCore, QtWebChannel, QtWebEngineWidgets, QtWidgets

W = QtWidgets
Qt = QtCore.Qt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                                  # DS3_Platform/
WEB_BACKEND = os.path.join(ROOT, "material_library", "backend")


def _free_port(preferred=8765):
    for port in [preferred] + list(range(preferred + 1, preferred + 60)):
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise RuntimeError("no free port for the DS3 web modules")


class WebBackend:
    """One API server per program (started on first use)."""
    _inst = None

    def __init__(self):
        self.base = None
        self.error = None
        self.proc = None
        self.db_shown = None

    @classmethod
    def get(cls):
        if cls._inst is None:
            cls._inst = WebBackend()
        return cls._inst

    def start(self, db_url=None):
        """Start the API server (first time only) as a SEPARATE PROCESS: the image readers of the web modules (rasterio /
        GDAL, HDF5, large cubes in memory) then cannot disturb the Qt modules - no clashing DLLs, no shared memory."""
        if self.base or self.error:
            return self.base
        try:
            url = os.environ.get("DS3_WEB_DB") or db_url
            env = dict(os.environ)
            if url:
                env["DATABASE_URL"] = url
            env["DS3_LOCAL_APP"] = "1"                         # open image files in place
            env.setdefault("AUTO_IMPORT", "0")                 # the Qt Library already manages data/libraries
            env["PYTHONPATH"] = os.pathsep.join([ROOT, WEB_BACKEND] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
            env["PYTHONUNBUFFERED"] = "1"
            self.db_shown = _hide_password(url) if url else "SQLite file of the web modules (material_library/data)"
            port = _free_port()
            cmd = [sys.executable, "-m", "uvicorn", "app.main:app", "--app-dir", WEB_BACKEND,
                   "--host", "127.0.0.1", "--port", str(port), "--log-level", "warning"]
            self.proc = subprocess.Popen(cmd, cwd=ROOT, env=env)
            atexit.register(self.stop)
            base = f"http://127.0.0.1:{port}"
            end = time.time() + 90
            while time.time() < end:
                if self.proc.poll() is not None:
                    self.error = (f"The DS3 web server stopped at start-up (exit code {self.proc.returncode}). "
                                  "The reason is printed in the terminal window.")
                    return None
                try:
                    urllib.request.urlopen(base + "/health", timeout=1).read()
                    self.base = base
                    return base
                except Exception:
                    time.sleep(0.3)
            self.error = "The DS3 web server did not start in 90 s."
            self.stop()
        except Exception as e:                                  # noqa: BLE001 - shown on the page
            self.error = f"{type(e).__name__}: {e}"
        return None

    def stop(self):
        p = getattr(self, "proc", None)
        if p is not None and p.poll() is None:
            try:
                p.terminate()
                p.wait(timeout=5)
            except Exception:                                   # noqa: BLE001
                try:
                    p.kill()
                except Exception:                               # noqa: BLE001
                    pass


def _hide_password(u):
    if "@" in u and "://" in u:
        scheme, rest = u.split("://", 1)
        cred, host = rest.rsplit("@", 1)
        return f"{scheme}://{cred.split(':', 1)[0]}:****@{host}"
    return u


class _Bridge(QtCore.QObject):
    """Qt side of window.pywebview.api (file / folder pick)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.last_dir = ""

    @QtCore.pyqtSlot(bool, result="QVariantList")
    def pick(self, folder):
        parent = QtWidgets.QApplication.activeWindow()
        if folder:
            d = W.QFileDialog.getExistingDirectory(parent, "Choose a folder", self.last_dir)
            res = [d] if d else []
        else:
            res, _ = W.QFileDialog.getOpenFileNames(parent, "Open image / spectra file(s)", self.last_dir,
                                                    "All files (*.*)")
        if res:
            self.last_dir = os.path.dirname(res[0]) if not folder else res[0]
        return list(res)


_SHIM = r"""
(function(){
  function init(){
    new QWebChannel(qt.webChannelTransport, function(ch){
      var b = ch.objects.ds3bridge;
      window.pywebview = { api: { pick_files: function(folder){
        return new Promise(function(res){ b.pick(!!folder, function(r){ res(r || []); }); });
      } } };
    });
  }
  if (typeof QWebChannel !== 'undefined') init();
})();
"""


_PROFILE = None


class _LogPage(QtWebEngineWidgets.QWebEnginePage):
    """A web page that prints its JavaScript warnings / errors in the terminal (lines starting 'js:')."""

    def javaScriptConsoleMessage(self, level, msg, line, src):
        if level >= 1:                                        # 1 = warning, 2 = error
            print(f"js: {msg}  [{str(src).rsplit('/', 1)[-1]}:{line}]", flush=True)


def _profile():
    """One off-the-record profile for all web pages: nothing is cached on disk (no 'Unable to move the cache' errors,
    no stale css / js)."""
    global _PROFILE
    if _PROFILE is None:
        _PROFILE = QtWebEngineWidgets.QWebEngineProfile(QtWidgets.QApplication.instance())
    return _PROFILE


class WebPage(W.QWidget):
    """A DS3 page that shows one module of the web app (``embed`` = 'image' or 'library')."""

    def __init__(self, shell, embed, title):
        super().__init__()
        self.shell = shell
        self.embed = embed
        self.setObjectName("ds3Page")
        self.view = QtWebEngineWidgets.QWebEngineView(self)
        self.view.setPage(_LogPage(_profile(), self.view))
        self.info = W.QLabel("")
        self.info.setWordWrap(True)
        self.info.setAlignment(Qt.AlignCenter)
        lay = W.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.info)
        lay.addWidget(self.view, 1)
        self.info.setVisible(False)
        self._loaded = False
        self._channel = QtWebChannel.QWebChannel(self.view.page())
        self._bridge = _Bridge(self)
        self._channel.registerObject("ds3bridge", self._bridge)
        self.view.page().setWebChannel(self._channel)
        self._install_shim()
        self.view.loadFinished.connect(self._after_load)

    def _install_shim(self):
        f = QtCore.QFile(":/qtwebchannel/qwebchannel.js")
        js = ""
        if f.open(QtCore.QIODevice.ReadOnly):
            js = bytes(f.readAll()).decode("utf-8")
            f.close()
        sc = QtWebEngineWidgets.QWebEngineScript()
        sc.setName("ds3-pywebview-shim")
        sc.setInjectionPoint(QtWebEngineWidgets.QWebEngineScript.DocumentReady)
        sc.setWorldId(QtWebEngineWidgets.QWebEngineScript.MainWorld)
        sc.setSourceCode(js + "\n" + _SHIM)
        self.view.page().scripts().insert(sc)

    def _db_url(self):
        """The database of the Qt Library panel (same URL), so every module uses one database."""
        lp = getattr(self.shell, "libpanel", None)
        if lp is None:
            return None
        cfg = getattr(lp, "cfg", None) or {}
        if cfg.get("mode") == "api":                     # the Qt panel talks to an API server, not to the database
            return None
        url = cfg.get("db_url") or None
        if url and url.startswith("sqlite:///") and not url.startswith("sqlite:////") and ":memory:" not in url:
            url = "sqlite:///" + os.path.abspath(url[len("sqlite:///"):]).replace("\\", "/")   # same file for every module
        return url

    def ensure(self):
        """Start the API (first time only) and load the page."""
        if self._loaded:
            return
        db = self._db_url()
        be = WebBackend.get()
        QtWidgets.QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            base = be.start(db)
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        if not base:
            self.info.setText("The DS3 web module could not start:\n" + (be.error or "unknown error"))
            self.info.setVisible(True)
            self.view.setVisible(False)
            return
        self._loaded = True
        print("[DS3 web] database of the web modules:", be.db_shown, flush=True)
        self.shell.flash("Library / Image database: " + str(be.db_shown))
        self.view.setUrl(QtCore.QUrl(f"{base}/app/?embed={self.embed}"))

    def _after_load(self, ok):
        """Open the requested module (the page's own start-up may have left it on its home screen)."""
        if not ok:
            return
        target = "image" if self.embed == "image" else "library-list"
        self.view.page().runJavaScript(
            "try{document.body.classList.add('embed');if(typeof state!=='undefined'&&state.view!=='%s'){setView('%s');}}"
            "catch(e){console.warn(e)}" % (target, target))

    def refresh_data(self):
        """Called whenever this page is shown: reload the list of libraries (another module may have saved one)."""
        if not self._loaded:
            return
        self.view.page().runJavaScript(
            "(function(){try{if(typeof loadLibraries!=='function'||typeof state==='undefined'){return;}"
            "Promise.resolve(loadLibraries()).then(function(){renderMenubar();"
            "if(state.view==='library-list'){render();}}).catch(function(e){console.warn(e)})}catch(e){console.warn(e)}})()")

    def reload(self):
        if self._loaded:
            self.view.reload()
