"""DS3 Library (main) and Image modules inside the Qt window - native PyQt5 pages.

The data side is unchanged: the FastAPI backend of the DS3 Library app (folder ``material_library/backend``) runs as a
separate process on 127.0.0.1 and owns the database, the image readers and the analysis code. The pages themselves are
now drawn with Qt widgets (``ds3/native``): no QtWebEngine, no HTML / JavaScript.

* Database: the connection of the Qt Library panel is reused (``libpanel.cfg['db_url']``), so the Qt modules and these
  modules use the same PostgreSQL / SQLite database.  Without a connection they use the SQLite file in
  ``material_library/data``.  Override with the environment variable DS3_WEB_DB.
* Files are opened in place (the backend runs on this computer), file / folder choosers are Qt dialogs.

The class names (WebBackend, WebPage) are kept so that shell.py / home.py / app.py need no change.
"""
from __future__ import annotations

import atexit
import os
import socket
import subprocess
import sys
import time
import urllib.request

from PyQt5 import QtCore, QtWidgets

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

    def start_async(self, db_url, callback):
        """Start the server in the background; callback(base or None) runs in the GUI thread."""
        if self.base or self.error:
            return callback(self.base)
        self._waiting = getattr(self, "_waiting", [])
        self._waiting.append(callback)
        if len(self._waiting) > 1:
            return                                              # already starting

        def done(base):
            cbs, self._waiting = self._waiting, []
            for cb in cbs:
                cb(base)
        from .native.api import run_bg
        run_bg(self.start, db_url, done=done, failed=lambda m: (setattr(self, "error", m), done(None)))

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


class WebPage(W.QWidget):
    """A DS3 page with one native module: ``embed`` = 'library' (Library (main)) or 'image' (Image)."""

    def __init__(self, shell, embed, title):
        super().__init__()
        self.shell = shell
        self.embed = embed
        self.title = title
        self.setObjectName("ds3Page")
        self.native = None
        self._starting = False
        self.info = W.QLabel("")
        self.info.setWordWrap(True)
        self.info.setAlignment(Qt.AlignCenter)
        self.lay = W.QVBoxLayout(self)
        self.lay.setContentsMargins(0, 0, 0, 0)
        self.lay.addWidget(self.info, 1)

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
        """Start the backend (first time only, in the background) and build the page."""
        if self.native is not None or self._starting:
            return
        self._starting = True
        self.info.setText(f"Starting the DS3 data server for {self.title}…")
        self.info.setVisible(True)
        WebBackend.get().start_async(self._db_url(), self._started)

    def _started(self, base):
        self._starting = False
        be = WebBackend.get()
        if not base:
            self.info.setText("The DS3 Library / Image server could not start:\n" + (be.error or "unknown error") +
                              "\n\nCheck that the database is running (e.g. the PostgreSQL Docker container) and that the "
                              "packages of requirements_web.txt are installed, then restart DS3.")
            return
        if self.native is not None:
            return
        from .native.api import Api
        from .native.common import Store
        api = Api(base)
        store = Store.get(api)
        try:
            if self.embed == "image":
                from .native.image_main import ImageMainPage
                self.native = ImageMainPage(self.shell, api, store)
            else:
                from .native.library_main import LibraryMainPage
                self.native = LibraryMainPage(self.shell, api, store)
        except Exception as e:                                   # noqa: BLE001 - shown on the page
            import traceback
            traceback.print_exc()
            self.info.setText(f"The {self.title} page could not be built: {type(e).__name__}: {e}")
            return
        self.info.setVisible(False)
        self.lay.addWidget(self.native, 1)
        print("[DS3] database of the Library (main) / Image modules:", be.db_shown, flush=True)
        try:
            self.shell.flash("Library / Image database: " + str(be.db_shown))
        except Exception:                                        # noqa: BLE001
            pass
        self.refresh_data()

    def refresh_data(self):
        """Called whenever this page is shown: reload libraries (another module may have saved one) and images."""
        if self.native is None:
            return
        self.native.store.refresh()
        if self.embed == "image":
            self.native.refresh()

    def reload(self):
        self.refresh_data()
