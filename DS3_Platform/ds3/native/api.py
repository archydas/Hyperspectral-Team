"""HTTP client for the DS3 backend (FastAPI in a separate process) + background calls for the Qt pages."""
from __future__ import annotations

import base64
import json
import mimetypes
import os
import traceback
import urllib.error
import urllib.parse
import urllib.request
import uuid

from PyQt5 import QtCore, QtGui


class ApiError(Exception):
    pass


class Api:
    """Minimal JSON client: get / post / put / patch / delete, file download and multipart upload."""

    def __init__(self, base):
        self.base = base.rstrip("/")

    def url(self, path, params=None):
        u = self.base + path
        if params:
            q = {k: v for k, v in params.items() if v is not None and v != ""}
            if q:
                u += ("&" if "?" in u else "?") + urllib.parse.urlencode(q)
        return u

    def _open(self, req, timeout):
        try:
            return urllib.request.urlopen(req, timeout=timeout)
        except urllib.error.HTTPError as e:
            detail = ""
            try:
                j = json.loads(e.read().decode("utf-8", "replace"))
                d = j.get("detail")
                if isinstance(d, list):
                    detail = "; ".join(f"{'.'.join(str(x) for x in it.get('loc', [])[1:])}: {it.get('msg')}" for it in d)
                else:
                    detail = str(d)
            except Exception:                                   # noqa: BLE001
                detail = e.reason
            raise ApiError(detail or f"HTTP {e.code}") from None
        except urllib.error.URLError as e:
            raise ApiError(f"The DS3 server does not answer ({e.reason}).") from None

    def request(self, method, path, body=None, params=None, timeout=600, raw=False):
        data, headers = None, {}
        if body is not None:
            data = json.dumps(body).encode("utf-8")
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(self.url(path, params), data=data, headers=headers, method=method)
        with self._open(req, timeout) as r:
            payload = r.read()
            if raw:
                cd = r.headers.get("Content-Disposition") or ""
                name = cd.split('filename="')[1].split('"')[0] if 'filename="' in cd else None
                return payload, name
        return json.loads(payload.decode("utf-8")) if payload else None

    def get(self, path, **params):
        return self.request("GET", path, params=params)

    def post(self, path, body=None, **params):
        return self.request("POST", path, body=body if body is not None else {}, params=params)

    def put(self, path, body=None):
        return self.request("PUT", path, body=body or {})

    def patch(self, path, body=None):
        return self.request("PATCH", path, body=body or {})

    def delete(self, path):
        return self.request("DELETE", path)

    def download(self, path, body):
        """POST returning a file: (bytes, filename)."""
        return self.request("POST", path, body=body, raw=True)

    def upload(self, path, files, fields=None, timeout=600):
        """multipart/form-data POST: files = [(field, filepath)]."""
        bnd = "----ds3" + uuid.uuid4().hex
        out = []
        for k, v in (fields or {}).items():
            if v is None or v == "":
                continue
            out += [f"--{bnd}\r\n".encode(), f'Content-Disposition: form-data; name="{k}"\r\n\r\n'.encode(),
                    str(v).encode("utf-8"), b"\r\n"]
        for field, fp in files:
            name = os.path.basename(fp)
            ctype = mimetypes.guess_type(name)[0] or "application/octet-stream"
            with open(fp, "rb") as f:
                content = f.read()
            out += [f"--{bnd}\r\n".encode(),
                    f'Content-Disposition: form-data; name="{field}"; filename="{name}"\r\n'.encode(),
                    f"Content-Type: {ctype}\r\n\r\n".encode(), content, b"\r\n"]
        out.append(f"--{bnd}--\r\n".encode())
        req = urllib.request.Request(self.url(path), data=b"".join(out), method="POST",
                                     headers={"Content-Type": f"multipart/form-data; boundary={bnd}"})
        with self._open(req, timeout) as r:
            return json.loads(r.read().decode("utf-8"))


# ------------------------------------------------------------------ background calls
class _Signals(QtCore.QObject):
    done = QtCore.pyqtSignal(object)
    failed = QtCore.pyqtSignal(str)


class _Job(QtCore.QRunnable):
    def __init__(self, fn, args, kwargs):
        super().__init__()
        self.fn, self.args, self.kwargs = fn, args, kwargs
        self.sig = _Signals()

    def run(self):
        try:
            res = self.fn(*self.args, **self.kwargs)
        except ApiError as e:
            self.sig.failed.emit(str(e))
            return
        except Exception as e:                                  # noqa: BLE001 - shown to the user
            traceback.print_exc()
            self.sig.failed.emit(f"{type(e).__name__}: {e}")
            return
        self.sig.done.emit(res)


_JOBS = set()


def _safe(cb, arg):
    """Call a GUI callback; never let an exception escape a Qt slot (PyQt5 would abort the program)."""
    try:
        cb(arg)
    except RuntimeError as e:                                   # widget closed while the call was running
        if "deleted" not in str(e):
            traceback.print_exc()
    except Exception:                                           # noqa: BLE001
        traceback.print_exc()


def run_bg(fn, *args, done=None, failed=None, **kwargs):
    """Run fn(*args) in the thread pool; done(result) / failed(message) are called in the GUI thread."""
    job = _Job(fn, args, kwargs)
    _JOBS.add(job.sig)

    def _done(r):
        _JOBS.discard(job.sig)
        if done:
            _safe(done, r)

    def _fail(m):
        _JOBS.discard(job.sig)
        if failed:
            _safe(failed, m)
    job.sig.done.connect(_done)
    job.sig.failed.connect(_fail)
    job.setAutoDelete(True)
    QtCore.QThreadPool.globalInstance().start(job)
    return job


def wait_all(timeout_s=30):
    """Process events until every background call has finished (used by the tests)."""
    import time
    from PyQt5 import QtWidgets
    end = time.time() + timeout_s
    while _JOBS and time.time() < end:
        QtWidgets.QApplication.processEvents()
        time.sleep(0.02)
    QtWidgets.QApplication.processEvents()
    return not _JOBS


def pixmap_from_data_url(s):
    """'data:image/png;base64,...' -> QPixmap."""
    if not s:
        return QtGui.QPixmap()
    b64 = s.split(",", 1)[1] if "," in s else s
    img = QtGui.QImage.fromData(base64.b64decode(b64))
    return QtGui.QPixmap.fromImage(img)


def data_url_bytes(s):
    return base64.b64decode(s.split(",", 1)[1] if "," in s else s)
