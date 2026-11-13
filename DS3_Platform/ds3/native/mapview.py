"""A small slippy web map drawn with Qt (tiles from OpenStreetMap / Esri World Imagery / OpenTopoMap, cached on disk).
Wheel = zoom, drag = move, click = a point (lon, lat). It can show markers and an image placed by its 4 corners.
Used by the Image page for 'View map' and 'Georeference' (instead of Leaflet in QtWebEngine)."""
from __future__ import annotations

import math
import os
import re
import threading
import urllib.request

from PyQt5 import QtCore, QtGui, QtWidgets

from .api import ApiError, run_bg

W = QtWidgets
Qt = QtCore.Qt

try:                                                    # reuse the tile servers + disk cache of the DS3 viewer
    from hvva.core.basemap import CACHE, PROVIDERS, USER_AGENT
except Exception:                                       # noqa: BLE001
    CACHE = os.path.join(os.path.expanduser("~"), ".hvva", "tiles")
    USER_AGENT = "DS3/2 (desktop hyperspectral tool)"
    PROVIDERS = {
        "OpenStreetMap": {"url": "https://tile.openstreetmap.org/{z}/{x}/{y}.png", "max_zoom": 19,
                          "attribution": "© OpenStreetMap contributors"},
        "Esri World Imagery (satellite)": {
            "url": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            "max_zoom": 19, "attribution": "Imagery © Esri, Maxar, Earthstar Geographics"},
        "OpenTopoMap": {"url": "https://tile.opentopomap.org/{z}/{x}/{y}.png", "max_zoom": 17,
                        "attribution": "© OpenStreetMap contributors, SRTM | © OpenTopoMap (CC-BY-SA)"}}

ESRI = "Esri World Imagery (satellite)"
TILE = 256
_MEM = {}
_LOCK = threading.Lock()


def world_xy(lon, lat):
    """lon/lat -> world coordinates 0..1 (web mercator, y down)."""
    lat = max(-85.05112878, min(85.05112878, lat))
    x = (lon + 180.0) / 360.0
    s = math.sin(math.radians(lat))
    y = 0.5 - math.log((1 + s) / (1 - s)) / (4 * math.pi)
    return x, y


def world_lonlat(x, y):
    lon = x * 360.0 - 180.0
    n = math.pi - 2 * math.pi * y
    lat = math.degrees(math.atan(math.sinh(n)))
    return lon, lat


def tile_bytes(provider, z, x, y, timeout=10):
    key = (provider, z, x, y)
    with _LOCK:
        if key in _MEM:
            return _MEM[key]
    safe = re.sub(r"[^A-Za-z0-9]+", "_", provider)
    path = os.path.join(CACHE, safe, str(z), str(x), f"{y}.tile")
    data = None
    if os.path.exists(path):
        with open(path, "rb") as f:
            data = f.read()
    if data is None:
        req = urllib.request.Request(PROVIDERS[provider]["url"].format(z=z, x=x, y=y), headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = r.read()
        except Exception as e:                          # noqa: BLE001 - offline: shown on the map, no traceback
            raise ApiError(f"map tiles: {e}") from None
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "wb") as f:
                f.write(data)
        except OSError:
            pass
    with _LOCK:
        _MEM[key] = data
    return data


class MapView(W.QWidget):
    """clicked(lon, lat). Markers: [(lon, lat, text, color)]. Image: set_image(pixmap, corners[(lon,lat) UL, UR, LR, LL])."""
    clicked = QtCore.pyqtSignal(float, float)
    moved = QtCore.pyqtSignal(float, float)

    def __init__(self, provider=ESRI, parent=None):
        super().__init__(parent)
        self.provider = provider
        self.z = 2
        self.cx, self.cy = 0.5, 0.5                      # centre in world coordinates
        self.tiles = {}
        self.pending = set()
        self.failed = set()
        self.markers = []
        self.image = None
        self.corners = None
        self.opacity = 0.8
        self.error = ""
        self._press = None
        self._drag = False
        self.setMouseTracking(True)
        self.setMinimumSize(300, 240)
        self.setCursor(Qt.CrossCursor)
        self.setFocusPolicy(Qt.StrongFocus)

    # ---- coordinates
    def world_size(self, z=None):
        return TILE * 2 ** (self.z if z is None else z)

    def to_screen(self, lon, lat):
        x, y = world_xy(lon, lat)
        ws = self.world_size()
        return QtCore.QPointF((x - self.cx) * ws + self.width() / 2, (y - self.cy) * ws + self.height() / 2)

    def to_lonlat(self, px, py):
        ws = self.world_size()
        return world_lonlat(self.cx + (px - self.width() / 2) / ws, self.cy + (py - self.height() / 2) / ws)

    def max_zoom(self):
        return int(PROVIDERS.get(self.provider, {}).get("max_zoom", 19))

    def set_provider(self, name):
        self.provider = name
        self.z = min(self.z, self.max_zoom())
        self.tiles.clear()
        self.failed.clear()
        self.update()

    def center_on(self, lon, lat, z=None):
        self.cx, self.cy = world_xy(lon, lat)
        if z is not None:
            self.z = int(max(1, min(self.max_zoom(), z)))
        self.update()

    def fit(self, lonlats, pad=40):
        if not lonlats:
            return
        xs, ys = zip(*[world_xy(lo, la) for lo, la in lonlats])
        self.cx, self.cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
        span = max(max(xs) - min(xs), (max(ys) - min(ys)) * self.width() / max(1, self.height()), 1e-9)
        z = math.log2(max(1, self.width() - 2 * pad) / (TILE * span))
        self.z = int(max(1, min(self.max_zoom(), math.floor(z))))
        self.update()

    def set_markers(self, markers):
        self.markers = list(markers)
        self.update()

    def set_image(self, pixmap, corners, opacity=None):
        self.image, self.corners = pixmap, corners
        if opacity is not None:
            self.opacity = opacity
        self.update()

    def image_transform(self):
        """QTransform pixmap (x, y) -> screen, or None."""
        if self.image is None or not self.corners:
            return None
        w, h = self.image.width(), self.image.height()
        src = QtGui.QPolygonF([QtCore.QPointF(0, 0), QtCore.QPointF(w, 0), QtCore.QPointF(w, h), QtCore.QPointF(0, h)])
        dst = QtGui.QPolygonF([self.to_screen(lo, la) for lo, la in self.corners])
        t = QtGui.QTransform()
        return t if QtGui.QTransform.quadToQuad(src, dst, t) else None

    # ---- tiles
    def _need(self, z, x, y):
        key = (self.provider, z, x, y)
        if key in self.tiles or key in self.pending or key in self.failed or len(self.pending) > 24:
            return
        self.pending.add(key)
        prov = self.provider

        def done(data):
            self.pending.discard(key)
            img = QtGui.QImage.fromData(data)
            if not img.isNull():
                self.tiles[key] = img
                if len(self.tiles) > 600:
                    for k in list(self.tiles)[:200]:
                        self.tiles.pop(k, None)
            self.update()

        def fail(m):
            self.pending.discard(key)
            self.failed.add(key)
            self.error = m
            self.update()
        run_bg(tile_bytes, prov, z, x, y, done=done, failed=fail)

    def paintEvent(self, e):
        p = QtGui.QPainter(self)
        p.fillRect(self.rect(), QtGui.QColor("#dfe3e8"))
        p.setRenderHint(QtGui.QPainter.SmoothPixmapTransform)
        n = 2 ** self.z
        ws = self.world_size()
        ox = self.cx * ws - self.width() / 2
        oy = self.cy * ws - self.height() / 2
        x0, x1 = int(math.floor(ox / TILE)), int(math.floor((ox + self.width()) / TILE))
        y0, y1 = max(0, int(math.floor(oy / TILE))), min(n - 1, int(math.floor((oy + self.height()) / TILE)))
        missing = False
        for ty in range(y0, y1 + 1):
            for tx in range(x0, x1 + 1):
                key = (self.provider, self.z, tx % n, ty)
                img = self.tiles.get(key)
                rx, ry = tx * TILE - ox, ty * TILE - oy
                if img is not None:
                    p.drawImage(QtCore.QRectF(rx, ry, TILE, TILE), img)
                else:
                    missing = True
                    # a lower zoom tile, scaled, while this one loads
                    for dz in (1, 2, 3):
                        zz = self.z - dz
                        if zz < 0:
                            break
                        f = 2 ** dz
                        k2 = (self.provider, zz, (tx % n) // f, ty // f)
                        im2 = self.tiles.get(k2)
                        if im2 is not None:
                            sx, sy = ((tx % n) % f) * TILE / f, (ty % f) * TILE / f
                            p.drawImage(QtCore.QRectF(rx, ry, TILE, TILE), im2, QtCore.QRectF(sx, sy, TILE / f, TILE / f))
                            break
                    self._need(self.z, tx % n, ty)
        t = self.image_transform()
        if t is not None:
            p.save()
            p.setOpacity(self.opacity)
            p.setTransform(t)
            p.drawPixmap(0, 0, self.image)
            p.restore()
            p.setPen(QtGui.QPen(QtGui.QColor("#fbbf24"), 1.5, Qt.DashLine))
            p.drawPolygon(QtGui.QPolygonF([self.to_screen(lo, la) for lo, la in self.corners]))
        for m in self.markers:
            lon, lat, text, color = (list(m) + [None, None])[:4]
            pt = self.to_screen(lon, lat)
            col = QtGui.QColor(color or "#fbbf24")
            p.setPen(QtGui.QPen(Qt.black, 3))
            p.drawEllipse(pt, 6, 6)
            p.setPen(QtGui.QPen(col, 2))
            p.drawEllipse(pt, 6, 6)
            if text:
                f = p.font()
                f.setBold(True)
                p.setFont(f)
                p.setPen(Qt.black)
                p.drawText(pt + QtCore.QPointF(9, -7), str(text))
                p.setPen(col)
                p.drawText(pt + QtCore.QPointF(8, -8), str(text))
        p.setPen(QtGui.QColor("#333"))
        att = PROVIDERS.get(self.provider, {}).get("attribution", "")
        fm = p.fontMetrics()
        tw = fm.horizontalAdvance(att) + 10 if hasattr(fm, "horizontalAdvance") else fm.width(att) + 10
        p.fillRect(QtCore.QRectF(self.width() - tw, self.height() - 16, tw, 16), QtGui.QColor(255, 255, 255, 190))
        p.drawText(QtCore.QRectF(self.width() - tw, self.height() - 16, tw - 5, 16), Qt.AlignRight | Qt.AlignVCenter, att)
        if missing and self.failed and not self.tiles:
            p.setPen(QtGui.QColor("#b91c1c"))
            p.drawText(self.rect(), Qt.AlignCenter, "Map tiles could not be loaded (no internet?)\n" + self.error[:120])
        p.end()

    # ---- mouse
    def wheelEvent(self, e):
        d = e.angleDelta().y()
        if not d:
            return
        nz = max(1, min(self.max_zoom(), self.z + (1 if d > 0 else -1)))
        if nz == self.z:
            return
        pos = e.pos()
        lon, lat = self.to_lonlat(pos.x(), pos.y())
        self.z = nz
        # keep the point under the cursor in place
        wx, wy = world_xy(lon, lat)
        ws = self.world_size()
        self.cx = wx - (pos.x() - self.width() / 2) / ws
        self.cy = wy - (pos.y() - self.height() / 2) / ws
        self.update()

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self._press = (e.pos(), self.cx, self.cy)
            self._drag = False

    def mouseMoveEvent(self, e):
        lon, lat = self.to_lonlat(e.x(), e.y())
        self.moved.emit(lon, lat)
        if self._press and e.buttons() & Qt.LeftButton:
            p0, cx, cy = self._press
            d = e.pos() - p0
            if self._drag or abs(d.x()) + abs(d.y()) > 4:
                self._drag = True
                self.setCursor(Qt.ClosedHandCursor)
                ws = self.world_size()
                self.cx, self.cy = cx - d.x() / ws, min(1.0, max(0.0, cy - d.y() / ws))
                self.update()

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.LeftButton and self._press:
            if not self._drag:
                lon, lat = self.to_lonlat(e.x(), e.y())
                self.clicked.emit(lon, lat)
            self._press = None
            self._drag = False
            self.setCursor(Qt.CrossCursor)
