"""Small DS3 building blocks: drawn icons, chips, cards, a reflowing card grid, page header."""
from __future__ import annotations

from PyQt5 import QtCore, QtGui, QtWidgets

from .style import C, MODULE, prop

W = QtWidgets
Qt = QtCore.Qt


# ------------------------------------------------------------------ icons (drawn, no image files needed)
def icon_pixmap(kind: str, color: str, size: int = 18) -> QtGui.QPixmap:
    s = size
    pm = QtGui.QPixmap(s * 2, s * 2)                     # 2x for crisp HiDPI
    pm.fill(Qt.transparent)
    p = QtGui.QPainter(pm)
    p.setRenderHint(QtGui.QPainter.Antialiasing)
    pen = QtGui.QPen(QtGui.QColor(color), s * 0.16)
    pen.setCapStyle(Qt.RoundCap)
    pen.setJoinStyle(Qt.RoundJoin)
    p.setPen(pen)
    p.scale(2, 2)
    m = s * 0.18
    if kind == "library":                                 # book
        r = QtCore.QRectF(m * 1.2, m * 0.8, s - 2.4 * m, s - 1.6 * m)
        p.drawRoundedRect(r, 2, 2)
        p.drawLine(QtCore.QPointF(r.left(), r.bottom() - s * 0.2), QtCore.QPointF(r.right(), r.bottom() - s * 0.2))
    elif kind == "image":                                 # picture
        r = QtCore.QRectF(m, m, s - 2 * m, s - 2 * m)
        p.drawRoundedRect(r, 2, 2)
        p.drawEllipse(QtCore.QPointF(r.left() + r.width() * 0.33, r.top() + r.height() * 0.33), s * 0.06, s * 0.06)
        path = QtGui.QPainterPath(QtCore.QPointF(r.left(), r.bottom() - r.height() * 0.15))
        path.lineTo(r.left() + r.width() * 0.45, r.top() + r.height() * 0.5)
        path.lineTo(r.right(), r.bottom() - r.height() * 0.05)
        p.drawPath(path)
    elif kind == "analysis":                              # bar chart
        for i, h in enumerate((0.3, 0.55, 0.8)):
            x = s * (0.3 + 0.2 * i)
            p.drawLine(QtCore.QPointF(x, s * 0.8), QtCore.QPointF(x, s * (0.8 - h * 0.65)))
    elif kind == "logo":                                  # ((•))
        c = QtCore.QPointF(s / 2, s / 2)
        p.setBrush(QtGui.QColor(color))
        p.drawEllipse(c, s * 0.07, s * 0.07)
        p.setBrush(Qt.NoBrush)
        for rad in (0.2, 0.34):
            rr = QtCore.QRectF(c.x() - s * rad, c.y() - s * rad, 2 * s * rad, 2 * s * rad)
            p.drawArc(rr, 135 * 16, 90 * 16)
            p.drawArc(rr, -45 * 16, 90 * 16)
    elif kind == "home":
        path = QtGui.QPainterPath(QtCore.QPointF(m, s * 0.5))
        path.lineTo(s / 2, m)
        path.lineTo(s - m, s * 0.5)
        p.drawPath(path)
        p.drawRect(QtCore.QRectF(s * 0.3, s * 0.48, s * 0.4, s * 0.35))
    elif kind == "folder":
        path = QtGui.QPainterPath(QtCore.QPointF(m, s * 0.3))
        path.lineTo(s * 0.42, s * 0.3)
        path.lineTo(s * 0.5, s * 0.4)
        path.lineTo(s - m, s * 0.4)
        path.lineTo(s - m, s - m * 1.2)
        path.lineTo(m, s - m * 1.2)
        path.closeSubpath()
        p.drawPath(path)
    p.end()
    pm.setDevicePixelRatio(2)
    return pm


class Chip(W.QLabel):
    """Rounded square with a module icon (DS3 'chip')."""

    def __init__(self, module: str, size: int = 26, kind: str = None, solid=False):
        super().__init__()
        col, fill, border, _btn = MODULE[module]
        self.setFixedSize(size, size)
        self.setAlignment(Qt.AlignCenter)
        if solid:
            self.setStyleSheet(f"QLabel {{ background: {col}; border-radius: {size // 3.5}px; }}")
            self.setPixmap(icon_pixmap(kind or module, "white", int(size * 0.62)))
        else:
            self.setStyleSheet(f"QLabel {{ background: {fill}; border: 1px solid {border}; border-radius: {size // 3.5}px; }}")
            self.setPixmap(icon_pixmap(kind or module, col, int(size * 0.62)))


def label(text, kind=None, wrap=False, rich=False):
    l = W.QLabel(text)
    if kind:
        prop(l, ds3=kind)
    l.setWordWrap(wrap)
    if rich:
        l.setTextFormat(Qt.RichText)
    return l


def primary(text, accent, fn=None):
    b = W.QPushButton(text)
    prop(b, ds3="primary", accent=accent)
    b.setCursor(Qt.PointingHandCursor)
    if fn:
        b.clicked.connect(fn)
    return b


class Card(W.QFrame):
    clicked = QtCore.pyqtSignal()

    def __init__(self, clickable=False):
        super().__init__()
        prop(self, ds3="card")
        self.lay = W.QVBoxLayout(self)
        self.lay.setContentsMargins(16, 14, 16, 14)
        self.lay.setSpacing(6)
        if clickable:
            self.setCursor(Qt.PointingHandCursor)
        self._clickable = clickable

    def mouseReleaseEvent(self, e):
        if self._clickable and e.button() == Qt.LeftButton:
            self.clicked.emit()
        super().mouseReleaseEvent(e)


class StatCard(Card):
    def __init__(self, label_text):
        super().__init__()
        self.lay.setContentsMargins(14, 10, 14, 10)
        self.lay.setSpacing(0)
        self.value = label("0", "statvalue")
        self.lay.addWidget(self.value)
        self.lay.addWidget(label(label_text, "statlabel"))

    def set(self, v):
        self.value.setText("–" if v is None else f"{v:,}" if isinstance(v, int) else str(v))


class ModuleCard(Card):
    def __init__(self, module, title, text, button, fn):
        super().__init__()
        self.lay.setContentsMargins(20, 18, 20, 18)
        self.lay.setSpacing(8)
        self.lay.addWidget(Chip(module, 40))
        self.lay.addWidget(label(title, "cardtitle"))
        t = label(text, "cardtext", wrap=True)
        self.lay.addWidget(t, 1)
        self.lay.addSpacing(4)
        self.lay.addWidget(primary(button, module, fn))


class Pill(W.QLabel):
    """Small rounded tag (e.g. 'Main library')."""
    STYLES = [(C["emerald_d"], C["emerald"]), (C["sky_d"], C["sky"]), (C["amber_d"], C["amber"]),
              (C["purple_d"], C["purple"]), (C["orange_d"], C["orange"]), (C["blue_d"], C["blue"])]

    def __init__(self, text, idx=0):
        super().__init__(text)
        bg, fg = self.STYLES[idx % len(self.STYLES)]
        self.setStyleSheet(f"QLabel {{ background: {bg}; color: {fg}; border-radius: 9px; padding: 2px 9px; "
                           f"font-size: 8.5pt; }}")
        self.setSizePolicy(W.QSizePolicy.Maximum, W.QSizePolicy.Fixed)


class CardGrid(W.QWidget):
    """Cards in as many columns as fit (min_width each), re-flowed on resize."""

    def __init__(self, min_width=300, max_cols=4, spacing=14):
        super().__init__()
        self.grid = W.QGridLayout(self)
        self.grid.setContentsMargins(0, 0, 0, 0)
        self.grid.setHorizontalSpacing(spacing)
        self.grid.setVerticalSpacing(spacing)
        self.min_width, self.max_cols = min_width, max_cols
        self.cards = []
        self._cols = 0

    def set_cards(self, cards):
        for c in self.cards:
            self.grid.removeWidget(c)
            c.setParent(None)
            c.deleteLater()
        self.cards = list(cards)
        self._cols = 0
        self._relayout(force=True)

    def _relayout(self, force=False):
        w = max(self.width(), 1)
        cols = max(1, min(self.max_cols, (w + self.grid.horizontalSpacing()) //
                          (self.min_width + self.grid.horizontalSpacing())))
        if cols == self._cols and not force:
            return
        self._cols = cols
        for c in self.cards:
            self.grid.removeWidget(c)
        for i, c in enumerate(self.cards):
            self.grid.addWidget(c, i // cols, i % cols)
        for k in range(self.max_cols):
            self.grid.setColumnStretch(k, 1 if k < cols else 0)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._relayout()


def page_header(title, sub):
    box = W.QVBoxLayout()
    box.setSpacing(2)
    box.addWidget(label(title, "pagetitle"))
    s = label(sub, "pagesub", wrap=True)
    s.setMaximumWidth(760)
    box.addWidget(s)
    return box


def scroll_page(inner: W.QWidget) -> W.QScrollArea:
    sa = W.QScrollArea()
    sa.setObjectName("ds3Scroll")
    sa.setWidgetResizable(True)
    sa.setFrameShape(W.QFrame.NoFrame)
    inner.setObjectName("ds3ScrollBody")
    sa.setWidget(inner)
    return sa


def hline():
    f = W.QFrame()
    f.setFrameShape(W.QFrame.HLine)
    f.setStyleSheet(f"color: {C['border']};")
    return f
