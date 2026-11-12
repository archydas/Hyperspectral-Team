"""DS3 light theme (colours of the DS3 web prototype, 'UI v2: light, full-screen, ENVI-style shell')."""
from __future__ import annotations

from PyQt5 import QtCore, QtGui, QtWidgets

C = {
    "bg": "#e8ecf1", "panel": "#ffffff", "panel2": "#eef1f5", "side": "#f4f5f7",
    "border": "#cdd4de", "border2": "#b4bfcd", "text": "#1b2430", "title": "#0f2744",
    "muted": "#6b7684", "muted2": "#4a5563", "hover": "#e3e9f1", "active": "#d6e4f6", "menuhover": "#cfe0f5",
    "accent": "#1f5fae",
    "emerald": "#059669", "emerald_d": "#d1fae5", "emerald_b": "#a7f3d0",
    "blue": "#2563eb", "blue_d": "#dbeafe", "blue_b": "#bfdbfe",
    "violet": "#6d4fd8", "violet_btn": "#7c3aed", "violet_d": "#ede9fe", "violet_b": "#ddd6fe",
    "amber": "#d97706", "amber_d": "#fef3c7", "amber_b": "#fde68a",
    "sky": "#0284c7", "sky_d": "#e0f2fe", "purple": "#9333ea", "purple_d": "#f3e8ff",
    "orange": "#ea580c", "orange_d": "#ffedd5", "pink": "#db2777",
    "red": "#dc2626", "green_dot": "#10b981",
}
# accent per module (chip colour, chip fill, chip border, button colour)
MODULE = {
    "library": (C["emerald"], C["emerald_d"], C["emerald_b"], C["emerald"]),
    "image": (C["blue"], C["blue_d"], C["blue_b"], C["blue"]),
    "analysis": (C["violet"], C["violet_d"], C["violet_b"], C["violet_btn"]),
    "home": (C["accent"], C["blue_d"], C["blue_b"], C["accent"]),
}

QSS = f"""
QWidget {{ color: {C['text']}; font-size: 9.5pt; }}
QMainWindow, QDialog {{ background: {C['bg']}; }}
QWidget#ds3Main, QWidget#ds3Page, QScrollArea#ds3Scroll, QWidget#ds3ScrollBody {{ background: {C['bg']}; }}
QWidget#ds3Sidebar {{ background: {C['side']}; border-right: 1px solid {C['border2']}; }}

QMenuBar {{ background: {C['side']}; border-bottom: 1px solid {C['border2']}; padding: 0 4px; }}
QMenuBar::item {{ padding: 6px 12px; background: transparent; color: {C['text']}; }}
QMenuBar::item:selected, QMenuBar::item:pressed {{ background: {C['menuhover']}; }}
QMenu {{ background: #fbfcfd; border: 1px solid {C['border2']}; padding: 3px 0; }}
QMenu::item {{ padding: 6px 28px 6px 16px; color: {C['text']}; }}
QMenu::item:selected {{ background: {C['menuhover']}; }}
QMenu::item:disabled {{ color: #a0a8b4; }}
QMenu::separator {{ height: 1px; background: #d9dfe7; margin: 3px 0; }}

QStatusBar {{ background: {C['side']}; border-top: 1px solid {C['border2']}; color: #5a6575; }}
QStatusBar QLabel {{ color: #5a6575; }}
QToolTip {{ background: #fbfcfd; color: {C['text']}; border: 1px solid {C['border2']}; padding: 4px; }}

QPushButton {{ background: #e6ebf2; color: {C['text']}; border: 1px solid #cdd6e2; border-radius: 6px;
              padding: 5px 12px; min-height: 20px; }}
QPushButton:hover {{ background: #d8e1ed; }}
QPushButton:pressed {{ background: #cad6e6; }}
QPushButton:disabled {{ color: #a0a8b4; background: #eef1f5; border-color: #dde3ea; }}
QPushButton:checked {{ background: {C['active']}; border-color: {C['accent']}; }}
QPushButton[ds3="primary"] {{ color: white; border: none; border-radius: 8px; padding: 7px 14px; font-weight: 500; }}
QPushButton[ds3="primary"][accent="library"] {{ background: {C['emerald']}; }}
QPushButton[ds3="primary"][accent="image"] {{ background: {C['blue']}; }}
QPushButton[ds3="primary"][accent="analysis"] {{ background: {C['violet_btn']}; }}
QPushButton[ds3="primary"][accent="home"] {{ background: {C['accent']}; }}
QPushButton[ds3="primary"]:hover {{ background: #1e40af; }}
QPushButton[ds3="primary"][accent="library"]:hover {{ background: #047857; }}
QPushButton[ds3="primary"][accent="analysis"]:hover {{ background: #6d28d9; }}
QPushButton[ds3="primary"]:disabled {{ background: #9ad7bf; color: #f0fdf4; }}
QPushButton[ds3="link"] {{ background: transparent; border: none; color: {C['muted']}; padding: 2px 0; text-align: left; }}
QPushButton[ds3="link"]:hover {{ color: {C['title']}; }}

QPushButton[ds3="nav"] {{ background: transparent; border: none; border-radius: 9px; text-align: left;
                          padding: 7px 8px; color: #364152; font-size: 10.5pt; }}
QPushButton[ds3="nav"]:hover {{ background: {C['hover']}; color: {C['title']}; }}
QPushButton[ds3="nav"]:checked {{ background: {C['active']}; color: {C['title']}; font-weight: 600; }}

QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QPlainTextEdit, QTextEdit {{
    background: white; color: {C['text']}; border: 1px solid {C['border2']}; border-radius: 6px; padding: 4px 6px; }}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, QPlainTextEdit:focus {{ border: 1px solid {C['accent']}; }}
QLineEdit[ds3="search"] {{ padding: 7px 10px; font-size: 10pt; border-radius: 6px; }}
QComboBox QAbstractItemView {{ background: white; selection-background-color: {C['menuhover']}; selection-color: {C['text']}; }}

QGroupBox {{ background: white; border: 1px solid {C['border']}; border-radius: 10px; margin-top: 14px;
             padding: 10px 8px 8px 8px; font-weight: 600; color: {C['title']}; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 4px; color: {C['title']}; }}

QTabWidget::pane {{ border: 1px solid {C['border']}; border-radius: 8px; background: white; top: -1px; }}
QTabBar::tab {{ background: transparent; color: {C['muted2']}; padding: 6px 12px; border: none;
               border-bottom: 2px solid transparent; margin-right: 2px; }}
QTabBar::tab:selected {{ color: {C['title']}; border-bottom: 2px solid {C['accent']}; font-weight: 600; }}
QTabBar::tab:hover {{ color: {C['title']}; background: {C['hover']}; }}

QTableWidget, QTableView, QTreeWidget, QTreeView, QListWidget {{
    background: white; alternate-background-color: #f7f9fb; border: 1px solid {C['border']}; border-radius: 8px;
    gridline-color: #e3e8ef; selection-background-color: {C['menuhover']}; selection-color: {C['text']}; }}
QHeaderView::section {{ background: #f4f6f9; color: {C['muted2']}; border: none; border-bottom: 1px solid {C['border']};
                        padding: 5px 6px; font-weight: 600; }}
QScrollArea {{ border: none; background: transparent; }}
QSplitter::handle {{ background: {C['bg']}; }}
QSplitter::handle:horizontal {{ width: 6px; }}
QSplitter::handle:vertical {{ height: 6px; }}
QScrollBar:vertical {{ background: transparent; width: 11px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: #c3ccd8; border-radius: 4px; min-height: 30px; }}
QScrollBar:horizontal {{ background: transparent; height: 11px; margin: 2px; }}
QScrollBar::handle:horizontal {{ background: #c3ccd8; border-radius: 4px; min-width: 30px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QProgressBar {{ border: 1px solid {C['border']}; border-radius: 6px; background: white; text-align: center; }}
QProgressBar::chunk {{ background: {C['accent']}; border-radius: 5px; }}
QSlider::groove:horizontal {{ height: 4px; background: {C['border']}; border-radius: 2px; }}
QSlider::handle:horizontal {{ background: {C['accent']}; width: 14px; margin: -6px 0; border-radius: 7px; }}

QFrame[ds3="card"] {{ background: white; border: 1px solid {C['border']}; border-radius: 12px; }}
QFrame[ds3="card"]:hover {{ border-color: {C['border2']}; }}
QFrame[ds3="banner"] {{ background: white; border: 1px solid {C['border']}; border-radius: 10px; }}
QLabel[ds3="pagetitle"] {{ font-size: 15pt; font-weight: 600; color: {C['title']}; }}
QLabel[ds3="pagesub"] {{ font-size: 10pt; color: {C['muted2']}; }}
QLabel[ds3="cardtitle"] {{ font-size: 11pt; font-weight: 600; color: {C['title']}; }}
QLabel[ds3="cardtext"] {{ font-size: 10pt; color: {C['muted2']}; }}
QLabel[ds3="statvalue"] {{ font-size: 16pt; font-weight: 700; color: {C['title']}; }}
QLabel[ds3="statlabel"] {{ font-size: 9.5pt; color: {C['muted']}; }}
QLabel[ds3="section"] {{ font-size: 10.5pt; font-weight: 600; color: {C['title']}; }}
QLabel[ds3="seclabel"] {{ font-size: 8pt; color: #7b8696; letter-spacing: 1px; }}
QLabel[ds3="muted"] {{ color: {C['muted']}; }}
QLabel[ds3="error"] {{ color: {C['red']}; }}
QLabel[ds3="small"] {{ font-size: 8.5pt; color: {C['muted']}; }}
QLabel[role="fieldlabel"] {{ color: {C['muted']}; font-size: 8.5pt; padding-left: 2px; }}
QLabel[role="crumb"] {{ color: {C['muted']}; }}
"""

# matplotlib colours for both tools (registered as theme "ds3")
MPL = {
    "figure.facecolor": "white", "figure.edgecolor": "white", "savefig.facecolor": "white",
    "axes.facecolor": "white", "axes.edgecolor": "#9aa6b6", "axes.labelcolor": "#364152",
    "axes.titlecolor": C["title"], "text.color": C["text"], "xtick.color": "#4a5563", "ytick.color": "#4a5563",
    "grid.color": "#e3e8ef", "legend.facecolor": "white", "legend.edgecolor": "#cdd4de",
}
THEME_ENTRY = {"label": "DS3 light", "mpl": MPL, "pane_border": C["border"],
               "info_style": f"background:#ffffff; color:{C['text']}; border:1px solid {C['border']};"}


def register_theme(*theme_modules):
    """Add the 'ds3' theme to the viewer's and the library tool's theme tables and make it current."""
    for m in theme_modules:
        m.THEMES["ds3"] = THEME_ENTRY
        m.set_matplotlib_theme("ds3")


def apply(app: QtWidgets.QApplication):
    app.setStyle("Fusion")
    pal = QtGui.QPalette()
    for role, col in ((QtGui.QPalette.Window, C["bg"]), (QtGui.QPalette.Base, "#ffffff"),
                      (QtGui.QPalette.AlternateBase, "#f7f9fb"), (QtGui.QPalette.Text, C["text"]),
                      (QtGui.QPalette.WindowText, C["text"]), (QtGui.QPalette.Button, "#e6ebf2"),
                      (QtGui.QPalette.ButtonText, C["text"]), (QtGui.QPalette.Highlight, C["menuhover"]),
                      (QtGui.QPalette.HighlightedText, C["text"]), (QtGui.QPalette.ToolTipBase, "#fbfcfd"),
                      (QtGui.QPalette.ToolTipText, C["text"]), (QtGui.QPalette.Link, C["accent"])):
        pal.setColor(role, QtGui.QColor(col))
    pal.setColor(QtGui.QPalette.Disabled, QtGui.QPalette.Text, QtGui.QColor("#a0a8b4"))
    pal.setColor(QtGui.QPalette.Disabled, QtGui.QPalette.ButtonText, QtGui.QColor("#a0a8b4"))
    app.setPalette(pal)
    f = app.font()
    for fam in ("Segoe UI", "Inter", "Roboto", "Helvetica Neue", "Arial"):
        if fam in QtGui.QFontDatabase().families():
            f.setFamily(fam)
            break
    app.setFont(f)
    app.setStyleSheet(QSS)


def prop(w, **kw):
    """Set dynamic properties used by the QSS (e.g. prop(btn, ds3='primary', accent='library'))."""
    for k, v in kw.items():
        w.setProperty(k, v)
    w.style().unpolish(w)
    w.style().polish(w)
    return w
