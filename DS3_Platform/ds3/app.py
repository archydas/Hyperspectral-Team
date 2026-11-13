"""Start DS3:  python run_ds3.py  [--db postgresql://postgres:postgres@localhost:5432/material_library | sqlite:///file.db]
                                  [--init] [image file]"""
from __future__ import annotations

import argparse
import os
import sys


def main(argv=None):
    ap = argparse.ArgumentParser(prog="DS3", description="DS3 - Deep Spectral Signature System")
    ap.add_argument("image", nargs="?", help="hyperspectral image to open in Viewer and Analysis")
    ap.add_argument("--db", help="library database URL (overrides the saved connection)")
    ap.add_argument("--init", action="store_true", help="create the database tables if they are missing")
    args = ap.parse_args(argv)

    os.environ.setdefault("QT_AUTO_SCREEN_SCALE_FACTOR", "1")
    try:
        from PyQt5 import QtWebEngineWidgets  # noqa: F401  (Qt rule: before the QApplication is created)
    except Exception:                                    # DS3 web modules are optional
        pass
    from PyQt5 import QtCore, QtWidgets
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
    from . import APP_NAME, SETTINGS_ORG, style
    app.setOrganizationName(SETTINGS_ORG)
    app.setApplicationName(APP_NAME)
    style.apply(app)
    from .shell import DS3Window
    win = DS3Window()
    lp = win.libpanel
    if args.db:
        lp.cfg["mode"] = "direct"
        lp.cfg["db_url"] = args.db
        QtCore.QTimer.singleShot(400, lambda: lp.connect_db(init=args.init, quiet=False))
    win.show()
    if args.image:
        QtCore.QTimer.singleShot(600, lambda: win.open_image_path(os.path.abspath(args.image)))
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
