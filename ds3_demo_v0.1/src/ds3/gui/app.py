from __future__ import annotations
import argparse, sys


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="ds3-library", description="DS3 Library - spectral library desktop app")
    ap.add_argument("--db", help="database URL (default: DATABASE_URL, last used, or ~/.ds3/ds3_library.db)")
    args = ap.parse_args(argv)
    try:
        from PySide6.QtWidgets import QApplication
    except ImportError:
        print("PySide6 is not installed. Run:  pip install -e \".[gui]\"", file=sys.stderr)
        return 2
    from .main_window import MainWindow
    app = QApplication(sys.argv[:1])
    app.setStyle("Fusion")
    app.setOrganizationName("Kapih"); app.setApplicationName("DS3Library")
    win = MainWindow(db_url=args.db)
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
