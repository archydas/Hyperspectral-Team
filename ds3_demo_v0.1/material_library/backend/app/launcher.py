"""Start the DS3 Library app as one program: database + API + the web interface in its own window.

    python run_gui.py                         # window (needs `pip install pywebview`) or the default browser
    python run_gui.py --db sqlite:///my.db    # another database (PostgreSQL URLs work too)
    python run_gui.py --no-window             # server only; open http://127.0.0.1:8000/app/ yourself
"""
from __future__ import annotations
import argparse, os, socket, sys, threading, time, urllib.request, webbrowser


def free_port(preferred: int) -> int:
    for port in [preferred] + list(range(preferred + 1, preferred + 50)):
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise RuntimeError("no free port found")


def wait_until_up(url: str, seconds: float = 60) -> bool:
    end = time.time() + seconds
    while time.time() < end:
        try:
            urllib.request.urlopen(url, timeout=1).read()
            return True
        except Exception:
            time.sleep(0.3)
    return False


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="DS3 Library — spectral library, image module and data aggregation")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--db", help="database URL, e.g. sqlite:///my.db or postgresql+psycopg://user:pw@host:5432/name (default: DATABASE_URL, .env, or a local SQLite file)")
    ap.add_argument("--no-window", action="store_true", help="do not open a window or browser")
    ap.add_argument("--no-import", action="store_true", help="do not load data/libraries at start-up")
    a = ap.parse_args(argv)
    if a.db:
        os.environ["DATABASE_URL"] = a.db
    if a.no_import:
        os.environ["AUTO_IMPORT"] = "0"
    os.environ["DS3_LOCAL_APP"] = "1"                  # desktop program: files may be opened in place (no upload copy)
    import uvicorn
    from .main import app
    from .database import display_url, DATA_HOME
    port = free_port(a.port)
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    base = f"http://127.0.0.1:{port}"
    if not wait_until_up(base + "/health"):
        print("The server did not start; see the messages above.", file=sys.stderr)
        return 1
    print(f"DS3 Library running at {base}/app/   database: {display_url()}   data folder: {DATA_HOME / 'data'}")
    url = base + "/app/"
    if a.no_window:
        print("Press Ctrl+C to stop.")
    else:
        try:
            import webview

            class Api:                                   # exposed to the page as window.pywebview.api
                def pick_files(self, folder=False):
                    w = webview.windows[0]
                    fd = getattr(webview, "FileDialog", None)          # pywebview >= 5.3; older versions use the constants
                    kind = (fd.FOLDER if folder else fd.OPEN) if fd else (webview.FOLDER_DIALOG if folder else webview.OPEN_DIALOG)
                    r = w.create_file_dialog(kind, allow_multiple=not folder)
                    return list(r) if r else []

            webview.create_window("DS3 — Deep Spectral Signature System", url, width=1360, height=880, js_api=Api(),
                                  maximized=True)
            webview.start()
            server.should_exit = True
            return 0
        except ImportError:
            webbrowser.open(url)
            print("(install pywebview for a separate window: pip install pywebview)  Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        server.should_exit = True
    return 0


if __name__ == "__main__":
    sys.exit(main())
