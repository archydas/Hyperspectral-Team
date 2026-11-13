# DS3 integrated project (Archy's DS3 Platform + DS3 Library and Image modules)

Base: `archydas/Hyperspectral-Team`, branch `ds3-platform`, commit `be02c4f` ("added roi analysis", v2.4).
Added: the Library and Image modules of the DS3 Library app, inside the DS3 Qt window.

## Native PyQt5 pages (no QtWebEngine)
The **Library (main)** and **Image** pages are now Qt widgets (`ds3/native/`), not HTML/JS in a QWebEngineView.
The FastAPI backend (`material_library/backend`) is unchanged and still runs as a separate process on 127.0.0.1; the pages
talk to it over HTTP in background threads, so the window never freezes.

| File | What |
|---|---|
| `ds3/web_pages.py` | `WebBackend` (starts the server, now in the background) + `WebPage` (builds the native page). Same names, so `shell.py`/`home.py` are unchanged |
| `ds3/native/api.py` | HTTP client + background jobs |
| `ds3/native/common.py` | library store, spectrum chart (matplotlib), library picker, save dialog |
| `ds3/native/library_main.py` | Library (main): cards (main / pixel / object), material search, library page, material page, compare |
| `ds3/native/library_tools.py` | import dialog, spectra picker, image-patch viewer, Spectral Library Builder, Viewer, Resampling |
| `ds3/native/image_main.py` | Image: open image, bands, zoom/pan, pixel spectra, ROI object samples + patches, collected list |
| `ds3/native/image_dialogs.py` | Available Bands List, Spectral menu windows + results, Spectral Analyst, SAM map, georeference, map view, aggregation |
| `ds3/native/mapview.py` | small web map in Qt (OSM / Esri / OpenTopoMap tiles, cached in `~/.hvva/tiles`) |

`material_library/frontend/` (HTML/JS) is no longer used by the Qt window (kept for the stand-alone web app).
`PyQtWebEngine` is not needed any more. Object samples (ROIs) and the patch folder are kept between sessions in the Qt settings.

## Sidebar
| Entry | Source |
|---|---|
| Library (main) | native Qt page, DS3 Library app backend: main / pixel-based / object-based libraries, Builder, Viewer, Resampling, Compare, View image |
| Image | native Qt page, DS3 Library app backend: image, pixel spectra collection, ROI object samples (`<class>.tif`), georeferencing, aggregate, save to library |
| Data Aggregation | Archy's, unchanged |
| Viewer and Analysis | Archy's / team's, unchanged |

Archy's own Qt Library page is **kept** and reachable by **View ▸ Library (classic)** (Ctrl+5), because other modules call into
it (database connection, Library tab of the Viewer). It is not in the sidebar. Its extra features are not yet merged into the new Library.

## Run in VS Code (Windows)
1. Open this folder in VS Code (File ▸ Open Folder).
2. Terminal ▸ New Terminal, then run once:  `.\setup_windows.ps1`   (creates `.venv` with Python 3.12 and installs everything)
   If PowerShell blocks the script:  `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`
3. Ctrl+Shift+P ▸ "Python: Select Interpreter" ▸ choose `.venv`.
4. Run and Debug (Ctrl+Shift+D), choose one of:
   * **DS3 - SQLite quick test** - no PostgreSQL needed (file `data/ds3_local.db`)
   * **DS3 - PostgreSQL, create missing tables** - first run on an empty database
   * **DS3 - PostgreSQL** - next runs
   The two PostgreSQL entries ask for the URL, e.g. `postgresql://postgres:<password>@localhost:5433/ds3_team`
   (a `@` in the password is written `%40`).  Press F5.

## One database for all modules
The Library and Image pages use the same database URL as the Qt Library panel (the one in `--db` or the saved connection).
The status line shows `Library / Image database: ...` when a web page opens. Changing the connection afterwards needs a restart.
Schema = Archy's `database/schema.sql`. On start the web modules add the columns they need (`images.reader_kw`) and fill
her required `images.fingerprint`.

## Layout of what was added
```
ds3/web_pages.py        starts the backend process, hosts the native pages
ds3/native/             the native PyQt5 Library (main) and Image pages
ds3lib/                 core package of the DS3 Library app (was `ds3`, renamed: DS3 Platform already has a `ds3` package)
material_library/       backend/ (FastAPI) and frontend/ (HTML/JS) of the Library and Image modules
tools/add_web_modules.py  patch for ds3/shell.py, ds3/app.py, ds3/home.py (already applied in this folder)
requirements_web.txt, setup_windows.ps1, .vscode/
```
After `git pull` brings new `shell.py`/`app.py`/`home.py` from Archy, run `python tools/add_web_modules.py` again.

## Tested (Linux sandbox) / not tested
Tested: window start with the 4-entry sidebar; Library and Image open on their own views; SQLite and PostgreSQL; end-to-end flows
(image collect, object samples, menu/aggregation) on PostgreSQL built from Archy's schema; 90 unit tests of the Library app.
NOT tested: Windows; Archy's Viewer and Analysis (`spectral` could not be installed in the sandbox); her own GUI tests; the `.exe`.

## Known limits
* Map tiles (View map, Georeference) need internet; once seen they are cached on disk.
* Two ROI tools exist (Image page object samples; Viewer's ROI Analysis / Data Aggregation); they do not share data yet.
