# DS3 integrated project (Archy's DS3 Platform + DS3 Library and Image modules)

Base: `archydas/Hyperspectral-Team`, branch `ds3-platform`, commit `be02c4f` ("added roi analysis", v2.4).
Added: the Library and Image modules of the DS3 Library app (FastAPI + HTML/JS), shown inside the DS3 Qt window.

## Sidebar
| Entry | Source |
|---|---|
| Library | DS3 Library app: main / pixel-based / object-based libraries, Builder, Viewer, Resampling, Compare, View image |
| Image | DS3 Library app: image, pixel spectra collection, ROI object samples (`<class>.tif`), georeferencing, aggregate, save to library |
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
ds3/web_pages.py        Qt page hosting a web module (API in a background thread, Qt file dialogs)
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
* The web pages still show their own menu row below the Qt menu bar.
* Leaflet, geotiff and fflate are bundled in `material_library/frontend/vendor/` (works offline; only the map *tiles* need internet).
* Windows ships an old Qt WebEngine (Chromium 83). `frontend/js/compat.js` adds the missing JavaScript functions (replaceAll, FinalizationRegistry, ...).
* Two ROI tools exist (Image page object samples; Viewer's ROI Analysis / Data Aggregation); they do not share data yet.
