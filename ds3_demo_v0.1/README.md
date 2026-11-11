# DS3 Library — spectral library, image module and data aggregation (Task 1) · version 0.1

Part of the DS3 (Deep Spectral Signature System) platform. This tool brings hyperspectral **images** and **spectral libraries**
from many formats into one open representation, stores them in a database, and lets you use them together.

| Function | Where |
|---|---|
| **Library** page: every library as a card, search across all libraries, upload any file (CSV/TXT tables, ENVI .sli, .sig, .asd, .xlsx, .json, Parquet, USGS, ECOSTRESS, .zip) | `material_library/frontend`, `app/routes/libraries.py`, `app/services/importer.py` |
| Your libraries in `data/libraries/` are read into the database at start-up (each file = one library; re-starts add nothing twice) | `app/main.py`, `app/services/importer.py` |
| **Menu bar** (light, full-screen, ENVI Classic style): **File** (Open Image File, Available Bands List, Open Spectral Library File), **View**, **Tools**, **Spectral**, **Help**; the DS3 logo returns to the three modules (Library · Image · Analysis) | `frontend/index.html` |
| **File ▸ Open Image File** → *Available Bands List* (Gray Scale / RGB, pick a band per colour, Dims) → *Load RGB* → opens in the Image module; in the desktop program the file is opened in place (nothing is copied) | `frontend/index.html`, `app/routes/admin.py` (`/api/import/path`), `app/launcher.py` |
| **Spectral ▸ Spectral Libraries ▸ Builder / Viewer / Resampling** — each opens that function directly | `app/routes/tools.py`, `app/routes/builder.py` |
| **Image module: collected spectra and object samples** — click a pixel: a small window shows its spectrum, give it a label and press *Add*; draw an ROI (double-click or right-click to finish) and *Add* it as an object sample. Both lists: select all / unselect all, delete all, save to a new or an existing library; *Show* pixel / ROI / both; object samples keep the polygon and the image file address (`.roi` export / import) | `frontend/index.html`, `app/routes/spectra.py`, `app/routes/images.py` |
| **Spectral ▸ Spectral Analyst, Mapping Methods (SAM, linear unmixing, k-means), Transforms (PCA, Band Math), Spectral Indices (NDVI, NDRE, SAVI, EVI, NDWI, NDBI, PRI), Anomaly (RX), Signal-to-Noise** — run on the backend on the open image (functions taken over from the HyperVista viewer of the repository) | `app/services/imageops.py`, `app/routes/imageops.py` |
| Builder: output wavelengths from *Data File* (image or library, with band subset and *Open ▾*), *ASCII File* or *First Input Spectrum*; add spectra from libraries, any spectra file or the open image (ROI/pixel); resample; save with category and acquisition tag | `app/routes/builder.py`, `app/services/resampling.py` |
| Viewer: plot up to 12 spectra, filter by category tree and acquisition type, export CSV | `app/routes/taxonomy.py`, `app/routes/spectra.py` |
| Resampling: whole library to a range, the bands of an image, or a list of wavelengths → new library | `app/routes/tools.py` |
| **Image** page: all formats, band selection, mouse-wheel zoom and pan, ROI polygon, georeference with control points (image beside the Esri map), map background, Show tick boxes (pixel spectra / object samples) in the tool line | `app/routes/images.py`, `app/services/images.py` |
| **Object sample collection (ROI)** (Image page, full width below the image): table with label, class, latitude, longitude, pixels and the image patch file `<class>.tif`; per-ROI surface-reflectance spectrum, export and locate; **Aggregate**: all spectra together + mean / standard deviation / variance per class (pooled over all pixels), export CSV / JSON / ENVI .sli, save to a library. *Analysis* is kept for AI / ML models | `src/ds3/aggregate/`, `app/services/aggregate.py`, `frontend/index.html` |
| Categories as paths (`Vegetation/Tree/Tsuga`) with a counted tree, rename/move; acquisition tag (lab / field / airborne / satellite) | `app/services/taxonomy.py`, `app/routes/taxonomy.py` |
| **Main libraries**: the files in `material_library/data/libraries/` are loaded into the database at start and listed under *Library ▸ Main libraries* (folder + *Reload* button on the page); libraries you build are listed as *Pixel-based libraries*, and libraries made from ROIs as *Object-based libraries* (patch file name + *View image* / *Locate*) | `app/main.py`, `app/services/importer.py` |
| Database: SQLite (no setup) or **PostgreSQL via Docker** (`docker compose up -d db`, guide in `docs/POSTGRES_DOCKER.md`); schema in `material_library/database/` | `app/models/`, `app/database.py` |
| Command line and a one-click launcher, packaged as a Windows `.exe` | `run_cli.py`, `run_gui.py`, `build_exe.bat` |

## Quick start (Windows, VS Code terminal)
```
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python run_gui.py
```
Your five `.txt` libraries: copy them into `material_library\data\libraries\` first. See **docs/RUN_AND_CHECK.md** for the full
step-by-step guide and a checklist to verify every function, and **DEVELOPER_GUIDE.md** for the structure.

## Tests
```
pytest tests                                   # DS3 core (readers, ingest, library builder)
cd material_library\backend && pytest tests    # API, database, aggregation, builder
```

## Status (honest)
Verified in a Linux sandbox: both test suites (core 83 tests; backend 34 tests, also run on PostgreSQL 16, SQLite) and two scripted browser runs (`tests/e2e/`: menus, File ▸ Open Image, band list, collect/compare/build library, all Spectral-menu analyses, builder dialogs,
viewer filters, aggregation) on synthetic data; the command line and the launcher without a window.
**Not verified:** a Windows `.exe` build (`build_exe.bat` is written but has not been run on Windows), the `pywebview` window, `docker compose` itself (PostgreSQL 16 was tested as a normal server, not inside Docker),
PostgreSQL with the newest columns, the new analyses on a full-size AVIRIS-NG scene (they work on a decimated copy of ≤ 250 000 pixels), and your real AVIRIS-NG / EnMAP / PaviaU data with the new aggregation page.
Limits: no login; images stay on disk and only their description is in the database; JSON files inside a folder import are skipped (import them directly);
aggregation reads at most ~120 000 pixels per ROI (regular sub-sampling for larger ROIs); mosaicking and georeferenced reprojection are not part of this tool.
