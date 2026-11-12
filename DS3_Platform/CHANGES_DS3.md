# DS3 — what changed (main branch → DS3 interface)

## v2.2
* **Data Aggregation:** the Georeferencing tab is removed (files that carry georeferencing, such as EnMAP,
  still show on the map with *Map* under the scene view). Tools ▸ Georeference removed.
* **Viewer:** the ROI Analysis tab is removed; ROIs are drawn in Data Aggregation only.
  * Tools ▸ *Draw ROIs (Data Aggregation)* sends / selects the Viewer's image there and opens its ROI Analysis tab.
  * Tools ▸ *Get ROIs from Data Aggregation (into the Viewer)* copies the rectangle / polygon ROIs of the
    same-size scene into the Viewer, so Classification (RF / SVM), SAM with ROI classes, Unmixing and the
    empirical line keep working. Tools ▸ *Load ROI File into the Viewer* reads a DS3-ROIs JSON.
  * Data Correction tools that asked for "ROIs selected in the ROI Manager" now ask which ROI to use.
  * Library tab: *ROI class means to library* removed; *Send image to Data Aggregation* sends the cube only.
  * Spectral ▸ Anomaly Detection (RX) runs directly (it used to live in the ROI tab).
  * `viewer/HyperVista_v1_0.py`: only 5 message texts changed (they pointed to the ROI Analysis tab).

## v2.1
* "Analysis" renamed **Data Aggregation**, "Image" renamed **Viewer** (sidebar, home cards, page titles, menus,
  buttons, help).
* Module order everywhere: **Library → Data Aggregation → Viewer** (sidebar, home cards, View menu:
  Ctrl+1 / Ctrl+2 / Ctrl+3). Sidebar a bit wider so "Data Aggregation" fits.

## v2.0

## Starting point (main branch)

* `library/library/Library and Data aggregation/` — HyperVista Library (we use **v3.4**, the latest)
* `Viewer/HyperVista_v1_0.py` — HyperVista viewer
* `DS3_demo` — the web prototype (FastAPI + one HTML page) whose **interface** we copied

Both tools were separate windows with their own menus and dark/light themes. Now they run **inside one DS3
window**, with the DS3 look and menus, and share one library database connection.

## Files

| File | Status | Why |
|---|---|---|
| `run_ds3.py`, `run_ds3.bat` | new | start DS3 |
| `ds3/__init__.py` | new | name, version |
| `ds3/app.py` | new | QApplication, `--db`, `--init`, image argument |
| `ds3/shell.py` | new | DS3 main window: menu bar, sidebar, pages, status bar, spectral indices, zoom, help |
| `ds3/home.py` | new | Home page: 4 counters + 3 module cards |
| `ds3/library_page.py` | new | Library page: library cards (main / built), library + material search, libraries-folder banner, detail view |
| `ds3/image_page.py` | new | Viewer page hosting the viewer + the new **Library** tab |
| `ds3/analysis_page.py` | new | Data Aggregation page hosting the data-aggregation window |
| `ds3/style.py`, `ds3/widgets.py` | new | DS3 colours (light ENVI style), Qt style sheet, matplotlib theme, cards, chips, icons |
| `hvva/library/repository.py` | **changed** | `list_libraries` also returns `spectrum_count` (Home "Spectra" counter, library cards) |
| `hvva/**` (rest), `database/`, `plugins/`, `assets/` | unchanged | HyperVista Library v3.4 |
| `viewer/HyperVista_v1_0.py` | 5 message texts changed (v2.2) | copied from main `Viewer/` |
| `data/libraries/` | new | default folder of main libraries (material + soil spectra) |
| `requirements.txt` | new | library + viewer requirements merged |
| `tests/test_ds3_gui.py` | new | end-to-end test of the DS3 window |
| `README.md`, `CHANGES_DS3.md`, `docs/screenshots/` | new | docs |

The two programs are embedded, not rewritten: every feature keeps working exactly as before and fixes to
`hvva/` or `viewer/` carry over to DS3 directly.

## DS3_demo features → where they are now

| DS3_demo | DS3 desktop |
|---|---|
| Sidebar: logo, MAIN MODULES Library / Data Aggregation / Viewer, status | ✔ same, plus recent images and database status (click = connect) |
| Home: Libraries / Materials / Images opened / Spectra + module cards | ✔ |
| Library: search libraries, search materials, "N main libraries loaded · Folder · Reload main libraries", Main / Built library cards | ✔ (main libraries are imported at first start, like the prototype) |
| Library detail (materials, spectrum) | ✔ the full v3.4 library panel: category tree, filters, compare, match / save, edit |
| File ▸ Open Image File / Open Previously Opened / Open Spectral Library File / Available Bands List / Download Image Preview / Home | ✔ (preview = Save Image View: PNG, JPG, PDF, SVG) |
| View ▸ Home, Library, Image, Analysis, Zoom In / Out, Fit, Map Background | ✔ |
| Tools ▸ Collect Spectra, Draw ROI Polygon, Clear Collected, Georeference, Add ROI to Data Aggregation | ✔ |
| Spectral ▸ Spectral Libraries ▸ Builder / Viewer / Resampling | ✔ (+ Export) |
| Spectral ▸ Spectral Analyst | ✔ Best matches of the pixel / collected spectrum against the database |
| Spectral ▸ Mapping Methods ▸ SAM / Linear Unmixing / K-Means | ✔ SAM against the library (new) or ROI classes, Unmixing tab, K-Means |
| Spectral ▸ Transforms ▸ PCA / Band Math | ✔ |
| Spectral ▸ Spectral Indices NDVI, NDRE, SAVI, EVI, NDWI, NDBI, PRI | ✔ same formulas; `wNNN` = band nearest NNN nm, computed by the viewer's band math |
| Spectral ▸ Anomaly (RX), Signal-to-Noise, Data Aggregation | ✔ |
| Help ▸ Supported Formats, Using the Image Module, About | ✔ |
| Image: Collect on, Build library, Compare angles, Export CSV, Classify (SAM) | ✔ new **Library** tab of the Viewer |
| Image: library choice, ignore unit differences, Best matches, Classification map, max angle | ✔ new Library tab (+ SAM / RMSE, top N, max materials) |
| Image: Class spectra from map (ground-truth map → class means) | ✔ new (Library tab ▸ Store and hand over) |
| Image: store pixel to library | ✔ + ROI class means to library, Register image |
| File ▸ Remove Image | – (open another image instead) |

## Features our tools had that DS3_demo did not (all kept)

Viewer: Preprocessing (normalisation, units, common grid, 30+ indices and band features, statistics), Band
Management, Data Correction, ROI Analysis with rectangle / polygon ROIs and classes, Unmixing, PCA, RF / SVM
classification, 2D CNN, processing history / revert, many input formats (ENVI, GeoTIFF / EnMAP, MATLAB, HDF5,
NetCDF, Zarr, ASD …).
Library: PostgreSQL / SQLite / API, import of any spectra format, category tree, acquisition / unit / purity /
covers filters, compare, matching, Builder, resampling, export.
Data Aggregation: multi-scene aggregation on a wavelength grid, mosaic / stack, signature export (CSV / JSON / ENVI .sli /
library), georeferencing (origin, bounds, control points) with a web-map background, plugins.

## New features in the Viewer (Library tab)

* pixel window 1×1 / 3×3 / 5×5 / 7×7; numbered markers on the image for collected spectra
* compare angles (matrix of spectral angles), export CSV, build a library from the collected spectra
* best matches against one or all libraries (SAM or RMSE, unit check or shape-only)
* SAM classification map against a library or against the collected spectra, max angle, legend with %
* store the pixel, ROI class means or class means from a ground-truth map in a library; register the image
* send the processed cube + ROIs to Data Aggregation, or open the same file in Data Aggregation for map / georeferencing

## Tests

`python -m pytest -q tests` → 121 passed, 2 skipped (PostgreSQL tests skip without a server).
