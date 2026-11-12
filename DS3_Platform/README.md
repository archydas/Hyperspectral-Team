# DS3 — Deep Spectral Signature System (desktop, v2.3)

One PyQt5 program with the **DS3 interface** (the look of the `DS3_demo` web prototype) and **three modules** (sidebar order: Library · Data Aggregation · Viewer and Analysis)
that carry every feature of our tools:

| Module | Comes from | What it does |
|---|---|---|
| **Library** | HyperVista Library v3.4 (`hvva` library panel) | All spectral libraries as cards (main libraries from the libraries folder + built libraries), search libraries and materials across all libraries, then the full library browser: category tree, materials, spectrum, compare, match / save, acquisition · unit · purity · covers filters, Builder, Resampling, Export, import of any spectra format |
| **Data Aggregation** | HyperVista Library v3.4 (data aggregation window) | Scenes, ROI Analysis, Wavelength Grid, Aggregation, Mosaic / Stack, Signature Export, map view (georeferenced files), Plugins. The Georeferencing tab is left out in DS3. |
| **Viewer and Analysis** | DS3 Hyperspectral Viewer & Analysis Tool (`viewer/HyperVista_v1_0.py`, the team's modified viewer) + DS3 **Library** tab | Display (single band / RGB), Preprocessing, Band Management, Data Correction, Spectral Analysis, Unmixing, Dimensionality Reduction (PCA / MNF), Classification (classical models, cross-validation, saved models), Spectral Matching (SAM), Segmentation (deep models, ground-truth files) — plus the DS3 image↔library features (collect spectra, compare angles, CSV, best matches, library SAM map, store to library, send to Data Aggregation). No ROI Analysis tab: ROIs come from Data Aggregation. |

All three share **one database connection** (PostgreSQL in Docker or a local SQLite file).

![home](docs/screenshots/1_home.png)

### Where ROIs live

ROIs are drawn only in **Data Aggregation ▸ ROI Analysis** (Viewer and Analysis has no ROI tab in DS3). Its
Classification, Spectral Matching, Segmentation, Unmixing and empirical-line tools still use ROIs: bring them over with
**Tools ▸ Get ROIs from Data Aggregation** (the Data Aggregation scene of the same size, e.g. after
*Library tab ▸ Send image to Data Aggregation*), or **Tools ▸ Load ROI File into Viewer and Analysis** (DS3-ROIs JSON).

## Install and run (Windows)

```bat
cd "D:\Hyperspectral code files\DS3_Platform"
pip install -r requirements.txt
docker start material_library_db          :: the PostgreSQL container (postgres/postgres, port 5432)
python run_ds3.py                          :: or double-click run_ds3.bat
```

First start: **File ▸ Connect Library Database…** (or click the red dot in the sidebar) →
`postgresql://postgres:postgres@localhost:5432/material_library`. The connection is remembered.
Command-line options:

```bat
python run_ds3.py --db postgresql://postgres:postgres@localhost:5432/material_library
python run_ds3.py --db sqlite:///D:/ds3_library.db --init        :: no Docker: local SQLite file (tables created)
python run_ds3.py "D:\data\ENMAP...\SPECTRAL_IMAGE.TIF"           :: open an image at start
```

Main libraries: every file in `data/libraries/` (default folder; *Change folder…* on the Library page) is
imported automatically the first time, and *Reload main libraries* imports new files (nothing is imported twice).

TensorFlow is optional (only the Viewer's 2D CNN tab).

## The interface

* **Menu bar** (File · View · Tools · Spectral · Help) — same structure as DS3_demo, see the menu map below.
* **Sidebar** — DS3 logo (= Home), MAIN MODULES Library / Data Aggregation / Viewer, recent images, database status.
* **Home** — counters (Libraries, Materials, Images opened, Spectra) and the three module cards.
* **Status bar** — messages of every module.

### Menu map

| Menu | Entry → what it opens |
|---|---|
| File | Open Image File (Ctrl+O), Open Previously Opened Image, Open Zarr Folder, Open Spectral Library File, Import ASD Field Spectra, Available Bands List, Save Image View (PNG/JPG/PDF/SVG), Export Active Cube, Add Scenes / Raw Binary Cube / Demo Scenes to Data Aggregation, Connect Library Database, Home, Exit |
| View | Home, Library (Ctrl+1), Data Aggregation (Ctrl+2), Viewer (Ctrl+3), Zoom In / Out, Fit (Ctrl+0), Map Background, Processing History, Revert to Original Cube |
| Tools | Collect Spectra, Draw ROIs (Data Aggregation), Clear Collected, Get ROIs from Data Aggregation (into Viewer and Analysis), Load ROI File into Viewer and Analysis, Send Image to Data Aggregation, Preprocessing ▸ (Normalisation, Units, Common Grid, Indices & Features, Statistics), Band Management, Data Correction, Collected Spectra (Data Aggregation), Library Images, Plugins |
| Spectral | Spectral Libraries ▸ Builder / Viewer / Resampling / Export · Spectral Analyst · Mapping Methods ▸ SAM (library) / SAM (ROI classes) / Linear Unmixing / K-Means · Transforms ▸ PCA / Band Math · Spectral Indices ▸ NDVI, NDRE, SAVI, EVI, NDWI, NDBI, PRI (+ more in Indices & Features) · Anomaly Detection (RX) · Signal-to-Noise · Classification ▸ RF / SVM, 2D CNN · Data Aggregation |
| Help | Supported Formats, Using the Viewer, Using Library and Data Aggregation, About |

## Project layout

```
DS3_Platform/
  run_ds3.py / run_ds3.bat     start DS3
  requirements.txt             everything (library + viewer)
  ds3/                         NEW - the DS3 interface
    app.py                     QApplication, --db / --init / image arguments
    shell.py                   main window: menus, sidebar, pages, status, spectral indices, zoom
    home.py                    Home page (counters + module cards)
    library_page.py            Library page (cards, search, folder banner) + detail view (library panel)
    image_page.py              Viewer and Analysis page (viewer) + the new Library tab (collect / match / SAM / store / send)
    analysis_page.py           Data Aggregation page (data aggregation window)
    style.py, widgets.py       DS3 colours, QSS, matplotlib theme, cards, chips, icons
  hvva/                        HyperVista Library v3.4 (one change: spectrum counts per library)
  viewer/HyperVista_v1_0.py    DS3 Viewer & Analysis Tool (team's modified viewer; ROI messages point to Data Aggregation)
  database/ plugins/ data/ assets/ tests/ docs/
```

## Tests

```bat
set QT_QPA_PLATFORM=offscreen
python -m pytest -q tests
```

121 passed (incl. `tests/test_ds3_gui.py`, which drives the whole DS3 window end to end).
See **CHANGES_DS3.md** for the feature comparison with DS3_demo.
