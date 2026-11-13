# DS3 - who owns which files

One window (`ds3/shell.py`), five sidebar pages, one database. Each page is separate code, so a teammate can work on
one module without touching the others. **Edit only the files of your own module** (list below); changes in the shared
files need a word with the team.

| Sidebar page | What it is | Its files (edit these) | Owner |
|---|---|---|---|
| **Library (main)** | spectral libraries: browse, import, builder, resample, compare, viewer (web page in the Qt window) | `material_library/frontend/js/`: `views-library.js`, `database.js`, `library-viewer.js`, `library-store-cnn.js`, `compare.js`, `builder-*.js`<br>`material_library/backend/app/routes/`: `libraries.py`, `materials.py`, `datasets.py`, `spectra.py`, `samples.py`, `builder.py`, `taxonomy.py`, `tools.py`<br>`material_library/backend/app/services/`: `importer.py`, `csv_importer.py`, `readers.py`, `resampling.py`, `taxonomy.py` | Arju |
| **Library 2** | Archy's Qt library (purity / covers filters, rename and move categories) | `ds3/library_page.py`, `hvva/gui/library_panel.py`, `hvva/gui/library_dialogs.py`, `hvva/library/` | Archy |
| **Image** | open an image, pixel spectra, ROI object samples, patches, georeferencing, aggregation of ROIs (web page in the Qt window) | `material_library/frontend/js/`: `image-*.js`, `object-samples.js`, `georef*.js`, `roi-labeling.js`<br>`material_library/backend/app/routes/`: `images.py`, `imageops.py`, `patches.py`, `aggregate.py`<br>`material_library/backend/app/services/`: `images.py`, `imageops.py`, `patches.py`, `georef.py`, `aggregate.py`<br>`ds3lib/ingest/` (image readers) | Arju |
| **Data Aggregation** | scenes, ROI Analysis, aggregation, mosaic, export | `ds3/analysis_page.py`, `hvva/gui/main_window.py`, `hvva/gui/roi_panel.py`, `hvva/gui/open_dialog.py`, `hvva/gui/canvases.py`, `hvva/core/` | Archy |
| **Viewer and Analysis** | open and analyse an image: preprocessing, PCA / MNF, classification, segmentation, unmixing | `ds3/image_page.py`, `viewer/` | third teammate |

## Shared files (change only after agreeing with the team)
- `ds3/shell.py`, `ds3/home.py`, `ds3/app.py`, `ds3/style.py`, `ds3/widgets.py` - the window: sidebar, menus, Home page
- `ds3/web_pages.py` - runs the web server of Library (main) / Image and shows them in the window
- the database: `database/schema.sql`, `hvva/library/models.py`, `material_library/backend/app/models/`. Tables are shared
  (`libraries`, `samples`, `spectra`, `images`, `rois`). **Add columns, never rename or remove them.**

## How the parts talk to each other
- Through the database only (a library saved in Image is read by Library (main), Library 2 and Data Aggregation).
- Library (main) and Image run as a web server in a separate process (`python -m uvicorn app.main:app`, started by
  `ds3/web_pages.py`), the other three modules run inside the Qt window.
- Run a module's tests: `pytest tests` (Qt modules), `pytest material_library/backend/tests` (Library (main) / Image).

## Working on your module
1. Pull the latest main branch, create your own branch (`git checkout -b yourname-library2`).
2. Change only your files, run the app (`python run_ds3.py --db sqlite:///data/ds3_local.db --init`) and check your page.
3. Commit and push your branch, and open a pull request. Merge into main after the others have looked at it.

Not done yet: the Qt menu bar and the small menu row inside the two web pages are still separate; one `.exe` is not built.
