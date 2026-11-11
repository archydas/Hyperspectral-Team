# Library module → PostgreSQL material library (DS3 backend)

## v3.2 — satellite / airborne GeoTIFF products (EnMAP and others)
* GeoTIFF loading reads the **no-data value** (GDAL nodata tag; EnMAP -32768) and turns it into gaps (NaN), so
  borders and bad bands no longer show as -32768 spikes; ENVI `data ignore value` likewise.
* **Band wavelengths** are read from GDAL / ENVI band metadata or band descriptions, and from sensor metadata next
  to the image: an EnMAP `*-METADATA.XML` gives the centre wavelength and FWHM of all 224 bands; L2A reflectance
  (×10000) is scaled to 0–1, L1B/L1C DN converted with the per-band gain/offset; the VNIR/SWIR overlap
  (900–1000 nm) keeps the SWIR bands so wavelengths increase (209 bands).
* Quicklooks (`QL_VNIR/QL_SWIR`, 8-bit RGB) and quality layers are recognised and the open dialog says to use the
  `SPECTRAL_IMAGE` instead. The dialog shows what was found and no longer silently spreads 400–1000 nm over files
  without wavelengths (that is now an explicit, unticked option, default range 400–2500 nm).
* Multi-page TIFFs open with all bands also when rasterio is installed; compressed TIFFs without rasterio ask for
  `imagecodecs`.
* Library import of images (class means / image mean / pixels) skips no-data bands instead of rejecting every
  pixel.
* Test: synthetic EnMAP L2A product (`tests/test_loaders.py`).

## v3.1 — ROI Analysis moved from the viewer; ROIs class by class in the database
* New **ROI Analysis** tab (next to *Scenes*) with the DS3 viewer's ROI tools and ROI Manager: rectangle and
  polygon ROIs, editable class, pending selection, outlines in class colours, anomaly map, delete / clear / set
  class / rename, plot ROI spectra, plot class means, save / load DS3-ROIs JSON (compatible with the viewer). It
  works across all loaded images: one table for every image, class means pooled over images.
* **Database:** migration `002_rois.sql` — table `rois` (one row per ROI with class, image, pixel and map
  geometry, pixel count), columns `spectra.roi_id` and `spectra.roi_class`, link table `spectrum_rois`.
  Applied automatically on connect (PostgreSQL and SQLite); `schema.sql` updated for new installs.
* *Save ROIs to library database...*: selected / current image / all ROIs, stored as pooled class means, one mean
  per ROI or pixel spectra; class = material, ROIs = samples; idempotent. *Load ROIs from library database...*
  puts stored ROIs back on the same image or, through map coordinates, on another georeferenced image; deletes
  ROIs (with or without their spectra).
* Repository / both clients / API: `save_roi_spectra`, `list_rois`, `get_roi`, `roi_classes`, `roi_spectra`,
  `spectrum_roi_info`, `delete_roi`; `spectrum_dict` includes `roi_id` and `roi_class`.
* Scenes: ROIs have id / name / kind and may be polygons; `class_masks()`, aggregation, mosaic, `.npz` files and
  plugins handle them; old scene files still load. The rectangle-ROI controls left the *Scenes* tab.
* Library tab: a spectrum saved from ROIs shows its class and source ROI(s) / image(s).
* Also included: library-wide duplicate check on import, and the data-folder auto-import asks before importing.
* Tests: `tests/test_rois.py` (core + database, SQLite and PostgreSQL), `tests/test_gui_rois.py` (whole tab,
  headless). Verified on Python 3.10 and 3.13, and migration of an existing v3.0 PostgreSQL database.

## v3.0 — universal import (any spectral data format)
* New `hvva/library/formats/` package: a registry of readers, each detecting its own files, plus shared
  normalisation (units → nm incl. µm and wavenumber, reflectance → 0–1 from percent / ×10000, bad values → gaps,
  sorted grid, category and acquisition inference). Formats: CSV/TSV/TXT in any layout (columns, rows with
  metadata, long; any delimiter, decimal comma, `key: value` headers), Excel, JSON, Parquet, ENVI spectral
  libraries (any data type / byte order / scale factor), hyperspectral images (ENVI, GeoTIFF, MAT, HDF5, NetCDF,
  NumPy → class means, image mean or pixels), spectra matrices in MAT/HDF5/NPZ, ASD `.asd` (v1–v8, DN + white
  reference; matches specdal), SVC `.sig`, Spectral Evolution `.sed`, USGS splib07, ECOSTRESS, OpenSpecLib, and
  folders / zips of all of these (subfolders → categories).
* `hvva/library/ingest.py`: one import path for everything (`python -m hvva.library.ingest`, with `--preview`).
  `speclib`, `sources` (data folder) and the GUI all go through it.
* GUI: **Import ▸ Spectral data files (any format)... / A folder or zip...** opens an import wizard with live
  preview (format, counts, units, scale, categories, acquisition, plot) and overrides. The old per-format menu
  entries are gone; the data-folder scan now recognises every format.
* Tests: `tests/spectral_fixtures.py` generates one fixture per format; `tests/test_formats.py` (30 tests) checks
  detection, values against ground truth, overrides, and import of the whole folder into the database.
  Verified on Python 3.10 and 3.13.

## v2.3 — import from a local data folder (no command line needed)
* **Import ▸ From data folder...** in the Library tab: scans a folder (and its subfolders) for OpenSpecLib Parquet,
  USGS splib07 ASCII zips/folders, ECOSTRESS files and wide CSVs. For each one it proposes an editable library name,
  shows *new* / *imported* (by checksum or existing dataset), and imports the selected sources in one go, creating
  libraries as needed. No library has to be selected first.
* Optional **auto-import at startup**: new files dropped into the folder are imported when HyperVista connects.
  The folder and this option are remembered.
* Script equivalent: `python -m hvva.library.sources <folder> [--import] [--only name] [--limit N]`.
* `hvva/library/sources.py` (new). Every import now records its source file (SHA-256) or folder in `data_files`.
* Tests: folder scan, batch import, re-scan shows *imported*, a new file is picked up on its own.

## v2.2 — OpenSpecLib (direct-download) importer
* `python -m hvva.library.speclib openspeclib <file>.parquet` and *Import ▸ OpenSpecLib Parquet...* in the GUI read the
  OpenSpecLib release files (USGS splib07, ECOSTRESS, EcoSIS), which download directly from GitHub with no request
  form. Tested on the real v0.0.6 files: 1,947 ECOSTRESS + 1,810 USGS spectra.
* The v0.0.6 ECOSTRESS file stores percent reflectance although it is labelled 0–1. This is detected per file and
  rescaled.
* Duplicates are now detected by a per-spectrum source ID instead of the material name, so two measurements of
  the same sample are both kept and re-imports still add nothing.
* Needs `pyarrow` (optional; only for this importer).

## v2.1 — reference library importers
* `hvva/library/speclib.py` (new): imports the **USGS Spectral Library v7** ASCII release (e.g.
  `ASCIIdata_splib07b_cvASD.zip`) and **ECOSTRESS** spectra, from a zip or a folder, with category paths
  (`Minerals/Kaolinite`, `Vegetation/Tree/Acer/Rubrum`) and acquisition tags. Bad bands become NULL; µm → nm,
  % → 0–1. Idempotent. CLI `python -m hvva.library.speclib`, and in the GUI the Library tab's **Import** button
  is now a menu (Wide CSV / USGS / ECOSTRESS).
* `DATA_SOURCES.md` (new): download links, which file to take, citations, demo recipe.
* Tests for both importers (`tests/test_library_db.py`).

## v2 — category tree, acquisition type, search bar
* **Category tree**: categories are paths (`Vegetation/Tree/Tsuga/Canadensis`); new *Category tree* pane with counts
  rolled up to every level, click a node to filter to it and everything below, rename/move a subtree, "uncategorised"
  node. Paths are normalised (`>`, `->`, `|`, `\` also accepted as separators).
* **Acquisition type** per spectrum: lab / field / airborne / satellite (or untagged), coloured chips in the materials
  table, filter in the search bar (also applies to the all-materials search and pixel matching), set per spectrum on
  the Spectrum tab, in bulk via *Edit...*, on CSV import and when saving signatures/pixels; the comparison plot uses
  a line style per acquisition type and CSV export writes the tag.
* **Search bar** like the DS3 web module: "Search libraries by name" / "Search all materials by name" (results pop up
  under the field) / Acquisition filter; four resizable panes below it.
* **Bulk edit** of several selected materials (category and acquisition at once).
* Database: new column `spectra.acquisition_type` (+ CHECK constraint, indexes) —
  `database/migrations/001_category_tree_acquisition.sql`, applied automatically on connect; `schema.sql` updated
  for new installs. API: `/facets`, `/categories/rename`, `PATCH /api/materials`, `PATCH /api/spectra/{id}`,
  `category_path` / `acquisition` filters. Importer: `--category "A/B/C"`, `--acquisition lab`.
* Files changed in v2: `hvva/gui/library_panel.py` (rewritten), `hvva/library/{repository,client,importer,database,models}.py`,
  `hvva/library/api/main.py`, `hvva/library/{schema,migrations}.sql`, `database/schema.sql`,
  `database/migrations/001_category_tree_acquisition.sql` (new), `tests/test_library_db.py`, docs.

## v1

Only the library part was changed; scenes, aggregation, mosaic, export, georeferencing and plugins are untouched.

## New
| File | From new_hyperspectral | Purpose |
|---|---|---|
| `hvva/library/database.py` | `backend/app/database.py` | engine/sessions per URL, no crash without DATABASE_URL, psycopg2/psycopg auto-pick, `init_schema()` |
| `hvva/library/models.py` | `backend/models/*.py` | the six models in one file, same tables/columns; portable types (also runs on SQLite) |
| `hvva/library/repository.py` | `backend/app/routes/*.py` | route logic as plain functions (GUI + API share it) + save/match/delete helpers |
| `hvva/library/importer.py` | `backend/app/services/csv_importer.py` | parameterised CSV import (CLI + GUI), no pandas |
| `hvva/library/client.py` | – | `DirectClient` (SQLAlchemy) / `ApiClient` (HTTP), same interface |
| `hvva/library/api/` | `backend/app/main.py` | FastAPI server: original endpoints + delete, categories, bulk spectra, save spectra, CSV upload |
| `hvva/library/schema.sql`, `database/schema.sql`, `database/docker-compose.yml` | `database/`, `docker-compose.yml` | unchanged schema |
| `data/*.csv` | `data/` | sample spectra |
| `hvva/gui/library_panel.py` | DS3 prototype library UI | the new Library tab |
| `tests/test_library_db.py`, `.env.example` | – | tests (SQLite; PostgreSQL via HVVA_TEST_PG_URL), config template |

## Modified
* `hvva/gui/main_window.py` — Library tab is now `LibraryPanel`; the in-memory SQLite library and its handlers were
  removed; *Signature Export → Save to library database...*; remembers the clicked pixel position; File menu entries
  *Connect library database...* / *Import spectra CSV into library...*; closes the connection on exit.
* `hvva/library/__init__.py`, `requirements.txt`, `README.md`, `DEVELOPER_GUIDE.md`.

## Fixed relative to the original backend
* `csv_importer.py` passed `metadata=` to the models; SQLAlchemy treats that as the reserved `Base.metadata`
  attribute, so the JSON metadata was never written. Now stored in the real columns.
* `csv_importer.py` was hard-wired to `soil_spectra.csv` but named the library "Water Library"; now parameters.
* SQLAlchemy 2.1 defaults `postgresql://` to psycopg 3 while requirements install psycopg2 — resolved automatically.
