# Changes

## Main libraries + PostgreSQL (4 Nov 2026)
* Library page: **Main libraries** (loaded from `data/libraries/`, flagged `main` in the library metadata) and **Built libraries** (made with the Builder / from pixels) in separate sections; a panel shows the folder, loading progress, a warning when it is empty and a **Reload main libraries** button (`POST /api/startup_import/reload`).
* `docker-compose.yml`: `db` (PostgreSQL 16) and an optional `app` service (`--profile app`), `Dockerfile`, `requirements-server.txt`, `.env.postgres.example`, `docs/POSTGRES_DOCKER.md`.
* Verified on a real PostgreSQL 16 server: backend tests (34), both browser walk-throughs, start-up import, builder, aggregation, image analyses.

## Interface update (4 Nov 2026)
* Light, full-screen ENVI-style interface; menu bar **File · View · Tools · Spectral · Help** (all functional). The *Dashboard* tab and the *Library function* page/tabs were removed (same functions are reached from the **Spectral** menu); the **DS3** logo opens the three modules.
* **File ▸ Open Image File** → *Available Bands List* (Gray Scale / RGB, band per colour, Dims) → *Load RGB* → Image module. Desktop window opens files in place (`/api/import/path`, `pywebview` file dialog); browser uses upload.
* Image module: **Collect spectra** (every click adds a colour-coded spectrum), pairwise spectral angles, export CSV, *Build library* with category / acquisition, markers on the image.
* **Spectral** menu: Spectral Libraries (Builder / Viewer / Resampling), Spectral Analyst, Mapping Methods (SAM, linear unmixing, k-means), Transforms (PCA, Band Math), Spectral Indices, Anomaly (RX), Signal-to-Noise — backend `app/services/imageops.py`, `POST /api/images/{id}/analyze/{op}`. These bring over the corresponding functions of the HyperVista viewer (PCA, k-means, SAM, band math, anomaly, SNR, linear unmixing).
* Fix in the packaged build: `rasterio` sub-modules are now bundled, so the packaged program reads ENVI / GeoTIFF / Excel files; the real import error is printed when the DS3 core cannot be loaded.
* Tests: backend 33; browser walk-throughs in `tests/e2e/`.

## v6 — Task 1 version (this release)
Kept from v5, unchanged in behaviour: Library page, startup import of `data/libraries`, Image module (band selection, zoom, ROI, georeference, map, library overlay, best matches, classification map, class spectra, store spectrum), Library function (viewer, resampling), Add library, SQLite/PostgreSQL.

Added
* Header menu **Spectral ▸ Spectral Libraries ▸ Viewer / Resampling / Builder** (ENVI Classic order).
* **Spectral Library Builder** rebuilt on the backend in the ENVI Classic flow: *Input Spectral Wavelength From* (Data File / ASCII File / First Input Spectrum),
  *File Containing Output Wavelength* with file information, *Spectral Subset* and *Open ▾*; spectra from a library, any spectra file, or the open image (ROI / pixel, wired to the new Image module); save with category and acquisition tag.
* **Data aggregation** (Analysis page): ROIs from several scenes pooled per class on a common grid, export CSV / JSON / ENVI .sli, save as library (std and scenes kept in metadata).
* **Category tree** (`A/B/C` paths, counts, rename/move, filters) and **acquisition type** (lab / field / airborne / satellite) in the database, the API, import, viewer and builder.
* Readers for **Excel (.xlsx), JSON and Parquet** spectra (JSON reads back the aggregation export).
* `run_gui.py` (window or browser), `run_cli.py` (import / list / formats / aggregate), `build_exe.bat` + `ds3_library.spec`, `.env.example`, documentation.
* Database: `spectra.acquisition_type` (migration 003); `init_db()` adds it to older databases; SQLite and PostgreSQL both supported by the same code.
* Tests: backend 23 → 30 (taxonomy, acquisition, aggregation incl. pooled-statistics check, exports, Excel/JSON, builder endpoints).

Changed
* ROI reading shared between "ROI mean" and aggregation (`_roi_blocks`), resampling shared between Library resampling and the Builder (`services/resampling.py`).
* Paths: program files and data folder are separated so a packaged `.exe` can write next to itself.

Known gaps
* ENVI ROI (.roi) and EVF vector files are listed in the builder's Open menu as unavailable.
* The Library page has no category-tree pane yet (the tree and filters are in the Viewer).
* Plugins, mosaicking and georeferenced reprojection are not in this tool.
