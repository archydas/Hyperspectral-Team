# Changes

## Version 0.3 (11 Nov 2026): simpler Image page, ROI click-to-delete, one library chooser
* **Delete an ROI from the image:** click its outline or its number → a small menu (*Spectrum*, *Export*, *Delete ROI*); the *Delete* key also works while it is selected.
* **Zoom stays while drawing an ROI:** clicking a point only redraws the polygon (before, the whole view was redrawn and jumped). The scroll position is also kept after other redraws.
* **Buttons:** *Draw ROI polygon* is green only while you are drawing (click it again to stop); *Clear ROI* is always there (greyed when there is nothing to clear); *Georeference* is green only while its window is open (the ✓ is gone).
* **The two tick boxes control whole parts of the page:** *Pixel spectra* = chart + collected spectra on the right (and the pixel markers); *Object samples (ROI)* = the table under the image (and the ROI outlines). Unticked = hidden, the image gets the free space.
* **Top line:** the wavelengths box and the *Bands…* button are gone (the R/G/B lists do the same; File ▸ Available Bands List is still there). A file without wavelengths now asks for the range in the "Could not open" window, with *Open again*.
* **Object sample table:** no more Export .roi / Import .roi / Export table / Export spectra (each row has its own *Export*); *Aggregate ticked ROIs* is one button — the aggregate (all spectra, mean / std. deviation / variance / min–max, table) appears only after pressing it, with *Export CSV* and *Save class means to library…*.
* **Save to library (Image page and Library Builder):** one drop-down — *＋ New library…* (then type the name) or an existing library (pixel libraries for spectra, object libraries for ROI samples; the main library is never offered).
* **Image patches (.tif):** all bands are in the file; the R, G, B bands shown on the page (else the bands nearest 650/550/470 nm) are stored as the default display bands (tags `default_bands` / `display_bands_rgb`). ENVI uses them; ArcMap and QGIS always open a multi-band TIFF on bands 1–3 — choose the bands in the layer's Symbology (RGB composite).
* Removed the duplicate *Export CSV* under the chart (the collected-spectra list has it).
* **Library ▸ View image (was “Load image”):** a saved ROI patch opens in a viewer window for **looking only**: all bands in the R/G/B lists (it starts on the bands that were on screen when the patch was saved), click a pixel for its spectrum, *Locate file*. It is **not** added to the Image module's list. (`GET /api/samples/{id}/patch/view|pixel`; the old `patch/load` is gone.)
* **Fixed:** the menu bar stayed empty when the mouse pointer was already over it at start-up.
* **Tests:** new/updated browser checks for all of the above, one more patch test (default display bands).

## Version 0.2 (11 Nov 2026): image page rebuilt, ROI patches, georeferencing, object libraries
**Code structure.** `frontend/index.html` is now only the page shell. Styles are in `frontend/css/app.css`, the code in `frontend/js/*.js` (21 plain scripts, loaded in the order listed at the bottom of `index.html`; every file starts with a comment saying what it holds). Functions that were declared twice (the old copy never ran) and 37 functions nothing called were removed. Behaviour is unchanged: the existing browser walk-throughs pass on the split version.
**Image page**
* The **Show** tick boxes (*Pixel spectra*, *Object samples (ROI)*) are in the tool line next to *Download image*, together with zoom −/+/Fit.
* **Mouse-wheel zoom** around the pointer, **drag to move**. When you zoom in beyond the drawn detail the backend sends a sharper preview (`/preview?max_px=`), so single pixels become visible. A drag never picks a pixel.
* The Library panel (Add library, upload, best matches, max angle, class spectra, **classification map**) is gone from the Image page; library matching stays under *Spectral ▸ Spectral Analyst*. The side panel is chart + collected spectra, as high as the image, so there is no blank area.
* **Object sample collection (ROI)** replaces *ROI aggregation*: a **full-width table under the image** (label, class, latitude, longitude, pixels, image = patch file name, source scene) with per-row *Spectrum*, *Export*, *Locate*, remove. *Select all / Unselect all*, *Save to library…*, *Export .roi / table / spectra*, *Delete all* (two clicks).
* **Image patches:** every ROI is written as a georeferenced GeoTIFF `<class name>.tif` (second ROI of a class: `_2`, `_3`). The folder is asked **once, at the first ROI** (system folder dialog in the desktop window, typed path in a browser); the library keeps the file name and address. Wavelengths are inside the file, so a patch opens again in DS3.
* **Per-ROI spectrum** window (surface reflectance, mean ± standard deviation, summary). **Aggregate** below the table: all ticked spectra together, *Calculate* gives mean, standard deviation, variance (and min–max) per class, pooled over all pixels at full resolution; export and save as before.
* **Georeference** is now the standard method: the image beside the **Esri World Imagery** map; click a point on the image, then the same point on the map; at least 3 pairs (4+ shows the error of every point and the RMSE in metres). The affine fit is stored with the image, so *View map* (Esri base, OpenStreetMap switch), ROI latitude/longitude and patches use it. Typing four corner coordinates is gone.
**Library module**
* The Library page has **Add library (import file)…** and **Build library…**, and lists **Main**, **Pixel-based** and **Object-based (ROI)** libraries separately.
* An object library opens as a table with the **patch file name only** and **Load image** (opens the file from where it was saved, not copied), **Locate** (shows it in the file browser) and *Spectrum*. A moved file is reported as "not found". No folder path is shown anywhere.
**Backend:** `routes/patches.py` (patch export/download, folder check, reveal, lon/lat of pixels, georeference get/fit/put/delete, object-library list, load/reveal patch), `services/patches.py`, `services/georef.py`; `/api/images/{id}/roi` also returns per-band std/min/max; `/preview` takes `max_px`; `geo()` falls back to the saved control points.
**Fixed:** a leftover timer could cancel a second *Delete all* click.
**Tests:** `tests/test_object_patches.py` (6), browser flows `run_image_collect_flow.py` (updated), `run_menu_image_analysis_flow.py`, `run_object_samples_flow.py` (new). The old `run_builder_aggregation_flow.py` was removed (it no longer matched the interface since the Builder redesign).

**DS3-version0.1** (10 Nov 2026): first complete package of the Task 1 library platform, including everything below.

## Image module v3: collected spectra, object samples, ROI aggregation (10 Nov 2026)
* **Click a pixel** → a small window shows its spectrum (axes: wavelength nm / reflectance) with a **Label** and **Add**; only then it goes to *Collected spectra*. The *Collect on/off* switch is gone.
* **ROI**: click points, **double-click or right-click** to finish (the *Finish ROI* button is gone). The same small window opens (label + class); **Add** puts it in *Collected object samples (ROI)*, which works like the collected spectra (select all / unselect all, click a name to view only that spectrum, rename, remove). *Clear* was replaced by **Delete all** (bottom right, asks for a second click).
* **Show** tick boxes: pixel spectra, ROI (object samples), or both: in the chart and on the image.
* **Save to library…** for both lists: new library name or an existing library (dropdown). Object samples keep the polygon and the **image file name and address** (not the pixels); *Export / Import .roi* writes/reads a small file with the polygons and the image address; *Locate file* opens the image's folder (desktop use). The Library viewer shows the source image address of such samples.
* **ROI aggregation** moved from *Analysis* to the **Image page, below the image** (class = pooled across images; mean and standard deviation, export, save to library). *Analysis* is kept for AI / ML models.
* **Builder**: opening spectra from files also saves the originals untouched as `<name> (raw)` (tick box); spectra taken from a database library leave that library unchanged and are saved as a separate, modified library. Library metadata records `raw`, `raw_library`, `derived_from_libraries`.
* Backend: `POST /api/spectra/save` accepts `library_info` and per-item `sample_metadata`; `GET /api/images/` returns `path`; `POST /api/images/{id}/reveal`.
* Tests: `tests/e2e/run_image_collect_flow.py` (new), `run_menu_image_analysis_flow.py` updated; `run_builder_aggregation_flow.py` predates the Builder redesign and no longer matches the interface.

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

## Compare spectra (multi-library)
- Spectrum page: "Add to comparison" replaced by **+ Add more spectra to compare**.
- New Compare Spectra workspace (also View ▸ Compare Spectra): choose the same or any other library, search/filter, click spectra to add them (up to 16); all are overlaid on one chart with colours, show/hide, remove, scale-to-maximum (shape only), spectral-angle table and CSV export.
