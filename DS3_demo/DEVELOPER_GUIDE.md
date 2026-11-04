# Developer guide

## Layout
    run_gui.py / run_cli.py       entry points (add src/ and the backend to the path; work without `pip install -e`)
    build_exe.bat, ds3_library.spec   Windows packaging (PyInstaller)
    requirements.txt, .env.example
    src/ds3/                      core — no web, no database server needed
      ingest/                       image readers (ENVI, GeoTIFF/EnMAP, MAT, HDF5, NetCDF, Zarr, raw), the output contract (Cube), QC, ROI
      spectra/                      spectra readers: csv, tabular (any text table), svc/asd/sed, ENVI .sli, USGS, ECOSTRESS, Excel, JSON, Parquet
      library/                      core-side library builder, grid, search, image_ops (preview, SAM map)
      aggregate/                    common wavelength grid, resampling, pooled per-class statistics, export (numpy only)
      db/, api/, gui/, pipeline.py  the original DS3 core database, API and desktop GUI (separate from the web tool below)
    material_library/
      backend/app/                web tool: FastAPI + SQLAlchemy
        models/                     libraries → datasets → materials → samples → spectra, data_files, images
        routes/                     HTTP endpoints (one file per area)
        services/                   logic the routes call: importer, images, aggregate, resampling, taxonomy, readers
        launcher.py, cli.py         start-up window / command line
      backend/tests/              API tests
      frontend/index.html         the interface (one file; the Spectral menu, Library, Image, Analysis pages)
      database/                   schema.sql + idempotent migrations 001…003
      data/libraries/             files read into the database at start-up
    tests/                        core tests

Rules: `src/ds3/aggregate` and `src/ds3/spectra` never import FastAPI or SQLAlchemy; routes only parse the request and call a service;
services own the database work. All wavelengths are **nm**, ascending. Values keep their unit (reflectance 0–1, radiance, …) and units are never mixed silently.

## Data model
`Library 1–n Dataset`, `Library 1–n Material 1–n Sample 1–n Spectrum`, `Dataset 1–n DataFile` (file name, SHA-256, size: provenance and duplicate detection).
A spectrum stores `values[]`, `wavelength_start/end/step`, `wavelengths[]` only when the grid is not evenly spaced, `measurement_unit`,
`acquisition_type` (lab|field|airborne|satellite|NULL) and `metadata` (JSON: source file, `purity`, and for aggregated signatures `std`, `n_pixels`, `source_scenes`).
`materials.category` is a `/`-separated path; the tree is computed from it (`services/taxonomy.py`), so no extra table exists.
Images: one row per registered file (shape, unit, wavelengths, bad-band mask, reader options); the pixels stay in the file.
Schema changes: add an idempotent file to `database/migrations/`, mirror it in `schema.sql` and in the model; `init_db()` adds missing columns on start-up.

## Aggregation algorithm
For every ROI: read the polygon's pixels block by block at full resolution (regular sub-sampling above ~120 000 pixels), mask bad bands,
resample each block linearly to the common grid (no extrapolation), and merge it into the class accumulator with the exact parallel
update of Chan et al. Mean and sample standard deviation (n − 1) are therefore identical to computing them on all pooled pixels
(this is tested). Grid: `intersection` (range all scenes cover) or `union` (NaN where a scene has no data); default step = the coarsest native spacing.

## Adding things
* **A spectra format:** write `read_xxx(path, **kw)` returning `list[Spectrum]` in `src/ds3/spectra/`, add it to `detect_spectra_format` and `_READERS`; the importer, builder and CLI pick it up. Add a test.
* **An endpoint:** logic in `services/`, a thin route in `routes/`, register it in `main.py`, test it in `backend/tests/`.
* **An interface function:** the page is plain JavaScript; new blocks go at the end of `index.html` (later declarations replace earlier ones).

## Packaging
`build_exe.bat` runs PyInstaller with `ds3_library.spec`. The interface and schema are bundled; the database, uploads and `data\libraries` live next to the `.exe`
(`DS3_HOME` changes that folder). Build on Windows; copy the whole `dist\DS3Library` folder to the client.

## Relation to the team's code
The same database layout as the team's library tool (libraries → datasets → materials → samples → spectra) is used so spectra can move between the tools;
the extra column `acquisition_type` and the `images` table are the differences. Their PyQt5 viewer and this tool exchange data through files (ENVI .hdr/.sli, CSV, JSON).
