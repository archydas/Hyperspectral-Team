# DS3 Material Library — web interface + FastAPI + PostgreSQL (v6)

> **v6:** the one-click launcher, command line, data aggregation, the ENVI-style builder, category/acquisition filters and the `.exe` build are described in the project's top-level **README.md**, **DEVELOPER_GUIDE.md** and **docs/RUN_AND_CHECK.md**. This file stays as the reference for the web tool itself.

The **DS3 interface from your first `ds3-platform.zip`** (dark shell: Dashboard · Library · Image · Analysis, Comparison, Library Builder, browser-only georeference viewer) — now with **no dummy data**: libraries, materials and spectra come from the database; uploads go to the database; the Image page opens any format and brings a database library into the image.
HTML → FastAPI → PostgreSQL (or SQLite with zero setup).
Your five `.txt` libraries: copy them into `material_library/data/libraries/` — they are read into the database when the backend starts and appear on the **Library** page when the app opens.
Lives inside the `ds3-platform` folder so it can use the DS3 readers (all formats).

## 1. Run it (SQLite, no Docker)  — 5 steps
```
cd ds3-platform\material_library\backend
python -m venv .venv
.venv\Scripts\activate                 (Linux/Mac: source .venv/bin/activate)
pip install -r requirements.txt        (also installs the DS3 core from the folder above)
uvicorn app.main:app --reload
```
Open **http://127.0.0.1:8000/app/** (the old dummy libraries are gone; the Library page lists what is in the database) (API docs: http://127.0.0.1:8000/docs).
No `.env` → database file `data/material_library.db`.

## 2. Run it on PostgreSQL (Docker)
```
cd ds3-platform\material_library
docker compose up -d                   # postgres:16, database material_library, schema.sql loaded on first start
copy backend\.env.example backend\.env
```
(Port **5433** is used so it does not clash with the `ds3-platform` compose file, which uses 5432.) Edit `backend\.env`: `DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5433/material_library`, then start `uvicorn` as above.
The left sidebar shows which database the page is connected to. Change the default password for anything beyond a local demo; keep `.env` private.
Database created from an older `schema.sql`? The API adds missing columns/tables at start; or run `database/migrations/001…sql`, `002…sql`.

## 3. Make your libraries appear every time the app opens
1. Copy your library files into **`material_library\data\libraries\`** (e.g. `material_spectra.txt`, `soil_spectra.txt`, `hsdos_lib_spectra.txt`, `samples_spectra_with_metadata.txt`, `ecostress_library_all_spectra.txt`).
2. Start the app (`uvicorn app.main:app`). At start-up each file becomes its **own library** (named after the file) and is stored in the database. The 9.6 MB ecostress file takes a few seconds: press **Refresh** on the Library page.
3. The Library page shows what happened (file by file, with the reason for any file it could not read); the same report is at http://127.0.0.1:8000/api/startup_import.
4. Next start-ups: files already in the database are recognised (SHA-256) and skipped, so nothing is duplicated; the libraries are simply there. Add a new file to the folder (or use **Upload…**) and it is picked up.
Another folder: set `AUTO_IMPORT_DIR=D:\your\folder` in `backend\.env`. Switch off: `AUTO_IMPORT=0`.
The libraries live in the database (SQLite file or PostgreSQL), not in the folder: with PostgreSQL in Docker they survive restarts as long as the `docker compose` volume is kept (don't run `docker compose down -v`).

One-off command instead (same result): `python -m app.services.import_cli D:\data\folder --each-file-a-library`

## 4. Add more data later (nothing is hard-coded)
**In the page**
- *Library* page → **Upload…**: one or many files — `.csv`, `.txt`/`.tsv`/`.dat` tables, `.sli`, `.sig`, USGS/ECOSTRESS files, or a `.zip` of them. Name the library or leave blank (file name is used).
- *Image* page → **Open image…**: AVIRIS-NG (**select the `.hdr` and its data file together**, e.g. `…_corr_v2m2_img` + `…_corr_v2m2_img.hdr`), EnMAP `.zip` (or the `SPECTRAL_IMAGE.TIF`), `PaviaU.mat`, GeoTIFF, HDF5/NetCDF.
  Files with no wavelengths (.mat): type the range in the box, e.g. `430-860`. PaviaU is recognised and its nominal 430–860 nm is **assumed and flagged**.

**From the command line** (good for your big `.txt` libraries)
```
cd backend
python -m app.services.import_cli D:\data\material_spectra.txt --library "Materials"
python -m app.services.import_cli D:\data\soil_spectra.txt hsdos_lib_spectra.txt samples_spectra_with_metadata.txt
python -m app.services.import_cli D:\data\ecostress_library_all_spectra.txt --library "ECOSTRESS"
python -m app.services.import_cli D:\data\PaviaU.mat --wavelengths 430-860
python -m app.services.import_cli --list
```
If a file's layout is not recognised, the import says what it saw (delimiter, first cells) instead of guessing — send me that line.

## 5. The two modules
**Library** — cards for every library (your five `.txt` files appear when the app opens). Open a library to browse and search its materials; click a material for its spectrum; *Add to comparison* overlays several.
**Library function** (button on the Library page) has three tools:
- *Spectral library builder* — pick a wavelength grid, add spectra (files, other libraries), resample, check, save as a library.
- *Spectral library viewer* — choose a library, tick materials, plot them together, export CSV.
- *Spectral library resampling* — resample a whole library to a new grid (start/end/step, the bands of an opened image, or a list of wavelengths) and save it as a **new** library (linear interpolation, no extrapolation, gaps not bridged, coverage check).

**Image** — *Open image…* takes any supported format (AVIRIS-NG `.hdr` + data file together, EnMAP `.zip`/`.TIF`, `PaviaU.mat`, GeoTIFF, HDF5/NetCDF). Then:
- **R/G/B band boxes** + *Apply bands*, **zoom** − / + / Fit, 1 px / 3×3 / 5×5 window, *Download image*.
- Click a pixel for its spectrum; **Draw ROI polygon** (click corners, *Finish ROI*) gives the ROI mean spectrum (computed on the server at full resolution).
- **Georeference** shows the corner coordinates read from the file; if the file has none, type north/south/west/east (north-up assumed; kept in this browser).
- **View map** puts the image on an OpenStreetMap background (needs internet; opacity slider; click the map to read a pixel).
- **Library → Add library** picks a library from the Library module (or uploads a library file); its materials appear in a scroll box (tick to overlay). Then **Best matches** (spectral angle, for the pixel or ROI), **Classification map**, **Class spectra from map…** (ground-truth map such as `PaviaU_gt.mat` → one mean spectrum per class) and **Store pixel/ROI spectrum to library**.

## 6. Demo without real data
`python demo/make_demo_data.py demo/out` writes a **synthetic** scene, label map and library (clearly labelled, not measurements). Upload them in the page.

## Tests
`cd backend && pytest -q` → 32 tests. Same tests on PostgreSQL: create an EMPTY scratch database and
`set TEST_DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5433/ml_test` first (the tests drop/re-create its tables).

## Status (honest)
Verified in a Linux sandbox (v5): 23 tests on SQLite **and** on a native PostgreSQL 16 server; `schema.sql` applies cleanly there; the full page (upload → image → library overlay → classify → save) was driven in a browser against PostgreSQL with no console or server errors.
**Not verified:** `docker compose` itself (no Docker in the sandbox), your real AVIRIS-NG / EnMAP / PaviaU / `.txt` files, Windows.
Limits: no login and no upload-size limit (local/demo server only); images stay on disk under `data/uploads/` and only their description is in the database;
classification uses the decimated image (≤300 000 px) and ≤200 library spectra. `database/schema.sql` and the SQLAlchemy models must stay in step.
