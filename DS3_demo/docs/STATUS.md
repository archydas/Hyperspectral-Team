# Status — 31 Oct 2026 (honest)

## Works and is tested
Linux sandbox, synthetic data + the team's real CSV library: **109 ds3 tests + 20 web-backend tests**.
The same GUI tests and the same web-backend tests were also run on a **real PostgreSQL 16 server** (all pass), and the
full web demo (upload → image → library overlay → SAM map) was driven in a browser against PostgreSQL with no errors.

- Cube readers: ENVI (3 interleaves, scale, nodata, bbl, map info), GeoTIFF (7 wavelength sources incl. sibling EnMAP METADATA.XML),
  Zarr, EMIT-style HDF5/NetCDF, **MATLAB .mat (v5 and v7.3)**
- Spectra readers: CSV, SVC .sig, ENVI .sli, USGS ASCII, ECOSTRESS, **tolerant text tables (.txt/.csv/.tsv/.dat; 4 layouts)**
- **.zip archives** (EnMAP L2A zips: only SPECTRAL_IMAGE is imported; zip-slip refused); **AVIRIS-NG ancillary files** (glt/igm/loc/obs/h2o) are skipped with a reason
- **Label maps** (PaviaU_gt.mat): one mean spectrum per class → library
- **Desktop app: Image viewer** — open imagery, click a pixel, bring a library in (from the database or any library file), overlay, best matches (spectral angle), classify the image (SAM), save a pixel to a library
- **Web interface (HTML + FastAPI + PostgreSQL/SQLite): Library · Upload (any format) · Image** with the same library-in-image workflow
- Database models + idempotent ingest, REST API, CLI, Library Builder, search, SAM/RMSE match, `.sli`/CSV export

## NOT verified
- **Real AVIRIS-NG, EnMAP, PaviaU.mat and your real .txt libraries.** The `.txt` reader recognises four layouts (see `spectra/tabular.py`) and was
  tested on synthetic files in each layout; if one of your files has another layout it is refused with a description of what was seen — send that message.
  PaviaU wavelengths are NOT in the .mat file: 430–860 nm evenly spaced is assumed (shown in the app), or type your own range.
- **`docker compose`** itself (no Docker in the sandbox). The identical schema/app was run on a native PostgreSQL 16 server.
- **Windows** run of the new features and the **.exe build** (`scripts/build_exe.ps1` never run).
- The `.sli` export opened in real ENVI; ENVI ASCII ROI pixel base (1-based assumed); ASD/SED binary.
- Large cubes: the viewer decimates for display and reads in row chunks, but a 1 GB+ cube was not timed.

## Known limits
- SAM classification of the image uses at most 200 library spectra and the decimated image (≤300 000 px shown); it is a library-matching aid, not a validated classifier.
- If image and library units differ (image DN/radiance vs library reflectance) the app says so; spectral angle ignores gain but not illumination shape.
- Web upload: no size limit or login — for a local/demo server only.
- Two database schemas exist: the desktop app (integer ids) and the web backend (UUID). Same readers and same library format; unifying them is a next step.

## Not written
PRISMA VNIR+SWIR merge · NEON scale/no-data · EMIT swath geolocation · `.evf` / binary `.roi`.

## Next
1. Run on your real files (`scripts/check_my_data.py`, or just open them in the app) and send any error text. 2. Windows `.exe` build. 3. One shared schema for desktop + web.
