# Task 1 — Spectral sensing software tools for HSI data aggregation and open-framework spectral representation

Owner: Arju Verma · Project: DS3 (Kapih / IIST) · Status date: 30 Oct 2026

## 1. What the item asks for
Hyperspectral data arrives from satellites, airborne/drone sensors, field spectroradiometers and public libraries,
each with its own format, units and metadata. The tool must **read all of them, convert them to one standard
representation, and keep them in one database**, in open Python code.

## 2. What was built
Two Python packages-in-one-repo, as agreed with the team (separate products; shared data contract):

| Part | Package | Role |
|---|---|---|
| Core (no GUI) | `ds3.ingest`, `ds3.spectra`, `ds3.db`, `ds3.pipeline`, `ds3.library` | readers, QC, contract, database, builder, search, match, export |
| Desktop app | `ds3.gui` (`ds3-library`) | PySide6 interface over the core: Import · Search · Cubes → Library · Library Builder · Match · Export · Database |
| Interfaces | `ds3` CLI, FastAPI (`ds3 serve`) | scripting and integration |

**Standard representation (contract v1.0):** cube = float32 `[rows, cols, bands]`, wavelengths in nm ascending, FWHM, bad-band mask
(True = exclude), unit (`reflectance | radiance | dn | unknown`), nodata → NaN, CRS + affine transform. Spectrum = wavelengths + values
+ unit + purity (`ground_measured | image_pixel | synthetic`) + category + metadata. QC flags problems; it never deletes data.

**Database:** `sources → cubes`, `libraries → datasets → samples ← materials`, `samples → spectra`. PostgreSQL (arrays) or SQLite (JSON);
files are identified by SHA-256, so re-importing is idempotent. Cube pixels stay in their files; the database holds metadata, QC and path.

**Supported inputs:** ENVI (BSQ/BIL/BIP) · GeoTIFF (incl. EnMAP: wavelengths from the sibling METADATA.XML) · Zarr · HDF5/NetCDF (EMIT-style, generic) ·
CSV (two layouts) · SVC `.sig` · ASD/SED (via specdal) · ENVI `.sli` · USGS splib07 · ECOSTRESS · ENVI ASCII ROI · GeoJSON.

**Spectral Library Builder (ENVI feature):** target grid from image / ASCII / first spectrum / regular range; linear or Gaussian resampling;
no extrapolation; wide gaps not bridged; one unit per library; rejected spectra reported with reasons; search by text, category, unit,
purity, library and wavelength coverage; SAM / RMSE matching; export to `.sli` and CSV.

## 3. How it was verified (and what that does and does not show)
| Check | Result | Environment |
|---|---|---|
| Automated tests | **93 passed** (readers, QC, ROI, DB, API, builder, EnMAP-style adapter, categories, ROI-to-library, CSV round trip, 10 end-to-end tests driving the app window) | Linux, Python 3.13, offscreen Qt, SQLite |
| Team's real CSV library (`material_spectra.csv`, 100 materials, 350–2500 nm, 1 nm) | imported (100 spectra), categorised, searched, resampled to 400–2400 nm / 10 nm (100 added, 0 rejected), matched | same sandbox |
| Synthetic scenes in ENVI, GeoTIFF, Zarr, NetCDF | all read; ENVI vs GeoTIFF agree to 5e-5 (integer rounding) | same sandbox |
| Large cube (1000×1000×224 int16, 448 MB) | open + full QC 3.5 s; 100×100 ROI 29 ms; Python heap peak 141 MB | same sandbox |

**Not yet verified:** real AVIRIS-NG and real EnMAP files (the EnMAP XML parser and the ×10000 scaling follow the published product layout and were tested
on a synthetic file written to that layout); real PostgreSQL through `ds3`; the app on Windows; the `.exe` build; the `.sli` export opened in real ENVI;
ENVI ASCII ROI pixel base (1-based assumed). `scripts/check_my_data.py` runs these checks on real files in one command and writes a report.

## 4. Known limits
* New product families may need small adapters (NEON scaling, PRISMA VNIR+SWIR merge, MATLAB `.mat`, EMIT swath geolocation are **not written**).
* ENVI `.evf` and binary/XML `.roi` files are unsupported (clear error).
* Category assignment is a keyword rule, not a classifier.
* Matching is O(N) per query; beyond ~10⁵ spectra a cached matrix or pgvector is the next step.
* The app has no image viewer: picking ROIs by clicking on an image belongs to the DS3 viewer; here ROIs are given as pixel/box/ROI file/GeoJSON.
* Instrument-level preprocessing (dark current, white-reference, splice correction) is **not** part of this item's code; the contract carries the flags.

## 5. Run it
See [SETUP.md](SETUP.md) and [USER_GUIDE.md](USER_GUIDE.md). Quick check: `pip install -e ".[dev]"` → `pytest -q` (93 passed) → `ds3-library`.
