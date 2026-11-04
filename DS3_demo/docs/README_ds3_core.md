# DS3 — Deep Spectral Signature System · Task 1: Data Aggregation

Reads hyperspectral data from **satellites, airborne/drone sensors, field spectroradiometers and public
spectral libraries** — each with its own format, units and metadata — converts everything to **one
standard form** (the [output contract](docs/CONTRACT.md)), and stores it in **one database** (PostgreSQL).

```
 raw files ──► readers ──► standard form ──► QC ──► database ──► CLI / REST API
 .hdr/.img .tif .nc .h5     Cube / Spectrum    flags    PostgreSQL     ds3 ...   /docs
 .zarr .csv .sig .sli ...   (nm, float32, NaN)  (never   (SQLite for
                                                deletes)  zero-setup)
```

## Quick start (5 minutes, no database needed)

```bash
python -m venv .venv && .venv\Scripts\activate        # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -e ".[dev]"         # [dev] includes the desktop-app dependencies (PySide6, matplotlib)
pytest -q                       # 93 tests
ds3 sample-data                 # writes synthetic files in every format to data/sample/
ds3 ingest data/sample          # reads all of them into ds3.db (SQLite)
ds3-library                     # desktop app (Import / Search / Builder / Match / Export)
ds3 serve                       # REST API + docs at http://127.0.0.1:8000/docs
```
Check your own files: `python scripts/check_my_data.py --aviris <.hdr> --enmap <...SPECTRAL_IMAGE.TIF> --csv <.csv>`.
App guide: [docs/USER_GUIDE.md](docs/USER_GUIDE.md) · Task report: [docs/TASK1_REPORT.md](docs/TASK1_REPORT.md).
Switch to PostgreSQL: see [docs/SETUP.md](docs/SETUP.md#5-postgresql-with-docker).

## ENVI-style library builder + search
```bash
ds3 library build demo --grid ascii:data/sample/builder/grid_wavelengths.txt \
  --cube data/sample/airborne/synthetic_scene_bsq.img.hdr \
  --inputs data/sample/builder/rois.txt data/sample/builder/fine_grid.csv
ds3 library search --text veg        # also: --unit --purity --library --covers 450 2400
ds3 library match --spectrum-id 3    # most similar spectra (SAM)
ds3 library export demo out/demo.sli # open in ENVI's Spectral Library Viewer
```
Details and ENVI-step mapping: [docs/LIBRARY_BUILDER.md](docs/LIBRARY_BUILDER.md).

## Web interface (HTML + FastAPI + PostgreSQL)
`material_library/` — library browser, upload of any format, and an Image module with library overlay. See `material_library/README.md`.

## Supported inputs
| Kind | Formats |
|---|---|
| Image cubes | ENVI (BSQ/BIL/BIP) · GeoTIFF · Zarr · HDF5/NetCDF (EMIT-style + generic) |
| Field spectra | CSV (wide / row-wise) · SVC `.sig` · ASD `.asd` / SED `.sed` (needs `pip install -e ".[field]"`) |
| Libraries | ENVI `.sli` · USGS splib07 ASCII · ECOSTRESS `.spectrum.txt` |
| Builder-only inputs | ENVI ASCII ROI export · GeoJSON polygons (measured from an image) |

`ds3 formats` prints the live list. **Honest scope:** a new *product family* (NEON, EnMAP, PRISMA, MATLAB `.mat`…)
may need a small adapter — see [docs/STATUS.md](docs/STATUS.md) for what is verified and what is not.

## Repo map
| Path | What |
|---|---|
| `src/ds3/ingest/` | cube readers, QC, ROI extraction, the output contract |
| `src/ds3/spectra/` | point-spectra / library readers |
| `src/ds3/library/` | ENVI-style builder: grid, resampling, ROI inputs, search, `.sli` export |
| `src/ds3/db/` | SQLAlchemy models + session (Postgres & SQLite) |
| `src/ds3/pipeline.py` | file/folder → reader → DB, idempotent (sha256) |
| `src/ds3/api/app.py` | FastAPI service |
| `src/ds3/gui/` | DS3 Library desktop app (PySide6) — separate product from the viewer |
| `src/ds3/cli.py` | `ds3` command |
| `docs/` | setup, architecture, contract, data links, status |
| `tests/` | pytest suite (93) |
| `scripts/` | `check_my_data.py` (verify real files) · `build_exe.ps1` (Windows .exe, untested) |
| `data/` | `raw/` (yours, git-ignored) · `sample/` (generated) |

## Docs
- [docs/FORMATS.md](docs/FORMATS.md) — what each of your file types goes through
[USER_GUIDE](docs/USER_GUIDE.md) · [TASK1_REPORT](docs/TASK1_REPORT.md) · [SETUP](docs/SETUP.md) · [ARCHITECTURE](docs/ARCHITECTURE.md) · [CONTRACT](docs/CONTRACT.md) ·
[LIBRARY_BUILDER](docs/LIBRARY_BUILDER.md) · [DATA_SOURCES](docs/DATA_SOURCES.md) · [STATUS](docs/STATUS.md) · [CONTRIBUTING](CONTRIBUTING.md)
