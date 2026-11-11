# Architecture

```
src/ds3/
  ingest/   readers/{envi,geotiff,zarr_io,hdf}.py  → Cube     (lazy, memmap/windowed)
            metadata.py (wavelength→nm, sorting, bad bands)  qc.py (streaming)  roi.py
  spectra/  readers.py  csv|svc|asd/sed|envi_sli|usgs|ecostress → list[Spectrum]
  db/       models.py   session.py                           (SQLAlchemy 2)
  library/  grid.py sources.py builder.py search.py export.py   ENVI-style builder + search
  pipeline.py           classify → read → store, sha256 dedupe
  api/app.py            FastAPI        cli.py  argparse      sampledata.py  synthetic files
```

## Key decisions
- **Cube pixels stay in files; the DB stores metadata + QC + path.** Putting 100s of MB of pixels in Postgres
  rows is slow and pointless; the API reads windows from the file on demand. Convert to Zarr for cloud/chunked use.
- **Point spectra are stored in the DB** as `DOUBLE PRECISION[]` (Postgres) / JSON (SQLite). NaN → NULL.
- **One schema, two engines:** PostgreSQL for production; SQLite for zero-setup dev/tests. DDL is generated
  from the models (`ds3 export-schema`) so there is no hand-written schema to drift.
- **Idempotent ingest:** sha256 of file(s) → re-running never duplicates.
- **Provenance:** `sources(kind=satellite|airborne|drone|field|library)`; kind inferred from folder name.
- **Adding a format:** write `read_xxx(path, **kw) -> Cube | list[Spectrum]`, register it in
  `ingest/api.py::_READERS` or `spectra/readers.py::_READERS`, add detection + a test. Nothing else changes.

## Schema
`sources` → `cubes` · `sources` → `libraries` → `datasets` → `samples` ← `materials`; `samples` → `spectra`.
See `docs/schema_postgres.sql`.
