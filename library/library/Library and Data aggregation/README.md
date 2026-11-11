# Library Builder and Aggregation Tool (HyperVista_va) — prototype

Companion to the DS3 viewer/analysis tool. Same look and feel, focused on building spectral libraries
from several scenes.

| Function | Where |
|---|---|
| Multi-scene aggregation (common wavelength grid, per-class / per-ROI mean ± std, pooled across scenes) | `hvva/core/resample.py`, `stats.py` — GUI tabs *Wavelength Grid*, *Aggregation* |
| Mosaic (georeferenced or side by side), composite, 4-D stack | `hvva/core/mosaic.py` — tab *Mosaic / Stack* |
| Signature export: CSV, JSON, ENVI spectral library `.sli` (+ `_std.sli`) | `hvva/core/export.py` — tab *Signature Export* |
| **Material spectral library in PostgreSQL** (libraries → datasets → materials → samples → spectra): browse, search, add, import CSV, compare, save aggregated signatures / pixels, spectral-angle matching | `hvva/library/` + `hvva/gui/library_panel.py` — tab *Library* (see below) |
| **ROI Analysis** (moved from the DS3 viewer): rectangle / polygon ROIs on any number of images, ROI Manager across images, class means pooled over images, anomaly map, DS3-ROIs JSON, ROIs saved class by class to the library database | `hvva/core/rois.py` + `hvva/gui/roi_panel.py` — tab *ROI Analysis* (see below) |
| Georeferencing kept with every scene (CRS + affine transform; ENVI map info, GeoTIFF, `.npz`) | `hvva/core/geo.py` — tab *Georeferencing* |
| Plugins (drop a `.py` into `plugins/`) | `hvva/core/plugins.py`, `plugins/` — tab *Plugins* |

## Supported input formats

ENVI (any interleave, `.hdr` + data) - MATLAB `.mat` (v5 and v7.3) - HDF5 (`.h5 .hdf5 .he5 .hdf`) - NetCDF (`.nc`) -
Zarr - TIFF / GeoTIFF (also bands-first) - NumPy (`.npy .npz`) - headerless raw binary (BSQ/BIL/BIP, you give the size).

The tool finds the cube (largest 3-D array), the spectral axis (shortest axis), a class map (2-D integer array,
names like `gt`/`labels`) and a wavelength vector (names like `wavelength`) by itself. In the GUI a dialog shows what
was found and lets you change every choice; headless use the options `--cube-key --band-axis --labels-key --wl-key
--wl-range --raw --labels-file`. Ground truth in a separate file (e.g. `PaviaU_gt.mat`) works in both.
`.mat`/HDF5 need `scipy`/`h5py`; NetCDF4 needs `netCDF4`; Zarr needs `zarr` (each is only required for its format).

**Sensor products (GeoTIFF).** No-data values become gaps; band wavelengths come from the file's metadata or, for
**EnMAP**, from the product's `*-METADATA.XML` (keep it in the same folder as `*-SPECTRAL_IMAGE.TIF`): 224 bands →
209 sorted bands (VNIR/SWIR overlap resolved), L2A reflectance scaled to 0–1, -32768 → gaps. Open the
`SPECTRAL_IMAGE` file — `QL_*` files are RGB quicklooks and `QL_QUALITY_*` are masks; the open dialog warns about
both.

## ROI Analysis (tab *ROI Analysis*)

The DS3 viewer's *ROI Tools* and *ROI Manager*, moved here and extended to several images and the database.

* **Draw.** Tick *Enable ROI selection*, pick *Rectangle (drag)* or *Polygon (click points; click the first point
  to close, Esc restarts)*, type or pick a class, then *Add selection to ROI list* (or tick *Add as soon as it is
  drawn*). The pending selection is dashed yellow; ROIs are outlined in their class colour and numbered.
  Switch images in the *Scenes* tab and keep drawing — one class can have ROIs in many images, several ROIs per
  image. *Show anomaly map* (distance from the image mean spectrum) helps to find rare materials.
* **ROI Manager.** One table for all images (or the current one): image, ROI, class, shape, pixels and the
  libraries it was saved to. Delete / Clear / Set class / Rename; double-click a row to jump to its image.
  *Plot ROI spectra* (mean ± std per ROI) and *Plot class means* (pixels of all ROIs of a class pooled over all
  images, resampled to a common grid if the images differ). *Save / Load ROIs...* use the viewer's DS3-ROIs JSON
  (version 2 adds the image name and ROI id; the viewer opens it, and HyperVista opens the viewer's files).
* **Save ROIs to library database...** stores the selected ROIs, the current image's or all of them, class by
  class: *class mean ± std pooled over all ROIs of the class*, *one mean ± std per ROI*, or *individual pixel
  spectra* (capped per ROI). Each class becomes a material (category e.g. `Field ROIs/vegetation`), its spectra
  are samples, and every ROI is a row of the new `rois` table (class, image, rectangle/polygon pixel geometry, map
  geometry + CRS when georeferenced, bounding box, pixel count). Each spectrum records its ROI class
  (`spectra.roi_class`), its ROI (`spectra.roi_id`) and every ROI it was computed from (`spectrum_rois`). Saving
  the same ROIs again adds nothing; a ROI that was changed is updated and gets a new sample.
* **Load ROIs from library database...** lists the stored ROIs (all libraries or one). ROIs drawn on the current
  image (same name and size) come back with their ids; on another georeferenced image of the same area (same CRS)
  they are placed through their map coordinates. The same dialog deletes ROIs from the database (keeping or
  deleting their spectra).
* The Library tab shows, for a spectrum saved this way, its ROI class and where it came from.
* ROIs are stored with the scene (`.npz`), and the *Aggregation*, *Mosaic* and export tabs use them (polygons
  included) like label-map classes.

An existing database gets the `rois` and `spectrum_rois` tables and the `spectra.roi_id` / `roi_class` columns
the first time HyperVista or the API server connects (`database/migrations/002_rois.sql`); no data is changed.

## Library database (tab *Library*)

The Library tab is connected to the material spectral library database of the DS3 platform (same PostgreSQL
schema and backend logic as `new_hyperspectral/material_library`).

    # 1. database (PostgreSQL 16 in Docker; runs database/schema.sql on first start)
    docker compose -f database/docker-compose.yml up -d
    # 2. load the bundled spectra (idempotent - re-running adds nothing)
    python -m hvva.library.importer data/material_spectra.csv --library "Material Library" --dataset "Material Spectra"
    python -m hvva.library.importer data/soil_spectra.csv --library "Soil Library" --dataset "Soil Spectra" --category "Soil/Mineral" --acquisition lab
    # 3. GUI - connects automatically to $DATABASE_URL (or .env, or the default below)
    python run_gui.py

Default URL: `postgresql://postgres:postgres@localhost:5432/material_library` — change it with *Connect...* in the
Library tab (or *File > Connect library database...*). Two connection modes:

* **Direct** — SQLAlchemy straight to PostgreSQL, no server needed. `sqlite:///my_library.db` also works for
  offline use (tick *create tables if missing*).
* **API** — through the FastAPI server, e.g. when only the server can reach the database:
  `pip install fastapi uvicorn` then `python -m hvva.library.api` (docs at http://127.0.0.1:8000/docs). It serves the
  original backend's endpoints unchanged, so the DS3 web prototype can use the same server.

**Category tree.** `materials.category` holds a path with `/` between levels, e.g.
`Vegetation/Tree/Tsuga/Canadensis` (typing `Vegetation > Tree > Tsuga` works too). The *Category tree* pane shows
the hierarchy with counts; clicking a node lists that category and everything below it; *Rename / move...*
(or right-click) renames a node and moves its whole subtree. Set categories in *Edit...* (several materials at
once), when adding a material, importing a CSV or saving from HyperVista.

**Acquisition type.** Every spectrum is tagged **lab**, **field**, **airborne** or **satellite** (or left
untagged), shown as coloured chips in the materials list. Filter with *Acquisition* in the search bar (also limits
the all-materials search and pixel matching); change a spectrum's tag on the *Spectrum* tab, or all spectra of the
selected materials in *Edit...*; choose it when importing a CSV or saving signatures/pixels.
An existing database gets the new `spectra.acquisition_type` column automatically the first time HyperVista (or the
API server) connects — see `database/migrations/001_category_tree_acquisition.sql`; no data is changed.

**Import any spectral data.** *Import ▸ Spectral data files (any format)...* opens an import wizard. For each
file, folder or zip it shows the detected format, the number of spectra, wavelength range and unit, value scale,
categories and acquisition tags, plus a plot of the first spectra. Every interpretation choice can be overridden:
unit, scale, table layout, columns, category, acquisition, image mode, wavelength range. Supported:

| Family | Formats |
|---|---|
| Text tables | CSV / TSV / TXT / DAT: spectra in columns, in rows (wavelengths in the header, metadata columns such as name, category, acquisition) or long format (name, wavelength, value); any delimiter, decimal comma, comment lines, `key: value` headers |
| Spreadsheets / columnar | Excel `.xlsx` (every sheet), Parquet (wide, long or list columns), JSON (HyperVista export, lists of spectra) |
| ENVI | Spectral libraries `.sli/.hdr` (any data type, byte order, scale factor, µm or nm); images → class means / image mean / sampled pixels |
| Arrays | MAT, HDF5, NetCDF, NumPy, Zarr, GeoTIFF cubes (as images) or spectra matrices with a wavelength vector |
| Field spectrometers | ASD FieldSpec `.asd` (v1–v8, reflectance or DN with white reference), Spectra Vista `.sig`, Spectral Evolution `.sed` |
| Reference libraries | USGS splib07 ASCII release, ECOSTRESS files, OpenSpecLib Parquet |
| Containers | Folders and zips of any of the above; subfolder names become category levels |

Wavelengths are converted to nm (from µm or wavenumber cm⁻¹). Reflectance is scaled to 0–1 (from percent or
×10000), while radiance and counts are kept as measured. Bad values become gaps. Acquisition is inferred from
columns, keys and instrument names (AVIRIS / UAV → airborne, Sentinel / PRISMA → satellite, …). Script equivalent:
`python -m hvva.library.ingest <files or folders> --library NAME [--preview] [--unit … --scale … --orientation …]`.

**Import from a data folder.** *Import ▸ From data folder...* scans a local folder, shows what is new and imports it into (new) libraries in one click, optionally at every start; script: `python -m hvva.library.sources <folder> --import`.

**Reference libraries for testing.** `python -m hvva.library.speclib usgs|ecostress <zip or folder>` (or
*Import ▸* in the tab) loads the USGS Spectral Library v7 ASCII release, ECOSTRESS files or the OpenSpecLib Parquet files (direct GitHub download), with category paths
and acquisition tags filled in. Download links and a demo recipe are in `DATA_SOURCES.md`.

What the tab does: list/filter libraries with material counts · add/delete a library · import a wide CSV
(`Wavelength` column + one column per material) · search materials across all libraries · per-library material
search (server-side, debounced, min 2 characters) with category filter and *See more* paging · add/delete
materials · open a material → choose a sample → spectrum (± std) in the Spectra pane → export CSV · compare up to 8
spectra and export the comparison as CSV/JSON/ENVI `.sli` · **Save to library database** from the *Signature
Export* tab (aggregated mean ± std, pixel count and source scenes are stored) · save the last clicked pixel · match
the last clicked pixel against one or all libraries by spectral angle (on the overlapping wavelengths).

## Run

    pip install -r requirements.txt
    python run_gui.py                     # GUI (File > Load demo scenes for a quick look)
    python -m hvva demo --out demo        # headless: write 3 synthetic scenes
    python -m hvva aggregate demo/*.npz --out sigs --db lib.sqlite   # CLI keeps the simple SQLite store (hvva/library/db.py)
    python -m hvva mosaic demo/*.npz --out mosaic.npz --envi
    python -m hvva plugins
    python -m pytest tests

See `DEVELOPER_GUIDE.md` for the architecture and how to write a plugin. Licence: MIT (`LICENSE`).

Prototype status: synthetic-data tested only; wavelengths are handled in nm; mosaicking needs north-up scenes with
the same CRS and pixel size (no reprojection yet).
