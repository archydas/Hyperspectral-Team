# Developer guide

## Layout

    hvva/core/      algorithms, no GUI, no database      (numpy only)
      scene.py      Scene dataclass; load_scene(); save .npz / ENVI
      quality.py    bad bands (water windows, bbl, empty), value units, purity, zip products, ancillary files
      rois.py       ROI geometry (rect / polygon / pixels), masks, pooling across images, anomaly map, DS3-ROIs JSON,
                    map-coordinate transfer, records for the library database (incl. image_info)
      roifiles.py   GeoJSON (pixel / map / lon-lat) and ENVI ASCII ROI read / write
      display.py    band selection for the scene view: gray / RGB, presets, percentile stretch (Scene.display)
      basemap.py    CRS <-> lon/lat (pyproj, rasterio or built-in UTM), web-mercator tiles (cached), scene_affine()
                    for drawing a scene on the map, affine_from_corners() / affine_from_gcps() for georeferencing
      loaders.py    universal format loaders (.mat, HDF5, NetCDF, Zarr, TIFF, NumPy, raw) + guessing
      geo.py        GeoInfo: CRS + affine transform, ENVI map info <-> transform
      resample.py   common_grid(), resample_spectra() / resample_cube()
      stats.py      aggregate_classes(): streaming, exact pooled mean/std per band
      mosaic.py     mosaic(), stack_cube(), composite()
      signature.py  Signature dataclass, spectral angle
      export.py     write_csv / write_json / write_sli / read_sli
      plugins.py    plugin discovery
    hvva/library/   spectral library storage (depends on core only, never Qt)
      database.py   Database: engine + sessions for a URL (PostgreSQL or SQLite), init_schema() from schema.sql
      models.py     SQLAlchemy models = database/schema.sql (libraries, datasets, materials, samples, spectra, data_files)
      repository.py all queries/commands, returning API-shaped dicts (shared by GUI and API server)
      importer.py   wide-CSV importer (python -m hvva.library.importer)
      speclib.py    USGS splib07 ASCII / ECOSTRESS / OpenSpecLib importers (python -m hvva.library.speclib)
      sources.py    data-folder scanner + batch import + status (python -m hvva.library.sources)
      ingest.py     the one import path: detect -> read -> normalise -> write (python -m hvva.library.ingest)
      formats/      universal readers (no Qt, no database)
        base.py         SpectrumRecord, ReadOptions, registry, detect(), read_path(), normalize(), inference
        text.py         table engine (columns / rows / long) for CSV/TXT, Excel, Parquet; JSON
        binary.py       ASD / SVC / SED, ENVI spectral library, image cubes, spectra matrices in array files
        refcollections.py  USGS splib07, OpenSpecLib, folder, zip
      builder.py    Spectral Library Builder: output grids, linear / Gaussian resampling, build() with rejection
                    reasons, whole-library export (.sli / CSV / JSON)
      client.py     DirectClient (SQLAlchemy) / ApiClient (HTTP) - one interface for the GUI
      api/          FastAPI server (python -m hvva.library.api), backend routes + HyperVista additions
      schema.sql    copy of database/schema.sql used by init_schema()
      db.py         legacy single-file SQLite signature store (used by the CLI --db option and tests)
    hvva/gui/       PyQt5 window, theme and canvases (depends on core + library)
      library_panel.py  the Library tab (async via QThreadPool; never blocks the window)
      roi_panel.py      the ROI Analysis tab (ROI tools + ROI Manager; database work through the Library tab's client)
      viewer.py         controls under the scene view (bands, zoom toolbar, pixel window, collect, map) +
                        the collected-spectra dialog
      library_dialogs.py  Library Builder, Resample, Export library, Library images dialogs
      canvases.py       ImageCanvas (image or map mode: pix_transform / event_pixel for overlays and clicks,
                        wheel zoom, view kept on redraw) and PlotCanvas
    database/       docker-compose.yml + schema.sql for the PostgreSQL server
    data/           sample spectra CSVs (material_spectra.csv, soil_spectra.csv)
    hvva/cli.py     headless entry point:  python -m hvva <command>
    plugins/        user methods
    tests/          pytest unit tests

Rule: `core/` never imports Qt or sqlite; `library/` never imports Qt. That keeps the CLI, tests and any
future server usable without a display.

## Library database data model
`Library 1-n Dataset`, `Library 1-n Material 1-n Sample 1-n Spectrum`, every Sample also belongs to a Dataset,
`Dataset 1-n DataFile` (provenance of imported files with SHA-256). A spectrum stores `wavelength_start/end/step`
and `values DOUBLE PRECISION[]`; HyperVista adds, inside the spectrum's JSON `metadata`: `std` (aggregated
signatures), `n_pixels`, `source_scenes`, and `wavelengths` when the grid is not regular (then `wavelength_step`
is NULL). NaN is stored as NULL. Saving a spectrum under an existing material name adds a new sample to it.
Category tree: `materials.category` is a `/`-separated path (no extra table, so the original backend keeps
working); `repository.normalize_category` canonicalises input, `library_facets` returns paths + counts for the tree,
`list_materials(category_path=...)` matches the node and its descendants (`category = p OR category LIKE 'p/%'`,
wildcards escaped), `rename_category` rewrites a subtree. Acquisition: `spectra.acquisition_type` in
(lab, field, airborne, satellite) or NULL; `list_materials(acquisition=...)` keeps materials having such a spectrum;
`'__none__'` means uncategorised / untagged in both filters. Schema changes go in `database/migrations/` (idempotent)
and are applied by `Database.migrate()` on connect; `database/schema.sql` already contains them for new installs.
ROIs (migration 002): `rois` (one row per ROI: `library_id`, `roi_uid` = HyperVista's ROI id, unique per
library, `class_name`, `roi_name`, `scene_name/path`, `image_rows/cols`, `geometry_type` rect|polygon, `geometry`
JSONB in pixels — rect `[r0, c0, r1, c1]` end-exclusive, polygon `[[col, row], ...]` pixel centres —,
`geometry_map` + `crs`, `bbox_*`, `n_pixels`, `metadata`), `spectra.roi_id` (FK, SET NULL; set when the spectrum
comes from one ROI), `spectra.roi_class`, and `spectrum_rois(spectrum_id, roi_id)` (all source ROIs; a pooled class
mean links to every ROI of its class). `repository.save_roi_spectra` writes ROIs + spectra in one transaction and
skips spectra whose `metadata.uid` is already in the library; `list_rois`, `roi_classes`, `roi_spectra`,
`spectrum_roi_info`, `delete_roi` read and manage them (`/api/libraries/{id}/rois`, `/roi-classes`,
`/roi-spectra`, `/api/rois/{id}`). `core.rois.library_items(items, mode)` builds the payload from scenes.
Scene ROI dicts: `{id, name, class, kind: rect|polygon, rect (r0, c0, r1, c1) | vertices [(x, y)]}`; old
`{class, rect}` dicts are upgraded on load. Mask rules match the viewer (pixel centre inside the polygon).
Data quality (v3.3): `Scene.bad_bands` = file `bbl` | water windows | empty bands, computed on demand (flags,
the cube is untouched); use `scene.spectrum_at()`, `scene.pixels(mask)` or `scene.masked(X)` for every spectrum
taken from a scene. `Scene.unit` / `unit_note` / `synthetic`; `quality.common_unit()` refuses to pool two known
units. `repository.add_spectrum(unit=, purity=)` calls `check_unit()` (one unit per library, recorded in
`libraries.metadata.unit`, `UnitMismatch` otherwise); `spectra.purity` (migration 003). `SpectrumRecord.unit /
purity` are filled by `formats.normalize()` (option > reader > implied by the value scale). `quality.resolve_input()`
turns a product zip into the image inside and raises for ancillary rasters; `load_scene()` and the Scenes tab call it.
Adding a file format: subclass `formats.base.Reader` in one of the formats/ modules and decorate it with
`@register`. Set `name`, `label` and `extensions`; implement `sniff(path)` (0 = no, 1 = certain) and
`read(path, opts)`, yielding `SpectrumRecord`s with wavelengths and values as found, plus hints (`wl_unit`,
`scale`, `quantity`, `category`, `acquisition`) and a source-unique `spectrum_metadata["uid"]`. That's all:
`normalize()` converts units and scale, and the import wizard, data-folder scan, CLI and API pick the reader up
automatically. Add a fixture to `tests/spectral_fixtures.py` and a case to `tests/test_formats.py`.

v3.4 (migration 004): `images` (one row per image, unique `fingerprint` = sha256 of file size + first / last MiB,
or of a cube sample for in-memory scenes; rows / cols / bands, unit, sensor, wavelengths, bad-band mask, CRS,
transform, metadata - never pixels), `rois.image_id` (FK, SET NULL; `upsert_roi` registers the record's `image`
and links it), `rois.geometry_type` may be `pixels` (`geometry = {"pixels": [[row, col], ...]}`), and
`spectra.wavelengths` (explicit wavelength of every value, always written by `add_spectrum`; filled for older rows
from `metadata.wavelengths` or start + i * step; `repository.spectrum_wavelengths()` reads column -> metadata ->
grid). `repository`: `register_image / list_images / get_image / delete_image`, `resample_library` (new library),
`match_rank(method="sam"|"rmse")`, and `unit / purity / covers` filters on `list_materials`,
`search_materials_global` and `library_spectra` (`covers` = (lo, hi) nm, spectra spanning it).

Scene view: overlays drawn on `ImageCanvas` must use `transform=canvas.pix_transform` with pixel-centre
coordinates (x = col, y = row) and clicks must use `canvas.event_pixel(ev)` - then they work in image and map
mode alike. Map tiles: `basemap.fetch_tile` is the only network call (mock it in tests).

To add a feature: write it in `repository.py`, expose it in `client.py` (both clients) and, if it should be
reachable over HTTP, in `api/main.py`; the GUI only ever calls the client.

## Conventions
* Cubes are `(rows, cols, bands)` float32. Wavelengths are nm (ENVI micrometres are converted on load).
* Georeferencing uses the rasterio convention `x = a*col + b*row + c`, `y = d*col + e*row + f`.
* Classes are matched across scenes **by name**. Label maps use 0 for unlabelled.
* Standard deviation is the sample std (n - 1).

## Writing a plugin
Create `plugins/my_method.py`:

    def run(cube, wavelengths, strength=1.0):
        return cube * strength            # correction: return a cube with the same rows/cols

    PLUGIN = {
        "name": "My method",
        "kind": "correction",             # or "analysis"
        "description": "One line shown in the GUI.",
        "params": {"strength": {"type": "float", "default": 1.0, "min": 0, "max": 10}},
        "run": run,
    }

* `correction` returns a cube (or `(cube, new_wavelengths)` if the band count changes) and becomes a new scene.
* `analysis` returns a 2-D image or `{"image": ..., "table": {...}, "plot": (x, y)}`.
* Parameter types: `int`, `float`, `bool`, `str`. The GUI builds the form from `params`.
* A plugin that raises on import is reported in the Plugins tab and skipped; it cannot crash the app.
* Test it headless: `python -m hvva run-plugin "My method" scene.npz --param strength=2 --out out.npz`.

## Extending
* New file format: add a `_from_xxx(path)` function to `core/loaders.py` that returns a `Source` (array names/shapes + a reader) and register its extension in `open_source`; guessing, the GUI dialog and the CLI options then work for it automatically.
* New export format: add a function to `core/export.py` and register it in `export_all`.
* New GUI tab: add a `_tab_xxx` method in `gui/main_window.py` returning `_pair(group, group)`.

## Contributing / licence
MIT licensed. Run `python -m pytest tests` before sending changes.
