# DS3 Library — user guide

DS3 Library is the desktop app for the spectral library: it **imports** data from many formats into one database,
lets you **search** it, **builds** libraries on a common wavelength grid, **matches** spectra, and **exports** them.
It is a separate product from the DS3 viewer (preprocessing); the two share only the data contract
([CONTRACT.md](CONTRACT.md)).

## Start it
```bash
pip install -e ".[gui]"        # once (adds PySide6 + matplotlib)
ds3-library                    # or:  python -m ds3.gui
ds3-library --db "postgresql+psycopg://user:pass@localhost:5432/ds3"   # optional: use PostgreSQL
```
Default database: a local SQLite file `~/.ds3/ds3_library.db` (nothing to install). The last database you chose is remembered.

## The window
* **Top left — Library table:** the current search results (100 per page). Select rows to plot them.
* **Top right — plot:** selected spectra overlaid; grey bands are the water-absorption windows (1350–1450, 1800–1950 nm).
  Below it: details of what is selected.
* **Bottom — tabs:** Import · Search · Cubes → Library · Library Builder · Match · Export · Database.

![Search](screenshots/1_search.png)

## Workflows
### 1. Import real data (Import tab)
1. *File…* or *Folder…* → choose a file (`.hdr`, `.tif`, `.h5`, `.nc`, `.zarr`, `.csv`, `.sig`, `.sli`, `.spectrum.txt`) or a whole folder.
2. *Source kind*: leave on **auto** if the file sits in a folder named `satellite`, `airborne`, `drone`, `field` or `library`; otherwise choose it.
3. **Import into database.** The log shows one line per file: `ingested`, `duplicate` (already there — safe to re-run), `skipped`, or `error` with the reason.

| Data | What to select | Notes |
|---|---|---|
| AVIRIS-NG | the `.hdr` file (the binary sits next to it) | wavelengths, scale and nodata come from the header |
| EnMAP L2A | the `…-SPECTRAL_IMAGE.TIF` | keep `…-METADATA.XML` in the same folder (band centres are read from it). The file name is recognised and the int16 ×10000 reflectance scaling is applied; check the unit/values in the log |
| EnMAP without the XML | the TIF + **Wavelengths file** (a `.txt/.csv` list or the XML) | wavelengths are remembered with the cube |
| CSV library | the `.csv` | wavelengths down the first column, one spectrum per column — or wavelengths as the header row, one spectrum per row |
| Wrong scale | **Reflectance divisor** (e.g. `10000`) and **Unit** | only if the log/values show it is needed |

Materials get a **category** from a keyword rule on their name (concrete → construction, grass → vegetation, …). It is a heuristic,
labelled `category_source: keyword_rule` in the data, and is shown as `—` when nothing matches.

### 2. Search (Search tab)
Text (material / category / library), category, unit, purity (`ground_measured`, `image_pixel`, `synthetic`), library,
and **Must cover** a wavelength window (only spectra that span the whole window). Results page with Prev / Next.

### 3. Image → library (Cubes → Library tab)
Select a registered cube, choose *Single pixel*, *Box* (end row/col exclusive) or an *ROI file* (ENVI ASCII export or GeoJSON),
name it, **Preview** (mean ± std is drawn; nothing is saved), then **Save to library**. Entries are tagged `image_pixel`.
Pixels that are nodata in every band are excluded and counted. Saving the same ROI twice is refused.

![Cubes](screenshots/4_cubes.png)

### 4. Library Builder (ENVI Spectral Library Builder equivalent)
1. Name the library; choose the **wavelength grid**: first input spectrum · an image/cube file · an ASCII wavelength file · a regular range.
2. Choose resampling (**linear** or **gaussian**), unit, minimum coverage (default 50 %).
3. Add inputs: spectra files, ENVI ASCII ROI files, GeoJSON (these need *Cube for ROI inputs*) and/or tick *Include rows selected in the table*.
4. **Build library.** The report lists every spectrum added and every spectrum **rejected, with the reason**
   (grid not covered, different unit — radiance and reflectance are never mixed or converted).
   Resampling never extrapolates and does not bridge gaps wider than 3× the median band spacing.

![Builder](screenshots/2_builder.png)

### 5. Match
Pick a row in the table (or a spectrum file), choose SAM (angle, shown in degrees) or RMSE, Top N, unit filter (default: same unit as the query).
Stored spectra are resampled onto the query's wavelengths before comparing. Click a result to overlay it on the query.
Matching reads candidates in chunks (memory stays flat) and is O(N) per query — fine to roughly 10⁵ spectra.

![Match](screenshots/3_match.png)

### 6. Export
ENVI `.sli` + `.hdr` (opens in ENVI's Spectral Library Viewer — not yet checked in real ENVI) or CSV (same layout as the team's `material_spectra.csv`).
Both need a library on **one** wavelength grid; otherwise the app says so — build it first with the Library Builder.

### 7. Database tab
Switch to another SQLite file or a PostgreSQL URL, **delete a library**, or **reset** the database (asks for confirmation).
To replace dummy libraries with real ones: *Delete library* for each dummy one, then import your real files.

### 8. Image viewer (bring a library into an image)
Top tab **Image viewer**.
1. **Open image…** — choose a file: AVIRIS-NG `.hdr` (the `*_corr_*_img` / `*_rdn_*_img` file; geometry files such as `_glt/_igm/_loc/_obs` and `_h2o` are skipped automatically),
   an EnMAP `.zip` (or the `…-SPECTRAL_IMAGE.TIF`), `PaviaU.mat`, a GeoTIFF, HDF5/NetCDF. It is added to the database and shown as a colour composite.
   *Files with no wavelengths* (.mat): the program asks for the range when the file cannot be opened, e.g. `430-860`. Known benchmark scenes (PaviaU) get their nominal range **and the app tells you it was assumed**.
2. **Click the image** — the spectrum of that pixel appears (window 1 px / 3×3 / 5×5 / 9×9). Bad bands are blanked.
3. **Library in this view** — choose a library from the database, or **Bring in library file…** (`.txt`, `.csv`, `.sli`, `.sig`): it is read, stored in the database and listed.
4. Tick/select spectra → they are overlaid on the pixel spectrum.
   **Best matches for this pixel** ranks the library by spectral angle. **Classify image (SAM)** colours every pixel by its best library spectrum
   (select 2+ spectra, or none to use the whole library; *Max angle* leaves poor matches unclassified).
5. **Save pixel to library** stores the pixel spectrum (purity `image_pixel`).
If the image is DN/radiance and the library is reflectance the app warns: shapes are comparable, absolute values are not (tick *ignore unit differences* to allow it).

### 9. Your libraries are there when the app opens (Database tab)
- **Libraries folder** (default `C:\Users\<you>\.ds3\libraries`; change it with *Browse…*): copy your library files there (`material_spectra.txt`, `soil_spectra.txt`, `hsdos_lib_spectra.txt`, `samples_spectra_with_metadata.txt`, `ecostress_library_all_spectra.txt`, …).
  Every file becomes **its own library**, named after the file. Press **Load now** once; with *Load this folder every time the app starts* ticked they are loaded at every start (files already in the database are recognised and skipped, so nothing is duplicated). Results and any unreadable file (with the reason) are listed in the **Import** tab log.
- **Docker PostgreSQL:** in the repository root run `docker compose up -d`, then Database tab → **Docker PostgreSQL (default)** (user/password/database `ds3`, port 5432). The choice is remembered; the folder libraries load into PostgreSQL. SQLite file is used otherwise.
- The database, not the folder, holds the libraries: use *Import* / *Bring in library file…* for one-off files.

## Troubleshooting
| Message | Meaning / fix |
|---|---|
| `no wavelengths found … sibling METADATA.XML` | put the XML next to the TIF, or give a *Wavelengths file* |
| `wavelengths= has N values but file has M bands` | wrong wavelength file for this image |
| `UNIT_SUSPECT` / values > 1 for reflectance | set *Reflectance divisor* (e.g. 10000) |
| `library … mixes wavelength grids` (export) | build a library on one grid first |
| `only x % of the grid is covered` (builder) | the spectrum does not span your grid; widen its range or lower *Min. coverage* |
| `already saved` / `already built` | same inputs were processed before; nothing changed |
