# Run it and check it — step by step (Windows, VS Code)

## 1. Open and install (once)
1. Unzip `ds3-platform-v6.zip` to a short path, e.g. `D:\work\ds3-platform-v6`.
2. VS Code → **File → Open Folder…** → that folder (you should see `run_gui.py`, `src`, `material_library`).
3. **Terminal → New Terminal**, then:
```
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```
If `pip` fails on one optional package (`pyarrow`, `pywebview`), delete that line from `requirements.txt` and run it again.
Python 3.11 or 3.12 is recommended (`python --version`).

## 2. Put your libraries in place
Copy your five `.txt` libraries into `material_library\data\libraries\`.

## 3. Start
```
python run_gui.py
```
A window opens (or your browser, at http://127.0.0.1:8000/app/). The first start reads the libraries into the database; the large ECOSTRESS file takes a few seconds, press **Refresh**.
The terminal prints the database in use and the data folder. Stop with Ctrl+C.

## 4. Checklist (tick each one)
| # | Do this | You should see |
|---|---|---|
| 1 | Start the app | full-screen light window with a menu bar **File · View · Tools · Spectral · Help**, left list **Library · Image · Analysis** (no Dashboard tab) |
| 2 | Click **DS3** (top left) | the three modules as cards |
| 3 | **Library** page | your five library cards (the `.txt` files from `data/libraries/`), bottom bar “5 libraries · N spectra”; restart → no duplicates |
| 4 | **Spectral ▸ Spectral Libraries ▸ Viewer** | the Viewer opens directly (no tabs, no “Library function” page); choose a library, use the category and acquisition filters, tick 2–3 materials |
| 5 | **Spectral ▸ Spectral Libraries ▸ Resampling**: library → range 400–2500 step 10, name it, run | new library on the Library page |
| 6 | **Spectral ▸ Spectral Libraries ▸ Builder**: Data File ▸ pick an image or library ▸ OK (try *Spectral Subset* and *Open ▾*) | “N bands … from …” above the table |
| 7 | Builder ▸ **Open ▾** ▸ spectra file(s) / library / data file / ASCII / open image (ROI or pixel); type a library name ▸ Save as library | spectra in the table; the library appears on the Library page; for spectra files the originals are also saved as `<name> (raw)` (tick box) |
| 8 | **File ▸ Open Image File** → choose your AVIRIS-NG `.hdr` (or EnMAP `.zip`, `PaviaU.mat`, GeoTIFF) | the **Available Bands List** (all bands with wavelengths) |
| 9 | In the list click a band for **R**, **G**, **B** (or choose *Gray Scale*), press **Load RGB** | the Image module opens with that RGB composite |
| 10 | Image: click a pixel, type a label, press **Add** (repeat 4–5 times) | a small window shows each spectrum; after *Add* numbered markers appear on the image and the spectrum is listed under *Collected spectra* and plotted |
| 11 | *Select all / Unselect all*, click a name, *Compare angles*, then *Save to library…* (new name or an existing library) | angle table; the library appears on the Library page; *Delete all* needs a second click |
| 12 | **Spectral ▸ Spectral Analyst** (library + current spectrum), **Mapping Methods ▸ K-Means / SAM / Unmixing**, **Transforms ▸ PCA / Band Math**, **Spectral Indices ▸ NDVI**, **Anomaly**, **Signal-to-Noise** | a result window with image, statistics and plots; class maps can be overlaid on the image; k-means class spectra can be saved as a library |
| 13 | Image: *Draw ROI polygon*, click points, **double-click (or right-click) to finish**, give label and class, **Add** | a small window shows the mean spectrum; after *Add* the program asks **where to save the image patches** (once). Choose a folder: the ROI appears in the full-width table *Object sample collection (ROI)* under the image with the patch file `<class>.tif` (open the folder to see it) |
| 14 | Turn the mouse wheel over the image; drag it; press **Fit**; tick / untick **Pixel spectra** and **Object samples (ROI)** (they are in the line with *Download image*) | zoom around the pointer, the image moves, it never picks a pixel after a drag; the chart and the markers follow the tick boxes |
| 15 | **📍 Georeference**: click a recognisable point on the image (left), then the same point on the Esri map (right); repeat for 4 points; **Apply georeference** (needs internet for the map) | a table with the error of each point and the RMSE; afterwards the button shows ✓, *View map* works, and the ROI table shows latitude / longitude |
| 16 | In the ROI table: **Spectrum** (one ROI), **Export** (downloads the patch), **Locate** (opens the folder); tick several ROIs of different classes; **Aggregate ticked ROIs**; switch *Show*: Mean / Std. deviation / Variance / Min–Max | surface-reflectance plot with ± std; all spectra together; table with mean, std, variance per class |
| 17 | **Save to library…** in the ROI table (new name), then open **Library** | the library is listed under *Object-based libraries (ROI)*; pixel spectra saved from the Image page are under *Pixel-based libraries* |
| 18 | Open the object library | a table with the patch file name only, and **Load image** (opens the patch in the Image module from where it was saved), **Locate**, **Spectrum**. Move the file and reopen the library: it says *not found* |
| 19 | In the aggregate: *Export CSV*, *Save class means to library…*. Click an ROI (outline or number) on the image: *Delete ROI* | files downloaded; the aggregated class signatures appear as a library |

Big scenes: in the desktop window (`python run_gui.py`) **File ▸ Open Image File** shows the system file dialog and opens the file where it is
(nothing is copied). In a browser tab the file is uploaded into `data/uploads/` first.

## 5. Automated tests
```
pytest tests
cd material_library\backend
pytest tests
```
Expected: 89 passed (2 skipped) and 34 passed.

## 6. Command line
```
python run_cli.py formats
python run_cli.py import D:\data\soil_spectra.txt --library Soil --acquisition lab
python run_cli.py import D:\data\scene.hdr
python run_cli.py list
python run_cli.py aggregate rois.json --out results
```
`rois.json` layout: see `material_library\backend\app\cli.py` (top of `cmd_aggregate`).

## 7. PostgreSQL with Docker (recommended for the real database)
Full step-by-step with checks: **docs/POSTGRES_DOCKER.md**. Short version (Windows, Docker Desktop running):
```
docker compose up -d db                  :: starts PostgreSQL 16 (user ds3, password ds3, database ds3, port 5432)
copy .env.postgres.example .env          :: makes the program use it
python run_gui.py                        :: the console must say  database: postgresql+psycopg://ds3:****@localhost:5432/ds3
```
Your five main library files go in `material_library\data\libraries\` before the first start; they are loaded into PostgreSQL automatically and appear under **Library ▸ Main libraries**.

## 8. Build the .exe (on Windows)
```
build_exe.bat
```
Result: `dist\DS3Library\DS3Library.exe`. Give the client the whole `dist\DS3Library` folder. Their main library files go into `dist\DS3Library\data\libraries`;
for PostgreSQL put a `.env` file (copy of `.env.postgres.example`, with their own server address) next to `DS3Library.exe`.
If the build stops, send me the last lines of the output.

## If something fails
| Message | Fix |
|---|---|
| `python` not found | use `py -3.11`, or install Python and tick “Add to PATH” |
| `No module named …` | the environment is not active (`.venv\Scripts\activate`) or `pip install -r requirements.txt` was run elsewhere |
| Library page says “No main library files found” | copy the five files into the folder named on that page, press **Reload main libraries** |
| Library page shows a red line | read it: it names the file that could not be read and why |
| Map is grey | the map background needs internet |
| Port in use | the launcher takes the next free port; the address is printed in the terminal |
