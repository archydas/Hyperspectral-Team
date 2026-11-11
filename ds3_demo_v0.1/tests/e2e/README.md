# Browser end-to-end checks (optional)

Scripted walk-throughs of the web interface with Playwright. They are not part of `pytest` (they need a running server and a browser).

```powershell
pip install playwright requests ; playwright install chromium
# terminal 1 — a throw-away server on port 8765 with an empty database
$env:DATABASE_URL="sqlite:///e2e.db"; $env:AUTO_IMPORT="0"; cd material_library\backend; python -m uvicorn app.main:app --port 8765
# terminal 2 — run one flow (use a fresh database for each run)
python tests\e2e\run_menu_image_analysis_flow.py       # menus, File > Open image, band list, collect spectra, build library, analyses
python tests\e2e\run_image_collect_flow.py            # pixel popup, collected spectra, ROI table + patches, aggregation, libraries, .roi, Builder raw library
python tests\e2e\run_object_samples_flow.py           # toolbar, wheel zoom + drag, ROI patch folder, per-ROI spectrum, georeference, Library: object libraries + View image
#   the georeference step needs the map script: set LEAFLET_DIST to the folder with leaflet.js and leaflet.css (npm i leaflet) when there is no internet
```
`data/` holds two tiny synthetic ENVI scenes (`sceneA`, `sceneB`), a CSV library and a wavelength file. Screenshots are written next to the scripts.
