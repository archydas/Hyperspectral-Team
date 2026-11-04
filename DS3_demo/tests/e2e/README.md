# Browser end-to-end checks (optional)

Scripted walk-throughs of the web interface with Playwright. They are not part of `pytest` (they need a running server and a browser).

```powershell
pip install playwright requests ; playwright install chromium
# terminal 1 — a throw-away server on port 8765 with an empty database
$env:DATABASE_URL="sqlite:///e2e.db"; $env:AUTO_IMPORT="0"; cd material_library\backend; python -m uvicorn app.main:app --port 8765
# terminal 2 — run one flow (use a fresh database for each run)
python tests\e2e\run_menu_image_analysis_flow.py       # menus, File > Open image, band list, collect spectra, build library, analyses
python tests\e2e\run_builder_aggregation_flow.py       # Builder dialogs, Viewer filters, two-scene aggregation, export, save
```
`data/` holds two tiny synthetic ENVI scenes (`sceneA`, `sceneB`), a CSV library and a wavelength file. Screenshots are written next to the scripts.
