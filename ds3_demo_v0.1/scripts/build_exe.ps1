# Build DS3Library.exe on WINDOWS (PowerShell, from the repo root, venv active).
# STATUS: written from PyInstaller's documented options; NOT yet run on Windows. rasterio/h5py/netCDF4 ship
# compiled libraries, which is the usual source of "works from VS Code, fails as .exe" - if the .exe starts and
# then reports a missing module/DLL, send the message and add it to the --collect-all / --hidden-import lines.
python -m pip install --upgrade pip
pip install ".[gui,exe]"            # NON-editable install, so PyInstaller bundles the real package
pyinstaller --noconfirm --clean --windowed --name DS3Library `
  --collect-all rasterio --collect-all h5py --collect-all netCDF4 --collect-all zarr --collect-all numcodecs `
  --collect-submodules sqlalchemy.dialects --hidden-import psycopg --hidden-import psycopg_binary `
  --collect-data matplotlib `
  scripts\ds3_library_launcher.py
Write-Host "Built: dist\DS3Library\DS3Library.exe  (zip the whole dist\DS3Library folder to share it)"
