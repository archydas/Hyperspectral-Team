# One-time setup (Windows, PowerShell, run inside this folder):  .\setup_windows.ps1
# Needs Python 3.12 (py launcher).  Python 3.13+ is too new for some packages (PyQt5, spectral, rasterio).
$ErrorActionPreference = "Stop"
if (-not (Test-Path ".venv")) { py -3.12 -m venv .venv }
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -r requirements_web.txt
.\.venv\Scripts\python.exe -c "import PyQt5.QtWidgets, fastapi, uvicorn, psycopg; print('DS3 setup OK')"
