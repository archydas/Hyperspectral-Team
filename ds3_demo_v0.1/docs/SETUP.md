# Setup guide (Windows-first; Linux/macOS notes inline)

## 1. Install the tools
| Tool | Why | Check |
|---|---|---|
| Python 3.11 – 3.14 (python.org, tick "Add to PATH") | runtime | `python --version` |
| Git (git-scm.com) | version control | `git --version` |
| VS Code (code.visualstudio.com) | editor | — |
| Docker Desktop *(only for PostgreSQL)* | database | `docker --version` |

## 2. Get the code
```bash
git clone <repo-url> ds3-platform
cd ds3-platform
code .          # open in VS Code; accept the "install recommended extensions" prompt
```

## 3. Python environment
```bash
python -m venv .venv
.venv\Scripts\activate          # PowerShell. If blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
pip install -e ".[dev]"         # add ,field for ASD/SED support:  pip install -e ".[dev,field]"
```
VS Code: `Ctrl+Shift+P` → *Python: Select Interpreter* → `.venv`.

## 4. Verify
```bash
pytest -q                       # expect: 93 passed (needs the [dev] install, which includes PySide6)
ds3 formats
ds3 sample-data && ds3 ingest data/sample && ds3 db status
```
Default database for the CLI is a local SQLite file (`ds3.db`); the desktop app uses `~/.ds3/ds3_library.db`. Nothing else to install.

Desktop app: `ds3-library` (or `python -m ds3.gui`). If installation of PySide6 is not wanted: `pip install -e .` gives the CLI/API only.

## 5. PostgreSQL with Docker
```bash
copy .env.example .env          # then edit .env: uncomment the postgresql DATABASE_URL line
docker compose up -d            # starts postgres:16 on localhost:5432 (user/pass/db = ds3)
ds3 db init                     # creates the tables
ds3 ingest data/sample --source synthetic
ds3 db status
```
Inspect with the VS Code "PostgreSQL" extension, or `docker exec -it ds3-db psql -U ds3 -d ds3`.
Plain SQL schema (for DBAs): `docs/schema_postgres.sql` (regenerate: `ds3 export-schema --out docs/schema_postgres.sql`).
Stop: `docker compose down` (add `-v` to wipe data).
`DATABASE_URL` is read from the environment; set it in the shell if you don't use `.env`:
`$env:DATABASE_URL="postgresql+psycopg://ds3:ds3@localhost:5432/ds3"`.

## 6. Use your own data
1. Download files — see [DATA_SOURCES.md](DATA_SOURCES.md).
2. Put them in `data/raw/<kind>/` where `<kind>` ∈ `satellite | airborne | drone | field | library`
   (the folder name becomes the provenance `kind` automatically). Keep ENVI `.hdr` next to its binary.
3. `ds3 info data/raw/satellite/file.nc` → inspect/QC one file first.
4. `ds3 ingest data/raw --source "<name>" --license "<licence>"`.
5. Re-running is safe: files already stored (same sha256) are reported as `duplicate`.

## 7. Run the API
```bash
ds3 serve          # http://127.0.0.1:8000/docs  (interactive)
```
Main calls: `GET /cubes`, `GET /cubes/{id}/spectrum?row=&col=`, `POST /cubes/{id}/roi`, `GET /spectra?material=`.

## Troubleshooting
| Symptom | Fix |
|---|---|
| `ds3` not found | venv not activated, or run `python -m ds3.cli ...` |
| rasterio / h5py install fails | use Python ≤ 3.14 from python.org (64-bit); `pip install --upgrade pip` first |
| `connection refused` on 5432 | `docker compose ps`; wait for healthy; check nothing else uses port 5432 |
| `MissingWavelengthsError` | file has no wavelengths in its metadata — pass `--unit`/see docs/CONTRACT.md or supply them in Python via `open_cube(path, wavelengths=[...])` |
| USGS `.txt` error | put the `splib07a_Wavelengths_*.txt` file in the same folder |
