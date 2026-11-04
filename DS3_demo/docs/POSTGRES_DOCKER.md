# PostgreSQL + Docker — set-up and check (Windows)

You need: **Docker Desktop** (https://www.docker.com/products/docker-desktop, WSL 2 backend) and the project already set up as in `RUN_AND_CHECK.md` (virtual environment + `pip install -r requirements.txt`).

## A. Database in Docker, program on your computer (recommended)

| Step | Command (PowerShell, in the project folder) | What you should see |
|---|---|---|
| 1 | Start **Docker Desktop**, wait until it says *Engine running*. Then `docker --version` | a version number |
| 2 | `docker compose up -d db` | first time: downloads `postgres:16`, then `Container ds3-db Started` |
| 3 | `docker ps` | `ds3-db` with status `Up … (healthy)` (wait ~10 s if it says *starting*) |
| 4 | `copy .env.postgres.example .env` | creates `.env` containing `DATABASE_URL=postgresql+psycopg://ds3:ds3@localhost:5432/ds3` |
| 5 | Copy your five library files (`material_spectra.txt`, `soil_spectra.txt`, `hsdos_lib_spectra.txt`, `samples_spectra_with_metadata.txt`, `ecostress_library_all_spectra.txt`) into `material_library\data\libraries\` | files in that folder |
| 6 | `python run_gui.py` | console line `database: postgresql+psycopg://ds3:****@localhost:5432/ds3`; the window opens |
| 7 | In the window: **Library** | after a short time (the ECOSTRESS file takes about a minute) five cards under **Main libraries**; the panel on top shows the folder and the count |
| 8 | Check the database itself: `docker exec -it ds3-db psql -U ds3 -d ds3 -c "select name, metadata->>'main' as main from libraries;"` | your libraries, `main = true` |
| 9 | `docker exec -it ds3-db psql -U ds3 -d ds3 -c "select count(*) from spectra;"` | number of spectra |

The libraries are read into PostgreSQL **once** (already loaded files are skipped at every later start). After that the files in the folder are no longer needed for the program to show them.
Libraries you build with **Spectral ▸ Spectral Libraries ▸ Builder** or from clicked pixels are stored in the same database and listed under **Built libraries**.

Look at the data with a tool: pgAdmin or DBeaver → host `localhost`, port `5432`, database `ds3`, user `ds3`, password `ds3`.

## B. Everything in Docker (web application at http://localhost:8000/app/)
```
docker compose --profile app up -d --build
```
The main library files are read from `material_library\data\libraries\` (mounted into the container). In this mode the program runs as a web page: images are **uploaded** through File ▸ Open Image File (the desktop window opens big files in place; a container cannot).
Logs: `docker logs -f ds3-app`.

## C. Stop, reset, back up
| Do | Command |
|---|---|
| Stop (data stays) | `docker compose stop` |
| Start again | `docker compose up -d db` |
| Delete the database completely | `docker compose down -v` |
| Back up | `docker exec ds3-db pg_dump -U ds3 ds3 > ds3_backup.sql` |
| Restore | `type ds3_backup.sql \| docker exec -i ds3-db psql -U ds3 -d ds3` |

## D. For the client's server
Use any PostgreSQL 14+ database: create it, then set `DATABASE_URL=postgresql+psycopg://USER:PASSWORD@HOST:5432/DBNAME` in the `.env` next to `run_gui.py` / `DS3Library.exe`. The tables and the missing columns are created at start (`init_db`); `material_library\database\schema.sql` and `migrations\` document the structure.

## Problems
| Message | Cause / fix |
|---|---|
| `failed to connect to the docker API` | Docker Desktop is not running |
| `port is already allocated` / `5432` in use | another PostgreSQL runs on your PC: change `"5432:5432"` to `"5433:5432"` in `docker-compose.yml` and use `…@localhost:5433/ds3` in `.env` |
| `password authentication failed` | the volume was made with other credentials: `docker compose down -v`, then `up -d db` again |
| `connection refused` at start | the container is still starting: `docker ps` must show *healthy* |
| Library page: *No main library files found* | the files are not in the folder named on the page; copy them and press **Reload main libraries** |

## What has been verified
The whole application (all backend tests, the browser walk-throughs, start-up import of the libraries folder, builder, aggregation, image analyses) was run against a real **PostgreSQL 16** server in the development sandbox.
**Not run there:** Docker itself (`docker compose up`, the `app` image build). The compose file and Dockerfile follow the standard pattern — if a step fails, send me the last lines of the output.
