# Hyperspectral Tools

A collection of tools for hyperspectral data processing, data aggregation, analysis, and visualization.

## Repository Overview

This repository contains two independent tools designed to support hyperspectral data workflows. Each has its own README, `requirements.txt`, and run instructions.

### 1. Library & Data Aggregation (HyperVista_va)

Located in [`library-aggregation/`](library-aggregation/).

Builds spectral libraries from several scenes: multi-scene aggregation on a common wavelength grid, mosaicking, signature export, georeferencing, plugins, and a PostgreSQL material spectral library with spectral-angle matching.

**Key components:**
- Python package (`hvva/`) with core processing, library database and GUI modules
- Database schema and Docker setup (`database/`)
- Sample spectra (`data/`), plugins (`plugins/`), and tests (`tests/`)
- Command-line and graphical interfaces

**Documentation:** [Library README](library-aggregation/README.md) · [Developer guide](library-aggregation/DEVELOPER_GUIDE.md) · [Data sources](library-aggregation/DATA_SOURCES.md)

### 2. Hyperspectral Viewer (DS3 Viewer & Analysis Tool)

Located in [`viewer/`](viewer/).

A PyQt5 desktop application for loading, visualizing, preprocessing, correcting, and analysing hyperspectral cubes (ENVI, MATLAB, HDF5, NetCDF, GeoTIFF and more).

**Key components:**
- Main application (`HyperVista_v1_0.py`)
- User guides (`docs/`)
- Sample scenes (`sample_data/`) and helper scripts (`examples/`, `additional_codes/`)
- Earlier development versions (`archive/previous_versions/`)

**Documentation:** [Viewer README](viewer/README.md) · [User guide (PDF)](viewer/docs/DS3_v1.0_User_Guide.pdf)

## Repository Structure

```text
Hyperspectral-Team/
├── README.md
├── LICENSE
├── .gitignore
├── .gitattributes
│
├── library-aggregation/          Library & data aggregation tool
│   ├── README.md
│   ├── hvva/                     core, library, gui
│   ├── database/                 schema.sql, docker-compose.yml
│   ├── data/                     sample spectra (CSV)
│   ├── plugins/   tests/   assets/
│   ├── requirements.txt   .env.example
│   └── run_gui.py   run_cli.py
│
└── viewer/                       Hyperspectral viewer
    ├── README.md
    ├── HyperVista_v1_0.py
    ├── hypervista.ico            keep next to the .py file
    ├── requirements.txt
    ├── docs/
    ├── sample_data/
    ├── examples/
    ├── additional_codes/
    └── archive/previous_versions/
```

## Getting Started

### Requirements

- **Python 3.10 or 3.11** (check with `python --version`)
- **Git** (to clone the repository)
- **Docker Desktop** (optional, only for the PostgreSQL material library)

The two tools are independent. Each has its own `requirements.txt`, so use a separate virtual environment for each.

### 1. Get the code

```bash
git clone https://github.com/archydas/Hyperspectral-Team.git
cd Hyperspectral-Team
```

### 2. Run the Viewer

```bash
cd viewer
python -m venv .venv

# activate the environment
.venv\Scripts\activate            # Windows (Command Prompt / PowerShell)
source .venv/Scripts/activate     # Windows (Git Bash)
source .venv/bin/activate         # macOS / Linux

pip install -r requirements.txt
python HyperVista_v1_0.py
```

- In the app, open an ENVI `.hdr` file. The matching data file (`.img`) must be in the same folder.
- Sample data is in `viewer/sample_data/`. The large `Seurat_BEFORE.img` (~413 MB) is on the
  [Releases page](https://github.com/archydas/Hyperspectral-Team/releases). Download it and place it next to `Seurat_BEFORE.hdr`.
- `tensorflow` in `requirements.txt` is only needed for the optional *2D CNN* tab. Delete that line to skip the large download.

### 3. Run the Library & Aggregation tool

```bash
cd library-aggregation
python -m venv .venv
.venv\Scripts\activate            # or the activation command for your system, as above

pip install -r requirements.txt
python run_gui.py
```

In the app, choose **File → Load demo scenes** for a quick look without any data of your own.

The **Library** tab connects to a spectral library database. There are two ways to set it up.

**Option A: PostgreSQL (full library, needs Docker Desktop running)**

```bash
# from the library-aggregation folder
docker compose -f database/docker-compose.yml up -d

python -m hvva.library.importer data/material_spectra.csv --library "Material Library" --dataset "Material Spectra"
python -m hvva.library.importer data/soil_spectra.csv --library "Soil Library" --dataset "Soil Spectra" --category Soil

python run_gui.py
```

The app connects automatically to `postgresql://postgres:postgres@localhost:5432/material_library`.
To use another database, copy `.env.example` to `.env` and edit `DATABASE_URL`, or use **Connect…** in the Library tab.
Stop the database with `docker compose -f database/docker-compose.yml down`.

**Option B: No Docker (offline, single file)**

Run `python run_gui.py`, open the **Library** tab, click **Connect…**, enter `sqlite:///my_library.db`, and tick **create tables if missing**.

### 4. Command-line use (library tool)

```bash
python -m hvva demo --out demo                                   # write 3 synthetic scenes
python -m hvva aggregate demo/*.npz --out sigs --db lib.sqlite   # aggregate signatures
python -m hvva mosaic demo/*.npz --out mosaic.npz --envi         # mosaic scenes
python -m hvva plugins                                           # list plugins
python -m pytest tests                                           # run the tests
```

## Troubleshooting

| Problem | Fix |
|---|---|
| `python` not found | Install Python 3.10/3.11 and tick **Add to PATH**. On Windows you can also try `py -3.11` instead of `python`. |
| `No module named PyQt5` (or another package) | The virtual environment is not active, or `pip install -r requirements.txt` was run in a different folder. |
| Viewer opens a `.hdr` but shows no data | The `.img` file is missing or is not in the same folder as the `.hdr`. |
| Library tab cannot connect | Docker Desktop is not running, or the database container has not started. Check with `docker ps`, or use Option B (SQLite). |
| `pip install` fails for `rasterio` or `netCDF4` | These packages are optional. Delete those lines from `requirements.txt` and install the rest. |

## License

MIT. See [LICENSE](LICENSE). The sample datasets may carry their own terms; see
[`library-aggregation/DATA_SOURCES.md`](library-aggregation/DATA_SOURCES.md).
