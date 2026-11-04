import os, threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .database import init_db, display_url, PROJECT_ROOT, DATA_HOME
from .routes import libraries, datasets, materials, samples, spectra, admin, images, tools, taxonomy, aggregate, builder, imageops


STARTUP_REPORT: dict = {"folder": None, "running": False, "files": []}


def _auto_import():
    """Load everything in the libraries folder (default data/libraries, or AUTO_IMPORT_DIR) once per start, in the background."""
    folder = Path(os.getenv("AUTO_IMPORT_DIR") or DATA_HOME / "data" / "libraries")
    STARTUP_REPORT.update(folder=str(folder), running=True, files=[], folder_exists=folder.is_dir())
    try:
        from .services.importer import import_folder_as_libraries
        STARTUP_REPORT["files"] = import_folder_as_libraries(folder)
    finally:
        STARTUP_REPORT["running"] = False
        for f in STARTUP_REPORT["files"]:
            print(f"[startup import] {f['status']:9s} {f['file']} -> {f['library']}: {f['detail']}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    if os.getenv("AUTO_IMPORT", "1") != "0":
        threading.Thread(target=_auto_import, daemon=True).start()
    yield


app = FastAPI(title="Material Spectral Library API", version="2.0.0", lifespan=lifespan)

# Allow the HTML to talk to the API even when opened from a file or another port.
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=False, allow_methods=["*"], allow_headers=["*"])

app.include_router(libraries.router)
app.include_router(datasets.router)
app.include_router(materials.router)
app.include_router(samples.router)
app.include_router(spectra.router)
app.include_router(admin.router)
app.include_router(images.router)
app.include_router(tools.router)
app.include_router(taxonomy.router)
app.include_router(aggregate.router)
app.include_router(builder.router)
app.include_router(imageops.router)


@app.get("/")
def root():
    return {"message": "Material Spectral Library API", "database": display_url(), "frontend": "/app/", "docs": "/docs"}


@app.get("/api/startup_import")
def startup_import():
    """What the automatic load of the libraries folder did at start-up."""
    return STARTUP_REPORT


@app.post("/api/startup_import/reload")
def reload_main_libraries():
    """Read the libraries folder again (e.g. after copying new files into it). Safe to repeat: known files are skipped."""
    if STARTUP_REPORT.get("running"):
        return STARTUP_REPORT
    STARTUP_REPORT["running"] = True
    threading.Thread(target=_auto_import, daemon=True).start()
    return STARTUP_REPORT


@app.get("/health")
def health():
    return {"status": "ok"}


FRONTEND = PROJECT_ROOT / "frontend"
if FRONTEND.is_dir():
    app.mount("/app", StaticFiles(directory=str(FRONTEND), html=True), name="frontend")
