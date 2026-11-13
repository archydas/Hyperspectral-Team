"""ROI patches (object samples), georeferencing with control points, and the small file helpers the desktop program needs.

Writing to / revealing a folder on the computer is allowed only when the request comes from this computer.
"""
import os, shutil, subprocess, sys, tempfile, uuid
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from ..database import get_db, DATA_HOME
from ..models import Image, Sample, Material, Dataset
from ..services import images as svc
from ..services import georef as gr
from ..services import patches as pt

router = APIRouter(prefix="/api", tags=["Patches and georeferencing"])
_LOCAL_HOSTS = ("127.0.0.1", "::1", "localhost", "testclient")


def _img(db, image_id):
    img = db.query(Image).filter(Image.id == image_id).first()
    if img is None:
        raise HTTPException(404, "Image not found")
    return img


def _local_only(request: Request, what="this"):
    if (request.client.host if request.client else "") not in _LOCAL_HOSTS:
        raise HTTPException(403, f"{what} works only when the app runs on your own computer")


def _wrap(fn):
    try:
        return fn()
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(400, f"{type(e).__name__}: {e}")


# ---- folders and files on this computer ---------------------------------------------------------------------
@router.get("/system/patch_folder")
def default_patch_folder():
    """Suggested folder for ROI patches (the page asks the user to confirm or change it at the first ROI)."""
    p = DATA_HOME / "data" / "patches"
    return {"default": str(p), "exists": p.is_dir(), "local": os.environ.get("DS3_LOCAL_APP") == "1"}


class PathBody(BaseModel):
    path: str


class PathsBody(BaseModel):
    paths: List[str]


@router.post("/system/check_folder")
def check_folder(body: PathBody, request: Request):
    """Make sure a folder exists (create it) and can be written to, before the first patch is saved."""
    _local_only(request, "Choosing a folder")
    p = Path(body.path.strip()).expanduser()
    if not str(body.path).strip():
        raise HTTPException(400, "type or choose a folder")
    try:
        p.mkdir(parents=True, exist_ok=True)
        t = p / f".ds3_write_test_{uuid.uuid4().hex[:6]}"
        t.write_text("x"); t.unlink()
    except Exception as e:
        raise HTTPException(400, f"Cannot write to {p}: {e}")
    return {"path": str(p)}


@router.post("/system/path_info")
def path_info(body: PathsBody, request: Request):
    """Which of these files are still where they were saved? (the Library page shows 'View image' only for those)"""
    _local_only(request, "Checking files")
    out = []
    for s in body.paths[:500]:
        p = Path(s).expanduser() if s else None
        ok = bool(p and p.exists())
        out.append({"path": s, "exists": ok, "name": p.name if p else "", "size": p.stat().st_size if ok and p.is_file() else None})
    return out


@router.post("/system/reveal")
def reveal(body: PathBody, request: Request):
    """Show a file (or folder) in the system file browser."""
    _local_only(request, "Locating a file")
    p = Path(body.path).expanduser()
    if not p.exists():
        raise HTTPException(404, f"Not found at its saved address any more: {p}")
    try:
        if sys.platform.startswith("win"):
            subprocess.Popen(["explorer", "/select,", str(p)] if p.is_file() else ["explorer", str(p)])
        elif sys.platform == "darwin":
            subprocess.Popen(["open", "-R", str(p)] if p.is_file() else ["open", str(p)])
        else:
            subprocess.Popen(["xdg-open", str(p.parent if p.is_file() else p)])
    except Exception as e:
        raise HTTPException(400, f"Could not open the file browser: {e}")
    return {"path": str(p)}


# ---- ROI patches -----------------------------------------------------------------------------------------------
class PatchBody(BaseModel):
    polygon: List[List[float]] = Field(..., min_length=3)       # [[row, col], ...] full-resolution pixel coordinates
    class_name: str = Field(..., min_length=1, max_length=200)
    folder: Optional[str] = None
    bands: Optional[List[int]] = None                           # [R, G, B] band numbers (1-based) the page shows; stored as the file's default display bands


@router.post("/images/{image_id}/patch")
def save_patch(image_id: uuid.UUID, body: PatchBody, request: Request, db: Session = Depends(get_db)):
    """Write the ROI as <folder>/<class name>.tif (georeferenced when the image is) and return where it is."""
    _local_only(request, "Saving a patch into a folder")
    folder = (body.folder or "").strip() or str(DATA_HOME / "data" / "patches")
    return _wrap(lambda: pt.write_patch(_img(db, image_id), body.polygon, body.class_name, folder, rgb=body.bands))


@router.post("/images/{image_id}/patch/download")
def download_patch(image_id: uuid.UUID, body: PatchBody, db: Session = Depends(get_db)):
    """Same patch as a download (for use from a browser); nothing is kept on the server."""
    tmp = Path(tempfile.mkdtemp(prefix="ds3_patch_"))
    r = _wrap(lambda: pt.write_patch(_img(db, image_id), body.polygon, body.class_name, tmp, rgb=body.bands))
    return FileResponse(r["path"], media_type="image/tiff", filename=r["file"])


class PointsBody(BaseModel):
    points: List[List[float]] = Field(..., min_length=1)        # [[row, col], ...]


@router.post("/images/{image_id}/lonlat")
def lonlat(image_id: uuid.UUID, body: PointsBody, db: Session = Depends(get_db)):
    """Latitude / longitude of image pixels (null when the image is not georeferenced)."""
    return {"points": _wrap(lambda: pt.pixels_to_lonlat(_img(db, image_id), body.points))}


# ---- georeferencing with control points ----------------------------------------------------------------------
class GcpBody(BaseModel):
    points: List[dict] = Field(..., min_length=3)               # [{row, col, lat, lon}, ...]


@router.get("/images/{image_id}/georef")
def get_georef(image_id: uuid.UUID, db: Session = Depends(get_db)):
    img = _img(db, image_id)
    return {"georef": gr.saved(img), "file_has_geo": bool(_wrap(lambda: svc.geo(img)).get("has_geo")) and not gr.saved(img)}


@router.post("/images/{image_id}/georef/fit")
def fit_georef(image_id: uuid.UUID, body: GcpBody, db: Session = Depends(get_db)):
    """Try a set of control points (affine fit, residuals, RMSE) without saving anything."""
    _img(db, image_id)
    try:
        return _wrap(lambda: gr.fit_gcps([{k: float(p[k]) for k in ("row", "col", "lat", "lon")} for p in body.points]))
    except KeyError as e:
        raise HTTPException(400, f"each control point needs row, col, lat and lon ({e})")


@router.put("/images/{image_id}/georef")
def put_georef(image_id: uuid.UUID, body: GcpBody, db: Session = Depends(get_db)):
    """Fit the affine map from >= 3 control points and keep it with the image."""
    img = _img(db, image_id)

    def go():
        pts = [{k: float(p[k]) for k in ("row", "col", "lat", "lon")} for p in body.points]
        fit = gr.fit_gcps(pts)
        fit["points"] = pts
        img.image_metadata = {**(img.image_metadata or {}), "georef": fit}
        db.commit()
        svc.forget(img.id)
        return {"georef": fit, "geo": svc.geo(img)}
    try:
        return _wrap(go)
    except KeyError as e:
        raise HTTPException(400, f"each control point needs row, col, lat and lon ({e})")


@router.delete("/images/{image_id}/georef")
def delete_georef(image_id: uuid.UUID, db: Session = Depends(get_db)):
    img = _img(db, image_id)
    md = dict(img.image_metadata or {}); md.pop("georef", None)
    img.image_metadata = md
    db.commit()
    return {"ok": True}


# ---- object (ROI) libraries: the saved image patch of each sample --------------------------------------------------
def _object_meta(db, sample_id):
    s = db.query(Sample).filter(Sample.id == sample_id).first()
    o = ((s.sample_metadata or {}).get("object") if s else None)
    if not o or not o.get("patch_path"):
        raise HTTPException(404, "This sample has no saved image patch")
    return o


@router.get("/libraries/{library_id}/objects")
def library_objects(library_id: uuid.UUID, request: Request, db: Session = Depends(get_db)):
    """The object (ROI) samples of a library: label, class, position, pixel count and the NAME of the saved patch file.
    The folder is not sent to the page; View image / Locate go through the sample id."""
    local = (request.client.host if request.client else "") in _LOCAL_HOSTS
    rows = (db.query(Sample, Material).join(Material, Sample.material_id == Material.id)
            .join(Dataset, Sample.dataset_id == Dataset.id).filter(Dataset.library_id == library_id).order_by(Material.name).all())
    out = []
    for s, m in rows:
        o = (s.sample_metadata or {}).get("object")
        if not o:
            continue
        pf = o.get("patch_path")
        out.append(dict(sample_id=str(s.id), material_id=str(m.id), name=m.name, class_name=o.get("class_name"), n_pixels=o.get("n_pixels"),
                        lat=o.get("lat"), lon=o.get("lon"), image_name=o.get("image_name"), patch_file=o.get("patch_file"),
                        has_patch=bool(pf), patch_exists=(Path(pf).is_file() if (pf and local) else None)))
    return {"objects": out}


def _view_image(db: Session, sample_id: uuid.UUID):
    """The saved patch as a throw-away Image: it is read from where it was saved and NOT added to the Image module's list."""
    from ..models import Image
    o = _object_meta(db, sample_id)
    p = Path(o["patch_path"])
    if not p.is_file():
        raise HTTPException(404, f"{o.get('patch_file') or p.name} is not at its saved location any more. Move it back, or save the ROI again.")
    key = uuid.uuid5(uuid.NAMESPACE_URL, f"{p.resolve()}|{p.stat().st_mtime_ns}")          # a changed file gets a fresh cache entry
    return Image(id=key, name=p.stem, path=str(p), reader_kw={}, unit=None, bad_band_mask=None), p


@router.get("/samples/{sample_id}/patch/view")
def view_patch(sample_id: uuid.UUID, request: Request, r: Optional[int] = None, g: Optional[int] = None, b: Optional[int] = None,
               db: Session = Depends(get_db)):
    """Visualisation only (Library > View image): colour preview, all band wavelengths; the bands shown first are the ones that were on screen when the patch was saved."""
    _local_only(request, "Viewing an image from its saved location")
    img, p = _view_image(db, sample_id)
    def go():
        cube = svc.open_image(img)
        bands = [r, g, b] if None not in (r, g, b) else None
        if bands is None:
            import rasterio
            try:
                with rasterio.open(p) as d:
                    t = (d.tags().get("display_bands_rgb") or "").split(",")
                    if len(t) == 3 and all(1 <= int(x) <= d.count for x in t):
                        bands = [int(x) for x in t]
            except Exception:
                pass
        pv = svc.preview(img, bands, max_px=1_000_000)
        return dict(name=p.name, rows=cube.rows, cols=cube.cols, bands=cube.bands, unit=str(cube.unit),
                    wavelengths=[float(x) for x in cube.wavelengths_nm], preview=pv)
    return _wrap(go)


@router.get("/samples/{sample_id}/patch/pixel")
def patch_pixel(sample_id: uuid.UUID, request: Request, row: int, col: int, db: Session = Depends(get_db)):
    _local_only(request, "Viewing an image from its saved location")
    img, _ = _view_image(db, sample_id)
    return _wrap(lambda: svc.pixel(img, row, col, 1)[0])


@router.post("/samples/{sample_id}/patch/reveal")
def reveal_patch(sample_id: uuid.UUID, request: Request, db: Session = Depends(get_db)):
    o = _object_meta(db, sample_id)
    return reveal(PathBody(path=o["patch_path"]), request)
