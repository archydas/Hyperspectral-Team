"""One interface, two ways to reach the material library - what the GUI talks to.

DirectClient  SQLAlchemy session straight to PostgreSQL (or SQLite) using repository.py.  No server needed.
ApiClient     HTTP/JSON to the FastAPI backend (hvva/library/api, a superset of the new_hyperspectral
              backend routes), e.g. when the database is only reachable through the API server.

Both return the same dicts, so the GUI does not care which one it has.  No Qt here.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Callable, List, Optional

import numpy as np

from . import repository as repo


class LibraryClient:
    mode = "?"

    def describe(self) -> str: ...
    def close(self): ...


UNSET = repo._UNSET


def _sig_items(sigs, category=None):
    """Signature objects (core.signature) or dicts -> JSON-able items for save_spectra."""
    out = []
    for s in sigs:
        if isinstance(s, dict):
            out.append(s)
            continue
        out.append({"name": s.name, "wavelengths": np.asarray(s.wavelengths, float).tolist(),
                    "values": np.asarray(s.mean, float).tolist(),
                    "std": None if s.std is None else np.asarray(s.std, float).tolist(),
                    "category": category, "material_metadata": {"source": s.source, "n_pixels": int(s.n_pixels),
                                                                "created_by": "HyperVista", **(s.meta or {})},
                    "spectrum_metadata": {"n_pixels": int(s.n_pixels), "source_scenes": s.source},
                    "unit": getattr(s, "unit", None), "purity": getattr(s, "purity", None)})
    return out


# ====================================================================== direct
class DirectClient(LibraryClient):
    mode = "direct"

    def __init__(self, url: Optional[str] = None, init_schema: bool = False):
        from .database import connect
        self.db = connect(url, init=init_schema)
        if not self.db.has_schema():
            raise RuntimeError("the database has no library tables - tick 'create tables if missing' to set them up")

    def describe(self):
        return f"PostgreSQL: {self.db.safe_url()}" if self.db.is_postgres else f"SQLite: {self.db.safe_url()}"

    def close(self):
        self.db.dispose()

    def _run(self, fn, *a, **kw):
        with self.db.session() as s:
            return fn(s, *a, **kw)

    def list_libraries(self):
        return self._run(repo.list_libraries)

    def create_library(self, name, description=None):
        return self._run(repo.create_library, name, description)

    def delete_library(self, library_id):
        return self._run(repo.delete_library, library_id)

    def list_datasets(self, library_id):
        return self._run(repo.list_datasets, library_id)

    def list_materials(self, library_id, search=None, category=None, limit=20, offset=0, category_path=None,
                       acquisition=None, unit=None, purity=None, covers=None):
        return self._run(repo.list_materials, library_id, search, category, limit, offset, category_path, acquisition,
                         unit, purity, covers)

    def library_facets(self, library_id):
        return self._run(repo.library_facets, library_id)

    def update_materials(self, material_ids, category=UNSET, acquisition=UNSET, purity=UNSET):
        return self._run(repo.update_materials, material_ids, category, acquisition, purity)

    def library_unit(self, library_id):
        return self._run(repo.library_unit, library_id)

    def rename_category(self, library_id, old_path, new_path):
        return self._run(repo.rename_category, library_id, old_path, new_path)

    def set_spectrum_acquisition(self, spectrum_id, acquisition):
        return self._run(repo.set_spectrum_acquisition, spectrum_id, acquisition)

    def list_categories(self, library_id):
        return self._run(repo.list_categories, library_id)

    def create_material(self, library_id, name, category=None, description=None):
        return self._run(repo.create_material, library_id, name, category, description)

    def delete_material(self, material_id):
        return self._run(repo.delete_material, material_id)

    def search_materials(self, q, limit=20, acquisition=None, unit=None, purity=None, covers=None):
        return self._run(repo.search_materials_global, q, limit, acquisition, unit, purity, covers)

    def get_material(self, material_id):
        return self._run(repo.get_material, material_id)

    def list_samples(self, material_id):
        return self._run(repo.list_samples, material_id)

    def list_spectra(self, sample_id):
        return self._run(repo.list_spectra, sample_id)

    def library_spectra(self, library_id, acquisition=None, category_path=None, unit=None, purity=None, covers=None,
                        all_samples=False):
        return self._run(repo.library_spectra, library_id, None, acquisition, category_path, unit, purity, covers,
                         all_samples)

    def resample_library(self, library_id, grid, new_name, method="linear", fwhm=None, min_coverage=0.5):
        return self._run(repo.resample_library, library_id, list(map(float, grid)), new_name, method,
                         None if fwhm is None else list(map(float, fwhm)), min_coverage)

    # ---- registered images (migration 004)
    def register_image(self, info):
        return self._run(repo.register_image, info)

    def list_images(self, search=None):
        return self._run(repo.list_images, search)

    def get_image(self, image_id):
        return self._run(repo.get_image, image_id)

    def delete_image(self, image_id):
        return self._run(repo.delete_image, image_id)

    # ---- ROIs (migration 002)
    def save_roi_spectra(self, library_id, dataset_name, rois, items, category=None, acquisition=None):
        return self._run(repo.save_roi_spectra, library_id, dataset_name, rois, items, category, acquisition)

    def list_rois(self, library_id=None, scene_name=None, class_name=None):
        return self._run(repo.list_rois, library_id, scene_name, class_name)

    def get_roi(self, roi_id):
        return self._run(repo.get_roi, roi_id)

    def roi_classes(self, library_id):
        return self._run(repo.roi_classes, library_id)

    def roi_spectra(self, roi_id=None, library_id=None, class_name=None):
        return self._run(repo.roi_spectra, roi_id, library_id, class_name)

    def spectrum_roi_info(self, spectrum_id):
        return self._run(repo.spectrum_roi_info, spectrum_id)

    def delete_roi(self, roi_id, delete_spectra=False):
        return self._run(repo.delete_roi, roi_id, delete_spectra)

    def save_spectra(self, library_id, dataset_name, items, category=None, acquisition=None):
        items = _sig_items(items, category)

        def go(s):
            ds = repo.get_or_create_dataset(s, library_id, dataset_name, source_name="HyperVista",
                                            description="Signatures saved from HyperVista.", commit=False)
            res = [repo.add_spectrum(s, library_id, ds.id, it["name"], it["wavelengths"], it["values"],
                                     category=it.get("category") or category, std=it.get("std"),
                                     material_metadata=it.get("material_metadata"),
                                     spectrum_metadata=it.get("spectrum_metadata"), commit=False,
                                     acquisition=it.get("acquisition_type", acquisition),
                                     unit=it.get("unit"), purity=it.get("purity")) for it in items]
            s.commit()
            return res
        return self._run(go)

    def import_csv(self, path, library_id, dataset_name=None, category=None,
                   progress: Optional[Callable[[int, int, str], None]] = None, acquisition=None):
        from .importer import import_wide_csv
        return self._run(import_wide_csv, path, library_id=library_id, dataset_name=dataset_name,
                         category=category, progress=progress, acquisition=acquisition)


    def import_speclib(self, kind, path, library_id, dataset_name=None, chapters=None, acquisition=None,
                       limit=None, progress=None, categories=None):
        """USGS splib07 ASCII release, ECOSTRESS files (folder or zip) or an OpenSpecLib .parquet -> this library."""
        from .speclib import import_speclib
        kw = {"dataset_name": dataset_name} if dataset_name else {}
        return self._run(import_speclib, kind, path, chapters=chapters, library_id=library_id, categories=categories,
                         acquisition=acquisition, limit=limit, progress=progress, **kw)


    # ---- any supported file / folder / zip (universal pipeline)
    def preview_file(self, path, options=None, n_spectra=12):
        from .formats import ReadOptions
        from .ingest import preview_path
        return preview_path(path, ReadOptions.from_dict(options), n_spectra)

    def import_file(self, path, library_id=None, library_name=None, dataset_name=None, options=None, progress=None):
        from .formats import ReadOptions
        from .ingest import import_path
        return self._run(import_path, path, library_id=library_id, library_name=library_name,
                         dataset_name=dataset_name, options=ReadOptions.from_dict(options), progress=progress)

    # ---- local data folder (scan + batch import; the files must be on this computer)
    def scan_folder(self, folder):
        """[Source dicts] for every importable file/folder under *folder*, with status new / imported / error."""
        from . import sources
        return [s.to_dict() for s in self._run(lambda db: sources.status(db, sources.scan(folder)))]

    def import_sources(self, items, progress=None):
        """items = [{"source": dict from scan_folder, "library": name, acquisition?, limit?, ...}]."""
        from . import sources
        cur = {"label": ""}

        def item(k, n, s):
            cur["label"] = f"[{k}/{n}] {s.name}"
            if progress:
                progress(0, 1, cur["label"])

        def prog(i, n, name):
            if progress:
                progress(i, n, f"{cur['label']}: {name}")
        return self._run(sources.import_many, items, prog, item)

    def import_new_from_folder(self, folder, progress=None):
        from . import sources
        return self._run(sources.import_new, folder, progress)


# ====================================================================== HTTP
class ApiClient(LibraryClient):
    mode = "api"

    def __init__(self, base_url="http://127.0.0.1:8000", timeout=15):
        self.base = base_url.rstrip("/")
        self.timeout = timeout
        h = self._req("GET", "/health")
        if not isinstance(h, dict) or h.get("status") != "ok":
            raise RuntimeError(f"{self.base} did not answer like the library API")

    def describe(self):
        return f"API: {self.base}"

    def close(self):
        pass

    def _req(self, method, path, params=None, body=None, raw=None, content_type="application/json"):
        url = self.base + path
        if params:
            url += "?" + urllib.parse.urlencode({k: v for k, v in params.items() if v not in (None, "")})
        data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
        req = urllib.request.Request(url, data=data, method=method)
        if data is not None:
            req.add_header("Content-Type", content_type)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                txt = r.read()
        except urllib.error.HTTPError as e:
            try:
                detail = json.loads(e.read()).get("detail")
            except Exception:                                                  # noqa: BLE001
                detail = None
            msg = detail if isinstance(detail, str) else (json.dumps(detail) if detail else e.reason)
            if e.code == 404:
                raise repo.NotFound(msg) from None
            raise RuntimeError(f"API {e.code}: {msg}") from None
        except urllib.error.URLError as e:
            raise ConnectionError(f"cannot reach {self.base}: {e.reason}") from None
        return json.loads(txt) if txt else None

    @staticmethod
    def _q(x):
        return urllib.parse.quote(str(x), safe="")

    def list_libraries(self):
        return self._req("GET", "/api/libraries/")

    def create_library(self, name, description=None):
        return self._req("POST", "/api/libraries/", body={"name": name, "description": description or None})

    def delete_library(self, library_id):
        return self._req("DELETE", f"/api/libraries/{self._q(library_id)}")

    def list_datasets(self, library_id):
        return self._req("GET", f"/api/libraries/{self._q(library_id)}/datasets")

    @staticmethod
    def _covers(covers):
        return None if not covers else f"{float(covers[0]):g},{float(covers[1]):g}"

    def list_materials(self, library_id, search=None, category=None, limit=20, offset=0, category_path=None,
                       acquisition=None, unit=None, purity=None, covers=None):
        return self._req("GET", f"/api/libraries/{self._q(library_id)}/materials",
                         {"search": search, "category": category, "limit": limit, "offset": offset,
                          "category_path": category_path, "acquisition": acquisition, "unit": unit,
                          "purity": purity, "covers": self._covers(covers)})

    def library_facets(self, library_id):
        return self._req("GET", f"/api/libraries/{self._q(library_id)}/facets")

    def update_materials(self, material_ids, category=UNSET, acquisition=UNSET, purity=UNSET):
        body = {"ids": [str(i) for i in material_ids]}
        if category is not UNSET:
            body["category"] = category
        if acquisition is not UNSET:
            body["acquisition_type"] = acquisition
        if purity is not UNSET:
            body["purity"] = purity
        return self._req("PATCH", "/api/materials", body=body)

    def library_unit(self, library_id):
        lib = next((l for l in self.list_libraries() if l["id"] == str(library_id)), None)
        return (lib or {}).get("unit")

    def rename_category(self, library_id, old_path, new_path):
        return self._req("POST", f"/api/libraries/{self._q(library_id)}/categories/rename",
                         body={"old_path": old_path, "new_path": new_path})

    def set_spectrum_acquisition(self, spectrum_id, acquisition):
        return self._req("PATCH", f"/api/spectra/{self._q(spectrum_id)}", body={"acquisition_type": acquisition})

    def list_categories(self, library_id):
        return self._req("GET", f"/api/libraries/{self._q(library_id)}/categories")

    def create_material(self, library_id, name, category=None, description=None):
        return self._req("POST", f"/api/libraries/{self._q(library_id)}/materials",
                         body={"name": name, "category": category or None, "description": description or None})

    def delete_material(self, material_id):
        return self._req("DELETE", f"/api/materials/{self._q(material_id)}")

    def search_materials(self, q, limit=20, acquisition=None, unit=None, purity=None, covers=None):
        return self._req("GET", "/api/materials/search", {"q": q, "limit": limit, "acquisition": acquisition,
                                                           "unit": unit, "purity": purity,
                                                           "covers": self._covers(covers)})

    def get_material(self, material_id):
        return self._req("GET", f"/api/materials/{self._q(material_id)}")

    def list_samples(self, material_id):
        return self._req("GET", f"/api/materials/{self._q(material_id)}/samples")

    def list_spectra(self, sample_id):
        return self._req("GET", f"/api/samples/{self._q(sample_id)}/spectra")

    def library_spectra(self, library_id, acquisition=None, category_path=None, unit=None, purity=None, covers=None,
                        all_samples=False):
        return self._req("GET", f"/api/libraries/{self._q(library_id)}/spectra",
                         {"acquisition": acquisition, "category_path": category_path, "unit": unit, "purity": purity,
                          "covers": self._covers(covers), "all_samples": "true" if all_samples else None})

    def resample_library(self, library_id, grid, new_name, method="linear", fwhm=None, min_coverage=0.5):
        return self._req("POST", f"/api/libraries/{self._q(library_id)}/resample",
                         body={"grid": [float(v) for v in grid], "name": new_name, "method": method,
                               "fwhm": None if fwhm is None else [float(v) for v in fwhm],
                               "min_coverage": min_coverage})

    # ---- registered images (migration 004)
    def register_image(self, info):
        return self._req("POST", "/api/images", body=info)

    def list_images(self, search=None):
        return self._req("GET", "/api/images", {"search": search})

    def get_image(self, image_id):
        return self._req("GET", f"/api/images/{self._q(image_id)}")

    def delete_image(self, image_id):
        return self._req("DELETE", f"/api/images/{self._q(image_id)}")

    # ---- ROIs (migration 002)
    def save_roi_spectra(self, library_id, dataset_name, rois, items, category=None, acquisition=None):
        return self._req("POST", f"/api/libraries/{self._q(library_id)}/roi-spectra",
                         body={"dataset_name": dataset_name, "category": category, "acquisition_type": acquisition,
                               "rois": rois, "items": items})

    def list_rois(self, library_id=None, scene_name=None, class_name=None):
        if library_id is None:
            return self._req("GET", "/api/rois", {"scene_name": scene_name, "class_name": class_name})
        return self._req("GET", f"/api/libraries/{self._q(library_id)}/rois",
                         {"scene_name": scene_name, "class_name": class_name})

    def get_roi(self, roi_id):
        return self._req("GET", f"/api/rois/{self._q(roi_id)}")

    def roi_classes(self, library_id):
        return self._req("GET", f"/api/libraries/{self._q(library_id)}/roi-classes")

    def roi_spectra(self, roi_id=None, library_id=None, class_name=None):
        if roi_id is not None:
            return self._req("GET", f"/api/rois/{self._q(roi_id)}/spectra")
        return self._req("GET", f"/api/libraries/{self._q(library_id)}/roi-classes/{self._q(class_name)}/spectra")

    def spectrum_roi_info(self, spectrum_id):
        return self._req("GET", f"/api/spectra/{self._q(spectrum_id)}/rois")

    def delete_roi(self, roi_id, delete_spectra=False):
        return self._req("DELETE", f"/api/rois/{self._q(roi_id)}", {"delete_spectra": "true" if delete_spectra else None})

    def save_spectra(self, library_id, dataset_name, items, category=None, acquisition=None):
        return self._req("POST", f"/api/libraries/{self._q(library_id)}/spectra",
                         body={"dataset_name": dataset_name, "category": category, "acquisition_type": acquisition,
                               "items": _sig_items(items, category)})

    def import_csv(self, path, library_id, dataset_name=None, category=None, progress=None, acquisition=None):
        with open(path, "rb") as f:
            raw = f.read()
        if progress:
            progress(0, 1, os.path.basename(path))
        return self._req("POST", f"/api/libraries/{self._q(library_id)}/import-csv",
                         {"filename": os.path.basename(path), "dataset_name": dataset_name, "category": category,
                          "acquisition": acquisition},
                         raw=raw, content_type="text/csv")


def _api_speclib(self, *a, **kw):
    raise RuntimeError("importing local data files needs a direct database connection (Connect... -> Direct), "
                       "or run: python -m hvva.library.sources <folder> --import")


ApiClient.import_speclib = _api_speclib
ApiClient.scan_folder = ApiClient.import_sources = ApiClient.import_new_from_folder = _api_speclib
ApiClient.import_file = _api_speclib


def _api_preview(self, path, options=None, n_spectra=12):
    """Preview needs no database: read the local file here."""
    from .formats import ReadOptions
    from .ingest import preview_path
    return preview_path(path, ReadOptions.from_dict(options), n_spectra)


ApiClient.preview_file = _api_preview


def make_client(mode: str, url: str, init_schema: bool = False) -> LibraryClient:
    return ApiClient(url) if mode == "api" else DirectClient(url, init_schema)


__all__: List[str] = ["LibraryClient", "DirectClient", "ApiClient", "make_client"]
