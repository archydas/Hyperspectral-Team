"""How a scene is shown: gray or RGB composite of chosen bands, with a percentile stretch (no GUI code).

The choice lives in ``Scene.display`` = {"mode": "rgb" | "gray", "bands": [r, g, b] or [band], "stretch": [lo, hi]}
and is kept in the .npz scene file.  Presets pick the bands nearest to target wavelengths.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

PRESETS: Dict[str, Sequence[float]] = {          # target wavelengths (nm), R G B
    "True colour": (650, 550, 450),
    "Colour infrared": (850, 650, 550),
    "SWIR composite": (2200, 850, 650),
    "Vegetation (SWIR/NIR/Red)": (1650, 850, 650),
}


def nearest_band(scene, target_nm: float) -> int:
    if scene.has_wavelengths:
        return int(np.argmin(np.abs(np.asarray(scene.wavelengths) - target_nm)))
    return int(np.clip(round(scene.n_bands * (target_nm - 400) / 2100), 0, scene.n_bands - 1))


def preset_bands(scene, name: str) -> List[int]:
    return [nearest_band(scene, t) for t in PRESETS[name]]


def band_label(scene, i: int) -> str:
    bad = scene.bad_bands[i] if hasattr(scene, "bad_bands") else False
    if scene.has_wavelengths:
        return f"{i + 1}: {scene.wavelengths[i]:.1f} nm" + (" (bad)" if bad else "")
    return f"Band {i + 1}" + (" (bad)" if bad else "")


def default_display(scene) -> dict:
    return {"mode": "rgb", "bands": preset_bands(scene, "True colour"), "stretch": [2.0, 98.0]}


def normalize_display(scene, d: Optional[dict]) -> dict:
    d = dict(d or default_display(scene))
    nb = scene.n_bands
    mode = d.get("mode") if d.get("mode") in ("rgb", "gray") else "rgb"
    bands = [int(np.clip(int(b), 0, nb - 1)) for b in (d.get("bands") or [])]
    if mode == "rgb" and len(bands) != 3:
        bands = preset_bands(scene, "True colour")
    if mode == "gray":
        bands = bands[:1] or [nearest_band(scene, 650)]
    lo, hi = (d.get("stretch") or [2.0, 98.0])[:2]
    lo, hi = float(np.clip(lo, 0, 49)), float(np.clip(hi, 51, 100))
    return {"mode": mode, "bands": bands, "stretch": [lo, hi]}


def _stretch(band: np.ndarray, lo: float, hi: float) -> np.ndarray:
    b = band.astype(np.float32)
    fin = b[np.isfinite(b)]
    if fin.size == 0:
        return np.zeros_like(b)
    step = max(1, fin.size // 2_000_000)                      # percentiles on a sample: fast on big scenes
    plo, phi = np.percentile(fin[::step], [lo, hi])
    return np.nan_to_num(np.clip((b - plo) / (phi - plo + 1e-12), 0, 1))


def render(scene, display: Optional[dict] = None):
    """(image array, matplotlib cmap) for the scene view."""
    d = normalize_display(scene, display if display is not None else getattr(scene, "display", None))
    lo, hi = d["stretch"]
    if d["mode"] == "gray":
        return _stretch(scene.cube[:, :, d["bands"][0]], lo, hi), "gray"
    return np.dstack([_stretch(scene.cube[:, :, b], lo, hi) for b in d["bands"]]), None


def describe(scene, display: Optional[dict] = None) -> str:
    d = normalize_display(scene, display if display is not None else getattr(scene, "display", None))
    names = "/".join(band_label(scene, b).split(": ")[-1] for b in d["bands"])
    return ("Gray " if d["mode"] == "gray" else "RGB ") + names
