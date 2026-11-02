"""A spectral signature = mean +/- std of a class on a wavelength grid."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np


@dataclass
class Signature:
    name: str
    wavelengths: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    n_pixels: int = 0
    source: str = ""                                   # e.g. "scene_a, scene_b"
    meta: Dict = field(default_factory=dict)


def signatures_from_result(result, classes=None) -> List[Signature]:
    out = []
    for name, cs in result.classes.items():
        if classes is not None and name not in classes:
            continue
        out.append(Signature(name, result.wavelengths, cs.mean, cs.std, cs.n_pixels,
                             ", ".join(cs.per_scene.keys())))
    return out


def spectral_angle(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = ~(np.isnan(a) | np.isnan(b))
    a, b = a[ok], b[ok]
    return float(np.arccos(np.clip(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30), -1, 1)))
