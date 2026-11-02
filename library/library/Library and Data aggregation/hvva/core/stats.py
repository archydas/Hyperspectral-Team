"""Per-class / per-ROI statistics pooled across scenes (streaming, NaN-aware, per band)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from .resample import resample_spectra
from .scene import Scene


class _Acc:
    """Per-band count / mean / M2 (Chan et al. parallel update) so scenes can be merged exactly."""

    def __init__(self, nb):
        self.n = np.zeros(nb)
        self.mean = np.zeros(nb)
        self.m2 = np.zeros(nb)
        self.lo = np.full(nb, np.inf)
        self.hi = np.full(nb, -np.inf)

    def add(self, X):
        ok = ~np.isnan(X)
        n_b = ok.sum(0).astype(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean_b = np.where(n_b > 0, np.nansum(X, 0) / np.maximum(n_b, 1), 0.0)
        m2_b = np.nansum((X - mean_b) ** 2, 0)
        lo = np.where(n_b > 0, np.nanmin(np.where(ok, X, np.inf), 0), np.inf)
        hi = np.where(n_b > 0, np.nanmax(np.where(ok, X, -np.inf), 0), -np.inf)
        self._merge(n_b, mean_b, m2_b, lo, hi)

    def merge(self, o: "_Acc"):
        self._merge(o.n, o.mean, o.m2, o.lo, o.hi)

    def _merge(self, n_b, mean_b, m2_b, lo, hi):
        tot = self.n + n_b
        safe = np.maximum(tot, 1)
        delta = mean_b - self.mean
        self.mean = np.where(tot > 0, self.mean + delta * n_b / safe, 0.0)
        self.m2 = self.m2 + m2_b + delta ** 2 * self.n * n_b / safe
        self.n = tot
        self.lo, self.hi = np.minimum(self.lo, lo), np.maximum(self.hi, hi)

    def result(self):
        n = self.n
        mean = np.where(n > 0, self.mean, np.nan)
        std = np.where(n > 1, np.sqrt(self.m2 / np.maximum(n - 1, 1)), np.nan)    # sample std (ddof = 1)
        lo = np.where(n > 0, self.lo, np.nan)
        hi = np.where(n > 0, self.hi, np.nan)
        return n, mean, std, lo, hi


@dataclass
class ClassStats:
    name: str
    n_pixels: int
    mean: np.ndarray
    std: np.ndarray
    minimum: np.ndarray
    maximum: np.ndarray
    per_scene: Dict[str, dict] = field(default_factory=dict)      # scene -> {n, mean, std}

    @property
    def n_scenes(self):
        return len(self.per_scene)


@dataclass
class AggregateResult:
    wavelengths: np.ndarray
    classes: Dict[str, ClassStats]
    scene_names: List[str]


def aggregate_classes(scenes, grid, max_pixels_per_scene=None, seed=0) -> AggregateResult:
    """Resample every scene to *grid* and pool the pixels of each class across scenes.

    Classes are matched by name.  max_pixels_per_scene randomly caps pixels (speed/balance).
    """
    grid = np.asarray(grid, float)
    rng = np.random.default_rng(seed)
    per_scene: Dict[str, Dict[str, _Acc]] = {}
    for sc in scenes:
        if not sc.has_wavelengths:
            raise ValueError(f"scene '{sc.name}' has no wavelengths - assign them first")
        for cname, mask in sc.class_masks().items():
            X = sc.cube[mask]
            if max_pixels_per_scene and len(X) > max_pixels_per_scene:
                X = X[rng.choice(len(X), max_pixels_per_scene, replace=False)]
            acc = _Acc(len(grid))
            acc.add(resample_spectra(X, sc.wavelengths, grid))
            per_scene.setdefault(cname, {})[sc.name] = acc
    classes = {}
    for cname, by_scene in per_scene.items():
        pooled = _Acc(len(grid))
        ps = {}
        for sname, acc in by_scene.items():
            pooled.merge(acc)
            n, m, s, _, _ = acc.result()
            ps[sname] = {"n": int(n.max()), "mean": m, "std": s}
        n, m, s, lo, hi = pooled.result()
        classes[cname] = ClassStats(cname, int(n.max()), m, s, lo, hi, ps)
    return AggregateResult(grid, classes, [s.name for s in scenes])
