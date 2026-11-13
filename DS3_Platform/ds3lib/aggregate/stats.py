"""Per-class statistics pooled across scenes: streaming, NaN-aware, exact (Chan et al. parallel update)."""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


@dataclass
class ClassStats:
    name: str
    n_pixels: int                    # pixels that contributed (any band)
    n: np.ndarray                    # valid samples per band
    mean: np.ndarray
    std: np.ndarray                  # sample standard deviation (n - 1)
    minimum: np.ndarray
    maximum: np.ndarray
    scenes: list = field(default_factory=list)


class ClassAccumulator:
    """Add blocks of spectra (N, bands) from any number of scenes; get exact pooled mean / std / min / max."""

    def __init__(self, n_bands: int):
        self.n = np.zeros(n_bands)
        self.mean = np.zeros(n_bands)
        self.m2 = np.zeros(n_bands)
        self.lo = np.full(n_bands, np.inf)
        self.hi = np.full(n_bands, -np.inf)
        self.n_pixels = 0
        self.scenes: list = []

    def add(self, X, scene: str | None = None):
        X = np.asarray(X, np.float64)
        if X.ndim != 2 or X.shape[1] != self.n.size:
            raise ValueError("block must have shape (N, bands) on the common grid")
        ok = np.isfinite(X)
        nb = ok.sum(0).astype(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            mb = np.where(nb > 0, np.where(ok, X, 0).sum(0) / np.maximum(nb, 1), 0.0)
        m2b = np.where(ok, (X - mb) ** 2, 0).sum(0)
        lo = np.where(nb > 0, np.where(ok, X, np.inf).min(0), np.inf)
        hi = np.where(nb > 0, np.where(ok, X, -np.inf).max(0), -np.inf)
        tot = self.n + nb
        safe = np.maximum(tot, 1)
        delta = mb - self.mean
        self.mean = np.where(tot > 0, self.mean + delta * nb / safe, 0.0)
        self.m2 = self.m2 + m2b + delta ** 2 * self.n * nb / safe
        self.n = tot
        self.lo, self.hi = np.minimum(self.lo, lo), np.maximum(self.hi, hi)
        self.n_pixels += int(ok.any(1).sum())
        if scene and scene not in self.scenes:
            self.scenes.append(scene)

    def result(self, name: str) -> ClassStats:
        n = self.n
        with np.errstate(invalid="ignore", divide="ignore"):
            std = np.where(n > 1, np.sqrt(self.m2 / np.maximum(n - 1, 1)), np.nan)
        return ClassStats(name, self.n_pixels, n.copy(), np.where(n > 0, self.mean, np.nan), std,
                          np.where(n > 0, self.lo, np.nan), np.where(n > 0, self.hi, np.nan), list(self.scenes))
