"""Tiny numpy k-means so a scene can get quick classes without extra dependencies."""
from __future__ import annotations

import numpy as np


def kmeans_labels(cube, k=4, seed=0, max_iter=30, sample=20000):
    """Label map (rows, cols) with values 1..k (NaN pixels -> 0)."""
    h, w, b = cube.shape
    X = cube.reshape(-1, b).astype(np.float32)
    ok = ~np.isnan(X).any(1)
    rng = np.random.default_rng(seed)
    idx = np.flatnonzero(ok)
    sub = X[rng.choice(idx, min(sample, len(idx)), replace=False)]
    cent = [sub[rng.integers(len(sub))]]                      # k-means++ seeding
    for _ in range(1, k):
        d = np.min([((sub - c) ** 2).sum(1) for c in cent], axis=0)
        cent.append(sub[rng.choice(len(sub), p=d / d.sum())] if d.sum() > 0 else sub[rng.integers(len(sub))])
    cent = np.array(cent)
    for _ in range(max_iter):
        lab = np.argmin(((sub[:, None, :] - cent[None]) ** 2).sum(2), 1)
        new = np.array([sub[lab == j].mean(0) if (lab == j).any() else cent[j] for j in range(k)])
        if np.allclose(new, cent):
            break
        cent = new
    out = np.zeros(h * w, np.int32)
    chunk = 50000
    for s in range(0, len(idx), chunk):
        sel = idx[s:s + chunk]
        out[sel] = np.argmin(((X[sel][:, None, :] - cent[None]) ** 2).sum(2), 1) + 1
    return out.reshape(h, w)
