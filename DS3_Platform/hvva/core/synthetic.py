"""Synthetic overlapping, georeferenced scenes for demos and tests (different band sampling per scene)."""
from __future__ import annotations

import numpy as np

from .geo import GeoInfo
from .scene import Scene


def _sig(kind, wl):
    x = wl.astype(float)
    sig = lambda c, w: 1 / (1 + np.exp(-(x - c) / w))
    if kind == "vegetation":
        return 0.04 + 0.45 * sig(715, 12) + 0.06 * np.exp(-((x - 550) / 30) ** 2) - 0.02 * np.exp(-((x - 670) / 20) ** 2)
    if kind == "water":
        return 0.08 * np.exp(-(x - 400) / 220) + 0.01
    if kind == "urban":
        return 0.22 + 0.04 * np.sin(x / 90) + 0.00015 * (x - 400)
    return 0.10 + 0.00045 * (x - 400) + 0.03 * sig(900, 40)              # soil


CLASSES = ["soil", "vegetation", "water", "urban"]
_PLAN = [  # (wavelength start, stop, step) per scene
    (400, 1000, 5), (420, 990, 6), (400, 1000, 4)]


def make_demo_scenes(n=3, seed=0, size=(100, 100)):
    rng = np.random.default_rng(seed)
    H, W = size[0], 220
    world = np.ones((H, W), np.int32)                                  # 1 = soil
    world[10:50, 60:130] = 2                                           # vegetation
    world[55:95, 20:80] = 3                                            # water
    world[55:95, 110:190] = 4                                          # urban
    scenes = []
    for i in range(n):
        lo, hi, st = _PLAN[i % len(_PLAN)]
        wl = np.arange(lo, hi + 1e-6, st, dtype=float)
        c0 = i * 55
        lab = world[:, c0:c0 + size[1]]
        cube = np.zeros(lab.shape + (len(wl),), np.float32)
        gain = rng.uniform(0.95, 1.05)
        for cid, cname in enumerate(CLASSES, 1):
            m = lab == cid
            cube[m] = _sig(cname, wl)[None, :] * gain
        cube *= 1 + rng.normal(0, 0.05, lab.shape + (1,)).astype(np.float32)      # texture
        cube += rng.normal(0, 0.006, cube.shape).astype(np.float32)               # sensor noise
        geo = GeoInfo("EPSG:32633", (30.0, 0.0, 500000.0 + c0 * 30.0, 0.0, -30.0, 4100000.0))
        sc = Scene(f"scene_{chr(65 + i)}", cube, wl, geo, unit="reflectance",
                   unit_note="synthetic demo reflectance", synthetic=True)
        if i < 2:                                                      # labelled by a map
            sc.labels = lab.copy()
            sc.class_names = {k + 1: v for k, v in enumerate(CLASSES)}
        else:                                                          # labelled by ROI rectangles
            for roi in ({"class": "urban", "rect": (60, 5, 90, 70)},
                        {"class": "vegetation", "rect": (15, 0, 45, 18)},
                        {"class": "soil", "rect": (0, 20, 8, 100)}):
                sc.add_roi(roi)
        scenes.append(sc)
    return scenes
