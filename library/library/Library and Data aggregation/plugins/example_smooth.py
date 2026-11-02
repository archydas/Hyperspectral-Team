"""Example CORRECTION plugin: moving-average smoothing along the spectral axis."""
import numpy as np


def run(cube, wavelengths, window=5):
    k = int(window)
    pad = k // 2
    padded = np.pad(cube, ((0, 0), (0, 0), (pad, pad)), mode="edge")
    csum = np.cumsum(np.pad(padded, ((0, 0), (0, 0), (1, 0))), axis=2)
    return (csum[:, :, k:] - csum[:, :, :-k]) / k


PLUGIN = {
    "name": "Spectral moving average",
    "kind": "correction",
    "description": "Smooths every spectrum with a moving average (window in bands).",
    "params": {"window": {"type": "int", "default": 5, "min": 1, "max": 51}},
    "run": run,
}
