"""Example CORRECTION plugin: subtract a scalar dark value from every pixel and band."""
import numpy as np


def run(cube, wavelengths, dark=0.01):
    return np.clip(cube - dark, 0, None)


PLUGIN = {
    "name": "Dark value subtraction",
    "kind": "correction",
    "description": "Subtracts a constant dark value (reflectance units) and clips at zero.",
    "params": {"dark": {"type": "float", "default": 0.01, "min": 0.0, "max": 100000.0}},
    "run": run,
}
