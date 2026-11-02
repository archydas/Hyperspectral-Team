"""Example ANALYSIS plugin: NDVI image from the bands nearest to the given wavelengths."""
import numpy as np


def run(cube, wavelengths, nir=800.0, red=670.0):
    if wavelengths is None:
        raise ValueError("this scene has no wavelengths")
    i_nir = int(np.argmin(np.abs(wavelengths - nir)))
    i_red = int(np.argmin(np.abs(wavelengths - red)))
    n, r = cube[:, :, i_nir], cube[:, :, i_red]
    ndvi = (n - r) / (n + r + 1e-12)
    return {"image": ndvi, "table": {"mean NDVI": float(np.nanmean(ndvi)),
                                     "NIR band (nm)": float(wavelengths[i_nir]), "red band (nm)": float(wavelengths[i_red])}}


PLUGIN = {
    "name": "NDVI",
    "kind": "analysis",
    "description": "Normalised difference vegetation index (NIR - red) / (NIR + red).",
    "params": {"nir": {"type": "float", "default": 800.0, "min": 400, "max": 2500},
               "red": {"type": "float", "default": 670.0, "min": 400, "max": 2500}},
    "run": run,
}
