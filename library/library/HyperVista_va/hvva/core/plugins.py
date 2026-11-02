"""Plugin discovery.  Drop a .py file in the plugins/ folder that defines a PLUGIN dict:

    PLUGIN = {
        "name": "My method",
        "kind": "correction",            # or "analysis"
        "description": "one line",
        "params": {"window": {"type": "int", "default": 5, "min": 1, "max": 99}},
        "run": run,                      # run(cube, wavelengths, **params)
    }

correction -> run returns a cube (rows, cols, bands[, new bands]); wavelengths may be returned as
              a tuple (cube, wavelengths) when the band count changes.
analysis   -> run returns a 2-D image, or a dict {"image": 2-D array, "table": {...}, "plot": (x, y)}.
"""
from __future__ import annotations

import importlib.util
import os
import traceback
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple

import numpy as np

DEFAULT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "plugins")


@dataclass
class Plugin:
    name: str
    kind: str
    description: str
    params: Dict[str, dict]
    run: Callable
    file: str = ""

    def default_params(self):
        return {k: v.get("default") for k, v in self.params.items()}

    def apply(self, scene, **params):
        """Run on a Scene; returns a new Scene (correction) or the analysis dict."""
        from .scene import Scene
        p = {**self.default_params(), **params}
        out = self.run(scene.cube.copy() if self.kind == "correction" else scene.cube, scene.wavelengths, **p)
        if self.kind == "correction":
            wl = scene.wavelengths
            if isinstance(out, tuple):
                out, wl = out
            out = np.asarray(out, np.float32)
            if out.ndim != 3 or out.shape[:2] != scene.cube.shape[:2]:
                raise ValueError("a correction plugin must return a cube with the same rows and columns")
            return Scene(f"{scene.name}+{self.name}", out, wl, scene.geo, scene.labels, dict(scene.class_names),
                         list(scene.rois))
        if not isinstance(out, dict):
            out = {"image": np.asarray(out)}
        return out


def _coerce(spec, text):
    t = spec.get("type", "float")
    return {"int": int, "float": float, "bool": lambda s: str(s).lower() in ("1", "true", "yes"), "str": str}[t](text)


def discover(folder=None) -> Tuple[List[Plugin], List[str]]:
    """Import every .py in *folder*; returns (plugins, error messages).  A broken plugin never stops the app."""
    folder = folder or DEFAULT_DIR
    plugins, errors = [], []
    if not os.path.isdir(folder):
        return plugins, [f"plugin folder not found: {folder}"]
    for fn in sorted(os.listdir(folder)):
        if not fn.endswith(".py") or fn.startswith("_"):
            continue
        path = os.path.join(folder, fn)
        try:
            spec = importlib.util.spec_from_file_location(f"hvva_plugin_{fn[:-3]}", path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            d = mod.PLUGIN
            if d.get("kind") not in ("correction", "analysis"):
                raise ValueError("kind must be 'correction' or 'analysis'")
            plugins.append(Plugin(d["name"], d["kind"], d.get("description", ""), d.get("params", {}), d["run"], path))
        except Exception as e:                                       # noqa: BLE001
            errors.append(f"{fn}: {e}")
    return plugins, errors
