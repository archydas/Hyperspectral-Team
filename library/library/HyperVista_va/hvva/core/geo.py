"""Georeferencing: a CRS plus an affine transform that travels with every scene.

Transform convention (same as rasterio / GDAL):
    x = a*col + b*row + c
    y = d*col + e*row + f
(c, f) is the upper-left *corner* of pixel (0, 0).  North-up images have b = d = 0
and e < 0.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class GeoInfo:
    crs: Optional[str] = None                      # "EPSG:32633", WKT, or free text
    transform: Tuple[float, ...] = (1.0, 0.0, 0.0, 0.0, -1.0, 0.0)
    map_info: Optional[str] = None                 # raw ENVI "map info" text, kept for round trips

    # ---- basic geometry ------------------------------------------------
    @property
    def pixel_size(self):
        a, b, c, d, e, f = self.transform
        return math.hypot(a, d), math.hypot(b, e)

    @property
    def is_north_up(self):
        return abs(self.transform[1]) < 1e-12 and abs(self.transform[3]) < 1e-12

    def xy(self, row, col, center=True):
        a, b, c, d, e, f = self.transform
        o = 0.5 if center else 0.0
        col, row = col + o, row + o
        return a * col + b * row + c, d * col + e * row + f

    def rowcol(self, x, y):
        """Fractional (row, col) of a map coordinate (inverse transform)."""
        a, b, c, d, e, f = self.transform
        det = a * e - b * d
        if abs(det) < 1e-18:
            raise ValueError("singular geotransform")
        col = (e * (x - c) - b * (y - f)) / det
        row = (a * (y - f) - d * (x - c)) / det
        return row, col

    def bounds(self, height, width):
        """(xmin, ymin, xmax, ymax) of the image footprint."""
        pts = [self.xy(r, c, center=False) for r in (0, height) for c in (0, width)]
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        return min(xs), min(ys), max(xs), max(ys)

    def shifted(self, row0, col0):
        """GeoInfo of a sub-window whose upper-left pixel is (row0, col0) of this image."""
        x, y = self.xy(row0, col0, center=False)
        a, b, _, d, e, _ = self.transform
        return GeoInfo(self.crs, (a, b, x, d, e, y), self.map_info)

    def same_crs(self, other: "GeoInfo") -> bool:
        return (self.crs or "").strip().lower() == (other.crs or "").strip().lower()

    # ---- (de)serialisation ----------------------------------------------
    def to_dict(self):
        return {"crs": self.crs, "transform": list(self.transform), "map_info": self.map_info}

    @classmethod
    def from_dict(cls, d):
        if not d:
            return None
        return cls(d.get("crs"), tuple(float(v) for v in d["transform"]), d.get("map_info"))

    # ---- ENVI "map info" -----------------------------------------------------
    @classmethod
    def from_envi_map_info(cls, text, coord_sys=None):
        """Parse e.g. '{UTM, 1.0, 1.0, 500000, 4100000, 30, 30, 33, North, WGS-84}'."""
        raw = text.strip().strip("{}")
        p = [s.strip() for s in raw.split(",")]
        if len(p) < 7:
            raise ValueError(f"cannot parse map info: {text!r}")
        name = p[0]
        rx, ry, east, north, xs, ys = (float(v) for v in p[1:7])
        x0 = east - (rx - 1.0) * xs            # ENVI reference pixel (1,1) = upper-left corner
        y0 = north + (ry - 1.0) * ys
        crs = None
        datum = p[-1] if len(p) > 9 else ""
        if name.lower().startswith("geographic"):
            crs = "EPSG:4326" if "84" in datum or not datum else f"ENVI:{name} {datum}"
        elif name.upper() == "UTM" and len(p) >= 9:
            zone, hemi = int(float(p[7])), p[8].lower()
            if "84" in datum:
                crs = f"EPSG:{32600 + zone if hemi.startswith('n') else 32700 + zone}"
            else:
                crs = f"UTM zone {zone} {p[8]} {datum}".strip()
        else:
            crs = f"ENVI:{name}"
        if coord_sys:
            crs = coord_sys.strip().strip("{}")
        return cls(crs, (xs, 0.0, x0, 0.0, -ys, y0), text.strip())

    def to_envi_map_info(self):
        if self.map_info:
            return self.map_info
        xs, ys = self.pixel_size
        x0, y0 = self.transform[2], self.transform[5]
        crs = (self.crs or "").upper()
        if crs.startswith("EPSG:326") or crs.startswith("EPSG:327"):
            z = int(crs[-2:])
            hemi = "North" if crs.startswith("EPSG:326") else "South"
            return f"{{UTM, 1.0, 1.0, {x0:.6f}, {y0:.6f}, {xs:.6f}, {ys:.6f}, {z}, {hemi}, WGS-84}}"
        if crs == "EPSG:4326":
            return f"{{Geographic Lat/Lon, 1.0, 1.0, {x0:.10f}, {y0:.10f}, {xs:.10f}, {ys:.10f}, WGS-84}}"
        return f"{{Arbitrary, 1.0, 1.0, {x0:.6f}, {y0:.6f}, {xs:.6f}, {ys:.6f}}}"

    def describe(self, height=None, width=None):
        xs, ys = self.pixel_size
        s = f"CRS: {self.crs or 'unknown'}\nPixel size: {xs:g} x {ys:g}\n" \
            f"Origin (upper-left): {self.transform[2]:.3f}, {self.transform[5]:.3f}"
        if height and width:
            b = self.bounds(height, width)
            s += f"\nBounds: x {b[0]:.1f} .. {b[2]:.1f}, y {b[1]:.1f} .. {b[3]:.1f}"
        return s
