"""Category paths and acquisition tags.

A category is a path with "/" between levels, e.g. ``Vegetation/Tree/Tsuga``. It is stored as plain text in
``materials.category`` so no extra table is needed and older databases keep working.
"""
from __future__ import annotations
import re

ACQUISITION_TYPES = ("lab", "field", "airborne", "satellite")
NONE_TOKEN = "__none__"                   # filter value meaning "not categorised" / "not tagged"

_SEPARATORS = re.compile(r"\s*(?:>|->|\||\\|/)\s*")


def normalize_category(text: str | None) -> str | None:
    """'Vegetation > Tree | Tsuga' -> 'Vegetation/Tree/Tsuga'. Empty input gives None."""
    if text is None:
        return None
    parts = [p.strip() for p in _SEPARATORS.split(str(text)) if p and p.strip()]
    return "/".join(parts) or None


def normalize_acquisition(value: str | None) -> str | None:
    if value is None or str(value).strip() == "":
        return None
    v = str(value).strip().lower()
    if v not in ACQUISITION_TYPES:
        raise ValueError(f"acquisition type must be one of {', '.join(ACQUISITION_TYPES)} (got {value!r})")
    return v


_HINTS = (
    ("field", (".asd", ".sig", ".sed", "fieldspec", "spectra vista", "field")),
    ("lab", ("ecostress", "usgs", "splib", "ossl", "lab")),
    ("airborne", ("aviris", "uav", "airborne", "hyspex", "aisa", "casi", "ang2")),
    ("satellite", ("enmap", "prisma", "sentinel", "landsat", "emit", "hyperion", "desis", "satellite")),
)


def infer_acquisition(*texts: str | None) -> str | None:
    """Best guess from file / instrument names; None when nothing matches (it is never invented)."""
    blob = " ".join(t.lower() for t in texts if t)
    for kind, keys in _HINTS:
        if any(k in blob for k in keys):
            return kind
    return None


def category_filter(column, path: str | None):
    """SQLAlchemy condition: the category node `path` and everything below it. NONE_TOKEN = uncategorised."""
    if not path:
        return None
    if path == NONE_TOKEN:
        return column.is_(None)
    p = normalize_category(path)
    esc = p.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return (column == p) | column.like(esc + "/%", escape="\\")


def rollup(rows):
    """[(category, n)] -> {path: total} with every count added to all its ancestors (for the category tree)."""
    out: dict[str, int] = {}
    for cat, n in rows:
        if not cat:
            out[NONE_TOKEN] = out.get(NONE_TOKEN, 0) + n
            continue
        parts = cat.split("/")
        for i in range(1, len(parts) + 1):
            k = "/".join(parts[:i])
            out[k] = out.get(k, 0) + n
    return out
