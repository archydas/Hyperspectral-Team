"""Material category guess from the material NAME (keyword rule).

This is a transparent HEURISTIC, not a classification model: the first matching rule wins and anything that
does not match stays None ('unspecified'). The category is stored on the material; the sample metadata records
`category_source: keyword_rule` so it can be told apart from a category supplied by the data provider.
Edit RULES to change behaviour - order matters.
"""
from __future__ import annotations

RULES = [
    ("metal", ("aluminum", "aluminium", "steel", "copper", "brass", "bronze", "zinc", "galvanized", "tin foil", " metal", "iron sheet")),
    ("paint_coating", ("paint", "coating", "varnish", "lacquer", "primer")),
    ("construction", ("concrete", "brick", "asphalt", "cement", "glass", "shingle", "roof", "tile", "slate", "plaster", "gravel",
                      "cinder", "tar ", "tar_", "wood", "plywood", "lumber", "paving", "pavement", "marble", "road")),
    ("polymer_fabric", ("rubber", "plastic", "polyethylene", "pvc", "nylon", "fabric", "cotton", "polyester", "carpet", "fiberglass", "canvas")),
    ("vegetation", ("grass", "leaf", "leaves", "tree", "vegetation", "conifer", "crop", "wheat", "corn", "maize", "soybean",
                    "alfalfa", "hay", "moss", "lichen", "shrub", "forest", "canopy", "weed", "oak ", "birch", "maple")),
    ("water_snow", ("water", "snow", "ice", "frost")),
    ("soil", ("soil", "loam", "dirt", "sand ", "sandy")),
    ("mineral_rock", ("quartz", "calcite", "kaolinite", "alunite", "gypsum", "hematite", "goethite", "muscovite", "montmorillonite",
                      "dolomite", "basalt", "granite", "limestone", "sandstone", "rock", "mineral", "olivine", "pyroxene",
                      "feldspar", "illite", "chlorite", "epidote", "jarosite", "serpentine", "talc")),
]


def guess_category(name: str | None) -> str | None:
    if not name:
        return None
    n = f" {name.lower()} "
    for cat, words in RULES:
        if any(w in n for w in words):
            return cat
    return None
