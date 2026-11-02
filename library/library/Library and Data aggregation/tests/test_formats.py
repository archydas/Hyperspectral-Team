"""Universal spectral-data readers: every format -> nm, 0-1 reflectance, names, categories, acquisition."""
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))
import spectral_fixtures  # noqa: E402

from hvva.library import formats  # noqa: E402
from hvva.library.formats import ReadOptions, read_path  # noqa: E402

WL = np.arange(400, 2501, 10.0)


def veg(k=1.0):
    return np.clip(0.05 + 0.4 * k / (1 + np.exp(-(WL - 715) / 15)) - 0.15 * np.exp(-((WL - 1450) / 60) ** 2)
                   - 0.2 * np.exp(-((WL - 1940) / 80) ** 2), 0.01, 1)


@pytest.fixture(scope="module")
def fx(tmp_path_factory):
    return Path(spectral_fixtures.build(tmp_path_factory.mktemp("formats") / "fx"))


def recs(path, **opts):
    r, info = read_path(path, ReadOptions(**opts))
    return {x.name: x for x in r}, info


def close(rec, truth, tol=2e-3):
    return np.allclose(np.interp(WL, rec.wavelengths, rec.values), truth, atol=tol)


@pytest.mark.parametrize("name,reader,n", [
    ("wide_columns.csv", "table", 4), ("rows_with_metadata.csv", "table", 4), ("long_format.csv", "table", 4),
    ("european_percent.csv", "table", 2), ("single_spectrum.txt", "table", 1), ("wavenumber_x10000.tsv", "table", 1),
    ("spectra_list.json", "json", 4), ("envi_int16_be.hdr", "envi_sli", 4), ("envi_int16_be.sli", "envi_sli", 4),
    ("image_envi.hdr", "image", 1), ("scene_with_labels.npz", "image", 3), ("field", "folder", 4),
    ("hvva_export", "folder", 12), ("bundle.zip", "zip", 5)])
def test_detect_and_count(fx, name, reader, n):
    r, score, _ = formats.detect(fx / name)
    assert r is not None and r.name == reader
    got, info = recs(fx / name)
    assert info.n_kept == n


def test_junk_is_not_spectral(fx):
    for n in ("readme.txt", "notes.csv"):
        assert formats.detect(fx / n)[0] is None


def test_wide_and_hvva_exports_round_trip(fx):
    for f in ("wide_columns.csv", "hvva_export/signatures.csv", "hvva_export/signatures.json", "hvva_export/signatures.sli"):
        got, info = recs(fx / f)
        assert close(got["Oak leaf"], veg(1)), f
        assert got["Oak leaf"].wavelengths[0] == pytest.approx(400) and got["Oak leaf"].wavelengths[-1] == pytest.approx(2500)
    got, _ = recs(fx / "hvva_export/signatures.sli")
    assert got["Oak leaf"].std is not None and np.allclose(got["Oak leaf"].std, 0.01)    # *_std.sli companion
    got, _ = recs(fx / "hvva_export/signatures.csv")
    assert got["Oak leaf"].std is not None and str(got["Oak leaf"].material_metadata.get("n_pixels")) == "50"


def test_rows_layout_metadata(fx):
    got, info = recs(fx / "rows_with_metadata.csv")
    assert info.wl_unit == "um" and close(got["Oak leaf"], veg(1))
    assert got["Oak leaf"].category == "Vegetation/Tree/Quercus" and got["Oak leaf"].acquisition == "field"
    assert got["Pine needles"].acquisition == "lab" and got["Copper sheet"].acquisition == "airborne"   # "UAV"
    assert got["Oak leaf"].material_metadata.get("sample_id") == "S-8"


def test_long_european_wavenumber_single(fx):
    got, _ = recs(fx / "long_format.csv")
    assert got["Oak leaf"].category == "Demo" and close(got["Oak leaf"], veg(1), tol=0.03)   # 20 nm sampling
    got, info = recs(fx / "european_percent.csv")
    assert "percent" in info.scale and close(got["Probe A"], veg(1))
    got, info = recs(fx / "wavenumber_x10000.tsv")
    g = got["Granite"]
    assert info.wl_unit == "cm-1" and g.wavelengths[0] == pytest.approx(400, abs=0.01)
    assert np.nanmax(g.values) == pytest.approx(0.25, abs=1e-3)                  # ×10000 -> 0-1
    got, info = recs(fx / "single_spectrum.txt")
    assert info.wl_unit == "um" and len(got) == 1


def test_overrides(fx):
    got, info = recs(fx / "european_percent.csv", value_scale="none")
    assert np.nanmax(got["Probe A"].values) > 40                                  # kept as measured
    got, _ = recs(fx / "wide_columns.csv", category="Lab/Test", acquisition="satellite", force_acquisition=True)
    assert all(r.category == "Lab/Test" and r.acquisition == "satellite" for r in got.values())
    got, _ = recs(fx / "wide_columns.csv", spectral_range="vswir", limit=2)
    assert len(got) == 2
    with pytest.raises(ValueError):
        recs(fx / "wide_columns.csv", orientation="rows")


def test_envi_big_endian_scaled(fx):
    got, info = recs(fx / "envi_int16_be.hdr")
    assert info.wl_unit == "um" and close(got["Oak leaf"], veg(1), tol=1e-4)


def test_images(fx):
    got, _ = recs(fx / "scene_with_labels.npz")
    assert set(got) == {"soil", "vegetation", "water"} and got["vegetation"].std is not None
    assert got["vegetation"].material_metadata["n_pixels"] > 0
    got, _ = recs(fx / "image_envi.hdr", image_mode="sample", image_samples=7)
    assert len(got) == 7 and all("pixel (" in n for n in got)
    got, _ = recs(fx / "scene_with_labels.npz", image_mode="mean")
    assert len(got) == 1


def test_field_spectrometers_and_folder_categories(fx):
    got, info = recs(fx / "field")
    maple = got["maple 00001"]                                        # ASD as7, DN / white reference
    assert maple.category == "Vegetation/Tree" and close(maple, veg(1.1), tol=1e-6)
    assert got["sand 00002"].category == "Soil"                        # ASD v1 reflectance
    assert got["grass_001"].acquisition == "field" and got["grass_001"].category == "Vegetation"   # SVC .sig, percent
    assert np.nanmax(got["grass_001"].values) < 1
    assert got["concrete"].category is None and np.nanmax(got["concrete"].values) < 1     # SED Reflect. %


def test_asd_matches_specdal(fx):
    specdal = pytest.importorskip("specdal")
    df, _ = specdal.read(str(fx / "field/Vegetation/Tree/maple_00001.asd"))
    got, _ = recs(fx / "field/Vegetation/Tree/maple_00001.asd")
    r = next(iter(got.values()))
    assert np.allclose(df.iloc[:, 0] / df.iloc[:, 1], np.interp(df.index, r.wavelengths, r.values))


def test_zip_folder_categories(fx):
    got, _ = recs(fx / "bundle.zip")
    assert got["Oak leaf"].category == "Lab/Batch1"


@pytest.mark.parametrize("name", ["library_matrix.mat", "library_matrix.h5", "workbook.xlsx", "lists.parquet",
                                  "wide.parquet"])
def test_optional_formats(fx, name):
    if not (fx / name).exists():
        pytest.skip("optional dependency missing")
    got, info = recs(fx / name)
    assert info.n_kept >= 3
    oak = got.get("Oak leaf") or got.get("Vegetation Oak leaf") or next(iter(got.values()))
    assert close(oak, veg(1))
    if name == "library_matrix.mat":
        assert "Oak leaf" in got                                       # names from a cell array
    if name == "lists.parquet":
        assert got["Dry soil"].category == "Soil"


def test_ingest_everything_into_db(fx, tmp_path):
    """Whole folder through sources -> ingest -> database, then nothing new on a second pass."""
    from hvva.library.client import DirectClient
    c = DirectClient(f"sqlite:///{tmp_path / 'lib.db'}", init_schema=True)
    srcs = {s["name"]: s for s in c.scan_folder(fx)}
    assert {"readme.txt", "notes.csv", "image_envi.img", "envi_int16_be.sli"}.isdisjoint(srcs)
    assert srcs["field"]["kind"] == "folder" and srcs["bundle.zip"]["kind"] == "zip"
    res = c.import_sources([{"source": s, "library": s["library"], "acquisition": "lab"} for s in srcs.values()])
    assert all(r["ok"] for r in res), [r for r in res if not r["ok"]]
    assert sum(r["spectra_added"] for r in res) >= 50
    assert all(s["status"] == "imported" for s in c.scan_folder(fx))
    assert c.import_new_from_folder(fx) == []
    lib = next(l for l in c.list_libraries() if l["name"] == "Field")
    f = c.library_facets(lib["id"])
    assert {x["path"] for x in f["categories"]} == {"Soil", "Vegetation", "Vegetation/Tree"}
    assert f["acquisition"]["field"] == 1 and f["acquisition"]["lab"] == 3     # inferred beats the default
    one = c.import_file(fx / "rows_with_metadata.csv", library_name="Single", options={"category": "X"})
    assert one["spectra_added"] == 4 and one["reader"] == "table"
    c.close()
