import json, numpy as np, pytest
from conftest_ingest_helpers import *
from ds3.ingest import (open_cube, extract_roi, write_zarr, run_qc, MissingWavelengthsError,
                        InvalidHeaderError, UnsupportedFormatError)
from ds3.ingest.metadata import parse_wavelength_strings, water_absorption_mask

T = truth()

def close(a, b, tol=2e-4):
    assert np.allclose(a, b, atol=tol, equal_nan=True), np.nanmax(np.abs(a - b))

# ---------------- ENVI
@pytest.mark.parametrize("il", ["bsq", "bil", "bip"])
def test_envi_interleaves_int16_scale(tmp, il):
    cube = open_cube(write_envi(tmp / f"c_{il}.img", il))
    assert cube.shape == (R, C, B) and cube.unit == "reflectance" and cube.source_format == "envi"
    close(cube.data, T)
    close(cube.wavelengths_nm, WL, 1e-3)
    assert cube.fwhm_nm is not None and np.all(cube.fwhm_nm == 10.0)

@pytest.mark.parametrize("il", ["bsq", "bil", "bip"])
def test_envi_lazy_window_equals_full(tmp, il):
    cube = open_cube(write_envi(tmp / "c.img", il))
    full = cube.data
    close(cube.read(rows=(3, 8), cols=(2, 7)), full[3:8, 2:7])
    close(cube.read(rows=(3, 8), cols=(2, 7), bands=slice(10, 20)), full[3:8, 2:7, 10:20])

def test_envi_bbl_to_bad_band_mask(tmp):
    cube = open_cube(write_envi(tmp / "c.img"), qc=False)
    assert cube.bad_band_mask[[5, 6]].all()

def test_envi_float32_infers_unit_and_notes(tmp):
    cube = open_cube(write_envi(tmp / "c.img", dtype="f4"))
    assert cube.unit == "reflectance"
    assert any("inferred" in n for n in cube.metadata["notes"])

def test_envi_micrometers_converted(tmp):
    wl_um = WL / 1000
    cube = open_cube(write_envi(tmp / "c.img", wl=wl_um, units="Micrometers"))
    close(cube.wavelengths_nm, WL, 1e-2)
    close(cube.fwhm_nm, np.full(B, 10000.0), 1)  # header fwhm unit follows wavelengths

def test_envi_descending_wavelengths_reordered(tmp):
    d = T[..., ::-1].copy()
    cube = open_cube(write_envi(tmp / "c.img", wl=WL[::-1], data=d), qc=False)
    assert np.all(np.diff(cube.wavelengths_nm) > 0)
    close(cube.data, T)           # after reorder, identical to ascending truth

def test_envi_nodata_to_nan(tmp):
    d = T.copy(); d[2, 3, :] = np.nan
    cube = open_cube(write_envi(tmp / "c.img", nodata=-9999, data=d))
    out = cube.data
    assert np.all(np.isnan(out[2, 3])) and not np.isnan(out[0, 0]).any()
    assert cube.qc.n_nodata_pixels == 1

def test_envi_geotransform_and_xy(tmp):
    cube = open_cube(write_envi(tmp / "c.img"))
    assert cube.transform == (30.0, 0.0, 500000.0, 0.0, -30.0, 1000000.0)
    assert cube.xy_to_pixel(500000 + 30 * 4 + 5, 1000000 - 30 * 2 - 5) == (2, 4)

def test_envi_size_mismatch_raises(tmp):
    p = write_envi(tmp / "c.img")
    open(str(tmp / "c.img"), "r+b").truncate(100)
    with pytest.raises(InvalidHeaderError):
        open_cube(p)

def test_envi_wavelength_count_mismatch_raises(tmp):
    p = write_envi(tmp / "c.img", wl=WL)
    txt = open(p).read().replace(f"bands = {B}", f"bands = {B+1}")
    open(p, "w").write(txt)
    with pytest.raises(InvalidHeaderError):
        open_cube(p)

def test_open_by_binary_path_or_hdr(tmp):
    write_envi(tmp / "c.img")
    a = open_cube(tmp / "c.img"); b = open_cube(tmp / "c.img.hdr")
    close(a.data, b.data)

# ---------------- GeoTIFF
def test_geotiff_band_descriptions(tmp):
    cube = open_cube(write_tif(tmp / "a.tif"))
    close(cube.data, T, 1e-6); close(cube.wavelengths_nm, WL, 1e-2)
    assert cube.crs == "EPSG:32643" and cube.transform[:3] == (30.0, 0.0, 500000.0)
    assert any("band descriptions" in n for n in cube.metadata["notes"])

def test_geotiff_dataset_tag_micrometers(tmp):
    cube = open_cube(write_tif(tmp / "a.tif", desc=False, tags_wl=True))
    close(cube.wavelengths_nm, WL, 0.01)

def test_geotiff_envi_tag_domain(tmp):
    cube = open_cube(write_tif(tmp / "a.tif", desc=False, env_tags=True))
    close(cube.wavelengths_nm, WL, 1e-2)

def test_geotiff_missing_wavelengths_clear_error_and_override(tmp):
    p = write_tif(tmp / "a.tif", desc=False)
    with pytest.raises(MissingWavelengthsError, match="wavelengths="):
        open_cube(p)
    cube = open_cube(p, wavelengths=WL)
    close(cube.wavelengths_nm, WL)

def test_geotiff_band_names_not_misread_as_wavelengths():
    assert parse_wavelength_strings([f"Band_{i}" for i in range(1, 6)]) == (None, False)
    assert parse_wavelength_strings([str(i) for i in range(1, 6)]) == (None, False)

def test_geotiff_nodata(tmp):
    d = T.copy(); d[0, :, :] = -9999
    cube = open_cube(write_tif(tmp / "a.tif", d=d, nodata=-9999.0))
    assert np.isnan(cube.data[0]).all() and not np.isnan(cube.data[1]).any()

def test_geotiff_int16_reflectance_divisor(tmp):
    d = np.round(T * 10000).astype("int16")
    cube = open_cube(write_tif(tmp / "a.tif", d=d, dtype="int16"), unit="reflectance", reflectance_divisor=10000)
    close(cube.data, T, 2e-4)

def test_geotiff_int_without_divisor_is_dn(tmp):
    d = np.round(T * 10000).astype("int16")
    cube = open_cube(write_tif(tmp / "a.tif", d=d, dtype="int16"))
    assert cube.unit == "dn"

# ---------------- Zarr / HDF
def test_zarr_read(tmp):
    cube = open_cube(write_zarr_store(tmp / "s.zarr"))
    close(cube.data, T, 1e-6); assert cube.unit == "reflectance"

def test_emit_like_netcdf(tmp):
    cube = open_cube(write_emit_like(tmp / "emit.nc"))
    assert cube.sensor == "EMIT" and cube.metadata["layout"] == "emit"
    assert cube.bad_band_mask[[5, 6]].all()
    out = cube.data
    assert np.isnan(out[0, 0]).all()          # -9999 fill -> NaN
    close(out[1:], T[1:])
    assert cube.qc.n_nodata_pixels == 1

def test_generic_hdf5(tmp):
    import h5py
    p = tmp / "g.h5"
    with h5py.File(p, "w") as f:
        f.create_dataset("cube/refl", data=np.moveaxis(T, -1, 0))      # (bands, rows, cols)
        f.create_dataset("cube/wavelength", data=WL)
    cube = open_cube(p)
    assert cube.shape == (R, C, B)
    close(cube.data, T, 1e-6)

# ---------------- round trip / standardisation
@pytest.mark.parametrize("src", ["envi", "tif"])
def test_convert_to_standard_zarr_roundtrip(tmp, src):
    p = write_envi(tmp / "c.img", "bil") if src == "envi" else write_tif(tmp / "a.tif")
    a = open_cube(p)
    out = write_zarr(a, tmp / "std.zarr")
    b = open_cube(out)
    close(a.data, b.data, 1e-6); close(a.wavelengths_nm, b.wavelengths_nm, 1e-6)
    assert np.array_equal(a.bad_band_mask, b.bad_band_mask | a.bad_band_mask)
    assert b.unit == a.unit

# ---------------- QC
def test_qc_flags_problem_bands_without_modifying_data(tmp):
    d = T.copy(); d[..., 20] = np.nan; d[..., 21] = 0.3; d[..., 22] = 5.0
    cube = open_cube(write_tif(tmp / "a.tif", d=d), unit="reflectance")
    codes = {i["code"] for i in cube.qc.issues}
    assert {"NAN_BANDS", "CONSTANT_BANDS", "OVER_RANGE"} <= codes
    assert cube.bad_band_mask[[20, 21, 22]].all()
    assert np.isnan(cube.data[..., 20]).all()      # data untouched

def test_qc_water_bands_flagged_not_removed(tmp):
    cube = open_cube(write_envi(tmp / "c.img"))
    w = water_absorption_mask(cube.wavelengths_nm)
    assert w.any() and cube.bad_band_mask[w].all() and cube.bands == B

def test_qc_catches_unscaled_dn_declared_reflectance(tmp):
    d = np.round(T * 10000).astype("float32")
    cube = open_cube(write_tif(tmp / "a.tif", d=d), unit="reflectance")
    assert not cube.qc.passed and any(i["code"] == "UNIT_SUSPECT" for i in cube.qc.issues)

def test_qc_json_serialisable(tmp):
    cube = open_cube(write_envi(tmp / "c.img"))
    json.loads(cube.qc.to_json())

# ---------------- ROI
def test_roi_pixel_and_bbox(tmp):
    cube = open_cube(write_envi(tmp / "c.img"))
    r = extract_roi(cube, pixel=(3, 4)); close(r.spectra[0], T[3, 4]); assert r.n_pixels == 1
    r = extract_roi(cube, bbox_px=(2, 3, 6, 8))
    assert r.n_pixels == 4 * 5
    close(r.mean, T[2:6, 3:8].mean(axis=(0, 1))); close(r.std, T[2:6, 3:8].std(axis=(0, 1)), 5e-4)
    assert r.pixel_coords[0].tolist() == [2, 3]

def test_roi_xy(tmp):
    cube = open_cube(write_envi(tmp / "c.img"))
    r = extract_roi(cube, xy=(500000 + 30 * 4 + 5, 1000000 - 30 * 2 - 5))
    close(r.spectra[0], T[2, 4])

def test_roi_polygon_map_coords_and_pixel_coords_agree(tmp):
    cube = open_cube(write_envi(tmp / "c.img"))
    pix = [(2, 2), (7, 2), (7, 6), (2, 6)]               # (col,row)
    r_pix = extract_roi(cube, polygon=pix, polygon_coords="pixel")
    mp = [(500000 + 30 * x, 1000000 - 30 * y) for x, y in pix]
    r_map = extract_roi(cube, polygon=mp, polygon_coords="map")
    assert r_pix.n_pixels == r_map.n_pixels == 5 * 4
    close(r_pix.mean, r_map.mean)

def test_roi_polygon_geojson_feature(tmp):
    cube = open_cube(write_envi(tmp / "c.img"))
    feat = {"type": "Feature", "properties": {}, "geometry": {"type": "Polygon", "coordinates": [[[2, 2], [5, 2], [5, 5], [2, 5], [2, 2]]]}}
    assert extract_roi(cube, polygon=feat, polygon_coords="pixel").n_pixels == 9

def test_roi_excludes_nodata_and_reports(tmp):
    d = T.copy(); d[3, 4, :] = np.nan; d[3, 5, :] = np.nan
    cube = open_cube(write_envi(tmp / "c.img", nodata=-9999, data=d))
    r = extract_roi(cube, bbox_px=(3, 3, 4, 8))
    assert r.n_pixels == 3 and r.n_excluded_nodata == 2 and not np.isnan(r.mean).any()

def test_roi_outside_returns_empty(tmp):
    cube = open_cube(write_envi(tmp / "c.img"))
    r = extract_roi(cube, polygon=[(50, 50), (60, 50), (60, 60)], polygon_coords="pixel")
    assert r.n_pixels == 0
    with pytest.raises(ValueError):
        extract_roi(cube, pixel=(99, 99))

def test_roi_mask_and_selector_validation(tmp):
    cube = open_cube(write_envi(tmp / "c.img"))
    m = np.zeros((R, C), bool); m[1, 1] = m[5, 5] = True
    assert extract_roi(cube, mask=m).n_pixels == 2
    with pytest.raises(ValueError):
        extract_roi(cube, pixel=(1, 1), mask=m)

def test_roi_to_library_spectrum(tmp):
    cube = open_cube(write_envi(tmp / "c.img"))
    s = extract_roi(cube, bbox_px=(0, 0, 4, 4)).to_spectrum("vegetation_patch", category="vegetation")
    assert s.purity == "image_pixel" and s.unit == "reflectance" and len(s.values) == B
    assert s.uncertainty is not None
    json.dumps(s.to_dict())

# ---------------- API misc
def test_unsupported_and_missing(tmp):
    (tmp / "x.xyz").write_text("hi")
    with pytest.raises(UnsupportedFormatError):
        open_cube(tmp / "x.xyz")

def test_cli_info_and_convert(tmp, capsys):
    from ds3.cli import main
    p = write_envi(tmp / "c.img")
    main(["info", p]); assert "QC" in capsys.readouterr().out
    main(["convert", p, str(tmp / "o.zarr")]); assert (tmp / "o.zarr").exists()
