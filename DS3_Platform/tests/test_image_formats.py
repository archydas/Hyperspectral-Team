"""Image module readers (ds3lib.ingest.open_cube): TIFF stacks, NumPy, ERDAS .img, ENVI without wavelengths,
and the 'give the wavelength range' path for files that carry none."""
import numpy as np
import pytest

from ds3lib.ingest import MissingWavelengthsError, open_cube

X = (np.random.default_rng(0).random((64, 72, 50)) * 4000).astype(np.uint16)       # rows, cols, bands (bands = the shortest axis, as in real scenes)


def _check(c, unit=None):
    assert (c.rows, c.cols, c.bands) == (64, 72, 50)
    np.testing.assert_allclose(c.read(rows=(3, 4), cols=(7, 8))[0, 0], X[3, 7].astype(np.float32))
    assert c.wavelengths_nm[0] == pytest.approx(430) and c.wavelengths_nm[-1] == pytest.approx(860)
    if unit:
        assert c.unit == unit


def test_multipage_tiff_needs_range_then_opens(tmp_path):
    tifffile = pytest.importorskip("tifffile")
    p = tmp_path / "stack.tif"
    tifffile.imwrite(p, np.moveaxis(X, 2, 0))                                       # one page per band
    with pytest.raises(MissingWavelengthsError, match="wavelength range"):
        open_cube(p, qc=False)
    _check(open_cube(p, qc=False, wavelength_range=(430, 860)), "dn")


def test_npz_with_wavelength_array(tmp_path):
    p = tmp_path / "cube.npz"
    np.savez(p, data=X, wavelengths=np.linspace(430, 860, 50))
    _check(open_cube(p, qc=False))


def test_npy_with_range(tmp_path):
    p = tmp_path / "cube.npy"
    np.save(p, X)
    _check(open_cube(p, qc=False, wavelength_range=(430, 860)))


def test_envi_without_wavelengths(tmp_path):
    np.moveaxis(X, 2, 0).astype("<u2").tofile(tmp_path / "s.dat")
    (tmp_path / "s.hdr").write_text("ENVI\nsamples = 72\nlines = 64\nbands = 50\nheader offset = 0\n"
                                    "data type = 12\ninterleave = bsq\nbyte order = 0\n")
    with pytest.raises(MissingWavelengthsError):
        open_cube(tmp_path / "s.hdr", qc=False)
    _check(open_cube(tmp_path / "s.hdr", qc=False, wavelength_range=(430, 860)))


def test_erdas_img_through_gdal(tmp_path):
    rasterio = pytest.importorskip("rasterio")
    p = tmp_path / "scene.img"
    with rasterio.open(p, "w", driver="HFA", height=64, width=72, count=50, dtype="uint16") as d:
        d.write(np.moveaxis(X, 2, 0))
    _check(open_cube(p, qc=False, wavelength_range=(430, 860)))


def test_enmap_ancillary_layer_gets_a_clear_message(tmp_path):
    tifffile = pytest.importorskip("tifffile")
    from ds3lib.ingest import IngestError
    p = tmp_path / "ENMAP01-____L2A-DT0000182114_20260311T054111Z_001_V010506_20260531T110835Z-ACOUT_SICOR_CWV.TIF"
    tifffile.imwrite(p, np.ones((20, 30), np.float32))
    with pytest.raises(IngestError, match="water-vapour map.*SPECTRAL_IMAGE"):
        open_cube(p, qc=False)


def test_enmap_style_tiff_with_quoted_units(tmp_path):
    """EnMAP L2A SPECTRAL_IMAGE.TIF: per-band 'wavelength' tags, wavelength_units = "'nanometers'" (with quotes),
    int16 reflectance x 10000, no data -32768."""
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin
    p = tmp_path / "ENMAP01-____L2A-DT0000179049_20260220T054805Z_002_V010506_20260531T110846Z-SPECTRAL_IMAGE.TIF"
    wl = np.linspace(418.4, 2445.3, 50)
    with rasterio.open(p, "w", driver="GTiff", height=64, width=72, count=50, dtype="int16", nodata=-32768,
                       crs="EPSG:32644", transform=from_origin(500000, 1800000, 30, 30)) as d:
        d.write(np.moveaxis(X, 2, 0).astype(np.int16))
        d.update_tags(wavelength_units="'nanometers'", data_ignore_value="'-32768'")
        for i, w in enumerate(wl, 1):
            d.update_tags(i, wavelength=f"{w:.3f}")
    c = open_cube(p, qc=False)
    assert c.bands == 50 and c.unit == "reflectance" and c.crs == "EPSG:32644"
    assert c.wavelengths_nm[0] == pytest.approx(418.4, abs=0.01)
    np.testing.assert_allclose(c.read(rows=(3, 4), cols=(7, 8))[0, 0], X[3, 7] / 10000, rtol=1e-5)
