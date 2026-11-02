import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hvva.core.loaders import UnsupportedFormat, open_source, read_raw
from hvva.core.scene import load_scene

RNG = np.random.default_rng(1)
CUBE = RNG.random((30, 40, 12)).astype("float32")           # rows, cols, bands
GT = RNG.integers(0, 4, (30, 40)).astype("uint8")
WL = np.linspace(0.45, 0.95, 12)                             # micrometres on purpose


def check(sc, labels=True, wl=True):
    assert sc.cube.shape == (30, 40, 12)
    assert np.allclose(sc.cube, CUBE, atol=1e-6)
    if labels:
        assert sc.labels is not None and np.array_equal(sc.labels, GT)
    if wl:
        assert np.allclose(sc.wavelengths, WL * 1000)         # converted to nm


def test_mat_v5(tmp_path):
    import scipy.io as sio
    p = str(tmp_path / "d.mat")
    sio.savemat(p, {"paviaU": CUBE, "paviaU_gt": GT, "wavelengths": WL})
    check(load_scene(p))


def test_mat_v73_hdf5(tmp_path):
    h5py = pytest.importorskip("h5py")
    p = str(tmp_path / "d73.mat")
    with h5py.File(p, "w") as f:                              # MATLAB v7.3 stores axes reversed
        f["cube"] = CUBE.T
        f["gt"] = GT.T
        f["wavelength"] = WL
    check(load_scene(p))


def test_hdf5_nested_groups(tmp_path):
    h5py = pytest.importorskip("h5py")
    p = str(tmp_path / "d.h5")
    with h5py.File(p, "w") as f:
        f["scan/reflectance"] = CUBE
        f["scan/classes"] = GT
        f["meta/wavelengths"] = WL
    check(load_scene(p))


def test_netcdf(tmp_path):
    nc = pytest.importorskip("netCDF4")
    p = str(tmp_path / "d.nc")
    ds = nc.Dataset(p, "w")
    for n, v in (("y", 30), ("x", 40), ("band", 12)):
        ds.createDimension(n, v)
    ds.createVariable("reflectance", "f4", ("y", "x", "band"))[:] = CUBE
    ds.createVariable("wavelength", "f8", ("band",))[:] = WL
    ds.close()
    check(load_scene(p), labels=False)


def test_zarr(tmp_path):
    zarr = pytest.importorskip("zarr")
    p = str(tmp_path / "d.zarr")
    g = zarr.open_group(p, mode="w")
    g.create_array("cube", data=CUBE) if hasattr(g, "create_array") else g.create_dataset("cube", data=CUBE)
    check(load_scene(p), labels=False, wl=False)


def test_tiff_bands_first(tmp_path):
    import tifffile
    p = str(tmp_path / "d.tif")
    tifffile.imwrite(p, CUBE.transpose(2, 0, 1))              # (bands, rows, cols) as most GeoTIFFs
    check(load_scene(p), labels=False, wl=False)


def test_npy_band_axis_override(tmp_path):
    p = str(tmp_path / "d.npy")
    np.save(p, CUBE.transpose(2, 0, 1))                       # bands first
    check(load_scene(p), labels=False, wl=False)
    sq = np.random.rand(10, 10, 10).astype("float32")         # ambiguous shape: user must choose the axis
    np.save(p, sq)
    sc = load_scene(p, band_axis=0)
    assert np.allclose(sc.cube, np.moveaxis(sq, 0, 2))


def test_npz_arbitrary_keys(tmp_path):
    p = str(tmp_path / "d.npz")
    np.savez(p, data=CUBE, truth=GT, wl=WL)
    check(load_scene(p))


@pytest.mark.parametrize("il", ["bsq", "bil", "bip"])
def test_raw_binary(tmp_path, il):
    p = str(tmp_path / "d.raw")
    arr = (CUBE * 1000).astype("<u2")
    {"bsq": arr.transpose(2, 0, 1), "bil": arr.transpose(0, 2, 1), "bip": arr}[il].tofile(p)
    sc = load_scene(p, raw=dict(samples=40, lines=30, bands=12, dtype="uint16", interleave=il),
                    wavelength_range=(450, 950))
    assert np.allclose(sc.cube, arr) and np.isclose(sc.wavelengths[-1], 950)


def test_raw_size_mismatch_message(tmp_path):
    p = str(tmp_path / "d.raw")
    np.zeros(100, "<u2").tofile(p)
    with pytest.raises(ValueError, match="expected"):
        load_scene(p, raw=dict(samples=40, lines=30, bands=12))


def test_wavelength_range_for_files_without_wl(tmp_path):
    p = str(tmp_path / "d.npy")
    np.save(p, CUBE)
    sc = load_scene(p, wavelength_range=(400, 1000))
    assert sc.wavelengths[0] == 400 and sc.wavelengths[-1] == 1000


def test_unknown_file_gives_helpful_error(tmp_path):
    p = str(tmp_path / "mystery.xyz")
    open(p, "wb").write(b"not a known format")
    with pytest.raises(UnsupportedFormat, match="raw"):
        load_scene(p)


def test_no_cube_inside(tmp_path):
    import scipy.io as sio
    p = str(tmp_path / "flat.mat")
    sio.savemat(p, {"a": np.zeros((5, 5))})
    with pytest.raises(UnsupportedFormat, match="3-D"):
        load_scene(p)


def test_source_lists_arrays_for_gui(tmp_path):
    import scipy.io as sio
    p = str(tmp_path / "d.mat")
    sio.savemat(p, {"cubeA": CUBE, "cubeB": CUBE[:, :, :5], "gt": GT})
    src = open_source(p)
    assert set(src.cube_candidates()) == {"cubeA", "cubeB"}
    g = src.guess()
    assert g["cube_key"] == "cubeA" and g["labels_key"] == "gt" and g["band_axis"] == 2
    assert src.build(cube_key="cubeB", band_axis=2).cube.shape[2] == 5
