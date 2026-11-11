import numpy as np, pytest, rasterio, zarr, h5py
from rasterio.transform import from_origin

R, C, B = 12, 10, 40
WL = np.linspace(400, 2400, B)


def truth():
    """Known reflectance cube [R, C, B]."""
    r = np.arange(R)[:, None, None]; c = np.arange(C)[None, :, None]; b = np.arange(B)[None, None, :]
    return (0.1 + 0.5 * b / B + 0.002 * r + 0.0005 * c).astype(np.float32)


def write_envi(path, interleave="bsq", dtype="i2", scale=10000, wl=WL, units="Nanometers", bbl=True,
               nodata=None, data=None, extra=""):
    d = truth() if data is None else data
    arr = np.round(d * scale).astype(dtype) if dtype != "f4" else d.astype("f4")
    if nodata is not None:
        arr = np.where(np.isnan(d), nodata, arr).astype(arr.dtype)
    lay = {"bsq": (2, 0, 1), "bil": (0, 2, 1), "bip": (0, 1, 2)}[interleave]
    arr.transpose(lay).tofile(str(path))
    code = {"i2": 2, "f4": 4, "u2": 12}[dtype]
    h = ["ENVI", f"samples = {C}", f"lines = {R}", f"bands = {len(wl)}", "header offset = 0",
         "file type = ENVI Standard", f"data type = {code}", f"interleave = {interleave}", "byte order = 0",
         "map info = {UTM, 1, 1, 500000.0, 1000000.0, 30.0, 30.0, 43, North, WGS-84}",
         f"wavelength units = {units}"]
    if scale and dtype != "f4":
        h.append(f"reflectance scale factor = {scale}")
    if nodata is not None:
        h.append(f"data ignore value = {nodata}")
    h.append("wavelength = {\n " + ",\n ".join(f"{w:.6f}" for w in wl) + "}")
    h.append("fwhm = {" + ", ".join("10.0" for _ in wl) + "}")
    if bbl:
        flags = [0 if i in (5, 6) else 1 for i in range(len(wl))]
        h.append("bbl = {" + ", ".join(map(str, flags)) + "}")
    h.append(extra)
    open(str(path) + ".hdr", "w").write("\n".join(h))
    return str(path) + ".hdr"


def write_tif(path, desc=True, tags_wl=False, env_tags=False, d=None, nodata=None, dtype="float32"):
    d = truth() if d is None else d
    tr = from_origin(500000, 1000000, 30, 30)
    with rasterio.open(path, "w", driver="GTiff", height=R, width=C, count=B, dtype=dtype,
                       crs="EPSG:32643", transform=tr, nodata=nodata) as ds:
        ds.write(np.moveaxis(d, -1, 0).astype(dtype))
        if desc:
            for i, w in enumerate(WL):
                ds.set_band_description(i + 1, f"{w:.2f} nm")
        if tags_wl:
            ds.update_tags(wavelength="{" + ",".join(f"{w/1000:.5f}" for w in WL) + "}", wavelength_units="micrometers")
        if env_tags:
            ds.update_tags(ns="ENVI", wavelength="{" + ",".join(f"{w:.3f}" for w in WL) + "}", wavelength_units="nm")
    return str(path)


def write_zarr_store(path):
    g = zarr.open_group(str(path), mode="w")
    a = g.create_array("data", shape=(R, C, B), chunks=(4, 4, B), dtype="float32")
    a[:] = truth()
    g.attrs.update(dict(axes="yxb", wavelengths_nm=WL.tolist(), unit="reflectance"))
    return str(path)


def write_emit_like(path):
    import netCDF4
    ds = netCDF4.Dataset(str(path), "w", format="NETCDF4")
    ds.createDimension("downtrack", R); ds.createDimension("crosstrack", C); ds.createDimension("bands", B)
    v = ds.createVariable("reflectance", "f4", ("downtrack", "crosstrack", "bands"), fill_value=-9999.0)
    d = truth().copy(); d[0, 0, :] = np.nan
    v[:] = np.ma.masked_invalid(d)
    g = ds.createGroup("sensor_band_parameters")
    g.createDimension("bands", B)
    w = g.createVariable("wavelengths", "f4", ("bands",)); w[:] = WL; w.units = "nm"
    f = g.createVariable("fwhm", "f4", ("bands",)); f[:] = 8.0
    gw = g.createVariable("good_wavelengths", "i1", ("bands",)); gw[:] = [0 if i in (5, 6) else 1 for i in range(B)]
    ds.close()
    return str(path)


@pytest.fixture
def tmp(tmp_path):
    return tmp_path
