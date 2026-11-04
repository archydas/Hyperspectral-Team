"""Generate small SYNTHETIC files in every supported format (for demos, tests, CI).
Not real data - see docs/DATA_SOURCES.md for real downloads."""
from __future__ import annotations
from pathlib import Path
import numpy as np

R, C, B = 40, 30, 120
WL = np.linspace(400, 2500, B)


def _veg(wl):
    w = wl
    base = 0.05 + 0.45 / (1 + np.exp(-(w - 720) / 15))          # red edge
    base = base - 0.35 * np.exp(-((w - 1450) / 60) ** 2) - 0.3 * np.exp(-((w - 1930) / 70) ** 2)
    return np.clip(base - 0.1 * (w > 1300) * (w - 1300) / 1200, 0.01, 0.9)

def _soil(wl): return np.clip(0.08 + 0.35 * (wl - 400) / 2100, 0, 0.6)
def _water(wl): return np.clip(0.06 * np.exp(-(wl - 400) / 250), 0.001, 0.1)
MATERIALS = {"vegetation": _veg, "bare_soil": _soil, "water": _water}


def make_cube(seed=0):
    rng = np.random.default_rng(seed)
    cube = np.zeros((R, C, B), np.float32)
    cube[:] = _soil(WL)
    cube[:, : C // 2] = _veg(WL)
    cube[R // 2:, C // 2:] = _water(WL)
    return np.clip(cube + rng.normal(0, 0.003, cube.shape), 0, 1).astype(np.float32)


def _envi(path, d, interleave, dtype="i2", scale=10000):
    arr = np.round(d * scale).astype(dtype)
    lay = {"bsq": (2, 0, 1), "bil": (0, 2, 1), "bip": (0, 1, 2)}[interleave]
    arr.transpose(lay).tofile(str(path))
    h = ["ENVI", f"samples = {C}", f"lines = {R}", f"bands = {B}", "header offset = 0", "file type = ENVI Standard",
         "data type = 2", f"interleave = {interleave}", "byte order = 0",
         "map info = {UTM, 1, 1, 500000.0, 1000000.0, 5.0, 5.0, 43, North, WGS-84}",
         "wavelength units = Nanometers", f"reflectance scale factor = {scale}", "sensor type = synthetic-VNIR-SWIR",
         "wavelength = {" + ", ".join(f"{w:.3f}" for w in WL) + "}",
         "fwhm = {" + ", ".join("10.0" for _ in WL) + "}"]
    Path(str(path) + ".hdr").write_text("\n".join(h))


def generate(out_dir="data/sample") -> list[str]:
    import rasterio, zarr, netCDF4
    from rasterio.transform import from_origin
    out = Path(out_dir); made = []
    d = make_cube()
    for sub in ("airborne", "satellite", "drone", "field", "library"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    # --- cubes
    _envi(out / "airborne/synthetic_scene_bsq.img", d, "bsq"); made.append("airborne/synthetic_scene_bsq.img")
    _envi(out / "airborne/synthetic_scene_bil.img", d, "bil"); made.append("airborne/synthetic_scene_bil.img")
    _envi(out / "drone/synthetic_scene_bip.img", d, "bip"); made.append("drone/synthetic_scene_bip.img")
    p = out / "satellite/synthetic_scene.tif"
    with rasterio.open(p, "w", driver="GTiff", height=R, width=C, count=B, dtype="float32", crs="EPSG:32643",
                       transform=from_origin(500000, 1000000, 30, 30)) as ds:
        ds.write(np.moveaxis(d, -1, 0))
        for i, w in enumerate(WL): ds.set_band_description(i + 1, f"{w:.2f} nm")
    made.append("satellite/synthetic_scene.tif")
    g = zarr.open_group(str(out / "drone/synthetic_scene.zarr"), mode="w")
    a = g.create_array("data", shape=d.shape, chunks=(10, 10, B), dtype="float32"); a[:] = d
    g.attrs.update(dict(axes="yxb", wavelengths_nm=WL.tolist(), unit="reflectance")); made.append("drone/synthetic_scene.zarr")
    p = out / "satellite/synthetic_emit_like.nc"
    nc = netCDF4.Dataset(str(p), "w", format="NETCDF4")
    nc.createDimension("downtrack", R); nc.createDimension("crosstrack", C); nc.createDimension("bands", B)
    v = nc.createVariable("reflectance", "f4", ("downtrack", "crosstrack", "bands"), fill_value=-9999.0); v[:] = d
    gr = nc.createGroup("sensor_band_parameters"); gr.createDimension("bands", B)
    w = gr.createVariable("wavelengths", "f4", ("bands",)); w[:] = WL; w.units = "nm"
    f = gr.createVariable("fwhm", "f4", ("bands",)); f[:] = 8.0
    nc.close(); made.append("satellite/synthetic_emit_like.nc")
    # --- field / library spectra (several different layouts)
    fd = out / "field"
    names = list(MATERIALS)
    (fd / "wide.csv").write_text("wavelength_nm," + ",".join(names) + "\n" + "\n".join(
        f"{w:.1f}," + ",".join(f"{MATERIALS[n](np.array([w]))[0]:.5f}" for n in names) for w in WL))
    (fd / "rowwise.csv").write_text("sample," + ",".join(f"{w:.1f}" for w in WL) + "\n" + "\n".join(
        f"{n}_row," + ",".join(f"{x:.5f}" for x in MATERIALS[n](WL)) for n in names))
    veg = _veg(WL) * 100
    (fd / "leaf_001.sig").write_text("\n".join(
        ["/*** Spectra Vista HR-1024i ***/", "name= leaf_001", "instrument= HR-1024i", "time= 10/10/2026, 11:02:00 AM",
         "data="] + [f"{w:.2f} {r:.2f} {r*0.9:.2f} {r:.3f}" for w, r in zip(WL, veg)]))
    (out / "library/oak_leaf.spectrum.txt").write_text("\n".join(
        ["Name: Oak leaf", "Type: vegetation", "Class: tree", "X Units: Wavelength (micrometers)",
         "Y Units: Reflectance (percent)", ""] + [f"{w/1000:.4f} {r:.4f}" for w, r in zip(WL, veg)]))
    (out / "library/splib07a_Wavelengths_synthetic.txt").write_text("splib07a Wavelengths (micrometers)\n" +
        "\n".join(f"{w/1000:.6f}" for w in WL))
    sp = _soil(WL).copy(); sp[10:12] = -1.23e34
    (out / "library/splib07a_Dry_soil_synthetic.txt").write_text("Dry_Soil_Synthetic splib07a Record=1: Soil\n" +
        "\n".join(f"{x:.6f}" for x in sp))
    arr = np.stack([f(WL) for f in MATERIALS.values()]).astype("f4"); arr.tofile(str(out / "library/synthetic.sli"))
    (out / "library/synthetic.hdr").write_text("\n".join(
        ["ENVI", f"samples = {B}", f"lines = {len(names)}", "bands = 1", "header offset = 0",
         "file type = ENVI Spectral Library", "data type = 4", "interleave = bsq", "byte order = 0",
         "wavelength units = Nanometers", "spectra names = {" + ", ".join(names) + "}",
         "wavelength = {" + ", ".join(f"{w:.3f}" for w in WL) + "}"]))
    # --- builder demo inputs (ENVI Spectral Library Builder workflow)
    bd = out / "builder"; bd.mkdir(exist_ok=True)
    g = np.arange(450, 2401, 10.0)
    (bd / "grid_wavelengths.txt").write_text("; target wavelength grid (nm), one per line\n" + "\n".join(f"{x:.1f}" for x in g))
    def block(name, r0, r1, c0, c1):
        pts = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]
        return [f"; ROI name: {name}", "; ROI rgb value: {0, 255, 0}", f"; ROI npts: {len(pts)}", ";   ID     X     Y"] + \
               [f"   {i+1:4d} {c:5d} {r:5d}" for i, (r, c) in enumerate(pts)]
    (bd / "rois.txt").write_text("\n".join(["; ENVI Output of ROIs (ASCII)", f"; File Dimension: {C} x {R}", "; Number of ROIs: 2", ";"] +
        block("roi_vegetation", 2, 11, 2, 11) + [";"] + block("roi_water", 30, 38, 18, 27)))
    fw = np.arange(350, 2501, 5.0)
    (bd / "fine_grid.csv").write_text("wavelength_nm,vegetation_fine\n" + "\n".join(f"{w:.1f},{_veg(np.array([w]))[0]:.5f}" for w in fw))
    return sorted(str(x.relative_to(out)) for x in out.rglob("*")
                  if (x.is_file() and ".zarr" not in str(x.relative_to(out))) or x.suffix == ".zarr")
