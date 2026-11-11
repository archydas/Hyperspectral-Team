"""EnMAP-style product (wavelengths only in a sibling XML, int16 reflectance x10000, nodata bands),
wavelength files, material categories, ROI from registered cubes, CSV export."""
import numpy as np, pytest, rasterio
from rasterio.transform import from_origin
from ds3.ingest import open_cube, MissingWavelengthsError
from ds3.ingest.readers.wavelength_files import read_wavelengths_file, parse_wavelength_xml
from ds3.spectra.categories import guess_category
from ds3.pipeline import ingest_file
from ds3.db import session_scope, init_db
from ds3.library import cube_roi
from ds3.library.access import list_cubes, get_spectra, open_registered_cube
from ds3.library.search import find_spectra
from ds3.library.export import export_csv
from ds3.library.builder import build_library
from ds3.library.grid import Grid

NAME = "ENMAP01-____L2A-DT0000120914_20250324T051236Z_003_V010506_20260724T115758Z-SPECTRAL_IMAGE.TIF"
B, R, C = 30, 12, 10
WL = np.linspace(420, 2450, B)


def make_enmap_like(folder, xml=True):
    d = np.clip(0.05 + 0.4 * (WL - 420) / 2030, 0, 1)[None, None, :] * np.ones((R, C, B))
    arr = np.round(d * 10000).astype("int16")
    arr[:, :, [3, 4, 20]] = -32768                      # bands filled with nodata
    arr[0, :, :] = -32768                               # a nodata row
    p = folder / NAME
    with rasterio.open(p, "w", driver="GTiff", height=R, width=C, count=B, dtype="int16", nodata=-32768,
                       crs="EPSG:32633", transform=from_origin(390000, 5820000, 30, 30)) as ds:
        ds.write(np.moveaxis(arr, -1, 0))
    if xml:
        bands = "".join(f'<bandID number="{i+1}"><wavelengthCenterOfBand>{w:.2f}</wavelengthCenterOfBand>'
                        f'<FWHMOfBand>{8.0:.1f}</FWHMOfBand></bandID>' for i, w in enumerate(WL))
        (folder / NAME.replace("SPECTRAL_IMAGE.TIF", "METADATA.XML")).write_text(
            f'<?xml version="1.0"?><level_X><specific><bandCharacterisation>{bands}</bandCharacterisation></specific></level_X>')
    return p


def test_xml_wavelengths_and_enmap_defaults(tmp_path):
    p = make_enmap_like(tmp_path)
    c = open_cube(p)
    assert c.bands == B and np.allclose(c.wavelengths_nm, WL, atol=0.01)
    assert c.unit == "reflectance" and c.sensor == "EnMAP"
    assert np.isclose(np.nanmax(c.read(rows=(3, 4), cols=(0, 1))), 0.05 + 0.4 * 0.99, atol=0.06)
    assert c.bad_band_mask[[3, 4, 20]].all()             # nodata-filled bands are flagged, not deleted
    assert c.fwhm_nm is not None and np.allclose(c.fwhm_nm, 8.0)


def test_missing_xml_gives_actionable_error(tmp_path):
    p = make_enmap_like(tmp_path, xml=False)
    with pytest.raises(MissingWavelengthsError) as e:
        open_cube(p)
    assert "METADATA.XML" in str(e.value) and "wavelengths=" in str(e.value)


def test_manual_wavelength_file_and_unknown_xml(tmp_path):
    p = make_enmap_like(tmp_path, xml=False)
    f = tmp_path / "wl.txt"; f.write_text("# nm\n" + "\n".join(f"{w:.3f}" for w in WL))
    wl, fw, notes = read_wavelengths_file(f)
    assert fw is None and len(wl) == B
    c = open_cube(p, wavelengths=wl.tolist())
    assert c.bands == B
    bad = tmp_path / "x.xml"; bad.write_text("<a><b>1</b></a>")
    with pytest.raises(MissingWavelengthsError):
        parse_wavelength_xml(bad)


def test_micrometre_wavelength_file(tmp_path):
    f = tmp_path / "um.csv"; f.write_text("\n".join(f"{w/1000:.5f},{0.01}" for w in WL))
    wl, fw, _ = read_wavelengths_file(f)
    assert np.allclose(wl, WL, atol=0.01) and np.allclose(fw, 10.0)


@pytest.mark.parametrize("name,cat", [("Construction  Concrete", "construction"), ("Galvanized Steel Metal_2", "metal"),
                                      ("Olive green gloss paint", "paint_coating"), ("Pine Wood", "construction"),
                                      ("Black unspecified rubber", "polymer_fabric"), ("Dry grass", "vegetation"),
                                      ("Kaolinite KGa-1", "mineral_rock"), ("Lake water", "water_snow"),
                                      ("Mystery sample", None)])
def test_category_rule(name, cat):
    assert guess_category(name) == cat


def test_cube_roi_save_and_search(tmp_path):
    db = f"sqlite:///{tmp_path}/t.db"
    p = make_enmap_like(tmp_path)
    r = ingest_file(p, kind="satellite", db_url=db)
    assert r.status == "ingested", r.detail
    with session_scope(db) as s:
        cubes = list_cubes(s)
        cube, rec = open_registered_cube(s, cubes[0]["id"])
    assert cube.sensor == "EnMAP" and cube.bad_band_mask[3]
    spectra, _ = cube_roi.measure(cubes[0]["id"], bbox_px=(2, 2, 6, 6), name="field A", category="soil", db_url=db)
    assert spectra[0].purity == "image_pixel" and spectra[0].metadata["n_pixels"] == 16
    ids = cube_roi.save(spectra, cubes[0]["id"], library="ROI spectra", db_url=db)
    with pytest.raises(Exception, match="already saved"):
        cube_roi.save(spectra, cubes[0]["id"], library="ROI spectra", db_url=db)
    with session_scope(db) as s:
        hit = find_spectra(s, text="field A", purity="image_pixel")
        assert hit["total"] == 1 and hit["results"][0]["category"] == "soil"
        v = get_spectra(s, ids)[0]["values"]
    assert np.isfinite(v[0]) and np.isnan(v[3])          # nodata band stays NaN in the library entry (flagged, not invented)
    with pytest.raises(Exception, match="no valid pixels"):
        cube_roi.measure(cubes[0]["id"], pixel=(0, 0), db_url=db)    # row 0 is all nodata


def test_export_csv_roundtrip(tmp_path):
    db = f"sqlite:///{tmp_path}/t.db"
    g = Grid(np.arange(450, 2401, 50.0), origin="explicit")
    f = tmp_path / "a.csv"
    f.write_text("wavelength_nm,Grass,Sand\n" + "\n".join(f"{w},{0.1+w/5000:.4f},{0.2+w/9000:.4f}" for w in np.arange(400, 2501, 10)))
    build_library("lib1", [str(f)], g, db_url=db)
    out = export_csv("lib1", tmp_path / "out", db_url=db)
    head = open(out).readline().strip().split(",")
    assert head == ["Wavelength", "Grass", "Sand"]
    r = ingest_file(out, kind="library", db_url=db)
    assert r.status == "ingested" and r.detail.startswith("2 spectra")
