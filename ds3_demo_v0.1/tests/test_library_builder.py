import numpy as np, pytest
from fastapi.testclient import TestClient
from ds3.sampledata import generate, _veg
from ds3.library.grid import Grid, resample
from ds3.library.sources import read_envi_roi_ascii, is_envi_roi, spectra_from_roi_ascii, read_evf
from ds3.library.builder import build_library
from ds3.library.search import find_spectra, match_spectrum
from ds3.db import session_scope
from ds3.ingest import open_cube, IngestError, UnsupportedFormatError


@pytest.fixture(scope="module")
def sample(tmp_path_factory):
    d = tmp_path_factory.mktemp("lb"); generate(d); return d


@pytest.fixture
def dburl(tmp_path, monkeypatch):
    url = f"sqlite:///{tmp_path/'t.db'}"; monkeypatch.setenv("DATABASE_URL", url); return url


def test_resample_linear_accuracy_and_no_extrapolation():
    src = np.arange(400, 2501, 5.0); g = Grid(np.arange(350, 2601, 10.0))
    v, cov = resample(src, _veg(src), g)
    ins = (g.wavelengths_nm >= 400) & (g.wavelengths_nm <= 2500)
    assert np.nanmax(np.abs(v[ins] - _veg(g.wavelengths_nm[ins]))) < 1e-4
    assert np.isnan(v[~ins]).all() and 0 < cov < 1


def test_resample_does_not_bridge_wide_gaps():
    src = np.r_[400:1300:10, 1500:2000:10].astype(float); v = np.ones_like(src)
    out, _ = resample(src, v, Grid(np.array([1000., 1400., 1800.])))
    assert np.isfinite(out[0]) and np.isnan(out[1]) and np.isfinite(out[2])


def test_resample_gaussian_needs_full_window():
    src = np.arange(400, 1000, 2.0); g = Grid(np.array([402., 700.]), fwhm_nm=np.array([20., 20.]))
    out, _ = resample(src, np.full_like(src, 0.3), g, "gaussian")
    assert np.isnan(out[0]) and out[1] == pytest.approx(0.3)


def test_grid_ascii_and_cube(sample):
    g = Grid.from_ascii(sample / "builder/grid_wavelengths.txt"); assert g.wavelengths_nm[0] == 450 and len(g) == 196
    c = Grid.from_cube(sample / "airborne/synthetic_scene_bsq.img.hdr"); assert len(c) == 120 and c.fwhm_nm is not None


def test_roi_file_parsing_and_measurement(sample):
    f = sample / "builder/rois.txt"; assert is_envi_roi(f)
    rois = read_envi_roi_ascii(f); assert [r[0] for r in rois] == ["roi_vegetation", "roi_water"] and rois[0][1] == 100
    cube = open_cube(sample / "airborne/synthetic_scene_bsq.img.hdr")
    veg, wat = spectra_from_roi_ascii(f, cube)
    assert veg.purity == "image_pixel" and veg.values[60] > wat.values[60]          # vegetation brighter in NIR
    # 1-based X/Y: pixel listed as (x=2,y=2) is row 1, col 1 -> inside the vegetation half of the scene
    assert veg.metadata["roi_pixels_used"] == 100


def test_roi_outside_cube_gives_clear_error(sample, tmp_path):
    f = tmp_path / "bad.txt"
    f.write_text("; ENVI Output of ROIs (ASCII)\n; ROI name: far\n; ROI npts: 1\n  1 9999 9999\n")
    with pytest.raises(Exception, match="none of its"): spectra_from_roi_ascii(f, open_cube(sample / "airborne/synthetic_scene_bsq.img.hdr"))


def test_evf_clear_error(tmp_path):
    with pytest.raises(UnsupportedFormatError, match="evf"): read_evf(tmp_path / "a.evf")


def _build(sample, **kw):
    S = sample
    return build_library("lib", [S / "builder/rois.txt", S / "builder/fine_grid.csv", S / "field/leaf_001.sig",
                                 S / "library/oak_leaf.spectrum.txt"],
                         Grid.from_ascii(S / "builder/grid_wavelengths.txt"),
                         cube=S / "airborne/synthetic_scene_bsq.img.hdr", **kw)


def test_builder_one_grid_and_report(sample, dburl):
    r = _build(sample)
    assert len(r.added) == 5 and not r.rejected
    with session_scope() as s:
        from ds3.db.models import SpectrumRecord
        rows = s.query(SpectrumRecord).all()
        assert {tuple(x.wavelengths_nm) for x in rows} == {tuple(np.arange(450, 2401, 10.0))}


def test_builder_rebuild_refused_and_unit_mismatch_rejected(sample, dburl, tmp_path):
    _build(sample)
    with pytest.raises(IngestError, match="already built"): _build(sample)
    rad = tmp_path / "rad.csv"; rad.write_text("wavelength_nm,x\n500,1\n600,2\n700,3\n800,2\n900,1\n")
    r = build_library("mix", [sample / "builder/fine_grid.csv", rad], "first", unit="reflectance")
    assert len(r.added) == 1
    r2 = build_library("mix2", [sample / "builder/fine_grid.csv", sample / "field/leaf_001.sig"], "first", min_coverage=0.99)
    assert any("covered" in x["reason"] for x in r2.rejected)


def test_builder_rois_need_cube(sample, dburl):
    with pytest.raises(IngestError, match="cube"): build_library("x", [sample / "builder/rois.txt"], "first")


def test_search_and_match(sample, dburl):
    _build(sample)
    with session_scope() as s:
        r = find_spectra(s, text="vegetation"); assert r["total"] >= 2
        assert find_spectra(s, text="nothing-like-this")["total"] == 0
        assert find_spectra(s, covers_nm=(450, 2400), unit="reflectance")["total"] == 5
        assert find_spectra(s, covers_nm=(300, 2400))["total"] == 0
        assert find_spectra(s, purity="image_pixel")["total"] == 2
        assert len(find_spectra(s, limit=2)["results"]) == 2 and find_spectra(s, offset=4)["results"]
        veg = find_spectra(s, text="roi_vegetation")["results"][0]
        from ds3.db.models import SpectrumRecord
        rec = s.get(SpectrumRecord, veg["id"])
        q = np.array([np.nan if x is None else x for x in rec.values], float)
        top = match_spectrum(s, rec.wavelengths_nm, q, top_k=5, unit="reflectance", exclude_id=rec.id)
        names = [t["material"] for t in top]
        assert "roi_vegetation" not in names and top[0]["material"] in ("vegetation_fine", "leaf_001", "Oak leaf")
        assert top[0]["score"] < top[-1]["score"]


def test_ingest_dir_skips_roi_files(sample, dburl):
    from ds3.pipeline import ingest_path
    res = ingest_path(sample, source_name="t")
    assert not [r for r in res if r.path.endswith("rois.txt") or r.status == "error"]


def test_api_build_search_match(sample, dburl):
    from ds3.api.app import app
    S = sample
    with TestClient(app) as c:
        r = c.post("/libraries/build", json=dict(name="api-lib", grid=f"ascii:{S/'builder/grid_wavelengths.txt'}",
                   cube=str(S / "airborne/synthetic_scene_bsq.img.hdr"),
                   inputs=[str(S / "builder/rois.txt"), str(S / "builder/fine_grid.csv")]))
        assert r.status_code == 200 and len(r.json()["added"]) == 3
        assert c.post("/libraries/build", json=dict(name="b", inputs=[str(S / "builder/rois.txt")])).status_code == 422
        s = c.get("/spectra/search?text=roi&library=api-lib").json(); assert s["total"] == 2
        sid = s["results"][0]["id"]
        m = c.post("/spectra/match", json=dict(spectrum_id=sid, top_k=2, library="api-lib")).json()
        assert len(m) == 2 and m[0]["score"] <= m[1]["score"]
        assert c.post("/spectra/match", json=dict(wavelengths_nm=[1, 2], values=[1, 2])).status_code == 422
        assert c.get(f"/spectra/{sid}").status_code == 200       # route order: /search before /{id}


def test_cli_library_flow(sample, dburl, capsys):
    from ds3.cli import main
    S = sample
    main(["library", "build", "cli-lib", "--grid", f"ascii:{S/'builder/grid_wavelengths.txt'}", "--cube",
          str(S / "airborne/synthetic_scene_bsq.img.hdr"), "--inputs", str(S / "builder/rois.txt"), str(S / "builder/fine_grid.csv")])
    main(["library", "search", "--text", "roi"]); out = capsys.readouterr().out
    assert "added 3" in out and "2 match(es)" in out


def test_export_envi_sli_roundtrip(sample, dburl, tmp_path):
    from ds3.library.export import export_envi_sli
    from ds3.spectra import read_spectra
    _build(sample)
    out = export_envi_sli("lib", tmp_path / "out.sli")
    back = read_spectra(out); assert len(back) == 5 and len(back[0].values) == 196
    assert back[0].wavelengths_nm[0] == 450
    assert min(np.nanmin(b.values) for b in back) > -1          # -9999 fill must come back as NaN, not as a value
    assert any(np.isnan(b.values).any() for b in back)          # ROI spectra have flagged water bands
    with session_scope() as s:
        orig = find_spectra(s, library="lib")["results"][0]["material"]
    assert back[0].material == orig
    with pytest.raises(IngestError, match="no spectra"): export_envi_sli("nope", tmp_path / "x.sli")
