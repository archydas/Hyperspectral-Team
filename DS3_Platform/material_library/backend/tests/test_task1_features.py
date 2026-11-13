"""Category paths, acquisition tags and multi-scene aggregation."""
import io, zipfile

import numpy as np
from test_material_library import _envi_with_map, make_csv


def _upload_scene(client, tmp, nb):
    tmp.mkdir(parents=True, exist_ok=True)
    d, h = _envi_with_map(tmp, nb=nb)
    r = client.post("/api/import", files=[("files", (f"scene{nb}", d.read_bytes())), ("files", (f"scene{nb}.hdr", h.read_bytes()))])
    assert r.status_code == 200 and r.json()["images"] == 1, r.text
    return [i for i in client.get("/api/images/").json() if i["bands"] == nb][0]["id"]


def test_category_paths_are_normalised_and_tree_rolls_up(client):
    items = [{"name": n, "wavelengths_nm": [400, 500, 600, 700], "values": [0.1, 0.2, 0.3, 0.4], "category": c}
             for n, c in (("Hemlock", "Vegetation > Tree | Tsuga"), ("Oak", "Vegetation/Tree/Quercus"), ("Sand", "Soil"), ("Mystery", None))]
    assert client.post("/api/spectra/save", json={"library": "Tax", "items": items}).json()["saved"] == 4
    f = client.get("/api/facets").json()
    cats = {c["path"]: c["count"] for c in f["categories"]}
    assert cats["Vegetation"] == 2 and cats["Vegetation/Tree"] == 2 and cats["Vegetation/Tree/Tsuga"] == 1
    assert cats["Soil"] == 1 and cats["__none__"] == 1
    hits = client.get("/api/spectra/search", params={"category_path": "Vegetation/Tree"}).json()
    assert {h["material"] for h in hits["results"]} == {"Hemlock", "Oak"}
    assert client.get("/api/spectra/search", params={"category_path": "Vegetation/Tr"}).json()["total"] == 0     # not a prefix match
    r = client.post("/api/categories/rename", json={"old": "Vegetation/Tree", "new": "Plants/Woody"}).json()
    assert r["materials"] == 2
    cats = {c["path"] for c in client.get("/api/facets").json()["categories"]}
    assert "Plants/Woody/Tsuga" in cats and "Vegetation/Tree" not in cats


def test_acquisition_tags_filter_and_bulk_edit(client):
    items = [{"name": n, "wavelengths_nm": [400, 500, 600, 700], "values": [0.1, 0.2, 0.3, 0.4], "acquisition_type": a}
             for n, a in (("A", "lab"), ("B", "field"), ("C", None))]
    assert client.post("/api/spectra/save", json={"library": "Acq", "items": items}).json()["saved"] == 3
    assert client.post("/api/spectra/save", json={"library": "Acq", "items": [dict(items[0], name="Bad", acquisition_type="drone")]}).json()["rejected"]
    lab = client.get("/api/spectra/search", params={"acquisition": "lab"}).json()["results"]
    assert [r["material"] for r in lab] == ["A"] and lab[0]["acquisition_type"] == "lab"
    assert [r["material"] for r in client.get("/api/spectra/search", params={"acquisition": "__none__"}).json()["results"]] == ["C"]
    lib = client.get("/api/libraries/").json()[0]
    mats = client.get(f"/api/libraries/{lib['id']}/materials", params={"acquisition": "field"}).json()
    assert [m["name"] for m in mats] == ["B"] and mats[0]["acquisition"] == ["field"]
    everyone = [m["id"] for m in client.get(f"/api/libraries/{lib['id']}/materials").json()]
    assert client.patch("/api/materials", json={"material_ids": everyone, "set_category": False, "set_acquisition": True,
                                                 "acquisition_type": "airborne"}).json()["spectra_tagged"] == 3
    assert client.get("/api/facets").json()["acquisition"] == {"airborne": 3}
    assert client.patch("/api/materials", json={"material_ids": everyone, "set_category": False, "set_acquisition": True,
                                                 "acquisition_type": "boat"}).status_code == 400
    sid = client.get("/api/spectra/search").json()["results"][0]["spectrum_id"]
    assert client.patch(f"/api/spectra/{sid}", json={"acquisition_type": "satellite"}).json()["acquisition_type"] == "satellite"


def test_import_infers_acquisition_from_file_name_and_accepts_override(client, tmp_path):
    p = make_csv(tmp_path / "ecostress_sample.csv")
    r = client.post("/api/import", files=[("files", ("ecostress_sample.csv", p.read_bytes()))], data={"library": "E"})
    assert r.status_code == 200
    assert {x["acquisition_type"] for x in client.get("/api/spectra/search").json()["results"]} == {"lab"}
    q = make_csv(tmp_path / "plain.csv")
    client.post("/api/import", files=[("files", ("plain.csv", q.read_bytes()))], data={"library": "P", "acquisition": "field"})
    got = {x["library"]: x["acquisition_type"] for x in client.get("/api/spectra/search").json()["results"]}
    assert got["P"] == "field"


def test_aggregation_pools_two_scenes_on_a_common_grid_and_saves(client, tmp_path):
    a = _upload_scene(client, tmp_path / "a", 12)           # 450.. nm, 12 bands
    b = _upload_scene(client, tmp_path / "b", 16)           # different band count and spacing
    left = [[2, 2], [2, 10], [15, 10], [15, 2]]
    body = {"rois": [{"image_id": a, "class_name": "left", "polygon": left}, {"image_id": b, "class_name": "left", "polygon": left}],
            "mode": "intersection"}
    r = client.post("/api/aggregate", json=body)
    assert r.status_code == 200, r.text
    d = r.json()
    cl = d["classes"][0]
    assert cl["name"] == "left" and len(cl["scenes"]) == 2 and not d["mixed_units"]
    per_scene = client.post(f"/api/images/{a}/roi", json={"polygon": left}).json()["n_pixels"]        # the single-ROI endpoint
    assert cl["n_pixels"] == 2 * per_scene and len(cl["mean"]) == len(d["wavelengths"]) == len(cl["std"])
    assert d["wavelengths"][0] == 450 and d["wavelengths"][-1] == 890                         # range both scenes cover
    assert abs(cl["mean"][0] - 0.2) < 1e-6 and cl["std"][0] == 0                              # flat test pixels: std 0
    # union keeps the whole range; where only the 16-band scene reaches, fewer pixels contribute
    u = client.post("/api/aggregate", json=dict(body, mode="union")).json()["classes"][0]
    assert u["n"][0] == 2 * per_scene and u["n"][-1] == per_scene and len(u["mean"]) == 16
    # save as a library entry with the std kept in the metadata
    s = client.post("/api/aggregate", json=dict(body, save={"library": "Aggregated", "category": "Test/Left", "acquisition_type": "airborne"})).json()
    assert s["saved"]["saved"] == ["left"]
    hit = client.get("/api/spectra/search", params={"library_id": s["saved"]["library_id"]}).json()["results"][0]
    assert hit["category"] == "Test/Left" and hit["acquisition_type"] == "airborne"
    sp = client.get(f"/api/spectra/{hit['spectrum_id']}").json()
    assert sp["metadata"]["purity"] == "aggregated" and len(sp["metadata"]["std"]) == len(sp["values"]) and len(sp["metadata"]["source_scenes"]) == 2
    # exports
    assert client.post("/api/aggregate/export/csv", json=body).text.startswith("wavelength_nm,left_mean,left_std,left_n")
    assert client.post("/api/aggregate/export/json", json=body).json()["classes"][0]["name"] == "left"
    z = zipfile.ZipFile(io.BytesIO(client.post("/api/aggregate/export/sli", json=body).content))
    assert sorted(z.namelist()) == ["aggregated_signatures.hdr", "aggregated_signatures.sli", "aggregated_signatures_std.hdr", "aggregated_signatures_std.sli"]
    assert "ENVI Spectral Library" in z.read("aggregated_signatures.hdr").decode()
    assert client.post("/api/aggregate/export/xyz", json=body).status_code == 400
    assert client.post("/api/aggregate", json={"rois": [{"image_id": a, "polygon": [[100, 100], [100, 110], [110, 110]]}]}).status_code == 400


def test_excel_and_json_spectra_can_be_imported(client, tmp_path):
    import json, openpyxl
    wb = openpyxl.Workbook(); ws = wb.active; ws.title = "lab"
    ws.append(["Wavelength", "Grass", "Soil"])
    for i, w in enumerate(range(400, 800, 50)):
        ws.append([w, 0.1 + 0.01 * i, 0.3 - 0.01 * i])
    p = tmp_path / "samples.xlsx"; wb.save(p)
    r = client.post("/api/import", files=[("files", ("samples.xlsx", p.read_bytes()))], data={"library": "Excel lib"})
    assert r.status_code == 200 and r.json()["imported"] == 2, r.text
    j = {"wavelengths": [0.4, 0.5, 0.6, 0.7], "spectra": [{"name": "Moss", "values": [0.1, 0.2, 0.3, 0.2]}]}   # micrometres
    r = client.post("/api/import", files=[("files", ("moss.json", json.dumps(j).encode()))], data={"library": "Json lib"})
    assert r.status_code == 200 and r.json()["imported"] == 1, r.text
    hit = [h for h in client.get("/api/spectra/search", params={"q": "Moss"}).json()["results"]][0]
    assert hit["wavelength_start"] == 400 and hit["wavelength_end"] == 700
    assert client.post("/api/import", files=[("files", ("bad.json", b"{}"))]).status_code == 400


def test_aggregation_json_export_can_be_imported_back(client, tmp_path):
    a = _upload_scene(client, tmp_path / "a", 12)
    body = {"rois": [{"image_id": a, "class_name": "left", "polygon": [[2, 2], [2, 10], [15, 10], [15, 2]]}]}
    exported = client.post("/api/aggregate/export/json", json=body).content
    r = client.post("/api/import", files=[("files", ("agg.json", exported))], data={"library": "Round trip"})
    assert r.status_code == 200 and r.json()["imported"] == 1, r.text


def test_builder_grid_sources_ascii_parse_and_resample(client, tmp_path):
    a = _upload_scene(client, tmp_path / "a", 12)
    src = client.get("/api/builder/sources").json()
    assert [s for s in src if s["kind"] == "image" and s["bands"] == 12]
    w = client.get(f"/api/builder/sources/image/{a}/wavelengths").json()
    assert w["total_bands"] == 12 and w["wavelengths"][0] == 450
    sub = client.get(f"/api/builder/sources/image/{a}/wavelengths", params={"band_from": 3, "band_to": 6}).json()
    assert sub["wavelengths"] == [530, 570, 610, 650]
    assert client.get(f"/api/builder/sources/image/{a}/wavelengths", params={"band_from": 5, "band_to": 99}).status_code == 400
    # ASCII wavelengths: header + extra columns skipped, micrometres converted
    txt = b"# output wavelengths\nwavelength\n0.45 band1\n0.50, band2\n0.55\n0.60\n"
    r = client.post("/api/builder/wavelengths/ascii", files={"file": ("wl.txt", txt)}).json()
    assert r["wavelengths"] == [450, 500, 550, 600] and r["converted_from_um"]
    assert client.post("/api/builder/wavelengths/ascii", files={"file": ("wl.txt", b"1\n2\n")}).status_code == 400
    # parse any spectra file (CSV) and resample to the image grid
    csv_bytes = make_csv(tmp_path / "pts.csv").read_bytes()
    parsed = client.post("/api/builder/parse", files=[("files", ("pts.csv", csv_bytes)), ("files", ("junk.bin", b"\x00\x01"))]).json()
    assert len(parsed["spectra"]) == 2 and parsed["errors"][0]["file"] == "junk.bin"
    sp = parsed["spectra"][0]
    body = {"spectra": [{"name": sp["name"], "wavelengths": sp["wavelengths"], "values": sp["values"], "unit": sp["unit"]}],
            "grid": w["wavelengths"], "min_coverage": 0.2}
    got = client.post("/api/builder/resample", json=body).json()
    assert got["grid"] == w["wavelengths"] and len(got["spectra"][0]["values"]) == 12
    # 'first input spectrum': the grid is that spectrum's own wavelengths
    first = client.post("/api/builder/resample", json=dict(body, grid=None)).json()
    assert first["grid"] == sorted(sp["wavelengths"]) and first["spectra"][0]["coverage"] == 1.0
    # too little overlap is rejected with the reason
    far = {"spectra": [{"name": "far", "wavelengths": [3000, 3100, 3200], "values": [1, 2, 3]}], "grid": w["wavelengths"]}
    assert "covers only 0%" in client.post("/api/builder/resample", json=far).json()["rejected"][0]["reason"]


def test_taxonomy_helpers():
    from app.services.taxonomy import normalize_category, normalize_acquisition, infer_acquisition, rollup
    assert normalize_category("Vegetation > Tree -> Tsuga | Canadensis\\x") == "Vegetation/Tree/Tsuga/Canadensis/x"
    assert normalize_category("  ") is None and normalize_category(None) is None
    assert normalize_acquisition(" Lab ") == "lab" and normalize_acquisition("") is None
    import pytest
    with pytest.raises(ValueError):
        normalize_acquisition("drone")
    assert infer_acquisition("ecostress_library_all_spectra.txt") == "lab" and infer_acquisition("scene.sig") == "field"
    assert infer_acquisition("soil_spectra.txt") is None                      # nothing is invented
    assert rollup([("A/B", 2), ("A", 1), (None, 4)]) == {"A": 3, "A/B": 2, "__none__": 4}


def test_command_line_import_list_and_aggregate(client, tmp_path, capsys):
    import json
    from app import cli
    d, h = _envi_with_map(tmp_path / "s" if (tmp_path / "s").mkdir() is None else tmp_path, nb=10)
    csv_path = make_csv(tmp_path / "lib.csv")
    assert cli.main(["import", str(csv_path), "--library", "CLI lib", "--acquisition", "field", "--category", "A > B"]) == 0
    assert cli.main(["import", str(h), str(tmp_path / "does_not_exist")]) == 1          # a bad path does not stop the good one
    out = capsys.readouterr()
    assert "1 image(s)" in out.out and "does_not_exist" in out.err
    assert cli.main(["list"]) == 0
    assert "CLI lib" in capsys.readouterr().out
    spec = tmp_path / "rois.json"
    spec.write_text(json.dumps({"rois": [{"image": "scene", "class_name": "L", "polygon": [[2, 2], [2, 10], [15, 10], [15, 2]]}],
                                "save": {"library": "CLI agg", "acquisition_type": "airborne"}}))
    assert cli.main(["aggregate", str(spec), "--out", str(tmp_path / "res")]) == 0
    assert sorted(p.name for p in (tmp_path / "res").iterdir()) == ["aggregated_signatures.csv", "aggregated_signatures.json", "aggregated_signatures_sli.zip"]
    hit = client.get("/api/spectra/search", params={"q": "L"}).json()["results"]
    assert any(x["library"] == "CLI agg" and x["acquisition_type"] == "airborne" for x in hit)
    assert cli.main(["formats"]) == 0


def test_image_analysis_operations(client, tmp_path):
    iid = _upload_scene(client, tmp_path / "ops", 16)
    url = f"/api/images/{iid}/analyze"
    for op, params, kind in [("pca", {"n_components": 3}, "rgb"), ("kmeans", {"k": 3}, "classes"), ("anomaly", {}, "scalar"),
                             ("snr", {}, "table"), ("bandmath", {"expression": "(b10 - b3) / (b10 + b3)"}, "scalar"),
                             ("bandmath", {"expression": "mean(w600, w700) * 2"}, "scalar")]:
        r = client.post(f"{url}/{op}", json={"params": params})
        assert r.status_code == 200, (op, r.text)
        assert r.json()["kind"] == kind
    k = client.post(f"{url}/kmeans", json={"params": {"k": 3}}).json()
    assert k["spectra"] and len(k["spectra"][0]["values"]) == 16
    em = [{"name": s["name"], "values": s["values"]} for s in k["spectra"][:3]]
    r = client.post(f"{url}/sam", json={"params": {"endmembers": em}}); assert r.status_code == 200, r.text
    assert sum(l["pixels"] for l in r.json()["legend"]) > 0
    r = client.post(f"{url}/unmix", json={"params": {"endmembers": em}}); assert r.status_code == 200, r.text
    assert len(r.json()["extra"]) == len(em) + 1            # one abundance map per endmember + RMSE
    for bad in ("__import__('os')", "b999", "b1 +", "open('x')"):
        assert client.post(f"{url}/bandmath", json={"params": {"expression": bad}}).status_code == 400
    assert client.post(f"{url}/nonsense", json={}).status_code == 400
    assert client.get("/api/local").status_code == 200
    assert client.post("/api/import/path", json={"paths": ["/tmp/x"]}).status_code == 403   # not the desktop program


def test_folder_libraries_are_marked_main_and_can_be_reloaded(client, tmp_path, monkeypatch):
    import time
    folder = tmp_path / "libs"; folder.mkdir()
    make_csv(folder / "soil_spectra.csv")
    monkeypatch.setenv("AUTO_IMPORT_DIR", str(folder))
    assert client.post("/api/startup_import/reload").status_code == 200
    for _ in range(60):
        st = client.get("/api/startup_import").json()
        if not st["running"]: break
        time.sleep(0.2)
    assert st["folder"] == str(folder) and st["folder_exists"]
    libs = client.get("/api/libraries/").json()
    assert libs and all(l["metadata"].get("main") for l in libs), libs
    mine = client.post("/api/spectra/save", json={"library": "Mine", "items": [
        {"name": "a", "wavelengths_nm": [400, 500, 600], "values": [.1, .2, .3], "unit": "reflectance"}]}).json()
    assert not [l for l in client.get("/api/libraries/").json() if l["name"] == "Mine"][0]["metadata"].get("main")
