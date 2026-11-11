import re, requests
from pathlib import Path
from playwright.sync_api import sync_playwright
D=Path(__file__).parent/"data"; OUT=Path(__file__).parent; API="http://127.0.0.1:8765"
errs=[]
with sync_playwright() as p:
    b=p.chromium.launch(); pg=b.new_page(viewport={"width":1300,"height":900})
    pg.on("pageerror", lambda e: errs.append(str(e)))
    pg.route(re.compile(r"https://(cdnjs|cdn\.jsdelivr)\..*"), lambda r: r.abort())
    pg.route(re.compile(r"https://.*tile.*"), lambda r: r.abort())
    for n in ("sceneA","sceneB"):
        r=requests.post(API+"/api/import",files=[("files",(n,(D/n).read_bytes())),("files",(n+".hdr",(D/(n+".hdr")).read_bytes()))]); assert r.status_code==200,r.text
    r=requests.post(API+"/api/import",files=[("files",("mylib.csv",(D/"mylib.csv").read_bytes()))],data={"library":"My lib","acquisition":"lab","category":"Test > Plants"}); assert r.status_code==200,r.text
    pg.goto(API+"/app/"); pg.wait_for_timeout(1500)
    # header menu -> Builder
    pg.hover("#menubar [data-menu='Spectral']"); pg.hover(".mnu-item.sub:has-text('Spectral Libraries')"); pg.click(".mnu-item:not(.sub):has-text('Spectral Library Builder')"); pg.wait_for_timeout(700)
    assert pg.locator(".lf-title:has-text('Spectral Library Builder')").count()==1, "step 1 dialog missing"
    pg.screenshot(path=str(OUT/"s1.png"))
    pg.click(".lf-btns >> text=OK"); pg.wait_for_timeout(500)
    assert pg.locator(".lf-title:has-text('File Containing Output Wavelength')").count()==1
    pg.locator(".lf-item:has-text('sceneA')").first.click(); pg.wait_for_timeout(600)
    pg.screenshot(path=str(OUT/"s2.png"))
    assert "30/30 Bands" in pg.inner_text(".lf-dialog"), pg.inner_text(".lf-dialog")
    pg.click("text=Spectral Subset"); pg.locator(".lf-dialog input[type=number]").nth(1).fill("20"); pg.locator(".lf-dialog input[type=number]").nth(1).dispatch_event("change"); pg.wait_for_timeout(300)
    assert "20/30 Bands" in pg.inner_text(".lf-dialog")
    pg.click(".lf-btns >> text=OK"); pg.wait_for_timeout(700)
    t=pg.inner_text("#main"); assert "20 bands" in t and "sceneA" in t, t[:400]
    # import from library
    pg.hover(".lfmenu"); pg.click("text=Spectral library (database)"); pg.wait_for_timeout(500)
    pg.select_option(".lf-dialog select", label="My lib (2)"); pg.wait_for_timeout(800)
    pg.locator(".lf-dialog input[type=checkbox]").nth(0).check(); pg.locator(".lf-dialog input[type=checkbox]").nth(1).check()
    pg.click("text=Import 2 selected"); pg.wait_for_timeout(1200)
    assert "Spectra (2)" in pg.inner_text("#main"), pg.inner_text("#main")[:500]
    pg.set_input_files("#bldFiles", str(D/"mylib.csv")); pg.wait_for_timeout(1200)
    assert "Spectra (4)" in pg.inner_text("#main")
    pg.screenshot(path=str(OUT/"s3.png"))
    pg.fill("#bldLib","Built lib"); pg.fill("#bldCat","Plants > Grass"); pg.select_option("#bldAcq","field"); pg.click("text=Save as library"); pg.wait_for_timeout(1500)
    res=requests.get(API+"/api/spectra/search",params={"limit":50}).json()["results"]
    built=[x for x in res if x["library"]=="Built lib"]
    assert len(built)==4 and {x["acquisition_type"] for x in built}=={"field"} and {x["category"] for x in built}=={"Plants/Grass"}, built
    # ASCII wavelengths
    pg.on("dialog", lambda d: d.accept())
    pg.click("text=Change…"); pg.wait_for_timeout(500)
    pg.check("input[value=ascii]"); pg.click(".lf-btns >> text=OK"); pg.wait_for_timeout(200)
    pg.set_input_files("#bldAsciiFile", str(D/"wl.txt")); pg.wait_for_timeout(1000)
    t=pg.inner_text("#main"); assert "16 bands" in t and "ASCII file wl.txt" in t, t[:300]
    # Viewer filters
    pg.hover("#menubar [data-menu='Spectral']"); pg.hover(".mnu-item.sub:has-text('Spectral Libraries')"); pg.click(".mnu-item:not(.sub):has-text('Spectral Library Viewer')"); pg.wait_for_timeout(600)
    pg.select_option("#main select >> nth=0", label="My lib (2)"); pg.wait_for_timeout(1000)
    pg.screenshot(path=str(OUT/"s4.png"))
    assert pg.locator("#main select").count()>=3 and "lab" in pg.inner_text("#main")
    pg.select_option("#main select >> nth=2", value="lab"); pg.wait_for_timeout(700)
    assert pg.locator("#lfList label").count()==2
    # aggregation: two scenes, one class each, run, export, save
    ids={i["name"]:i["id"] for i in requests.get(API+"/api/images/").json()}
    pg.click('[data-nav="image"]'); pg.wait_for_timeout(800)
    for nm in ("sceneA","sceneB"):
        pg.evaluate("async id=>{ await openImage(id); }", ids[nm]); pg.wait_for_timeout(800)
        pg.evaluate("()=>{ I.roi={pts:[[5,5],[5,25],[35,25],[35,5]],closed:true,drawing:false}; refreshAll(); }"); pg.wait_for_timeout(300)
        pg.click("text=+ ROI to aggregation"); pg.fill("#agName","Left half"); pg.click(".lf-btns >> text=Add"); pg.wait_for_timeout(300)
    pg.click('[data-nav="analysis"]'); pg.wait_for_timeout(500)
    assert "ROIs (2)" in pg.inner_text("#main")
    pg.click("text=Run aggregation"); pg.wait_for_timeout(2500)
    t=pg.inner_text("#main"); assert "Result" in t and "Left half" in t and "sceneA, sceneB" in t, t[:600]
    pg.screenshot(path=str(OUT/"s5.png"), full_page=True)
    with pg.expect_download() as dl: pg.click("text=ENVI .sli (zip)")
    assert dl.value.suggested_filename=="aggregated_signatures_sli.zip"
    pg.fill("input[title='library name']","Agg lib"); pg.click("text=Save mean spectra"); pg.wait_for_timeout(1500)
    agg=[x for x in requests.get(API+"/api/spectra/search",params={"limit":50}).json()["results"] if x["library"]=="Agg lib"]; assert len(agg)==1, agg
    b.close()
print("page errors:", errs)
print("E2E OK")
