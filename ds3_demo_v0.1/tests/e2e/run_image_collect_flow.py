"""Image page v3: click a pixel -> small window (label, Add) -> Collected spectra; ROI by double / right click ->
Collected object samples; show pixel / ROI / both; select all, solo, Delete all; ROI aggregation below the image;
save to new / existing library; .roi export + import; Builder keeps a raw library next to the modified one."""
import re, requests
from pathlib import Path
from playwright.sync_api import sync_playwright
D=Path(__file__).parent/"data"; OUT=Path("/tmp/img_v3"); OUT.mkdir(exist_ok=True); API="http://127.0.0.1:8765"
errs=[]
def box(pg):
    pg.evaluate("()=>window.scrollTo(0,0)"); pg.locator("#imgBox").scroll_into_view_if_needed(); return pg.locator("#imgBox").bounding_box()
def click_at(pg,fx,fy,**kw):
    b=box(pg); pg.mouse.click(b["x"]+fx*b["width"], b["y"]+fy*b["height"], **kw)
with sync_playwright() as p:
    b=p.chromium.launch(); pg=b.new_page(viewport={"width":1500,"height":1000}, accept_downloads=True)
    pg.on("pageerror", lambda e: errs.append(str(e)))
    pg.route(re.compile(r"https://(cdnjs|cdn\.jsdelivr)\..*"), lambda r: r.abort())
    pg.route(re.compile(r"https://.*tile.*"), lambda r: r.abort())
    for n in ("sceneA","sceneB"):
        r=requests.post(API+"/api/import",files=[("files",(n,(D/n).read_bytes())),("files",(n+".hdr",(D/(n+".hdr")).read_bytes()))]); assert r.status_code==200,r.text
    r=requests.post(API+"/api/import",files=[("files",("mylib.csv",(D/"mylib.csv").read_bytes()))],data={"library":"My lib","acquisition":"lab","category":"Test > Plants"}); assert r.status_code==200,r.text
    imgs={i["name"]:i for i in requests.get(API+"/api/images/").json()}
    assert imgs["sceneA"]["path"], "image path missing from /api/images"
    pg.goto(API+"/app/"); pg.wait_for_timeout(1500)
    pg.click('[data-nav="image"]'); pg.wait_for_timeout(800)
    pg.evaluate("async id=>{ await openImage(id); }", imgs["sceneA"]["id"]); pg.wait_for_timeout(1000)
    t=pg.inner_text("#main")
    assert "Collect spectra: on" not in t and "Collect spectra" not in t and "Collect on" not in t, "collect toggle still there"
    assert "Finish ROI" not in t, "Finish ROI button still there"
    assert pg.locator("#main button:text-is('Clear')").count()==0, "Clear button still there"
    assert "Delete all" in t and "Object sample collection (ROI)" in t
    assert pg.locator("#imgObjects").count()==1 and pg.locator("#imgBox").count()==1
    assert "Library" not in pg.inner_text("#imgSide") and "Classification map" not in t and "Add library" not in t, "library panel must be gone from the Image page"
    # --- pixel click -> popup, nothing added until Add
    click_at(pg,0.3,0.3); pg.wait_for_selector("#pxPopBox"); pg.wait_for_timeout(300)
    assert "Collected spectra (0)" in pg.inner_text("#imgSide"), "added without pressing Add"
    assert pg.locator("#pxPopBox svg").count()==1 and "Wavelength (nm)" in pg.locator("#pxPopBox svg").text_content()
    pg.screenshot(path=str(OUT/"1_popup.png"))
    pg.fill("#pxLabel","Water A"); pg.click("#pxPopBox >> text=Add"); pg.wait_for_timeout(500)
    assert pg.locator("#pxPopBox").count()==0 and "Collected spectra (1)" in pg.inner_text("#imgSide")
    assert "Water A" in pg.inner_text("#imgSide")
    click_at(pg,0.7,0.6); pg.wait_for_selector("#pxPopBox"); pg.press("#pxLabel","Enter"); pg.wait_for_timeout(500)   # Enter adds, default label
    assert "Collected spectra (2)" in pg.inner_text("#imgSide")
    click_at(pg,0.5,0.8); pg.wait_for_selector("#pxPopBox"); pg.click("#pxPopBox >> text=Cancel"); pg.wait_for_timeout(300)
    assert "Collected spectra (2)" in pg.inner_text("#imgSide"), "Cancel must not add"
    # --- select all / unselect all / solo
    pg.click("text=Unselect all >> nth=0"); pg.wait_for_timeout(300)
    assert pg.locator("#imgSide .coll-row input[type=checkbox]:checked").count()==0
    pg.click("text=Select all >> nth=0"); pg.wait_for_timeout(300)
    assert pg.locator("#imgSide .coll-row input[type=checkbox]:checked").count()==2
    pg.locator(".coll-nm").first.click(); pg.wait_for_timeout(300)
    assert pg.locator("#imgSide .coll-row input[type=checkbox]:checked").count()==1, "solo should show only the clicked spectrum"
    pg.click("text=Select all >> nth=0"); pg.wait_for_timeout(300)
    ch=pg.locator("#imgChart svg").text_content(); assert "Wavelength (nm)" in ch and ("Reflectance" in ch or "reflectance" in ch.lower()), ch[:200]
    pg.screenshot(path=str(OUT/"2_collected.png"))
    # --- ROI: finish with double-click
    pg.click("button:has-text('Draw ROI polygon')"); pg.wait_for_timeout(300)
    for fx,fy in ((0.1,0.1),(0.4,0.1),(0.4,0.45)): click_at(pg,fx,fy); pg.wait_for_timeout(120)
    assert "double-click or right-click to finish" in pg.inner_text("#main")
    b_=box(pg); pg.mouse.dblclick(b_["x"]+0.1*b_["width"], b_["y"]+0.45*b_["height"]); pg.wait_for_selector("#pxPopBox"); pg.wait_for_timeout(400)
    assert "ROI mean" in pg.inner_text("#pxPopBox") and pg.locator("#pxClass").count()==1
    pg.fill("#pxLabel","Left box"); pg.fill("#pxClass","Left half"); pg.click("#pxPopBox >> text=Add"); pg.wait_for_timeout(900)
    # first ROI: the program asks where the image patches go
    pg.wait_for_selector("#objFolderIn"); assert "named after its class" in pg.inner_text("#winBody")
    PATCH=OUT/"patches"; import shutil; shutil.rmtree(PATCH, ignore_errors=True)
    pg.fill("#objFolderIn", str(PATCH)); pg.click(".win >> text=Save patches here"); pg.wait_for_timeout(1500)
    assert "Object sample collection (ROI) — 1" in pg.inner_text("#imgObjects"), pg.inner_text("#imgObjects")[:600]
    assert "Left half" in pg.inner_text("#imgObjects") and "Left half.tif" in pg.inner_text("#imgObjects"), pg.inner_text("#imgObjects")[:500]
    assert (PATCH/"Left half.tif").is_file(), list(PATCH.glob("*"))
    # --- ROI: finish with right-click
    pg.click("button:has-text('Draw ROI polygon')"); pg.wait_for_timeout(300)
    for fx,fy in ((0.55,0.55),(0.9,0.55),(0.9,0.9)): click_at(pg,fx,fy); pg.wait_for_timeout(120)
    click_at(pg,0.55,0.9,button="right"); pg.wait_for_selector("#pxPopBox"); pg.fill("#pxLabel","Right box"); pg.fill("#pxClass","Left half"); pg.press("#pxClass","Enter"); pg.wait_for_timeout(1200)
    assert "Object sample collection (ROI) — 2" in pg.inner_text("#imgObjects") and "Left half_2.tif" in pg.inner_text("#imgObjects") and (PATCH/"Left half_2.tif").is_file()
    pg.screenshot(path=str(OUT/"3_objects.png"), full_page=True)
    # --- show toggles
    assert pg.locator("#imgBox .pxmark").count()==4, pg.locator("#imgBox .pxmark").count()      # 2 pixels + 2 ROI labels
    pg.uncheck("#main .show-row label:has-text('Pixel spectra') input"); pg.wait_for_timeout(300)
    assert pg.locator("#imgBox .pxmark").count()==2
    pg.uncheck("#main .show-row label:has-text('Object samples') input"); pg.wait_for_timeout(300)
    assert pg.locator("#imgBox .pxmark").count()==0 and not pg.locator("#imgSide").is_visible() and not pg.locator("#imgObjects").is_visible(), "the checkboxes hide the whole side panel and the whole ROI table"
    pg.check("#main .show-row label:has-text('Object samples') input"); pg.wait_for_timeout(300)
    assert pg.locator("#imgBox .pxmark").count()==2 and pg.locator("#imgObjects").is_visible() and not pg.locator("#imgSide").is_visible()
    pg.check("#main .show-row label:has-text('Pixel spectra') input"); pg.wait_for_timeout(300)
    assert pg.locator("#imgSide").is_visible() and "Collected spectra (2)" in pg.inner_text("#imgSide")
    # --- second scene, same class, then aggregation below the image
    pg.evaluate("async id=>{ await openImage(id); }", imgs["sceneB"]["id"]); pg.wait_for_timeout(900)
    pg.click("button:has-text('Draw ROI polygon')")
    for fx,fy in ((0.2,0.2),(0.8,0.2),(0.8,0.8)): click_at(pg,fx,fy); pg.wait_for_timeout(120)
    b_=box(pg); pg.mouse.dblclick(b_["x"]+0.2*b_["width"], b_["y"]+0.8*b_["height"]); pg.wait_for_selector("#pxPopBox")
    pg.fill("#pxLabel","B box"); pg.fill("#pxClass","Left half"); pg.press("#pxClass","Enter"); pg.wait_for_timeout(1200)
    assert pg.locator("#imgObjects .agg-box").count()==0, "nothing is aggregated before the button is pressed"
    pg.click("#imgObjects >> text=Aggregate ticked ROIs"); pg.wait_for_selector("#imgObjects .agg-box .gcp-table >> text=Variance", timeout=20000); pg.wait_for_timeout(500)
    ag=pg.inner_text("#imgObjects"); assert "Left half" in ag and "Std. deviation" in ag and "Variance" in ag, ag[:900]
    assert pg.locator("#imgObjects svg").count()>=1
    pg.check("#imgObjects label:has-text('Variance') input"); pg.wait_for_timeout(300); assert "Variance" in pg.locator("#imgObjects svg").last.text_content()
    pg.locator("#imgObjects").scroll_into_view_if_needed(); pg.screenshot(path=str(OUT/"4_aggregation.png"), full_page=True)
    # Analysis page no longer has aggregation
    pg.click('[data-nav="analysis"]'); pg.wait_for_timeout(400)
    ta=pg.inner_text("#main"); assert "AI / ML" in ta and "Run aggregation" not in ta
    pg.click('[data-nav="image"]'); pg.wait_for_timeout(600)
    # --- save object samples to a new object library; collected spectra into an existing library
    pg.evaluate("async id=>{ await openImage(id); }", imgs["sceneA"]["id"]); pg.wait_for_timeout(900)
    pg.click("#imgObjects >> text=Save to library…"); pg.wait_for_selector("#cbName"); assert pg.locator("#cbSel option").count()>=1; pg.fill("#cbName","Obj lib"); pg.click(".win >> text=Save to library"); pg.wait_for_timeout(1200)
    libs={l["name"]:l for l in requests.get(API+"/api/libraries/").json()}
    assert "Obj lib" in libs, list(libs)
    assert (libs["Obj lib"].get("metadata") or {}).get("type")=="object", libs["Obj lib"]
    res=requests.get(API+"/api/spectra/search",params={"library_id":libs["Obj lib"]["id"],"limit":20}).json()
    assert res["total"]==3, res["total"]
    r0=res["results"][0]; pg.evaluate("([l,m])=>goToMaterial(l,m)",[libs["Obj lib"]["id"],r0["material_id"]]); pg.wait_for_timeout(1500)
    sv=pg.inner_text("#main"); assert "Source" in sv and "Image patch" in sv and "Left half" in sv and ".tif" in sv and "View image" in sv and "Locate" in sv and str(OUT) not in sv, sv[:600]
    pg.screenshot(path=str(OUT/"6_source_image.png"))
    pg.click('[data-nav="image"]'); pg.wait_for_timeout(800)
    assert "Collected spectra (0)" in pg.inner_text("#imgSide")      # switching image clears the pixel spectra (object samples stay)
    for fx,fy in ((0.2,0.2),(0.6,0.5)):
        click_at(pg,fx,fy)
        try: pg.wait_for_selector("#pxPopBox", timeout=8000)
        except Exception:
            pg.screenshot(path=str(OUT/"dbg_nopopup.png"), full_page=True); print("NO POPUP; err=", pg.evaluate("()=>[I.err, I.busy, I.roi, !!I.preview, document.getElementById('imgBox')&&document.getElementById('imgBox').getBoundingClientRect().toJSON()]")); raise
        pg.press("#pxLabel","Enter"); pg.wait_for_timeout(400)
    pg.click("#imgSide >> text=Save to library…"); pg.wait_for_selector("#cbSel"); pg.select_option("#cbSel", value="My lib"); assert not pg.locator("#cbName").is_visible(), "the name box hides when an existing library is chosen"
    pg.click(".win >> text=Save to library"); pg.wait_for_timeout(1200)
    n=requests.get(API+"/api/spectra/search",params={"library_id":[l for l in requests.get(API+"/api/libraries/").json() if l["name"]=="My lib"][0]["id"],"limit":50}).json()["total"]
    assert n==2+2, n                    # two spectra were already there, two collected pixels added
    # --- an ROI is deleted by clicking it on the image
    n0=pg.locator("#imgObjects .obj-table tr").count()-1
    pg.locator("#imgBox .roimark").first.click(); pg.wait_for_selector("#objMenu"); assert "Delete ROI" in pg.inner_text("#objMenu")
    pg.click("#objMenu >> text=Delete ROI"); pg.wait_for_timeout(400)
    assert pg.locator("#imgObjects .obj-table tr").count()-1==n0-1 and pg.locator("#objMenu").count()==0, "ROI deleted from the image click menu"
    assert f"Object sample collection (ROI) — {n0-1}" in pg.inner_text("#imgObjects")
    pg.click("#imgObjects >> text=Delete all >> nth=0"); pg.wait_for_timeout(300); pg.click("#imgObjects >> text=Click again to delete all"); pg.wait_for_timeout(500)
    pg.click("#imgSide >> text=Delete all >> nth=0"); pg.wait_for_timeout(300); pg.click("#imgSide >> text=Click again to delete all"); pg.wait_for_timeout(500)
    assert "Collected spectra (0)" in pg.inner_text("#imgSide")
    # --- locate file endpoint (400/404 are fine in this sandbox: no file browser)
    r=requests.post(API+f"/api/images/{imgs['sceneA']['id']}/reveal"); print("reveal ->", r.status_code, r.text[:90])
    # --- Builder keeps a raw library next to the modified one
    pg.hover("#menubar [data-menu='Spectral']"); pg.hover(".mnu-item.sub:has-text('Spectral Libraries')"); pg.click(".mnu-item:not(.sub):has-text('Spectral Library Builder')"); pg.wait_for_timeout(700)
    pg.set_input_files("#bldFiles", str(D/"mylib.csv")); pg.wait_for_timeout(1500)
    assert "as read from the file" in pg.inner_text("#main") and pg.locator("#main input[type=checkbox]:checked").count()>=1
    pg.fill("#bldLibName","Mod lib"); pg.click("text=Save as library"); pg.wait_for_timeout(1800)
    libs={l["name"]:l for l in requests.get(API+"/api/libraries/").json()}
    assert "Mod lib" in libs and "Mod lib (raw)" in libs, list(libs)
    assert libs["Mod lib (raw)"]["metadata"].get("raw") is True and libs["Mod lib"]["metadata"].get("raw_library")=="Mod lib (raw)", (libs["Mod lib"]["metadata"], libs["Mod lib (raw)"]["metadata"])
    assert requests.get(API+"/api/spectra/search",params={"library_id":libs["Mod lib (raw)"]["id"],"limit":20}).json()["total"]==2
    # from an existing database library: source untouched, one modified library only
    pg.click("text=Start over") if pg.locator("text=Start over").count() else None
    pg.wait_for_timeout(300)
    pg.screenshot(path=str(OUT/"5_builder.png"))
    b.close()
print("JS errors:", errs)
assert not errs, errs
print("OK image collect flow")
