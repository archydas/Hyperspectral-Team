"""Object samples (ROI), zoom, georeferencing and the Library module:
 toolbar line (Show checkboxes next to Download image) · mouse-wheel zoom and drag · ROI -> folder question -> <class>.tif patch ·
 per-ROI spectrum · georeference with control points (image beside the map) · lat/lon in the table ·
 Library page: pixel-based vs object-based libraries, patch file name only, Load image.
The map needs Leaflet: set LEAFLET_DIST to the folder with leaflet.js / leaflet.css (npm i leaflet) when there is no internet."""
import os, re, shutil, requests
from pathlib import Path
from playwright.sync_api import sync_playwright
D=Path(__file__).parent/"data"; OUT=Path("/tmp/img_obj"); OUT.mkdir(exist_ok=True); API="http://127.0.0.1:8765"
LEAF=Path(os.environ["LEAFLET_DIST"]) if os.environ.get("LEAFLET_DIST") else None
errs=[]; PATCH=OUT/"patches"; shutil.rmtree(PATCH, ignore_errors=True)
# a scene without map information: georeferencing has something to do
hdr="\n".join(l for l in (D/"sceneA.hdr").read_text().splitlines() if not l.startswith("map info"))
def fr(pg,sel,fx,fy,**kw):
    pg.locator(sel).scroll_into_view_if_needed(); b=pg.locator(sel).bounding_box(); pg.mouse.click(b["x"]+fx*b["width"], b["y"]+fy*b["height"], **kw)
with sync_playwright() as p:
    b=p.chromium.launch(); pg=b.new_page(viewport={"width":1500,"height":1000}, accept_downloads=True)
    pg.on("pageerror", lambda e: errs.append(str(e)))
    def route(r):
        u=r.request.url
        if LEAF and "leaflet" in u and u.endswith(".js"): return r.fulfill(path=str(LEAF/"leaflet.js"), content_type="application/javascript")
        if LEAF and "leaflet" in u and u.endswith(".css"): return r.fulfill(path=str(LEAF/"leaflet.css"), content_type="text/css")
        return r.abort()
    pg.route(re.compile(r"https://.*"), route)
    r=requests.post(API+"/api/import",files=[("files",("sceneN",(D/"sceneA").read_bytes())),("files",("sceneN.hdr",hdr.encode()))]); assert r.status_code==200,r.text
    img=[i for i in requests.get(API+"/api/images/").json() if i["name"]=="sceneN"][0]
    pg.goto(API+"/app/"); pg.wait_for_timeout(1500); pg.click('[data-nav="image"]'); pg.wait_for_timeout(600)
    pg.evaluate("async id=>{ await openImage(id); }", img["id"]); pg.wait_for_timeout(1200)
    # ---- toolbar: Show checkboxes in the same line as Download image; library panel gone
    y1=pg.locator("button:has-text('Download image')").bounding_box()["y"]; y2=pg.locator(".show-row label:has-text('Pixel spectra')").bounding_box()["y"]
    y3=pg.locator(".show-row label:has-text('Object samples')").bounding_box()["y"]; assert abs(y1-y2)<14 and abs(y1-y3)<14, (y1,y2,y3)
    t=pg.inner_text("#main"); assert "Classification map" not in t and "Add library" not in t and "Class spectra from map" not in t
    # ---- wheel zoom (around the pointer) and drag
    bx=pg.locator("#imgBox").bounding_box(); w0=bx["width"]; pg.mouse.move(bx["x"]+bx["width"]*0.3, bx["y"]+bx["height"]*0.3)
    pg.mouse.wheel(0,-240); pg.wait_for_timeout(300); w1=pg.locator("#imgBox").bounding_box()["width"]; assert w1>w0*1.1, (w0,w1)
    pg.mouse.wheel(0,480); pg.wait_for_timeout(300); w2=pg.locator("#imgBox").bounding_box()["width"]; assert w2<w1, (w1,w2)
    pg.click("button:has-text('Fit')"); pg.wait_for_timeout(300)
    b0=pg.locator('#imgVp').bounding_box(); pg.mouse.move(b0['x']+b0['width']/2, b0['y']+b0['height']/2)
    for _ in range(8): pg.mouse.wheel(0,-300); pg.wait_for_timeout(80)
    pg.wait_for_timeout(300)
    sl=pg.evaluate("()=>document.getElementById('imgVp').scrollLeft"); b_=pg.locator("#imgVp").bounding_box()
    pg.mouse.move(b_["x"]+300,b_["y"]+200); pg.mouse.down(); pg.mouse.move(b_["x"]+160,b_["y"]+120,steps=6); pg.mouse.up(); pg.wait_for_timeout(200)
    assert pg.evaluate("()=>document.getElementById('imgVp').scrollLeft")!=sl, "drag should pan"
    assert pg.locator("#pxPopBox").count()==0, "a drag must not pick a pixel"
    pg.click("button:has-text('Fit')"); pg.wait_for_timeout(300)
    assert pg.locator("#imgWl").count()==0 and "wavelengths, only" not in pg.inner_html("#main"), "no wavelength box in the top line"
    assert pg.locator("button:has-text('Clear ROI')").count()==1 and pg.locator("button:has-text('Clear ROI')").is_disabled()
    assert "emerald" not in pg.locator("button:has-text('Draw ROI polygon')").get_attribute("class")
    # ---- ROI -> patch folder question -> <class>.tif   (zoomed in: the view must not jump while points are clicked)
    pg.click("button:has-text('Draw ROI polygon')"); pg.wait_for_timeout(200)
    assert "emerald" in pg.locator("button:has-text('Draw ROI polygon')").get_attribute("class") and pg.locator("button:has-text('Clear ROI')").count()==1
    b0=pg.locator('#imgVp').bounding_box(); pg.mouse.move(b0['x']+b0['width']*0.4, b0['y']+b0['height']*0.4)
    for _ in range(3): pg.mouse.wheel(0,-300); pg.wait_for_timeout(100)
    pg.wait_for_timeout(600)
    zoom0=pg.evaluate("()=>[I.zoom, document.getElementById('imgVp').scrollLeft, document.getElementById('imgVp').scrollTop, document.getElementById('imgBox').getBoundingClientRect().width]")
    vb=pg.locator("#imgVp").bounding_box()
    for fx,fy in ((0.3,0.3),(0.6,0.3),(0.6,0.6)): pg.mouse.click(vb["x"]+vb["width"]*fx, vb["y"]+vb["height"]*fy); pg.wait_for_timeout(200)
    zoom1=pg.evaluate("()=>[I.zoom, document.getElementById('imgVp').scrollLeft, document.getElementById('imgVp').scrollTop, document.getElementById('imgBox').getBoundingClientRect().width]")
    assert zoom0==zoom1, ("zoom / scroll changed while drawing", zoom0, zoom1)
    assert pg.evaluate("()=>I.roi.pts.length")==3
    pg.click("button:has-text('Clear ROI')"); pg.wait_for_timeout(200)
    assert pg.evaluate("()=>I.roi.pts.length")==0 and "emerald" not in pg.locator("button:has-text('Draw ROI polygon')").get_attribute("class"), "Clear ROI also ends drawing: the button is no longer green"
    pg.click("button:has-text('Fit')"); pg.wait_for_timeout(300)
    pg.click("button:has-text('Draw ROI polygon')"); pg.wait_for_timeout(200)
    for fx,fy in ((0.1,0.1),(0.45,0.1),(0.45,0.5)): fr(pg,"#imgBox",fx,fy); pg.wait_for_timeout(120)
    pg.locator("#imgBox").scroll_into_view_if_needed(); bb=pg.locator("#imgBox").bounding_box(); pg.mouse.dblclick(bb["x"]+0.1*bb["width"], bb["y"]+0.5*bb["height"]); pg.wait_for_selector("#pxPopBox")
    pg.fill("#pxLabel","Pond 1"); pg.fill("#pxClass","Water"); pg.click("#pxPopBox >> text=Add"); pg.wait_for_selector("#objFolderIn")
    pg.fill("#objFolderIn", str(PATCH)); pg.click(".win >> text=Save patches here"); pg.wait_for_timeout(1500)
    tbl=pg.inner_text("#imgObjects"); assert "Water.tif" in tbl and (PATCH/"Water.tif").is_file(), tbl[:500]
    assert str(PATCH) not in tbl, "the long folder path must not be shown"
    row=pg.locator("#imgObjects .obj-table tr").nth(1).inner_text(); assert "—" in row, "not georeferenced yet: coordinates are empty"
    # ---- per-ROI spectrum (surface reflectance)
    pg.click("#imgObjects >> text=Spectrum >> nth=0"); pg.wait_for_selector(".win svg"); wt=pg.inner_text(".win")
    assert "Surface reflectance" in pg.locator(".win svg").first.text_content() and "Water" in wt and "Mean std. deviation" in wt, wt[:300]
    pg.screenshot(path=str(OUT/"1_roi_spectrum.png")); pg.click(".win >> text=Close"); pg.wait_for_timeout(200)
    # ---- georeference: image beside the map, 4 control points
    if LEAF:
        assert "emerald" not in pg.get_attribute("#btnGeo","class")
        pg.click("button:has-text('Georeference')"); assert "emerald" in pg.get_attribute("#btnGeo","class"); pg.wait_for_selector("#grMap .leaflet-container, #grMap.leaflet-container"); pg.wait_for_timeout(400)
        assert pg.locator("#grImg").count()==1 and pg.locator("#grMap").count()==1
        lb=lambda fx,fy:(50-1e-4*fy*40, 10+1e-4*fx*60)
        for k,(fx,fy) in enumerate(((0.1,0.1),(0.9,0.1),(0.1,0.9),(0.9,0.9))):
            pg.locator("#grBox").scroll_into_view_if_needed(); bb=pg.locator("#grBox").bounding_box(); pg.mouse.click(bb["x"]+fx*bb["width"], bb["y"]+fy*bb["height"]); pg.wait_for_timeout(150)
            lat,lon=lb(fx,fy); pg.evaluate("([la,lo])=>GR.map.fire('click',{latlng:L.latLng(la,lo)})",[lat,lon]); pg.wait_for_timeout(500)
            if k==1: assert "at least" in pg.inner_text("#grTable") and pg.locator("#grApply").is_disabled(), "needs 3 points"
        assert pg.locator("#grApply").is_enabled(); tx=pg.inner_text("#grTable"); assert "RMSE" in tx, tx
        pg.screenshot(path=str(OUT/"2_georef.png"))
        pg.click("#grApply"); pg.wait_for_timeout(1500)
        assert "emerald" not in pg.get_attribute("#btnGeo","class"), "Georeference is green only while its window is open"
        row=pg.locator("#imgObjects .obj-table tr").nth(1).inner_text(); m=re.findall(r"\d+\.\d{5}",row); assert len(m)>=2, row
        lat=float(m[0]); assert 49.99<lat<50.0, row
    # ---- Library module: pixel-based and object-based libraries
    pg.click("#imgObjects >> text=Save to library…"); pg.wait_for_selector("#cbName"); pg.fill("#cbName","Pond objects"); pg.click(".win >> text=Save to library"); pg.wait_for_timeout(1200)
    fr(pg,"#imgBox",0.7,0.7); pg.wait_for_selector("#pxPopBox"); pg.press("#pxLabel","Enter"); pg.wait_for_timeout(400)
    pg.click("#imgSide >> text=Save to library…"); pg.wait_for_selector("#cbName"); pg.fill("#cbName","Pond pixels"); pg.click(".win >> text=Save to library"); pg.wait_for_timeout(1200)
    pg.click('[data-nav="library-list"]'); pg.wait_for_timeout(900); lt=pg.inner_text("#main")
    assert "Pixel-based libraries" in lt and "Object-based libraries (ROI)" in lt and "Pond objects" in lt and "Pond pixels" in lt, lt[:800]
    assert "Add library" in lt and "Build library" in lt
    assert lt.index("Pond pixels")<lt.index("Object-based libraries (ROI)")<lt.index("Pond objects"), "libraries must sit in their own sections"
    pg.click(".lib-card:has-text('Pond objects')"); pg.wait_for_selector("#matRows .obj-table"); pg.wait_for_timeout(500)
    ot=pg.inner_text("#matRows"); assert "Water.tif" in ot and "Load image" in ot and "Locate" in ot and str(PATCH) not in ot and "/" not in ot.split("Image (patch file)")[1].split("The image patches")[0].replace("The image patches",""), ot[:500]
    pg.screenshot(path=str(OUT/"3_object_library.png"))
    n_img=len(requests.get(API+"/api/images/").json())
    pg.click("#matRows >> text=Load image >> nth=0"); pg.wait_for_selector("#imgBox", timeout=15000); pg.wait_for_timeout(1200)
    assert len(requests.get(API+"/api/images/").json())==n_img+1
    assert "Water" in pg.locator("#imgInfo").inner_text() and "zoom" in pg.locator("#imgInfo").inner_text(), pg.locator("#imgInfo").inner_text()
    pg.screenshot(path=str(OUT/"4_patch_loaded.png"))
    print("JS errors:", errs); b.close()
assert not errs, errs
print("OK object samples flow" + ("" if LEAF else " (map steps skipped: set LEAFLET_DIST)"))
