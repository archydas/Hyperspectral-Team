import re, requests, sys
from pathlib import Path
from playwright.sync_api import sync_playwright
D=Path(__file__).parent/"data"; OUT=Path(__file__).parent; API="http://127.0.0.1:8765"
errs=[]
def shot(pg,n): pg.screenshot(path=str(OUT/f"u_{n}.png"))
with sync_playwright() as p:
    b=p.chromium.launch(); pg=b.new_page(viewport={"width":1440,"height":900})
    pg.on("pageerror", lambda e: errs.append(str(e)))
    pg.on("console", lambda m: errs.append("console: "+m.text) if m.type=="error" and "Failed to load resource" not in m.text else None)
    pg.route(re.compile(r"https://(cdnjs|cdn\.jsdelivr)\..*"), lambda r: r.abort())
    pg.route(re.compile(r"https://.*tile.*"), lambda r: r.abort())
    pg.goto(API+"/app/"); pg.wait_for_timeout(1500)
    # --- home: three modules, no Dashboard tab, no Library function
    t=pg.inner_text("body")
    assert "Dashboard" not in pg.inner_text(".sidebar"), pg.inner_text(".sidebar")
    assert pg.locator(".module-card").count()==3
    assert "Library function" not in t
    shot(pg,"1_home")
    # --- Spectral menu
    pg.hover("#menubar [data-menu='Spectral']"); pg.hover(".mnu-item.sub:has-text('Spectral Libraries')"); pg.wait_for_timeout(300)
    shot(pg,"2_menu")
    pg.click("text=Spectral Library Viewer"); pg.wait_for_timeout(600)
    assert "Spectral Library Viewer" in pg.inner_text("#main") and "Library function" not in pg.inner_text("#main")
    assert pg.locator("#main button:has-text('Spectral Library Builder')").count()==0      # no tab strip any more
    shot(pg,"3_viewer")
    # --- File > Open Image File
    pg.hover("#menubar [data-menu='File']"); pg.wait_for_timeout(200)
    with pg.expect_file_chooser() as fc:
        pg.click(".mnu-item:not(.sub):has-text('Open Image File')")
    fc.value.set_files([str(D/"sceneA"), str(D/"sceneA.hdr")]); pg.wait_for_timeout(2500)
    assert pg.locator("#winTitle:has-text('Available Bands List')").count()==1, pg.inner_text("body")[:600]
    shot(pg,"4_bands")
    # choose bands: R=band 25, G=band 15, B=band 5
    for n in (25,15,5): pg.locator(f".bl-item:has-text('Band {n} ')").first.click()
    assert "Band 25" in pg.inner_text("#blfR") and "Band 5 " in pg.inner_text("#blfB"), pg.inner_text("#blChans")
    pg.click("text=Load RGB"); pg.wait_for_timeout(2000)
    assert pg.locator("#imgBox img").count()==1, pg.inner_text("#main")[:400]
    # --- collect pixels one after another
    box=pg.locator("#imgBox").bounding_box()
    for fx,fy in ((0.2,0.3),(0.5,0.5),(0.8,0.7),(0.3,0.8)):
        pg.mouse.click(box["x"]+box["width"]*fx, box["y"]+box["height"]*fy); pg.wait_for_selector("#pxPopBox"); pg.press("#pxLabel","Enter"); pg.wait_for_timeout(500)   # small window: label, Add
    assert pg.locator(".coll-row").count()==4, pg.locator(".coll-row").count()
    assert pg.locator(".pxmark").count()==4
    pg.click("text=Compare angles"); pg.wait_for_timeout(300)
    shot(pg,"5_collect")
    # --- build library from collected
    pg.click("#imgSide >> text=Save to library… >> nth=0"); pg.fill("#cbName","Clicked spectra"); pg.fill("#cbCat","Scene/Pixels"); pg.select_option("#cbAcq","airborne")
    pg.click(".win >> text=Save to library"); pg.wait_for_timeout(1200)
    res=[x for x in requests.get(API+"/api/spectra/search",params={"limit":50}).json()["results"] if x["library"]=="Clicked spectra"]
    assert len(res)==4 and {x["acquisition_type"] for x in res}=={"airborne"} and {x["category"] for x in res}=={"Scene/Pixels"}, res
    # --- Spectral analyst against that library
    pg.hover("#menubar [data-menu='Spectral']"); pg.click(".mnu-item:not(.sub):has-text('Spectral Analyst')"); pg.wait_for_timeout(300)
    pg.select_option("#anLib", label=re.compile("Clicked spectra")) if False else pg.select_option("#anLib", index=1)
    pg.click("#anGo"); pg.wait_for_timeout(1800)
    assert "best matches" in pg.inner_text("#winTitle"), pg.inner_text("#winBody")[:300]
    shot(pg,"6_analyst"); pg.click(".win >> text=Close")
    # --- analyses from the Spectral menu
    def run_menu(path, wait=2500, shotn=None, expect=None):
        pg.hover("#menubar [data-menu='Spectral']")
        top=path[0]
        pg.hover(f".mnu-item:not(.sub):has-text('{top}')")
        for label in path[1:]:
            pg.click(f".mnu-item:not(.sub):has-text('{label}')")
        pg.wait_for_timeout(300)
        return
    pg.hover("#menubar [data-menu='Spectral']"); pg.hover(".mnu-item.sub:has-text('Mapping Methods')"); pg.click(".mnu-item:not(.sub):has-text('K-Means')"); pg.wait_for_timeout(300)
    pg.fill("#anK","4"); pg.click("#anGo"); pg.wait_for_timeout(3500)
    assert "K-means" in pg.inner_text("#winTitle"), pg.inner_text("#winBody")[:300]
    shot(pg,"7_kmeans")
    pg.click("text=Overlay on the image"); pg.wait_for_timeout(700)
    assert pg.locator("#imgBox img").count()==2
    shot(pg,"8_overlay")
    # SAM with collected spectra
    pg.hover("#menubar [data-menu='Spectral']"); pg.hover(".mnu-item.sub:has-text('Mapping Methods')"); pg.click(".mnu-item:not(.sub):has-text('Spectral Angle')"); pg.wait_for_timeout(300)
    pg.click("#anGo"); pg.wait_for_timeout(3000); assert "Spectral Angle Mapper" in pg.inner_text("#winTitle"), pg.inner_text("#winBody")[:300]; pg.click(".win >> text=Close")
    # Unmixing
    pg.hover("#menubar [data-menu='Spectral']"); pg.hover(".mnu-item.sub:has-text('Mapping Methods')"); pg.click(".mnu-item:not(.sub):has-text('Linear Spectral Unmixing')"); pg.wait_for_timeout(300)
    pg.click("#anGo"); pg.wait_for_timeout(4000); assert "unmixing" in pg.inner_text("#winTitle"), pg.inner_text("#winBody")[:300]
    shot(pg,"9_unmix"); pg.click(".win >> text=Close")
    # NDVI
    pg.hover("#menubar [data-menu='Spectral']"); pg.hover(".mnu-item.sub:has-text('Spectral Indices')"); pg.click(".mnu-item:not(.sub):has-text('NDVI')"); pg.wait_for_timeout(3000)
    assert "Band math" in pg.inner_text("#winTitle"), pg.inner_text("#winBody")[:300]
    shot(pg,"10_ndvi"); pg.click(".win >> text=Close")
    # PCA, anomaly, SNR
    for label,title in (("Principal Components","PCA"),("Anomaly","Anomaly"),("Signal-to-Noise","Signal-to-noise")):
        pg.hover("#menubar [data-menu='Spectral']")
        if label=="Principal Components":
            pg.hover(".mnu-item.sub:has-text('Transforms')")
        pg.click(f".mnu-item:not(.sub):has-text('{label}')"); pg.wait_for_timeout(300); pg.click("#anGo"); pg.wait_for_timeout(3500)
        assert title.lower() in pg.inner_text("#winTitle").lower(), (label, pg.inner_text("#winBody")[:300])
        if title=="Signal-to-noise": shot(pg,"11_snr")
        pg.click(".win >> text=Close")
    # Image view via logo goes home
    pg.click(".sidebar .logo"); pg.wait_for_timeout(500); assert pg.locator(".module-card").count()==3
    pg.click("text=Open Image >> nth=0") if False else None
    print("ERRORS:", errs)
    b.close()
sys.exit(1 if errs else 0)
