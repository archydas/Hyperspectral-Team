import re,sys
from playwright.sync_api import sync_playwright
API="http://127.0.0.1:8765"; errs=[]
with sync_playwright() as p:
    b=p.chromium.launch(); pg=b.new_page(viewport={"width":1440,"height":900})
    pg.on("pageerror",lambda e:errs.append(str(e)))
    pg.route(re.compile(r"https://(cdnjs|cdn\.jsdelivr)\..*"),lambda r:r.abort())
    pg.goto(API+"/app/"); pg.wait_for_timeout(1500)
    pg.hover("#menubar [data-menu='View']"); pg.click(".mnu-item:not(.sub):has-text('Compare Spectra')"); pg.wait_for_timeout(1200)
    assert "Compare spectra" in pg.inner_text("#main")
    n=pg.locator("#cmpList .lf-item").count(); print("rows",n); assert n>2
    for i in range(3): pg.locator("#cmpList .lf-item").nth(i).click(); pg.wait_for_timeout(500)
    opts=pg.locator("select").first.locator("option").all_inner_texts(); print(opts)
    if len(opts)>2:
        pg.locator("#main select").first.select_option(index=2); pg.wait_for_timeout(800)
        pg.locator("#cmpList .lf-item").nth(0).click(); pg.wait_for_timeout(500)
    assert pg.locator("#cmpSel .coll-row").count()==4, pg.locator("#cmpSel .coll-row").count()
    assert pg.locator("#cmpChart svg").count()>=1
    pg.screenshot(path="/tmp/cmp.png")
    print("ERR",errs); b.close()
sys.exit(1 if errs else 0)
