/* ui-shell.js — ENVI-style shell: windows, menu bar, Open image, collected spectra
   (classic script, loaded in order from index.html; all functions are global) */
/* =========================================================================
   UI v2 — ENVI-style shell: menu bar, File ▸ Open image + Available Bands List, spectra collected by clicking pixels,
   Spectral menu (libraries, analyst, mapping methods, transforms, indices, anomaly, SNR) running on the backend.
   ========================================================================= */

/* ---------- ENVI-style window (replaces the old blurred modal for the new dialogs) ---------- */
function win(title, body, o={}){
  let h=document.getElementById("winHost"); if(!h){ h=document.createElement("div"); h.id="winHost"; document.body.appendChild(h); }
  try{ if(document.activeElement&&h.contains(document.activeElement)) document.activeElement.blur(); }catch(_){}
  if(body===null){ h.innerHTML=""; if(typeof GR!=="undefined"&&GR.open) georefMark(false); return; }
  h.innerHTML=`<div class="win-back" onmousedown="if(event.target===this)winClose()"><div class="win" id="winBox" style="width:${o.width||520}px">
    <div class="win-title" id="winTitle"><span>${escapeHtml(title)}</span><button onclick="winClose()" title="Close">&#10005;</button></div>
    <div class="win-body" id="winBody">${body}</div></div></div>`;
  const t=document.getElementById("winTitle"), b=document.getElementById("winBox"); let dx=0,dy=0,sx,sy,drag=false;
  t.onmousedown=e=>{ if(e.target.tagName==="BUTTON") return; drag=true; sx=e.clientX-dx; sy=e.clientY-dy; e.preventDefault(); };
  window.onmousemove=e=>{ if(!drag) return; dx=e.clientX-sx; dy=e.clientY-sy; b.style.transform=`translate(${dx}px,${dy}px)`; };
  window.onmouseup=()=>{ drag=false; };
}
function winClose(){ win("",null); }
function winBody(html){ const b=document.getElementById("winBody"); if(b) b.innerHTML=html; }
function needImage(){
  if(I.active&&I.preview) return true;
  showToast("Open an image first: File ▸ Open Image File…"); return false;
}

/* ---------- menu bar ---------- */
const MENU_ACT={}; let _mi=0;
function mi(label, fn, o={}){ const id="m"+(++_mi); MENU_ACT[id]=fn; return Object.assign({label,id},o); }
const sep=()=>({sep:true}), head=t=>({head:t});
function menuRun(id){
  const bar=document.getElementById("menubar"); bar.classList.add("shut");
  try{ MENU_ACT[id](); }catch(e){ showToast(e.message); }
}
function menuHtml(items){
  return items.map(m=>{
    if(m.sep) return `<div class="mnu-sep"></div>`;
    if(m.head) return `<div class="mnu-head">${escapeHtml(m.head)}</div>`;
    if(m.sub) return `<div class="mnu-item sub ${m.disabled?"off":""}">${escapeHtml(m.label)} <i>&#9656;</i><div class="mnu-pop">${menuHtml(m.sub)}</div></div>`;
    return `<div class="mnu-item ${m.disabled?"off":""}" onclick="event.stopPropagation();${m.disabled?"":`menuRun('${m.id}')`}">${escapeHtml(m.label)}${m.key?`<span class="key">${m.key}</span>`:`<span></span>`}</div>`;
  }).join("");
}
function lfOpen(tab){ return ()=>openLibFunctions(tab); }
function indexItem(label, expr){ return mi(label, ()=>anOpen("bandmath",{expression:expr,run:true})); }
function menuModel(){
  const imgs=I.list.length?I.list.map(i=>mi(`${i.name}  (${i.cols}×${i.rows}×${i.bands})`,()=>showBandsList(i.id))):[{label:"(no images opened yet)",disabled:true,id:"x"}];
  return [
    {label:"File", items:[
      mi("Open Image File…",openImageDialog,{key:"Ctrl+O"}),
      {label:"Open Previously Opened Image",sub:imgs},
      mi("Open Spectral Library File…",importSpectraDialog),
      sep(),
      mi("Available Bands List…",()=>showBandsList()),
      mi("Download Image Preview (PNG)",()=>{ if(needImage()) downloadImage(); }),
      mi("Remove Image from List",()=>{ if(needImage()){ state.view="image"; render(); deleteImage(); } }),
      sep(),
      mi("Home",()=>setView("dashboard"))]},
    {label:"View", items:[
      mi("Home (3 modules)",()=>setView("dashboard")),
      mi("Library",()=>setView("library-list")),
      mi("Image",()=>setView("image")),
      mi("Compare Spectra",()=>cmpOpen()),
      mi("Analysis (AI / ML models)",()=>setView("analysis")),
      sep(),
      mi("Zoom In",()=>{ if(needImage()&&state.view==="image") setZoom(1.25); }),
      mi("Zoom Out",()=>{ if(needImage()&&state.view==="image") setZoom(1/1.25); }),
      mi("Fit Image to Window",()=>{ if(needImage()&&state.view==="image") setZoom("fit"); }),
      mi("Map Background On/Off",()=>{ if(needImage()){ state.view="image"; render(); toggleMap(); } })]},
    {label:"Tools", items:[
      mi("Pick Pixels (click, label, Add)",()=>{ if(needImage()){ state.view="image"; render(); showToast("Click a pixel: a small window shows its spectrum. Label it and press Add."); } }),
      mi("Draw ROI Polygon",()=>{ if(needImage()){ state.view="image"; render(); startRoi(); } }),
      mi("Delete All Collected Spectra",()=>{ collClear(); }),
      sep(),
      mi("Georeference…",()=>{ if(needImage()){ state.view="image"; render(); georefOpen(); } }),
      mi("Add Current ROI to Object Samples…",()=>{ if(needImage()){ if(I.roi.closed&&I.spec&&I.spec.kind==="roi") popOpen("roi"); else showToast("Draw an ROI first: click points, double-click to finish"); } })]},
    {label:"Spectral", items:[
      {label:"Spectral Libraries",sub:[
        mi("Spectral Library Builder",lfOpen("builder")),
        mi("Spectral Library Viewer",lfOpen("viewer")),
        mi("Spectral Library Resampling",lfOpen("resample"))]},
      mi("Spectral Analyst…",()=>anOpen("analyst")),
      {label:"Mapping Methods",sub:[
        mi("Spectral Angle Mapper…",()=>anOpen("sam")),
        mi("Linear Spectral Unmixing…",()=>anOpen("unmix")),
        mi("K-Means Clustering…",()=>anOpen("kmeans"))]},
      {label:"Transforms",sub:[
        mi("Principal Components (PCA)…",()=>anOpen("pca")),
        mi("Band Math…",()=>anOpen("bandmath"))]},
      {label:"Spectral Indices",sub:[
        indexItem("NDVI — vegetation greenness","(w800 - w670) / (w800 + w670)"),
        indexItem("NDRE — red edge","(w790 - w720) / (w790 + w720)"),
        indexItem("SAVI — soil-adjusted vegetation","1.5 * (w800 - w670) / (w800 + w670 + 0.5)"),
        indexItem("EVI — enhanced vegetation","2.5 * (w800 - w670) / (w800 + 6 * w670 - 7.5 * w475 + 1)"),
        indexItem("NDWI — canopy water","(w860 - w1240) / (w860 + w1240)"),
        indexItem("NDBI — built-up","(w1650 - w830) / (w1650 + w830)"),
        indexItem("PRI — photochemical reflectance","(w531 - w570) / (w531 + w570)")]},
      mi("Anomaly Detection (RX)…",()=>anOpen("anomaly")),
      mi("Signal-to-Noise Ratio…",()=>anOpen("snr")),
      sep(),
      mi("Data Aggregation (ROI)…",()=>aggGo())]},
    {label:"Help", items:[
      mi("Supported Formats",helpFormats),
      mi("Using the Image Module",helpImage),
      sep(),
      mi("About DS3",()=>win("About DS3",`<p style="margin:0 0 6px"><b>DS3 — Deep Spectral Signature System</b></p><p class="note" style="margin:0">Spectral library, hyperspectral image module and data aggregation, backed by a real database (SQLite or PostgreSQL).</p><p class="note">Database: ${escapeHtml(dbKind())}</p>`,{width:440}))]}
  ];
}
function renderMenubar(){
  const bar=document.getElementById("menubar"); if(!bar) return;
  if(bar.firstChild&&bar.matches(":hover")){ clearTimeout(renderMenubar._t); renderMenubar._t=setTimeout(renderMenubar,700); return; }   // never rebuild a menu under the mouse
  bar.innerHTML=menuModel().map(m=>`<div class="mnu" data-menu="${m.label}">${m.label}<div class="mnu-pop">${menuHtml(m.items)}</div></div>`).join("")+`<div class="grow"></div><div class="dbinfo" id="dbInfo">${escapeHtml(dbKind())}</div>`;
}
document.addEventListener("mouseover",e=>{ const b=document.getElementById("menubar"); if(b&&b.classList.contains("shut")&&!b.contains(e.target)) b.classList.remove("shut"); });
document.addEventListener("keydown",e=>{ if((e.ctrlKey||e.metaKey)&&e.key.toLowerCase()==="o"){ e.preventDefault(); openImageDialog(); } if(e.key==="Escape") winClose(); });

function helpFormats(){
  win("Supported formats",`<div class="box" style="font-size:12.5px;line-height:1.7"><b>Images</b>: ENVI (.hdr + data file, e.g. AVIRIS-NG, any interleave), GeoTIFF / EnMAP (.tif, .zip), MATLAB (.mat, e.g. PaviaU), HDF5, NetCDF.<br>
    <b>Spectra</b>: CSV / TXT tables, Excel (.xlsx), JSON, Parquet, ENVI spectral library (.sli/.hdr), ASD, SIG, SED, USGS and ECOSTRESS text files.<br>
    <b>Export</b>: CSV, JSON, ENVI .sli (+ standard deviation).</div><p class="note">For an ENVI image choose the <b>.hdr</b> file (the program finds the data file next to it).</p>`,{width:560});
}
function helpImage(){
  win("Using the image module",`<div class="box" style="font-size:12.5px;line-height:1.7"><b>1.</b> File ▸ Open Image File, choose the .hdr (or .tif, .mat, …).<br><b>2.</b> In the Available Bands List pick Gray Scale or RGB, click a band for each colour, then <b>Load RGB</b>.<br>
    <b>3.</b> Click pixels one after another: a small window shows the spectrum: give it a label and press <b>Add</b> to list it under Collected spectra. Draw an ROI (double-click or right-click to finish) for a mean spectrum; Add puts it under Collected object samples. ROI aggregation is below the image.<br><b>4.</b> Compare the spectra, check their spectral angles, or <b>Build library</b> from them.<br>
    <b>5.</b> Spectral menu: SAM, unmixing, k-means, PCA, band math / indices, anomaly, SNR.</div>`,{width:560});
}

/* ---------- opening images: File > Open Image File > Available Bands List > Load > Image module ---------- */
const OPENED={ paths:null, files:null };            // what was chosen last: lets "Open again" retry with a wavelength range
async function openImageDialog(wl){
  try{
    if(window.pywebview&&window.pywebview.api&&window.pywebview.api.pick_files){
      const paths=await window.pywebview.api.pick_files(false);
      if(!paths||!paths.length) return;
      showToast("Reading the file(s)…");
      OPENED.paths=paths; OPENED.files=null;
      const rep=await apiSend("POST","/api/import/path",{paths,wavelengths:wl&&String(wl).trim()?String(wl).trim():null});
      await afterOpen(rep);
    }else{
      document.getElementById("fileOpenInput").click();
    }
  }catch(e){ showToast(e.message); I.err=e.message; if(state.view==="image") refreshAll(); }
}
async function openFromInput(input,wl){
  const files=input?Array.from(input.files||[]):OPENED.files||[]; if(input) input.value=""; if(!files.length) return;
  OPENED.files=files; OPENED.paths=null;
  const fd=new FormData(); files.forEach(f=>fd.append("files",f));
  if(wl&&String(wl).trim()) fd.append("wavelengths",String(wl).trim());
  showToast(`Reading ${files.length} file(s)…`);
  try{ await afterOpen(await apiSend("POST","/api/import",fd,true)); }catch(e){ showToast(e.message); }
}
async function afterOpen(rep){
  await loadLibraries();
  I.list=await apiGet("/api/images/");
  let got=(rep.files||[]).find(f=>f.kind==="image"&&f.id);
  const bad=(rep.files||[]).filter(f=>f.status==="error");
  if(!got&&rep.duplicates){ const stem=((rep.files||[]).find(f=>f.status==="duplicate")||{}).file||""; const m=I.list.find(i=>stem.startsWith(i.name)||i.name.startsWith(stem.replace(/\.[^.]+$/,""))); if(m) got={id:m.id}; }
  renderMenubar();
  if(got){ await showBandsList(got.id); return; }
  if(rep.imported){ showToast(`${rep.imported} spectra were added to the library "${rep.library}" (no image in these files).`); return; }
  win("Could not open",`<div class="err" style="margin:0">${bad.length?bad.map(f=>escapeHtml(f.file+": "+f.detail)).join("<br>"):"No hyperspectral image was found in the chosen file(s). For ENVI, choose the .hdr file."}</div>
    <p class="note" style="margin-top:10px">If the file has no wavelengths, give the range of the sensor (for example 430-860) and open it again.</p>
    <div style="display:flex;gap:6px"><input id="wlRetry" type="text" placeholder="wavelengths, e.g. 430-860" style="flex:1" onkeydown="if(event.key==='Enter')openAgain()"><button class="btn emerald btn-sm" onclick="openAgain()">Open again</button></div>`,{width:520});
}
async function importSpectraDialog(){
  try{
    if(window.pywebview&&window.pywebview.api&&window.pywebview.api.pick_files){
      const paths=await window.pywebview.api.pick_files(false); if(!paths||!paths.length) return;
      const rep=await apiSend("POST","/api/import/path",{paths}); await loadLibraries(); renderMenubar();
      showToast(`${rep.imported} spectra imported into "${rep.library}"`+(rep.errors?` · ${rep.errors} file(s) failed`:""));
    }else document.getElementById("fileSpecInput").click();
  }catch(e){ showToast(e.message); }
}
async function specFromInput(input){
  const files=Array.from(input.files||[]); input.value=""; if(!files.length) return;
  const fd=new FormData(); files.forEach(f=>fd.append("files",f));
  try{ const rep=await apiSend("POST","/api/import",fd,true); await loadLibraries(); renderMenubar();
    showToast(`${rep.imported} spectra imported into "${rep.library}"`+(rep.errors?` · ${rep.errors} file(s) failed`:"")); if(state.view==="library-list") render(); }
  catch(e){ showToast(e.message); }
}

/* Available Bands List (ENVI): list of bands, Gray Scale / RGB, R-G-B targets, Dims, Load */
const BL={ id:null, info:null, wl:[], mode:"rgb", cur:"R", sel:{R:0,G:0,B:0} };
function defaultRgb(wl){
  const n=wl.length, near=t=>{ let b=0; wl.forEach((w,i)=>{ if(Math.abs(w-t)<Math.abs(wl[b]-t)) b=i; }); return b; };
  if(wl.length&&Math.min(...wl)<=480&&Math.max(...wl)>=660){ const r=near(650),g=near(550),b=near(470); if(new Set([r,g,b]).size===3) return [r,g,b]; }
  return [Math.floor(n*.75),Math.floor(n*.5),Math.floor(n*.25)].map(x=>Math.min(n-1,Math.max(0,x)));
}
const blName=i=>`Band ${i+1}${BL.wl[i]!==undefined?` (${(+BL.wl[i]).toFixed(2)} nm)`:""}`;
async function showBandsList(id){
  id=id||(I.active&&I.active.id);
  if(!id){ if(I.list.length) id=I.list[0].id; else { openImageDialog(); return; } }
  try{
    const info=I.list.find(i=>i.id===id)||await apiGet(`/api/images/${id}`);
    const bd=await apiGet(`/api/images/${id}/bands`);
    Object.assign(BL,{id,info,wl:bd.wavelengths,mode:"rgb",cur:"R"});
    const d=(I.active&&I.active.id===id&&I.bands)?I.bands.map(b=>b-1):defaultRgb(BL.wl);
    BL.sel={R:d[0],G:d[1],B:d[2]};
    win("Available Bands List",blHtml(),{width:430});
    setTimeout(()=>{ const el=document.querySelector(".bl-item.sel"); if(el) el.scrollIntoView({block:"center"}); },30);
  }catch(e){ showToast(e.message); }
}
function blHtml(){
  const info=BL.info, n=BL.wl.length, rgb=BL.mode==="rgb";
  const tag=i=>["R","G","B"].filter(c=>(rgb||c==="R")&&BL.sel[c]===i).join("");
  const items=BL.wl.map((w,i)=>`<div class="bl-item ${tag(i)?"sel":""}" data-i="${i}" onclick="blPick(${i})"><span class="sq ${tag(i)?"on":""}"></span>${blName(i)}<span class="tag">${tag(i)}</span></div>`).join("");
  const chan=c=>`<div class="bl-ch"><label style="width:42px"><input type="radio" name="blc" ${BL.cur===c?"checked":""} ${!rgb&&c!=="R"?"disabled":""} onclick="BL.cur='${c}'"> ${rgb?c:"Gray"}</label><div class="fld" id="blf${c}">${blName(BL.sel[c])} : ${escapeHtml(info.name)}</div></div>`;
  return `<div class="bl-list" id="blList"><div class="bl-head">&#9662; ${escapeHtml(info.name)}</div>${items}</div>
    <div class="bl-box"><label style="margin-right:18px"><input type="radio" name="blm" ${!rgb?"checked":""} onclick="blMode('gray')"> Gray Scale</label><label><input type="radio" name="blm" ${rgb?"checked":""} onclick="blMode('rgb')"> RGB Color</label></div>
    <div class="bl-box" id="blChans">${rgb?chan("R")+chan("G")+chan("B"):chan("R")}</div>
    <div class="bl-dims">Dims ${info.cols} x ${info.rows} x ${n} · ${escapeHtml(info.unit||"")}${info.format?` · ${escapeHtml(info.format)}`:""}${info.assumed_wavelengths?` · <span style="color:#b45309">wavelengths assumed</span>`:""}</div>
    <div class="row" style="justify-content:space-between"><span><button class="btn blue btn-sm" onclick="blLoad()">${rgb?"Load RGB":"Load Gray"}</button> <span class="note">Display #1</span></span><button class="btn ghost btn-sm" onclick="winClose()">Cancel</button></div>`;
}
function blMode(m){ BL.mode=m; BL.cur="R"; winBody(blHtml()); const el=document.querySelector(".bl-item.sel"); if(el) el.scrollIntoView({block:"center"}); }
function blPick(i){
  BL.sel[BL.cur]=i;
  if(BL.mode==="rgb"){ BL.cur=BL.cur==="R"?"G":BL.cur==="G"?"B":"B"; }
  const sc=document.getElementById("blList").scrollTop; winBody(blHtml()); document.getElementById("blList").scrollTop=sc;
}
async function blLoad(){
  const b=BL.mode==="rgb"?[BL.sel.R,BL.sel.G,BL.sel.B]:[BL.sel.R,BL.sel.R,BL.sel.R];
  const id=BL.id; winClose(); await showImage(id,b.map(x=>x+1));
}
async function showImage(id,bands){
  state.view="image"; render();
  try{ I.list=await apiGet("/api/images/"); await openImage(id,bands); }catch(e){ I.err=e.message; refreshAll(); }
  renderMenubar();
}

/* ---------- spectra collected by clicking pixels ---------- */
const COLL_COLORS=["#059669","#2563eb","#db2777","#d97706","#7c3aed","#ea580c","#0891b2","#dc2626","#4d7c0f","#be185d","#0f766e","#6d28d9"];
I.coll=[]; I.collect=true; I.collN=0; I.angles=false;
function collClear(){ I.coll=[]; I.collN=0; I.angles=false; if(state.view==="image"){ refreshView(); refreshSide(); } }
function collRemove(i){ I.coll.splice(i,1); refreshView(); refreshSide(); }
function collShow(i,v){ I.coll[i].show=v; drawSpectrumChart(); refreshView(); }
function angleDeg(a,b){
  let ab=0,aa=0,bb=0,n=0; for(let i=0;i<a.length;i++){ const x=a[i],y=b[i]; if(x==null||y==null) continue; ab+=x*y; aa+=x*x; bb+=y*y; n++; }
  if(n<3||!aa||!bb) return null; return Math.acos(Math.max(-1,Math.min(1,ab/Math.sqrt(aa*bb))))*180/Math.PI;
}
function collCsv(){
  const c=I.coll; if(!c.length) return; const wl=c[0].wavelengths;
  csvDownload(`${(I.active&&I.active.name)||"image"}_collected_spectra.csv`,["wavelength_nm,"+c.map(s=>`"${s.name.replace(/"/g,'""')}"`).join(","),...wl.map((w,k)=>[w,...c.map(s=>s.values[k]==null?"":s.values[k])].join(","))]);
}

/* the plot shows every collected spectrum (colour-coded), the current one, and ticked library spectra */

async function openAgain(){
  const wl=((document.getElementById("wlRetry")||{}).value||"").trim(); if(!wl) return; winClose();
  if(OPENED.paths) await openImageDialogWith(OPENED.paths,wl); else await openFromInput(null,wl);
}
async function openImageDialogWith(paths,wl){
  try{ showToast("Reading the file(s)…"); await afterOpen(await apiSend("POST","/api/import/path",{paths,wavelengths:wl})); }catch(e){ showToast(e.message); }
}
