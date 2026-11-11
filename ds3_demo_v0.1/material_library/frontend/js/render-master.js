/* render-master.js — Master render: which page is drawn for the current view
   (classic script, loaded in order from index.html; all functions are global) */
/* =========================================================================
   MASTER RENDER
   ========================================================================= */
function renderMainOnly(){
  const main = document.getElementById("main");
  switch(state.view){
    case "dashboard": main.innerHTML = renderDashboard(); break;
    case "library-list": main.innerHTML = renderLibraryList(); break;
    case "library-materials": main.innerHTML = renderMaterialList(); break;
    case "library-spectrum": main.innerHTML = renderSpectrumViewer(); break;
    case "comparison": main.innerHTML = renderComparison(); break;
    case "library-map": main.innerHTML = renderLibraryMap(); break;
    case "image": destroyMap(); main.innerHTML = renderImagePage(); drawSpectrumChart(); if(I.mapOn) initMap(); else afterImageDraw(); break;
    case "image-local": main.innerHTML = renderImage(); if(imgState.tiff) renderTiffComposite(); else drawCanvas(); break;
    case "image-georef": main.innerHTML = renderGeorefPanel(); initGeorefMap(); drawGeoImageCanvas(); break;
    case "builder": case "lib-functions": main.innerHTML = renderLibFunctions(); break;
    case "analysis": main.innerHTML = renderAnalysis(); break;
    default: main.innerHTML = renderLibraryList();
  }
}
function renderRecent(){
  // Recent files UI removed from the sidebar; kept as a no-op so addRecent()
  // calls elsewhere don't need to be touched.
}
function render(){
  document.querySelectorAll(".nav-item").forEach(el=>{
    el.classList.toggle("active", el.dataset.nav === state.view ||
      (el.dataset.nav==="library-list" && ["library-list","library-materials","library-spectrum","comparison","library-map","builder","lib-functions"].includes(state.view)) ||
      (el.dataset.nav==="image" && ["image-georef","image-local"].includes(state.view)));
  });
  renderMainOnly();
  renderRecent();
}

/* BUILDER-START — ENVI-style Spectral Library Builder: pick grid, add inputs, resample, search, match, save */
const bld = { grid:null, gridSrc:"", items:[], rejected:[], refId:null, query:"", mask:true, divisor:1, libName:"Builder Library" };
const WATER_NM = [[1340,1460],[1790,1960]];
const toNm = wl => Math.max(...wl) > 50 ? wl.slice() : wl.map(w=>w*1000);
function bldSetGrid(nm, src){ nm=nm.slice().sort((a,b)=>a-b); bld.grid=nm; bld.gridSrc=src; bld.items=[]; bld.rejected=[]; bld.refId=null; renderMainOnly(); showToast(`Grid set: ${nm.length} bands, ${nm[0].toFixed(0)}–${nm[nm.length-1].toFixed(0)} nm`); }
function bldReject(name, why){ bld.rejected.push({name, why}); }
function bldParseText(text){ const w=[],v=[]; text.split(/\r?\n/).forEach(l=>{ const p=l.split(/[\s,;\t]+/).map(parseFloat).filter(Number.isFinite); if(p.length>=2){ w.push(p[0]); v.push(p[1]); } }); return {w,v}; }
function bldImgSpectra(pixels){
  const t=imgState.tiff; if(!t||!t.cube){ showToast("Load the full spectral cube in Image first"); return null; }
  if(!t.wavelengths){ showToast("Image has no wavelengths"); return null; }
  const nb=t.cube.length, sum=new Float64Array(nb); let n=0;
  for(const [x,y] of pixels){ const i=y*t.width+x; const nd=t.noDataValue; if(nd!==null&&t.cube[0][i]===nd) continue; for(let b=0;b<nb;b++) sum[b]+=t.cube[b][i]; n++; }
  return n? {mean:Array.from(sum,s=>s/n), n} : null;
}
/* BUILDER-END */
/* BACKEND-START — Option B: the page talks to the Python API (ds3 serve) and the database */
const bk = { base: location.protocol==="file:" ? "http://127.0.0.1:8000" : location.origin, ok:false, info:null, cubes:[], libs:[], q:"", hits:null, ref:null, matches:null, err:"", r0:0,c0:0,r1:10,c1:10, cubeId:null };
async function bkFetch(path, opts){
  const r = await fetch(bk.base+path, opts);
  if(!r.ok){ let d=""; try{ const j=await r.json(); d=typeof j.detail==="string"?j.detail:JSON.stringify(j.detail); }catch(e){} throw new Error(`${r.status} ${d||r.statusText}`); }
  return r.json();
}
async function bkConnect(){
  bk.base=(document.getElementById("bkBase")?.value||bk.base).replace(/\/$/,""); bk.err="";
  try{ bk.info=await bkFetch("/health"); bk.ok=true; await bkRefresh(); showToast(`Connected · ${bk.info.db} database`); }
  catch(e){ bk.ok=false; bk.err=`Cannot reach ${bk.base} — is "ds3 serve" running? (${e.message})`; renderMainOnly(); }
}
async function bkRefresh(){ try{ bk.cubes=await bkFetch("/cubes"); bk.libs=await bkFetch("/libraries"); }catch(e){ bk.err=e.message; } renderMainOnly(); }
async function bkUpload(e){
  const fs=[...e.target.files]; if(!fs.length) return; const fd=new FormData(); fs.forEach(f=>fd.append("files",f,f.name)); fd.append("source","ui-upload");
  showToast(`Uploading ${fs.length} file(s)…`);
  try{ const res=await bkFetch("/ingest/upload-many",{method:"POST",body:fd});
    res.forEach(r=>{ if(r.status!=="ingested") bldReject(r.path.split(/[\\/]/).pop(), `${r.status}: ${r.detail||""}`); });
    showToast(res.map(r=>r.status).join(", ")||"nothing ingested"); }
  catch(err){ bk.err=err.message; }
  e.target.value=""; await bkRefresh();
}
async function bkGridFromCube(id){
  try{ const c=await bkFetch(`/cubes/${id}`); bk.cubeId=id; bldSetGrid(c.wavelengths_nm, `database cube #${id} ${c.name}`); }catch(e){ showToast(e.message); }
}
function bkAddFrom(name, wl, vals, how, cube){
  const ok=bldAdd(name, wl, vals.map(v=>v===null?NaN:v), "image_pixel", `db cube #${cube.id} ${cube.name} ${how}`); renderMainOnly(); return ok;
}
async function bkAddRoi(id, mode){
  const cube=bk.cubes.find(c=>c.id===id); if(!bld.grid){ showToast("Choose a wavelength grid first (step 1)"); return; }
  let body;
  if(mode==="poly"){ const r=imgState.roi; if(!(r.closed&&r.points.length>=3)){ showToast("Draw and finish an ROI polygon in Image first"); return; } body={polygon:r.points.map(p=>[p.x,p.y])}; }
  else body={bbox_px:[bk.r0,bk.c0,bk.r1,bk.c1].map(x=>Math.round(+x))};
  try{ const r=await bkFetch(`/cubes/${id}/roi`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)});
    bkAddFrom(`ROI mean (${r.n_pixels} px) cube ${id} #${bld.items.length+1}`, r.wavelengths_nm, r.mean, mode==="poly"?"polygon ROI":`box ${body.bbox_px}`, cube);
    if(r.n_excluded_nodata) showToast(`${r.n_excluded_nodata} no-data pixels excluded`); }
  catch(e){ bldReject("ROI",e.message); renderMainOnly(); }
}
async function bkAddPixel(id){
  const cube=bk.cubes.find(c=>c.id===id), p=imgState.pixel; if(!bld.grid){ showToast("Choose a wavelength grid first (step 1)"); return; }
  if(!p){ showToast("Click a pixel in Image first (row/col are taken from it)"); return; }
  try{ const r=await bkFetch(`/cubes/${id}/spectrum?row=${p.y}&col=${p.x}`); bkAddFrom(`Pixel (${p.x},${p.y}) cube ${id}`, r.wavelengths_nm, r.values, `pixel row ${p.y} col ${p.x}`, cube); }
  catch(e){ bldReject("Pixel",e.message); renderMainOnly(); }
}
async function bkBuildServer(e){
  const fs=[...e.target.files]; if(!fs.length) return; const fd=new FormData(); fs.forEach(f=>fd.append("files",f,f.name));
  fd.append("name",(document.getElementById("bldLibName")?.value||bld.libName)); 
  if(bk.cubeId&&bld.gridSrc.startsWith("database cube")){ fd.append("grid","cube"); fd.append("cube_id",bk.cubeId); } else fd.append("grid","first");
  try{ const r=await bkFetch("/libraries/build/upload",{method:"POST",body:fd});
    r.rejected.forEach(x=>bldReject(x.input.split(/[\\/]/).pop(),x.reason)); showToast(`Server build: added ${r.added.length}, rejected ${r.rejected.length} (${r.n_bands} bands)`); }
  catch(err){ bldReject("server build",err.message); }
  e.target.value=""; await bkRefresh();
}
async function bkSearch(){
  try{ const r=await bkFetch(`/spectra/search?limit=25&text=${encodeURIComponent(bk.q)}`); bk.hits=r; bk.matches=null; bk.ref=null; }catch(e){ bk.err=e.message; } renderMainOnly();
}
async function bkMatch(id){
  try{ bk.ref=id; bk.matches=await bkFetch("/spectra/match",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({spectrum_id:id,top_k:8})}); }catch(e){ bk.err=e.message; bk.matches=null; } renderMainOnly();
}
/* BACKEND-END */

/* DBLAYER-START — everything below talks to the FastAPI + database backend. Nothing here is simulated. */
const API_BASE_URL = (new URLSearchParams(location.search).get("api")
  || (location.protocol.startsWith("http") ? location.origin : "http://127.0.0.1:8000")).replace(/\/$/, "");
function apiUrl(p){ return API_BASE_URL + p; }
function escapeHtml(v){ return String(v ?? "").replaceAll("&","&amp;").replaceAll("<","&lt;").replaceAll(">","&gt;").replaceAll('"',"&quot;").replaceAll("'","&#039;"); }
function unwrapList(p){ if(Array.isArray(p)) return p; for(const k of ["items","results","data"]) if(p && Array.isArray(p[k])) return p[k]; return []; }
async function apiGet(path){
  const r = await fetch(apiUrl(path), {headers:{"Accept":"application/json"}});
  if(!r.ok){ let d=`Request failed (${r.status})`; try{ const b=await r.json(); d=typeof b.detail==="string"?b.detail:JSON.stringify(b.detail||b); }catch(_){} throw new Error(d); }
  return r.json();
}
async function apiSend(method, path, body, isForm){
  const r = await fetch(apiUrl(path), { method, headers: isForm ? {} : {"Content-Type":"application/json","Accept":"application/json"},
    body: isForm ? body : (body===undefined ? undefined : JSON.stringify(body)) });
  let data=null; try{ data=await r.json(); }catch(_){}
  if(!r.ok) throw new Error((data && data.detail) ? (typeof data.detail==="string" ? data.detail : JSON.stringify(data.detail)) : `Request failed (${r.status})`);
  return data;
}

/* chart that fits any unit / wavelength range (the original chart assumes reflectance 0-1 over 0.4-2.5 um) */
function dsChart(seriesList, opts={}){
  const valid=(seriesList||[]).filter(s=>Array.isArray(s.data)&&s.data.length);
  if(!valid.length) return `<div class="placeholder-card">No spectral values available.</div>`;
  const w=opts.width||560, h=opts.height||220, padL=58, padR=10, padT=10, padB=40;
  const cap=t=>t?String(t).charAt(0).toUpperCase()+String(t).slice(1):"";
  const xLab=opts.xLabel||"Wavelength (nm)", yLab=opts.yLabel||(opts.unit?cap(opts.unit):"Reflectance");
  const aX=valid.flatMap(s=>s.data.map(d=>+d.x)).filter(Number.isFinite), aY=valid.flatMap(s=>s.data.map(d=>+d.y)).filter(Number.isFinite);
  const mm=a=>{ let lo=Infinity,hi=-Infinity; for(const v of a){ if(v<lo) lo=v; if(v>hi) hi=v; } return [lo,hi]; };   // loop, not Math.min(...a): spreading >100k values overflows the call stack
  const [xMin,xMax]=mm(aX); let [yMin,yMax]=mm(aY);
  if(yMin>=0 && yMax<=1.2){ yMin=Math.min(0,yMin); yMax=Math.max(1,yMax); } else { const pad=(yMax-yMin)*0.06||1; yMin-=pad; yMax+=pad; }
  const xs=x=>padL+((x-xMin)/(xMax-xMin||1))*(w-padL-padR), ys=y=>(h-padB)-((y-yMin)/(yMax-yMin||1))*(h-padT-padB);
  const yd = (yMax-yMin)>=50?0:((yMax-yMin)>=5?1:2), xd = xMax<10?2:0;
  let grid="";
  for(let i=0;i<=4;i++){ const gy=padT+i*(h-padT-padB)/4; grid+=`<line x1="${padL}" y1="${gy}" x2="${w-padR}" y2="${gy}" class="chart-grid"/><text x="${padL-5}" y="${gy+3}" text-anchor="end" class="chart-tick">${(yMax-i*(yMax-yMin)/4).toFixed(yd)}</text>`; }
  for(let i=0;i<=5;i++){ const gx=padL+i*(w-padL-padR)/5; grid+=`<text x="${gx}" y="${h-padB+13}" text-anchor="middle" class="chart-tick">${(xMin+i*(xMax-xMin)/5).toFixed(xd)}</text>`; }
  let lines=""; valid.forEach(s=>{ lines+=`<polyline points="${s.data.filter(d=>Number.isFinite(+d.x)&&Number.isFinite(+d.y)).map(d=>`${xs(+d.x)},${ys(+d.y)}`).join(" ")}" fill="none" stroke="${s.color||"#34d399"}" stroke-width="${s.width||2}"/>`; });
  return `<svg viewBox="0 0 ${w} ${h}" width="100%" height="${h}" role="img" aria-label="Spectrum chart"><text x="${padL+(w-padL-padR)/2}" y="${h-4}" text-anchor="middle" class="chart-title">${escapeHtml(xLab)}</text><text transform="translate(11 ${padT+(h-padT-padB)/2}) rotate(-90)" text-anchor="middle" class="chart-title">${escapeHtml(yLab)}</text><line x1="${padL}" y1="${padT}" x2="${padL}" y2="${h-padB}" class="chart-axis"/><line x1="${padL}" y1="${h-padB}" x2="${w-padR}" y2="${h-padB}" class="chart-axis"/>${grid}${lines}</svg>`;
}
const DS_COLORS=["#34d399","#60a5fa","#f472b6","#fbbf24","#a78bfa","#fb923c","#38bdf8","#f87171"];
const LIB_COLORS=["emerald","sky","amber","purple","orange","blue","violet"];
