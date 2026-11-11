/* database.js — Backend (database) layer: libraries, materials, search, upload, save
   (classic script, loaded in order from index.html; all functions are global) */
/* ---------------- state ---------------- */
const DB = { stats:null, startup:null, hits:null, report:null, err:null, target:null, loaded:false };
function dbKind(){ const d=(DB.stats&&DB.stats.database)||""; return d.startsWith("postgres")?"PostgreSQL":(d.startsWith("sqlite")?"SQLite (local file)":(d.split(":")[0]||"database")); }
function updateStatusbar(){
  { const di=document.getElementById("dbInfo"); if(di) di.textContent=dbKind(); }
  const sb=document.getElementById("statusbar"), sd=document.getElementById("sideStatus");
  if(DB.stats){
    if(sb) sb.textContent=`${DB.stats.libraries} libraries · ${DB.stats.spectra} spectra`;
    if(sd) sd.innerHTML=`<span class="d"></span> Ready`;
  } else {
    if(sb) sb.textContent=`Cannot reach the backend at ${API_BASE_URL} — start it with: uvicorn app.main:app`;
    if(sd) sd.innerHTML=`<span class="d" style="background:#ef4444"></span> Backend not running`;
  }
  bk.ok = !!DB.stats;
}

/* ---------------- libraries / materials from the database ---------------- */
async function loadLibraries(){
  try{
    const libs=unwrapList(await apiGet("/api/libraries/"));
    const keep=state.libraries.filter(l=>l.user&&!l.db);
    state.libraries=libs.map((l,i)=>({id:String(l.id),name:l.name,main:!!(l.metadata&&l.metadata.main),type:(l.metadata&&l.metadata.type)||"spectral",source:(l.metadata&&l.metadata.main)?"Main library":((l.metadata&&l.metadata.type)==="object"?"Object library":"Pixel library"),color:LIB_COLORS[i%LIB_COLORS.length],
      count:l.material_count,spectra:l.spectrum_count,db:true,loaded:false,materials:[]})).concat(keep);
    DB.err=null;
  }catch(e){ DB.err=e.message; }
  try{ DB.stats=await apiGet("/api/stats"); }catch(_){ DB.stats=null; }
  try{ DB.startup=await apiGet("/api/startup_import"); }catch(_){}
  DB.loaded=true; updateStatusbar();
  if(DB.startup&&DB.startup.running){ clearTimeout(window._suT); window._suT=setTimeout(loadLibraries,2500); }   // main libraries are still being read
  const lib=state.libraries.find(l=>l.id===state.activeLibId);
  if(state.view==="library-materials" && lib && lib.db) await dbLoadMaterials(lib); else renderMainOnly();
}
async function dbLoadMaterials(lib){
  lib.loading=true; lib.err=null; renderMainOnly();
  try{
    const ms=unwrapList(await apiGet(`/api/libraries/${lib.id}/materials`));
    lib.materials=ms.map(m=>({id:String(m.id),name:m.name,type:m.category||"Uncategorised",db:true,description:m.description,meta:m.metadata,
      latitude:null,longitude:null,locationName:null,resolution:""}));
    if(lib.type==="object") lib.objects=(await apiGet(`/api/libraries/${lib.id}/objects`)).objects;      // ROI samples with their patch file names
    lib.loaded=true;
  }catch(e){ lib.err=e.message; }
  lib.loading=false; renderMainOnly();
}
function libTypes(lib){ return [...new Set(lib.materials.map(m=>m.type))].sort(); }
function dbCountText(lib){ return lib.db ? `${lib.count} materials · ${lib.spectra} spectra` : `${lib.materials.length} materials · this browser only`; }

function goToLibrary(libId){
  state.activeLibId=libId; state.activeMatId=null; state.view="library-materials";
  state.matQuery=""; state.showAddMaterial=false; state.matTypeFilter=""; state.importPreview=null;
  render();
  const lib=state.libraries.find(l=>l.id===libId);
  if(lib&&lib.db&&!lib.loaded) dbLoadMaterials(lib);
}
async function goToMaterial(libId,matId){
  state.activeLibId=libId; state.activeMatId=matId; state.view="library-spectrum"; state.spectraExtracted++;
  let lib=state.libraries.find(l=>l.id===libId);
  if(lib&&lib.db&&!lib.loaded) await dbLoadMaterials(lib);
  const m=lib&&lib.materials.find(x=>x.id===matId);
  if(m) addRecent({name:m.name,type:"Spectrum"});
  render();
  if(m&&m.db&&!m.spectraLoaded) await dbLoadSpectra(m);
}
async function dbLoadSpectra(m){
  m.loadingSp=true; m.err=null; renderMainOnly();
  try{
    const samples=unwrapList(await apiGet(`/api/materials/${m.id}/samples`));
    m.nSamples=samples.length; const out=[];
    for(const s of samples.slice(0,12)){
      for(const sp of unwrapList(await apiGet(`/api/samples/${s.id}/spectra`))) out.push({...sp,sample:s.name,sample_meta:s.metadata});
    }
    m.dbSpectra=out; m.spectraLoaded=true;
    const rows=dbRows(out[0]); m.customData=null;           // keep local-only fields clean
    m.firstRows=rows;
  }catch(e){ m.err=e.message; }
  m.loadingSp=false; renderMainOnly();
}
function dbRows(sp){
  if(!sp) return [];
  return sp.values.map((v,i)=>({wavelength:Number(sp.wavelengths[i]),reflectance:v===null?null:Number(v)})).filter(r=>r.reflectance!==null&&Number.isFinite(r.reflectance));
}
/* library materials turned into the {wavelength(um), reflectance} rows the browser-side classifier expects (real spectra only) */
async function dbHydrate(lib, maxN=300){
  if(!lib||!lib.db) return;
  if(!lib.loaded) await dbLoadMaterials(lib);
  const r=await apiGet(`/api/spectra/search?library_id=${lib.id}&unit=reflectance&limit=${maxN}`);
  const byMat=new Map(); r.results.forEach(h=>{ if(!byMat.has(h.material_id)) byMat.set(h.material_id,h.spectrum_id); });
  await Promise.all(lib.materials.map(async m=>{
    if(m.customData||!byMat.has(m.id)) return;
    try{ const sp=await apiGet(`/api/spectra/${byMat.get(m.id)}`); const rows=dbRows(sp).map(x=>({wavelength:x.wavelength/1000,reflectance:x.reflectance}));
      if(rows.length>=3) m.customData=rows; }catch(_){}
  }));
}

/* ---------------- library list / search / upload ---------------- */
function libCardHtml(l){
  return `<div class="lib-card" onclick="goToLibrary('${l.id}')"><span class="lib-chip" style="background:var(--${l.color}-d,#e5e7eb); color:var(--${l.color}-l,#374151);">${escapeHtml(l.source)}</span>
    <h4>${escapeHtml(l.name)}</h4><div class="meta">${l.db?`${l.count} materials · ${l.spectra} spectra`:`${l.materials.length} materials`}</div></div>`;
}
function buildLibraryCards(){
  const q=state.libQuery.toLowerCase(), list=state.libraries.filter(l=>l.name.toLowerCase().includes(q));
  if(!state.libraries.length) return `<div class="placeholder-card" style="grid-column:1/-1">${DB.stats?"No libraries yet.":"The backend is not running — see the red line at the bottom."}</div>`;
  if(!list.length) return '<p style="color:var(--muted)">No libraries match your search.</p>';
  const main=list.filter(l=>l.main), obj=list.filter(l=>!l.main&&l.type==="object"), pix=list.filter(l=>!l.main&&l.type!=="object");
  const sec=(t,sub,arr)=>arr.length?`<div style="grid-column:1/-1;margin-top:6px"><b style="font-size:13px">${t}</b> <span class="note" style="margin:0 0 0 6px">${sub}</span></div>${arr.map(libCardHtml).join("")}`:"";
  return sec("Main libraries",`${main.length} · loaded from the libraries folder`,main)
    +sec("Pixel-based libraries",`${pix.length} · one spectrum per pixel or measurement: built with Spectral Library Builder, collected on an image, or imported`,pix)
    +sec("Object-based libraries (ROI)",`${obj.length} · one mean spectrum per ROI; each keeps its image patch file (<class>.tif)`,obj);
}
function mainLibPanel(){
  const su=DB.startup; if(!su||!su.folder) return "";
  const nMain=state.libraries.filter(l=>l.main).length;
  const state_=su.running?`<span style="color:#b45309">&#8987; Reading the libraries folder\u2026 large files (e.g. ECOSTRESS) can take a minute.</span>`
    :(nMain?`${nMain} main librar${nMain===1?"y":"ies"} loaded.`:`<b style="color:#b45309">No main library files found.</b> Copy your library files (.txt / .csv / \u2026) into the folder below, then press <b>Reload main libraries</b>.`);
  return `<div class="card" style="padding:10px 14px;margin:0 0 14px;display:flex;gap:12px;align-items:center;flex-wrap:wrap">
    <div style="flex:1;min-width:260px;font-size:12.5px">${state_}<br><span class="note" style="margin:0">Folder: <code>${escapeHtml(su.folder)}</code>${su.folder_exists===false?" (does not exist yet \u2014 create it)":""}</span></div>
    <button class="btn ghost btn-sm" onclick="reloadMain()" ${su.running?"disabled":""}>Reload main libraries</button></div>`;
}
async function reloadMain(){
  try{ await apiSend("POST","/api/startup_import/reload"); showToast("Reading the libraries folder\u2026"); }catch(e){ showToast(e.message); }
  await loadLibraries();
}
function buildGlobalMatResults(){
  const q=state.globalMatQuery.trim(), h=DB.hits; if(!q||!h) return "";
  if(h.error) return `<p style="color:var(--amber-l);font-size:12.5px">Search failed: ${escapeHtml(h.error)}</p>`;
  if(!h.results.length) return `<p style="color:var(--muted); font-size:12.5px; margin-bottom:20px;">No materials match "${escapeHtml(q)}".</p>`;
  return `<div class="panel-list" style="margin-bottom:20px;">${h.results.map(x=>`<div class="mat-row" onclick="goToMaterial('${x.library_id}','${x.material_id}')">
    <span>${escapeHtml(x.material)} <span style="color:var(--muted); font-size:11.5px;">— ${escapeHtml(x.library)} · ${x.n_points} pts · ${escapeHtml(x.unit||"")}${x.acquisition_type?` · ${x.acquisition_type}`:""}</span></span><span class="mat-tag">${escapeHtml(x.category||"—")}</span></div>`).join("")}
    ${h.total>h.results.length?`<div class="small" style="padding:8px 12px;color:var(--muted)">showing ${h.results.length} of ${h.total}</div>`:""}</div>`;
}
let _gq=0;
function filterLibraries(){
  // library name box: filter the cards right away
  const grid=document.getElementById("libCardsGrid"); if(grid) grid.innerHTML=buildLibraryCards();
  // material box: ask the database (debounced) and show matches from ALL libraries
  const q=(state.globalMatQuery||"").trim(), box=document.getElementById("libGlobalResults");
  clearTimeout(_gq);
  if(!q){ DB.hits=null; if(box) box.innerHTML=""; return; }
  _gq=setTimeout(async()=>{
    try{ DB.hits=await apiGet(`/api/spectra/search?q=${encodeURIComponent(q)}&limit=60`); }
    catch(e){ DB.hits={error:e.message,results:[],total:0}; }
    if((state.globalMatQuery||"").trim()!==q) return;          // a newer keystroke is pending
    const b=document.getElementById("libGlobalResults"); if(b) b.innerHTML=buildGlobalMatResults();
  },250);
}
function renderStartupIssues(){
  const su=DB.startup, bad=su?(su.files||[]).filter(f=>f.status==="error"||f.status==="skipped"):[];
  if(!bad.length && DB.stats) return "";
  return `<p style="color:#fca5a5;font-size:12.5px;margin:0 0 14px">${DB.stats?`${bad.length} file(s) in the libraries folder could not be read:<br>${bad.map(f=>`${escapeHtml(f.file)}: ${escapeHtml(f.detail||"")}`).join("<br>")}`:`The backend is not reachable ${DB.err?"("+escapeHtml(DB.err)+")":""}.`}</p>`;
}
async function dbUpload(files, libName, extra){
  if(!files.length) return;
  const fd=new FormData(); files.forEach(f=>fd.append("files",f)); if(libName) fd.append("library",libName);
  Object.entries(extra||{}).forEach(([k,v])=>{ if(v) fd.append(k,v); });
  showToast(`Uploading ${files.length} file(s)…`);
  try{ DB.report=await apiSend("POST","/api/import",fd,true); showToast(`"${DB.report.library}": ${DB.report.imported} spectra imported`); }
  catch(e){ DB.report={error:e.message}; showToast("Upload failed"); }
  await loadLibraries();
}
function triggerImportFile(libId){ DB.target=libId; const i=document.getElementById("importFileInput"); i.multiple=true; i.removeAttribute("accept"); i.click(); }
async function handleImportFile(e){
  const lib=state.libraries.find(l=>l.id===DB.target); const f=[...e.target.files]; e.target.value="";
  await dbUpload(f, lib?lib.name:"");
}

/* ---------------- spectrum page for database materials ---------------- */
function dbCsv(sp, name){
  const rows=dbRows(sp), a=document.createElement("a");
  a.href=URL.createObjectURL(new Blob([`wavelength_nm,${sp.measurement_unit||"value"}\n`+rows.map(r=>`${r.wavelength},${r.reflectance}`).join("\n")],{type:"text/csv"}));
  a.download=(name||"spectrum").replace(/\s+/g,"_")+".csv"; a.click();
}
function dbCsvIdx(matId,i){ const lib=state.libraries.find(l=>l.id===state.activeLibId), m=lib.materials.find(x=>x.id===matId); dbCsv(m.dbSpectra[i], m.name+"_"+(i+1)); }
function renderSpectrumViewerDb(lib,mat){
  const isComparing=state.compareList.some(c=>c.libId===lib.id&&c.matId===mat.id);
  const head=`<div class="crumb" onclick="goToLibrary('${lib.id}')">&larr; ${escapeHtml(lib.name)}</div><h1 class="pagetitle">${escapeHtml(mat.name)}</h1>`;
  if(mat.loadingSp||!mat.spectraLoaded&&!mat.err) return head+`<p class="pagesub">Loading the spectrum…</p>`;
  if(mat.err) return head+`<div class="banner-info" style="color:#fca5a5"><span>&#9888;</span><span>${escapeHtml(mat.err)}</span></div>`;
  const sps=mat.dbSpectra||[];
  if(!sps.length) return head+`<p class="pagesub">${escapeHtml(mat.type)} · ${escapeHtml(lib.name)}</p><div class="placeholder-card">This material has no stored spectrum.</div>`;
  const units=[...new Set(sps.map(s=>s.measurement_unit))];
  const series=sps.map((s,i)=>({name:s.sample,color:DS_COLORS[i%DS_COLORS.length],data:dbRows(s).map(r=>({x:r.wavelength,y:r.reflectance}))}));
  const first=sps[0], wl=first.wavelengths;
  return head+`<p class="pagesub">${escapeHtml(mat.type)} · ${mat.nSamples} sample${mat.nSamples===1?"":"s"} · ${sps.length} spectr${sps.length===1?"um":"a"} · ${escapeHtml(lib.name)} · ${escapeHtml(units.join(", "))} · wavelength in nm</p>
  ${units.length>1?`<div class="banner-info"><span>&#9888;</span><span>These spectra have different units (${escapeHtml(units.join(", "))}); the scales are not comparable.</span></div>`:""}
  ${mat.description?`<p class="small" style="color:var(--muted2)">${escapeHtml(mat.description)}</p>`:""}
  <div class="chart-wrap">${dsChart(series,{height:260,unit:units.length===1?units[0]:""})}</div>
  <div style="font-size:11.5px;margin-top:6px;display:flex;gap:10px;flex-wrap:wrap">${series.map(s=>`<span style="color:${s.color}">&#9632; ${escapeHtml(s.name)}</span>`).join("")}</div>
  <table class="gcp-table" style="margin-top:12px"><tr><th>Sample</th><th>Points</th><th>Range (nm)</th><th>Unit</th><th></th></tr>
   ${sps.map((s,i)=>`<tr><td>${escapeHtml(s.sample)}</td><td>${s.values.length}</td><td>${Math.min(...s.wavelengths).toFixed(0)}–${Math.max(...s.wavelengths).toFixed(0)}</td><td>${escapeHtml(s.measurement_unit||"")}</td><td><button class="btn ghost btn-sm" onclick="dbCsvIdx('${mat.id}',${i})">Export CSV</button></td></tr>`).join("")}</table>
  ${srcImageHtml(sps)}
  ${mat.nSamples>12?`<p class="small" style="color:var(--muted)">showing the first 12 of ${mat.nSamples} samples</p>`:""}
  <div style="display:flex; gap:10px; margin-top:14px; flex-wrap:wrap;">
    <button class="btn emerald btn-sm" onclick="cmpOpen('${lib.id}','${mat.id}')">&#43; Add more spectra to compare${CMP.items.length?` (${CMP.items.length} selected)`:""}</button>
  </div>`;
}
function renderSpectrumViewer(){
  const lib=state.libraries.find(l=>l.id===state.activeLibId), mat=lib&&lib.materials.find(m=>m.id===state.activeMatId);
  if(lib&&lib.db&&!mat&&lib.loading) return `<p class="pagesub">Loading…</p>`;
  if(lib&&mat&&mat.db) return renderSpectrumViewerDb(lib,mat);
  return renderSpectrumViewerLocal();
}
function buildMaterialRows(lib){
  if(lib.loading) return `<p style="padding:12px;color:var(--muted);font-size:12.5px">Loading materials…</p>`;
  if(lib.err) return `<p style="padding:12px;color:var(--amber-l);font-size:12.5px">${escapeHtml(lib.err)}</p>`;
  if(lib.type==="object"&&lib.objects) return objectRowsHtml(lib);
  const mq=state.matQuery.trim().toLowerCase(), tf=state.matTypeFilter;
  const mats=lib.materials.filter(m=>m.name.toLowerCase().includes(mq)&&(!tf||m.type===tf));
  const rows=mats.slice(0,300).map(m=>`<div class="mat-row" onclick="goToMaterial('${lib.id}','${m.id}')"><span>${escapeHtml(m.name)}${m.locationName?` <span style="color:var(--muted);font-size:11px;">· 📍 ${escapeHtml(m.locationName)}</span>`:""}</span><span class="mat-tag">${escapeHtml(m.type)}</span></div>`).join("");
  const more=mats.length>300?`<p style="padding:10px 12px;color:var(--muted);font-size:12px">showing 300 of ${mats.length} — type in the search box to narrow down</p>`:"";
  return (rows||`<p style="padding:12px;color:var(--muted);font-size:12.5px;">No materials match your search.</p>`)+more;
}

/* ---------------- object (ROI) libraries: the saved image patch of each sample ---------------- */
/* Only the patch FILE NAME is shown. "Load image" opens the file from the folder where it was saved (it is not copied);
   "Locate" shows it in the file browser. Both go through the sample id: the page never handles the folder path. */
function objectRowsHtml(lib){
  const q=(state.matQuery||"").trim().toLowerCase(), list=(lib.objects||[]).filter(o=>!q||(o.name||"").toLowerCase().includes(q)||(o.class_name||"").toLowerCase().includes(q));
  const f5=v=>v==null?"—":(+v).toFixed(5);
  const rows=list.map(o=>{
    const ok=o.has_patch&&o.patch_exists!==false;
    return `<tr><td>${escapeHtml(o.name)}</td><td>${escapeHtml(o.class_name||"")}</td><td class="num">${f5(o.lat)}</td><td class="num">${f5(o.lon)}</td><td class="num">${o.n_pixels??"—"}</td>
      <td>${o.has_patch?`&#128247; ${escapeHtml(o.patch_file||"")}${o.patch_exists===false?` <span style="color:#c62828" title="The file is not at its saved location any more">· not found</span>`:""}`:`<span class="note" style="margin:0">no patch</span>`}</td>
      <td style="white-space:nowrap"><button class="btn emerald btn-sm" onclick="libObjLoad('${o.sample_id}')" ${ok?"":"disabled"} title="Open this image patch in the Image module">Load image</button>
        <button class="btn ghost btn-sm" onclick="libObjLocate('${o.sample_id}')" ${ok?"":"disabled"} title="Show the file in the file browser">Locate</button>
        <button class="btn ghost btn-sm" onclick="goToMaterial('${lib.id}','${o.material_id}')">Spectrum</button></td></tr>`; }).join("");
  return `<div style="overflow:auto;padding:4px 8px"><table class="gcp-table obj-table"><tr><th>Label</th><th>Class</th><th>Latitude</th><th>Longitude</th><th>Pixels</th><th>Image (patch file)</th><th></th></tr>${rows||`<tr><td colspan=7 class="note">No object samples match.</td></tr>`}</table></div>
    <p class="note" style="padding:0 12px 10px">The image patches stay in the folder where they were saved; the database keeps only the file name and its address.</p>`;
}
async function libObjLoad(sampleId){
  try{
    showToast("Loading the image…");
    const r=await apiSend("POST",`/api/samples/${sampleId}/patch/load`,{});
    I.list=await apiGet("/api/images/"); state.view="image"; render(); await openImage(r.image_id);
    showToast(`${r.name} loaded in the Image module`);
  }catch(e){ showToast(e.message); }
}
async function libObjLocate(sampleId){
  try{ await apiSend("POST",`/api/samples/${sampleId}/patch/reveal`,{}); }catch(e){ showToast(e.message); }
}

/* ---------------- builder: save into the database ---------------- */
async function bldAddFromDb(){
  if(!bld.grid){ showToast("Choose a wavelength grid first (step 1)"); return; }
  const id=document.getElementById("bldDbLib").value;
  try{
    const r=await apiGet(`/api/spectra/search?library_id=${id}&unit=reflectance&limit=150`); let ok=0;
    for(const h of r.results){ const sp=await apiGet(`/api/spectra/${h.spectrum_id}`);
      if(bldAdd(h.material, sp.wavelengths.filter((w,i)=>sp.values[i]!==null), sp.values.filter(v=>v!==null), "library_import", `library · ${h.library}`)) ok++; }
    showToast(`Added ${ok} of ${r.results.length} spectra (the rest are listed under Rejected)`);
  }catch(e){ bldReject("database",e.message); }
  renderMainOnly();
}
