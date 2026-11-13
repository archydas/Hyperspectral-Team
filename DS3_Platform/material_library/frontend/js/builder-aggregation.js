/* builder-aggregation.js — Spectral Library Builder, data aggregation (Analysis), categories filters
   (classic script, loaded in order from index.html; all functions are global) */
/* =====================================================================================================================
   TASK 1 ADDITIONS — Spectral Library Builder (ENVI Classic style, runs on the backend), data aggregation across
   scenes, category / acquisition filters. Later declarations replace earlier ones with the same name.
   ===================================================================================================================== */
function lfModal(html){
  let h=document.getElementById("lfModalHost");
  try{ if(document.activeElement&&h&&h.contains(document.activeElement)) document.activeElement.blur(); }catch(_){}
  if(!h){ h=document.createElement("div"); h.id="lfModalHost"; document.body.appendChild(h); }
  h.innerHTML=html?`<div class="lf-overlay"><div class="lf-dialog">${html}</div></div>`:"";
}
function lfModalClose(){ lfModal(""); }
async function dlPost(path, body, fallback){
  const r=await fetch(apiUrl(path),{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)});
  if(!r.ok){ let d=`Request failed (${r.status})`; try{ const j=await r.json(); d=typeof j.detail==="string"?j.detail:JSON.stringify(j.detail); }catch(_){} throw new Error(d); }
  const cd=r.headers.get("Content-Disposition")||"", m=cd.match(/filename="([^"]+)"/);
  const a=document.createElement("a"); a.href=URL.createObjectURL(await r.blob()); a.download=m?m[1]:fallback; a.click();
}
function csvDownload(name, rows){
  const a=document.createElement("a"); a.href=URL.createObjectURL(new Blob([rows.join("\n")],{type:"text/csv"})); a.download=name; a.click();
}

/* ---------------------------------------------------- Spectral Library Builder ---------------------------------------- */
const BLD2={ grid:null, gridSrc:"", gridMode:"first", items:[], rejected:[], libName:"", pick:"", category:"", acq:"",
  maskWater:false, minCov:0.05, ref:null, busy:false, err:"", picked:null };
const BDD={ list:[], sel:null, total:0, from:1, to:1, recent:[], err:"", subsetOpen:false, openMenu:false };

function bldStep1(){
  lfModal(`<div class="lf-title">Spectral Library Builder</div>
   <div class="lf-box"><div style="margin-bottom:8px">Input Spectral Wavelength From:</div>
    ${[["data","Data File..."],["ascii","ASCII File..."],["first","First Input Spectrum"]].map(([v,l],i)=>`<label class="lf-radio"><input type="radio" name="bldSrc" value="${v}" ${i===0?"checked":""}> ${l}</label>`).join("")}</div>
   <div class="lf-btns"><button class="btn violet btn-sm" onclick="bldStep1Ok()">OK</button><button class="btn ghost btn-sm" onclick="lfModalClose()">Cancel</button></div>`);
}
function bldStep1Ok(){
  const v=document.querySelector('input[name="bldSrc"]:checked')?.value;
  if(v==="data") return bldDataDialog();
  if(v==="ascii"){ lfModalClose(); return document.getElementById("bldAsciiFile").click(); }
  BLD2.grid=null; BLD2.gridMode="first"; BLD2.gridSrc="first input spectrum"; lfModalClose(); renderMainOnly();
}
async function bldAsciiPicked(e){
  const f=e.target.files[0]; e.target.value=""; if(!f) return;
  const fd=new FormData(); fd.append("file",f);
  try{ const r=await apiSend("POST","/api/builder/wavelengths/ascii",fd,true);
    BLD2.grid=r.wavelengths; BLD2.gridMode="ascii"; BLD2.gridSrc=`ASCII file ${r.name}${r.converted_from_um?" (µm converted to nm)":""}`; BLD2.err="";
    showToast(`${r.wavelengths.length} wavelengths read`); }
  catch(err){ BLD2.err=err.message; }
  renderMainOnly();
}
async function bldDataDialog(){
  BDD.err=""; BDD.sel=null; BDD.subsetOpen=false; BDD.openMenu=false;
  try{ BDD.list=await apiGet("/api/builder/sources"); }catch(e){ BDD.list=[]; BDD.err=e.message; }
  bldDataDraw();
}
function bldDataDraw(){
  const s=BDD.sel, L=BDD.list;
  const info=s?`File: ${escapeHtml(s.name)}<br>Dims: ${escapeHtml(s.info)}<br>File Type: ${escapeHtml(s.type)}<br>Wavelength: ${s.wl_min!=null?`${(+s.wl_min).toFixed(1)} to ${(+s.wl_max).toFixed(1)} nm`:"unknown"}<br>Bands: ${s.bands}<br>Unit: ${escapeHtml(s.unit||"")}`:"Select a file on the left.";
  lfModal(`<div class="lf-title">File Containing Output Wavelength</div>
   <div class="lf-cols"><div><div class="lf-cap">Select Input File:</div><div class="lf-list">${L.map((x,i)=>`<div class="lf-item ${s&&s.id===x.id?"sel":""}" onclick="bldDataPick(${i})">${escapeHtml(x.name)} <span class="small" style="color:var(--muted)">${x.kind}</span></div>`).join("")||`<div class="small" style="padding:8px">Nothing opened yet. Use Open ▾ below.</div>`}</div></div>
   <div><div class="lf-cap">File Information:</div><div class="lf-info">${info}</div></div></div>
   <div class="lf-row"><button class="btn ghost btn-sm" onclick="BDD.subsetOpen=!BDD.subsetOpen;bldDataDraw()" ${s?"":"disabled"}>Spectral Subset</button>
     <span class="lf-box" style="padding:4px 10px">${s?`${BDD.to-BDD.from+1}/${BDD.total} Bands`:"–"}</span>
     ${BDD.subsetOpen&&s?`<span class="small">bands <input type="number" min="1" max="${BDD.total}" value="${BDD.from}" style="width:70px" onchange="BDD.from=Math.max(1,+this.value||1);bldDataDraw()"> to <input type="number" min="1" max="${BDD.total}" value="${BDD.to}" style="width:70px" onchange="BDD.to=Math.min(BDD.total,+this.value||BDD.total);bldDataDraw()"></span>`:""}</div>
   ${BDD.err?`<p style="color:#fca5a5;font-size:12px">${escapeHtml(BDD.err)}</p>`:""}
   <div class="lf-btns" style="position:relative">
     <button class="btn violet btn-sm" onclick="bldDataOk()" ${s?"":"disabled"}>OK</button>
     <button class="btn ghost btn-sm" onclick="lfModalClose()">Cancel</button>
     <button class="btn ghost btn-sm" onclick="bldStep1()">Previous</button>
     <button class="btn ghost btn-sm" onclick="BDD.openMenu=!BDD.openMenu;bldDataDraw()">Open ▾</button>
     ${BDD.openMenu?`<div class="lf-menu">
        <div class="lf-mi" style="position:relative" onmouseenter="this.lastElementChild.style.display='block'" onmouseleave="this.lastElementChild.style.display='none'">Previous File &#9656;<div class="lf-sub" style="display:none">${BDD.recent.map(r=>`<div class="lf-mi" onclick="bldDataRecent('${r.id}')">${escapeHtml(r.name)}</div>`).join("")||`<div class="lf-mi off">none yet</div>`}</div></div>
        <div class="lf-mi" onclick="document.getElementById('bldNewFile').click()">New File...</div>
        <hr><div class="lf-mi" onclick="document.getElementById('bldNewLib').click()">Spectral Library...</div>
        <div class="lf-mi off" title="Not available yet">ROI File...</div><div class="lf-mi off" title="Not available yet">EVF File...</div></div>`:""}
   </div>`);
}
async function bldDataPick(i){
  BDD.sel=BDD.list[i]; BDD.total=BDD.sel.bands; BDD.from=1; BDD.to=BDD.total; BDD.subsetOpen=false; BDD.openMenu=false;
  try{ const w=await apiGet(`/api/builder/sources/${BDD.sel.kind}/${BDD.sel.id}/wavelengths`); BDD.total=w.total_bands; BDD.to=w.total_bands; }catch(e){ BDD.err=e.message; }
  bldDataDraw();
}
function bldDataRecent(id){ const i=BDD.list.findIndex(x=>x.id===id); BDD.openMenu=false; if(i>=0) bldDataPick(i); else bldDataDraw(); }
async function bldDataOk(){
  const s=BDD.sel; if(!s) return;
  try{ const w=await apiGet(`/api/builder/sources/${s.kind}/${s.id}/wavelengths?band_from=${BDD.from}&band_to=${BDD.to}`);
    BLD2.grid=w.wavelengths; BLD2.gridMode="data"; BLD2.gridSrc=`${s.kind==="image"?"image":"library"} ${s.name}${BDD.to-BDD.from+1<BDD.total?` (bands ${BDD.from}–${BDD.to})`:""}`;
    if(!BDD.recent.find(r=>r.id===s.id)) BDD.recent.unshift({id:s.id,name:s.name});
    BLD2.err=""; lfModalClose(); renderMainOnly(); }
  catch(e){ BDD.err=e.message; bldDataDraw(); }
}
async function bldNewFiles(e, asLibrary){
  const fs=[...e.target.files]; e.target.value=""; if(!fs.length) return;
  const fd=new FormData(); fs.forEach(f=>fd.append("files",f));
  if(asLibrary) fd.append("library", fs[0].name.replace(/\.[^.]+$/,""));
  BDD.openMenu=false; showToast("Reading…");
  try{ const before=new Set(BDD.list.map(x=>x.id)); await apiSend("POST","/api/import",fd,true);
    BDD.list=await apiGet("/api/builder/sources"); await loadLibraries();
    const fresh=BDD.list.findIndex(x=>!before.has(x.id)); BDD.err=""; if(fresh>=0) return bldDataPick(fresh); }
  catch(err){ BDD.err=err.message; }
  bldDataDraw();
}

async function bldFiles(e){
  const fs=[...e.target.files]; e.target.value=""; if(!fs.length) return;
  const fd=new FormData(); fs.forEach(f=>fd.append("files",f));
  try{ const r=await apiSend("POST","/api/builder/parse",fd,true);
    r.errors.forEach(x=>BLD2.rejected.push({name:x.file,reason:x.reason}));
    if(r.spectra.length) await bldAdd(r.spectra,"file"); else { BLD2.busy=false; renderMainOnly(); } }
  catch(err){ BLD2.err=err.message; renderMainOnly(); }
}
async function bldFromImage(){
  const a=I.active;
  if(!a){ showToast("Open an image in the Image module first, then select a pixel or draw an ROI"); return; }
  try{
    let r, nm;
    if(I.roi&&I.roi.closed){ r=await apiSend("POST",`/api/images/${a.id}/roi`,{polygon:I.roi.pts}); nm=`${a.name} ROI`; }
    else if(I.pixel){ r=await apiGet(`/api/images/${a.id}/pixel?row=${I.pixel.row}&col=${I.pixel.col}&size=${I.win}`); nm=`${a.name} pixel ${I.pixel.row},${I.pixel.col}`; }
    else { showToast("In the Image module, click a pixel or finish an ROI first"); return; }
    await bldAdd([{name:nm,wavelengths:r.wavelengths,values:r.values,unit:r.unit,source:a.name,origin:r.n_pixels?"image_roi":"image_pixel"}]);
  }catch(e){ BLD2.err=e.message; renderMainOnly(); }
}
async function bldLibDialog(){
  BLD2.picked=BLD2.picked||{libId:"",list:[],sel:new Set(),q:""};
  lfLibDraw();
}
async function bldLibLoad(){
  const p=BLD2.picked; if(!p.libId){ p.list=[]; return lfLibDraw(); }
  try{ p.list=(await apiGet(`/api/spectra/search?library_id=${p.libId}&limit=200${p.q?`&q=${encodeURIComponent(p.q)}`:""}`)).results; }catch(_){ p.list=[]; }
  lfLibDraw();
}
function lfLibDraw(){
  const p=BLD2.picked, libs=dbLibs();
  lfModal(`<div class="lf-title">Import Spectra from Spectral Library</div>
   <select onchange="BLD2.picked.libId=this.value;BLD2.picked.sel=new Set();bldLibLoad()" style="width:100%"><option value="">— choose a library —</option>${libs.map(l=>`<option value="${l.id}" ${p.libId===l.id?"selected":""}>${escapeHtml(l.name)} (${l.spectra})</option>`).join("")}</select>
   <input type="text" placeholder="search" value="${escapeHtml(p.q)}" style="width:100%;margin-top:6px" onchange="BLD2.picked.q=this.value;bldLibLoad()">
   <div class="lf-list" style="margin-top:8px;max-height:280px">${p.list.map(s=>`<label class="lf-item" style="display:flex;gap:6px"><input type="checkbox" ${p.sel.has(s.spectrum_id)?"checked":""} onchange="this.checked?BLD2.picked.sel.add('${s.spectrum_id}'):BLD2.picked.sel.delete('${s.spectrum_id}');lfLibDraw()"> ${escapeHtml(s.material)}</label>`).join("")||`<div class="small" style="padding:8px">${p.libId?"no spectra":"choose a library"}</div>`}</div>
   <div class="lf-btns"><button class="btn violet btn-sm" onclick="bldLibAdd()" ${p.sel.size?"":"disabled"}>Import ${p.sel.size||""} selected</button><button class="btn ghost btn-sm" onclick="lfModalClose()">Cancel</button></div>`);
}
async function bldLibAdd(){
  const ids=[...BLD2.picked.sel]; lfModalClose(); const raw=[];
  try{ for(const id of ids){ const sp=specCache[id]||(specCache[id]=await apiGet(`/api/spectra/${id}`)); const h=BLD2.picked.list.find(x=>x.spectrum_id===id);
      raw.push({name:h?h.material:id,wavelengths:sp.wavelengths,values:sp.values,unit:sp.measurement_unit,source:h?h.library:"",origin:"library"}); }
    await bldAdd(raw,"library"); }
  catch(e){ BLD2.err=e.message; renderMainOnly(); }
}
function bldRemove(id){ BLD2.items=BLD2.items.filter(x=>x.id!==id); if(BLD2.ref===id) BLD2.ref=null; renderMainOnly(); }
function bldReset(){ if(BLD2.items.length&&!confirm("Remove all spectra and start again? The wavelengths will be taken from the next file you open.")) return; BLD2.items=[]; BLD2.grid=null; BLD2.gridMode="first"; BLD2.gridSrc=""; BLD2.rejected=[]; BLD2.ref=null; BLD2.err=""; renderMainOnly(); }
function bldAngle(a,b){
  let ab=0,aa=0,bb=0,n=0; for(let i=0;i<a.values.length;i++){ const x=a.values[i],y=b.values[i]; if(x==null||y==null) continue; ab+=x*y; aa+=x*x; bb+=y*y; n++; }
  if(n<3||!aa||!bb) return null; return {deg:Math.acos(Math.min(1,ab/Math.sqrt(aa*bb)))*180/Math.PI,n};
}
function bldExportCsv(){
  const g=BLD2.grid, rows=["wavelength_nm,"+BLD2.items.map(i=>`"${i.name.replace(/"/g,'""')}"`).join(",")];
  g.forEach((w,k)=>rows.push([w,...BLD2.items.map(i=>i.values[k]==null?"":i.values[k])].join(",")));
  csvDownload("spectral_library_builder.csv",rows);
}
BLD2.menu=false;
document.addEventListener("click",e=>{ if(BLD2.menu&&!e.target.closest("[onclick^='bldMenu']")&&!e.target.closest('.lf-menu')){ BLD2.menu=false; if(state.view==="lib-functions"&&LF.tab==="builder") renderMainOnly(); } });
function bldMenu(open){ BLD2.menu=open===undefined?!BLD2.menu:open; renderMainOnly(); }
function bldPick(kind){
  BLD2.menu=false;
  if(kind==="files") document.getElementById("bldFiles").click();
  else if(kind==="ascii") document.getElementById("bldAscii").click();
  else if(kind==="library") bldLibDialog();
  else if(kind==="image") bldFromImage();
  else if(kind==="data") openImageDialog();            // open an image (ENVI, GeoTIFF, MAT, HDF...), then click pixels / draw an ROI
  renderMainOnly();
}
function renderBuilder(){
  const B=BLD2, g=B.grid, ref=B.items.find(i=>i.id===B.ref);
  const series=g?B.items.map((it,k)=>({name:it.name,color:DS_COLORS[k%DS_COLORS.length],data:g.map((w,j)=>({x:w,y:it.values[j]})).filter(d=>d.y!=null)})):[];
  const matches=ref?B.items.filter(i=>i!==ref).map(i=>({i,a:bldAngle(ref,i)})).filter(x=>x.a).sort((a,b)=>a.a.deg-b.a.deg):[];
  return `<input type="file" id="bldAsciiFile" style="display:none" onchange="bldAsciiPicked(event)">
  <input type="file" id="bldNewFile" multiple style="display:none" onchange="bldNewFiles(event,false)">
  <input type="file" id="bldNewLib" style="display:none" onchange="bldNewFiles(event,true)">
  <input type="file" id="bldFiles" multiple style="display:none" onchange="bldFiles(event)">
  <input type="file" id="bldAscii" multiple accept=".txt,.csv,.asc,.dat,.tsv,.prn,.spectrum" style="display:none" onchange="bldFiles(event)">
  <div class="card" style="padding:14px;margin-bottom:12px">
    <div style="display:flex;gap:10px;align-items:center;flex-wrap:wrap"><b>Spectral Library Builder</b>
      <span style="position:relative;display:inline-block">
        <button class="btn emerald btn-sm" onclick="bldMenu()" ${B.busy?"disabled":""}>Open &#9662;</button>
        ${B.menu?`<div class="lf-menu" style="position:absolute;left:0;top:100%;z-index:50;min-width:330px;margin-top:2px">
          <div class="lf-mi" onclick="bldPick('files')">Spectra file(s) — any format (CSV, TXT, SLI, SIG, ASD, XLSX, JSON…)</div>
          <div class="lf-mi" onclick="bldPick('library')">Spectral library (database)…</div>
          <div class="lf-mi" onclick="bldPick('data')">Data file — image (ENVI, GeoTIFF, MAT, HDF…)…</div>
          <div class="lf-mi" onclick="bldPick('ascii')">ASCII file (wavelength, value columns)…</div>
          <hr><div class="lf-mi" onclick="bldPick('image')">From the open image (selected pixel or ROI)</div></div>`:""}
      </span>
      <span class="small">${g?`Output wavelengths: <b>${g.length} bands</b>, ${g[0].toFixed(1)}–${g[g.length-1].toFixed(1)} nm`:"Use Open to add spectra. The output wavelengths are set automatically from the first spectrum you add."}</span>
      ${g||B.items.length?`<button class="btn ghost btn-sm" onclick="bldReset()">Start over</button>`:""}
      ${B.busy?`<span class="small">working…</span>`:""}</div>
    ${B.err?`<p style="color:#c62828;font-size:12.5px;margin:8px 0 0">${escapeHtml(B.err)}</p>`:""}
    <p class="note" style="margin:8px 0 0">Every spectrum you open is resampled (linear, no extrapolation) onto the wavelengths of the first spectrum. ROI / pixel come from the Image module.</p>
  </div>
  ${B.rejected.length?`<div class="card" style="padding:12px;margin-bottom:12px;border-color:#f59e0b"><b style="color:var(--amber-l)">Rejected (${B.rejected.length})</b>${B.rejected.slice(0,20).map(r=>`<div class="small">${escapeHtml(r.name)} — ${escapeHtml(r.reason)}</div>`).join("")}<button class="btn ghost btn-sm" style="margin-top:6px" onclick="BLD2.rejected=[];renderMainOnly()">Clear</button></div>`:""}
  <div class="card" style="padding:14px">
    <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap"><b>Spectra (${B.items.length})</b>
      ${libPickHtml("bldLib",B.libName,"pixel",{pick:B.pick,ref:"BLD2"})}
      <input id="bldCat" type="text" value="${escapeHtml(B.category)}" placeholder="category e.g. Vegetation/Tree" style="max-width:190px">
      <select id="bldAcq" title="acquisition type"><option value="">acquisition: —</option>${["lab","field","airborne","satellite"].map(a=>`<option ${B.acq===a?"selected":""}>${a}</option>`).join("")}</select>
      <button class="btn blue btn-sm" onclick="bldSave()" ${B.items.length?"":"disabled"}>Save as library</button>
      <button class="btn ghost btn-sm" onclick="bldExportCsv()" ${B.items.length?"":"disabled"}>Export CSV</button></div>
    ${bldRawHtml()}
    ${series.length?`<div class="chart-wrap" style="margin-top:10px">${dsChart(series,{height:240,width:760})}</div><p class="note">x: wavelength (nm). Gaps are real gaps (outside the source range).</p>`:`<p class="note" style="margin-top:10px">No spectra yet. Click <b>Open</b> and choose a source.</p>`}
    <table class="gcp-table"><tr><th>Name</th><th>Origin</th><th>Source</th><th>Src points</th><th>Unit</th><th></th></tr>
    ${B.items.map(i=>`<tr><td>${escapeHtml(i.name)}</td><td>${escapeHtml(i.origin)}</td><td>${escapeHtml(i.source||"")}</td><td>${i.n_source}</td><td>${escapeHtml(i.unit||"")}</td>
      <td><button class="btn ghost btn-sm" onclick="BLD2.ref='${i.id}';renderMainOnly()">Match</button> <button class="btn ghost btn-sm" onclick="bldRemove('${i.id}')">Remove</button></td></tr>`).join("")||`<tr><td colspan=6>Empty</td></tr>`}</table>
    ${ref?`<h4 style="margin:14px 0 4px">Spectral angle vs “${escapeHtml(ref.name)}” (smaller = more similar)</h4><table class="gcp-table"><tr><th>Spectrum</th><th>Angle</th><th>Bands used</th></tr>${matches.map(x=>`<tr><td>${escapeHtml(x.i.name)}</td><td>${x.a.deg.toFixed(2)}°</td><td>${x.a.n}</td></tr>`).join("")||`<tr><td colspan=3>no other spectra</td></tr>`}</table>`:""}
  </div>`;
}

/* ---------------------------------------------------- Data aggregation (Analysis page) ------------------------------- */
const AG={ rois:[], mode:"intersection", step:"", mask:true, result:null, busy:false, err:"", lib:"Aggregated signatures", cat:"", acq:"", draft:null };
try{ const s=JSON.parse(localStorage.getItem("ds3_ag_rois")||"[]"); if(Array.isArray(s)) AG.rois=s; }catch(_){}
function agStore(){ try{ localStorage.setItem("ds3_ag_rois",JSON.stringify(AG.rois)); }catch(_){} }
function agConfirmAdd(){
  const n=(document.getElementById("agName")?.value||"").trim(); if(!n){ showToast("Give the class a name"); return; }
  AG.rois.push({...AG.draft,class_name:n}); AG.draft=null; agStore(); AG.result=null; lfModalClose(); showToast(`ROI added to class "${n}" (${AG.rois.length} total)`);
}
function agBody(save){
  const b={rois:AG.rois.filter(r=>r.show!==false).map(r=>({image_id:r.image_id,class_name:r.class_name,polygon:r.polygon})),mode:AG.mode,mask_bad_bands:!!AG.mask};
  if(AG.step!=="" && +AG.step>0) b.step=+AG.step;
  if(save) b.save={library:AG.lib.trim(),category:AG.cat.trim()||null,acquisition_type:AG.acq||null};
  return b;
}

/* ---------------------------------------------------- Categories and acquisition filters (Viewer) ---------------------- */
const FAC={ libId:null, data:null };
async function facLoad(libId){
  if(FAC.libId===libId&&FAC.data) return; FAC.libId=libId;
  try{ FAC.data=await apiGet("/api/facets"+(libId?`?library_id=${libId}`:"")); }catch(_){ FAC.data=null; }
}
async function lfViewerLoad(){
  const v=LF.v; v.err="";
  if(!v.libId){ v.list=[]; v.total=0; return; }
  await facLoad(v.libId);
  const q=[`library_id=${v.libId}`,"limit=500"]; if(v.q) q.push(`q=${encodeURIComponent(v.q)}`); if(v.cat) q.push(`category_path=${encodeURIComponent(v.cat)}`); if(v.acq) q.push(`acquisition=${encodeURIComponent(v.acq)}`);
  try{ const r=await apiGet("/api/spectra/search?"+q.join("&")); v.list=r.results; v.total=r.total; }catch(e){ v.err=e.message; }
}
async function lfViewerFilter(){ LF.v.picked=new Set(); await lfViewerLoad(); renderMainOnly(); }
function lfFilterHtml(){
  const v=LF.v, f=FAC.data; if(!v.libId||!f) return "";
  const cats=f.categories.map(c=>`<option value="${escapeHtml(c.path)}" ${v.cat===c.path?"selected":""}>${c.path==="__none__"?"(uncategorised)":"&nbsp;&nbsp;".repeat(c.path.split("/").length-1)+escapeHtml(c.path.split("/").pop())} (${c.count})</option>`).join("");
  const acq=Object.entries(f.acquisition).map(([k,n])=>`<option value="${k}" ${v.acq===k?"selected":""}>${k==="__none__"?"(untagged)":k} (${n})</option>`).join("");
  return `<div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;margin-top:8px"><select onchange="LF.v.cat=this.value;lfViewerFilter()"><option value="">all categories</option>${cats}</select>
    <select onchange="LF.v.acq=this.value;lfViewerFilter()"><option value="">all acquisition</option>${acq}</select></div>`;
}
function lfListHtml(){
  const v=LF.v, chip=a=>a?`<span class="acq-chip acq-${a}">${a}</span>`:"";
  return v.list.map(s=>`<label style="display:flex;gap:6px;align-items:center;font-size:12.5px;padding:2px 0"><input type="checkbox" ${v.picked.has(s.spectrum_id)?"checked":""} onchange="lfViewerToggle('${s.spectrum_id}')"> ${escapeHtml(s.material)} ${chip(s.acquisition_type)} <span style="color:var(--muted)">${escapeHtml(s.category||"")} ${escapeHtml(s.unit||"")}</span></label>`).join("")
    || `<span class="note">${v.libId?"no materials":"choose a library"}</span>`;
}
async function lfViewerPick(id){ LF.v.libId=id; LF.v.picked=new Set(); LF.v.q=""; LF.v.cat=""; LF.v.acq=""; FAC.data=null; await lfViewerLoad(); renderMainOnly(); }
