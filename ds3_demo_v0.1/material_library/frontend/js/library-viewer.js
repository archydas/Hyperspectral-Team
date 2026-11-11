/* library-viewer.js — Library function: viewer and resampling
   (classic script, loaded in order from index.html; all functions are global) */
/* ===================== LIBRARY FUNCTION: builder / viewer / resampling ===================== */
const LF = { tab:"builder",
  v:{ libId:"", list:[], total:0, q:"", picked:new Set(), err:"", cat:"", acq:"" },
  r:{ libId:"", mode:"range", start:400, end:2500, step:10, imageId:"", wl:"", name:"", mask:false, cov:0.5, busy:false, result:null, err:"", images:[] } };
function openLibFunctions(tab){ if(tab) LF.tab=tab; state.view="lib-functions"; render(); if(LF.tab==="resample") lfLoadImages(); }
async function lfLoadImages(){ try{ LF.r.images=await apiGet("/api/images/"); renderMainOnly(); }catch(_){} }
function dbLibs(){ return state.libraries.filter(l=>l.db); }
/* "Save to library" chooser used everywhere: ONE drop-down — "New library" (then type its name) or an existing library.
   kind = "object" lists the object libraries (ROI samples), anything else lists the pixel/spectral libraries (never the main library). */
function libPickHtml(id,defName,kind,store){       // store (optional): an object that remembers the choice while the page is redrawn
  const libs=dbLibs().filter(l=>!l.main&&(kind==="object")===(l.type==="object")), pick=store&&store.pick||"";
  return `<select id="${id}Sel" onchange="libPickChange('${id}')" style="min-width:200px"><option value="">＋ New library…</option>${libs.map(l=>`<option value="${escapeHtml(l.name)}" ${pick===l.name?"selected":""}>${escapeHtml(l.name)} (${l.spectra})</option>`).join("")}</select>
    <input id="${id}Name" type="text" value="${escapeHtml(defName||"")}" placeholder="name of the new library" style="min-width:200px;${pick?"display:none":""}" ${store?`oninput="${store.ref}.libName=this.value"`:""}>`;
}
function libPickChange(id){ const sel=document.getElementById(id+"Sel"), nm=document.getElementById(id+"Name"); if(sel&&nm) nm.style.display=sel.value?"none":""; if(id==="bldLib") BLD2.pick=sel.value; }
function libPickValue(id){ const sel=document.getElementById(id+"Sel"), nm=document.getElementById(id+"Name"); return sel?(sel.value||(nm&&nm.value)||"").trim():""; }
function libPickExisting(id){ const sel=document.getElementById(id+"Sel"); return !!(sel&&sel.value); }
function renderLibFunctions(){
  const T={viewer:["Spectral Library Viewer","Look at spectra from your libraries, filter by category and acquisition type, compare and export."],
           resample:["Spectral Library Resampling","Resample every spectrum of a library to another wavelength grid and store it as a new library."]};
  const head=LF.tab==="builder"?"":`<h1 class="pagetitle">${T[LF.tab][0]}</h1><p class="pagesub">${T[LF.tab][1]}</p>`;
  return `<div class="crumb" onclick="setView('library-list')">&larr; Library</div>${head}
  ${LF.tab==="builder"?renderBuilder():LF.tab==="viewer"?renderLibViewer():renderLibResample()}`;
}

/* ---- viewer ---- */
let _lfq; function lfViewerSearch(q){ LF.v.q=q; clearTimeout(_lfq); _lfq=setTimeout(async()=>{ await lfViewerLoad(); lfViewerRefreshList(); },250); }
function lfViewerToggle(id){ const p=LF.v.picked; if(p.has(id)) p.delete(id); else{ if(p.size>=12){ showToast("At most 12 spectra at a time"); lfViewerRefreshList(); return; } p.add(id); } lfViewerChart(); }
function lfViewerRefreshList(){ const b=document.getElementById("lfList"); if(b) b.innerHTML=lfListHtml(); }
async function lfViewerChart(){
  const box=document.getElementById("lfChart"); if(!box) return; const v=LF.v;
  if(!v.picked.size){ box.innerHTML=`<div class="placeholder-card">Tick materials in the list to plot them.</div>`; return; }
  const ids=[...v.picked], series=[], units=new Set(); LF.v.shown=[];
  for(let k=0;k<ids.length;k++){
    try{
      const sp=specCache[ids[k]]||(specCache[ids[k]]=await apiGet(`/api/spectra/${ids[k]}`));
      const h=v.list.find(x=>x.spectrum_id===ids[k]); units.add(sp.measurement_unit);
      series.push({name:h?h.material:ids[k],color:DS_COLORS[k%DS_COLORS.length],data:dbRows(sp).map(r=>({x:r.wavelength,y:r.reflectance})),sp});
    }catch(e){ box.innerHTML=`<p style="color:#fca5a5">${escapeHtml(e.message)}</p>`; return; }
  }
  LF.v.shown=series;
  box.innerHTML=`<div class="chart-wrap">${dsChart(series,{height:300,width:760})}</div>
   <div style="font-size:11.5px;margin-top:6px;display:flex;gap:10px;flex-wrap:wrap">${series.map(s=>`<span style="color:${s.color}">&#9632; ${escapeHtml(s.name)}</span>`).join("")}</div>
   ${units.size>1?`<p class="note">These spectra have different units (${escapeHtml([...units].join(", "))}); their scales are not comparable.</p>`:`<p class="note">x: wavelength (nm) · y: ${escapeHtml([...units][0]||"")}</p>`}
   <div style="margin-top:8px"><button class="btn ghost btn-sm" onclick="lfViewerExport()">Export CSV</button></div>`;
}
function lfViewerExport(){
  const rows=["material,wavelength_nm,value"]; (LF.v.shown||[]).forEach(s=>dbRows(s.sp).forEach(r=>rows.push(`"${s.name.replace(/"/g,'""')}",${r.wavelength},${r.reflectance}`)));
  const a=document.createElement("a"); a.href=URL.createObjectURL(new Blob([rows.join("\n")],{type:"text/csv"})); a.download="library_spectra.csv"; a.click();
}
function renderLibViewer(){
  const v=LF.v, libs=dbLibs();
  return `<div style="display:grid;grid-template-columns:320px minmax(0,1fr);gap:18px;align-items:start">
   <div class="card" style="padding:14px">
     <label class="small">Library</label>
     <select onchange="lfViewerPick(this.value)" style="width:100%"><option value="">— choose —</option>${libs.map(l=>`<option value="${l.id}" ${v.libId===l.id?"selected":""}>${escapeHtml(l.name)} (${l.spectra})</option>`).join("")}</select>
     <input type="text" placeholder="search materials" value="${escapeHtml(v.q)}" oninput="lfViewerSearch(this.value)" style="width:100%;margin-top:8px">
     ${lfFilterHtml()}
     <div id="lfList" style="max-height:420px;overflow:auto;margin-top:8px;border:1px solid var(--border);border-radius:8px;padding:6px">${lfListHtml()}</div>
     ${v.total>v.list.length?`<p class="note">showing ${v.list.length} of ${v.total} — use the search box</p>`:""}
     ${v.err?`<p style="color:#fca5a5;font-size:12px">${escapeHtml(v.err)}</p>`:""}
     <button class="btn ghost btn-sm" style="margin-top:8px" onclick="LF.v.picked=new Set();renderMainOnly()">Clear selection</button>
   </div>
   <div id="lfChart"></div></div>`;
}

/* ---- resampling ---- */
function rsSet(k,val){ LF.r[k]=val; if(k==="mode") renderMainOnly(); }
async function rsRun(){
  const r=LF.r; r.err=""; r.result=null;
  if(!r.libId){ r.err="Choose the library to resample."; renderMainOnly(); return; }
  if(!r.name.trim()){ r.err="Give the new library a name."; renderMainOnly(); return; }
  const body={name:r.name.trim(),mode:r.mode,min_coverage:+r.cov,mask_water:!!r.mask};
  if(r.mode==="range"){ body.start=+r.start; body.end=+r.end; body.step=+r.step; }
  else if(r.mode==="image"){ if(!r.imageId){ r.err="Choose an image."; renderMainOnly(); return; } body.image_id=r.imageId; }
  else body.wavelengths=r.wl.split(/[\s,;]+/).map(parseFloat).filter(Number.isFinite);
  r.busy=true; renderMainOnly();
  try{ r.result=await apiSend("POST",`/api/libraries/${r.libId}/resample`,body); await loadLibraries(); }
  catch(e){ r.err=e.message; }
  r.busy=false; renderMainOnly();
}
function renderLibResample(){
  const r=LF.r, libs=dbLibs(), res=r.result;
  return `<div class="card" style="padding:16px;max-width:760px">
    <p class="note" style="margin:0 0 10px">Every spectrum of the chosen library is resampled (linear interpolation, no extrapolation, gaps are not bridged) to a new wavelength grid and stored as a <b>new</b> library. The original is not changed.</p>
    <label class="small">Library to resample</label>
    <select onchange="rsSet('libId',this.value)" style="width:100%;max-width:420px"><option value="">— choose —</option>${libs.map(l=>`<option value="${l.id}" ${r.libId===l.id?"selected":""}>${escapeHtml(l.name)} (${l.spectra} spectra)</option>`).join("")}</select>
    <div style="margin-top:12px"><label class="small">New wavelength grid from</label>
      <div style="display:flex;gap:14px;flex-wrap:wrap;font-size:13px;margin-top:4px">
        ${[["range","A range (start, end, step)"],["image","The bands of an image"],["list","A list of wavelengths"]].map(([k,l])=>`<label><input type="radio" name="rsmode" ${r.mode===k?"checked":""} onchange="rsSet('mode','${k}')"> ${l}</label>`).join("")}</div></div>
    <div style="margin-top:10px">
    ${r.mode==="range"?`<div style="display:flex;gap:8px;flex-wrap:wrap">${[["start","Start (nm)"],["end","End (nm)"],["step","Step (nm)"]].map(([k,l])=>`<div><label class="small">${l}</label><input type="number" step="any" value="${r[k]}" onchange="rsSet('${k}',this.value)" style="width:110px"></div>`).join("")}</div>`
     :r.mode==="image"?`<select onchange="rsSet('imageId',this.value)" style="width:100%;max-width:520px"><option value="">— choose an opened image —</option>${r.images.map(i=>`<option value="${i.id}" ${r.imageId===i.id?"selected":""}>${escapeHtml(i.name)} (${i.bands} bands, ${i.wavelength_min.toFixed(0)}–${i.wavelength_max.toFixed(0)} nm)</option>`).join("")}</select>${r.images.length?"":`<p class="note">No image opened yet — open one in the Image module first.</p>`}<p class="note">Bands flagged as bad in the image are left out of the grid.</p>`
     :`<textarea rows="3" placeholder="450, 460, 470, … (nm)" onchange="rsSet('wl',this.value)" style="width:100%;max-width:520px">${escapeHtml(r.wl)}</textarea>`}</div>
    <div style="display:flex;gap:16px;flex-wrap:wrap;margin-top:12px;align-items:center;font-size:12.5px">
      <label><input type="checkbox" ${r.mask?"checked":""} onchange="rsSet('mask',this.checked)"> leave out water-absorption bands (1340–1460, 1790–1960 nm)</label>
      <label>a spectrum must cover at least <select onchange="rsSet('cov',this.value)">${[0.3,0.5,0.7,0.9].map(c=>`<option value="${c}" ${+r.cov===c?"selected":""}>${c*100}%</option>`).join("")}</select> of the grid</label></div>
    <div style="display:flex;gap:8px;margin-top:14px;align-items:center;flex-wrap:wrap"><input type="text" placeholder="name of the new library" value="${escapeHtml(r.name)}" onchange="rsSet('name',this.value)" style="width:280px">
      <button class="btn blue btn-sm" onclick="rsRun()" ${r.busy?"disabled":""}>${r.busy?"Resampling…":"Resample and save as new library"}</button></div>
    ${r.err?`<p style="color:#fca5a5;font-size:12.5px;margin-top:10px">${escapeHtml(r.err)}</p>`:""}
    ${res?`<p style="margin-top:12px;font-size:13px"><b>Done:</b> “${escapeHtml(res.library)}” — ${res.resampled} spectra on ${res.bands} bands (${res.range_nm[0].toFixed(0)}–${res.range_nm[1].toFixed(0)} nm).${res.n_rejected?` ${res.n_rejected} spectra were left out:`:""}</p>
      ${res.rejected.length?`<div class="note" style="max-height:120px;overflow:auto">${res.rejected.map(x=>`${escapeHtml(x.name)} — ${escapeHtml(x.reason)}`).join("<br>")}</div>`:""}
      <button class="btn emerald btn-sm" style="margin-top:8px" onclick="setView('library-list')">Show libraries</button>`:""}
  </div>`;
}
/* DBLAYER-END */
