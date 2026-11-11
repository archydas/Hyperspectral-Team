/* analysis-windows.js — Analysis windows (Spectral menu)
   (classic script, loaded in order from index.html; all functions are global) */
/* ---------- analysis windows (Spectral menu) ---------- */
const AN={ op:null, last:null, busy:false };
const AN_TITLES={ pca:"Principal Components (PCA)", kmeans:"K-Means Clustering", sam:"Spectral Angle Mapper", unmix:"Linear Spectral Unmixing",
  bandmath:"Band Math / Spectral Index", anomaly:"Anomaly Detection (RX)", snr:"Signal-to-Noise Ratio", analyst:"Spectral Analyst" };
function endmemberSource(){
  const libs=dbLibs();
  return `<h4>Spectra to use</h4>
    <div class="box" style="max-height:130px;overflow:auto">${I.coll.length?I.coll.map((s,i)=>`<label style="display:flex;gap:6px;align-items:center;padding:1px 0"><input type="checkbox" class="anC" value="${i}" checked><span class="coll-sw" style="background:${s.color}"></span>${escapeHtml(s.name)} <span class="note" style="margin:0">${s.kind==="roi"?s.n+" px":`${s.row}, ${s.col}`}</span></label>`).join(""):`<span class="note">No spectra collected yet — click pixels on the image and press Add in the small window, or use a library below.</span>`}</div>
    <div class="row"><label>or all spectra (max 60) of library</label><select id="anLib" style="max-width:240px"><option value="">— none —</option>${libs.map(l=>`<option value="${l.id}">${escapeHtml(l.name)} (${l.spectra})</option>`).join("")}</select></div>`;
}
function anOpen(op,o={}){
  if(!needImage()){ return; }
  AN.op=op; const a=I.active; let body="";
  const range=`${a.wavelength_min.toFixed(0)}–${a.wavelength_max.toFixed(0)} nm, ${a.bands} bands`;
  if(op==="pca") body=`<div class="row"><label>Number of components</label><select id="anN">${[3,4,5,6,8,10].map(n=>`<option>${n}</option>`).join("")}</select></div><p class="note">Shows PC1/PC2/PC3 as an RGB image, the scree plot and the first components.</p>`;
  else if(op==="kmeans") body=`<div class="row"><label>Number of classes</label><input id="anK" type="number" min="2" max="20" value="6" style="width:70px"><label>Iterations</label><input id="anIt" type="number" min="3" max="100" value="20" style="width:70px"></div><p class="note">Unsupervised clustering of all good bands. The class mean spectra can be saved to a library.</p>`;
  else if(op==="sam") body=`${endmemberSource()}<div class="row"><label>Maximum angle (°, 0 = none)</label><input id="anAng" type="number" min="0" max="90" value="0" style="width:70px"></div>`;
  else if(op==="unmix") body=`${endmemberSource()}<div class="row"><label><input id="anS1" type="checkbox" checked> sum-to-one constraint</label></div><p class="note">Non-negative least squares per pixel (decimated image). One abundance map per endmember, plus the dominant endmember and the fit error.</p>`;
  else if(op==="bandmath") body=`<label class="small">Expression — <b>b</b>N = band number (1–${a.bands}), <b>w</b>NNN = band nearest to that wavelength in nm (${range})</label>
    <textarea id="anExpr" rows="3" style="width:100%">${escapeHtml(o.expression||"(w800 - w670) / (w800 + w670)")}</textarea>
    <div class="row"><label>Preset</label><select onchange="if(this.value)document.getElementById('anExpr').value=this.value"><option value="">—</option>
      <option value="(w800 - w670) / (w800 + w670)">NDVI</option><option value="(w790 - w720) / (w790 + w720)">NDRE</option><option value="(w860 - w1240) / (w860 + w1240)">NDWI</option><option value="w2200 / w2100">Clay ratio</option><option value="mean(w450, w550, w650)">Visible mean</option></select></div>
    <p class="note">Operators + - * / **, functions sqrt, log, log10, exp, abs, min, max, mean, where. The bands actually used are listed with the result.</p>`;
  else if(op==="anomaly") body=`<div class="row"><label>PCA components used</label><input id="anComp" type="number" min="2" max="50" value="15" style="width:70px"></div><p class="note">Global RX: Mahalanobis distance of every pixel from the scene background (in the leading principal components).</p>`;
  else if(op==="snr") body=`<p class="note">Nearest-neighbour estimate: the signal is the mean spectrum, the noise is the standard deviation of the difference between neighbouring pixels. Uses the first rows of the scene.</p>`;
  else if(op==="analyst"){ analystOpen(); return; }
  win(AN_TITLES[op],`<p class="note" style="margin:0 0 4px">Image: <b>${escapeHtml(a.name)}</b> · ${a.cols}×${a.rows}×${a.bands}</p>${body}
    <div class="row"><button class="btn blue btn-sm" id="anGo" onclick="anRun()">Run</button><button class="btn ghost btn-sm" onclick="winClose()">Cancel</button></div><div class="err" id="anErr"></div>`,{width:op==="bandmath"?560:480});
  if(o.run) anRun();
}
function collEndmembers(){ return Array.from(document.querySelectorAll(".anC:checked")).map(e=>I.coll[+e.value]).map(s=>({name:s.name,values:s.values})); }
async function anRun(){
  if(AN.busy) return; const op=AN.op, g=id=>document.getElementById(id); let params={}, body={};
  try{
    if(op==="pca") params={n_components:+g("anN").value};
    else if(op==="kmeans") params={k:+g("anK").value,iterations:+g("anIt").value};
    else if(op==="sam"||op==="unmix"){
      params.endmembers=collEndmembers(); const lib=g("anLib").value; if(lib) body.library_id=lib;
      if(!params.endmembers.length&&!lib) throw new Error("Choose at least one spectrum or a library.");
      if(op==="sam") params.max_angle_deg=+g("anAng").value||null; else params.sum_to_one=g("anS1").checked;
    }
    else if(op==="bandmath") params={expression:g("anExpr").value};
    else if(op==="anomaly") params={components:+g("anComp").value};
  }catch(e){ const x=g("anErr"); if(x) x.textContent=e.message; return; }
  AN.busy=true; const go=g("anGo"); if(go){ go.disabled=true; go.textContent="Running…"; }
  const err=document.getElementById("anErr"); if(err) err.textContent="";
  try{
    const r=await apiSend("POST",`/api/images/${I.active.id}/analyze/${op}`,Object.assign({params},body));
    AN.busy=false; AN.last=r; anShow(r);
  }catch(e){ AN.busy=false; const x=g("anErr"), b=g("anGo"); if(x) x.textContent=e.message; else win(AN_TITLES[op],`<div class="err">${escapeHtml(e.message)}</div>`); if(b){ b.disabled=false; b.textContent="Run"; } }
}
const CMAPS={viridis:"#440154,#3b528b,#21918c,#5ec962,#fde725",RdYlGn:"#a50026,#f46d43,#ffffbf,#a6d96a,#006837",inferno:"#000004,#57106e,#bc3754,#f98e09,#fcffa4",magma_r:"#fcfdbf,#fc8961,#b73779,#51127c,#000004",gray:"#000,#fff"};
function fmtNum(v){ if(v==null) return "–"; if(typeof v!=="number") return escapeHtml(v); if(Number.isInteger(v)) return v.toLocaleString(); const a=Math.abs(v); return a>=1000||(a>0&&a<0.01)?v.toExponential(2):v.toFixed(3); }
function anShow(r){
  let body="";
  const imgBox=(src,w)=>`<div style="background:#222;border:1px solid #6d7b8e;display:inline-block;max-width:100%;line-height:0"><img src="${src}" style="width:${Math.min(560,Math.max(320,r.width||320))}px;max-width:100%;max-height:55vh;object-fit:contain;image-rendering:${r.width&&r.width<400?"pixelated":"auto"}"></div>`;
  if(r.image) body+=imgBox(r.image);
  if(r.kind==="scalar") body+=`<div style="max-width:420px;margin-top:6px"><div class="cbar" style="background:linear-gradient(90deg,${CMAPS[r.cmap]||CMAPS.viridis})"></div><div style="display:flex;justify-content:space-between;font-size:11px"><span>${fmtNum(r.range[0])}</span><span>${fmtNum(r.range[1])}</span></div></div>`;
  if(r.kind==="classes") body+=`<div style="display:flex;flex-wrap:wrap;gap:3px 14px;margin-top:8px;font-size:12px">${r.legend.map(l=>`<span><span style="display:inline-block;width:11px;height:11px;background:${l.color};margin-right:5px;vertical-align:-1px"></span>${escapeHtml(l.name)} <span class="note" style="margin:0">${l.pixels.toLocaleString()} px</span></span>`).join("")}${r.unmatched?`<span class="note" style="margin:0">unclassified ${r.unmatched.toLocaleString()} px</span>`:""}</div>`;
  if(r.plot&&r.plot.series&&r.plot.series.length){ const pal=["#2563eb","#dc2626","#059669"]; body+=`<h4>${escapeHtml(r.plot.title||"")}</h4><div class="box">${dsChart(r.plot.series.map((s,i)=>({name:s.name,color:pal[i%3],data:r.plot.x.map((x,k)=>({x,y:s.y[k]})).filter(d=>d.y!==null)})),{height:210,width:520,xLabel:r.plot.xlabel||"Wavelength (nm)",yLabel:r.plot.ylabel||"Value"})}<div class="note">x: ${escapeHtml(r.plot.xlabel||"")} · y: ${escapeHtml(r.plot.ylabel||"")}</div></div>`; }
  if(r.table&&Object.keys(r.table).length) body+=`<h4>Statistics</h4><table>${Object.entries(r.table).map(([k,v])=>`<tr><td>${escapeHtml(k)}</td><td>${fmtNum(v)}</td></tr>`).join("")}</table>`;
  if(r.extra&&r.extra.length) body+=`<h4>Further images</h4><div class="thumbs">${r.extra.map((e,i)=>`<figure onclick="anBig(${i})" title="Click to enlarge"><img src="${e.image}"><figcaption>${escapeHtml(e.name)}${e.range?` · ${fmtNum(e.range[0])} … ${fmtNum(e.range[1])}`:""}</figcaption></figure>`).join("")}</div>`;
  body+=`<div class="row" style="margin-top:12px">${r.kind==="classes"?`<button class="btn emerald btn-sm" onclick="anOverlay()">Overlay on the image</button>`:""}
    ${r.spectra&&r.spectra.length?`<button class="btn ghost btn-sm" onclick="anSpectraToColl()">Add ${r.spectra.length} spectra to the collected list</button><button class="btn ghost btn-sm" onclick="anSpectraSave()">Save as library…</button>`:""}
    ${r.image?`<button class="btn ghost btn-sm" onclick="anPng()">Save PNG</button>`:""}<button class="btn ghost btn-sm" onclick="winClose()">Close</button></div>
    <p class="note">Computed on a decimated copy of the scene (step ${r.step}); the full image is not modified.</p>`;
  win(r.title,body,{width:Math.min(760,Math.max(520,(r.width||0)+60))});
}
function anBig(i){ const e=AN.last.extra[i]; win(e.name,`<div style="background:#222;line-height:0"><img src="${e.image}" style="max-width:100%;max-height:70vh;image-rendering:pixelated"></div><div class="row"><button class="btn ghost btn-sm" onclick="anShow(AN.last)">Back</button></div>`,{width:720}); }
function anPng(){ const a=document.createElement("a"); a.href=AN.last.image; a.download=(AN.last.title||"result").replace(/[^\w.-]+/g,"_")+".png"; a.click(); }
function anOverlay(){
  const r=AN.last; I.overlay=r.image; I.legend={legend:r.legend}; winClose(); if(state.view!=="image") state.view="image"; render(); refreshAll();
}
function anSpectraToColl(){
  (AN.last.spectra||[]).forEach(s=>{ if(I.coll.length>=24) return; I.collN++; I.coll.push({key:"an:"+I.collN,color:COLL_COLORS[(I.collN-1)%COLL_COLORS.length],show:true,kind:"px",row:-1,col:-1,n:0,unit:s.unit,wavelengths:I.wl,values:s.values,name:s.name}); });
  winClose(); state.view="image"; render(); refreshAll();
}
function anSpectraSave(){
  const sp=AN.last.spectra; I.coll.length;
  win("Save spectra as library",`<label class="small">Library name</label><input id="cbLib" type="text" value="${escapeHtml(I.active.name)} ${escapeHtml(AN.last.title)}" style="width:100%">
    <div class="row"><div style="flex:1"><label class="small">Category (optional)</label><input id="cbCat" type="text" style="width:100%"></div><div><label class="small">Acquisition</label><select id="cbAcq"><option value="">—</option><option>airborne</option><option>satellite</option><option>field</option><option>lab</option></select></div></div>
    <div class="row"><button class="btn emerald btn-sm" onclick="anSpectraDo()">Save</button><button class="btn ghost btn-sm" onclick="anShow(AN.last)">Back</button></div><div class="err" id="cbErr"></div>`,{width:500});
}
async function anSpectraDo(){
  const lib=document.getElementById("cbLib").value.trim(), cat=document.getElementById("cbCat").value.trim()||null, acq=document.getElementById("cbAcq").value||null;
  try{ const r=await apiSend("POST","/api/spectra/save",{library:lib,items:AN.last.spectra.map(s=>({name:s.name,wavelengths_nm:I.wl,values:s.values,unit:s.unit,category:cat,origin:"image_pixel",acquisition_type:acq}))});
    await loadLibraries(); winClose(); showToast(`Saved ${r.saved} spectra to library "${r.library}"`); }catch(e){ document.getElementById("cbErr").textContent=e.message; }
}

/* Spectral Analyst: rank library spectra by spectral angle against the current pixel / ROI */
function analystOpen(){
  const t=target(), libs=dbLibs();
  if(!t&&!I.coll.length){ showToast("Click a pixel (or draw an ROI) first"); return; }
  win("Spectral Analyst",`<p class="note" style="margin:0 0 6px">Unknown spectrum: ${I.roi.closed?"ROI mean":I.pixel?`pixel (${I.pixel.row}, ${I.pixel.col})`:"collected spectrum"} — compared with every spectrum of the library by spectral angle.</p>
    <div class="row"><label>Library</label><select id="anLib" style="max-width:260px"><option value="">— choose —</option>${libs.map(l=>`<option value="${l.id}">${escapeHtml(l.name)} (${l.spectra})</option>`).join("")}</select></div>
    <div class="row"><label><input id="anIgn" type="checkbox" checked> ignore unit differences (compare shape only)</label></div>
    <div class="row"><button class="btn blue btn-sm" id="anGo" onclick="analystRun()">Run</button><button class="btn ghost btn-sm" onclick="winClose()">Cancel</button></div><div class="err" id="anErr"></div>`,{width:500});
}
async function analystRun(){
  const lib=document.getElementById("anLib").value, err=document.getElementById("anErr"); if(!lib){ err.textContent="Choose a library."; return; }
  let t=target(); if(!t){ const c=I.coll[I.coll.length-1]; t={row:c.row,col:c.col,size:1}; }
  try{
    const r=await apiSend("POST",`/api/images/${I.active.id}/match`,{...t,library_id:lib,ignore_unit:document.getElementById("anIgn").checked,top_k:15});
    await addLibraryToImage(lib);
    const rows=r.results.map((m,i)=>`<tr><td>${i+1}</td><td>${escapeHtml(m.material)}</td><td>${escapeHtml(m.category||"")}</td><td style="text-align:right">${m.angle_deg.toFixed(2)}°</td></tr>`).join("");
    I.matches=r.results; if(r.results.length) I.picked=new Set(r.results.slice(0,3).map(x=>x.spectrum_id));
    win("Spectral Analyst — best matches",`${r.results.length?`<table style="width:100%"><tr><th>#</th><th>Material</th><th>Category</th><th style="text-align:right">Angle</th></tr>${rows}</table>`:`<p class="note">No match.</p>`}${r.note?`<p class="note">${escapeHtml(r.note)}</p>`:""}
      <div class="row"><button class="btn emerald btn-sm" onclick="winClose();state.view='image';render();refreshAll()">Show best 3 in the plot</button><button class="btn ghost btn-sm" onclick="winClose()">Close</button></div>`,{width:560});
  }catch(e){ err.textContent=e.message; }
}

/* ---------- the Analysis page (placeholder: AI / ML models will be built here) ---------- */
function renderAnalysis(){
  return `<h1 class="pagetitle">Analysis</h1>
  <p class="pagesub">AI / ML models will be built here (classification and detection trained on the libraries and the object samples). Data aggregation now sits on the <b>Image</b> page, below the image.</p>
  <div class="card" style="padding:16px"><button class="btn emerald btn-sm" onclick="setView('image')">Go to Image</button> <button class="btn ghost btn-sm" onclick="aggGo()">Go to ROI aggregation</button></div>`;
}
