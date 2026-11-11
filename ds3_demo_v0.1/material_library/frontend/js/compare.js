/* compare.js — Compare spectra
   (classic script, loaded in order from index.html; all functions are global) */
/* =========================================================================
   COMPARE SPECTRA — pick spectra from any library (same or other), click to add, see all of them overlaid
   ========================================================================= */
const CMP={ items:[], libId:"", q:"", acq:"", list:[], total:0, norm:false, err:"", busy:false, angles:false, n:0 };
const CMP_COLORS=["#059669","#2563eb","#db2777","#d97706","#7c3aed","#ea580c","#0891b2","#dc2626","#4d7c0f","#be185d","#0f766e","#6d28d9","#92400e","#1d4ed8","#9d174d","#166534"];
const CMP_MAX=16;
function cmpOpen(libId,matId){
  CMP.err="";
  if(libId&&matId){                                  // coming from a spectrum page: its spectra are the first ones
    const lib=state.libraries.find(l=>l.id===libId), mat=lib&&lib.materials.find(m=>m.id===matId);
    ((mat&&mat.dbSpectra)||[]).slice(0,4).forEach((s,i)=>cmpPush({key:`m:${matId}:${i}`,name:mat.name+(mat.dbSpectra.length>1?` (${s.sample||i+1})`:""),library:lib.name,unit:s.measurement_unit,wavelengths:s.wavelengths,values:s.values}));
    CMP.libId=libId;
  }
  state.view="comparison"; render(); cmpLoadList();
}
function cmpPush(it){
  if(CMP.items.some(c=>c.key===it.key)) return false;
  if(CMP.items.length>=CMP_MAX){ CMP.err=`At most ${CMP_MAX} spectra can be compared at once. Remove some first.`; return false; }
  CMP.n++; CMP.items.push(Object.assign({color:CMP_COLORS[(CMP.n-1)%CMP_COLORS.length],show:true},it)); return true;
}
async function cmpLoadList(){
  const q=[`limit=200`]; if(CMP.libId) q.push(`library_id=${CMP.libId}`); if(CMP.q) q.push(`q=${encodeURIComponent(CMP.q)}`); if(CMP.acq) q.push(`acquisition=${encodeURIComponent(CMP.acq)}`);
  try{ const r=await apiGet("/api/spectra/search?"+q.join("&")); CMP.list=r.results; CMP.total=r.total; CMP.err=""; }catch(e){ CMP.err=e.message; CMP.list=[]; CMP.total=0; }
  cmpDrawList();
}
let _cmpT; function cmpSearch(v){ CMP.q=v; clearTimeout(_cmpT); _cmpT=setTimeout(cmpLoadList,250); }
function cmpSetLib(v){ CMP.libId=v; cmpLoadList(); }
function cmpSetAcq(v){ CMP.acq=v; cmpLoadList(); }
async function cmpToggle(sid){
  const i=CMP.items.findIndex(c=>c.key===sid);
  if(i>=0){ CMP.items.splice(i,1); CMP.err=""; cmpRefresh(); return; }
  const row=CMP.list.find(r=>r.spectrum_id===sid); if(!row) return;
  if(CMP.items.length>=CMP_MAX){ CMP.err=`At most ${CMP_MAX} spectra can be compared at once. Remove some first.`; cmpRefresh(); return; }
  try{
    const sp=specCache[sid]||(specCache[sid]=await apiGet(`/api/spectra/${sid}`));
    cmpPush({key:sid,name:row.material,library:row.library,unit:sp.measurement_unit||row.unit,wavelengths:sp.wavelengths,values:sp.values}); CMP.err="";
  }catch(e){ CMP.err=e.message; }
  cmpRefresh();
}
async function cmpAddShown(){
  for(const r of CMP.list.slice(0,CMP_MAX)){ if(CMP.items.length>=CMP_MAX) break; if(CMP.items.some(c=>c.key===r.spectrum_id)) continue; await cmpToggle(r.spectrum_id); }
}
function cmpRemove(i){ CMP.items.splice(i,1); CMP.err=""; cmpRefresh(); }
function cmpShow(i,v){ CMP.items[i].show=v; cmpDrawChart(); }
function cmpClear(){ CMP.items=[]; CMP.n=0; CMP.err=""; cmpRefresh(); }
function cmpRefresh(){ cmpDrawList(); cmpDrawChart(); cmpDrawSel(); }
function cmpSeries(){
  return CMP.items.filter(c=>c.show).map(c=>{
    let d=c.wavelengths.map((w,k)=>({x:+w,y:c.values[k]})).filter(p=>p.y!==null&&Number.isFinite(+p.y));
    if(CMP.norm){ const m=Math.max(...d.map(p=>p.y)); if(m>0) d=d.map(p=>({x:p.x,y:p.y/m})); }
    return {name:c.name,color:c.color,data:d};
  });
}
function cmpDrawChart(){
  const box=document.getElementById("cmpChart"); if(!box) return;
  const s=cmpSeries(), units=[...new Set(CMP.items.filter(c=>c.show).map(c=>c.unit||""))];
  box.innerHTML=s.length?dsChart(s,{height:350,width:760,yLabel:CMP.norm?"Reflectance (scaled to maximum)":(units.length===1&&units[0]?units[0][0].toUpperCase()+units[0].slice(1):"Value")})+(units.length>1&&!CMP.norm?`<div class="banner-info" style="margin:10px 0 0"><span>&#9888;</span><span>The spectra have different units (${escapeHtml(units.join(", "))}); the heights are not comparable. Tick <b>Scale each to its maximum</b> to compare shapes only.</span></div>`:""):`<div class="placeholder-card" style="padding:50px 10px">No spectrum selected yet.<br>Click spectra in the list on the left (from this or any other library).</div>`;
}
function cmpDrawSel(){
  const box=document.getElementById("cmpSel"); if(!box) return;
  const c=CMP.items;
  let ang="";
  if(CMP.angles&&c.length>1){
    const sh=c.slice(0,10), grid=(a,b)=>{ const wl=a.wavelengths, bw=b.wavelengths; if(wl.length===bw.length&&wl[0]===bw[0]&&wl[wl.length-1]===bw[bw.length-1]) return angleDeg(a.values,b.values); const bm=new Map(); bw.forEach((w,k)=>bm.set(Math.round(w),b.values[k])); const A=[],B=[]; wl.forEach((w,k)=>{ const y=bm.get(Math.round(w)); if(y!==undefined){ A.push(a.values[k]); B.push(y); } }); return angleDeg(A,B); };
    ang=`<div style="overflow:auto;margin-top:8px"><table style="font-size:11px;border-collapse:collapse"><tr><th style="padding:2px 6px">°</th>${sh.map((s,i)=>`<th style="padding:2px 6px">${i+1}</th>`).join("")}</tr>${sh.map((s,i)=>`<tr><th style="padding:2px 6px">${i+1}</th>${sh.map((t,j)=>{ const d=i===j?0:grid(s,t); return `<td style="padding:2px 6px;${d!=null&&d<3?"color:#047857":d>15?"color:#c62828":""}">${d==null?"–":d.toFixed(1)}</td>`; }).join("")}</tr>`).join("")}</table><p class="note">Spectral angle between the selected spectra (on shared wavelengths): small = similar shape.</p></div>`;
  }
  box.innerHTML=`<div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap"><b style="font-size:13px">Selected (${c.length}/${CMP_MAX})</b>
    <label style="font-size:12px"><input type="checkbox" ${CMP.norm?"checked":""} onchange="CMP.norm=this.checked;cmpDrawChart()"> Scale each to its maximum (shape only)</label>
    <span style="flex:1"></span>
    <button class="btn ghost btn-sm" onclick="CMP.angles=!CMP.angles;cmpDrawSel()" ${c.length>1?"":"disabled"}>Compare angles</button>
    <button class="btn ghost btn-sm" onclick="cmpCsv()" ${c.length?"":"disabled"}>Export CSV</button>
    <button class="btn ghost btn-sm" onclick="cmpClear()" ${c.length?"":"disabled"}>Clear all</button></div>
    ${CMP.err?`<p style="color:#c62828;font-size:12px;margin:6px 0 0">${escapeHtml(CMP.err)}</p>`:""}
    <div style="margin-top:6px;max-height:210px;overflow:auto">${c.map((s,i)=>`<div class="coll-row"><input type="checkbox" ${s.show?"checked":""} onchange="cmpShow(${i},this.checked)"><span class="coll-sw" style="background:${s.color}"></span><b style="width:16px;font-size:11px">${i+1}</b><span style="flex:1;min-width:0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis" title="${escapeHtml(s.name)}">${escapeHtml(s.name)} <span class="note" style="margin:0">· ${escapeHtml(s.library||"")}</span></span><button class="coll-x" title="Remove" onclick="cmpRemove(${i})">&#10005;</button></div>`).join("")||`<p class="note">Nothing selected.</p>`}</div>${ang}`;
}
function cmpDrawList(){
  const box=document.getElementById("cmpList"); if(!box) return;
  const on=new Set(CMP.items.map(c=>c.key));
  box.innerHTML=CMP.list.map(r=>{ const c=CMP.items.find(x=>x.key===r.spectrum_id);
    return `<div class="lf-item ${c?"sel":""}" onclick="cmpToggle('${r.spectrum_id}')" style="display:flex;gap:8px;align-items:center">
      <span class="coll-sw" style="background:${c?c.color:"transparent"};border:1px solid ${c?c.color:"#9aa6b6"}"></span>
      <span style="flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${escapeHtml(r.material)}</span>
      ${r.acquisition_type?`<span class="acq-chip acq-${r.acquisition_type}">${r.acquisition_type}</span>`:""}
      <span class="note" style="margin:0;white-space:nowrap">${escapeHtml(r.library)}</span></div>`; }).join("")||`<p class="note" style="padding:8px">No spectra match.</p>`;
  const t=document.getElementById("cmpTotal"); if(t) t.textContent=CMP.total>CMP.list.length?`showing ${CMP.list.length} of ${CMP.total} — narrow the search`:`${CMP.total} spectra`;
}
function cmpCsv(){
  const c=CMP.items.filter(x=>x.show); if(!c.length) return;
  const wl=[...new Set(c.flatMap(s=>s.wavelengths.map(w=>Math.round(+w*100)/100)))].sort((a,b)=>a-b);
  const maps=c.map(s=>{ const m=new Map(); s.wavelengths.forEach((w,k)=>{ if(s.values[k]!==null) m.set(Math.round(+w*100)/100,s.values[k]); }); return m; });
  csvDownload("compared_spectra.csv",["wavelength_nm,"+c.map(s=>`"${(s.name+" | "+s.library).replace(/"/g,'""')}"`).join(","),...wl.map(w=>[w,...maps.map(m=>m.has(w)?m.get(w):"")].join(","))]);
}
function renderComparison(){
  const libs=state.libraries.filter(l=>l.db);
  setTimeout(()=>{ cmpRefresh(); if(!CMP.list.length&&!CMP._loaded){ CMP._loaded=true; cmpLoadList(); } },0);
  return `<h1 class="pagetitle">Compare spectra</h1>
  <p class="pagesub">Click spectra in the list — from this library or any other — and they are drawn together so you can compare them visually.</p>
  <div style="display:grid;grid-template-columns:360px minmax(0,1fr);gap:16px;align-items:start">
    <div class="card" style="padding:12px">
      <label class="small">Library</label>
      <select onchange="cmpSetLib(this.value)" style="width:100%"><option value="">All libraries</option>${libs.map(l=>`<option value="${l.id}" ${CMP.libId===l.id?"selected":""}>${escapeHtml(l.name)} (${l.spectra})${l.main?"":" – built"}</option>`).join("")}</select>
      <input type="text" placeholder="search by material, category or library" value="${escapeHtml(CMP.q)}" oninput="cmpSearch(this.value)" style="width:100%;margin-top:8px">
      <select onchange="cmpSetAcq(this.value)" style="width:100%;margin-top:8px"><option value="">any acquisition</option>${["lab","field","airborne","satellite"].map(a=>`<option ${CMP.acq===a?"selected":""}>${a}</option>`).join("")}</select>
      <div class="lf-list" id="cmpList" style="height:380px;margin-top:8px"></div>
      <div style="display:flex;align-items:center;gap:8px;margin-top:6px"><span class="note" id="cmpTotal" style="margin:0"></span><span style="flex:1"></span><button class="btn ghost btn-sm" onclick="cmpAddShown()">Add the first ${CMP_MAX} shown</button></div>
    </div>
    <div>
      <div class="chart-wrap"><div id="cmpChart"></div></div>
      <div class="card" style="padding:12px;margin-top:12px" id="cmpSel"></div>
    </div>
  </div>`;
}
