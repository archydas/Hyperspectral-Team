/* image-spectra.js — Image page: clicking a pixel (popup, label, Add), ROI drawing, collected spectra, save to library, spectrum chart
   (classic script, loaded in order from index.html; all functions are global) */
document.addEventListener("keydown",e=>{ if(e.key==="Escape"&&I.pop) popClose(); });

I.collect=false; I.show={px:true,roi:true}; I.arm={c:false,o:false}; I.edit=null; I.pop=null; I.lastClick=null; I.panned=false;

/* ---------- collected pixel spectra ---------- */
function collAdd(s,kind,label){
  if(!s||!s.values) return false;
  const key=kind==="roi"?"roi:"+(I.roi.pts||[]).map(p=>p.join(",")).join(";"):`px:${I.active?I.active.id:""}:${s.row},${s.col},${s.size||1}`;
  if(I.coll.some(c=>c.key===key)){ showToast("This spectrum is already in the list"); return false; }
  if(I.coll.length>=60){ showToast("60 spectra collected: delete some first"); return false; }
  I.collN++; const color=COLL_COLORS[(I.collN-1)%COLL_COLORS.length];
  I.coll.push({key,color,show:true,kind,row:s.row,col:s.col,n:s.n_pixels||1,unit:s.unit,wavelengths:s.wavelengths,values:s.values,
    img:I.active?{id:I.active.id,name:I.active.name,path:I.active.path||""}:null,
    name:(label||"").trim()||(kind==="roi"?`ROI ${I.collN}`:`Pixel ${I.collN}`)});
  return true;
}
function collMarkers(){
  const a=I.active; if(!a||!I.show.px) return "";
  return I.coll.map((c,i)=>c.kind==="px"&&c.show&&c.row>=0&&(!c.img||c.img.id===a.id)?`<div class="pxmark" style="left:${(c.col+0.5)/a.cols*100}%;top:${(c.row+0.5)/a.rows*100}%;border-color:${c.color};background:${c.color}66">${i+1}</div>`:"").join("");
}

/* ---------- clicking on the image ---------- */
async function onImageClick(ev){
  if(!I.preview||I.panned) return; const p=imgCoords(ev); I.lastClick={x:ev.clientX,y:ev.clientY};
  if(I.roi.drawing){ const l=I.roi.pts[I.roi.pts.length-1]; if(!l||l[0]!==p.row||l[1]!==p.col) I.roi.pts.push([p.row,p.col]); refreshRoiLayer(); return; }
  if(OBJ.menu){ objMenuClose(); return; }
  if(!I.show.px){ showToast("Tick “Pixel spectra” to pick pixels"); return; }
  popClose(); await pickPixel(p.row,p.col);
}
function onImageDbl(ev){ if(I.panned||!I.roi.drawing) return; ev.preventDefault(); finishRoi(); }
function onImageCtx(ev){ if(!I.roi.drawing) return; ev.preventDefault(); finishRoi(); }
async function pickPixel(row,col){
  I.win=parseInt((document.getElementById("imgWin")||{value:I.win}).value,10)||1;
  let ok=false;
  try{
    const p=await apiGet(`/api/images/${I.active.id}/pixel?row=${row}&col=${col}&size=${I.win}`);
    I.pixel=p; I.spec={...p,kind:"pixel"}; I.matches=null; I.err=""; ok=true;
  }catch(e){ I.err=e.message; }
  refreshView(); refreshSide(); updateMapMarker();
  if(ok) popOpen("px");
}
function startRoi(){ popClose(); objMenuClose(); I.roi={pts:[],closed:false,drawing:true}; I.mapOn=false; I.show.roi=true; refreshAll(); showToast("Click to add points. Double-click or right-click to finish the ROI."); }
async function finishRoi(){
  if(I.roi.pts.length<3){ showToast("Click at least 3 points, then double-click or right-click to finish"); return; }
  I.roi.drawing=false; I.roi.closed=true;
  try{ const r=await apiSend("POST",`/api/images/${I.active.id}/roi`,{polygon:I.roi.pts}); I.spec={...r,kind:"roi"}; I.matches=null; I.err=""; refreshAll(); popOpen("roi"); return; }
  catch(e){ I.err=e.message; I.roi.closed=false; }
  refreshAll();
}
function clearRoi(){ popClose(); I.roi={pts:[],closed:false,drawing:false}; if(I.spec&&I.spec.kind==="roi") I.spec=I.pixel?{...I.pixel,kind:"pixel"}:null; refreshAll(); }

/* ---------- the small window after a click: spectrum, label, Add ---------- */
function popClose(){ I.pop=null; const h=document.getElementById("pxPop"); if(h) h.innerHTML=""; }
function popOpen(kind){
  const s=I.spec; if(!s||!I.active) return;
  I.pop={kind};
  let h=document.getElementById("pxPop"); if(!h){ h=document.createElement("div"); h.id="pxPop"; document.body.appendChild(h); }
  const isRoi=kind==="roi";
  const title=isRoi?`ROI mean · ${s.n_pixels} pixels`:`Pixel (${s.row}, ${s.col})${s.n_pixels>1?` · mean of ${s.n_pixels} px`:""}`;
  const ll=(!isRoi&&I.pixel&&typeof pixelToLatLon==="function")?pixelToLatLon(I.pixel.row,I.pixel.col):null;
  const defName=isRoi?`ROI ${objs().length+1}`:`Pixel ${I.collN+1}`;
  const classes=[...new Set(objs().map(o=>o.class_name))];
  const key=`onkeydown="if(event.key==='Enter')popAdd();if(event.key==='Escape')popClose()"`;
  h.innerHTML=`<div class="pxpop" id="pxPopBox"><div class="pxpop-t"><b>${escapeHtml(title)}</b><button onclick="popClose()" title="Close">&#10005;</button></div>
    ${ll?`<div class="note" style="margin:0 0 4px">${ll.lat.toFixed(5)}, ${ll.lon.toFixed(5)}</div>`:""}
    ${dsChart([{name:"s",color:isRoi?"#d97706":"#059669",data:s.wavelengths.map((w,i)=>({x:w,y:s.values[i]})).filter(d=>d.y!==null)}],{height:150,width:310,unit:s.unit})}
    <label class="small">Label</label><input id="pxLabel" type="text" value="${escapeHtml(defName)}" ${key}>
    ${isRoi?`<label class="small">Class for aggregation (same class = pooled)</label><input id="pxClass" type="text" list="pxClasses" placeholder="e.g. Water, Vegetation (default: the label)" ${key}><datalist id="pxClasses">${classes.map(c=>`<option value="${escapeHtml(c)}">`).join("")}</datalist>`:""}
    <div style="display:flex;gap:6px;margin-top:10px"><button class="btn emerald btn-sm" onclick="popAdd()">Add</button><button class="btn ghost btn-sm" onclick="popClose()">Cancel</button>
      <span class="note" style="margin:0 0 0 auto;align-self:center">${isRoi?"goes to Collected object samples":"goes to Collected spectra"}</span></div></div>`;
  const b=document.getElementById("pxPopBox"), w=330, hh=isRoi?390:330, lc=I.lastClick||{x:innerWidth/2-w/2,y:innerHeight/4};
  b.style.left=Math.max(8,Math.min(lc.x+16,innerWidth-w-12))+"px"; b.style.top=Math.max(8,Math.min(lc.y+16,innerHeight-hh-12))+"px";
  const inp=document.getElementById("pxLabel"); if(inp){ inp.focus(); inp.select(); }
}
function popAdd(){
  const p=I.pop, s=I.spec; if(!p||!s) return;
  const label=((document.getElementById("pxLabel")||{}).value||"").trim();
  if(p.kind==="px"){ const ok=collAdd(s,"px",label); popClose(); if(ok){ refreshView(); refreshSide(); showToast(`"${label||"Pixel"}" added to Collected spectra`); } }
  else { const cls=((document.getElementById("pxClass")||{}).value||"").trim(); popClose(); objAdd(s,label,cls); }
}

/* select all / unselect all, click a name to view only that spectrum, rename, delete all */
function collAll(v){ I.coll.forEach(c=>c.show=v); if(v) I.show.px=true; refreshView(); refreshSide(); }
function listSolo(k,i){
  I.coll.forEach((c,j)=>c.show=(k==="c"&&j===i)); objs().forEach((o,j)=>o.show=(k==="o"&&j===i));
  if(k==="c") I.show.px=true; else I.show.roi=true; agStore(); refreshView(); refreshSide();
}
const collSolo=listSolo;
function rowEdit(k,i){ I.edit={k,i}; refreshSide(); setTimeout(()=>{ const e=document.getElementById("rowEd1"); if(e){ e.focus(); e.select(); } },0); }
function rowEditOk(){
  const e=I.edit; if(!e) return; const a=document.getElementById("rowEd1"), b=document.getElementById("rowEd2");
  const v=((a&&a.value)||"").trim();
  if(e.k==="c"){ if(v&&I.coll[e.i]) I.coll[e.i].name=v; }
  else { const o=objs()[e.i]; if(o){ if(v) o.name=v; if(b){ const c=(b.value||"").trim(); o.class_name=c||o.name; } agStore(); AG.result=null; } }
  I.edit=null; refreshSide(); agRefresh();
}
function delAll(k){                                   // two clicks: the first arms the button (3.5 s), the second deletes
  const redraw=()=>{ if(state.view!=="image") return; if(k==="o") objRefresh(); else refreshSide(); };
  I.armT=I.armT||{}; clearTimeout(I.armT[k]);
  if(!I.arm[k]){ I.arm[k]=true; redraw(); I.armT[k]=setTimeout(()=>{ I.arm[k]=false; redraw(); },3500); return; }
  I.arm[k]=false; if(k==="c") collClear(); else objClear();
}

function rowName(k,i,s,extra){
  if(I.edit&&I.edit.k===k&&I.edit.i===i){
    const kd=`onkeydown="if(event.key==='Enter')rowEditOk();if(event.key==='Escape'){I.edit=null;refreshSide()}"`;
    return `<input id="rowEd1" type="text" value="${escapeHtml(s.name)}" ${kd} title="Label">${k==="o"?`<input id="rowEd2" type="text" value="${escapeHtml(s.class_name||"")}" placeholder="class" style="max-width:80px" ${kd} title="Class (pooled in the aggregation)">`:""}<button class="coll-x" title="Done" onclick="rowEditOk()">&#10003;</button>`;
  }
  return `<span class="coll-nm" title="Click to view only this spectrum" onclick="listSolo('${k}',${i})">${escapeHtml(s.name)}</span><span class="note" style="margin:0;white-space:nowrap">${extra}</span><button class="coll-x" title="Rename" onclick="rowEdit('${k}',${i})">&#9998;</button>`;
}
function collPanel(){
  const c=I.coll;
  const rows=c.map((s,i)=>`<div class="coll-row"><input type="checkbox" ${s.show?"checked":""} onchange="collShow(${i},this.checked)"><span class="coll-sw" style="background:${s.color}"></span><b style="width:14px;font-size:11px">${i+1}</b>
    ${rowName("c",i,s,s.kind==="roi"?`${s.n} px`:(s.row>=0?`${s.row}, ${s.col}`:""))}<button class="coll-x" title="Remove" onclick="collRemove(${i})">&#10005;</button></div>`).join("");
  let ang="";
  if(I.angles&&c.length>1){
    const sh=c.slice(0,10);
    ang=`<div style="overflow:auto;margin-top:8px"><table style="font-size:11px"><tr><th>°</th>${sh.map((s,i)=>`<th>${i+1}</th>`).join("")}</tr>${sh.map((s,i)=>`<tr><th>${i+1}</th>${sh.map((t,j)=>{ const d=i===j?0:angleDeg(s.values,t.values); return `<td style="${d!=null&&d<2?"color:#047857":d>10?"color:#c62828":""}">${d==null?"–":d.toFixed(1)}</td>`; }).join("")}</tr>`).join("")}</table><p class="note">Spectral angle between collected spectra: small = similar shape.</p></div>`;
  }
  return `<div class="chart-wrap coll-card" style="margin-top:12px">
    <div style="display:flex;align-items:center;gap:6px;flex-wrap:wrap"><b style="font-size:12.5px">Collected spectra (${c.length})</b>
      <span style="margin-left:auto;display:flex;gap:4px"><button class="btn ghost btn-sm" onclick="collAll(true)" ${c.length?"":"disabled"}>Select all</button><button class="btn ghost btn-sm" onclick="collAll(false)" ${c.length?"":"disabled"}>Unselect all</button></span></div>
    ${c.length?`<div class="coll-list" style="margin-top:6px">${rows}</div><p class="note" style="margin:4px 0 0">Tick to show in the chart · click a name to view only that spectrum.</p>`:`<p class="note">Click a pixel on the image: a small window shows its spectrum. Give it a label and press <b>Add</b>, and it is listed here.</p>`}
    ${ang}
    <div style="display:flex;gap:6px;flex-wrap:wrap;margin-top:8px;align-items:center">
      <button class="btn emerald btn-sm" onclick="saveDialog('c')" ${c.length?"":"disabled"}>Save to library…</button>
      <button class="btn ghost btn-sm" onclick="I.angles=!I.angles;refreshSide()" ${c.length>1?"":"disabled"}>Compare angles</button>
      <button class="btn ghost btn-sm" onclick="collCsv()" ${c.length?"":"disabled"}>Export CSV</button>
      <button class="btn ghost btn-sm" onclick="anOpen('sam')" ${c.length?"":"disabled"} title="Use the collected spectra as endmembers">Classify (SAM)…</button>
      <button class="btn danger btn-sm ${I.arm.c?"armed":""}" style="margin-left:auto" onclick="delAll('c')" ${c.length?"":"disabled"}>${I.arm.c?"Click again to delete all":"Delete all"}</button></div></div>`;
}

/* ---------- save collected spectra / object samples to a new or an existing library ---------- */
function saveDialog(k){
  const items=k==="c"?I.coll:k==="a"?(AG.result?AG.result.classes:[]):objs().filter(o=>o.values); if(!items.length){ showToast("Nothing to save"); return; }
  const nm=`${(I.active&&I.active.name)||"Image"} ${k==="c"?"spectra":k==="a"?"class means":"objects"}`;
  win(k==="c"?"Save collected spectra to a library":k==="a"?"Save class mean spectra to a library":"Save object samples to a library",
   `<p class="note" style="margin:0 0 8px">${items.length} ${k==="c"?"spectra":k==="a"?"class mean spectra":"object samples (ROI mean spectra)"} will be stored in the database.${k==="o"?" Each sample keeps the name of its patch file (<class name>.tif), the folder where that file was saved, the ROI polygon and the source image; the patch file itself stays in its folder.":""}</p>
    <label class="small">Library</label>
    <div style="display:flex;gap:6px;flex-wrap:wrap">${libPickHtml("cb",nm,k==="o"?"object":"pixel")}</div>
    <div class="row"><div style="flex:1"><label class="small">Category (optional, e.g. Vegetation/Trees)</label><input id="cbCat" type="text" style="width:100%"></div>
    <div><label class="small">Acquisition</label><select id="cbAcq"><option value="">—</option><option>airborne</option><option>satellite</option><option>field</option><option>lab</option></select></div></div>
    <div class="row"><button class="btn emerald btn-sm" onclick="saveDo('${k}')">Save to library</button><button class="btn ghost btn-sm" onclick="winClose()">Cancel</button></div><div class="err" id="cbErr"></div>`,{width:520});
}
async function saveDo(k){
  const ex=libPickExisting("cb"), lib=libPickValue("cb");
  if(!lib){ document.getElementById("cbErr").textContent="Give the library a name or choose an existing one."; return; }
  const cat=document.getElementById("cbCat").value.trim()||null, acq=document.getElementById("cbAcq").value||null;
  if(k==="a"){ AG.lib=lib; AG.cat=cat||""; AG.acq=acq||""; await agRun(true); if(AG.err) document.getElementById("cbErr")&&(document.getElementById("cbErr").textContent=AG.err); else winClose(); return; }
  let items, info=null;
  if(k==="c"){
    items=I.coll.map(s=>{ const im=s.img||(I.active?{id:I.active.id,name:I.active.name,path:I.active.path||""}:{});
      return {name:s.name,wavelengths_nm:s.wavelengths,values:s.values,unit:s.unit,category:cat,origin:"image_pixel",acquisition_type:acq,
        sample_metadata:{pixel:{image_id:im.id,image_name:im.name,image_path:im.path,row:s.row,col:s.col}}}; });
  } else {
    items=objs().filter(o=>o.values).map(o=>({name:o.name,wavelengths_nm:o.wavelengths,values:o.values,unit:o.unit,category:cat,origin:"image_roi",acquisition_type:acq,
      sample_metadata:{object:{kind:"roi",class_name:o.class_name,image_id:o.image_id,image_name:o.image_name,image_path:o.image_path,polygon:o.polygon,n_pixels:o.n,
        lat:o.lat??null,lon:o.lon??null,patch_file:o.patch?o.patch.file:null,patch_path:o.patch?o.patch.path:null}}}));
    if(!ex) info={type:"object",description:"Object (ROI) library: each entry is the mean spectrum of an ROI. The ROI image patch is a GeoTIFF file in a folder on disk; the database keeps its file name and address."};
  }
  try{
    const body={library:lib,items}; if(info) body.library_info=info;
    const r=await apiSend("POST","/api/spectra/save",body);
    await loadLibraries(); renderMenubar(); winClose(); showToast(`Saved ${r.saved} ${k==="c"?"spectra":"object samples"} to library "${r.library}"`+((r.rejected||[]).length?` (${r.rejected.length} rejected)`:""));
  }catch(e){ document.getElementById("cbErr").textContent=e.message; }
}

/* ---------- the chart: pixel spectra, ROI (object) spectra, or both ---------- */
async function drawSpectrumChart(){
  const box=document.getElementById("imgChart"); if(!box) return;
  const s=I.spec, series=[], names=[]; let unit=null;
  const toData=(wl,v)=>wl.map((w,k)=>({x:w,y:v[k]})).filter(d=>d.y!==null&&d.y!==undefined);
  if(I.show.px) I.coll.filter(c=>c.show).forEach(c=>{ series.push({name:c.name,color:c.color,data:toData(c.wavelengths,c.values)}); names.push(`<span style="color:${c.color}">&#9632; ${escapeHtml(c.name)}</span>`); unit=unit||c.unit; });
  if(I.show.roi) objs().filter(o=>o.show&&o.values).forEach(o=>{ series.push({name:o.name,color:o.color,data:toData(o.wavelengths,o.values)}); names.push(`<span style="color:${o.color}">&#9670; ${escapeHtml(o.name)} <i>(ROI)</i></span>`); unit=unit||o.unit; });
  const cur=s&&(s.kind==="roi"?I.show.roi:I.show.px);
  const inColl=s&&(s.kind==="roi"?objs().some(o=>o.image_id===(I.active||{}).id&&samePoly(o.polygon,I.roi.pts)):I.coll.some(c=>c.show&&c.row===s.row&&c.col===s.col&&(!c.img||c.img.id===(I.active||{}).id)));
  if(cur&&!inColl){ series.unshift({name:"current",color:"#1b2430",data:toData(s.wavelengths,s.values)}); names.unshift(`<span style="color:#1b2430">&#9632; ${s.kind==="roi"?`ROI mean (${s.n_pixels} px)`:`pixel (${s.row}, ${s.col})`} <i>(not added)</i></span>`); unit=unit||s.unit; }
  if(!series.length && !I.picked.size){
    box.innerHTML=`<div class="placeholder-card">${(!I.show.px&&!I.show.roi)?"Tick Pixel spectra and/or ROI above to plot them.":"Click a pixel on the image to see its spectrum here."}</div>`; return; }
  const ref=series[0]; let rescaled=false; const ids=Array.from(I.picked).slice(0,8);
  for(let k=0;k<ids.length;k++){
    try{
      const sp=specCache[ids[k]]||(specCache[ids[k]]=await apiGet(`/api/spectra/${ids[k]}`));
      let data=sp.wavelengths.map((w,i)=>({x:w,y:sp.values[i]})).filter(d=>d.y!==null);
      if(unit&&sp.measurement_unit!==unit&&ref){ let m=-Infinity,pm=-Infinity; for(const d of data) if(d.y>m) m=d.y; for(const d of ref.data) if(d.y>pm) pm=d.y; if(m>0){ data=data.map(d=>({x:d.x,y:d.y/m*pm})); rescaled=true; } }
      const h=I.libSpectra.find(x=>x.spectrum_id===ids[k])||(I.matches||[]).find(x=>x.spectrum_id===ids[k]);
      const col=["#9ca3af","#6b7280","#a16207","#0e7490","#9d174d","#3f6212","#4338ca","#b91c1c"][k%8];
      series.push({name:ids[k],color:col,data}); names.push(`<span style="color:${col}">&#9633; ${escapeHtml(h?h.material:ids[k].slice(0,8))} <i>(library)</i></span>`);
    }catch(_){}
  }
  box.innerHTML=`${dsChart(series,{height:230,width:460,unit:unit||undefined})}<div style="font-size:11.5px;margin-top:6px;display:flex;gap:4px 12px;flex-wrap:wrap">${names.join("")}</div>`
    +(rescaled?`<p class="note">Library curves are scaled to the image spectrum (units differ), so only the shape is comparable.</p>`:"");
}
