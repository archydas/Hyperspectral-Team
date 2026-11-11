/* object-samples.js — Object samples (ROI): the table under the image, image patches (<class>.tif), per-ROI spectrum, aggregation
   (classic script, loaded in order from index.html; all functions are global)

   An object sample is the mean spectrum of an ROI polygon drawn on an image. It is kept in AG.rois (also saved in the
   browser's localStorage) and looks like this:
     { image_id, image_name, image_path, polygon:[[row,col],...], class_name, name, color, show,
       n (pixels), unit, wavelengths, values (mean), std, lat, lon (centre of the ROI, null if the image is not georeferenced),
       patch:{file, path, folder} | null (the GeoTIFF written for this ROI) }
   The image patch is a real file in a folder on this computer; the database only keeps its name and address.
*/
const OBJ_COLORS=["#d97706","#7c3aed","#0891b2","#dc2626","#4d7c0f","#be185d","#2563eb","#0f766e","#ea580c","#6d28d9"];
const OBJ={ folder:"", edit:null, stat:"mean", menu:null };      // menu = index of the ROI selected on the image (its small menu is open)
try{ OBJ.folder=localStorage.getItem("ds3_patch_folder")||""; }catch(_){}

function objs(){ return AG.rois; }
function samePoly(a,b){ return JSON.stringify(a)===JSON.stringify(b); }
function objRefresh(){ const el=document.getElementById("imgObjects"); if(el) el.innerHTML=objectsCardHtml(); else if(state.view==="image") renderMainOnly(); }
const agRefresh=objRefresh;                                    // older name used by the aggregation buttons
function fmtCoord(v){ return v==null?"—":(+v).toFixed(5); }

/* ---------- ROI outlines drawn on the image: click the outline or its number to get a small menu (Spectrum / Export / Delete) ---------- */
function objOverlay(){
  const a=I.active; if(!a||!I.show.roi) return "";
  const mine=objs().map((o,i)=>({o,i})).filter(x=>x.o.image_id===a.id&&x.o.show&&x.o.polygon&&x.o.polygon.length>2);
  if(!mine.length) return "";
  const poly=({o,i})=>{ const pts=o.polygon.map(p=>p[1]+","+p[0]).join(" "), on=OBJ.menu===i;
    return `<polygon points="${pts}" fill="${o.color}${on?"55":"33"}" stroke="${o.color}" stroke-width="${on?4:2}" vector-effect="non-scaling-stroke"/>
      <polygon points="${pts}" fill="none" stroke="transparent" stroke-width="14" vector-effect="non-scaling-stroke" pointer-events="stroke" style="cursor:pointer" onclick="objPick(${i},event)"/>`; };
  return `<svg viewBox="0 0 ${a.cols} ${a.rows}" preserveAspectRatio="none" style="position:absolute;inset:0;width:100%;height:100%;pointer-events:none">${mine.map(poly).join("")}</svg>`
   +mine.map(({o,i})=>{ const cy=o.polygon.reduce((s,p)=>s+p[0],0)/o.polygon.length, cx=o.polygon.reduce((s,p)=>s+p[1],0)/o.polygon.length;
      return `<div class="pxmark roimark" style="left:${(cx+0.5)/a.cols*100}%;top:${(cy+0.5)/a.rows*100}%;border-color:${o.color};background:${o.color}99" onclick="objPick(${i},event)" title="Click: spectrum, export or delete this ROI">${i+1}</div>`; }).join("");
}
function objPick(i,ev){
  if(I.roi.drawing||I.panned) return;                      // while drawing, a click on an old outline is just another point
  ev.stopPropagation(); const o=objs()[i]; if(!o) return;
  objMenuClose(true); OBJ.menu=i; refreshView();
  const h=document.createElement("div"); h.id="objMenu"; h.className="objmenu";
  h.innerHTML=`<div class="pxpop-t"><b>${escapeHtml(o.name)}</b><button onclick="objMenuClose()" title="Close">&#10005;</button></div>
    <div class="note" style="margin:0 0 8px">${escapeHtml(o.class_name||"")} · ${o.n} pixels</div>
    <div style="display:flex;gap:6px;flex-wrap:wrap"><button class="btn ghost btn-sm" onclick="objMenuClose();objSpectrum(${i})">Spectrum</button>
      <button class="btn ghost btn-sm" onclick="objMenuClose();objExportPatch(${i})">Export</button>
      <button class="btn danger btn-sm" onclick="objRemove(${i})">Delete ROI</button></div>`;
  document.body.appendChild(h);
  h.style.left=Math.max(8,Math.min(ev.clientX+12,innerWidth-250))+"px"; h.style.top=Math.max(8,Math.min(ev.clientY+12,innerHeight-130))+"px";
}
function objMenuClose(keepView){
  const h=document.getElementById("objMenu"); if(h) h.remove();
  if(OBJ.menu!=null){ OBJ.menu=null; if(!keepView) refreshView(); }
}
document.addEventListener("keydown",e=>{
  if(OBJ.menu==null||state.view!=="image"||/INPUT|TEXTAREA|SELECT/.test((e.target||{}).tagName||"")) return;
  if(e.key==="Delete"||e.key==="Backspace"){ e.preventDefault(); objRemove(OBJ.menu); }
  if(e.key==="Escape") objMenuClose();
});

/* ---------- adding an ROI: spectrum + coordinates + image patch ---------- */
async function objAdd(s,label,cls){
  if(!I.active||!(I.roi&&I.roi.closed)) return;
  const poly=I.roi.pts.map(p=>[p[0],p[1]]);
  if(objs().some(o=>o.image_id===I.active.id&&samePoly(o.polygon,poly))){ showToast("This ROI is already in the list"); return; }
  const name=(label||"").trim()||`ROI ${objs().length+1}`;
  const o={image_id:I.active.id,image_name:I.active.name,image_path:I.active.path||"",polygon:poly,class_name:cls||name,name,
    color:OBJ_COLORS[objs().length%OBJ_COLORS.length],show:true,n:s.n_pixels||0,unit:s.unit,wavelengths:s.wavelengths,values:s.values,std:s.std||null,
    lat:null,lon:null,patch:null,patchErr:""};
  objs().push(o); agStore(); AG.result=null;
  I.roi={pts:[],closed:false,drawing:false}; I.spec=I.pixel?{...I.pixel,kind:"pixel"}:null;
  refreshView(); refreshSide(); objRefresh();
  showToast(`"${name}" added to the object samples`);
  await objFillCoords([o]);
  if(!OBJ.folder) objFolderDialog(); else await objSavePatch(o);
}
/* latitude / longitude of the ROI centre, from the image's georeference (file or control points) */
async function objFillCoords(list){
  const byImg={}; list.forEach(o=>{ (byImg[o.image_id]=byImg[o.image_id]||[]).push(o); });
  for(const id of Object.keys(byImg)){
    const os=byImg[id];
    try{
      const r=await apiSend("POST",`/api/images/${id}/lonlat`,{points:os.map(o=>[o.polygon.reduce((s,p)=>s+p[0],0)/o.polygon.length,o.polygon.reduce((s,p)=>s+p[1],0)/o.polygon.length])});
      os.forEach((o,k)=>{ const p=r.points?r.points[k]:null; o.lat=p?p.lat:null; o.lon=p?p.lon:null; });
    }catch(_){}
  }
  agStore(); objRefresh();
}
async function objRefreshCoords(){ await objFillCoords(objs()); }       // called after the image has been georeferenced

/* ---------- the folder for the image patches (asked once, at the first ROI) ---------- */
async function objFolderDialog(){
  let def=OBJ.folder;
  if(!def){ try{ def=(await apiGet("/api/system/patch_folder")).default; }catch(_){ def=""; } }
  const canBrowse=!!(window.pywebview&&window.pywebview.api&&window.pywebview.api.pick_files);
  win("Where should the ROI image patches be saved?",
   `<p class="note" style="margin:0 0 8px">Every ROI is also saved as an image patch (a GeoTIFF with all bands), named after its class — for example <b>Dry grass.tif</b>. Choose the folder once; the next ROIs go to the same folder. The Library module finds the patches again from the file name and this folder.</p>
    <label class="small">Folder</label>
    <div style="display:flex;gap:6px"><input id="objFolderIn" type="text" value="${escapeHtml(def)}" style="flex:1">${canBrowse?`<button class="btn ghost btn-sm" onclick="objBrowse()">Browse…</button>`:""}</div>
    <div class="row"><button class="btn emerald btn-sm" onclick="objFolderOk()">Save patches here</button><button class="btn ghost btn-sm" onclick="winClose()">Not now</button></div>
    <div class="err" id="objFolderErr"></div>`,{width:560});
}
async function objBrowse(){
  try{ const r=await window.pywebview.api.pick_files(true); if(r&&r.length) document.getElementById("objFolderIn").value=r[0]; }catch(e){ showToast(e.message); }
}
async function objFolderOk(){
  const v=(document.getElementById("objFolderIn").value||"").trim(), err=document.getElementById("objFolderErr");
  try{ const r=await apiSend("POST","/api/system/check_folder",{path:v}); OBJ.folder=r.path; try{ localStorage.setItem("ds3_patch_folder",OBJ.folder); }catch(_){} winClose(); await objSaveMissingPatches(); }
  catch(e){ err.textContent=e.message; }
}
/* the R, G, B bands shown on the page become the patch's default display bands (all bands are in the file) */
function objBands(o){ return I.active&&I.active.id===o.image_id?I.bands:null; }
async function objSavePatch(o){
  if(!OBJ.folder) { objFolderDialog(); return; }
  try{
    const r=await apiSend("POST",`/api/images/${o.image_id}/patch`,{polygon:o.polygon,class_name:o.class_name,folder:OBJ.folder,bands:objBands(o)});
    o.patch={file:r.file,path:r.path,folder:r.folder}; o.patchErr=""; if(r.lat!=null){ o.lat=r.lat; o.lon=r.lon; }
  }catch(e){ o.patch=null; o.patchErr=e.message; showToast("Patch not saved: "+e.message); }
  agStore(); objRefresh();
}
async function objSaveMissingPatches(){ for(const o of objs()) if(!o.patch) await objSavePatch(o); }
async function objExportPatch(i){                                          // download a copy (browser / another computer)
  const o=objs()[i]; try{ await dlPost(`/api/images/${o.image_id}/patch/download`,{polygon:o.polygon,class_name:o.class_name,bands:objBands(o)},(o.class_name||"roi")+".tif"); }catch(e){ showToast(e.message); }
}
async function fileLocate(path){
  try{ await apiSend("POST","/api/system/reveal",{path}); }catch(e){ showToast(e.message); }
}
function objLocate(i){ const o=objs()[i]; if(o&&o.patch) fileLocate(o.patch.path); }
async function imgLocate(id){
  try{ const r=await apiSend("POST",`/api/images/${id}/reveal`,{}); showToast("Opened the folder: "+r.path); }catch(e){ showToast(e.message); }
}

/* ---------- list actions ---------- */
function objRemove(i){ objMenuClose(true); OBJ.menu=null; objs().splice(i,1); agStore(); AG.result=null; refreshView(); refreshSide(); objRefresh(); }
function objClear(){ objs().splice(0); agStore(); AG.result=null; refreshView(); refreshSide(); objRefresh(); }
function objShow(i,v){ objs()[i].show=v; agStore(); AG.result=null; refreshView(); drawSpectrumChart(); objRefresh(); }
function objAll(v){ objs().forEach(o=>o.show=v); if(v) I.show.roi=true; agStore(); AG.result=null; refreshView(); drawSpectrumChart(); objRefresh(); }
function objEdit(i){ OBJ.edit=i; objRefresh(); setTimeout(()=>{ const e=document.getElementById("objEd1"); if(e){ e.focus(); e.select(); } },0); }
function objEditOk(){
  const i=OBJ.edit, o=objs()[i]; if(!o) return;
  const n=(document.getElementById("objEd1").value||"").trim(), c=(document.getElementById("objEd2").value||"").trim();
  if(n) o.name=n; o.class_name=c||o.name; OBJ.edit=null; agStore(); AG.result=null; refreshView(); drawSpectrumChart(); objRefresh();
}

/* ---------- one ROI: its spectrum (surface reflectance) ---------- */
function objSpectrum(i){
  const o=objs()[i]; if(!o||!o.values) return;
  const ok=o.values.map((v,k)=>v==null?null:k).filter(k=>k!==null);
  const mean=ok.reduce((s,k)=>s+o.values[k],0)/(ok.length||1);
  const sd=o.std?ok.filter(k=>o.std[k]!=null).map(k=>o.std[k]):[];
  const sdMean=sd.length?sd.reduce((s,v)=>s+v,0)/sd.length:null;
  const pt=(arr,f)=>o.wavelengths.map((w,k)=>({x:w,y:arr[k]==null?null:f(arr[k],k)})).filter(d=>d.y!=null);
  const series=[{name:o.name,color:o.color,data:pt(o.values,v=>v)}];
  if(o.std){ series.push({name:"mean + std",color:"#9ca3af",data:o.wavelengths.map((w,k)=>({x:w,y:(o.values[k]==null||o.std[k]==null)?null:o.values[k]+o.std[k]})).filter(d=>d.y!=null)});
             series.push({name:"mean − std",color:"#9ca3af",data:o.wavelengths.map((w,k)=>({x:w,y:(o.values[k]==null||o.std[k]==null)?null:o.values[k]-o.std[k]})).filter(d=>d.y!=null)}); }
  const yLab=o.unit==="reflectance"?"Surface reflectance":(o.unit||"value");
  win(`Spectrum — ${o.name}`,
   `<div class="small" style="margin-bottom:6px">Class <b>${escapeHtml(o.class_name)}</b> · ${o.n} pixels · ${escapeHtml(o.image_name)} · ${fmtCoord(o.lat)}, ${fmtCoord(o.lon)}${o.patch?` · patch <b>${escapeHtml(o.patch.file)}</b>`:""}</div>
    ${dsChart(series,{height:260,width:640,yLabel:yLab})}
    <p class="note">Coloured line = mean of the ROI pixels${o.std?"; grey lines = mean ± one standard deviation (sample, n − 1)":""}.</p>
    <table class="gcp-table" style="max-width:520px"><tr><th>Mean over all bands</th><th>Mean std. deviation</th><th>Wavelength range</th><th>Bands used</th></tr>
      <tr><td>${mean.toFixed(4)}</td><td>${sdMean==null?"—":sdMean.toFixed(4)}</td><td>${o.wavelengths[0].toFixed(0)}–${o.wavelengths[o.wavelengths.length-1].toFixed(0)} nm</td><td>${ok.length}</td></tr></table>
    <div class="row"><button class="btn ghost btn-sm" onclick="objSpectrumCsv(${i})">Export CSV</button><button class="btn ghost btn-sm" onclick="winClose()">Close</button></div>`,{width:700});
}
function objSpectrumCsv(i){
  const o=objs()[i]; if(!o) return;
  csvDownload(`${o.class_name||o.name}_spectrum.csv`,["wavelength_nm,mean"+(o.std?",std":""),...o.wavelengths.map((w,k)=>[w,o.values[k]==null?"":o.values[k],...(o.std?[o.std[k]==null?"":o.std[k]]:[])].join(","))]);
}

/* ---------- the object samples card (below the image) ---------- */
function objectsCardHtml(){
  const O=objs(), a=I.active;
  const rows=O.map((o,i)=>{
    const here=a&&o.image_id===a.id;
    const editing=OBJ.edit===i, kd=`onkeydown="if(event.key==='Enter')objEditOk();if(event.key==='Escape'){OBJ.edit=null;objRefresh()}"`;
    return `<tr>
      <td><input type="checkbox" ${o.show?"checked":""} onchange="objShow(${i},this.checked)" title="Show in the chart and on the image"></td>
      <td><span class="coll-sw" style="background:${o.color};border-radius:4px"></span> ${i+1}</td>
      <td>${editing?`<input id="objEd1" type="text" value="${escapeHtml(o.name)}" ${kd}>`:`<span class="coll-nm" onclick="objEdit(${i})" title="Click to rename">${escapeHtml(o.name)}</span>`}</td>
      <td>${editing?`<input id="objEd2" type="text" value="${escapeHtml(o.class_name||"")}" ${kd}><button class="coll-x" onclick="objEditOk()" title="Done">&#10003;</button>`:escapeHtml(o.class_name||"")}</td>
      <td class="num">${fmtCoord(o.lat)}</td><td class="num">${fmtCoord(o.lon)}</td><td class="num">${o.n}</td>
      <td>${o.patch?`<span title="${escapeHtml(o.patch.folder||"")}">&#128247; ${escapeHtml(o.patch.file)}</span>`:`<span class="note" style="margin:0" title="${escapeHtml(o.patchErr||"")}">no patch yet</span> <button class="btn ghost btn-sm" onclick="objSavePatch(objs()[${i}])">Save patch</button>`}</td>
      <td class="note" style="margin:0">${escapeHtml(o.image_name)}${here?"":" <i>(other image)</i>"}</td>
      <td style="white-space:nowrap"><button class="btn ghost btn-sm" onclick="objSpectrum(${i})">Spectrum</button>
        <button class="btn ghost btn-sm" onclick="objExportPatch(${i})" title="Download this ROI as a GeoTIFF patch">Export</button>
        <button class="btn ghost btn-sm" onclick="objLocate(${i})" ${o.patch?"":"disabled"} title="Show the patch file in the file browser">Locate</button>
        <button class="coll-x" onclick="objRemove(${i})" title="Remove from the list (the patch file stays on disk)">&#10005;</button></td></tr>`; }).join("");
  const folderLine=OBJ.folder?`Patch folder: <b>${escapeHtml(OBJ.folder.split(/[\\/]/).filter(Boolean).slice(-2).join("/"))}</b> <button class="btn ghost btn-sm" onclick="objFolderDialog()">Change</button>`
    :`<button class="btn ghost btn-sm" onclick="objFolderDialog()">Choose patch folder…</button>`;
  return `<div class="card" style="padding:14px;margin-top:16px">
    <div style="display:flex;gap:10px;align-items:center;flex-wrap:wrap"><b style="font-size:14px">Object sample collection (ROI) — ${O.length}</b>
      <span class="small">${folderLine}</span>
      <span style="margin-left:auto;display:flex;gap:4px"><button class="btn ghost btn-sm" onclick="objAll(true)" ${O.length?"":"disabled"}>Select all</button><button class="btn ghost btn-sm" onclick="objAll(false)" ${O.length?"":"disabled"}>Unselect all</button></span></div>
    ${O.length?`<div style="overflow:auto"><table class="gcp-table obj-table"><tr><th></th><th>#</th><th>Label</th><th>Class</th><th>Latitude</th><th>Longitude</th><th>Pixels</th><th>Image (patch)</th><th>Source scene</th><th></th></tr>${rows}</table></div>
      ${O.some(o=>o.lat==null)?`<p class="note">Latitude / longitude appear when the image is georeferenced (<b>Georeference</b> button above).</p>`:""}`
      :`<p class="note" style="margin-top:10px">Draw an ROI polygon on the image (double-click or right-click to finish). A small window shows its mean spectrum: give it a label and a class and press <b>Add</b>. The ROI is listed here and saved as an image patch <i>&lt;class name&gt;.tif</i>.</p>`}
    <div style="display:flex;gap:6px;flex-wrap:wrap;margin-top:10px;align-items:center">
      <button class="btn emerald btn-sm" onclick="saveDialog('o')" ${O.length?"":"disabled"}>Save to library…</button>
      <button class="btn violet btn-sm" onclick="agRun(false)" ${O.some(o=>o.show&&o.values)&&!AG.busy?"":"disabled"} title="Mean, standard deviation and variance of the ticked ROIs, per class">${AG.busy?"Aggregating…":"Aggregate ticked ROIs"}</button>
      <button class="btn danger btn-sm ${I.arm.o?"armed":""}" style="margin-left:auto" onclick="delAll('o')" ${O.length?"":"disabled"}>${I.arm.o?"Click again to delete all":"Delete all"}</button></div>
    ${AG.result||AG.err?aggregationHtml():""}
  </div>`;
}

/* ---------- aggregate: shown after the "Aggregate ticked ROIs" button — all spectra together + mean / std. deviation / variance per class ---------- */
function aggregationHtml(){
  const R=AG.result, O=objs(), used=O.filter(o=>o.show&&o.values), cls=R?R.classes:[];
  const classNames=[...new Set(used.map(o=>o.class_name))], colorOf=n=>DS_COLORS[Math.max(0,classNames.indexOf(n))%DS_COLORS.length];
  const toData=(wl,v)=>wl.map((w,k)=>({x:w,y:v[k]})).filter(d=>d.y!=null);
  const head=`<div style="display:flex;gap:10px;align-items:center;flex-wrap:wrap"><b style="font-size:13.5px">Aggregate</b>
      <span class="small">${R?`${used.length} ROI(s) · ${cls.length} class(es) · pixels of the same class are pooled`:""}</span>
      <button class="coll-x" style="margin-left:auto" onclick="AG.result=null;AG.err='';objRefresh()" title="Close the aggregate">&#10005;</button></div>`;
  if(!R) return `<div class="agg-box">${head}${AG.err?`<p style="color:#c62828;font-size:12.5px;margin:8px 0 0">${escapeHtml(AG.err)}</p>`:""}</div>`;
  /* thin lines: every ticked ROI; with "Mean" the class means are drawn on top, with the other statistics only the class curves */
  const g=(c,k)=>R.wavelengths.map((w,j)=>({x:w,y:c[k][j]})).filter(d=>d.y!=null);
  let series, yLabel, note;
  if(OBJ.stat==="mean"){
    series=used.map(o=>({name:o.name,color:colorOf(o.class_name)+"99",data:toData(o.wavelengths,o.values)})).concat(cls.map(c=>({name:c.name+" mean",color:colorOf(c.name),data:g(c,"mean")})));
    note="Thin lines: every ticked ROI. Thick lines: the mean of each class.";
  }else{
    const f=OBJ.stat;
    series=cls.map(c=>f==="minmax"?[{name:c.name+" min",color:colorOf(c.name),data:g(c,"min")},{name:c.name+" max",color:colorOf(c.name),data:g(c,"max")}]
      :[{name:c.name,color:colorOf(c.name),data:g(f==="variance"?{...c,variance:c.std.map(v=>v==null?null:v*v)}:c,f==="variance"?"variance":"std")}]).flat();
    yLabel=f==="std"?"Standard deviation":f==="variance"?"Variance":"Minimum and maximum"; note=f==="minmax"?"Lowest and highest value of any pixel of the class, per band.":"Per band, over all pixels of the class (sample statistics, n − 1).";
  }
  const avg=a=>{ const v=a.filter(x=>x!=null&&Number.isFinite(x)); return v.length?v.reduce((s,x)=>s+x,0)/v.length:null; }, f4=v=>v==null?"—":v.toFixed(4);
  const statRows=cls.map(c=>`<tr><td><span style="color:${colorOf(c.name)}">&#9632;</span> ${escapeHtml(c.name)}</td><td>${used.filter(o=>o.class_name===c.name).length}</td><td>${c.n_pixels}</td><td>${f4(avg(c.mean))}</td><td>${f4(avg(c.std))}</td><td>${f4(avg(c.std.map(v=>v==null?null:v*v)))}</td><td>${f4(avg(c.min))}</td><td>${f4(avg(c.max))}</td></tr>`).join("");
  return `<div class="agg-box">${head}
    <div style="display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin-top:8px"><span class="small"><b>Show</b></span>
      ${[["mean","Mean"],["std","Std. deviation"],["variance","Variance"],["minmax","Min–Max"]].map(([k,l])=>`<label class="small"><input type="radio" name="objStat" ${OBJ.stat===k?"checked":""} onchange="OBJ.stat='${k}';objRefresh()"> ${l}</label>`).join("")}
      <span style="margin-left:auto;display:flex;gap:6px"><button class="btn ghost btn-sm" onclick="agExport('csv')">Export CSV</button><button class="btn blue btn-sm" onclick="saveDialog('a')" title="Store the mean spectrum of every class in a library">Save class means to library…</button></span></div>
    <div class="chart-wrap" style="margin-top:8px">${dsChart(series,{height:270,width:1000,unit:(used[0]||{}).unit,yLabel})}
      <div style="font-size:11.5px;margin-top:6px;display:flex;gap:4px 12px;flex-wrap:wrap">${classNames.map(n=>`<span style="color:${colorOf(n)}">&#9632; ${escapeHtml(n)}</span>`).join("")}</div>
      <p class="note">${note}</p></div>
    <table class="gcp-table"><tr><th>Class</th><th>ROIs</th><th>Pixels</th><th>Mean</th><th>Std. deviation</th><th>Variance</th><th>Min</th><th>Max</th></tr>${statRows}</table>
    <p class="note">Each value is averaged over all ${R.wavelengths.length} bands (${R.wavelengths[0].toFixed(0)}–${R.wavelengths[R.wavelengths.length-1].toFixed(0)} nm). ${R.mixed_units?`<b style="color:#b45309">The scenes have different units (${escapeHtml(R.units.join(", "))}): the pooled values mix them.</b>`:""}</p>
  </div>`;
}
async function agRun(save){
  AG.err=""; if(save&&!AG.lib.trim()){ AG.err="Give the library a name."; objRefresh(); return; }
  AG.busy=true; objRefresh();
  try{ AG.result=await apiSend("POST","/api/aggregate",agBody(save)); if(save){ await loadLibraries(); showToast(`Saved ${AG.result.saved.saved.length} class signature(s) to "${AG.result.saved.library}"`); } }
  catch(e){ AG.err=e.message; AG.result=null; }
  AG.busy=false; objRefresh();
  if(!save&&AG.result){ const b=document.querySelector(".agg-box"); if(b) b.scrollIntoView({behavior:"smooth",block:"nearest"}); }
}
async function agExport(fmt){ try{ await dlPost(`/api/aggregate/export/${fmt}`,agBody(false),`aggregated_signatures.${fmt}`); }catch(e){ AG.err=e.message; objRefresh(); } }
function aggGo(){ state.view="image"; render(); setTimeout(()=>{ const e=document.getElementById("imgObjects"); if(e) e.scrollIntoView({behavior:"smooth"}); },60); }
