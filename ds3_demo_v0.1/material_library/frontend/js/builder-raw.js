/* builder-raw.js — Spectral Library Builder: the raw (original) spectra are saved next to the modified library
   (classic script, loaded in order from index.html; all functions are global) */
BLD2.keepRaw=true;
async function bldAdd(raw, origin){
  /* raw: [{name, wavelengths, values, unit, source}] -> resampled onto the output grid on the backend; the originals are kept for the "raw" library */
  BLD2.busy=true; BLD2.err=""; renderMainOnly();
  try{
    const r=await apiSend("POST","/api/builder/resample",{spectra:raw.map(s=>({name:s.name,wavelengths:s.wavelengths,values:s.values,unit:s.unit||"reflectance",origin:origin||s.origin||"file",source:s.source||""})),
      grid:BLD2.grid,mask_water:BLD2.maskWater,min_coverage:+BLD2.minCov});
    if(!BLD2.grid && r.grid){ BLD2.grid=r.grid; BLD2.gridSrc="first input spectrum"; }
    const q={}; raw.forEach(s=>{ (q[s.name]=q[s.name]||[]).push(s); });
    r.spectra.forEach(s=>{ const src=(q[s.name]||[]).shift();
      BLD2.items.push({...s,id:"b"+Math.random().toString(36).slice(2,9),category:"",raw:src?{wavelengths:src.wavelengths,values:src.values,unit:src.unit||"reflectance"}:null}); });
    r.rejected.forEach(x=>BLD2.rejected.push(x));
    showToast(`Added ${r.spectra.length}${r.rejected.length?`, rejected ${r.rejected.length}`:""}`);
  }catch(e){ BLD2.err=e.message; }
  BLD2.busy=false; renderMainOnly();
}
function bldRawHtml(){
  const B=BLD2, nFile=B.items.filter(i=>i.origin==="file"&&i.raw).length, libs=[...new Set(B.items.filter(i=>i.origin==="library").map(i=>i.source).filter(Boolean))];
  if(!nFile&&!libs.length) return "";
  return `<div style="margin:8px 0 0;font-size:12.5px">`
    +(nFile?`<label style="display:flex;gap:6px;align-items:center"><input type="checkbox" ${B.keepRaw?"checked":""} onchange="BLD2.keepRaw=this.checked"> Also save the ${nFile} original spectra exactly as read from the file(s) (no resampling) as a second library: the same name plus “(raw)”</label>`:"")
    +(libs.length?`<div class="note" style="margin:4px 0 0">Spectra taken from ${libs.map(l=>"“"+escapeHtml(l)+"”").join(", ")}: that library stays unchanged in the database; Save creates a separate, modified library.</div>`:"")
    +`</div>`;
}
async function bldSave(){
  if(!BLD2.items.length){ showToast("Nothing to save"); return; }
  const name=libPickValue("bldLib")||BLD2.libName.trim(); if(!name){ BLD2.err="Give the library a name."; renderMainOnly(); return; }
  BLD2.libName=name; BLD2.category=document.getElementById("bldCat")?.value||""; BLD2.acq=document.getElementById("bldAcq")?.value||"";
  try{
    const fileItems=BLD2.items.filter(i=>i.origin==="file"&&i.raw), srcLibs=[...new Set(BLD2.items.filter(i=>i.origin==="library").map(i=>i.source).filter(Boolean))];
    let rawName=null, extra="";
    if(BLD2.keepRaw&&fileItems.length){
      rawName=`${name} (raw)`;
      const rr=await apiSend("POST","/api/spectra/save",{library:rawName,
        library_info:{type:"spectral",raw:true,modified_library:name,source_files:[...new Set(fileItems.map(i=>i.source).filter(Boolean))],note:"Original spectra exactly as read from the source file(s): no resampling, no wavelength change."},
        items:fileItems.map(i=>({name:i.name,wavelengths_nm:i.raw.wavelengths,values:i.raw.values,unit:i.raw.unit||i.unit,category:BLD2.category||null,origin:"raw_file",acquisition_type:BLD2.acq||null}))});
      extra=` and ${rr.saved} original spectra to "${rr.library}"`;
      BLD2.rejected=BLD2.rejected.concat((rr.rejected||[]).map(x=>({name:x.name+" (raw)",reason:x.reason})));
    }
    const info={type:"spectral",modified:true,grid:BLD2.gridSrc||"first input spectrum",built_with:"Spectral Library Builder"};
    if(srcLibs.length) info.derived_from_libraries=srcLibs; if(rawName) info.raw_library=rawName;
    const r=await apiSend("POST","/api/spectra/save",{library:name,library_info:info,items:BLD2.items.map(i=>({name:i.name,wavelengths_nm:BLD2.grid,values:i.values,unit:i.unit,
        category:BLD2.category||null,origin:"builder",acquisition_type:BLD2.acq||null}))});
    BLD2.err=""; BLD2.rejected=BLD2.rejected.concat((r.rejected||[]).map(x=>({name:x.name,reason:x.reason}))); await loadLibraries();
    showToast(`Saved ${r.saved} spectra to library "${r.library}"${extra}`); }
  catch(e){ BLD2.err=e.message; }
  renderMainOnly();
}

/* ---------- where an object / pixel sample came from (spectrum page of a library material) ---------- */
/* ROI sample: the name of its patch file + View image / Locate. Pixel sample: the image name and the pixel. No folder paths are shown. */
function srcImageHtml(sps){
  const rows=sps.map(s=>{ const m=s.sample_meta||{}; return m.object?{o:m.object,kind:"ROI",sid:s.sample_id}:(m.pixel?{o:m.pixel,kind:"pixel",sid:s.sample_id}:null); }).filter(Boolean);
  if(!rows.length) return "";
  const f5=v=>v==null?"":(+v).toFixed(5);
  return `<div class="card" style="padding:12px;margin-top:12px"><b>Source</b> <span class="small">· the image is not stored in the database, only the name and address of its file</span>`
    +rows.slice(0,6).map(({o,kind,sid})=>kind==="ROI"
      ?`<div class="small" style="margin-top:8px">ROI of ${o.n_pixels||"?"} pixels${o.class_name?` · class <b>${escapeHtml(o.class_name)}</b>`:""}${o.lat!=null?` · ${f5(o.lat)}, ${f5(o.lon)}`:""} · scene ${escapeHtml(o.image_name||"")}<br>
         ${o.patch_file?`Image patch: <b>&#128247; ${escapeHtml(o.patch_file)}</b> <button class="btn emerald btn-sm" onclick="libObjView('${sid}')" title="Look at the saved patch (it is not opened in the Image module)">View image</button> <button class="btn ghost btn-sm" onclick="libObjLocate('${sid}')">Locate</button>`:`<span class="note" style="margin:0">no image patch was saved for this ROI</span>`}</div>`
      :`<div class="small" style="margin-top:8px">pixel (${o.row}, ${o.col}) · image <b>${escapeHtml(o.image_name||"")}</b> ${o.image_id?`<button class="btn ghost btn-sm" onclick="imgLocate('${o.image_id}')">Locate image file</button>`:""}</div>`).join("")+`</div>`;
}
