/* georef.js — Georeferencing with control points: the image beside the Esri World Imagery map
   (classic script, loaded in order from index.html; all functions are global)

   How it works: the user clicks a recognisable point on the image, then the same point on the map. Each pair is a control
   point (row, col  <->  latitude, longitude). From 3 or more pairs the backend fits an affine transformation
   (PUT /api/images/{id}/georef) and reports the error of every point and the RMSE in metres. The result is stored with the
   image, so View map, ROI latitude / longitude and the patch GeoTIFFs use it.
*/
const GR={ open:false, pts:[], pending:null, fit:null, map:null, layer:null, token:0, zoom:1, saved:null, fileGeo:false, err:"", busy:false };

async function georefOpen(){
  if(!I.active||!I.preview){ showToast("Open an image first"); return; }
  const a=I.active;
  Object.assign(GR,{pts:[],pending:null,fit:null,err:"",busy:false,zoom:1}); georefMark(true);
  try{
    const g=await apiGet(`/api/images/${a.id}/georef`);
    GR.saved=g.georef; GR.fileGeo=!!g.file_has_geo;
    if(g.georef&&g.georef.points) GR.pts=g.georef.points.map(p=>({...p}));
  }catch(_){ GR.saved=null; GR.fileGeo=false; }
  win("Georeference — match points of the image with the map",
   `<div class="small" style="margin-bottom:8px;line-height:1.55">${GR.fileGeo?`<b style="color:#047857">This image already carries map coordinates in its file</b> — use <b>View map</b>. Control points are only needed for images without them.<br>`:""}
    <b>1.</b> Click a point on the <b>image</b> that you can also find on the map (a road crossing, a building corner, a field corner…). &nbsp;<b>2.</b> Click the <b>same point on the map</b>. &nbsp;Repeat for at least <b>3 points</b> spread over the image (4–6 are better: the error of each point is shown). Mouse wheel = zoom, drag = move.</div>
    <div class="gr-grid">
      <div><div class="gr-title">Image · ${escapeHtml(a.name)}</div><div id="grImg" class="img-vp gr-pane"><div id="grBox" style="position:relative;line-height:0;margin:auto;flex:none;cursor:crosshair"><img id="grPic" src="${I.preview.image}" draggable="false" style="width:100%;display:block;user-select:none"><div id="grImgMarks"></div></div></div></div>
      <div><div class="gr-title" style="display:flex;gap:6px;align-items:center">Map · Esri World Imagery <span style="margin-left:auto;font-weight:400"><input id="grGo" type="text" placeholder="go to: lat, lon" style="width:140px" onkeydown="if(event.key==='Enter')grGoTo()"><button class="btn ghost btn-sm" onclick="grGoTo()">Go</button></span></div><div id="grMap" class="gr-pane"></div></div>
    </div>
    <div id="grTable"></div>
    <div class="row"><button class="btn emerald btn-sm" id="grApply" onclick="grApply()">Apply georeference</button>
      <button class="btn ghost btn-sm" onclick="grClear()">Clear points</button>
      ${GR.saved?`<button class="btn ghost btn-sm" onclick="grRemoveSaved()">Remove the saved georeference</button>`:""}
      <button class="btn ghost btn-sm" onclick="grClose()">Close</button></div><div class="err" id="grErr"></div>`,{width:1240});
  grInitImage(); grInitMap(); grRefresh();
}
function georefMark(on){ GR.open=on; const b=document.getElementById("btnGeo"); if(b){ b.classList.toggle("emerald",on); b.classList.toggle("ghost",!on); } }     // the button is green only while the window is open
function grClose(){ if(GR.map){ try{ GR.map.remove(); }catch(_){} GR.map=null; } winClose(); }

/* ----- the image pane: fit, wheel zoom, drag, click to set the pending point ----- */
function grInitImage(){
  const vp=document.getElementById("grImg"), box=document.getElementById("grBox"), pv=I.preview;
  const fit=Math.max(0.05,Math.min((vp.clientWidth-4)/pv.width,(vp.clientHeight-4)/pv.height,8));
  GR.zoom=fit; box.style.width=Math.round(pv.width*fit)+"px";
  vp.addEventListener("wheel",e=>{
    e.preventDefault(); const old=GR.zoom, nz=Math.max(fit*0.8,Math.min(old*(e.deltaY<0?1.25:0.8),24*I.active.cols/pv.width)); if(nz===old) return;
    const r=vp.getBoundingClientRect(), px=e.clientX-r.left, py=e.clientY-r.top, ix=(vp.scrollLeft+px-box.offsetLeft)/old, iy=(vp.scrollTop+py-box.offsetTop)/old;
    GR.zoom=nz; box.style.width=Math.round(pv.width*nz)+"px"; document.getElementById("grPic").style.imageRendering=(nz*pv.width/I.active.cols)>2?"pixelated":"auto";
    vp.scrollLeft=ix*nz+box.offsetLeft-px; vp.scrollTop=iy*nz+box.offsetTop-py;
  },{passive:false});
  let d=null,moved=false;
  vp.addEventListener("mousedown",e=>{ d={x:e.clientX,y:e.clientY,sl:vp.scrollLeft,st:vp.scrollTop}; moved=false; });
  window.addEventListener("mousemove",e=>{ if(!d) return; const dx=e.clientX-d.x, dy=e.clientY-d.y; if(!moved&&Math.hypot(dx,dy)<5) return; moved=true; vp.scrollLeft=d.sl-dx; vp.scrollTop=d.st-dy; });
  window.addEventListener("mouseup",()=>{ d=null; });
  box.addEventListener("click",e=>{
    if(moved){ moved=false; return; }
    const r=box.getBoundingClientRect(), a=I.active;
    GR.pending={row:(e.clientY-r.top)/r.height*a.rows-0.5, col:(e.clientX-r.left)/r.width*a.cols-0.5};     // continuous position; centre of pixel (r,c) = (r+.5, c+.5)
    GR.err=""; grRefresh();
  });
}

/* ----- the map pane ----- */
function grInitMap(){
  const el=document.getElementById("grMap");
  if(typeof L==="undefined"){ el.innerHTML=`<div class="placeholder-card" style="margin:20px">The map needs internet access (Leaflet and Esri tiles could not be loaded).</div>`; return; }
  GR.map=L.map(el,{zoomControl:true});
  const base=mapBaseLayers(); base["Esri World Imagery"].addTo(GR.map); L.control.layers(base,null,{collapsed:true}).addTo(GR.map);
  GR.layer=L.layerGroup().addTo(GR.map);
  const b=GR.pts.length?L.latLngBounds(GR.pts.map(p=>[p.lat,p.lon])).pad(0.5):(typeof geoBounds==="function"&&geoBounds()?geoBounds():null);
  if(b) GR.map.fitBounds(b); else GR.map.setView([20,0],2);
  GR.map.on("click",e=>{
    if(!GR.pending){ GR.err="First click the point on the image (left), then the same point on the map."; grRefresh(); return; }
    GR.pts.push({row:GR.pending.row,col:GR.pending.col,lat:e.latlng.lat,lon:e.latlng.lng}); GR.pending=null; GR.err=""; grRefit();
  });
  setTimeout(()=>GR.map&&GR.map.invalidateSize(),60);
}
function grGoTo(){
  const v=(document.getElementById("grGo").value||"").split(/[,\s;]+/).filter(Boolean).map(Number);
  if(v.length<2||!v.every(Number.isFinite)||Math.abs(v[0])>90||Math.abs(v[1])>180){ GR.err="Type latitude and longitude, for example: 50.11, 8.68"; grRefresh(); return; }
  if(GR.map) GR.map.setView([v[0],v[1]],15);
}

/* ----- control points: fit, table, markers ----- */
async function grRefit(){
  const my=++GR.token;
  if(GR.pts.length<3){ GR.fit=null; grRefresh(); return; }
  try{ const f=await apiSend("POST",`/api/images/${I.active.id}/georef/fit`,{points:GR.pts}); if(my===GR.token){ GR.fit=f; GR.err=""; } }
  catch(e){ if(my===GR.token){ GR.fit=null; GR.err=e.message; } }
  if(my===GR.token) grRefresh();
}
function grRemovePoint(i){ GR.pts.splice(i,1); grRefit(); }
function grClear(){ GR.pts=[]; GR.pending=null; GR.fit=null; GR.err=""; grRefresh(); }
function grRefresh(){
  const a=I.active, f=GR.fit, n=GR.pts.length;
  const marks=GR.pts.map((p,i)=>`<div class="pxmark" style="left:${(p.col+0.5)/a.cols*100}%;top:${(p.row+0.5)/a.rows*100}%;border-color:#2563eb;background:#2563eb88">${i+1}</div>`).join("")
    +(GR.pending?`<div class="pxmark" style="left:${(GR.pending.col+0.5)/a.cols*100}%;top:${(GR.pending.row+0.5)/a.rows*100}%;border-color:#f59e0b;background:#f59e0b88">?</div>`:"");
  const mk=document.getElementById("grImgMarks"); if(mk) mk.innerHTML=marks;
  if(GR.layer){
    GR.layer.clearLayers();
    GR.pts.forEach((p,i)=>L.marker([p.lat,p.lon],{icon:L.divIcon({className:"",html:`<div class="pxmark" style="position:relative;border-color:#2563eb;background:#2563eb88">${i+1}</div>`,iconSize:[16,16],iconAnchor:[8,8]})}).addTo(GR.layer));
  }
  const rows=GR.pts.map((p,i)=>`<tr><td>${i+1}</td><td class="num">${p.row.toFixed(1)}</td><td class="num">${p.col.toFixed(1)}</td><td class="num">${p.lat.toFixed(6)}</td><td class="num">${p.lon.toFixed(6)}</td>
     <td class="num ${f&&!f.exact_fit&&f.residuals_m[i]>Math.max(3*f.rmse_m,1)?"bad":""}">${f?(f.exact_fit?"—":f.residuals_m[i].toFixed(1)+" m"):"—"}</td><td><button class="coll-x" onclick="grRemovePoint(${i})" title="Remove this point">&#10005;</button></td></tr>`).join("");
  const status=n<3?`<b>${n}</b> control point${n===1?"":"s"} — at least <b>3</b> are needed.`
    :(f?(f.exact_fit?`3 points fit exactly (no check possible) — add a 4th point to see how well the image matches the map. Pixel size ≈ ${f.pixel_size_m[0].toFixed(1)} × ${f.pixel_size_m[1].toFixed(1)} m.`
        :`<b>RMSE ${f.rmse_m.toFixed(1)} m</b> over ${f.n} points · pixel size ≈ ${f.pixel_size_m[0].toFixed(1)} × ${f.pixel_size_m[1].toFixed(1)} m.${f.rmse_m>Math.max(3*Math.min(...f.pixel_size_m),5)?" The error is large: a point with a much larger error than the others is probably misplaced — remove it and click it again.":""}`):"Fitting…");
  const t=document.getElementById("grTable");
  if(t) t.innerHTML=`<div style="display:flex;gap:16px;align-items:flex-start;margin-top:8px"><div style="flex:1;max-height:150px;overflow:auto"><table class="gcp-table" style="margin-top:0"><tr><th>#</th><th>Image row</th><th>Image col</th><th>Latitude</th><th>Longitude</th><th>Error</th><th></th></tr>${rows||`<tr><td colspan=7 class="note">No points yet.</td></tr>`}</table></div>
     <div class="small" style="width:340px">${status}${GR.pending?`<br><span style="color:#b45309">Now click the same point on the map.</span>`:""}</div></div>`;
  const ap=document.getElementById("grApply"); if(ap) ap.disabled=n<3||!f||GR.fileGeo;
  const er=document.getElementById("grErr"); if(er) er.textContent=GR.err||"";
}
async function grApply(){
  try{
    const r=await apiSend("PUT",`/api/images/${I.active.id}/georef`,{points:GR.pts});
    I.geo=r.geo; grClose(); showToast(`Georeferenced with ${r.georef.n} control points (RMSE ${r.georef.rmse_m.toFixed(1)} m)`);
    refreshAll(); if(typeof objRefreshCoords==="function") objRefreshCoords();
  }catch(e){ GR.err=e.message; grRefresh(); }
}
async function grRemoveSaved(){
  try{ await apiSend("DELETE",`/api/images/${I.active.id}/georef`); I.geo=await apiGet(`/api/images/${I.active.id}/geo`); if(I.mapOn&&!geoCorners()) I.mapOn=false;
    grClose(); showToast("Saved georeference removed"); refreshAll(); }
  catch(e){ GR.err=e.message; grRefresh(); }
}
