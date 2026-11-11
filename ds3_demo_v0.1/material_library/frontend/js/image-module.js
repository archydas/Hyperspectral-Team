/* image-module.js — Image module: opening images, bands, zoom, pixel <-> lat/lon, drawing
   (classic script, loaded in order from index.html; all functions are global) */
/* ===================== IMAGE MODULE ===================== */
const I = { list:[], active:null, preview:null, previewA:null, nb:0, wl:[], bands:[1,1,1], zoom:1, win:1,
  pixel:null, spec:null, roi:{pts:[],closed:false,drawing:false}, geo:null,
  mapOn:false, mapOpacity:0.75, map:null, libId:"", libSpectra:[], libTotal:0, libQ:"", picked:new Set(),
  matches:null, overlay:null, legend:null, note:"", err:"", busy:"" };
const I_COLORS=["#34d399","#60a5fa","#f472b6","#fbbf24","#a78bfa","#fb923c","#38bdf8","#f87171"];
const specCache={};

/* ---- opening images ---- */
async function openImageView(){
  state.view="image"; render();
  try{ await loadImages(); }catch(e){ I.err=e.message; refreshAll(); }
}
async function loadImages(selectId){
  I.list=await apiGet("/api/images/");
  if(selectId){ await openImage(selectId); return; }
  if(!I.active && I.list.length){ await openImage(I.list[0].id); return; }
  refreshAll();
}
async function openImage(id,bands){
  I.busy="Opening image…"; I.err=""; refreshAll();
  try{
    const same=I.active&&I.active.id===id;
    I.active=I.list.find(i=>i.id===id)||await apiGet(`/api/images/${id}`);
    if(!same){ if(I.coll.length) showToast(`${I.coll.length} collected spectra cleared: they belong to the previous image (save them to a library first)`); I.coll=[]; I.collN=0; I.angles=false; }
    Object.assign(I,{pixel:null,spec:null,roi:{pts:[],closed:false,drawing:false},overlay:null,legend:null,matches:null,previewA:null,mapOn:false,note:""});
    const [pv,bd,geo]=await Promise.all([apiGet(`/api/images/${id}/preview${bands?`?r=${bands[0]}&g=${bands[1]}&b=${bands[2]}`:""}`),apiGet(`/api/images/${id}/bands`),apiGet(`/api/images/${id}/geo`)]);
    I.preview=pv; I.nb=I.active.bands; I.wl=bd.wavelengths; I.bands=pv.bands.slice(); I.geo=geo;
    I.zoom=Math.max(0.1,Math.min(4,620/pv.width)); I.fitPending=true; I.busy="";
  }catch(e){ I.busy=""; I.err=e.message; }
  refreshAll();
}
async function deleteImage(){
  if(!I.active||!confirm("Remove this image from the list? (the file on disk is not deleted)")) return;
  try{ await apiSend("DELETE",`/api/images/${I.active.id}`); I.active=null; I.preview=null; await loadImages(); }catch(e){ I.err=e.message; refreshAll(); }
}

/* ---- bands, zoom ---- */
async function applyBands(){
  const r=[0,1,2].map(k=>parseInt(document.getElementById("bandSel"+k).value,10));
  I.busy="Updating view…"; refreshAll();
  try{
    const shown=I.preview.width*I.zoom; I.preview=await apiGet(`/api/images/${I.active.id}/preview?r=${r[0]}&g=${r[1]}&b=${r[2]}`); I.zoom=shown/I.preview.width; I.bands=r; I.previewA=null;
    if(I.mapOn) await ensureAlphaPreview();
  }catch(e){ I.err=e.message; }
  I.busy=""; refreshAll();
}

/* ---- geometry: pixel <-> lat/lon ---- */
function geoCorners(){ return I.geo&&I.geo.has_geo?I.geo.corners:null; }     // from the file, or from the control points (see georef.js)
function pixelToLatLon(row,col){
  const c=geoCorners(); if(!c||!I.active) return null;
  const fx=(col+0.5)/I.active.cols, fy=(row+0.5)/I.active.rows;
  return {lat:c[0].lat+fx*(c[1].lat-c[0].lat)+fy*(c[3].lat-c[0].lat), lon:c[0].lon+fx*(c[1].lon-c[0].lon)+fy*(c[3].lon-c[0].lon)};
}
function latLonToPixel(lat,lon){
  const c=geoCorners(); if(!c) return null;
  const ax=c[1].lat-c[0].lat, ay=c[1].lon-c[0].lon, bx=c[3].lat-c[0].lat, by=c[3].lon-c[0].lon, px=lat-c[0].lat, py=lon-c[0].lon;
  const det=ax*by-ay*bx; if(!det) return null;
  const fx=(px*by-py*bx)/det, fy=(ax*py-ay*px)/det;
  if(fx<0||fx>=1||fy<0||fy>=1) return null;
  return {row:Math.floor(fy*I.active.rows), col:Math.floor(fx*I.active.cols)};
}
function geoBounds(){ const c=geoCorners(); if(!c) return null; const la=c.map(p=>p.lat), lo=c.map(p=>p.lon); return [[Math.min(...la),Math.min(...lo)],[Math.max(...la),Math.max(...lo)]]; }

/* ---- picking pixels and drawing the ROI ---- */
function imgCoords(ev){
  const el=document.getElementById("imgBox"), r=el.getBoundingClientRect();
  const fx=(ev.clientX-r.left)/r.width, fy=(ev.clientY-r.top)/r.height;
  return {row:Math.min(I.active.rows-1,Math.max(0,Math.floor(fy*I.active.rows))), col:Math.min(I.active.cols-1,Math.max(0,Math.floor(fx*I.active.cols)))};
}
function target(){ return I.roi.closed ? {polygon:I.roi.pts} : (I.pixel?{row:I.pixel.row,col:I.pixel.col,size:I.win}:null); }

/* ---- map in the background ---- */
async function ensureAlphaPreview(){
  const b=I.bands; I.previewA=await apiGet(`/api/images/${I.active.id}/preview?r=${b[0]}&g=${b[1]}&b=${b[2]}&alpha=true`);
}
async function toggleMap(){
  if(I.mapOn){ I.mapOn=false; destroyMap(); refreshAll(); return; }
  if(!geoCorners()){ showToast("This image has no map coordinates yet: match at least 3 points with the map first."); georefOpen(); return; }
  if(typeof L==="undefined"){ I.err="The map needs internet access (Leaflet and OpenStreetMap could not be loaded)."; refreshAll(); return; }
  I.err=""; I.busy="Preparing the map…"; refreshAll();
  try{ await ensureAlphaPreview(); I.mapOn=true; }catch(e){ I.err=e.message; }
  I.busy=""; refreshAll();
}
function destroyMap(){ if(I.map){ try{ I.map.remove(); }catch(_){} I.map=null; I.mapLayers={}; } }
function initMap(){
  destroyMap(); const el=document.getElementById("imgMap"); if(!el||!I.mapOn||!I.previewA) return;
  const b=geoBounds(); I.map=L.map(el,{zoomControl:true});
  const base=mapBaseLayers(); base["Esri World Imagery"].addTo(I.map); L.control.layers(base,null,{collapsed:true}).addTo(I.map);
  I.mapLayers={img:L.imageOverlay(I.previewA.image,b,{opacity:I.mapOpacity}).addTo(I.map)};
  if(I.overlay) I.mapLayers.cls=L.imageOverlay(I.overlay,b,{opacity:0.8}).addTo(I.map);
  I.map.fitBounds(b);
  I.map.on("click",async e=>{ const p=latLonToPixel(e.latlng.lat,e.latlng.lng); if(p) await pickPixel(p.row,p.col); else showToast("That point is outside the image"); });
  updateMapMarker();
}
function updateMapMarker(){
  if(!I.map||!I.pixel) return; const ll=pixelToLatLon(I.pixel.row,I.pixel.col); if(!ll) return;
  if(I.mapLayers.mk) I.map.removeLayer(I.mapLayers.mk);
  I.mapLayers.mk=L.circleMarker([ll.lat,ll.lon],{radius:6,color:"#fbbf24",weight:2,fillOpacity:0.2}).addTo(I.map);
}
function setMapOpacity(v){ I.mapOpacity=+v; if(I.mapLayers&&I.mapLayers.img) I.mapLayers.img.setOpacity(I.mapOpacity); }

/* ---- library spectra drawn next to the image spectrum (used by Spectral > Spectral Analyst: best matches) ---- */
function libById(id){ return state.libraries.find(l=>l.id===id); }
async function addLibraryToImage(id){
  if(!id) return; I.libId=id; I.picked=new Set(); I.matches=null; I.libQ=""; await loadLibSpectra();
}
async function loadLibSpectra(){
  if(!I.libId){ I.libSpectra=[]; I.libTotal=0; return; }
  try{ const r=await apiGet(`/api/spectra/search?library_id=${I.libId}&limit=500${I.libQ?`&q=${encodeURIComponent(I.libQ)}`:""}`); I.libSpectra=r.results; I.libTotal=r.total; }
  catch(e){ I.err=e.message; }
}

function clearOverlay(){ I.overlay=null; I.legend=null; refreshAll(); }
function exportSpectrum(){
  const s=I.spec; if(!s) return; const rows=s.wavelengths.map((w,i)=>[w,s.values[i]]).filter(r=>r[1]!==null);
  const a=document.createElement("a"); a.href=URL.createObjectURL(new Blob([`wavelength_nm,${s.unit}\n`+rows.map(r=>r.join(",")).join("\n")],{type:"text/csv"}));
  a.download=(s.kind==="roi"?"roi_mean":`pixel_${s.row}_${s.col}`)+".csv"; a.click();
}
function downloadImage(){ if(!I.preview) return; const a=document.createElement("a"); a.href=I.preview.image; a.download=(I.active.name||"image")+"_preview.png"; a.click(); }

/* ---- drawing ---- */
function refreshAll(){          // redraw the whole Image page, keeping the scroll position of the image view
  if(state.view!=="image") return;
  const old=document.getElementById("imgVp"), sl=old?old.scrollLeft:0, st=old?old.scrollTop:0;
  renderMainOnly();
  const vp=document.getElementById("imgVp"); if(vp&&old&&!I.fitPending){ vp.scrollLeft=sl; vp.scrollTop=st; }
}
function refreshView(){          // redraw the image view but keep the scroll position (the zoom lives in the box width, which is unchanged)
  const v=document.getElementById("imgView"); if(!v){ refreshAll(); return; }
  const old=document.getElementById("imgVp"), sl=old?old.scrollLeft:0, st=old?old.scrollTop:0;
  v.innerHTML=viewHtml();
  const vp=document.getElementById("imgVp"); if(vp&&old){ vp.scrollLeft=sl; vp.scrollTop=st; }
  if(I.mapOn) initMap(); else afterImageDraw();
}
function refreshSide(keepList){ const s=document.getElementById("imgSide"); if(s&&!I.show.px) return; if(!s){ refreshAll(); return; } s.innerHTML=sideHtml(); drawSpectrumChart(); }

