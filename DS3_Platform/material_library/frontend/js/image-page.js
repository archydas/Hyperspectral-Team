/* image-page.js — the Image page: toolbar, image view with wheel zoom and drag pan, side panel, map view
   (classic script, loaded in order from index.html; all functions are global)

   Layout:   toolbar rows (open / bands / tools + Show checkboxes + zoom)
             [ image view ........................ | chart + collected spectra ]
             [ object sample collection (ROI) — full width, see object-samples.js ]
*/

/* ---------- zoom: mouse wheel (around the pointer), drag to pan, buttons, Fit ---------- */
const ZOOM={ maxPxPerPixel:24, detailTimer:null };

function fitZoom(){
  const pv=I.preview, vp=document.getElementById("imgVp");
  const w=vp?vp.clientWidth-4:620, h=vp?vp.clientHeight-4:560;
  return Math.max(0.05,Math.min(w/pv.width,h/pv.height,8));
}
function zoomLimits(){ const f=fitZoom(); return [Math.min(f*0.6,0.5), Math.max(f,ZOOM.maxPxPerPixel*I.active.cols/I.preview.width)]; }

/* change the zoom without redrawing the page, keeping the point under the pointer where it is */
function zoomBy(factor,clientX,clientY){
  const vp=document.getElementById("imgVp"), box=document.getElementById("imgBox"); if(!vp||!box||!I.preview) return;
  const [lo,hi]=zoomLimits(), old=I.zoom, nz=Math.max(lo,Math.min(hi,old*factor)); if(Math.abs(nz-old)<1e-9) return;
  const r=vp.getBoundingClientRect(), px=(clientX==null?r.left+r.width/2:clientX)-r.left, py=(clientY==null?r.top+r.height/2:clientY)-r.top;
  const ix=(vp.scrollLeft+px-box.offsetLeft)/old, iy=(vp.scrollTop+py-box.offsetTop)/old;      // point under the pointer, in preview pixels
  setBoxZoom(nz);
  vp.scrollLeft=ix*nz+box.offsetLeft-px; vp.scrollTop=iy*nz+box.offsetTop-py;
  scheduleDetail();
}
function setBoxZoom(z){
  I.zoom=z; const box=document.getElementById("imgBox"); if(!box) return;
  box.style.width=Math.round(I.preview.width*z)+"px";
  const img=box.querySelector("img"); if(img) img.style.imageRendering=(z*I.preview.width/I.active.cols)>2?"pixelated":"auto";
  updateInfo();
}
function setZoom(f){ if(!I.preview) return; if(f==="fit"){ const vp=document.getElementById("imgVp"); setBoxZoom(fitZoom()); if(vp){ vp.scrollLeft=0; vp.scrollTop=0; } scheduleDetail(); } else zoomBy(f); }

/* zoomed in beyond the drawn detail -> ask the backend for a sharper preview (up to ~4 million pixels), same size on screen */
function scheduleDetail(){ clearTimeout(ZOOM.detailTimer); ZOOM.detailTimer=setTimeout(loadDetail,350); }
async function loadDetail(){
  const a=I.active, pv=I.preview; if(!a||!pv||I.mapOn||pv.step<=1) return;
  const shown=pv.width*I.zoom;                                   // width on screen (css px)
  if(shown<pv.width*1.3) return;                                 // still sharp enough
  const id=a.id, b=I.bands;
  try{
    const np=await apiGet(`/api/images/${id}/preview?r=${b[0]}&g=${b[1]}&b=${b[2]}&max_px=4000000`);
    if(!I.active||I.active.id!==id||np.width<=pv.width) return;
    const vp=document.getElementById("imgVp"), box=document.getElementById("imgBox"); if(!vp||!box) return;
    const fx=(vp.scrollLeft+vp.clientWidth/2-box.offsetLeft)/shown, fy=(vp.scrollTop+vp.clientHeight/2-box.offsetTop)/(shown*pv.height/pv.width);
    I.preview=np; I.previewA=null; const img=box.querySelector("img"); if(img) img.src=np.image;
    setBoxZoom(shown/np.width);
    const nh=shown*np.height/np.width;
    vp.scrollLeft=fx*shown+box.offsetLeft-vp.clientWidth/2; vp.scrollTop=fy*nh+box.offsetTop-vp.clientHeight/2;
  }catch(_){}
}
function infoLine(){
  const a=I.active, pv=I.preview; if(!a||!pv) return "";
  const pct=Math.round(pv.width*I.zoom/a.cols*100);
  return `${escapeHtml(a.name)} · ${a.rows}×${a.cols}×${a.bands} · ${escapeHtml(a.unit||"")} · ${a.wavelength_min.toFixed(0)}–${a.wavelength_max.toFixed(0)} nm · zoom ${pct}%${pv.step>1?` · every ${pv.step}. pixel drawn until you zoom in`:""} · mouse wheel = zoom, drag = move`
    +(I.roi.drawing?` · <b style="color:#047857">drawing ROI: ${I.roi.pts.length} point(s) — double-click or right-click to finish</b>`:"");
}
function updateInfo(){
  const el=document.getElementById("imgInfo"); if(el) el.innerHTML=infoLine();
  const z=document.getElementById("imgZoomPct"); if(z&&I.preview) z.textContent=Math.round(I.preview.width*I.zoom/I.active.cols*100)+"%";
}

/* wheel + drag are attached once, to the document, so they keep working when the view is redrawn */
(function(){
  document.addEventListener("wheel",e=>{
    const vp=e.target.closest&&e.target.closest("#imgVp"); if(!vp||state.view!=="image"||!I.preview) return;
    e.preventDefault(); zoomBy(e.deltaY<0?1.2:1/1.2,e.clientX,e.clientY);
  },{passive:false});
  let drag=null;
  document.addEventListener("mousedown",e=>{
    const vp=e.target.closest&&e.target.closest("#imgVp"); if(!vp||e.button!==0) return;
    drag={vp,x:e.clientX,y:e.clientY,sl:vp.scrollLeft,st:vp.scrollTop,moved:false};
  });
  document.addEventListener("mousemove",e=>{
    if(!drag) return; const dx=e.clientX-drag.x, dy=e.clientY-drag.y;
    if(!drag.moved&&Math.hypot(dx,dy)<5) return;
    drag.moved=true; drag.vp.classList.add("panning"); drag.vp.scrollLeft=drag.sl-dx; drag.vp.scrollTop=drag.st-dy;
  });
  document.addEventListener("mouseup",()=>{
    if(!drag) return; const moved=drag.moved; drag.vp.classList.remove("panning"); drag=null;
    if(moved){ I.panned=true; setTimeout(()=>{ I.panned=false; },60); }          // the click that ends a drag is not a pixel pick
  });
})();

/* called after the page or the view has been drawn */
function afterImageDraw(){
  if(I.fitPending&&I.preview&&document.getElementById("imgVp")){ I.fitPending=false; setBoxZoom(fitZoom()); }
  scheduleDetail();
}

/* ---------- the image view ---------- */
/* the ROI being drawn (points, line / polygon); redrawn alone on every click so the zoom and scroll position stay as they are */
function roiLayerHtml(){
  const a=I.active; if(!a||!I.roi.pts.length) return "";
  const pts=I.roi.pts.map(p=>`${p[1]},${p[0]}`).join(" ");
  return `<svg viewBox="0 0 ${a.cols} ${a.rows}" preserveAspectRatio="none" style="position:absolute;inset:0;width:100%;height:100%;pointer-events:none">
     ${I.roi.closed?`<polygon points="${pts}" fill="rgba(52,211,153,.18)" stroke="#34d399" stroke-width="2" vector-effect="non-scaling-stroke"/>`:`<polyline points="${pts}" fill="none" stroke="#34d399" stroke-width="2" vector-effect="non-scaling-stroke"/>`}
     ${I.roi.pts.map(p=>`<circle cx="${p[1]}" cy="${p[0]}" r="${Math.max(1.5,a.cols/260)}" fill="#34d399"/>`).join("")}</svg>`;
}
function refreshRoiLayer(){
  const el=document.getElementById("roiLayer"); if(!el){ refreshView(); return; }
  el.innerHTML=roiLayerHtml(); updateInfo();
  const h=document.getElementById("roiHint"); if(h) h.innerHTML=roiHint();
  const c=document.getElementById("btnClearRoi"); if(c) c.disabled=!(I.roi.pts.length||I.roi.closed);
}
function viewHtml(){
  const a=I.active, pv=I.preview;
  if(!a||!pv) return `<div class="placeholder-card" style="padding:50px 20px;text-align:center">No image yet.<br>Use <b>Open image…</b> — AVIRIS-NG (.hdr + data file together), EnMAP (.zip or .TIF), PaviaU.mat, GeoTIFF, HDF5 / NetCDF.</div>`;
  if(I.mapOn) return `<div style="display:flex;gap:10px;align-items:center;margin-bottom:6px" class="note">Map background (Esri World Imagery) · image opacity <input type="range" min="0.1" max="1" step="0.05" value="${I.mapOpacity}" oninput="setMapOpacity(this.value)" style="width:140px"> · click the map to read a pixel</div><div id="imgMap" class="img-vp" style="display:block"></div>`;
  const w=Math.round(pv.width*I.zoom);
  const mk=(I.pixel&&!I.roi.drawing)?`<div style="position:absolute;left:${(I.pixel.col+0.5)/a.cols*100}%;top:${(I.pixel.row+0.5)/a.rows*100}%;width:14px;height:14px;margin:-7px 0 0 -7px;border:2px solid #fbbf24;border-radius:50%;pointer-events:none"></div>`:"";
  return `<div id="imgVp" class="img-vp">
    <div id="imgBox" style="position:relative;width:${w}px;line-height:0;margin:auto;flex:none" onclick="onImageClick(event)" ondblclick="onImageDbl(event)" oncontextmenu="onImageCtx(event)">
      <img src="${pv.image}" style="width:100%;display:block;image-rendering:${(w/a.cols)>2?"pixelated":"auto"};user-select:none" draggable="false">
      ${I.overlay?`<img src="${I.overlay}" style="position:absolute;inset:0;width:100%;height:100%;image-rendering:pixelated;pointer-events:none">`:""}
      <div id="roiLayer">${roiLayerHtml()}</div>${objOverlay()}${collMarkers()}${mk}</div></div>
    <p class="note" id="imgInfo" style="margin-top:6px">${infoLine()}</p>
    ${I.legend?`<div style="font-size:11.5px;margin-top:8px;display:flex;flex-wrap:wrap;gap:4px 12px">${I.legend.legend.filter(l=>l.pixels>0).map(l=>`<span><span style="display:inline-block;width:10px;height:10px;background:${l.color};margin-right:4px"></span>${escapeHtml(l.name)} <span style="color:var(--muted)">${l.pixels}</span></span>`).join("")}${I.legend.unmatched?`<span style="color:var(--muted)">unclassified ${I.legend.unmatched}</span>`:""}</div>`:""}`;
}

/* ---------- the side panel: current spectrum chart + collected spectra ---------- */
function sideHtml(){
  const s=I.spec;
  return `${I.err?`<p style="color:#fca5a5;font-size:12px;margin:0 0 8px">${escapeHtml(I.err)}</p>`:""}
  <div class="chart-wrap">
    <p style="font-size:12px;color:#cbd5e1;margin:0 0 8px">${s?(s.kind==="roi"?`ROI mean · ${s.n_pixels} pixels`:`Pixel (${s.row}, ${s.col})${s.n_pixels>1?` · mean of ${s.n_pixels} px`:""}`)+` · ${escapeHtml(s.unit)}`:"Spectrum"}${(()=>{ const ll=(I.pixel&&s&&s.kind==="pixel")?pixelToLatLon(I.pixel.row,I.pixel.col):null; return ll?` · ${ll.lat.toFixed(5)}, ${ll.lon.toFixed(5)}`:""; })()}</p>
    <div id="imgChart"></div>
</div>
  ${collPanel()}`;
}

/* the two "Show" checkboxes sit in the tool line, next to Download image */
function showToggles(){
  return `<span class="show-row" style="margin:0"><b>Show</b>
    <label title="the spectra of the clicked pixels: chart and numbered markers on the image"><input type="checkbox" ${I.show.px?"checked":""} onchange="I.show.px=this.checked;showRefresh()"> Pixel spectra</label>
    <label title="the ROI object samples: chart and outlines on the image"><input type="checkbox" ${I.show.roi?"checked":""} onchange="I.show.roi=this.checked;showRefresh()"> Object samples (ROI)</label></span>`;
}
/* the two checkboxes show / hide a whole part of the page: Pixel spectra = chart + collected spectra (right) and the numbered pixel markers;
   Object samples = the table under the image and the ROI outlines */
function showRefresh(){
  const g=document.getElementById("imgGrid"), s=document.getElementById("imgSide"), o=document.getElementById("imgObjects");
  if(g) g.classList.toggle("one",!I.show.px);
  if(s){ s.style.display=I.show.px?"":"none"; s.innerHTML=I.show.px?sideHtml():""; }
  if(o){ o.style.display=I.show.roi?"":"none"; o.innerHTML=I.show.roi?objectsCardHtml():""; }
  refreshView(); if(I.show.px) drawSpectrumChart();
}
function roiHint(){ const n=I.roi.pts.length; return `<b>Drawing ROI</b> — ${n} point${n===1?"":"s"}. Click to add points · <b>double-click or right-click to finish</b>`; }

/* ---------- the page ---------- */
function renderImagePage(){
  const a=I.active, nb=I.nb;
  const bandOpts=sel=>Array.from({length:nb},(_,i)=>`<option value="${i+1}" ${sel===i+1?"selected":""}>Band ${i+1} · ${I.wl[i]!==undefined?Math.round(I.wl[i])+" nm":""}</option>`).join("");
  const imgOpts=I.list.map(i=>`<option value="${i.id}" ${a&&a.id===i.id?"selected":""}>${escapeHtml(i.name)} (${i.rows}×${i.cols}×${i.bands})</option>`).join("");
  const has=!!(a&&I.preview), geoOk=!!geoCorners();
  return `<h1 class="pagetitle" style="margin-bottom:4px">Image</h1>
  ${I.busy?`<p class="note">&#8987; ${escapeHtml(I.busy)}</p>`:""}
  <div class="tool-row">
    <button class="btn emerald btn-sm" onclick="openImageDialog()">Open image…</button>
    <select onchange="openImage(this.value)" style="min-width:220px;max-width:340px">${imgOpts||"<option>(no images yet)</option>"}</select>
    ${has?`<button class="btn ghost btn-sm" onclick="deleteImage()">Remove</button>
    <span class="tool-sep"></span>
    ${["R","G","B"].map((c,k)=>`<span class="small">${c}</span><select id="bandSel${k}" style="max-width:150px">${bandOpts(I.bands[k])}</select>`).join("")}
    <button class="btn ghost btn-sm" onclick="applyBands()">Apply bands</button>
    <select id="imgWin" onchange="I.win=parseInt(this.value,10)" title="pixel window: average of 1, 3×3 or 5×5 pixels per click"><option value="1" ${I.win==1?"selected":""}>1 px</option><option value="3" ${I.win==3?"selected":""}>3×3</option><option value="5" ${I.win==5?"selected":""}>5×5</option></select>
    ${I.overlay?`<button class="btn ghost btn-sm" onclick="clearOverlay()">Clear class overlay</button>`:""}`:""}
  </div>
  ${has?`  <div class="tool-row">
    <button id="btnGeo" class="btn ${GR.open?"emerald":"ghost"} btn-sm" onclick="georefOpen()" title="${geoOk?"This image is georeferenced. Open to check or change the control points":"Match points of the image with the Esri map"}">📍 Georeference</button>
    <button class="btn ${I.mapOn?"emerald":"ghost"} btn-sm" onclick="toggleMap()">🗺 ${I.mapOn?"Hide map":"View map"}</button>
    <button class="btn ${I.roi.drawing?"emerald":"ghost"} btn-sm" onclick="${I.roi.drawing?"clearRoi()":"startRoi()"}" title="${I.roi.drawing?"Click again to stop drawing":"Click points on the image, double-click or right-click to finish"}">✏ Draw ROI polygon</button>
    <button id="btnClearRoi" class="btn ghost btn-sm" onclick="clearRoi()" ${(I.roi.pts.length||I.roi.closed)?"":"disabled"}>Clear ROI</button>
    ${I.roi.drawing?`<span class="small" id="roiHint" style="color:#047857">${roiHint()}</span>`:""}
    <button class="btn ghost btn-sm" onclick="downloadImage()">Download image</button>
    ${showToggles()}
    <span class="zoom-box"><button class="btn ghost btn-sm" onclick="setZoom(1/1.25)" ${I.mapOn?"disabled":""} title="Zoom out (or mouse wheel)">−</button><span class="small" id="imgZoomPct" style="min-width:42px;text-align:center">${Math.round(I.preview.width*I.zoom/a.cols*100)}%</span><button class="btn ghost btn-sm" onclick="setZoom(1.25)" ${I.mapOn?"disabled":""} title="Zoom in (or mouse wheel)">+</button><button class="btn ghost btn-sm" onclick="setZoom('fit')" ${I.mapOn?"disabled":""}>Fit</button></span>
  </div>`:""}
  <div class="img-grid ${I.show.px?"":"one"}" id="imgGrid"><div id="imgView" style="min-width:0">${viewHtml()}</div><div id="imgSide" style="${I.show.px?"":"display:none"}">${I.show.px?sideHtml():""}</div></div>
  <div id="imgObjects" style="${I.show.roi?"":"display:none"}">${I.show.roi?objectsCardHtml():""}</div>`;
}

/* ---------- map background: Esri World Imagery (or OpenStreetMap) under the image ---------- */
const MAP_TILES={
  esri:{url:"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",opt:{maxZoom:19,attribution:"Tiles © Esri — Source: Esri, Maxar, Earthstar Geographics"}},
  osm:{url:"https://tile.openstreetmap.org/{z}/{x}/{y}.png",opt:{maxZoom:19,attribution:"© OpenStreetMap"}}
};
function mapBaseLayers(){
  return {"Esri World Imagery":L.tileLayer(MAP_TILES.esri.url,MAP_TILES.esri.opt),"OpenStreetMap":L.tileLayer(MAP_TILES.osm.url,MAP_TILES.osm.opt)};
}
