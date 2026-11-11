/* georef-legacy.js — Georeferencing helpers: ground control points, affine fit, Leaflet map
   (classic script, loaded in order from index.html; all functions are global) */
/* =========================================================================
   GEOREFERENCING — ground control points, affine fit, Leaflet map
   ========================================================================= */
function det3(m){
  return m[0][0]*(m[1][1]*m[2][2]-m[1][2]*m[2][1])
       - m[0][1]*(m[1][0]*m[2][2]-m[1][2]*m[2][0])
       + m[0][2]*(m[1][0]*m[2][1]-m[1][1]*m[2][0]);
}
function solve3x3(M, rhs){
  const D = det3(M);
  if(Math.abs(D) < 1e-9) return null; // singular — points are collinear
  const Mx = [[rhs[0],M[0][1],M[0][2]],[rhs[1],M[1][1],M[1][2]],[rhs[2],M[2][1],M[2][2]]];
  const My = [[M[0][0],rhs[0],M[0][2]],[M[1][0],rhs[1],M[1][2]],[M[2][0],rhs[2],M[2][2]]];
  const Mz = [[M[0][0],M[0][1],rhs[0]],[M[1][0],M[1][1],rhs[1]],[M[2][0],M[2][1],rhs[2]]];
  return [det3(Mx)/D, det3(My)/D, det3(Mz)/D];
}
function computeAffine(gcps){
  if(!gcps || gcps.length<3) return null;
  let sxx=0,sxy=0,sx=0,syy=0,sy=0,n=gcps.length;
  let sxLat=0,syLat=0,sLat=0, sxLon=0,syLon=0,sLon=0;
  gcps.forEach(g=>{
    const x=g.px, y=g.py;
    sxx+=x*x; sxy+=x*y; sx+=x; syy+=y*y; sy+=y;
    sxLat+=x*g.lat; syLat+=y*g.lat; sLat+=g.lat;
    sxLon+=x*g.lon; syLon+=y*g.lon; sLon+=g.lon;
  });
  const M = [[sxx,sxy,sx],[sxy,syy,sy],[sx,sy,n]];
  const latSol = solve3x3(M, [sxLat, syLat, sLat]);
  const lonSol = solve3x3(M, [sxLon, syLon, sLon]);
  if(!latSol || !lonSol) return null;
  return { a:latSol[0], b:latSol[1], c:latSol[2], d:lonSol[0], e:lonSol[1], f:lonSol[2] };
}
function pixelLatLon(x,y){
  if(!imgState.geo) return null;
  const {a,b,c,d,e,f} = imgState.geo;
  return { lat: a*x+b*y+c, lon: d*x+e*y+f };
}
function residualMeters(fit, g){
  const pred = { lat: fit.a*g.px+fit.b*g.py+fit.c, lon: fit.d*g.px+fit.e*g.py+fit.f };
  const dLatM = (pred.lat-g.lat)*111320;
  const dLonM = (pred.lon-g.lon)*111320*Math.cos(g.lat*Math.PI/180);
  return Math.sqrt(dLatM*dLatM + dLonM*dLonM);
}
function computeRmse(){
  if(!imgState.geo || imgState.gcps.length===0) return 0;
  const sum = imgState.gcps.reduce((s,g)=>s + Math.pow(residualMeters(imgState.geo,g),2), 0);
  return Math.sqrt(sum/imgState.gcps.length);
}
function clearImageGeoreference(){
  imgState.geo = null;
  imgState.gcps = [];
  renderMainOnly();
}

function renderGeorefPanel(){
  return `
  <div class="crumb" onclick="cancelGeoref()">&larr; Image</div>
  <h1 class="pagetitle">Georeference image</h1>
  <p class="pagesub">Click a point on your image, then click the matching real-world spot on the satellite map. Repeat for at least 3 points, spread across the image, for a good fit. Uses a live satellite basemap (Esri World Imagery) — needs an internet connection.</p>

  <div class="georef-grid">
    <div>
      <p style="font-size:11.5px; color:var(--muted); margin:0 0 6px;">Your image — click to place a point</p>
      <canvas id="geoImgCanvas" width="${imgState.tiff ? imgState.tiff.width : 480}" height="${imgState.tiff ? imgState.tiff.height : 270}" onclick="handleGeoImageClick(event)"></canvas>
    </div>
    <div>
      <p style="font-size:11.5px; color:var(--muted); margin:0 0 6px;">Satellite map — click the matching location</p>
      <div id="geoMapDiv"></div>
    </div>
  </div>

  <p id="geoHint" style="font-size:12px; color:${imgState.pendingImgPoint ? '#fbbf24' : 'var(--muted)'}; margin-top:10px;">
    ${imgState.pendingImgPoint ? 'Point placed on the image — now click the matching spot on the satellite map →' : 'Click a point on the image to begin ←'}
  </p>

  <table class="gcp-table">
    <thead><tr><th>#</th><th>Image (x, y)</th><th>Lat, Lon</th><th>Error</th><th></th></tr></thead>
    <tbody id="gcpTableBody">${buildGcpTableRows()}</tbody>
  </table>

  <div style="display:flex; gap:10px; margin-top:14px;">
    <button class="btn emerald btn-sm" id="applyGeorefBtn" onclick="applyGeoreference()" ${imgState.gcps.length<3 ? 'disabled' : ''}>Compute &amp; apply (${imgState.gcps.length}/3 points min)</button>
    <button class="btn ghost btn-sm" onclick="cancelGeoref()">Cancel</button>
  </div>
  `;
}
function buildGcpTableRows(){
  if(imgState.gcps.length===0) return `<tr><td colspan="5" style="color:var(--muted);">No points yet.</td></tr>`;
  const fit = imgState.gcps.length>=3 ? computeAffine(imgState.gcps) : null;
  return imgState.gcps.map((g,i)=>{
    const errCell = fit ? (()=>{ const err=residualMeters(fit,g); return `<span class="${err>50?'bad':''}">${err.toFixed(1)} m</span>`; })() : '—';
    return `<tr>
      <td>${i+1}</td>
      <td>${g.px}, ${g.py}</td>
      <td>${g.lat.toFixed(5)}, ${g.lon.toFixed(5)}</td>
      <td>${errCell}</td>
      <td><button class="btn ghost btn-sm" onclick="removeGcp(${g.id})">Remove</button></td>
    </tr>`;
  }).join("");
}
function refreshGeorefUI(){
  drawGeoImageCanvas();
  const tbody = document.getElementById("gcpTableBody");
  if(tbody) tbody.innerHTML = buildGcpTableRows();
  const btn = document.getElementById("applyGeorefBtn");
  if(btn){ btn.disabled = imgState.gcps.length<3; btn.textContent = `Compute & apply (${imgState.gcps.length}/3 points min)`; }
  const hint = document.getElementById("geoHint");
  if(hint){
    hint.textContent = imgState.pendingImgPoint
      ? "Point placed on the image — now click the matching spot on the satellite map →"
      : "Click a point on the image to begin ←";
    hint.style.color = imgState.pendingImgPoint ? "#fbbf24" : "var(--muted)";
  }
}
function initGeorefMap(){
  if(georefMap){ try{ georefMap.remove(); }catch(e){} georefMap=null; }
  const last = imgState.gcps[imgState.gcps.length-1];
  const center = last ? [last.lat, last.lon] : [20.5937, 78.9629];
  georefMap = L.map('geoMapDiv').setView(center, last ? 12 : 5);
  L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
    attribution: 'Tiles &copy; Esri — Source: Esri, Maxar, Earthstar Geographics, and the GIS User Community',
    maxZoom: 19,
  }).addTo(georefMap);
  georefMarkersLayer = L.layerGroup().addTo(georefMap);
  imgState.gcps.forEach((g,i)=>{
    L.marker([g.lat,g.lon]).addTo(georefMarkersLayer).bindTooltip(String(i+1), {permanent:true, direction:'top'});
  });
  georefMap.on('click', onGeorefMapClick);
}
function drawGeoImageCanvas(){
  const canvas = document.getElementById("geoImgCanvas");
  const t = imgState.tiff;
  const source = t ? t.previewCanvas : imgState.imgEl;
  if(!canvas || !source) return;
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0,0,canvas.width,canvas.height);
  ctx.drawImage(source,0,0,canvas.width,canvas.height);
  imgState.gcps.forEach((g,i)=>{
    ctx.beginPath(); ctx.arc(g.px,g.py,5,0,Math.PI*2);
    ctx.fillStyle="#34d399"; ctx.fill();
    ctx.strokeStyle="#022c22"; ctx.lineWidth=1.5; ctx.stroke();
    ctx.fillStyle="#e2e8f0"; ctx.font="bold 10px sans-serif";
    ctx.fillText(String(i+1), g.px+7, g.py-7);
  });
  if(imgState.pendingImgPoint){
    const {x,y} = imgState.pendingImgPoint;
    ctx.beginPath(); ctx.arc(x,y,5,0,Math.PI*2);
    ctx.fillStyle="#fbbf24"; ctx.fill();
    ctx.strokeStyle="#78350f"; ctx.lineWidth=1.5; ctx.stroke();
  }
}
function handleGeoImageClick(e){
  const canvas = document.getElementById("geoImgCanvas");
  const rect = canvas.getBoundingClientRect();
  const x = Math.round((e.clientX-rect.left)/rect.width*canvas.width);
  const y = Math.round((e.clientY-rect.top)/rect.height*canvas.height);
  imgState.pendingImgPoint = {x,y};
  refreshGeorefUI();
}
function onGeorefMapClick(e){
  if(!imgState.pendingImgPoint){
    showToast("Click a point on the image first");
    return;
  }
  const gcp = { id: Date.now(), px: imgState.pendingImgPoint.x, py: imgState.pendingImgPoint.y, lat: e.latlng.lat, lon: e.latlng.lng };
  imgState.gcps.push(gcp);
  imgState.pendingImgPoint = null;
  L.marker([gcp.lat,gcp.lon]).addTo(georefMarkersLayer).bindTooltip(String(imgState.gcps.length), {permanent:true, direction:'top'});
  refreshGeorefUI();
}
function removeGcp(id){
  imgState.gcps = imgState.gcps.filter(g=>g.id!==id);
  initGeorefMap();
  refreshGeorefUI();
}
function applyGeoreference(){
  const fit = computeAffine(imgState.gcps);
  if(!fit){ showToast("Need at least 3 points that aren't in a straight line"); return; }
  imgState.geo = fit;
  imgState.pendingImgPoint = null;
  if(georefMap){ try{ georefMap.remove(); }catch(e){} georefMap=null; }
  showToast(`Image georeferenced with ${imgState.gcps.length} control points`);
  setView("image");
}
function cancelGeoref(){
  imgState.pendingImgPoint = null;
  if(georefMap){ try{ georefMap.remove(); }catch(e){} georefMap=null; }
  setView("image");
}

function resetImage(){
  imgState = {
    imgEl:null, pixel:null, mode:"true", zoom:1,
    bands:{ r:"Band 30", g:"Band 20", b:"Band 10" },
    fileName:null, naturalW:null, naturalH:null,
    pixelHistory:[], showSaveForm:false,
    geo:null, gcps:[], pendingImgPoint:null,
    tiff:null, labels:[], labelInput:"",
    roi:{ active:false, points:[], closed:false }, roiResult:null,
  };
  if(georefMap){ try{ georefMap.remove(); }catch(e){} georefMap=null; }
  renderMainOnly();
  showToast("Image cleared");
}
function handleCanvasClick(e){
  const canvas = document.getElementById("imgCanvas");
  const rect = canvas.getBoundingClientRect();
  const x = Math.round((e.clientX-rect.left)/rect.width*canvas.width);
  const y = Math.round((e.clientY-rect.top)/rect.height*canvas.height);

  if(imgState.roi.active && !imgState.roi.closed){
    imgState.roi.points.push({x,y});
    const counter = document.getElementById("roiPointCount");
    if(counter) counter.textContent = `${imgState.roi.points.length} point${imgState.roi.points.length===1?'':'s'}`;
    redrawCanvasFast();
    return;
  }

  const ctx = canvas.getContext("2d");
  const [r,g,b] = ctx.getImageData(x,y,1,1).data;
  const seed = r*0.3 + g*0.59 + b*0.11 + x + y;
  const geo = pixelLatLon(x,y);
  imgState.pixel = {x,y,seed,geo};
  imgState.showSaveForm = false;
  imgState.pixelHistory = [{x,y,seed}, ...imgState.pixelHistory.filter(p=>!(p.x===x&&p.y===y))].slice(0,6);
  state.spectraExtracted++;
  document.getElementById("pixelPanel").innerHTML = renderPixelPanel();
  renderRecent();
}
function redrawCanvasFast(){
  // Cheap redraw for ROI vertex placement — reuses the last-composited pixels
  // (from the cache renderTiffComposite() already keeps) instead of re-fetching
  // and re-stretching bands on every single click.
  const canvas = document.getElementById("imgCanvas");
  if(!canvas) return;
  const t = imgState.tiff;
  if(t && t.previewCanvas){
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0,0,canvas.width,canvas.height);
    ctx.drawImage(t.previewCanvas,0,0);
    drawLabelMarkers(ctx);
    drawRoiOverlay(ctx);
  } else {
    drawCanvas();
  }
}
function revisitPixel(x,y,seed){
  imgState.pixel = {x,y,seed,geo:pixelLatLon(x,y)};
  imgState.showSaveForm = false;
  document.getElementById("pixelPanel").innerHTML = renderPixelPanel();
}
function saveCurrentPixelToLibrary(){
  const t = imgState.tiff;
  if(!(t && t.cube)){ showToast("Load the full spectral cube first — only real pixel spectra can be saved to the library"); return; }
  const libId = document.getElementById("saveLibSelect").value;
  const name = (document.getElementById("saveMatName").value || "").trim() || `Pixel (${imgState.pixel.x}, ${imgState.pixel.y})`;
  const latitude = document.getElementById("saveMatLat").value, longitude = document.getElementById("saveMatLon").value;
  const locationName = document.getElementById("saveMatLoc").value;
  const {x,y} = imgState.pixel, data = tiffPixelSpectrum(x,y);
  const lib = state.libraries.find(l=>l.id===libId);
  const existing = lib && lib.materials.find(m=>m.name.toLowerCase()===name.toLowerCase());
  if(existing){
    if(!existing.user){ showToast(`"${existing.name}" is a placeholder demo material — pick a different name`); return; }
    try{ addSpectrum(libId, existing.id, {data, origin:"image_pixel", pixel:[x,y]}); }catch(e){ showToast(e.message); return; }
    persistUserLibrary();
    showToast(`Real spectrum added to sample "${existing.name}" (${existing.spectra.length} spectra)`);
  } else {
    if(!addMaterial(libId, {name, type:"Manmade", samples:data.length, customData:data, latitude, longitude, locationName, sourceFormat:"Image pixel (real spectrum)", origin:"image_pixel", pixel:[x,y]})) return;
  }
  imgState.showSaveForm = false; renderMainOnly();
}
