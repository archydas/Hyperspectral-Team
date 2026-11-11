/* roi-labeling.js — ROI polygon, apply model to ROI, pixel labelling
   (classic script, loaded in order from index.html; all functions are global) */
/* =========================================================================
   ROI POLYGON + "APPLY MODEL TO ROI" — draw a boundary around an earth
   feature, then run a lightweight spectral-angle match (against a chosen
   library's materials) on the pixels inside it. This is a stand-in for the
   real Analysis module, not a trained classifier — the point is to prove
   out the ROI-scoping mechanism and let the resolution/stride control show
   a genuine, measured time-vs-detail trade-off.
   ========================================================================= */
function startRoiDraw(){
  imgState.roi = { active:true, points:[], closed:false };
  imgState.roiResult = null;
  showToast("Click points on the image to trace the ROI, then hit Finish ROI");
  renderMainOnly();
}
function finishRoiDraw(){
  if(imgState.roi.points.length<3){ showToast("Add at least 3 points first"); return; }
  imgState.roi.active=false; imgState.roi.closed=true;
  renderMainOnly();
}
function drawRoiOverlay(ctx){
  const roi = imgState.roi;
  if(roi.points.length){
    ctx.save();
    ctx.strokeStyle = "#fbbf24"; ctx.lineWidth=1.5;
    ctx.setLineDash(roi.closed ? [] : [4,3]);
    ctx.beginPath();
    roi.points.forEach((p,i)=> i===0 ? ctx.moveTo(p.x,p.y) : ctx.lineTo(p.x,p.y));
    if(roi.closed) ctx.closePath();
    ctx.stroke();
    ctx.setLineDash([]);
    roi.points.forEach(p=>{
      ctx.beginPath(); ctx.arc(p.x,p.y,3,0,Math.PI*2);
      ctx.fillStyle="#fbbf24"; ctx.fill();
    });
    ctx.restore();
  }
  if(imgState.roiResult){
    const { matches, colorFor, stride } = imgState.roiResult;
    ctx.save();
    ctx.globalAlpha = 0.55;
    matches.forEach(m=>{
      ctx.fillStyle = colorFor[m.materialName];
      ctx.fillRect(m.x, m.y, stride, stride);
    });
    ctx.restore();
  }
}
function pointInPolygon(x,y,pts){
  let inside=false;
  for(let i=0,j=pts.length-1;i<pts.length;j=i++){
    const xi=pts[i].x, yi=pts[i].y, xj=pts[j].x, yj=pts[j].y;
    const intersect = ((yi>y)!==(yj>y)) && (x < (xj-xi)*(y-yi)/(yj-yi)+xi);
    if(intersect) inside=!inside;
  }
  return inside;
}
function roiBoundingBox(pts){
  return {
    minX: Math.min(...pts.map(p=>p.x)), maxX: Math.max(...pts.map(p=>p.x)),
    minY: Math.min(...pts.map(p=>p.y)), maxY: Math.max(...pts.map(p=>p.y)),
  };
}
function getPixelSpectrumForRoi(x,y){
  const t = imgState.tiff;
  if(t && t.cube) return tiffPixelSpectrum(x,y).map(d=>d.reflectance);
  // Fallback demo mode: deterministic per-coordinate synthetic spectrum,
  // same generator the rest of the app uses when no real cube is loaded.
  const seed = (x*12.9898 + y*78.233) % 1000;
  return makeSpectrum(seed, 0.4, 2.5, 24).map(d=>d.reflectance);
}
function spectralAngle(a,b){
  let dot=0, na=0, nb=0;
  const n = Math.min(a.length,b.length);
  for(let i=0;i<n;i++){ dot+=a[i]*b[i]; na+=a[i]*a[i]; nb+=b[i]*b[i]; }
  if(na===0||nb===0) return Math.PI/2;
  const cos = Math.max(-1, Math.min(1, dot/(Math.sqrt(na)*Math.sqrt(nb))));
  return Math.acos(cos);
}
function materialReferenceVector(material, targetLength){
  if(material.customData && material.customData.length){
    const src = material.customData;
    const vec = [];
    for(let i=0;i<targetLength;i++){
      const srcIdx = Math.round(i * (src.length-1) / Math.max(targetLength-1,1));
      vec.push(src[srcIdx].reflectance);
    }
    return { vector: vec, isReal: true };
  }
  return { vector: new Array(targetLength).fill(0.0001), isReal: false };   // no data: never invented (such materials are filtered out before this)
}
async function runRoiClassification(){
  if(cnnState.roiModel==="cnn"){ return runRoiCnn(); }
  if(!imgState.roi.closed || imgState.roi.points.length<3){ showToast("Draw and finish a polygon first"); return; }
  const lib = state.libraries.find(l=>l.id===document.getElementById("roiLibSelect").value);
  const stride = parseInt(document.getElementById("roiResSelect").value,10);
  if(!lib || (lib.db ? lib.count===0 : lib.materials.length===0)){ showToast("Pick a library that has materials in it"); return; }
  if(lib.db){ showToast("Reading the library spectra from the database…"); try{ await dbHydrate(lib); }catch(e){ showToast(e.message); return; } }
  const usable = lib.materials.filter(m=>m.customData && m.customData.length);
  if(!usable.length){ showToast("This library has no reflectance spectra to classify with"); return; }

  const t = imgState.tiff;
  const canvasW = t ? t.width : 480, canvasH = t ? t.height : 270;
  const pts = imgState.roi.points;
  const bbox = roiBoundingBox(pts);
  const minX = Math.max(0, Math.floor(bbox.minX)), maxX = Math.min(canvasW-1, Math.ceil(bbox.maxX));
  const minY = Math.max(0, Math.floor(bbox.minY)), maxY = Math.min(canvasH-1, Math.ceil(bbox.maxY));

  showToast("Running classification…");
  await new Promise(res=>setTimeout(res,10)); // let the toast paint before the loop below blocks the thread

  const t0 = performance.now();
  const PALETTE = ["#34d399","#60a5fa","#f472b6","#fbbf24","#a78bfa","#fb7185","#22d3ee","#facc15"];
  const colorFor = {};
  usable.forEach((m,i)=> colorFor[m.name] = PALETTE[i % PALETTE.length]);

  let sampleLen = null;
  const refVectors = {};
  const referenceIsReal = {};
  const matches = [];
  const counts = {};
  let sampledCount = 0, totalInPoly = 0;

  for(let y=minY; y<=maxY; y+=stride){
    for(let x=minX; x<=maxX; x+=stride){
      if(!pointInPolygon(x,y,pts)) continue;
      totalInPoly++;
      const pixelVec = getPixelSpectrumForRoi(x,y);
      if(sampleLen===null){
        sampleLen = pixelVec.length;
        usable.forEach(m=>{
          const ref = materialReferenceVector(m, sampleLen);
          refVectors[m.id] = ref.vector;
          referenceIsReal[m.name] = ref.isReal;
        });
      }
      let best=null, bestAngle=Infinity;
      usable.forEach(m=>{
        const angle = spectralAngle(pixelVec, refVectors[m.id]);
        if(angle<bestAngle){ bestAngle=angle; best=m; }
      });
      matches.push({x,y,materialName:best.name});
      counts[best.name] = (counts[best.name]||0)+1;
      sampledCount++;
    }
  }
  const elapsedMs = performance.now()-t0;

  let groundInfo = null;
  if(imgState.geo){
    const p0 = pixelLatLon(0,0), px = pixelLatLon(stride,0), py = pixelLatLon(0,stride);
    const mPerDegLat = 111320, mPerDegLon = 111320*Math.cos(p0.lat*Math.PI/180);
    groundInfo = {
      dxM: Math.sqrt(Math.pow((px.lat-p0.lat)*mPerDegLat,2) + Math.pow((px.lon-p0.lon)*mPerDegLon,2)),
      dyM: Math.sqrt(Math.pow((py.lat-p0.lat)*mPerDegLat,2) + Math.pow((py.lon-p0.lon)*mPerDegLon,2)),
    };
  }

  imgState.roiResult = {
    stride, counts, colorFor, matches, elapsedMs, sampledCount, totalInPoly,
    boxW: maxX-minX+1, boxH: maxY-minY+1,
    referenceIsReal, groundInfo,
    usedRealPixelSpectra: !!(t && t.cube),
  };
  renderMainOnly();
  showToast(`Classified ${sampledCount} sampled pixels in ${elapsedMs.toFixed(0)} ms`);
}
function renderRoiResults(){
  const r = imgState.roiResult;
  if(!r) return "";
  const rows = Object.entries(r.counts).sort((a,b)=>b[1]-a[1]).map(([name,count])=>{
    const pct = ((count/r.sampledCount)*100).toFixed(1);
    const isReal = r.referenceIsReal[name];
    return `<tr>
      <td><span style="display:inline-block; width:10px; height:10px; border-radius:2px; background:${r.colorFor[name]}; margin-right:6px;"></span>${name}${isReal?'':' <span style="color:var(--muted); font-size:10.5px;">(synthetic ref)</span>'}</td>
      <td>${count}</td>
      <td>${pct}%</td>
    </tr>`;
  }).join("");
  const groundLine = r.groundInfo
    ? `Effective sampling at this resolution: ~${r.groundInfo.dxM.toFixed(1)} m × ${r.groundInfo.dyM.toFixed(1)} m per sampled point (from your georeferencing fit).`
    : `Not georeferenced — sampling shown in pixels only (1 sample every ${r.stride}px).`;
  return `
  <div style="margin-top:12px; padding-top:10px; border-top:1px solid var(--border);">
    <table class="gcp-table">
      <thead><tr><th>Matched material</th><th>Pixels</th><th>Share</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>
    <p style="font-size:11.5px; color:var(--muted); margin-top:8px;">
      ${r.sampledCount} of ${r.totalInPoly} points-in-polygon sampled (stride ${r.stride}) over a ${r.boxW}×${r.boxH}px bounding box, in <strong style="color:#e2e8f0;">${r.elapsedMs.toFixed(0)} ms</strong>.
      ${r.usedRealPixelSpectra ? '' : ' (simulated spectra — timing is still real, just not the spectra.)'}
    </p>
    <p style="font-size:11.5px; color:var(--muted); margin:2px 0 0;">${groundLine}</p>
    ${r.modelNote ? `<p style="font-size:11.5px; color:var(--muted); margin:2px 0 0;">${r.modelNote}</p>` : ""}
  </div>`;
}

/* =========================================================================
   PIXEL LABELING — click a pixel, assign it a class/material name, and
   build up a labeled training-sample dataset you can export (with real
   per-band spectra as feature columns, when a full cube is loaded).
   ========================================================================= */
function addPixelLabel(){
  const input = document.getElementById("pixelLabelInput");
  const label = input.value.trim();
  if(!label){ showToast("Type a label first"); return; }
  const {x,y,geo} = imgState.pixel;
  const existing = imgState.labels.find(l=>l.x===x && l.y===y);
  if(existing){
    existing.label = label;
  } else {
    imgState.labels.push({ id:Date.now(), x, y, label, lat: geo?geo.lat:null, lon: geo?geo.lon:null });
  }
  imgState.labelInput = "";
  showToast(`Labeled (${x}, ${y}) as "${label}"`);
  renderMainOnly();
}
function removeLabelPoint(id){
  imgState.labels = imgState.labels.filter(l=>l.id!==id);
  renderMainOnly();
}
function exportLabeledDataset(){
  if(imgState.labels.length===0){ showToast("No labeled points yet"); return; }
  const t = imgState.tiff;
  const hasSpectra = t && t.cube;
  let header = "id,x,y,lat,lon,label";
  if(hasSpectra){
    const wavelengths = t.cube.map((_,bi)=> t.wavelengths ? t.wavelengths[bi] : (bi+1));
    header += "," + wavelengths.map(w=>`wl_${w}`).join(",");
  }
  const rows = imgState.labels.map((l,i)=>{
    let row = `${i+1},${l.x},${l.y},${l.lat??''},${l.lon??''},"${l.label.replace(/"/g,'""')}"`;
    if(hasSpectra){
      const spec = tiffPixelSpectrum(l.x,l.y);
      row += "," + spec.map(d=>d.reflectance).join(",");
    }
    return row;
  });
  const csv = header + "\n" + rows.join("\n");
  const blob = new Blob([csv], {type:"text/csv"});
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href=url; a.download="labeled_dataset.csv"; a.click();
  URL.revokeObjectURL(url);
  showToast(hasSpectra ? "Exported labeled dataset with full spectra" : "Exported labeled dataset (load full spectrum data first for feature columns)");
}
