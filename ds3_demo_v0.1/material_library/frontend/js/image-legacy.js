/* image-legacy.js — Image page (first version): bands, display, pixel click, ROI, libraries in the image
   (classic script, loaded in order from index.html; all functions are global) */
/* =========================================================================
   RENDER — IMAGE MODULE
   ========================================================================= */
let imgState = {
  imgEl:null, pixel:null, mode:"true", zoom:1,
  bands:{ r:"Band 30", g:"Band 20", b:"Band 10" },
  fileName:null, naturalW:null, naturalH:null,
  pixelHistory:[], showSaveForm:false,
  geo:null,            // fitted affine transform {a,b,c,d,e,f} once >=3 GCPs are computed
  gcps:[],             // [{id, px, py, lat, lon}]
  pendingImgPoint:null,

  // Real multi-band TIFF/GeoTIFF/Zarr support (any .tif/.tiff or zipped Zarr v2 store)
  tiff:null,           // { image, width, height, bandCount, bandCache:Map, wavelengths:null|number[], cube:null|number[][], cubeLoading:false, rIdx, gIdx, bIdx }

  labels:[],           // [{id, x, y, label, lat, lon}] — pixels the user has tagged with a class/material name
  labelInput:"",       // current text in the "label this pixel" box, kept between clicks for convenience

  roi:{ active:false, points:[], closed:false },  // polygon being drawn/finished for ROI-scoped model runs
  roiResult:null,      // set once "Run model on ROI" completes — see runRoiClassification()
};
let georefMap = null, georefMarkersLayer = null;
const BAND_OPTIONS = ["Band 5","Band 10","Band 20","Band 30","Band 45","Band 60"];

function renderImage(){
  const t = imgState.tiff;

  const bandSelects = t
    ? ["r","g","b"].map(k=>{
        const idx = k==='r'?t.rIdx:k==='g'?t.gIdx:t.bIdx;
        const opts = [];
        for(let i=0;i<t.bandCount;i++) opts.push(`<option value="${i}" ${i===idx?'selected':''}>${bandLabel(t,i)}</option>`);
        return `<div class="crumb" onclick="setView('image')">&larr; Image (any format, database library)</div>
  <div class="banner-info"><span>&#9432;</span><span>Browser-only viewer for GeoTIFF / ENVI / Zarr with georeference, ROI and labelling. Until a real cube is loaded here the pixel spectra are <b>simulated</b>. For AVIRIS-NG, EnMAP, .mat and library overlay use the main Image page.</span></div>
  <div style="display:flex; align-items:center; gap:5px; font-size:11.5px; color:var(--muted);">
          ${k.toUpperCase()}
          <select onchange="setTiffBand('${k}', this.value)" style="max-width:170px;">${opts.join("")}</select>
        </div>`;
      }).join("")
    : ["r","g","b"].map(k=>`
      <div style="display:flex; align-items:center; gap:5px; font-size:11.5px; color:var(--muted);">
        ${k.toUpperCase()}
        <select onchange="setBand('${k}', this.value)">
          ${BAND_OPTIONS.map(b=>`<option ${imgState.bands[k]===b?'selected':''}>${b}</option>`).join("")}
        </select>
      </div>`).join("");

  const dispW = t ? 480 : 480;
  const dispH = t ? Math.round(480 * (t.height/t.width)) : 270;

  return `
  <h1 class="pagetitle">Image</h1>
  <p class="pagesub">Upload an image, adjust the band combination, and click a pixel to see its spectrum.</p>
  ${t ? `
  <div class="banner-info" style="color:#6ee7b7; background:rgba(2,44,34,.4); border-color:rgba(6,95,70,.5);">
    <span>&#9989;</span>
    Real ${t.bandCount}-band ${t.source==='zarr'?'Zarr':t.source==='envi'?'ENVI':'TIFF'} loaded (${t.width}×${t.height}px). Band selection below reads and stretches actual pixel values from the file — not simulated.
    ${t.source==='zarr' ? `<br/><span style="color:#fbbf24;">Band axis was auto-detected as the smallest array dimension — if the composite below looks striped or scrambled, that guess may be wrong for this array.</span>` : ""}
  </div>` : ""}

  <div style="display:grid; grid-template-columns:1fr 300px; gap:18px;" id="imageGrid">
    <div>
      <div class="img-toolbar">
        <label class="btn ghost btn-sm" style="cursor:pointer;">
          Upload image (Ctrl/Cmd-click both files for ENVI)
          <input type="file" accept="image/*,.tif,.tiff,.zip,.hdr" multiple style="display:none" onchange="handleImageUpload(event)"/>
        </label>
        ${bandSelects}
        ${!t ? `<button class="btn ghost btn-sm" onclick="toggleFalseColor()">${imgState.mode==='true' ? 'Show false color' : 'Show true color'}</button>` : ""}
        <button class="btn ghost btn-sm" onclick="zoomOut()">&minus;</button>
        <span class="zoom-readout">${Math.round(imgState.zoom*100)}%</span>
        <button class="btn ghost btn-sm" onclick="zoomIn()">+</button>
        <button class="btn ghost btn-sm" onclick="zoomReset()">Reset zoom</button>
        ${(imgState.imgEl||t) ? `<button class="btn ghost btn-sm" onclick="downloadImage()">Download image</button>` : ""}
        ${(imgState.imgEl||t) ? `<button class="btn ghost btn-sm" onclick="resetImage()">Clear image</button>` : ""}
        ${(imgState.imgEl||t) ? `<button class="btn ${imgState.geo ? 'emerald' : 'ghost'} btn-sm" onclick="setView('image-georef')">📍 Georeference (GCP)</button>` : ""}
      </div>

      ${t ? `
      <div class="img-toolbar" style="margin-top:-4px;">
        <label class="btn ghost btn-sm" style="cursor:pointer;">
          Upload metadata XML (optional — real band wavelengths + corner geo)
          <input type="file" accept=".xml" style="display:none" onchange="handleTiffMetadataXml(event)"/>
        </label>
        ${!t.cube ? `<button class="btn ghost btn-sm" onclick="loadFullSpectralCube()" ${t.cubeLoading?'disabled':''}>${t.cubeLoading ? 'Reading all bands…' : `Load full spectrum data (${t.bandCount} bands)`}</button>`
                  : `<span style="font-size:11.5px; color:var(--emerald-l);">✓ Full spectral cube loaded — click a pixel for its real spectrum</span>`}
      </div>` : ""}

      ${imgState.geo ? `<p style="font-size:11.5px; color:var(--muted); margin:6px 0 10px;">
        📍 Georeferenced from ${imgState.gcps.length} control points (RMSE ${computeRmse().toFixed(1)} m) —
        <a href="#" onclick="setView('image-georef'); return false;" style="color:#60a5fa;">Edit points</a> ·
        <a href="#" onclick="clearImageGeoreference(); return false;" style="color:#f87171;">Clear</a>
      </p>` : ""}

      ${(imgState.imgEl||t) ? `
      <div class="img-toolbar" style="margin-top:-4px;">
        ${!imgState.roi.active && !imgState.roi.closed ? `<button class="btn ghost btn-sm" onclick="startRoiDraw()">&#9998; Draw ROI polygon</button>` : ""}
        ${imgState.roi.active ? `<button class="btn emerald btn-sm" onclick="finishRoiDraw()">Finish ROI</button><span id="roiPointCount" style="font-size:11.5px; color:var(--muted);">${imgState.roi.points.length} point${imgState.roi.points.length===1?'':'s'}</span>` : ""}
        ${(imgState.roi.active || imgState.roi.closed) ? `<button class="btn ghost btn-sm" onclick="clearRoi()">Clear ROI</button>` : ""}
        ${imgState.roi.active ? `<span style="font-size:11.5px; color:var(--muted);">Click the image to add a vertex, then Finish ROI (min. 3 points).</span>` : ""}
      </div>` : ""}

      <div style="overflow:auto; max-height:420px; border-radius:10px; border:1px solid var(--border);">
        <canvas id="imgCanvas" width="480" height="270" onclick="handleCanvasClick(event)"
          style="display:block; width:${dispW*imgState.zoom}px; height:${dispH*imgState.zoom}px; cursor:crosshair;
          filter:${(!t && imgState.mode==='false') ? 'hue-rotate(130deg) saturate(1.6)' : 'none'};"></canvas>
      </div>

      <p style="font-size:12px; color:var(--muted); margin-top:8px;">
        ${t
          ? `R: ${bandLabel(t,t.rIdx)} · G: ${bandLabel(t,t.gIdx)} · B: ${bandLabel(t,t.bIdx)} — real stretched values. Click the image to inspect a pixel.`
          : `R: ${imgState.bands.r} · G: ${imgState.bands.g} · B: ${imgState.bands.b} (label only in this demo) · Click the image to inspect a pixel.`}
      </p>
      ${(imgState.imgEl||t) ? `<p style="font-size:11.5px; color:var(--muted);">${imgState.fileName} · source ${imgState.naturalW}×${imgState.naturalH}px${t?` · ${t.bandCount} bands`:''}</p>` : ""}
    </div>
    <div id="pixelPanel">${renderPixelPanel()}</div>
  </div>

  ${renderCnnPanel()}

  ${imgState.roi.closed ? `
  <div class="chart-wrap" style="margin-top:16px;">
    <p style="font-size:13px; color:#e2e8f0; margin:0 0 10px;">Apply model to selected ROI (${imgState.roi.points.length}-point polygon)</p>
    <div class="add-form">
      <div><label class="small">Reference library</label>
        <select id="roiLibSelect">${state.libraries.map(l=>`<option value="${l.id}">${l.name} (${l.materials.length} materials)</option>`).join("")}</select>
      </div>
      <div><label class="small">Resolution</label>
        <select id="roiResSelect">
          <option value="1">Full (every pixel)</option>
          <option value="2">1/2 (every 2nd pixel)</option>
          <option value="4" selected>1/4 (every 4th pixel)</option>
          <option value="8">1/8 (every 8th pixel)</option>
        </select>
      </div>
      <div><label class="small">Model</label>
        <select id="roiModelSelect" onchange="cnnState.roiModel=this.value; renderMainOnly()">
          <option value="sam" ${cnnState.roiModel==='sam'?'selected':''}>Spectral angle match (baseline)</option>
          <option value="cnn" ${cnnState.roiModel==='cnn'?'selected':''} ${cnnState.model?'':'disabled'}>3D-CNN (trained${cnnState.model?'':' — train it above first'})</option>
        </select>
      </div>
      <button class="btn emerald btn-sm" onclick="runRoiClassification()">Run model on ROI</button>
    </div>
    <p style="font-size:11px; color:var(--muted); margin:8px 0 0;">
      ${cnnState.roiModel==='cnn' ? `3D-CNN trained above, applied to 5×5-pixel patches; pixels under ${CNN_MIN_CONF*100}% confidence stay unclassified.` : `Spectral-angle match against the chosen library's materials — a non-trained baseline to compare the 3D-CNN against.`}
      ${imgState.tiff && imgState.tiff.cube ? 'Uses real per-pixel spectra.' : 'Uses simulated per-pixel spectra (load a real image and its full spectral cube above for real values).'}
    </p>
    <div id="roiResultsPanel">${renderRoiResults()}</div>
  </div>` : ""}

  ${imgState.labels.length ? `
  <div class="chart-wrap" style="margin-top:16px;">
    <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:8px;">
      <p style="font-size:13px; color:#e2e8f0; margin:0;">Labeled points (${imgState.labels.length})</p>
      <button class="btn emerald btn-sm" onclick="exportLabeledDataset()">Export labeled dataset (CSV)</button>
    </div>
    <table class="gcp-table">
      <thead><tr><th>x, y</th><th>Label</th><th>Lat, Lon</th><th></th></tr></thead>
      <tbody>${imgState.labels.map(l=>`
        <tr>
          <td>${l.x}, ${l.y}</td>
          <td>${l.label}</td>
          <td>${l.lat!=null ? `${l.lat.toFixed(5)}, ${l.lon.toFixed(5)}` : '—'}</td>
          <td><button class="btn ghost btn-sm" onclick="removeLabelPoint(${l.id})">Remove</button></td>
        </tr>`).join("")}</tbody>
    </table>
  </div>` : ""}
  `;
}

function renderPixelPanel(){
  const historyHtml = imgState.pixelHistory.length ? `
    <div class="pixel-history">
      <span class="lbl">Recent picks</span>
      ${imgState.pixelHistory.map(p=>`<button class="btn ghost btn-sm" onclick="revisitPixel(${p.x},${p.y},${p.seed})">(${p.x}, ${p.y})</button>`).join("")}
    </div>` : "";

  if(!imgState.pixel){
    return `<div class="placeholder-card">Click a pixel on the image to see its spectrum here.</div>${historyHtml}`;
  }
  const t = imgState.tiff;
  const isReal = t && t.cube;
  const data = isReal ? tiffPixelSpectrum(imgState.pixel.x, imgState.pixel.y) : makeSpectrum(imgState.pixel.seed, 0.4, 2.5);
  const chart = svgLineChart([{name:"pixel", color:"#34d399", data:data.map(d=>({x:d.wavelength,y:d.reflectance}))}], {height:150, yAuto:true});

  const saveForm = imgState.showSaveForm ? `
    <div class="add-form" style="margin:10px 0 0;">
      <div><label class="small">Save to library</label>
        <select id="saveLibSelect">${state.libraries.map(l=>`<option value="${l.id}">${l.name}</option>`).join("")}</select>
      </div>
      <div><label class="small">Material name</label>
        <input type="text" id="saveMatName" value="Pixel (${imgState.pixel.x}, ${imgState.pixel.y})"></div>
      <div><label class="small">Latitude</label><input type="text" id="saveMatLat" value="${imgState.pixel.geo ? imgState.pixel.geo.lat.toFixed(5) : ''}" placeholder="optional" style="width:100px;"></div>
      <div><label class="small">Longitude</label><input type="text" id="saveMatLon" value="${imgState.pixel.geo ? imgState.pixel.geo.lon.toFixed(5) : ''}" placeholder="optional" style="width:100px;"></div>
      <div><label class="small">Location name</label><input type="text" id="saveMatLoc" placeholder="optional" style="width:150px;"></div>
      <button class="btn emerald btn-sm" onclick="saveCurrentPixelToLibrary()">Save</button>
      <button class="btn ghost btn-sm" onclick="imgState.showSaveForm=false; renderMainOnly()">Cancel</button>
    </div>` : "";

  const existingLabel = imgState.labels.find(l=>l.x===imgState.pixel.x && l.y===imgState.pixel.y);
  const labelOptions = Array.from(new Set([
    ...imgState.labels.map(l=>l.label),
    ...state.libraries.flatMap(l=>l.materials.map(m=>m.name)),
  ])).slice(0,100);

  return `
  <div class="chart-wrap">
    <p style="font-size:12px; color:#cbd5e1; margin:0 0 8px;">Pixel (${imgState.pixel.x}, ${imgState.pixel.y}) ${isReal ? '<span style="color:#34d399; font-size:11px;">— real spectrum</span>' : (t ? '<span style="color:#fbbf24; font-size:11px;">— load full spectrum data above for a real spectrum</span>' : '')}</p>
    ${imgState.pixel.geo ? `<p style="font-size:11.5px; color:var(--muted2); margin:-4px 0 8px;">📍 ${imgState.pixel.geo.lat.toFixed(5)}, ${imgState.pixel.geo.lon.toFixed(5)}</p>` : ""}
    ${chart}
    <div style="display:flex; gap:8px; margin-top:10px; flex-wrap:wrap;">
      <button class="btn ghost btn-sm" onclick='downloadCSV("pixel_${imgState.pixel.x}_${imgState.pixel.y}.csv", ${JSON.stringify(data)})'>Export CSV</button>
      <button class="btn ghost btn-sm" onclick="imgState.showSaveForm=!imgState.showSaveForm; renderMainOnly()">Save to library</button>
    </div>
    ${saveForm}
    <div style="margin-top:12px; padding-top:10px; border-top:1px solid var(--border);">
      <label class="small">${existingLabel ? 'Labeled as:' : 'Label this pixel (for a training dataset)'}</label>
      <div style="display:flex; gap:6px; flex-wrap:wrap;">
        <input type="text" id="pixelLabelInput" list="labelOptionsList" placeholder="e.g. water, road, concrete…" value="${existingLabel ? existingLabel.label : imgState.labelInput}" style="width:170px;">
        <datalist id="labelOptionsList">${labelOptions.map(o=>`<option value="${o}">`).join("")}</datalist>
        <button class="btn ${existingLabel?'ghost':'emerald'} btn-sm" onclick="addPixelLabel()">${existingLabel ? 'Update' : 'Add label'}</button>
        ${existingLabel ? `<button class="btn ghost btn-sm" onclick="removeLabelPoint(${existingLabel.id})">Remove</button>` : ""}
      </div>
    </div>
  </div>
  ${historyHtml}`;
}

function drawCanvas(){
  const canvas = document.getElementById("imgCanvas");
  if(!canvas) return;
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0,0,canvas.width,canvas.height);
  if(imgState.imgEl){
    ctx.drawImage(imgState.imgEl,0,0,canvas.width,canvas.height);
  } else {
    const cols=16, rows=9, cw=canvas.width/cols, ch=canvas.height/rows;
    const palette = ["#3f6212","#65a30d","#78716c","#a16207","#57534e"];
    for(let i=0;i<cols;i++) for(let j=0;j<rows;j++){
      const idx = Math.floor(seededRandom(i*13+j*7)*palette.length);
      ctx.fillStyle = palette[idx];
      ctx.fillRect(i*cw, j*ch, cw, ch);
    }
  }
  drawLabelMarkers(ctx);
  drawRoiOverlay(ctx);
}
function drawLabelMarkers(ctx){
  imgState.labels.forEach(l=>{
    ctx.beginPath(); ctx.arc(l.x,l.y,4,0,Math.PI*2);
    ctx.fillStyle="#f472b6"; ctx.fill();
    ctx.strokeStyle="#1c0a2e"; ctx.lineWidth=1.2; ctx.stroke();
  });
}
function isTiffFile(file){
  const n = file.name.toLowerCase();
  return n.endsWith(".tif") || n.endsWith(".tiff") || file.type === "image/tiff";
}
function isZarrZipFile(file){
  const n = file.name.toLowerCase();
  return n.endsWith(".zarr.zip") || n.endsWith(".zip");
}
function isEnviHdrFile(file){
  return file.name.toLowerCase().endsWith(".hdr");
}
function handleImageUpload(e){
  const files = Array.from(e.target.files || []);
  if(files.length === 0) return;

  if(files.length >= 2){
    const hdr = files.find(f=>isEnviHdrFile(f));
    const dataFile = files.find(f=>f!==hdr);
    if(hdr && dataFile){ handleEnviFile(hdr, dataFile); return; }
    showToast("Select exactly one .hdr file together with its paired data file");
    return;
  }

  const file = files[0];
  if(isEnviHdrFile(file)){
    showToast("Also select the paired data file — Ctrl/Cmd-click both the .hdr and the data file in the picker");
    return;
  }
  if(isTiffFile(file)){ handleTiffFile(file); return; }
  if(isZarrZipFile(file)){ handleZarrFile(file); return; }

  const reader = new FileReader();
  reader.onload = () => {
    const img = new Image();
    img.onload = () => {
      imgState.imgEl = img;
      imgState.fileName = file.name;
      imgState.naturalW = img.naturalWidth;
      imgState.naturalH = img.naturalHeight;
      imgState.pixel = null;
      imgState.pixelHistory = [];
      imgState.showSaveForm = false;
      imgState.tiff = null;
      imgState.labels = [];
      imgState.geo = null;
      imgState.gcps = [];
      state.imagesLoaded++;
      addRecent({name:file.name, type:"Hyperspectral Image"});
      showToast("Image loaded (used as a visual stand-in in this demo)");
      renderMainOnly();
    };
    img.src = reader.result;
  };
  reader.readAsDataURL(file);
}
