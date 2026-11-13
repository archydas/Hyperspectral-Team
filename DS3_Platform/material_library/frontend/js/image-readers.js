/* image-readers.js — Reading images in the browser: GeoTIFF, Zarr, ENVI (.hdr + raw)
   (classic script, loaded in order from index.html; all functions are global) */
/* =========================================================================
   REAL TIFF / GeoTIFF SUPPORT — works with any .tif/.tiff, single- or
   multi-band (e.g. a 224-band EnMAP L2A SPECTRAL_IMAGE.TIF). Uses geotiff.js,
   loaded entirely client-side — the file never leaves the browser.
   ========================================================================= */
async function handleTiffFile(file){
  showToast(`Reading ${file.name}…`);
  try{
    const buf = await file.arrayBuffer();
    const geotiff = await GeoTIFF.fromArrayBuffer(buf);
    const image = await geotiff.getImage();
    const width = image.getWidth();
    const height = image.getHeight();
    const bandCount = image.getSamplesPerPixel();
    let noDataValue = null;
    try{ const nd = image.getGDALNoData(); if(nd !== null && nd !== undefined) noDataValue = nd; }catch(e){}

    imgState.imgEl = null;
    imgState.fileName = file.name;
    imgState.naturalW = width;
    imgState.naturalH = height;
    imgState.pixel = null;
    imgState.pixelHistory = [];
    imgState.showSaveForm = false;
    imgState.geo = null;
    imgState.gcps = [];
    imgState.labels = [];
    imgState.tiff = {
      source:'tiff', image, width, height, bandCount,
      bandCache: new Map(),      // bandIndex(0-based) -> {data, min, max}
      wavelengths: null,         // optional, filled in via metadata XML upload
      cube: null,                // filled in via "load full spectral cube"
      cubeLoading: false,
      noDataValue,
      rIdx: Math.max(0, Math.round(bandCount*0.7)-1),
      gIdx: Math.max(0, Math.round(bandCount*0.5)-1),
      bIdx: Math.max(0, Math.round(bandCount*0.25)-1),
      _readBand: async(idx) => (await image.readRasters({ samples:[idx], interleave:false }))[0],
    };
    state.imagesLoaded++;
    addRecent({name:file.name, type:"Hyperspectral Image (TIFF)"});
    showToast(`Loaded ${bandCount}-band TIFF (${width}×${height}px)`);
    renderMainOnly();
    await renderTiffComposite();
  }catch(err){
    console.error(err);
    showToast("Couldn't read that TIFF — see console for details");
  }
}
async function getTiffBand(idx){
  const t = imgState.tiff;
  if(t.bandCache.has(idx)) return t.bandCache.get(idx);
  const data = await t._readBand(idx);
  const nd = t.noDataValue;
  let min=Infinity, max=-Infinity;
  for(let i=0;i<data.length;i++){
    const v=data[i];
    if(nd!==null && v===nd) continue;
    if(v<min) min=v; if(v>max) max=v;
  }
  const entry = { data, min, max };
  t.bandCache.set(idx, entry);
  return entry;
}
async function renderTiffComposite(){
  const t = imgState.tiff;
  if(!t) return;
  const canvas = document.getElementById("imgCanvas");
  if(!canvas) return;
  canvas.width = t.width;
  canvas.height = t.height;

  const [rB,gB,bB] = await Promise.all([getTiffBand(t.rIdx), getTiffBand(t.gIdx), getTiffBand(t.bIdx)]);
  const n = t.width*t.height;
  const imgData = new ImageData(t.width, t.height);
  const stretch = (b,i) => {
    const range = (b.max-b.min) || 1;
    return Math.max(0, Math.min(255, Math.round((b.data[i]-b.min)/range*255)));
  };
  for(let i=0;i<n;i++){
    imgData.data[i*4]   = stretch(rB,i);
    imgData.data[i*4+1] = stretch(gB,i);
    imgData.data[i*4+2] = stretch(bB,i);
    imgData.data[i*4+3] = 255;
  }
  canvas.getContext("2d").putImageData(imgData, 0, 0);
  drawLabelMarkers(canvas.getContext("2d"));
  drawRoiOverlay(canvas.getContext("2d"));

  // Cache an off-DOM copy so the georeferencing panel (a different canvas,
  // on a different view) can draw the same composite without recomputing it.
  if(!t.previewCanvas){ t.previewCanvas = document.createElement("canvas"); }
  t.previewCanvas.width = t.width;
  t.previewCanvas.height = t.height;
  t.previewCanvas.getContext("2d").putImageData(imgData, 0, 0);
}
function bandLabel(t, idx){
  if(t.wavelengths && t.wavelengths[idx] != null) return `Band ${idx+1} (${t.wavelengths[idx].toFixed(1)} nm)`;
  return `Band ${idx+1}`;
}
function setTiffBand(channel, idx){
  const t = imgState.tiff;
  if(!t) return;
  idx = parseInt(idx,10);
  if(channel==='r') t.rIdx=idx; if(channel==='g') t.gIdx=idx; if(channel==='b') t.bIdx=idx;
  renderMainOnly();
  renderTiffComposite();
}
async function loadFullSpectralCube(){
  const t = imgState.tiff;
  if(!t || t.cube || t.cubeLoading) return;
  t.cubeLoading = true;
  renderMainOnly();
  showToast(`Reading all ${t.bandCount} bands — this can take a while for a large cube…`);
  try{
    if(t.source === 'tiff'){
      const rasters = await t.image.readRasters({ interleave:false });
      t.cube = rasters; // array of TypedArrays, one per band
    } else {
      // Generic path (e.g. Zarr): read band-by-band, reusing/populating the same cache
      const bands = [];
      for(let i=0;i<t.bandCount;i++){
        const {data} = await getTiffBand(i);
        bands.push(data);
        if(i % 20 === 0) showToast(`Reading band ${i+1}/${t.bandCount}…`);
      }
      t.cube = bands;
    }
    showToast("Full spectral cube loaded — click any pixel for its real spectrum");
  }catch(err){
    console.error(err);
    showToast("Couldn't load the full cube (likely out of memory for this image size)");
  }finally{
    t.cubeLoading = false;
    renderMainOnly();
  }
}
function tiffPixelSpectrum(x,y){
  const t = imgState.tiff;
  if(!t || !t.cube) return null;
  const i = y*t.width + x;
  return t.cube.map((bandArr, bi) => ({
    wavelength: t.wavelengths ? t.wavelengths[bi] : (bi+1),
    reflectance: bandArr[i],
  }));
}

/* =========================================================================
   REAL ZARR SUPPORT — a single .zip containing a Zarr v2 array (gzip or
   uncompressed chunks). Reads only the chunks needed for the requested band,
   same lazy philosophy as the TIFF path above. Feeds into the exact same
   imgState.tiff object shape via the generic _readBand() interface, so
   composite rendering, band selection, and pixel spectra all just work.
   ========================================================================= */
function zarrDtypeToTypedArray(bytes, dtype){
  const little = dtype[0] !== '>';
  const typeChar = dtype[1];
  const size = parseInt(dtype.slice(2), 10) || 1;
  if(size===1){
    return typeChar==='u' ? new Uint8Array(bytes.buffer, bytes.byteOffset, bytes.byteLength) : new Int8Array(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  }
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const n = Math.floor(bytes.byteLength/size);
  const out = new Float64Array(n);
  for(let i=0;i<n;i++){
    const off = i*size;
    if(typeChar==='f') out[i] = size===8 ? dv.getFloat64(off, little) : dv.getFloat32(off, little);
    else if(typeChar==='i') out[i] = size===4 ? dv.getInt32(off, little) : size===2 ? dv.getInt16(off, little) : dv.getInt8(off);
    else out[i] = size===4 ? dv.getUint32(off, little) : size===2 ? dv.getUint16(off, little) : dv.getUint8(off);
  }
  return out;
}
function zarrDecompressChunk(bytes, compressor){
  if(!compressor) return bytes;
  if(compressor.id === 'gzip') return fflate.gunzipSync(bytes);
  if(compressor.id === 'zlib') return fflate.unzlibSync(bytes);
  throw new Error(`Unsupported Zarr compressor: "${compressor.id}" (only gzip/zlib/uncompressed are supported)`);
}
async function zarrReadBand(t, bandIndex){
  const m = t.zarrMeta;
  const { shape, chunks, dtype, compressor, bandAxis, spatialAxes, files, prefix, sep } = m;
  const [axH, axW] = spatialAxes;
  const height = shape[axH], width = shape[axW];
  const chunkIdxBand = Math.floor(bandIndex / chunks[bandAxis]);
  const offsetInChunkBand = bandIndex % chunks[bandAxis];
  const out = new Float64Array(height*width);
  const chunkGridH = Math.ceil(height/chunks[axH]);
  const chunkGridW = Math.ceil(width/chunks[axW]);

  for(let ch=0; ch<chunkGridH; ch++){
    for(let cw=0; cw<chunkGridW; cw++){
      const idxArr = [0,0,0];
      idxArr[bandAxis] = chunkIdxBand; idxArr[axH] = ch; idxArr[axW] = cw;
      const key = prefix + idxArr.join(sep);
      const raw = files[key];
      if(!raw) continue; // missing chunk = treated as fill value (left as 0)
      const decompressed = zarrDecompressChunk(raw, compressor);
      const typed = zarrDtypeToTypedArray(decompressed, dtype);
      const dims = [0,0,0].map((_,ax)=>{
        const ci = idxArr[ax];
        return Math.min(chunks[ax], shape[ax]-ci*chunks[ax]);
      });
      const chH = dims[axH], chW = dims[axW];
      for(let ii=0; ii<chH; ii++){
        for(let jj=0; jj<chW; jj++){
          const coord = [0,0,0];
          coord[bandAxis]=offsetInChunkBand; coord[axH]=ii; coord[axW]=jj;
          let lin=0, mul=1;
          for(let ax=2; ax>=0; ax--){ lin += coord[ax]*mul; mul *= dims[ax]; }
          out[(ch*chunks[axH]+ii)*width + (cw*chunks[axW]+jj)] = typed[lin];
        }
      }
    }
  }
  return out;
}
async function handleZarrFile(file){
  showToast(`Unzipping ${file.name}…`);
  try{
    const buf = new Uint8Array(await file.arrayBuffer());
    const files = fflate.unzipSync(buf);
    const zarrayPaths = Object.keys(files).filter(p=>p.endsWith(".zarray"));
    if(zarrayPaths.length===0){
      showToast("No .zarray found inside this zip — is it really a Zarr store?");
      return;
    }
    let best=null;
    for(const p of zarrayPaths){
      const meta = JSON.parse(new TextDecoder().decode(files[p]));
      const count = meta.shape.reduce((a,b)=>a*b,1);
      if(!best || count>best.count) best = {path:p, meta, count};
    }
    const meta = best.meta;
    const prefix = best.path.slice(0, best.path.length - ".zarray".length);
    const sep = meta.dimension_separator || ".";
    const shape = meta.shape;
    if(shape.length !== 3){
      showToast(`This Zarr array has ${shape.length} dimensions — only 3D (band, height, width) arrays are supported right now`);
      return;
    }
    let bandAxis = 0;
    for(let a=1;a<3;a++) if(shape[a]<shape[bandAxis]) bandAxis=a;
    const spatialAxes = [0,1,2].filter(a=>a!==bandAxis);
    const height = shape[spatialAxes[0]], width = shape[spatialAxes[1]];
    const bandCount = shape[bandAxis];

    imgState.imgEl = null;
    imgState.fileName = file.name;
    imgState.naturalW = width;
    imgState.naturalH = height;
    imgState.pixel = null;
    imgState.pixelHistory = [];
    imgState.showSaveForm = false;
    imgState.geo = null;
    imgState.gcps = [];
    imgState.labels = [];
    imgState.tiff = {
      source:'zarr', width, height, bandCount,
      bandCache: new Map(),
      wavelengths: null,
      noDataValue: (meta.fill_value===undefined ? null : meta.fill_value),
      cube: null, cubeLoading:false,
      rIdx: Math.max(0, Math.round(bandCount*0.7)-1),
      gIdx: Math.max(0, Math.round(bandCount*0.5)-1),
      bIdx: Math.max(0, Math.round(bandCount*0.25)-1),
      zarrMeta: { shape, chunks:meta.chunks, dtype:meta.dtype, compressor:meta.compressor, bandAxis, spatialAxes, files, prefix, sep },
      _readBand: async(idx) => zarrReadBand(imgState.tiff, idx),
    };
    state.imagesLoaded++;
    addRecent({name:file.name, type:"Hyperspectral Image (Zarr)"});
    showToast(`Loaded ${bandCount}-band Zarr array (${width}×${height}px) — assumed band axis ${bandAxis} of shape [${shape.join(", ")}]`);
    renderMainOnly();
    await renderTiffComposite();
  }catch(err){
    console.error(err);
    showToast("Couldn't read that Zarr zip — see console for details");
  }
}

/* =========================================================================
   REAL ENVI SUPPORT — a raw flat binary data file + a paired .hdr text
   header (the format AVIRIS-NG and many airborne sensors ship in). No
   compression, so once the file is in memory, reading any single band for
   any interleave (BSQ/BIL/BIP) is just strided indexing — no decode step.
   ========================================================================= */
const ENVI_DTYPE_BYTES = { 1:1, 2:2, 3:4, 4:4, 5:8, 12:2, 13:4 };
function enviReadValue(dv, offset, code, little){
  switch(code){
    case 1: return dv.getUint8(offset);
    case 2: return dv.getInt16(offset, little);
    case 3: return dv.getInt32(offset, little);
    case 4: return dv.getFloat32(offset, little);
    case 5: return dv.getFloat64(offset, little);
    case 12: return dv.getUint16(offset, little);
    case 13: return dv.getUint32(offset, little);
    default: throw new Error(`Unsupported ENVI data type code: ${code}`);
  }
}
function parseEnviHeaderText(text){
  const body = text.replace(/\r\n/g, "\n");
  const meta = {};
  const re = /([a-zA-Z0-9 _]+?)\s*=\s*(\{[\s\S]*?\}|[^\n\r]*)/g;
  let m;
  while((m = re.exec(body))){
    const key = m[1].trim().toLowerCase();
    let val = m[2].trim();
    if(val.startsWith("{") && val.endsWith("}")) val = val.slice(1,-1);
    meta[key] = val;
  }
  return meta;
}
function enviReadBand(t, bandIndex){
  const { buffer, headerOffset, samples, lines, bands, dtypeCode, dtypeBytes, interleave, littleEndian } = t.enviMeta;
  const dv = new DataView(buffer);
  const out = new Float64Array(samples*lines);
  if(interleave === "bsq"){
    const bandStart = headerOffset + bandIndex*samples*lines*dtypeBytes;
    for(let i=0;i<samples*lines;i++) out[i] = enviReadValue(dv, bandStart + i*dtypeBytes, dtypeCode, littleEndian);
  } else if(interleave === "bil"){
    const rowSize = bands*samples*dtypeBytes;
    for(let row=0; row<lines; row++){
      const rowBandStart = headerOffset + row*rowSize + bandIndex*samples*dtypeBytes;
      for(let col=0; col<samples; col++) out[row*samples+col] = enviReadValue(dv, rowBandStart + col*dtypeBytes, dtypeCode, littleEndian);
    }
  } else { // bip
    const pixelStride = bands*dtypeBytes;
    for(let i=0;i<samples*lines;i++) out[i] = enviReadValue(dv, headerOffset + i*pixelStride + bandIndex*dtypeBytes, dtypeCode, littleEndian);
  }
  return out;
}
async function handleEnviFile(hdrFile, dataFile){
  showToast(`Reading ${hdrFile.name}…`);
  try{
    const meta = parseEnviHeaderText(await hdrFile.text());
    const samples = parseInt(meta["samples"], 10);
    const lines = parseInt(meta["lines"], 10);
    const bands = parseInt(meta["bands"], 10);
    if(!samples || !lines || !bands){
      showToast("Couldn't find samples/lines/bands in the .hdr file — is this a valid ENVI header?");
      return;
    }
    const dtypeCode = parseInt(meta["data type"], 10);
    const dtypeBytes = ENVI_DTYPE_BYTES[dtypeCode];
    if(!dtypeBytes){
      showToast(`Unsupported ENVI data type code ${dtypeCode} (complex/int64 types aren't supported)`);
      return;
    }
    const headerOffset = parseInt(meta["header offset"] || "0", 10) || 0;
    const interleave = (meta["interleave"] || "bsq").toLowerCase().trim();
    const littleEndian = (parseInt(meta["byte order"] || "0", 10)) === 0;
    const wavelengths = meta["wavelength"] ? meta["wavelength"].split(",").map(s=>parseFloat(s.trim())).filter(v=>!isNaN(v)) : null;
    const dataIgnoreValue = (meta["data ignore value"]!=null && meta["data ignore value"]!=="") ? parseFloat(meta["data ignore value"]) : null;

    const buffer = await dataFile.arrayBuffer();
    const expectedBytes = headerOffset + samples*lines*bands*dtypeBytes;
    if(buffer.byteLength < expectedBytes){
      showToast(`Warning: data file is smaller than the header describes (${buffer.byteLength} vs ${expectedBytes} bytes expected) — check you selected the matching pair`);
    }

    imgState.imgEl = null;
    imgState.fileName = dataFile.name;
    imgState.naturalW = samples;
    imgState.naturalH = lines;
    imgState.pixel = null;
    imgState.pixelHistory = [];
    imgState.showSaveForm = false;
    imgState.geo = null;
    imgState.gcps = [];
    imgState.labels = [];
    imgState.tiff = {
      source:'envi', width:samples, height:lines, bandCount:bands,
      bandCache: new Map(),
      wavelengths,
      noDataValue: dataIgnoreValue,
      cube: null, cubeLoading:false,
      rIdx: Math.max(0, Math.round(bands*0.7)-1),
      gIdx: Math.max(0, Math.round(bands*0.5)-1),
      bIdx: Math.max(0, Math.round(bands*0.25)-1),
      enviMeta: { buffer, headerOffset, samples, lines, bands, dtypeCode, dtypeBytes, interleave, littleEndian },
      _readBand: async(idx) => enviReadBand(imgState.tiff, idx),
    };
    state.imagesLoaded++;
    addRecent({name:dataFile.name, type:"Hyperspectral Image (ENVI)"});
    showToast(`Loaded ${bands}-band ENVI cube (${samples}×${lines}px, ${interleave.toUpperCase()})`);
    renderMainOnly();
    await renderTiffComposite();
  }catch(err){
    console.error(err);
    showToast("Couldn't read that ENVI file — see console for details");
  }
}

/* Optional: parse an EnMAP-style METADATA.XML for band wavelengths (nm) and
   the four scene corner coordinates, so georeferencing can be set instantly
   instead of manually picking GCPs. This is EnMAP-specific parsing layered
   on top of the generic TIFF support above — a plain TIFF simply won't have
   this file, and band selection still works fine without it. */
async function handleTiffMetadataXml(e){
  const file = e.target.files?.[0];
  if(!file || !imgState.tiff) return;
  const text = await file.text();
  const xml = new DOMParser().parseFromString(text, "application/xml");

  if(imgState.tiff.noDataValue === null){
    const bg = parseFloat(xml.querySelector("backgroundValue")?.textContent);
    if(!isNaN(bg)){
      imgState.tiff.noDataValue = bg;
      imgState.tiff.bandCache.clear(); // recompute min/max excluding no-data now that we know it
    }
  }

  const bandNodes = xml.querySelectorAll("bandCharacterisation bandID");
  if(bandNodes.length){
    const wavelengths = [];
    bandNodes.forEach(node=>{
      const n = parseInt(node.getAttribute("number"),10);
      const wl = parseFloat(node.querySelector("wavelengthCenterOfBand")?.textContent);
      if(!isNaN(n) && !isNaN(wl)) wavelengths[n-1] = wl;
    });
    imgState.tiff.wavelengths = wavelengths;
  }

  const corners = {};
  xml.querySelectorAll("boundingPolygon point").forEach(pt=>{
    const frame = pt.querySelector("frame")?.textContent;
    const lat = parseFloat(pt.querySelector("latitude")?.textContent);
    const lon = parseFloat(pt.querySelector("longitude")?.textContent);
    if(frame && !corners[frame] && !isNaN(lat) && !isNaN(lon)) corners[frame] = {lat,lon};
  });
  if(corners.upper_left && corners.upper_right && corners.lower_left){
    const t = imgState.tiff;
    imgState.gcps = [
      { id:1, px:0,         py:0,          lat:corners.upper_left.lat,  lon:corners.upper_left.lon },
      { id:2, px:t.width-1, py:0,          lat:corners.upper_right.lat, lon:corners.upper_right.lon },
      { id:3, px:0,         py:t.height-1, lat:corners.lower_left.lat,  lon:corners.lower_left.lon },
    ];
    if(corners.lower_right) imgState.gcps.push({ id:4, px:t.width-1, py:t.height-1, lat:corners.lower_right.lat, lon:corners.lower_right.lon });
    imgState.geo = computeAffine(imgState.gcps);
    showToast("Band wavelengths + corner georeferencing loaded from metadata");
  } else {
    showToast("Band wavelengths loaded (no corner coordinates found in this XML)");
  }
  renderMainOnly();
  renderTiffComposite();
}

function toggleFalseColor(){
  imgState.mode = imgState.mode==='true' ? 'false' : 'true';
  renderMainOnly();
}
function setBand(channel, value){ imgState.bands[channel] = value; renderMainOnly(); }
function zoomIn(){ imgState.zoom = Math.min(3, +(imgState.zoom+0.25).toFixed(2)); renderMainOnly(); }
function zoomOut(){ imgState.zoom = Math.max(0.5, +(imgState.zoom-0.25).toFixed(2)); renderMainOnly(); }
function zoomReset(){ imgState.zoom = 1; renderMainOnly(); }
