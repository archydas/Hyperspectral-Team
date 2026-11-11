/* core.js — Data helpers, application state and the SVG chart
   (classic script, loaded in order from index.html; all functions are global) */

/* =========================================================================
   DATA LAYER — browser-side helpers (the libraries themselves come from the database, see DBLAYER)
   ========================================================================= */
const MATERIAL_TYPES = ["Vegetation","Soil","Rock","Mineral","Metal","Manmade"];
const NAME_BANK = {
  Vegetation: ["Dry grass","Oak leaf","Pine needles","Corn canopy","Turf grass","Wheat stubble"],
  Soil: ["Sandy loam","Clay soil","Peat","Silty clay","Red laterite","Alluvial soil"],
  Rock: ["Basalt","Sandstone","Granite","Limestone","Shale","Quartzite"],
  Mineral: ["Calcium carbonate","Gypsum","Hematite","Kaolinite","Quartz sand","Mica schist"],
  Metal: ["Copper roofing","Galvanized steel","Aluminum panel","Zinc sheet","Brass fitting"],
  Manmade: ["Asphalt","Concrete","Plastic (HDPE)","Rubber membrane","Painted panel","Brick"],
};
function seededRandom(seed){ const x = Math.sin(seed)*10000; return x - Math.floor(x); }
function makeSpectrum(seed, wlMin=0.4, wlMax=2.5, points=24){
  const arr=[];
  for(let i=0;i<points;i++){
    const wavelength = Number((wlMin + (wlMax-wlMin)*i/(points-1)).toFixed(3));
    const v = 0.12 + 0.5*Math.abs(Math.sin(i*0.4+seed)) + 0.25*seededRandom(seed*13+i);
    arr.push({ wavelength, reflectance: Number(Math.min(v,0.98).toFixed(3)) });
  }
  return arr;
}
const LIBRARY_DEFS = [
  { id:"ecostress", name:"ECOSTRESS", source:"Open Source", color:"emerald", count:6, seed:1 },
  { id:"soil", name:"Open Soil Spectral Library", source:"Open Source", color:"amber", count:6, seed:2 },
  { id:"usgs", name:"USGS Spectral Library", source:"Open Source", color:"sky", count:6, seed:3 },
  { id:"astm", name:"ASTM Library", source:"Custom", color:"purple", count:5, seed:4 },
  { id:"custom1", name:"Custom Library 1", source:"Custom", color:"orange", count:4, seed:5 },
];
const COMPARE_COLORS = ["#34d399","#60a5fa","#f472b6","#fbbf24"];

/* =========================================================================
   STATE
   ========================================================================= */
let state = {
  view: "dashboard",
  libraries: [],
  activeLibId: null,
  activeMatId: null,
  compareList: [],
  recentFiles: [],
  imagesLoaded: 0,
  spectraExtracted: 0,
  libQuery: "",
  globalMatQuery: "",
  matQuery: "",
  showAddLibrary: false,
  showAddMaterial: false,
  matTypeFilter: "",
  importPreview: null,
};

function showToast(msg){
  const t = document.getElementById("toast");
  t.textContent = msg;
  t.classList.add("show");
  clearTimeout(window._toastTimer);
  window._toastTimer = setTimeout(()=>t.classList.remove("show"), 2400);
}
function addRecent(entry){
  state.recentFiles = [{ id:"r-"+Date.now(), when:"just now", ...entry }, ...state.recentFiles].slice(0,6);
  renderRecent();
}
function setView(v){ state.view = v; if(v==='library-list'){ state.activeLibId=null; state.activeMatId=null; } render(); if(v==='image') openImageView(); if(v==='dashboard'||v==='library-list') loadLibraries(); }
function addLibrary(name, source){
  if(!name || !name.trim()) return;
  const id = "lib-"+Date.now();
  const color = source==="Open Source" ? "sky" : "orange";
  const lib = { id, name:name.trim(), source, color, count:0, seed: state.libraries.length+10, materials:[], user:true };
  state.libraries.push(lib);
  persistUserLibrary();
  state.showAddLibrary = false;
  addRecent({ name:lib.name, type:"Library" });
  showToast(`"${lib.name}" added`);
  renderMainOnly();
}
function submitAddLibrary(){
  const name = document.getElementById("newLibName").value;
  const source = document.getElementById("newLibSource").value;
  addLibrary(name, source);
}
function addMaterial(libId, data){
  try{
    if(data.customData) validateSpectrumRows(sortRows(data.customData));
    const mat = createSample(libId, data);
    if(data.customData) addSpectrum(libId, mat.id, {data:data.customData, origin:data.origin||"library_import", pixel:data.pixel, sourceFormat:data.sourceFormat});
    const lib = state.libraries.find(l=>l.id===libId);
    state.showAddMaterial = false;
    persistUserLibrary();
    addRecent({ name:mat.name, type:"Spectrum" });
    showToast(`"${mat.name}" added to ${lib.name}${mat.customData ? "" : " — no spectrum yet, open it to add one"}`);
    renderMainOnly();
    return mat;
  }catch(e){ showToast(e.message); return null; }
}
function submitAddMaterial(libId){
  const name = document.getElementById("newMatName").value;
  const type = document.getElementById("newMatType").value;
  const samples = parseInt(document.getElementById("newMatSamples").value, 10) || 20;
  const latitude = document.getElementById("newMatLat").value;
  const longitude = document.getElementById("newMatLon").value;
  const locationName = document.getElementById("newMatLoc").value;
  addMaterial(libId, { name, type, samples, latitude, longitude, locationName });
}
function toggleCompare(libId, matId){
  const i = state.compareList.findIndex(c=>c.libId===libId && c.matId===matId);
  if(i>=0){ state.compareList.splice(i,1); }
  else{
    if(state.compareList.length>=4){ showToast("You can compare up to 4 materials"); return; }
    state.compareList.push({libId, matId});
  }
  render();
}
function downloadCSV(filename, rows){
  const header = "wavelength_um,reflectance\n";
  const body = rows.map(r=>`${r.wavelength},${r.reflectance}`).join("\n");
  const blob = new Blob([header+body], {type:"text/csv"});
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href=url; a.download=filename; a.click();
  URL.revokeObjectURL(url);
}
function downloadTXT(filename, rows){
  const body = rows.map(r=>`${r.wavelength}\t${r.reflectance}`).join("\n");
  const blob = new Blob([body], {type:"text/plain"});
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href=url; a.download=filename; a.click();
  URL.revokeObjectURL(url);
}
function downloadJSON(filename, rows){
  const blob = new Blob([JSON.stringify(rows, null, 2)], {type:"application/json"});
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href=url; a.download=filename; a.click();
  URL.revokeObjectURL(url);
}

/* ---- spectrum file import: CSV / TXT (ASCII) / JSON ---- */
function parseSpectrumText(text, ext){
  text = text.trim();
  if(ext === "json" || text.startsWith("[") || text.startsWith("{")){
    try{
      const parsed = JSON.parse(text);
      const arr = Array.isArray(parsed) ? parsed : (parsed.data || parsed.samples || []);
      const rows = arr.map(r => ({
        wavelength: Number(r.wavelength ?? r.wavelength_um ?? r.x),
        reflectance: Number(r.reflectance ?? r.y),
      })).filter(r => !isNaN(r.wavelength) && !isNaN(r.reflectance));
      return rows.length ? rows : null;
    } catch(e){ return null; }
  }
  // CSV / TXT / ASCII — comma, tab, or space delimited, one pair per line
  const rows = [];
  text.split(/\r?\n/).forEach(line=>{
    const parts = line.trim().split(/[,\t\s]+/).filter(Boolean);
    if(parts.length>=2){
      const w = Number(parts[0]), r = Number(parts[1]);
      if(!isNaN(w) && !isNaN(r)) rows.push({wavelength:w, reflectance:r});
    }
  });
  return rows.length ? rows : null;
}
function cancelImport(){ state.importPreview = null; renderMainOnly(); }
function submitImport(){
  const ip = state.importPreview;
  if(!ip) return;
  const name = document.getElementById("importMatName").value || ip.suggestedName;
  const type = document.getElementById("importMatType").value;
  const latitude = document.getElementById("importMatLat").value;
  const longitude = document.getElementById("importMatLon").value;
  const locationName = document.getElementById("importMatLoc").value;
  addMaterial(ip.libId, {
    name, type, samples: ip.rows.length, customData: ip.rows,
    latitude, longitude, locationName, sourceFormat: ip.format,
  });
  state.importPreview = null;
}

/* =========================================================================
   CHART — lightweight inline SVG line chart (no external chart library)
   ========================================================================= */
function svgLineChart(seriesList, opts={}){
  // seriesList: [{name, color, data:[{x,y}]}]
  const w = opts.width || 560, h = opts.height || 200;
  const padL=34, padR=10, padT=10, padB=22;
  const allX = seriesList.flatMap(s=>s.data.map(d=>d.x));
  const xMin = Math.min(...allX), xMax = Math.max(...allX);
  let yMin=0, yMax=1;
  if(opts.yAuto){
    const allY = seriesList.flatMap(s=>s.data.map(d=>d.y));
    yMin = Math.min(...allY); yMax = Math.max(...allY);
    if(yMin===yMax){ yMin-=1; yMax+=1; }
    const pad = (yMax-yMin)*0.08;
    yMin -= pad; yMax += pad;
  }
  const xs = x => padL + ( (x-xMin)/(xMax-xMin||1) ) * (w-padL-padR);
  const ys = y => (h-padB) - ( (y-yMin)/(yMax-yMin) ) * (h-padT-padB);

  let grid = "";
  for(let i=0;i<=4;i++){
    const gy = padT + i*(h-padT-padB)/4;
    grid += `<line x1="${padL}" y1="${gy}" x2="${w-padR}" y2="${gy}" class="chart-grid"/>`;
    grid += `<text x="4" y="${gy+3}" class="chart-tick">${(yMax - i*(yMax-yMin)/4).toFixed(opts.yAuto?3:2)}</text>`;
  }
  const xTicks = 5;
  for(let i=0;i<=xTicks;i++){
    const gx = padL + i*(w-padL-padR)/xTicks;
    const val = xMin + i*(xMax-xMin)/xTicks;
    grid += `<text x="${gx-8}" y="${h-6}" class="chart-tick">${val.toFixed(2)}</text>`;
  }

  let lines = "";
  seriesList.forEach(s=>{
    const pts = s.data.map(d=>`${xs(d.x)},${ys(d.y)}`).join(" ");
    lines += `<polyline points="${pts}" fill="none" stroke="${s.color}" stroke-width="2"/>`;
  });

  return `<svg viewBox="0 0 ${w} ${h}" width="100%" height="${h}">
    <line x1="${padL}" y1="${padT}" x2="${padL}" y2="${h-padB}" class="chart-axis"/>
    <line x1="${padL}" y1="${h-padB}" x2="${w-padR}" y2="${h-padB}" class="chart-axis"/>
    ${grid}
    ${lines}
  </svg>`;
}
