/* views-library.js — Pages: architecture, dashboard, library list, material list, spectrum viewer, comparison, library map
   (classic script, loaded in order from index.html; all functions are global) */
/* =========================================================================
   RENDER — ARCHITECTURE VIEW
   ========================================================================= */

/* =========================================================================
   RENDER — DASHBOARD
   ========================================================================= */
function renderDashboard(){
  const st = DB.stats || {libraries:state.libraries.length, materials:0, spectra:0, images:0};
  return `
  <h1 class="pagetitle">Welcome to DS3</h1>
  <p class="pagesub">A workspace for managing spectral libraries, exploring hyperspectral images, and — eventually — running material analysis. Your libraries are listed under Library when the app opens.</p>

  <div class="grid cols-4" style="margin-bottom:20px;">
    <div class="stat-card"><div class="v">${st.libraries}</div><div class="l">Libraries</div></div>
    <div class="stat-card"><div class="v">${st.materials}</div><div class="l">Materials</div></div>
    <div class="stat-card"><div class="v">${st.images}</div><div class="l">Images opened</div></div>
    <div class="stat-card"><div class="v">${st.spectra}</div><div class="l">Spectra</div></div>
  </div>

  <div class="grid cols-3">
    <div class="module-card">
      <span class="chip emerald"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/></svg></span>
      <h3>Library</h3>
      <p>Browse and search your spectral libraries; build, view and resample them.</p>
      <button class="btn emerald" style="margin-top:12px;" onclick="setView('library-list')">Open Library</button>
    </div>
    <div class="module-card">
      <span class="chip blue"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8.5" cy="8.5" r="1.5"/><path d="M21 15l-5-5L5 21"/></svg></span>
      <h3>Image</h3>
      <p>Open a hyperspectral image, pick pixels or draw ROIs, compare with a library, aggregate ROIs per class.</p>
      <button class="btn blue" style="margin-top:12px;" onclick="setView('image')">Open Image</button>
    </div>
    <div class="module-card">
      <span class="chip violet"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="12" y1="20" x2="12" y2="10"/><line x1="18" y1="20" x2="18" y2="4"/><line x1="6" y1="20" x2="6" y2="16"/></svg></span>
      <h3>Analysis</h3>
      <p>AI / ML models: classification and detection trained on the libraries and the object samples (next).</p>
      <button class="btn violet" style="margin-top:12px;" onclick="setView('analysis')">Open Analysis</button>
    </div>
  </div>
  `;
}

/* =========================================================================
   RENDER — LIBRARY LIST
   ========================================================================= */
function renderLibraryList(){
  const addForm = state.showAddLibrary ? `
    <div class="add-form">
      <div><label class="small">Name</label><input type="text" id="newLibName" placeholder="e.g. Rooftop Materials"></div>
      <div><label class="small">Source</label>
        <select id="newLibSource"><option>Open Source</option><option>Custom</option></select>
      </div>
      <button class="btn emerald btn-sm" onclick="submitAddLibrary()">Create</button>
      <button class="btn ghost btn-sm" onclick="state.showAddLibrary=false; renderMainOnly()">Cancel</button>
    </div>` : "";

  return `
  <h1 class="pagetitle">Library</h1>
  <p class="pagesub">Manage and explore spectral libraries.</p>

  <div class="search-row">
    <div>
      <label class="small">Search libraries by name</label>
      <input type="text" placeholder="e.g. ECOSTRESS" value="${state.libQuery}" oninput="state.libQuery=this.value; filterLibraries()"/>
    </div>
    <div>
      <label class="small">Search materials by name or location</label>
      <input type="text" placeholder="e.g. basalt, concrete, grass" value="${state.globalMatQuery}" oninput="state.globalMatQuery=this.value; filterLibraries()"/>
    </div>
  </div>

  <div style="display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin:0 0 14px">
    <button class="btn emerald btn-sm" onclick="importSpectraDialog()" title="Read a spectral library file (CSV, ASCII, USGS, ECOSTRESS, ENVI .sli, zip) into a new library">+ Add library (import file)…</button>
    <button class="btn ghost btn-sm" onclick="lfOpen('builder')()" title="Resample, merge and save spectra as a new library">Build library…</button>
    <span class="note" style="margin:0">Libraries made from collected pixel spectra or object samples are saved from the Image page and listed here.</span>
  </div>

  ${addForm}
  ${mainLibPanel()}
  ${renderStartupIssues()}
  <div id="libGlobalResults">${buildGlobalMatResults()}</div>

  <div id="libCardsGrid" class="grid cols-3">${buildLibraryCards()}</div>
  `;
}

/* =========================================================================
   RENDER — MATERIAL LIST (inside a library)
   ========================================================================= */
function filterMaterials(){
  const lib = state.libraries.find(l=>l.id===state.activeLibId);
  document.getElementById("matRows").innerHTML = buildMaterialRows(lib);
}
function renderMaterialList(){
  const lib = state.libraries.find(l=>l.id===state.activeLibId);
  if(!lib) return `<p>Library not found.</p>`;

  const addForm = state.showAddMaterial ? `
    <div class="add-form">
      <div><label class="small">Name</label><input type="text" id="newMatName" placeholder="e.g. River sand"></div>
      <div><label class="small">Type</label>
        <select id="newMatType">${MATERIAL_TYPES.map(t=>`<option>${t}</option>`).join("")}</select>
      </div>
      <div><label class="small">Samples</label><input type="text" id="newMatSamples" placeholder="20" style="width:70px;"></div>
      <div><label class="small">Latitude</label><input type="text" id="newMatLat" placeholder="optional" style="width:90px;"></div>
      <div><label class="small">Longitude</label><input type="text" id="newMatLon" placeholder="optional" style="width:90px;"></div>
      <div><label class="small">Location name</label><input type="text" id="newMatLoc" placeholder="optional" style="width:150px;"></div>
      <button class="btn emerald btn-sm" onclick="submitAddMaterial('${lib.id}')">Create</button>
      <button class="btn ghost btn-sm" onclick="state.showAddMaterial=false; renderMainOnly()">Cancel</button>
    </div>` : "";

  const ip = state.importPreview;
  const importPanel = (ip && ip.libId===lib.id) ? `
    <div class="add-form">
      <div style="width:100%; font-size:12.5px; color:var(--muted2); margin-bottom:2px;">
        Parsed ${ip.rows.length} wavelength/reflectance points from a ${ip.format} file. Review and confirm:
      </div>
      <div><label class="small">Name</label><input type="text" id="importMatName" value="${ip.suggestedName}"></div>
      <div><label class="small">Type</label>
        <select id="importMatType">${MATERIAL_TYPES.map(t=>`<option>${t}</option>`).join("")}</select>
      </div>
      <div><label class="small">Latitude</label><input type="text" id="importMatLat" placeholder="optional" style="width:90px;"></div>
      <div><label class="small">Longitude</label><input type="text" id="importMatLon" placeholder="optional" style="width:90px;"></div>
      <div><label class="small">Location name</label><input type="text" id="importMatLoc" placeholder="optional" style="width:150px;"></div>
      <button class="btn emerald btn-sm" onclick="submitImport()">Import</button>
      <button class="btn ghost btn-sm" onclick="cancelImport()">Cancel</button>
    </div>` : "";

  return `
  <div class="crumb" onclick="setView('library-list')">&larr; Library</div>
  <h1 class="pagetitle">${lib.name}</h1>
  <p class="pagesub">${dbCountText(lib)}</p>

  <div class="search-row">
    <div>
      <label class="small">Search materials by name</label>
      <input type="text" placeholder="e.g. Granite" value="${state.matQuery}" oninput="state.matQuery=this.value; filterMaterials()"/>
    </div>
    <div>
      <label class="small">Filter by type</label>
      <select onchange="state.matTypeFilter=this.value; filterMaterials()">
        <option value="">All types</option>
        ${libTypes(lib).map(t=>`<option ${state.matTypeFilter===t?'selected':''}>${escapeHtml(t)}</option>`).join("")}
      </select>
    </div>
    <button class="btn ghost btn-sm" onclick="triggerImportFile('${lib.id}')">Upload files into this library</button>
    ${lib.db?"":`<button class="btn emerald btn-sm" onclick="state.showAddMaterial=!state.showAddMaterial; renderMainOnly()">+ Add material</button>`}
  </div>
  <input type="file" id="importFileInput" multiple style="display:none" onchange="handleImportFile(event)"/>
  <p style="font-size:11.5px; color:var(--muted); margin:-8px 0 14px;">Uploaded files are read on the server (CSV, TXT tables, .sli, USGS/ECOSTRESS files, .zip) and added to this library.</p>

  ${addForm}
  ${importPanel}

  <div id="matRows" class="panel-list">${buildMaterialRows(lib)}</div>
  `;
}

/* =========================================================================
   RENDER — SPECTRUM VIEWER (single material)
   ========================================================================= */
function renderSpectrumViewerLocal(){
  const lib = state.libraries.find(l=>l.id===state.activeLibId);
  const mat = lib?.materials.find(m=>m.id===state.activeMatId);
  if(!lib || !mat) return `<p>Material not found.</p>`;
  const hasReal = !!mat.customData, empty = mat.user && !hasReal;
  const data = hasReal ? mat.customData : makeSpectrum(mat.seed);
  const chart = svgLineChart([{name:mat.name, color:"#34d399", data:data.map(d=>({x:d.wavelength,y:d.reflectance}))}]);
  const isComparing = state.compareList.some(c=>c.libId===lib.id && c.matId===mat.id);
  const fname = mat.name.replace(/\s+/g,"_");
  const hasLoc = mat.latitude!=null && mat.longitude!=null;

  return `
  <div class="crumb" onclick="goToLibrary('${lib.id}')">&larr; ${lib.name}</div>
  <h1 class="pagetitle">${mat.name}</h1>
  <p class="pagesub">${mat.type} · ${mat.samples} samples · resolution ${mat.resolution} · 0.4–2.5 &micro;m${mat.sourceFormat ? ` · imported from ${mat.sourceFormat}` : ""}</p>
  ${hasLoc ? `<p style="font-size:12.5px; color:var(--muted2); margin:-10px 0 16px;">📍 ${mat.locationName || "Location"} (${mat.latitude.toFixed(4)}, ${mat.longitude.toFixed(4)})</p>` : ""}

  ${empty ? `<div class="placeholder-card">No spectrum yet. Add one from a CSV/TXT/JSON file, or save an image pixel to a sample with this name.</div>` : `<div class="chart-wrap">${chart}</div>`}
  ${mat.spectra && mat.spectra.length ? `<p style="font-size:11.5px; color:var(--muted); margin:8px 0 0;">${mat.spectra.length} spectrum${mat.spectra.length===1?"":"s"} stored (${[...new Set(mat.spectra.map(s=>s.origin))].join(", ")}) — saved in this browser. The chart shows the first.</p>` : ""}

  <div style="display:flex; gap:10px; margin-top:14px; flex-wrap:wrap;">
    ${empty ? "" : `<button class="btn ghost btn-sm" onclick='downloadCSV("${fname}.csv", ${JSON.stringify(data)})'>Export CSV</button>
    <button class="btn ghost btn-sm" onclick='downloadTXT("${fname}.txt", ${JSON.stringify(data)})'>Export TXT</button>
    <button class="btn ghost btn-sm" onclick='downloadJSON("${fname}.json", ${JSON.stringify(data)})'>Export JSON</button>`}
    ${mat.user ? `<button class="btn ghost btn-sm" onclick="document.getElementById('attachFileInput').click()">Add spectrum from file</button>
    <input type="file" id="attachFileInput" accept=".csv,.txt,.json,.asc" style="display:none" onchange="handleAttachFile(event)"/>` : ""}
    <button class="btn ${isComparing ? 'ghost' : 'emerald'} btn-sm" onclick="toggleCompare('${lib.id}','${mat.id}')">
      ${isComparing ? 'Remove from comparison' : 'Add to comparison'}
    </button>
    ${state.compareList.length>0 ? `<button class="btn ghost btn-sm" onclick="setView('comparison')">View comparison (${state.compareList.length})</button>` : ""}
  </div>
  `;
}

/* =========================================================================
   RENDER — COMPARISON VIEW
   ========================================================================= */

/* =========================================================================
   RENDER — LIBRARY MAP (georeferencing)
   ========================================================================= */
function renderLibraryMap(){
  const geoMats = [];
  state.libraries.forEach(l=>{
    l.materials.forEach(m=>{
      if(m.latitude!=null && m.longitude!=null) geoMats.push({l,m});
    });
  });

  if(geoMats.length===0){
    return `
    <div class="crumb" onclick="setView('library-list')">&larr; Library</div>
    <h1 class="pagetitle">Map view</h1>
    <p class="pagesub">Materials with a saved location will show up here.</p>
    <div class="placeholder-card">No georeferenced materials yet — add latitude/longitude when creating or importing a material.</div>
    `;
  }

  const w=760, h=380;
  const toXY = (lat,lon) => ({ x: (lon+180)/360*w, y: (90-lat)/180*h });
  let grid = "";
  for(let lon=-180; lon<=180; lon+=30){
    const x = (lon+180)/360*w;
    grid += `<line x1="${x}" y1="0" x2="${x}" y2="${h}" stroke="#1e293b" stroke-width="1"/>`;
  }
  for(let lat=-90; lat<=90; lat+=30){
    const y = (90-lat)/180*h;
    grid += `<line x1="0" y1="${y}" x2="${w}" y2="${y}" stroke="#1e293b" stroke-width="1"/>`;
  }
  const pins = geoMats.map(({l,m})=>{
    const {x,y} = toXY(m.latitude, m.longitude);
    return `<g style="cursor:pointer;" onclick="goToMaterial('${l.id}','${m.id}')">
      <circle cx="${x}" cy="${y}" r="6" fill="#34d399" stroke="#022c22" stroke-width="1.5"/>
      <title>${m.name} — ${m.locationName || ''} (${m.latitude.toFixed(2)}, ${m.longitude.toFixed(2)})</title>
    </g>`;
  }).join("");

  const listRows = geoMats.map(({l,m})=>`
    <div class="mat-row" onclick="goToMaterial('${l.id}','${m.id}')">
      <span>${m.name} <span style="color:var(--muted); font-size:11.5px;">— ${l.name}</span></span>
      <span class="mat-tag">${m.locationName || `${m.latitude.toFixed(2)}, ${m.longitude.toFixed(2)}`}</span>
    </div>
  `).join("");

  return `
  <div class="crumb" onclick="setView('library-list')">&larr; Library</div>
  <h1 class="pagetitle">Map view</h1>
  <p class="pagesub">${geoMats.length} georeferenced material${geoMats.length>1?'s':''}. Click a pin to open its spectrum.</p>

  <div class="chart-wrap" style="overflow-x:auto;">
    <svg viewBox="0 0 ${w} ${h}" width="100%" style="min-width:520px; background:#020617; border-radius:6px;">
      ${grid}
      ${pins}
    </svg>
  </div>
  <p style="font-size:11px; color:var(--muted); margin-top:8px;">Simple lat/lon grid, not a real basemap — this demo has no map-tile service.</p>

  <div class="panel-list" style="margin-top:16px;">${listRows}</div>
  `;
}
