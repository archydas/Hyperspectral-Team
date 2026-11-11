/* library-store-cnn.js — Analysis placeholder, local library store, 3D-CNN baseline and its panel
   (classic script, loaded in order from index.html; all functions are global) */
/* =========================================================================
   RENDER — ANALYSIS (placeholder)
   ========================================================================= */

/* LIBSTORE-START */
/* =========================================================================
   LIBRARY STORE — validated sample + spectrum creation, persisted to localStorage.
   createSample() makes a material (sample); addSpectrum() attaches a real spectrum to it.
   Only user-created data is persisted; the seeded demo libraries stay placeholders.
   ========================================================================= */
const LIB_STORE_KEY = "ds3_user_library_v1";
function validateSpectrumRows(rows){
  if(!Array.isArray(rows) || rows.length < 2) throw new Error("Spectrum needs at least 2 wavelength/reflectance points");
  for(let i=0;i<rows.length;i++){
    const w=rows[i].wavelength, r=rows[i].reflectance;
    if(!Number.isFinite(w) || !Number.isFinite(r)) throw new Error(`Non-numeric value at point ${i+1}`);
    if(i>0 && w<=rows[i-1].wavelength) throw new Error(`Wavelengths must be strictly increasing (point ${i+1})`);
    if(r < -100) throw new Error(`Reflectance ${r} at point ${i+1} looks like a no-data value`);
  }
}
function sortRows(r){ return Array.isArray(r) ? r.slice().sort((a,b)=>a.wavelength-b.wavelength) : r; }
function createSample(libId, d){
  const lib = state.libraries.find(l=>l.id===libId);
  if(!lib) throw new Error("Library not found");
  const name = (d.name||"").trim();
  if(!name) throw new Error("Name is required");
  if(lib.materials.some(m=>m.name.toLowerCase()===name.toLowerCase())) throw new Error(`"${name}" already exists in ${lib.name}`);
  const num = (v,lo,hi,what)=>{ if(v===undefined||v===null||v==="") return null; const n=Number(v);
    if(!Number.isFinite(n)||n<lo||n>hi) throw new Error(`${what} must be a number between ${lo} and ${hi}`); return n; };
  const mat = { id:`mat-${libId}-${Date.now()}-${lib.materials.length}`, name, type:d.type||"Manmade", samples:d.samples||0,
    resolution:"1 nm", seed:(Date.now()%10000)+lib.materials.length, customData:null, spectra:[], user:true,
    latitude:num(d.latitude,-90,90,"Latitude"), longitude:num(d.longitude,-180,180,"Longitude"),
    locationName:d.locationName||null, sourceFormat:d.sourceFormat||null };
  lib.materials.push(mat);
  return mat;
}
function addSpectrum(libId, matId, d){
  const lib = state.libraries.find(l=>l.id===libId), mat = lib && lib.materials.find(m=>m.id===matId);
  if(!mat) throw new Error("Sample not found");
  d = {...d, data:sortRows(d.data)};
  validateSpectrumRows(d.data);
  if(!["image_pixel","field","library_import"].includes(d.origin)) throw new Error("origin must be image_pixel, field or library_import");
  mat.spectra = mat.spectra || (mat.customData ? [{id:"sp-0",origin:"library_import",pixel:null,data:mat.customData}] : []);
  const sp = { id:`sp-${Date.now()}-${mat.spectra.length}`, origin:d.origin, pixel:d.pixel||null, data:d.data.map(r=>({wavelength:r.wavelength, reflectance:r.reflectance})) };
  mat.spectra.push(sp);
  if(!mat.customData){ mat.customData = sp.data; mat.samples = sp.data.length; if(d.sourceFormat) mat.sourceFormat = d.sourceFormat; }
  mat.user = true;
  return sp;
}
function persistUserLibrary(){
  try{
    const r6 = v=>Number(v.toPrecision(6));
    const payload = { v:1,
      libs: state.libraries.filter(l=>l.user).map(l=>({id:l.id,name:l.name,source:l.source,color:l.color})),
      mats: state.libraries.flatMap(l=>l.materials.filter(m=>m.user).map(m=>({ libId:l.id, id:m.id, name:m.name, type:m.type,
        latitude:m.latitude, longitude:m.longitude, locationName:m.locationName, sourceFormat:m.sourceFormat,
        spectra:(m.spectra||[]).map(s=>({id:s.id,origin:s.origin,pixel:s.pixel,w:s.data.map(d=>r6(d.wavelength)),r:s.data.map(d=>r6(d.reflectance))})) }))) };
    localStorage.setItem(LIB_STORE_KEY, JSON.stringify(payload)); return true;
  }catch(e){ console.error(e); if(typeof showToast==="function") showToast("Couldn't persist the library (browser storage full or blocked)"); return false; }
}
function restoreUserLibrary(){
  try{
    const raw = localStorage.getItem(LIB_STORE_KEY); if(!raw) return 0;
    const p = JSON.parse(raw); let n=0;
    (p.libs||[]).forEach(l=>{ if(!state.libraries.some(x=>x.id===l.id)) state.libraries.push({...l, count:0, seed:state.libraries.length+10, materials:[], user:true}); });
    (p.mats||[]).forEach(m=>{
      const lib = state.libraries.find(l=>l.id===m.libId); if(!lib || lib.materials.some(x=>x.id===m.id)) return;
      const spectra = m.spectra.map(s=>({id:s.id,origin:s.origin,pixel:s.pixel,data:s.w.map((w,i)=>({wavelength:w,reflectance:s.r[i]}))}));
      lib.materials.push({ id:m.id,name:m.name,type:m.type,samples:spectra[0]?spectra[0].data.length:0,resolution:"1 nm",seed:1,
        customData:spectra[0]?spectra[0].data:null, spectra, user:true, latitude:m.latitude, longitude:m.longitude,
        locationName:m.locationName, sourceFormat:m.sourceFormat }); n++;
    });
    return n;
  }catch(e){ console.error(e); return 0; }
}
/* LIBSTORE-END */

/* CNN-START */
/* =========================================================================
   3D-CNN BASELINE — dependency-free. Input = 5x5 pixel patch x 16 spectral bins
   (spectra resampled by wavelength, z-scored per pixel). Conv3D(8 filters, 3x3x3)+ReLU -> Dense -> Softmax.
   ========================================================================= */
const CNN_D=16, CNN_P=5, CNN_F=8, CNN_MIN_CONF=0.5;
function mulberry32(a){ return function(){ a|=0; a=a+0x6D2B79F5|0; let t=Math.imul(a^a>>>15,1|a); t=t+Math.imul(t^t>>>7,61|t)^t; return ((t^t>>>14)>>>0)/4294967296; }; }
function gauss(r){ let u=0; while(u===0) u=r(); return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*r()); }
function cnnCreate(K, seed=1){
  const r=mulberry32(seed), Do=CNN_D-2, Po=CNN_P-2, H=CNN_F*Do*Po*Po;
  const net={K,Do,Po,H,W:new Float64Array(CNN_F*27),b:new Float64Array(CNN_F),Wd:new Float64Array(K*H),bd:new Float64Array(K)};
  for(let i=0;i<net.W.length;i++) net.W[i]=gauss(r)*Math.sqrt(2/27);
  for(let i=0;i<net.Wd.length;i++) net.Wd[i]=gauss(r)*Math.sqrt(1/H);
  return net;
}
function cnnForward(net,x){
  const {Do,Po,H,K}=net, PP=CNN_P*CNN_P, a=new Float64Array(H); let o=0;
  for(let f=0;f<CNN_F;f++){ const wf=f*27;
    for(let d=0;d<Do;d++) for(let i=0;i<Po;i++) for(let j=0;j<Po;j++){
      let s=net.b[f];
      for(let kd=0;kd<3;kd++) for(let ki=0;ki<3;ki++){ const base=(d+kd)*PP+(i+ki)*CNN_P+j, wb=wf+kd*9+ki*3;
        s+=net.W[wb]*x[base]+net.W[wb+1]*x[base+1]+net.W[wb+2]*x[base+2]; }
      a[o++]=s>0?s:0; } }
  const p=new Float64Array(K); let mx=-Infinity;
  for(let k=0;k<K;k++){ let s=net.bd[k]; const wk=k*H; for(let n=0;n<H;n++) s+=net.Wd[wk+n]*a[n]; p[k]=s; if(s>mx) mx=s; }
  let sum=0; for(let k=0;k<K;k++){ p[k]=Math.exp(p[k]-mx); sum+=p[k]; } for(let k=0;k<K;k++) p[k]/=sum;
  return {a,p};
}
function cnnNewGrad(net){ return {W:new Float64Array(net.W.length),b:new Float64Array(CNN_F),Wd:new Float64Array(net.Wd.length),bd:new Float64Array(net.K)}; }
function cnnGrad(net,x,y,g){
  const {a,p}=cnnForward(net,x), {Do,Po,H,K}=net, PP=CNN_P*CNN_P, da=new Float64Array(H);
  for(let k=0;k<K;k++){ const dz=p[k]-(k===y?1:0), wk=k*H; g.bd[k]+=dz;
    for(let n=0;n<H;n++){ g.Wd[wk+n]+=dz*a[n]; da[n]+=net.Wd[wk+n]*dz; } }
  let o=0;
  for(let f=0;f<CNN_F;f++){ const wf=f*27;
    for(let d=0;d<Do;d++) for(let i=0;i<Po;i++) for(let j=0;j<Po;j++){
      const dv=a[o]>0?da[o]:0; o++; if(dv===0) continue; g.b[f]+=dv;
      for(let kd=0;kd<3;kd++) for(let ki=0;ki<3;ki++){ const base=(d+kd)*PP+(i+ki)*CNN_P+j, wb=wf+kd*9+ki*3;
        g.W[wb]+=dv*x[base]; g.W[wb+1]+=dv*x[base+1]; g.W[wb+2]+=dv*x[base+2]; } } }
  let best=0; for(let k=1;k<K;k++) if(p[k]>p[best]) best=k;
  return {loss:-Math.log(Math.max(p[y],1e-12)), correct:best===y};
}
function cnnAdamInit(net){ const z=()=>({W:new Float64Array(net.W.length),b:new Float64Array(CNN_F),Wd:new Float64Array(net.Wd.length),bd:new Float64Array(net.K)}); return {t:0,m:z(),v:z()}; }
function cnnAdamStep(net,g,st,n,lr){
  st.t++; const c1=1-Math.pow(0.9,st.t), c2=1-Math.pow(0.999,st.t);
  for(const key of ["W","b","Wd","bd"]){ const P=net[key],G=g[key],M=st.m[key],V=st.v[key];
    for(let i=0;i<P.length;i++){ const gi=G[i]/n; M[i]=0.9*M[i]+0.1*gi; V[i]=0.999*V[i]+0.001*gi*gi; P[i]-=lr*(M[i]/c1)/(Math.sqrt(V[i]/c2)+1e-8); G[i]=0; } }
}
function zscore(v){ let m=0; for(const x of v) m+=x; m/=v.length; let s=0; for(const x of v) s+=(x-m)*(x-m); s=Math.sqrt(s/v.length)+1e-6; return v.map(x=>(x-m)/s); }
function toMicrons(wl){ return Math.max(...wl)>50 ? wl.map(w=>w/1000) : wl.slice(); }
function makeGrid(lo,hi){ return Array.from({length:CNN_D},(_,i)=>lo+(hi-lo)*i/(CNN_D-1)); }
function resampleToGrid(wl,vals,grid){          // linear interpolation, clamped at the ends
  return grid.map(g=>{ if(g<=wl[0]) return vals[0]; if(g>=wl[wl.length-1]) return vals[vals.length-1];
    let i=1; while(wl[i]<g) i++; const t=(g-wl[i-1])/(wl[i]-wl[i-1]); return vals[i-1]*(1-t)+vals[i]*t; });
}
function libRowsToGrid(rows,grid){
  const wl=toMicrons(rows.map(r=>r.wavelength)), cover=(Math.min(wl[wl.length-1],grid[grid.length-1])-Math.max(wl[0],grid[0]))/(grid[grid.length-1]-grid[0]);
  return cover<0.7 ? null : resampleToGrid(wl,rows.map(r=>r.reflectance),grid);   // null = library spectrum doesn't cover the image range
}
function synthPatch(spec,rnd){                  // library spectrum -> spatially uniform noisy patch
  const PP=CNN_P*CNN_P, x=new Float64Array(CNN_D*PP), mean=spec.reduce((a,b)=>a+b,0)/spec.length;
  for(let p=0;p<PP;p++){ const z=zscore(spec.map(v=>v*(1+0.03*gauss(rnd))+0.01*mean*gauss(rnd))); for(let d=0;d<CNN_D;d++) x[d*PP+p]=z[d]; }
  return x;
}
function jitterPatch(x,rnd){ const y=new Float64Array(x.length); for(let i=0;i<x.length;i++) y[i]=x[i]+0.15*gauss(rnd); return y; }
function cubeSampler(cube,w,h,imgWl,grid){      // returns patchAt(x,y) reading a real cube (array of per-band arrays)
  const wl=toMicrons(imgWl), idx=grid.map(g=>{ if(g<=wl[0]) return [0,0,0]; if(g>=wl[wl.length-1]) return [wl.length-1,wl.length-1,0];
    let i=1; while(wl[i]<g) i++; return [i-1,i,(g-wl[i-1])/(wl[i]-wl[i-1])]; });
  const PP=CNN_P*CNN_P, R=(CNN_P-1)/2;
  return (x,y)=>{ const out=new Float64Array(CNN_D*PP); let p=0;
    for(let dy=-R;dy<=R;dy++) for(let dx=-R;dx<=R;dx++,p++){
      const xx=Math.min(w-1,Math.max(0,x+dx)), yy=Math.min(h-1,Math.max(0,y+dy)), pos=yy*w+xx;
      const z=zscore(idx.map(([a,b,t])=>cube[a][pos]*(1-t)+cube[b][pos]*t)); for(let d=0;d<CNN_D;d++) out[d*PP+p]=z[d]; }
    return out; };
}
/* trainSet: [{name, real:[patch...], lib:[gridSpectrum...]}]; returns {net, report}. onEpoch is called each epoch (may be async). */
async function cnnTrain(trainSet, opts={}, onEpoch){
  const o={epochs:30,perClass:48,batch:16,lr:0.005,seed:7,valFrac:0.25,...opts}, rnd=mulberry32(o.seed), K=trainSet.length, t0=Date.now();
  const net=cnnCreate(K,o.seed), st=cnnAdamInit(net), g=cnnNewGrad(net), val=[], pools=[];
  trainSet.forEach((c,k)=>{ const real=c.real.slice().sort(()=>rnd()-0.5), nv=real.length>=4?Math.max(1,Math.round(real.length*o.valFrac)):0;
    real.slice(0,nv).forEach(p=>val.push({x:p,y:k})); pools.push({real:real.slice(nv),lib:c.lib}); });
  const draw=k=>{ const P=pools[k]; if(P.real.length && (!P.lib.length || rnd()<0.5)) return jitterPatch(P.real[Math.floor(rnd()*P.real.length)],rnd);
    return synthPatch(P.lib[Math.floor(rnd()*P.lib.length)],rnd); };
  let last={loss:0,acc:0};
  for(let ep=0;ep<o.epochs;ep++){
    const S=[]; for(let k=0;k<K;k++) for(let i=0;i<o.perClass;i++) S.push(k); S.sort(()=>rnd()-0.5);
    let loss=0,ok=0;
    for(let b=0;b<S.length;b+=o.batch){ const B=S.slice(b,b+o.batch);
      for(const k of B){ const r=cnnGrad(net,draw(k),k,g); loss+=r.loss; ok+=r.correct?1:0; }
      cnnAdamStep(net,g,st,B.length,o.lr); }
    last={loss:loss/S.length,acc:ok/S.length};
    if(onEpoch) await onEpoch(ep+1,o.epochs,last);
  }
  let vok=0; const perClass=trainSet.map(()=>({n:0,ok:0}));
  val.forEach(v=>{ const p=cnnForward(net,v.x).p; let b=0; for(let k=1;k<K;k++) if(p[k]>p[b]) b=k; perClass[v.y].n++; if(b===v.y){ vok++; perClass[v.y].ok++; } });
  return {net, report:{classes:trainSet.map((c,k)=>({name:c.name,nReal:c.real.length,nLib:c.lib.length,valN:perClass[k].n,valOk:perClass[k].ok})),
    epochs:o.epochs, loss:last.loss, trainAcc:last.acc, valN:val.length, valAcc:val.length?vok/val.length:null, seconds:(Date.now()-t0)/1000}};
}
function cnnPredict(net,x){ const p=cnnForward(net,x).p; let b=0; for(let k=1;k<net.K;k++) if(p[k]>p[b]) b=k; return {k:b,conf:p[b]}; }
/* CNN-END */

/* =========================================================================
   3D-CNN UI GLUE — training panel, train-from-labels+library, ROI inference
   ========================================================================= */
let cnnState = { model:null, report:null, training:false, roiModel:"sam", libId:"", usePlaceholders:false };
function imageWavelengths(t){ return t.wavelengths ? t.wavelengths : Array.from({length:t.bandCount},(_,i)=>0.4+2.1*i/Math.max(t.bandCount-1,1)); }
function renderCnnPanel(){
  const t = imgState.tiff; if(!t) return "";
  const ready = !!t.cube, rep = cnnState.report;
  const libOpts = `<option value="">None (labeled pixels only)</option>` + state.libraries.map(l=>{
    const n = l.materials.filter(m=>m.customData).length;
    return `<option value="${l.id}" ${cnnState.libId===l.id?"selected":""}>${l.name} (${n} with real spectra)</option>`; }).join("");
  const rows = rep ? rep.classes.map(c=>`<tr><td>${c.name}</td><td>${c.nReal}</td><td>${c.nLib}</td><td>${c.valN ? `${c.valOk}/${c.valN}` : "—"}</td></tr>`).join("") : "";
  return `
  <div class="chart-wrap" style="margin-top:16px;">
    <p style="font-size:13px; color:#e2e8f0; margin:0 0 6px;">3D-CNN detection model (baseline)</p>
    <p style="font-size:11.5px; color:var(--muted); margin:0 0 10px;">Trains on (1) the pixels you labeled below, as real 5×5×${CNN_D}-band patches, and (2) real spectra from a library, matched to labels by name and augmented with noise. Placeholder demo materials are excluded unless ticked.</p>
    <div class="add-form">
      <div><label class="small">Training library</label><select id="cnnLibSelect">${libOpts}</select></div>
      <div><label class="small">&nbsp;</label><label style="font-size:11.5px; color:var(--muted);"><input type="checkbox" id="cnnPlaceholders" ${cnnState.usePlaceholders?"checked":""}> include placeholder materials (demo only)</label></div>
      <button class="btn emerald btn-sm" onclick="trainCnnFromUi()" ${(!ready||cnnState.training)?"disabled":""}>${cnnState.training?"Training…":"Train 3D-CNN"}</button>
    </div>
    <p id="cnnStatus" style="font-size:11.5px; color:#fbbf24; margin:8px 0 0;">${!ready ? "Load the full spectral cube above first — training needs real pixel spectra." : (cnnState.training ? "Starting…" : "")}</p>
    ${rep ? `
    <table class="gcp-table" style="margin-top:8px;"><thead><tr><th>Class</th><th>Labeled pixels</th><th>Library spectra</th><th>Held-out correct</th></tr></thead><tbody>${rows}</tbody></table>
    <p style="font-size:11.5px; color:var(--muted); margin:8px 0 0;">${rep.epochs} epochs in ${rep.seconds.toFixed(1)} s · train accuracy ${(rep.trainAcc*100).toFixed(0)}% · held-out labeled pixels: ${rep.valAcc===null ? "none (need ≥4 labels in a class)" : `${(rep.valAcc*100).toFixed(0)}% of ${rep.valN}`}.
    Held-out pixels can sit next to training pixels, so treat this as optimistic. ${rep.skipped ? `${rep.skipped} library spectrum(s) skipped: they don't cover this image's wavelength range.` : ""} ${rep.placeholder ? '<span style="color:#fbbf24;">Placeholder (synthetic) materials were used.</span>' : ""}</p>` : ""}
  </div>`;
}
async function trainCnnFromUi(){
  const t = imgState.tiff;
  if(!t || !t.cube){ showToast("Load the full spectral cube first"); return; }
  if(cnnState.training) return;
  cnnState.libId = document.getElementById("cnnLibSelect").value;
  cnnState.usePlaceholders = document.getElementById("cnnPlaceholders").checked;
  const imgWl = imageWavelengths(t), wlu = toMicrons(imgWl), grid = makeGrid(wlu[0], wlu[wlu.length-1]);
  const patchAt = cubeSampler(t.cube, t.width, t.height, imgWl, grid);
  const byName = new Map(), get = n=>{ const k=n.trim().toLowerCase(); if(!byName.has(k)) byName.set(k,{name:n.trim(),real:[],lib:[]}); return byName.get(k); };
  imgState.labels.forEach(l=>get(l.label).real.push(patchAt(l.x,l.y)));
  let skipped = 0, placeholder = false;
  const lib = state.libraries.find(l=>l.id===cnnState.libId);
  if(lib && lib.db){ showToast("Reading the library spectra from the database…"); try{ await dbHydrate(lib); }catch(e){ showToast(e.message); return; } }
  if(lib) lib.materials.forEach(m=>{
    let specs = (m.spectra && m.spectra.length) ? m.spectra.map(s=>s.data) : (m.customData ? [m.customData] : []);
    if(!specs.length && cnnState.usePlaceholders && m.seed!==undefined){ specs = [makeSpectrum(m.seed,0.4,2.5,24)]; placeholder = true; }
    specs.forEach(rows=>{ const g = libRowsToGrid(rows, grid); if(g) get(m.name).lib.push(g); else skipped++; });
  });
  const set = [...byName.values()].filter(c=>c.real.length || c.lib.length);
  if(set.length < 2){ showToast("Need at least 2 classes: label pixels and/or choose a library with real spectra"); return; }
  cnnState.training = true; renderMainOnly();
  const {net, report} = await cnnTrain(set, {}, async(ep,n,s)=>{
    const el = document.getElementById("cnnStatus"); if(el) el.textContent = `Epoch ${ep}/${n} — loss ${s.loss.toFixed(3)}, train accuracy ${(s.acc*100).toFixed(0)}%`;
    await new Promise(r=>setTimeout(r,0)); });
  cnnState = {...cnnState, model:{net, names:set.map(c=>c.name), grid, bandCount:t.bandCount}, report:{...report, skipped, placeholder}, training:false, roiModel:"cnn"};
  renderMainOnly();
  showToast(`3D-CNN trained on ${set.length} classes (${report.seconds.toFixed(1)} s)`);
}
function roiGroundInfo(stride){
  if(!imgState.geo) return null;
  const p0 = pixelLatLon(0,0), px = pixelLatLon(stride,0), py = pixelLatLon(0,stride), mLat = 111320, mLon = 111320*Math.cos(p0.lat*Math.PI/180);
  return { dxM: Math.hypot((px.lat-p0.lat)*mLat,(px.lon-p0.lon)*mLon), dyM: Math.hypot((py.lat-p0.lat)*mLat,(py.lon-p0.lon)*mLon) };
}
async function runRoiCnn(){
  const m = cnnState.model, t = imgState.tiff;
  if(!m){ showToast("Train the 3D-CNN first"); return; }
  if(!(t && t.cube)){ showToast("Load the full spectral cube first — the 3D-CNN needs real pixel spectra"); return; }
  if(!imgState.roi.closed || imgState.roi.points.length<3){ showToast("Draw and finish a polygon first"); return; }
  const imgWl = imageWavelengths(t), wlu = toMicrons(imgWl), g0 = m.grid[0], g1 = m.grid[CNN_D-1];
  if((Math.min(wlu[wlu.length-1],g1)-Math.max(wlu[0],g0))/(g1-g0) < 0.9){ showToast("This image's wavelength range doesn't match the range the model was trained on"); return; }
  const stride = parseInt(document.getElementById("roiResSelect").value,10), pts = imgState.roi.points, bbox = roiBoundingBox(pts);
  const minX = Math.max(0,Math.floor(bbox.minX)), maxX = Math.min(t.width-1,Math.ceil(bbox.maxX)), minY = Math.max(0,Math.floor(bbox.minY)), maxY = Math.min(t.height-1,Math.ceil(bbox.maxY));
  showToast("Running 3D-CNN…"); await new Promise(r=>setTimeout(r,10));
  const patchAt = cubeSampler(t.cube, t.width, t.height, imgWl, m.grid), t0 = performance.now();
  const PALETTE = ["#34d399","#60a5fa","#f472b6","#fbbf24","#a78bfa","#fb7185","#22d3ee","#facc15"], colorFor = {"Unclassified":"#64748b"}, referenceIsReal = {"Unclassified":true};
  m.names.forEach((n,i)=>{ colorFor[n] = PALETTE[i%PALETTE.length]; referenceIsReal[n] = true; });
  const matches = [], counts = {}; let sampled = 0, inPoly = 0, confSum = 0;
  for(let y=minY; y<=maxY; y+=stride) for(let x=minX; x<=maxX; x+=stride){
    if(!pointInPolygon(x,y,pts)) continue; inPoly++;
    const r = cnnPredict(m.net, patchAt(x,y)), name = r.conf>=CNN_MIN_CONF ? m.names[r.k] : "Unclassified";
    matches.push({x,y,materialName:name}); counts[name] = (counts[name]||0)+1; sampled++; confSum += r.conf;
  }
  const elapsedMs = performance.now()-t0, unc = counts["Unclassified"]||0;
  imgState.roiResult = { stride, counts, colorFor, matches, elapsedMs, sampledCount:sampled, totalInPoly:inPoly, boxW:maxX-minX+1, boxH:maxY-minY+1,
    referenceIsReal, groundInfo:roiGroundInfo(stride), usedRealPixelSpectra:true,
    modelNote:`3D-CNN, ${m.names.length} classes · mean confidence ${(confSum/Math.max(sampled,1)*100).toFixed(0)}% · ${unc} of ${sampled} points under ${CNN_MIN_CONF*100}% confidence left unclassified.` };
  renderMainOnly(); showToast(`3D-CNN classified ${sampled} sampled pixels in ${elapsedMs.toFixed(0)} ms`);
}
function handleAttachFile(e){
  const file = e.target.files?.[0]; if(!file) return;
  const ext = (file.name.split(".").pop()||"").toLowerCase(), reader = new FileReader();
  reader.onload = ()=>{
    const rows = ["csv","txt","json","asc"].includes(ext) ? parseSpectrumText(reader.result, ext) : null;
    if(!rows){ showToast("Couldn't read wavelength/reflectance pairs (use CSV, TXT or JSON)"); return; }
    try{ addSpectrum(state.activeLibId, state.activeMatId, {data:rows, origin:"library_import", sourceFormat:ext.toUpperCase()}); persistUserLibrary(); showToast(`Spectrum added (${rows.length} points)`); renderMainOnly(); }
    catch(err){ showToast(err.message); }
  };
  reader.readAsText(file); e.target.value = "";
}
