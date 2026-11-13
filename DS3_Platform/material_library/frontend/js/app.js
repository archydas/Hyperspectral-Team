/* app.js — start-up. Loaded last, after every other file has defined its functions. */
restoreUserLibrary();
render();
loadLibraries();
renderMenubar();

/* Embedded in the DS3 Qt window (?embed=image or ?embed=library): no own sidebar, open the requested module.
   Without the sidebar there is no way back from a sub-page, so a slim bar with "← Back" appears on every page that
   is not the start page of the module (Back goes to the page you came from). */
(function(){
  try{
    const m = new URLSearchParams(location.search).get("embed");
    if(!m) return;
    document.body.classList.add("embed");
    const HOME = m === "image" ? "image" : "library-list", TITLE = m === "image" ? "Image" : "Library";
    const hist = []; let last = null;
    const bar = document.createElement("div");
    bar.id = "embedBar"; bar.style.display = "none";
    const menu = document.getElementById("menubar");
    menu.parentNode.insertBefore(bar, menu.nextSibling);
    window.embedBack = function(){
      const to = hist.length ? hist.pop() : HOME; last = null;           // the next render must not push again
      if(typeof winClose === "function") try{ winClose(); }catch(_){}
      setView(to);
    };
    window.embedHome = function(){ hist.length = 0; last = null; setView(HOME); };
    function paint(){
      const v = state.view;
      if(last !== null && last !== v) hist.push(last);
      if(hist.length > 30) hist.shift();
      last = v;
      if(v === HOME){ bar.style.display = "none"; hist.length = 0; return; }
      bar.style.display = "flex";
      bar.innerHTML = `<button class="btn ghost btn-sm" onclick="embedBack()">&larr; Back</button>`
        + `<button class="btn ghost btn-sm" onclick="embedHome()">${TITLE} home</button>`;
    }
    const _render = render;
    render = function(){ _render(); paint(); };                         // every page change goes through render()
    setView(HOME);
  }catch(e){ console.warn("embed mode", e); }
})();
