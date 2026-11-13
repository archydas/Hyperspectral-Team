/* Compatibility for older embedded browsers: the Qt WebEngine that ships for Windows (PyQtWebEngine 5.15.2) is
   Chromium 83 and lacks String.replaceAll, FinalizationRegistry, Array.at, Object.hasOwn, WeakRef.  Loaded first. */
(function(){
  if(!String.prototype.replaceAll){
    Object.defineProperty(String.prototype,"replaceAll",{configurable:true,writable:true,value:function(a,b){
      if(a instanceof RegExp){ if(!a.global) throw new TypeError("replaceAll must be called with a global RegExp"); return this.replace(a,b); }
      var s=String(this), p=String(a), out="", i=0, j;
      if(p===""){ return s.replace(/(?:)/g,typeof b==="function"?b:String(b)); }
      while((j=s.indexOf(p,i))!==-1){ out+=s.slice(i,j)+(typeof b==="function"?b(p,j,s):String(b)); i=j+p.length; }
      return out+s.slice(i);
    }});
  }
  if(typeof window.FinalizationRegistry==="undefined"){
    window.FinalizationRegistry=function(cb){ this.register=function(){}; this.unregister=function(){return false;}; };
  }
  if(typeof window.WeakRef==="undefined"){
    window.WeakRef=function(o){ this._o=o; }; window.WeakRef.prototype.deref=function(){ return this._o; };
  }
  if(!Array.prototype.at){
    Object.defineProperty(Array.prototype,"at",{configurable:true,writable:true,value:function(n){ n=Math.trunc(n)||0; if(n<0) n+=this.length; return this[n]; }});
  }
  if(!Object.hasOwn){ Object.hasOwn=function(o,k){ return Object.prototype.hasOwnProperty.call(o,k); }; }
})();
