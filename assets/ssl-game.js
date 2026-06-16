
/* ssl-game.js · 养成系：连续追踪 streak / 进度环 / 标记已读 / 里程碑 / 签到热力图
   仅使用自己的 localStorage 键：sslReadIds, sslVisitDays。 */
(function(){
  var RKEY='sslReadIds', VKEY='sslVisitDays';
  function load(k){ try{ return JSON.parse(localStorage.getItem(k)||'[]'); }catch(e){ return []; } }
  function save(k,v){ try{ localStorage.setItem(k,JSON.stringify(v)); }catch(e){} }
  function todayStr(d){ d=d||new Date(); return d.toISOString().slice(0,10); }

  var readSet=load(RKEY), visits=load(VKEY);
  (function visit(){ var t=todayStr(); if(visits.indexOf(t)===-1){ visits.push(t); save(VKEY,visits); } })();

  function streak(){
    var set={}; visits.forEach(function(d){ set[d]=1; });
    var n=0, d=new Date();
    while(set[todayStr(d)]){ n++; d.setDate(d.getDate()-1); }
    return n;
  }
  var listeners=[];
  function emit(){ listeners.forEach(function(fn){ try{ fn(); }catch(e){} }); }
  var Game={
    isRead:function(id){ return readSet.indexOf(id)!==-1; },
    toggleRead:function(id){ var i=readSet.indexOf(id); if(i===-1) readSet.push(id); else readSet.splice(i,1); save(RKEY,readSet); emit(); return this.isRead(id); },
    readCount:function(){ return readSet.length; },
    streak:streak,
    onChange:function(fn){ listeners.push(fn); }
  };
  window.SSLGame=Game;

  function ring(el){
    var total=+el.dataset.total||10, done=0;
    if(el.dataset.ids){ var ids=el.dataset.ids.split(',').filter(Boolean); done=ids.filter(function(id){return readSet.indexOf(id)!==-1;}).length; total=ids.length||1; }
    else { done=Math.min(readSet.length,total); }
    var pct=total?done/total:0, r=26, C=2*Math.PI*r;
    el.innerHTML='<svg viewBox="0 0 64 64" class="ringsvg"><circle cx="32" cy="32" r="'+r+'" class="ring-bg"></circle><circle cx="32" cy="32" r="'+r+'" class="ring-fg" stroke-dasharray="'+C+'" stroke-dashoffset="'+(C*(1-pct))+'"></circle></svg><div class="ring-num"><b>'+done+'</b><span>/'+total+'</span></div>';
  }
  var MILE=[10,50,100,250,500];
  function badges(el){ var c=readSet.length; el.innerHTML=MILE.map(function(m){ return '<span class="badge'+(c>=m?' on':'')+'" title="累计精读 '+m+' 篇">'+m+'</span>'; }).join(''); }
  function heat(el){
    var set={}; visits.forEach(function(d){ set[d]=1; });
    var weeks=+el.dataset.weeks||16, today=new Date();
    var start=new Date(today); start.setDate(start.getDate()-(weeks*7-1));
    var dow=(start.getDay()+6)%7; start.setDate(start.getDate()-dow);
    var html='<div class="heat-grid">';
    for(var w=0;w<weeks;w++){ html+='<div class="heat-col">';
      for(var dd=0;dd<7;dd++){ var d=new Date(start); d.setDate(d.getDate()+w*7+dd); var fut=d>today; html+='<i class="heat-cell'+(set[todayStr(d)]?' on':'')+(fut?' fut':'')+'" title="'+todayStr(d)+'"></i>'; }
      html+='</div>'; }
    html+='</div>'; el.innerHTML=html;
  }
  function renderAll(){
    document.querySelectorAll('[data-ring]').forEach(ring);
    document.querySelectorAll('[data-streak]').forEach(function(el){ el.textContent=streak(); });
    document.querySelectorAll('[data-readcount]').forEach(function(el){ el.textContent=readSet.length; });
    document.querySelectorAll('[data-badges]').forEach(badges);
    document.querySelectorAll('[data-heat]').forEach(heat);
    document.querySelectorAll('[data-pid]').forEach(function(node){
      var r=Game.isRead(node.dataset.pid);
      node.classList.toggle('is-read', r);
      var b=node.querySelector('.readbtn'); if(b) b.classList.toggle('on', r);
    });
    document.querySelectorAll('[data-dayids]').forEach(function(el){
      var ids=(el.dataset.dayids||'').split(',').filter(Boolean);
      var read=ids.filter(function(id){return Game.isRead(id);}).length;
      var badge=el.querySelector('[data-day-done]');
      if(badge){ if(read){ badge.hidden=false; badge.textContent='已读 '+read+'/'+ids.length; } else { badge.hidden=true; } }
    });
  }
  document.addEventListener('click', function(e){
    var btn=e.target.closest && e.target.closest('[data-read-toggle]');
    if(!btn) return;
    e.preventDefault(); e.stopPropagation();
    var now=Game.toggleRead(btn.dataset.readToggle);
    btn.classList.toggle('on', now);
    if(now){ btn.classList.remove('pop'); void btn.offsetWidth; btn.classList.add('pop'); }
  });
  Game.onChange(renderAll);
  if(document.readyState!=='loading') renderAll();
  else document.addEventListener('DOMContentLoaded', renderAll);
})();
