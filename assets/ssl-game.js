
/* ssl-game.js · 阅读进度 + 偏好池
   localStorage 键：sslReadIds, sslVisitDays, sslLikedIds, sslDismissedIds。 */
(function(){
  var RKEY='sslReadIds', VKEY='sslVisitDays', LKEY='sslLikedIds', DKEY='sslDismissedIds';
  var SYNC_URL='http://127.0.0.1:8765/visual-ssl/preferences';
  var syncTimer=null, syncInFlight=false;
  function load(k){ try{ return normalize(JSON.parse(localStorage.getItem(k)||'[]')); }catch(e){ return []; } }
  function save(k,v){ try{ localStorage.setItem(k,JSON.stringify(normalize(v))); }catch(e){} }
  function normalize(v){
    var out=[], seen={};
    if(!Array.isArray(v)) return out;
    v.forEach(function(id){ id=String(id||'').trim(); if(id && !seen[id]){ seen[id]=1; out.push(id); } });
    return out;
  }
  function has(list,id){ return list.indexOf(id)!==-1; }
  function add(list,id){ if(!has(list,id)) list.push(id); }
  function remove(list,id){ var i=list.indexOf(id); if(i!==-1) list.splice(i,1); }
  function todayStr(d){ d=d||new Date(); return d.toISOString().slice(0,10); }

  var readSet=load(RKEY), visits=load(VKEY), likedSet=load(LKEY), dismissedSet=load(DKEY);
  (function visit(){ var t=todayStr(); if(visits.indexOf(t)===-1){ visits.push(t); save(VKEY,visits); } })();

  function streak(){
    var set={}; visits.forEach(function(d){ set[d]=1; });
    var n=0, d=new Date();
    while(set[todayStr(d)]){ n++; d.setDate(d.getDate()-1); }
    return n;
  }
  var listeners=[];
  function emit(){ listeners.forEach(function(fn){ try{ fn(); }catch(e){} }); }
  function snapshot(){
    return {
      schema:'visual-ssl-preferences/v1',
      exportedAt:new Date().toISOString(),
      source:location.href,
      readIds:normalize(readSet),
      likedIds:normalize(likedSet),
      dismissedIds:normalize(dismissedSet),
      visitDays:normalize(visits)
    };
  }
  function announce(){
    emit();
    try{ document.dispatchEvent(new CustomEvent('sslPreferenceChanged',{detail:snapshot()})); }catch(e){}
  }
  function setStatus(text){
    document.querySelectorAll('[data-pref-status]').forEach(function(el){ el.textContent=text||''; });
  }
  function applyPrefs(data){
    if(!data || typeof data!=='object') return false;
    var before=JSON.stringify(snapshot());
    normalize(data.readIds).forEach(function(id){ add(readSet,id); });
    normalize(data.likedIds).forEach(function(id){ add(likedSet,id); remove(dismissedSet,id); });
    normalize(data.dismissedIds).forEach(function(id){ if(!has(likedSet,id)) add(dismissedSet,id); });
    normalize(data.visitDays).forEach(function(day){ add(visits,day); });
    save(RKEY,readSet); save(LKEY,likedSet); save(DKEY,dismissedSet); save(VKEY,visits);
    return before!==JSON.stringify(snapshot());
  }
  function pullPrefs(){
    if(!window.fetch) return Promise.resolve(false);
    return fetch(SYNC_URL,{method:'GET',mode:'cors',cache:'no-store'}).then(function(resp){
      if(!resp.ok) throw new Error('HTTP '+resp.status);
      return resp.json();
    }).then(function(data){
      if(!data || !data.preferences) return false;
      var changed=applyPrefs(data.preferences);
      if(changed) announce();
      setStatus('已从本机同步偏好');
      return changed;
    }).catch(function(){
      setStatus('本机同步未连接；仍可手动导出');
      return false;
    });
  }
  function scheduleSync(reason){
    if(syncTimer) clearTimeout(syncTimer);
    syncTimer=setTimeout(function(){ syncPrefs(reason||'change'); }, 550);
  }
  function syncPrefs(reason){
    if(syncInFlight || !window.fetch) return;
    syncInFlight=true;
    var data=snapshot();
    data.reason=reason || 'change';
    fetch(SYNC_URL,{
      method:'POST',
      mode:'cors',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify(data)
    }).then(function(resp){
      syncInFlight=false;
      if(!resp.ok) throw new Error('HTTP '+resp.status);
      setStatus('已自动同步到本机');
    }).catch(function(){
      syncInFlight=false;
      setStatus('本机同步未连接；仍可手动导出');
    });
  }
  var Game={
    isRead:function(id){ return readSet.indexOf(id)!==-1; },
    isLiked:function(id){ return likedSet.indexOf(id)!==-1; },
    isDismissed:function(id){ return dismissedSet.indexOf(id)!==-1; },
    toggleRead:function(id){ if(has(readSet,id)) remove(readSet,id); else add(readSet,id); save(RKEY,readSet); announce(); return this.isRead(id); },
    togglePreference:function(id,kind){
      if(kind==='like'){
        if(has(likedSet,id)) remove(likedSet,id); else { add(likedSet,id); remove(dismissedSet,id); }
      } else if(kind==='dismiss'){
        if(has(dismissedSet,id)) remove(dismissedSet,id); else { add(dismissedSet,id); remove(likedSet,id); }
      }
      save(LKEY,likedSet); save(DKEY,dismissedSet); announce();
      return kind==='like' ? this.isLiked(id) : this.isDismissed(id);
    },
    readCount:function(){ return readSet.length; },
    likedCount:function(){ return likedSet.length; },
    dismissedCount:function(){ return dismissedSet.length; },
    snapshot:snapshot,
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
    document.querySelectorAll('[data-pref-liked-count]').forEach(function(el){ el.textContent=likedSet.length; });
    document.querySelectorAll('[data-pref-dismissed-count]').forEach(function(el){ el.textContent=dismissedSet.length; });
    document.querySelectorAll('[data-badges]').forEach(badges);
    document.querySelectorAll('[data-heat]').forEach(heat);
    document.querySelectorAll('[data-pid]').forEach(function(node){
      var pid=node.dataset.pid, r=Game.isRead(pid), liked=Game.isLiked(pid), dismissed=Game.isDismissed(pid);
      node.classList.toggle('is-read', r);
      node.classList.toggle('is-liked', liked);
      node.classList.toggle('is-dismissed', dismissed);
      var b=node.querySelector('.readbtn'); if(b) b.classList.toggle('on', r);
      node.querySelectorAll('[data-pref-toggle]').forEach(function(btn){
        var on=(btn.dataset.prefToggle==='like' && liked) || (btn.dataset.prefToggle==='dismiss' && dismissed);
        btn.classList.toggle('on', on);
        btn.setAttribute('aria-pressed', on ? 'true' : 'false');
      });
    });
    document.querySelectorAll('[data-dayids]').forEach(function(el){
      var ids=(el.dataset.dayids||'').split(',').filter(Boolean);
      var read=ids.filter(function(id){return Game.isRead(id);}).length;
      var badge=el.querySelector('[data-day-done]');
      if(badge){ if(read){ badge.hidden=false; badge.textContent='已读 '+read+'/'+ids.length; } else { badge.hidden=true; } }
    });
  }
  function downloadPrefs(){
    var data=snapshot();
    var blob=new Blob([JSON.stringify(data,null,2)],{type:'application/json'});
    var a=document.createElement('a');
    a.href=URL.createObjectURL(blob);
    a.download='visual_ssl_preferences_'+todayStr()+'.json';
    document.body.appendChild(a);
    a.click();
    setTimeout(function(){ URL.revokeObjectURL(a.href); a.remove(); },0);
    setStatus('已导出 '+data.likedIds.length+' 个关注、'+data.dismissedIds.length+' 个略过。');
  }
  function mergePrefs(data){
    if(!data || typeof data!=='object') throw new Error('JSON 格式不正确');
    applyPrefs(data);
    announce();
    scheduleSync('import');
    setStatus('已导入偏好：关注 '+likedSet.length+'，略过 '+dismissedSet.length+'。');
  }
  document.addEventListener('click', function(e){
    var pref=e.target.closest && e.target.closest('[data-pref-toggle]');
    if(pref){
      e.preventDefault(); e.stopPropagation();
      Game.togglePreference(pref.dataset.prefId, pref.dataset.prefToggle);
      return;
    }
    var exp=e.target.closest && e.target.closest('[data-pref-export]');
    if(exp){ e.preventDefault(); downloadPrefs(); return; }
    var btn=e.target.closest && e.target.closest('[data-read-toggle]');
    if(!btn) return;
    e.preventDefault(); e.stopPropagation();
    var now=Game.toggleRead(btn.dataset.readToggle);
    btn.classList.toggle('on', now);
    if(now){ btn.classList.remove('pop'); void btn.offsetWidth; btn.classList.add('pop'); }
  });
  document.addEventListener('change', function(e){
    var input=e.target.closest && e.target.closest('[data-pref-import]');
    if(!input || !input.files || !input.files[0]) return;
    var reader=new FileReader();
    reader.onload=function(){
      try{ mergePrefs(JSON.parse(String(reader.result||''))); }
      catch(err){ setStatus('导入失败：'+(err && err.message ? err.message : '无法解析 JSON')); }
      input.value='';
    };
    reader.readAsText(input.files[0], 'utf-8');
  });
  Game.onChange(renderAll);
  Game.onChange(function(){ scheduleSync('change'); });
  function boot(){
    renderAll();
    pullPrefs().then(function(){ renderAll(); scheduleSync('load'); });
  }
  if(document.readyState!=='loading') boot();
  else document.addEventListener('DOMContentLoaded', boot);
})();
