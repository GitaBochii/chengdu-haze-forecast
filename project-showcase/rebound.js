(function(){
  'use strict';
  const A=window.ReboundAnalysis,$=id=>document.getElementById(id);
  let data=null,current=null,hour=0,timer=null,active=false,request=0,loading=null;
  const format=value=>new Intl.DateTimeFormat('zh-CN',{timeZone:'Asia/Shanghai',year:'numeric',month:'2-digit',day:'2-digit',hour:'2-digit',minute:'2-digit',hour12:false}).format(new Date(value));
  const categories={rebound:'预测提示反弹，随后实测出现反弹',sustained_recovery:'预测提示持续恢复，随后实测达标',missed_rebound:'预测漏掉反弹的真实案例'};
  function stop(){if(timer!==null)clearInterval(timer);timer=null;if($('rebound-play'))$('rebound-play').textContent='播放';}
  function init(){
    if($('rebound-case'))return;
    $('rebound-controls').innerHTML='<h2>反弹与恢复</h2><p class="help">查看短暂转好、持续恢复，以及预测未能识别的反弹。</p><label>历史案例<select id="rebound-case" disabled><option>正在读取案例</option></select></label><p id="rebound-case-note" class="help"></p><button type="button" class="secondary" id="rebound-retry" hidden>重新读取案例</button><p class="help">案例来自冻结的历史测试预测，经过定向筛选，不代表整体准确率或实时预警。</p>';
    $('rebound-view').innerHTML='<div class="rb-heading"><div><span class="small-label">历史路径对照</span><h2 id="rebound-title"></h2><span id="rebound-category" class="rb-tag"></span></div><button type="button" class="secondary" id="rebound-download">下载当前回放 JSON</button></div><p class="rb-notice">本页仅以 AQI ≤ 100 连续 6 小时判定恢复，不包含原消散模型的颗粒物条件。预测提示由冻结预测曲线计算，事后实测不参与预测。</p><div class="rb-status"><div><h3>预测路径提示</h3><p id="rebound-prediction"></p></div><div><h3>实测进展</h3><p id="rebound-actual" aria-live="polite"></p></div></div><div class="rb-legend"><span style="--rb-color:#275f4b">过去观测</span><span style="--rb-color:#38699b">24 小时预测</span><span style="--rb-color:#ac6533">已揭示实测</span></div><div id="rebound-chart" role="img" aria-label="历史 AQI、未来预测与逐小时揭示的实测 AQI"></div><div class="rb-timeline"><label for="rebound-hour">实测回放</label><input id="rebound-hour" type="range" min="0" max="24" step="1" value="0"><output id="rebound-counter" for="rebound-hour">+0 小时</output></div><div class="rb-buttons"><button type="button" class="primary" id="rebound-play">播放</button><button type="button" class="secondary" id="rebound-reset">重置</button><span class="help">实测仅显示回放时刻及以前的观测。</span></div><details class="rb-details"><summary>查看来源与小时数据</summary><dl><dt>预测起点</dt><dd id="rebound-origin"></dd><dt>训练 / 开发 / 测试</dt><dd id="rebound-split"></dd><dt>结果来源</dt><dd id="rebound-source"></dd></dl><div class="table-scroll" tabindex="0" role="region" aria-label="AQI 小时预测与已揭示实测"><table><caption>AQI 无量纲；未揭示的实测留空。</caption><thead><tr><th>北京时间</th><th>小时</th><th>预测 AQI</th><th>已揭示实测</th></tr></thead><tbody id="rebound-rows"></tbody></table></div></details><p id="rebound-error" class="rb-error" role="status" hidden></p>';
    $('rebound-case').addEventListener('change',()=>choose($('rebound-case').value));
    $('rebound-retry').addEventListener('click',()=>{data=null;loading=null;window.openReboundView();});
    $('rebound-hour').addEventListener('input',()=>{stop();hour=Number($('rebound-hour').value);renderProgress();});
    $('rebound-reset').addEventListener('click',()=>{stop();hour=0;renderProgress();});
    $('rebound-play').addEventListener('click',()=>{
      if(timer!==null){stop();return;}if(!current||!active)return;
      if(hour>=24){hour=0;renderProgress();} $('rebound-play').textContent='暂停';
      timer=setInterval(()=>{if(!active){stop();return;}hour++;renderProgress();if(hour>=24)stop();},850);
    });
    $('rebound-download').addEventListener('click',()=>{
      if(!current)return;
      const visibleCase={...current,actual:A.prefix(current.actual,hour)};
      if(hour<24){delete visibleCase.category;delete visibleCase.title;}
      const value={version:1,notice:'历史案例回放；actual 仅包含已揭示观测，预测未使用未来实测。',replay_hour:hour,criterion:data.criterion,source_hashes:data.source_hashes,case_selection:data.case_selection,case:visibleCase};
      const url=URL.createObjectURL(new Blob([JSON.stringify(value,null,2)],{type:'application/json'})),a=document.createElement('a');a.href=url;a.download=`rebound-${current.id}-hour${hour}.json`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
    });
    new ResizeObserver(()=>{if(active&&current)draw();}).observe($('rebound-chart'));
    document.addEventListener('visibilitychange',()=>{if(document.hidden)stop();});
  }
  function describe(rows,predicted){
    if(rows.length===0)return '尚未揭示后续观测。';
    const info=A.analyze(rows),parts=[];
    if(info.rebounds.length){const b=info.rebounds[0];parts.push(`${predicted?'预计':'已观测到'}第 ${b.start} 小时起短暂转好 ${b.lowHours} 小时，第 ${b.at} 小时 AQI 回升至 ${b.aqi.toFixed(1)}。`);}
    if(info.recovery){parts.push(`${predicted?'预测路径':'实测'}从第 ${info.recovery.start} 小时开始连续达标 6 小时${predicted?'':`，在第 ${info.recovery.confirmedAt} 小时得到确认`}。`);}
    if(!predicted&&rows.length===24&&!info.recovery&&info.rebounds.length)parts.push('完整 24 小时实测未达到连续 6 小时恢复条件。');
    if(!info.rebounds.length&&!info.recovery){if(info.pendingLowHours)parts.push(`末段 AQI 已低于或等于 100，连续 ${info.pendingLowHours} 小时，尚不足 6 小时。`);else parts.push(predicted?'未来 24 小时预测路径未出现连续 6 小时达标。':`已揭示 ${rows.length} 小时，尚未观测到连续 6 小时达标。`);}
    return parts.join(' ');
  }
  function choose(id){
    stop();current=data.cases.find(c=>c.id===id);hour=0;
    if(!current)return;
    const predicted=A.analyze(current.forecast);
    $('rebound-title').textContent=predicted.rebounds.length?'预测路径出现短暂转好后反弹':predicted.recovery?'预测路径提示持续恢复':'预测路径尚未达到恢复条件';$('rebound-category').textContent='定向筛选的历史案例 · 不代表整体准确率';
    $('rebound-case-note').textContent=`${format(current.timestamp)} 起预测 · 北京时间`;
    $('rebound-prediction').textContent=describe(current.forecast,true);
    $('rebound-prediction').classList.toggle('rb-warning',A.analyze(current.forecast).rebounds.length>0);
    $('rebound-origin').textContent=format(current.timestamp)+'（北京时间）';
    $('rebound-split').textContent=`截至 ${current.train_end_year} 年 / ${current.dev_year} 年 / ${current.test_year} 年`;
    $('rebound-source').textContent=typeof data.model_scope==='string'?data.model_scope:'冻结历史曲线实验；不替代当前消散时间模型。';
    $('rebound-view').querySelectorAll('button,input').forEach(el=>el.disabled=false);renderProgress();
  }
  function renderProgress(){
    if(!current)return;
    $('rebound-hour').value=String(hour);$('rebound-counter').textContent=`+${hour} 小时`;
    const actual=A.prefix(current.actual,hour);$('rebound-actual').textContent=describe(actual,false);
    $('rebound-category').textContent=hour===24?(categories[current.category]||'完整历史案例'):'定向筛选的历史案例 · 不代表整体准确率';
    $('rebound-actual').classList.toggle('rb-warning',A.analyze(actual).rebounds.length>0);
    $('rebound-rows').replaceChildren(...current.forecast.map((r,i)=>{const tr=document.createElement('tr');[format(r.timestamp),`+${i+1}`,Number.isFinite(r.aqi)?r.aqi.toFixed(1):'缺失',i<hour?(Number.isFinite(current.actual[i].aqi)?current.actual[i].aqi.toFixed(1):'缺失'):'未揭示'].forEach(value=>{const td=document.createElement('td');td.textContent=value;tr.append(td);});return tr;}));draw();
  }
  function draw(){
    if(!active||!current)return;const host=$('rebound-chart'),w=host.clientWidth,h=host.clientHeight;if(w<1)return;
    const l=40,r=18,t=26,b=34,origin=Date.parse(current.timestamp),x=time=>l+(Date.parse(time)-origin+24*3600000)/(48*3600000)*(w-l-r);
    const actual=A.prefix(current.actual,hour),max=Math.ceil(Math.max(150,...current.history.map(p=>p.aqi||0),...current.forecast.map(p=>p.aqi||0),...actual.map(p=>p.aqi||0))/50)*50,y=value=>t+(1-value/max)*(h-t-b);
    let grid='';for(let v=0;v<=max;v+=max>300?100:50)grid+=`<line x1="${l}" x2="${w-r}" y1="${y(v)}" y2="${y(v)}" stroke="#e1e7e3"/><text x="${l-7}" y="${y(v)+4}" text-anchor="end">${v}</text>`;
    function curve(rows,color){let previous=null,lines='',dots='';rows.forEach(p=>{if(!Number.isFinite(p.aqi)){previous=null;return;}const time=Date.parse(p.timestamp),px=x(p.timestamp),py=y(p.aqi);if(previous&&time-previous.time===3600000)lines+=`<line x1="${previous.x}" y1="${previous.y}" x2="${px}" y2="${py}"/>`;dots+=`<circle cx="${px}" cy="${py}" r="2"><title>${format(p.timestamp)} AQI ${p.aqi.toFixed(1)}</title></circle>`;previous={time,x:px,y:py};});return `<g stroke="${color}" stroke-width="2">${lines}</g><g fill="${color}">${dots}</g>`;}
    const divider=x(current.timestamp);
    const xHour=k=>divider+k/48*(w-l-r),predicted=A.analyze(current.forecast),confirmed=A.analyze(actual).recovery;
    let bands='',markers='';
    predicted.rebounds.forEach((event,i)=>{
      const left=xHour(event.start-.5),right=xHour(event.at-.5),mark=xHour(event.at);
      bands+=`<rect data-rb-marker="predicted-low-window" x="${left}" y="${t}" width="${right-left}" height="${h-t-b}" fill="#d79859" fill-opacity=".13"><title>预计短暂转好 ${event.lowHours} 小时</title></rect>`;
      markers+=`<line data-rb-marker="predicted-rebound" x1="${mark}" x2="${mark}" y1="${t}" y2="${h-b}" stroke="#ac6533" stroke-dasharray="3 4" stroke-opacity=".65"/>`;
      if(i===0)markers+=`<text data-rb-marker="predicted-rebound-label" x="${Math.min(w-r-3,Math.max(divider+102,mark))}" y="${t+14}" text-anchor="end" fill="#935720">预计反弹 +${event.at}h</text>`;
    });
    if(confirmed){const left=xHour(confirmed.start-.5),right=xHour(confirmed.confirmedAt+.5);bands+=`<rect data-rb-marker="actual-confirmed-window" x="${left}" y="${t}" width="${right-left}" height="${h-t-b}" fill="#539876" fill-opacity=".14"><title>实测第 ${confirmed.start} 至 ${confirmed.confirmedAt} 小时连续达标</title></rect>`;markers+=`<text data-rb-marker="actual-confirmed-label" x="${Math.min(w-r-3,Math.max(divider+112,right))}" y="${h-b-9}" text-anchor="end" fill="#275f4b">实测已连续达标 6h</text>`;}
    host.innerHTML=`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${w} ${h}" aria-hidden="true"><g font-family="Arial,sans-serif" font-size="11" fill="#55685d"><rect x="${divider}" y="${t}" width="${w-r-divider}" height="${h-t-b}" fill="#f3f6f8"/>${bands}${grid}<line x1="${l}" x2="${w-r}" y1="${y(100)}" y2="${y(100)}" stroke="#83978a" stroke-dasharray="5 4"/><text x="${w-r}" y="${y(100)-6}" text-anchor="end">AQI 100</text><line x1="${divider}" x2="${divider}" y1="${t}" y2="${h-b}" stroke="#a5b2aa" stroke-dasharray="3 3"/><text x="${divider}" y="15" text-anchor="middle">预测起点</text>${markers}${curve(current.history,'#275f4b')}${curve(current.forecast,'#38699b')}${curve(actual,'#ac6533')}<text x="${l}" y="${h-8}">过去 24 小时</text><text x="${divider}" y="${h-8}" text-anchor="middle">0</text><text x="${w-r}" y="${h-8}" text-anchor="end">未来 24 小时</text><text x="${l}" y="15">AQI</text></g></svg>`;
  }
  function fail(error){
    stop();current=null;$('rebound-case').disabled=true;$('rebound-case').replaceChildren(new Option('案例未能读取',''));$('rebound-retry').hidden=false;
    $('rebound-title').textContent='暂时无法显示案例';$('rebound-category').textContent='';$('rebound-case-note').textContent='';$('rebound-prediction').textContent='';$('rebound-actual').textContent='';$('rebound-origin').textContent='';$('rebound-split').textContent='';$('rebound-source').textContent='';$('rebound-chart').replaceChildren();$('rebound-rows').replaceChildren();$('rebound-error').textContent=error.message||'读取失败，请重新尝试。';$('rebound-error').hidden=false;$('rebound-view').querySelectorAll('button,input').forEach(el=>el.disabled=true);
  }
  async function load(){
    const controller=new AbortController(),timeout=setTimeout(()=>controller.abort(),20000);
    try{const response=await fetch('rebound-cases.json',{cache:'no-cache',signal:controller.signal});if(!response.ok)throw new Error('案例文件尚未就绪，请重新读取。');return A.validate(await response.json());}
    catch(error){if(error.name==='AbortError')throw new Error('读取案例超时，请重新读取。');throw error;}
    finally{clearTimeout(timeout);}
  }
  window.openReboundView=async function(){
    active=true;init();$('rebound-view').hidden=false;$('rebound-error').hidden=true;
    if(data){draw();return;}const token=++request;
    try{
      if(!loading)loading=load();
      const payload=await loading;if(token!==request||!active)return;data=payload;
      $('rebound-case').replaceChildren(...data.cases.map((c,i)=>new Option(`案例 ${i+1} · ${format(c.timestamp)}`,c.id)));$('rebound-case').disabled=false;$('rebound-retry').hidden=true;choose(data.cases[0].id);
    }catch(error){loading=null;if(token===request&&active)fail(error);}
  };
  window.hideReboundView=function(){active=false;request++;stop();if($('rebound-view'))$('rebound-view').hidden=true;};
})();
