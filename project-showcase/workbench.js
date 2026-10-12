const $ = id => document.getElementById(id);
let mode = 'replay', schema, result, busy = false, dateRequest = 0, disabledStates = [], availableDates = [], onlineSnapshot = null;
const tabs = [...document.querySelectorAll('[role=tab]')];
function message(text, error=false){$('feedback').textContent=text;$('feedback').classList.toggle('error',error);$('feedback').hidden=!text;}
function clearResult(){result=null;$('result').hidden=true;$('empty-result').hidden=false;$('online-preview').hidden=true;window.hideWeatherInputs?.();window.hideReboundView?.();}
function setBusy(value){busy=value;document.querySelector('.result-pane').setAttribute('aria-busy',String(value));if(value){disabledStates=[...document.querySelectorAll('.input-pane input,.input-pane select,.input-pane button,.mode-tabs button')].map(el=>[el,el.disabled]);disabledStates.forEach(([el])=>el.disabled=true);}else if(disabledStates.length){disabledStates.forEach(([el,disabled])=>el.disabled=disabled);disabledStates=[];}else{document.querySelectorAll('button[type=submit]').forEach(b=>b.disabled=!schema);}setOnlineControls();window.setMultimodalControls?.();}
function setOnlineControls(){$('predict-online').disabled=busy||!onlineSnapshot?.can_predict||!$('online-consent').checked;}
async function api(path, data){
  const controller=new AbortController();
  const timeout=setTimeout(()=>controller.abort(),30000);
  try{
    const response=await fetch(path,{signal:controller.signal,...(data?{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)}:{})});
    if(!response.headers.get('content-type')?.includes('application/json'))throw new Error('服务返回了无法读取的结果，请重新连接。');
    const payload=await response.json();
    if(!response.ok)throw new Error(payload.error||'请求未完成，请稍后重试。');
    return payload;
  }catch(error){
    if(error.name==='AbortError')throw new Error('等待服务超过 30 秒，请稍后重试。');
    if(error instanceof TypeError)throw new Error('连接中断，请确认本地网站服务正在运行。');
    throw error;
  }finally{clearTimeout(timeout);}
}
function formatTime(value){if(!value)return '未提供';const date=new Date(value);return Number.isNaN(date.getTime())?String(value):new Intl.DateTimeFormat('sv-SE',{timeZone:'Asia/Shanghai',year:'numeric',month:'2-digit',day:'2-digit',hour:'2-digit',minute:'2-digit'}).format(date);}
function metric(id,value){$(id).textContent=Number.isFinite(value)?`${value.toFixed(1)} 小时`:'—';}
function showResult(value){
  result=value;$('result').hidden=false;$('empty-result').hidden=true;message('');
  $('result-kind').textContent=mode==='replay'?'严格历史回测':mode==='online'?'在线估算实验 · 非实测':'新输入预测';
  $('result-time').textContent=formatTime(value.timestamp)+' · 北京时间';
  metric('prediction',value.prediction_hours);metric('actual',value.actual_hours);metric('error',value.error_hours);
  $('actual-label').textContent=mode==='replay'?'历史真实剩余时间':'真实剩余时间（未知）';
  $('error-label').textContent=mode==='replay'?'本条绝对误差':'预测误差（待观测）';
  $('result-notice').textContent=value.notice||'';
  if(!value.eligible&&!value.notice)$('result-notice').textContent='当前不满足模型适用的颗粒物污染条件，不给出消散倒计时。';
  $('model-name').textContent=value.model_label;
  $('recovery-label').textContent='预计恢复时刻';
  $('recovery-criterion').textContent='AQI ≤ 100，且不再满足颗粒物污染条件，连续保持 6 小时。';
  document.querySelector('.trend-section h3').textContent=mode==='online'?'在线估算趋势':'输入观测趋势';
  document.querySelector('.trend-section > .help').textContent=mode==='online'?'以下为网格模型估算；AQI按训练时期口径推算，不是监测站实测或官方发布值。':'仅展示预测时刻及以前的观测，不是未来浓度预测。';
  $('recovery-time').textContent=value.eligible&&Number.isFinite(value.prediction_hours)?formatTime(new Date(new Date(value.timestamp).getTime()+value.prediction_hours*3600000).toISOString())+'（北京时间）':'不适用';
  const rows=Array.isArray(value.trend)?value.trend:[];
  $('trend-rows').replaceChildren(...rows.map(row=>{const tr=document.createElement('tr');[formatTime(row.timestamp),row.aqi,row.pm2_5,row.pm10].forEach(v=>{const td=document.createElement('td');td.textContent=v??'缺失';tr.append(td);});return tr;}));
  $('trend-note').textContent=rows.length<2?'仅有单时刻数据，以数据点展示；不足以形成趋势。':'左轴：AQI；右轴：PM2.5、PM10（µg/m³）。两轴采用相同数值刻度，单位不同；缺失数据不连线。';
  document.querySelector('#result details summary').textContent='查看输入序列明细';
  document.querySelector('#result details .table-scroll').setAttribute('aria-label','污染物输入序列明细');
  if(mode==='multimodal')window.renderMultimodalResult?.(value);
  drawTrend();
}
function drawTrend(){
  if(!result||$('result').hidden)return;
  const rows=(result.trend||[]).filter(r=>Number.isFinite(new Date(r.timestamp).getTime()));
  const host=$('trend'),w=host.clientWidth,h=host.clientHeight;
  host.onpointerdown=host.onpointermove=host.onpointerleave=host.onfocus=host.onblur=host.onkeydown=null;
  host.removeAttribute('tabindex');
  document.getElementById('trend-inspector')?.replaceChildren();
  if(!w||!rows.length){host.textContent='暂无可用趋势数据';return;}
  const left=38,right=38,top=30,bottom=32;
  const times=rows.map(r=>new Date(r.timestamp).getTime());
  const low=Math.min(...times),high=Math.max(...times);
  const x=t=>low===high?(w+left-right)/2:left+(t-low)/(high-low)*(w-left-right);
  const series=[['aqi','#275f4b','AQI'],['pm2_5','#38699b','PM2.5'],['pm10','#ac6533','PM10']];
  const max=Math.ceil(Math.max(150,...rows.flatMap(r=>series.map(([key])=>Number.isFinite(r[key])?r[key]:0)))/50)*50;
  const y=v=>top+(1-v/max)*(h-top-bottom);
  const step=max<=200?50:max<=500?100:Math.ceil(max/200)*50;
  const ticks=Array.from({length:Math.floor(max/step)+1},(_,i)=>i*step);
  const grid=ticks.map(v=>`<line x1="${left}" x2="${w-right}" y1="${y(v)}" y2="${y(v)}" stroke="#e0e5e1"/><text x="${left-7}" y="${y(v)+4}" text-anchor="end">${v}</text><text x="${w-right+7}" y="${y(v)+4}">${v}</text>`).join('');
  const selected=new Set([...document.querySelectorAll('[data-curve]:checked')].map(el=>el.dataset.curve));
  if(!selected.size){host.textContent='尚未选择曲线';return;}
  const threshold=selected.has('aqi')?`<line x1="${left}" x2="${w-right}" y1="${y(100)}" y2="${y(100)}" stroke="#799386" stroke-dasharray="4 4"/><text x="${w-right-4}" y="${y(100)-6}" text-anchor="end">AQI 100</text>`:'';
  const curves=series.filter(([key])=>selected.has(key)).map(([key,color,label])=>{
      let lines='',dots='',previous=null;
      rows.forEach(r=>{const t=new Date(r.timestamp).getTime();if(!Number.isFinite(r[key])){previous=null;return;}const p={t,x:x(t),y:y(r[key])};if(previous&&t-previous.t<=3600000)lines+=`<line x1="${previous.x}" y1="${previous.y}" x2="${p.x}" y2="${p.y}"/>`;dots+=`<circle cx="${p.x}" cy="${p.y}" r="${rows.length===1?4:2}"><title>${label}: ${r[key]}</title></circle>`;previous=p;});
      return `<g data-series="${key}"><g stroke="${color}" stroke-width="1.8">${lines}</g><g fill="${color}">${dots}</g></g>`;
    }).join('');
  const plots=`<text x="${left}" y="13">AQI</text><text x="${w-right}" y="13" text-anchor="end">PM2.5 / PM10（µg/m³）</text>${grid}${threshold}${curves}`;
  const first=formatTime(rows[0].timestamp).slice(5),last=formatTime(rows[rows.length-1].timestamp).slice(5);
  host.setAttribute('aria-label','AQI、PM2.5 与 PM10 同图输入序列趋势，左轴 AQI，右轴颗粒物浓度');
  host.innerHTML=`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${w} ${h}" aria-hidden="true"><g font-size="11" font-family="Arial,sans-serif" fill="#626b67">${plots}<text x="${left}" y="${h-8}">${first}</text>${rows.length>1?`<text x="${w-right}" y="${h-8}" text-anchor="end">${last}</text>`:''}</g></svg>`;
  window.installTrendCursor?.(host,rows,{left,right,top,bottom,width:w,height:h,x,y,selected,series,formatTime});
}
new ResizeObserver(drawTrend).observe($('trend'));
document.querySelectorAll('[data-curve]').forEach(el=>el.addEventListener('change',drawTrend));
tabs.forEach((tab,index)=>{tab.addEventListener('click',()=>{if(busy)return;mode=tab.dataset.mode;tabs.forEach(t=>{const active=t===tab;t.setAttribute('aria-selected',String(active));t.tabIndex=active?0:-1;$(t.getAttribute('aria-controls')).hidden=!active;});clearResult();if(mode==='rebound'){message('');$('empty-result').hidden=true;window.openReboundView?.();return;}message(schema?'选择记录或提交观测后显示结果。':'本地服务尚未连接，请通过启动脚本打开网站。',!schema);});tab.addEventListener('keydown',event=>{if(!['ArrowLeft','ArrowRight','Home','End'].includes(event.key))return;event.preventDefault();const next=event.key==='Home'?0:event.key==='End'?tabs.length-1:(index+(event.key==='ArrowRight'?1:-1)+tabs.length)%tabs.length;tabs[next].click();tabs[next].focus();});});
tabs.forEach(tab=>tab.addEventListener('click',()=>{if(busy)return;onlineSnapshot=null;$('online-consent').checked=false;setOnlineControls();if(mode==='online'){$('empty-result').hidden=true;message('点击获取最近 24 小时，先检查数据再运行实验预测。');}}));
$('online-consent').addEventListener('change',setOnlineControls);
$('import-online').addEventListener('click',async()=>{
  if(busy)return;clearResult();onlineSnapshot=null;$('online-consent').checked=false;$('empty-result').hidden=true;setBusy(true);message('正在获取空气质量与气象估算，请稍候…');
  try{
    onlineSnapshot=await api('/api/online/import',{});$('online-preview').hidden=false;
    $('online-range').textContent=formatTime(onlineSnapshot.start_at)+' 至 '+formatTime(onlineSnapshot.end_at)+'（北京时间）';
    $('online-fetched').textContent='本机获取时间：'+formatTime(onlineSnapshot.fetched_at)+'；共 '+onlineSnapshot.rows.length+' 个小时。';
    $('online-notice').textContent=onlineSnapshot.notice;
    $('online-missing').textContent=onlineSnapshot.missing_fields.length?'缺失字段：'+onlineSnapshot.missing_fields.map(name=>schema?.fields.find(f=>f.name===name)?.label||name).join('、')+'。未使用虚构数值补齐。':'所需字段完整。';
    $('online-rows').replaceChildren(...onlineSnapshot.rows.map(row=>{const tr=document.createElement('tr');[formatTime(row.timestamp),row.aqi,row.pm2_5,row.pm10,row.temperature_c].forEach(value=>{const td=document.createElement('td');td.textContent=typeof value==='number'?value.toFixed(1):value??'缺失';tr.append(td);});return tr;}));
    message(onlineSnapshot.can_predict?'序列已获取。确认估算数据的适用范围后，可运行实验预测。':'数据不完整，本次不运行预测。');
  }catch(error){message(error.message,true);}finally{setBusy(false);}
});
$('online-form').addEventListener('submit',event=>{event.preventDefault();if(!onlineSnapshot?.can_predict||!$('online-consent').checked)return;const token=onlineSnapshot.token;run(()=>api('/api/predict',{mode:'online',token,acknowledge_estimates:true}));});
async function loadTimes(){
  const request=++dateRequest;$('history-time').disabled=true;$('replay-form').querySelector('button').disabled=true;if(mode!=='rebound')clearResult();
  try{const {records}=await api('/api/history?date='+encodeURIComponent($('history-date').value));if(request!==dateRequest)return;$('history-time').replaceChildren(...records.map(r=>new Option(formatTime(r.timestamp),r.id)));$('history-time').disabled=!records.length;$('replay-form').querySelector('button').disabled=!records.length;if(mode==='replay')message(records.length?'选择污染时刻，查看对应回测结果。':'当天没有可用回测记录。');}catch(error){if(request===dateRequest&&mode==='replay')message(error.message,true);}
}
function filterDates(preferred){
  const dates=availableDates.filter(date=>date.startsWith($('history-year').value));
  $('history-date').replaceChildren(...dates.map(date=>new Option(date,date)));
  if(preferred&&dates.includes(preferred))$('history-date').value=preferred;
  $('history-date').disabled=!dates.length;
  return loadTimes();
}
$('history-year').addEventListener('change',()=>filterDates());
$('history-date').addEventListener('change',loadTimes);
$('history-time').addEventListener('change',()=>{clearResult();message('时刻已更改，请查看新的回测记录。');});
async function run(action){if(busy)return;clearResult();$('empty-result').hidden=true;setBusy(true);message('正在计算，请稍候…');try{showResult(await action());}catch(error){message(error.message,true);}finally{setBusy(false);}}
$('replay-form').addEventListener('submit',event=>{event.preventDefault();run(()=>api('/api/replay?timestamp='+encodeURIComponent($('history-time').value)));});
$('manual-form').addEventListener('submit',event=>{event.preventDefault();const values=Object.fromEntries(schema.fields.map(f=>[f.name,Number($('field-'+f.name).value)]));run(()=>api('/api/predict',{mode:'manual',timestamp:$('manual-time').value+':00+08:00',values}));});
$('sequence-form').addEventListener('submit',event=>{event.preventDefault();run(async()=>{const file=$('csv-file').files[0];if(!file)throw new Error('请选择 CSV 文件。');if(file.size>900*1024)throw new Error('CSV 文件不能超过 900 KB。');return api('/api/predict',{mode:'sequence',csv:await file.text()});});});
['manual-form','sequence-form'].forEach(id=>$(id).addEventListener('input',()=>{clearResult();message('输入已更改，提交后显示新的预测结果。');}));
function download(content,name,type){const url=URL.createObjectURL(new Blob([content],{type}));const link=document.createElement('a');link.href=url;link.download=name;link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
$('template').addEventListener('click',()=>download('\ufeff'+schema.columns.join(',')+'\r\n','chengdu-observations-template.csv','text/csv;charset=utf-8'));
$('download-result').addEventListener('click',()=>{if(result)download(JSON.stringify(result,null,2),'chengdu-recovery-result.json','application/json');});
async function init(){
  if(location.protocol==='file:'){message('预测需要本地服务。请运行项目根目录的“启动项目网站.cmd”，然后使用它显示的网址。',true);$('connection').textContent='尚未启动本地服务';return;}
  try{
    $('retry-connection').hidden=true;message('正在加载模型与历史记录…');
    await api('/api/status');schema=await api('/api/schema');$('connection').textContent='本地服务已连接';$('connection').dataset.ready='true';
    $('manual-fields').replaceChildren(...schema.fields.map(f=>{const label=document.createElement('label');label.textContent=f.label;const input=document.createElement('input');input.type='number';input.id='field-'+f.name;input.required=true;input.value=f.default;input.step=f.step??'any';if(f.min!=null)input.min=f.min;if(f.max!=null)input.max=f.max;label.append(input);return label;}));
    const now=new Date();$('manual-time').value=new Date(now.getTime()+8*3600000).toISOString().slice(0,13)+':00';
    $('csv-columns').textContent=schema.columns.join(', ');$('template').disabled=false;
    const {dates}=await api('/api/history/dates');availableDates=dates;
    const years=[...new Set(dates.map(d=>d.slice(0,4)))];$('history-year').replaceChildren(...years.map(year=>new Option(year+' 年',year)));$('history-year').disabled=!years.length;
    const preferred=new URLSearchParams(location.search).get('date');if(dates.includes(preferred))$('history-year').value=preferred.slice(0,4);
    if(dates.length)await filterDates(preferred);else message('暂无可用历史回测，可切换至输入预测。');
    setBusy(false);$('replay-form').querySelector('button').disabled=!$('history-time').value;
  }catch(error){$('connection').textContent='服务或数据暂不可用';$('connection').dataset.ready='false';message(error.message||'无法连接本地服务，请检查启动窗口。',true);$('retry-connection').hidden=false;}
  finally{$('tab-multimodal').disabled=false;if(mode!=='rebound'&&new URLSearchParams(location.search).get('view')==='rebound')$('tab-rebound').click();}
}
$('retry-connection').addEventListener('click',init);
$('tab-multimodal').disabled=true;
init();
document.addEventListener('DOMContentLoaded',()=>{if(new URLSearchParams(location.search).get('view')==='rebound')$('tab-rebound').click();},{once:true});
