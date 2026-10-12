// Actual hourly AQI observations, 2022-11-26 00:00 through 2022-11-28 10:00 (UTC+8).
const aqi = [139,134,128,127,114,115,115,108,114,112,114,115,102.5,92,90,82,87.5,99.5,108.5,107,127,109.5,108,107,102,97,90,100,102,97,87,88,100,100,102,92,92,97.5,91,94,112,107.5,113.5,112.5,125,138.5,129,123,109.5,95,107,105,104,94,97,95,89,97,99];
const start = Date.UTC(2022,10,26);
function timestamp(index){return new Date(start + index * 3600000).toISOString().slice(0,16).replace('T',' ');}
document.getElementById('observations').innerHTML = aqi.map((value,index)=>`<tr><td>${timestamp(index)}</td><td>${value}</td></tr>`).join('');
function drawChart(){
  const host = document.getElementById('chart');
  const w = host.clientWidth, h = host.clientHeight;
  const left = 34, right = 14, top = 15, bottom = 36;
  const x = i => left + i / (aqi.length - 1) * (w-left-right);
  const y = v => top + (150-v)/80*(h-top-bottom);
  const grid = [80,100,120,140].map(v=>`<line x1="${left}" x2="${w-right}" y1="${y(v)}" y2="${y(v)}" stroke="${v===100?'#71897b':'#e2e6e2'}" ${v===100?'stroke-dasharray="4 4"':''}/><text x="${left-8}" y="${y(v)+4}" text-anchor="end">${v}</text>`).join('');
  const ticks = [[0,'11/26 00:00'],[24,'11/27 00:00'],[48,'11/28 00:00']].map(([i,label])=>`<text x="${x(i)}" y="${h-7}" text-anchor="${i===0?'start':'middle'}">${label}</text>`).join('');
  host.innerHTML = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${w} ${h}" aria-hidden="true"><g font-family="Arial,sans-serif" font-size="11" fill="#626b67"><rect x="${x(53)}" y="${top}" width="${x(58)-x(53)}" height="${h-top-bottom}" fill="#e1ede3"/>${grid}<polyline points="${aqi.map((v,i)=>`${x(i)},${y(v)}`).join(' ')}" fill="none" stroke="#275f4b" stroke-width="2" stroke-linejoin="round"/>${ticks}</g></svg>`;
  host.setAttribute('aria-label','2022 年 11 月 26 日至 28 日 AQI 实际观测，绿色区域为稳定恢复窗口。');
  const rows = aqi.map((value, i) => ({timestamp: timestamp(i).replace(' ', 'T') + ':00+08:00', aqi: value}));
  window.installTrendCursor?.(host, rows, {
    left, right, top, bottom, width: w, height: h,
    x: time => x((time - (start - 8 * 3600000)) / 3600000), y,
    selected: new Set(['aqi']), series: [['aqi', '#275f4b', 'AQI']],
    formatTime: value => value.slice(0,16).replace('T',' '),
  });
}
new ResizeObserver(drawChart).observe(document.getElementById('chart'));
drawChart();
