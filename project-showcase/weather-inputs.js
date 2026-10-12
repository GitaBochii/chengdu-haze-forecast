function gridPosition(row,col,latitudes,longitudes){
  return {x:longitudes[0]<longitudes.at(-1)?col:longitudes.length-1-col,
    y:latitudes[0]>latitudes.at(-1)?row:latitudes.length-1-row};
}
function fieldRange(frames){
  const values=frames.flat();
  if(!values.length||!values.every(Number.isFinite))throw new Error('气象场数据不完整');
  const low=Math.min(...values),high=Math.max(...values);
  return low===high?[low-.5,high+.5]:[low,high];
}
if(typeof module!=='undefined')module.exports={gridPosition,fieldRange};
if(typeof document!=='undefined'){
  let archivePromise,selected=null,generation=0;
  const panel=document.createElement('section');
  panel.className='weather-inputs';panel.hidden=true;panel.id='weather-inputs';
  panel.innerHTML='<div class="weather-heading"><h3>模型输入气象场</h3><label>气象变量<select id="weather-variable" aria-label="气象变量"></select></label></div><p id="weather-context" class="help"></p><p id="weather-state" role="status"></p><div class="weather-maps"><figure><canvas width="660" height="560" role="img"></canvas><figcaption></figcaption></figure><figure><canvas width="660" height="560" role="img"></canvas><figcaption></figcaption></figure></div><div class="weather-scale"><span id="weather-low"></span><span class="weather-ramp" aria-hidden="true"></span><span id="weather-high"></span></div><p id="weather-source" class="help"></p>';
  document.querySelector('.trend-section').after(panel);
  const select=panel.querySelector('select'),state=panel.querySelector('#weather-state');
  const colors=[[40,90,145],[168,202,214],[245,244,232],[231,171,111],[170,64,51]];
  function color(fraction){
    const v=Math.max(0,Math.min(1,fraction))*4,index=Math.min(3,Math.floor(v)),mix=v-index;
    return 'rgb('+colors[index].map((a,i)=>Math.round(a+(colors[index+1][i]-a)*mix)).join(',')+')';
  }
  function draw(canvas,values,range,data,field,time){
    const ctx=canvas.getContext('2d'),left=68,top=32,width=550,height=460;
    const lat=data.latitudes,lon=data.longitudes,nx=lon.length,ny=lat.length;
    if(values.length!==nx*ny)throw new Error('气象场尺寸与经纬度不一致');
    ctx.clearRect(0,0,660,560);ctx.fillStyle='#ffffff';ctx.fillRect(0,0,660,560);
    values.forEach((value,index)=>{
      const p=gridPosition(Math.floor(index/nx),index%nx,lat,lon);
      ctx.fillStyle=color((value-range[0])/(range[1]-range[0]));
      ctx.fillRect(left+p.x*width/nx,top+p.y*height/ny,width/nx+.5,height/ny+.5);
    });
    const minLon=Math.min(...lon),maxLon=Math.max(...lon),minLat=Math.min(...lat),maxLat=Math.max(...lat);
    const dx=Math.abs(lon[1]-lon[0]),dy=Math.abs(lat[1]-lat[0]);
    const x=value=>left+(value-minLon+dx/2)/(maxLon-minLon+dx)*width;
    const y=value=>top+(maxLat+dy/2-value)/(maxLat-minLat+dy)*height;
    ctx.strokeStyle='#ffffff80';ctx.lineWidth=1;ctx.font='18px sans-serif';
    [0,.5,1].forEach(t=>{
      const lng=minLon+(maxLon-minLon)*t,lt=minLat+(maxLat-minLat)*t;
      ctx.beginPath();ctx.moveTo(x(lng),top);ctx.lineTo(x(lng),top+height);ctx.stroke();
      ctx.beginPath();ctx.moveTo(left,y(lt));ctx.lineTo(left+width,y(lt));ctx.stroke();
      ctx.fillStyle='#33423c';ctx.textAlign='center';ctx.fillText(lng.toFixed(1)+'°E',x(lng),top+height+30);
      ctx.textAlign='right';ctx.fillText(lt.toFixed(1)+'°N',left-8,y(lt)+6);
    });
    const cx=x(104.06),cy=y(30.67);
    ctx.beginPath();ctx.arc(cx,cy,6,0,2*Math.PI);ctx.fillStyle='#182b24';ctx.fill();ctx.strokeStyle='#fff';ctx.lineWidth=3;ctx.stroke();
    ctx.fillStyle='#fff';ctx.fillRect(cx+10,cy-21,57,29);ctx.fillStyle='#182b24';ctx.textAlign='left';ctx.fillText('成都',cx+15,cy);
    canvas.setAttribute('aria-label',`${field.label}，${formatTime(time)} 北京时间，成都周边真实输入网格，单位 ${field.unit}`);
  }
  function render(){
    if(!selected)return;
    const {data,sample}=selected,field=data.fields.find(f=>f.id===select.value);
    const frames=sample.fields[field.id],range=fieldRange(frames);
    panel.querySelectorAll('figure').forEach((figure,index)=>{
      draw(figure.querySelector('canvas'),frames[index],range,data,field,sample.times[index]);
      figure.querySelector('figcaption').textContent=formatTime(sample.times[index])+' · 北京时间';
    });
    panel.querySelector('#weather-low').textContent=range[0].toFixed(2)+' '+field.unit;
    panel.querySelector('#weather-high').textContent=range[1].toFixed(2)+' '+field.unit;
    panel.querySelector('#weather-source').textContent='来源：MERRA-2 再分析。两图采用相同色标；方格为原始网格，不是卫星照片。'+(field.note||'');
    state.textContent='';
    panel.querySelector('.weather-maps').hidden=false;panel.querySelector('.weather-scale').hidden=false;
  }
  select.addEventListener('change',()=>{try{render();}catch(error){state.textContent=error.message;}});
  window.hideWeatherInputs=()=>{generation++;selected=null;panel.hidden=true;};
  window.showWeatherInputs=async value=>{
    const request=++generation;panel.hidden=false;selected=null;state.textContent='正在加载实际输入气象场…';
    panel.querySelector('.weather-maps').hidden=true;panel.querySelector('.weather-scale').hidden=true;
    panel.querySelector('#weather-source').textContent='';
    panel.querySelector('#weather-context').textContent=value.arm==='spatial_fusion'?'本次融合预测使用的两个历史气象时刻。':'该样本配对的气象网格；当前对照模型未使用这组空间特征。';
    try{
      if(!archivePromise)archivePromise=fetch('weather-inputs.json').then(r=>{if(!r.ok)throw new Error('气象输入图暂不可用');return r.json();}).catch(e=>{archivePromise=null;throw e;});
      const data=await archivePromise;if(request!==generation)return;
      const sample=data.samples.find(s=>s.timestamp===value.timestamp);
      if(!sample||sample.times.length!==2)throw new Error('没有找到此样本的真实输入图');
      if(sample.times.some(t=>Date.parse(t)>Date.parse(value.timestamp)))throw new Error('气象时刻超出预测起点');
      const previous=select.value;select.replaceChildren(...data.fields.map(f=>new Option(f.label,f.id)));
      if(data.fields.some(f=>f.id===previous))select.value=previous;
      selected={data,sample};render();
    }catch(error){if(request===generation)state.textContent=error.message;}
  };
}
