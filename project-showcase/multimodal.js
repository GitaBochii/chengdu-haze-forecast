function researchPresentation(value){
  const labels={train:'训练样本',dev:'开发样本',test:'最终审计样本'};
  const hours=number=>Number(number).toLocaleString('zh-CN',{maximumFractionDigits:1});
  const range=(low,high)=>low===high?`${hours(low)} 小时`:`${hours(low)}–${hours(high)} 小时`;
  const lower=Math.min(value.lower_bound_hours,68),upper=value.upper_bound_hours==null?68:Math.min(value.upper_bound_hours,68);
  return {
    kind:`多模态研究 · ${labels[value.partition]||'未分组'} · ${value.mode==='multimodal_reinference'?'本地重新推理':'离线归档'} · 种子 ${value.seed}`,
    prediction:value.capped_tail?'至少 68 小时':`${hours(value.prediction_hours)} 小时`,
    actual:lower>=68?'至少 68 小时':range(lower,upper),
    error:range(value.error_lower_hours,value.error_upper_hours),
    timeNotice:value.capped_tail?'预测落在 68 小时以上尾部，不提供确定消散时刻。':'消散时间中位数；离线研究结果，不是实时预警。'
  };
}

if(typeof module!=='undefined')module.exports={researchPresentation};
if(typeof document!=='undefined'){
  let researchStatus=null,researchCases=[];
  window.renderMultimodalResult=value=>{
    const view=researchPresentation(value);
    $('result-kind').textContent=view.kind;
    $('prediction').textContent=view.prediction;
    $('actual').textContent=view.actual;
    $('error').textContent=view.error;
    $('actual-label').textContent='历史标签（68 小时封顶）';
    $('error-label').textContent='绝对误差范围（同口径）';
    $('recovery-label').textContent='预测含义';
    $('recovery-time').textContent=view.timeNotice;
    $('recovery-criterion').textContent='AQI ≤ 100 连续 6 小时，以首次满足条件的窗口起点计时。';
    window.showWeatherInputs?.(value);
  };
  function controls(){
    $('research-reinfer').disabled=busy||!researchStatus?.reinfer_available||!$('research-time').value;
    $('research-form').querySelector('[type=submit]').disabled=busy||!researchStatus||!$('research-time').value;
  }
  window.setMultimodalControls=controls;
  function filterCases(){
    const records=researchCases.filter(row=>row.partition===$('research-partition').value);
    $('research-time').replaceChildren(...records.map(row=>new Option(formatTime(row.timestamp),row.timestamp)));
    controls();
  }
  async function loadResearch(){
    if(researchStatus){controls();return;}
    setBusy(true);message('正在核验多模态研究数据…');
    try{
      const status=await api('/api/multimodal/status');
      if(!status.available)throw new Error('多模态研究数据尚未就绪。');
      const cases=await api('/api/multimodal/cases');
      researchStatus=status;researchCases=cases.records;
      $('research-arm').replaceChildren(...status.arms.map(arm=>new Option(arm.label,arm.value)));
      $('research-arm').value='spatial_fusion';
      $('research-seed').replaceChildren(...status.seeds.map(seed=>new Option(String(seed),String(seed))));
      $('research-state').textContent=status.reinfer_available?'归档与本地重新推理可用':'当前环境仅支持归档查看';
      filterCases();message('请选择样本。训练、开发与最终审计结果分别呈现。');
    }catch(error){message(error.message,true);$('research-state').textContent='研究数据暂不可用';}
    finally{setBusy(false);controls();}
  }
  $('tab-multimodal').addEventListener('click',()=>{if(!busy&&mode==='multimodal')loadResearch();});
  $('research-partition').addEventListener('change',()=>{clearResult();filterCases();});
  ['research-time','research-arm','research-seed'].forEach(id=>$(id).addEventListener('change',()=>{clearResult();message('选择已更改，请重新查看结果。');}));
  function payload(){return {timestamp:$('research-time').value,arm:$('research-arm').value,seed:Number($('research-seed').value)};}
  $('research-form').addEventListener('submit',event=>{
    event.preventDefault();
    run(()=>api('/api/multimodal/replay?'+new URLSearchParams(payload())));
  });
  $('research-reinfer').addEventListener('click',()=>run(()=>api('/api/multimodal/reinfer',payload())));
}
