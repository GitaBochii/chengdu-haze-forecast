(function(root,factory){const api=factory();if(typeof module==='object'&&module.exports)module.exports=api;else root.ReboundAnalysis=api;})(typeof globalThis!=='undefined'?globalThis:this,function(){
  'use strict';
  function prefix(rows,hour){return rows.slice(0,Math.max(0,Math.min(24,Math.floor(Number(hour)||0))));}
  function analyze(rows){
    let run=0,start=0,previous=null,recovery=null;const rebounds=[];
    rows.forEach((row,i)=>{
      const time=Date.parse(row.timestamp),valid=Number.isFinite(time)&&Number.isFinite(row.aqi)&&row.aqi>=0;
      const consecutive=previous!==null&&time-previous===3600000;
      if(!valid||!consecutive){run=0;start=i+1;}
      if(valid&&row.aqi<=100){if(run===0)start=i+1;run++;if(run===6&&!recovery)recovery={start,confirmedAt:i+1};}
      else {if(valid&&consecutive&&run>0&&run<6)rebounds.push({start,at:i+1,lowHours:run,aqi:row.aqi});run=0;}
      previous=valid?time:null;
    });
    return {recovery,rebounds,pendingLowHours:run<6?run:0,observedHours:rows.length};
  }
  function validate(payload){
    if(!payload||payload.version!==1||!Array.isArray(payload.cases)||payload.cases.length===0)throw new Error('案例格式不正确或没有可用案例。');
    const ids=new Set();
    payload.cases.forEach(c=>{
      if(typeof c.id!=='string'||ids.has(c.id)||typeof c.title!=='string'||!Number.isFinite(Date.parse(c.timestamp)))throw new Error('案例标识或起始时间无效。');
      ids.add(c.id);
      [['history','aqi'],['forecast','aqi'],['actual','aqi']].forEach(([key,value])=>{
        if(!Array.isArray(c[key])||c[key].length!==24)throw new Error('案例的小时序列不完整。');
        c[key].forEach((r,i)=>{
          const t=Date.parse(r.timestamp),origin=Date.parse(c.timestamp),expected=origin+(key==='history'?i-24:i+1)*3600000;
          if(t!==expected||!(r[value]===null||(Number.isFinite(r[value])&&r[value]>=0)))throw new Error('案例存在无效数值或不连续的时间。');
        });
      });
    });return payload;
  }
  return {prefix,analyze,validate};
});
