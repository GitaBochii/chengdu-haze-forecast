import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {createRequire} from 'node:module';
import {runInNewContext} from 'node:vm';

const require=createRequire(import.meta.url);
const analysis=require('../project-showcase/rebound-analysis.js');
const payload=JSON.parse(readFileSync(new URL('../project-showcase/rebound-cases.json',import.meta.url),'utf8'));
const code=readFileSync(new URL('../project-showcase/rebound.js',import.meta.url),'utf8');

// A minimal DOM exercises the real UI handlers without a browser download dialog.
function fixture(ok=true){
  const nodes=new Map();
  let exported=null,clicked=false;
  function element(){return {
    handlers:{},classList:{toggle(){}},clientWidth:800,clientHeight:340,
    set innerHTML(value){this.html=value;for(const id of value.matchAll(/id="([^"]+)"/g))nodes.set(id[1],element());},
    get innerHTML(){return this.html;},
    addEventListener(name,fn){this.handlers[name]=fn;},
    replaceChildren(...children){this.children=children;},append(){},
    querySelectorAll(){return [];},click(){clicked=true;},
  };}
  nodes.set('rebound-controls',element());nodes.set('rebound-view',element());
  const window={ReboundAnalysis:analysis};
  runInNewContext(code,{window,document:{getElementById:id=>nodes.get(id),createElement:element,addEventListener(){}},
    Intl,Date,Number,Option:class{constructor(text,value){this.text=text;this.value=value;}},
    ResizeObserver:class{observe(){}},AbortController,Blob,
    URL:{createObjectURL(blob){exported=blob;return 'blob:test';},revokeObjectURL(){}},
    fetch:async()=>({ok,json:async()=>structuredClone(payload)}),
    setTimeout(){return 1;},clearTimeout(){},setInterval(){return 1;},clearInterval(){}});
  return {window,nodes,download:async()=>{clicked=false;nodes.get('rebound-download').handlers.click();assert.ok(clicked);return JSON.parse(await exported.text());}};
}
const f=fixture();await f.window.openReboundView();
let value=await f.download();assert.equal(value.replay_hour,0);assert.equal(value.case.actual.length,0);
assert.equal('category' in value.case,false);assert.equal('title' in value.case,false);
f.nodes.get('rebound-hour').value='5';f.nodes.get('rebound-hour').handlers.input();
value=await f.download();assert.equal(value.case.actual.length,5);assert.equal(value.case.forecast.length,24);
f.nodes.get('rebound-hour').value='24';f.nodes.get('rebound-hour').handlers.input();
value=await f.download();assert.equal(value.case.actual.length,24);assert.equal(value.case.category,payload.cases[0].category);
f.nodes.get('rebound-reset').handlers.click();assert.equal((await f.download()).case.actual.length,0);
f.window.hideReboundView();assert.equal(f.nodes.get('rebound-view').hidden,true);
const broken=fixture(false);await broken.window.openReboundView();
assert.equal(broken.nodes.get('rebound-retry').hidden,false);
assert.equal(broken.nodes.get('rebound-case').disabled,true);
assert.equal(broken.nodes.get('rebound-chart').children.length,0);
assert.ok(broken.nodes.get('rebound-error').textContent.includes('尚未就绪'));
console.log('Rebound UI export, reset, tab hiding and error-state checks passed');
