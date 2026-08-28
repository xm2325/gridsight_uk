// Isolated event-handler regressions; real browser QA remains separate.
const {test}=require('node:test');
const assert=require('node:assert/strict');
const vm=require('node:vm');
const fs=require('node:fs');
const path=require('node:path');
const templateName=process.env.GRIDSIGHT_REVIEW_TEMPLATE||'uk_component_review.html';
const html=fs.readFileSync(path.join(__dirname,'../templates',templateName),'utf8');
const script=html.match(/<script>\n([\s\S]*?)<\/script>/)[1];

function fixture(){
  const elements=new Map();
  function element(id=''){
    return {id,value:'',textContent:'',style:{},handlers:{},children:[],checked:false,
      addEventListener(name,fn){this.handlers[name]=fn;},
      setAttribute(){},replaceChildren(){this.children=[];},append(...v){this.children.push(...v);}};
  }
  function get(id){if(!elements.has(id))elements.set(id,element(id));return elements.get(id);}
  get('reviewData').textContent=JSON.stringify({classes:['pole','crossarm','insulator'],images:[{image_id:'a',width:800,height:600,image_file:'a.jpg',regions:[]}]});
  const context=vm.createContext({document:{getElementById:get,createElementNS:()=>element(),createElement:()=>element(),querySelectorAll:()=>[]},
    window:{addEventListener(){}},location:{search:''},URLSearchParams,
    fetch:()=>new Promise(()=>{}),crypto:{randomUUID:()=> 'manual-test'}});
  vm.runInContext(script,context);
  vm.runInContext(`drafts={a:{image_sha256:'source',status:'ready_for_second_review',reviewer:'QA',notes:'',revision:0,objects:[{id:'o',class_id:2,box:[10,20,100,120],material:null,material_evidence:'',origin:'manual_draft'}]}};ready=true;selected='o';`,context);
  for(const [id,value] of Object.entries({x1:'10',y1:'20',x2:'100',y2:'120',material:'',materialEvidence:'',reviewer:'QA',reviewStatus:'ready_for_second_review',notes:''}))get(id).value=value;
  return {get,fire:(id,event)=>get(id).handlers[event](),read:code=>vm.runInContext(code,context)};
}

test('coordinate input updates draft before blur and invalidates review readiness',()=>{
  const f=fixture();f.get('x1').value='25';f.fire('x1','input');
  assert.equal(f.read('object().box[0]'),25);assert.equal(f.read('draft().status'),'draft');
});
test('empty or invalid coordinate does not become a zero-valued saved box',()=>{
  const f=fixture();f.get('x1').value='';f.fire('x1','input');
  assert.equal(f.read('object().box[0]'),10);assert.equal(f.read('pending.size'),0);
  assert.match(f.get('formError').textContent,/坐标无效|Invalid coordinates/);
});
test('explicit save rereads current coordinates and evidence even without change event',()=>{
  const f=fixture();f.get('x1').value='30';f.get('material').value='glass';f.get('materialEvidence').value='QA only';f.fire('save','click');
  assert.equal(f.read('object().box[0]'),30);assert.equal(f.read('object().material_evidence'),'QA only');
  assert.equal(f.read('draft().status'),'draft');
});
test('invalid current form cannot claim explicit save success',()=>{
  const f=fixture();f.get('x1').value='999';f.fire('save','click');
  assert.equal(f.read('pending.size'),0);assert.equal(f.read('object().box[0]'),10);
});
test('evidence and notes are retained on input before any subsequent render',()=>{
  const f=fixture();f.get('materialEvidence').value='unknown due to resolution';f.fire('materialEvidence','input');
  f.get('notes').value='QA note';f.fire('notes','input');
  assert.equal(f.read('object().material_evidence'),'unknown due to resolution');assert.equal(f.read('draft().notes'),'QA note');
});
test('reviewer edit downgrades ready image and zoom label follows direct scene redraw',()=>{
  const f=fixture();f.get('reviewer').value='Second QA';f.fire('reviewer','input');
  assert.equal(f.read('draft().status'),'draft');f.read('zoom=2;drawScene()');
  assert.match(f.get('dimensions').textContent,/2\.0×/);
});
