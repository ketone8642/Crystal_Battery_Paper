import {EXAMPLE,crystalSystem,populationLabel,number,concise,capitalize,csvToRequests,inputTemplate,predictionCSV,friendlyError,validatePredictions,profileGeometry} from './ui_utils.mjs?v=8.1';

const $=id=>document.getElementById(id);
const state={ready:false,catalog:null,results:{single:null,batch:null,profile:null},requests:null,sequence:{single:0,batch:0,profile:0},controllers:{},busy:{},fileVersion:0};
const el=(tag,text,css)=>{const node=document.createElement(tag);if(text!==undefined)node.textContent=text;if(css)node.className=css;return node;};
const setText=(id,text)=>{$(id).textContent=text;};
function showError(kind,message){const box=$(kind+'-error');box.textContent=message||'';box.hidden=!message;}
function updateButtons(){for(const kind of ['single','batch','profile']){const button=$(kind+'-submit');button.disabled=!state.ready||!!state.busy[kind]||(kind==='batch'&&!state.requests);button.textContent=state.busy[kind]?'Predicting…':button.dataset.label+' →';button.setAttribute('aria-busy',String(!!state.busy[kind]));}}
function clearResult(kind){state.results[kind]=null;if(kind==='single'){$('single-result').hidden=true;$('single-empty').hidden=false;}else $(kind+'-results').hidden=true;}
function invalidate(kind){state.sequence[kind]++;state.controllers[kind]?.abort();state.busy[kind]=false;clearResult(kind);showError(kind,'');updateButtons();}

async function fetchJSON(path,options={}){
  let response;
  try{response=await fetch(path,{...options,headers:{'Content-Type':'application/json',...(options.headers||{})}});}catch(error){if(error.name==='AbortError')throw error;throw new Error('Cannot reach the model service. Check that the server is running, then reconnect.');}
  let data;try{data=await response.json();}catch{throw new Error('The server did not return a valid API response. Restart the server from the project folder.');}
  if(!response.ok){const error=new Error(friendlyError(data.detail));error.status=response.status;throw error;}
  return data;
}

async function connect(){
  $('reconnect').disabled=true;setText('status-pill','Connecting to models');$('status-pill').className='status-pill';
  try{
    const health=await fetchJSON('/health');
    if(!health.model_loaded||health.loaded_models!==6)throw new Error(health.message||'The trained models are not ready. Complete Step 7 configuration.');
    const catalog=await fetchJSON('/models');
    if(!Array.isArray(catalog.models)||catalog.models.length!==6||catalog.run_id!==health.run_id)throw new Error('The model catalog does not match the active run. Reconnect after restarting the server.');
    if(state.catalog&&state.catalog.run_id!==catalog.run_id)for(const kind of ['single','batch','profile'])invalidate(kind);
    state.catalog=catalog;state.ready=true;setText('status-pill','6 models ready');$('status-pill').className='status-pill ready';$('connection-error').hidden=true;
    $('test-banner').hidden=!String(catalog.data_source).toLowerCase().includes('synthetic');renderModels();updateRouting();
  }catch(error){state.ready=false;setText('status-pill','Models unavailable');$('status-pill').className='status-pill offline';setText('connection-error',error.message);$('connection-error').hidden=false;}
  finally{$('reconnect').disabled=false;updateButtons();}
}

function openTab(name,focus=false){
  for(const tab of document.querySelectorAll('[data-tab]')){const active=tab.dataset.tab===name;tab.setAttribute('aria-selected',String(active));tab.tabIndex=active?0:-1;$('panel-'+tab.dataset.tab).hidden=!active;if(active&&focus)tab.focus();}
}
for(const tab of document.querySelectorAll('[data-tab]')){
  tab.addEventListener('click',()=>openTab(tab.dataset.tab));
  tab.addEventListener('keydown',event=>{
    const tabs=[...document.querySelectorAll('[data-tab]')],index=tabs.indexOf(tab);let next;
    if(event.key==='ArrowRight')next=(index+1)%tabs.length;else if(event.key==='ArrowLeft')next=(index+tabs.length-1)%tabs.length;else if(event.key==='Home')next=0;else if(event.key==='End')next=tabs.length-1;else return;
    event.preventDefault();openTab(tabs[next].dataset.tab,true);
  });
}

function updateSystems(){for(const side of ['charge','discharge'])setText(side+'-system',capitalize(crystalSystem($(side+'-sg').value))||'Enter a number from 1 to 230');}
function updateRouting(){
  const ion=$('single-ion').value,selected=$('single-population').value;
  const population=selected==='auto'?(['Li','Na','K'].includes(ion)?'li_only':'mixed_ions'):selected;
  const family=$('single-family').value==='auto'?(state.catalog?.default_family_by_population[population]||'CV choice'):$('single-family').value;
  setText('single-routing',populationLabel(population)+' · '+family.toUpperCase()+(['Na','K'].includes(ion)?' · '+ion+' is absent from training ions.':''));
}
function singleRequest(){
  const a=Number($('charge-sg').value),b=Number($('discharge-sg').value);
  if(!crystalSystem(a)||!crystalSystem(b))throw new Error('Space-group numbers must be integers from 1 to 230.');
  return {working_ion:$('single-ion').value,formula_charge:$('charge-formula').value.trim(),formula_discharge:$('discharge-formula').value.trim(),crystal_system_charge:crystalSystem(a),spacegroup_number_charge:a,crystal_system_discharge:crystalSystem(b),spacegroup_number_discharge:b,model_family:$('single-family').value,training_population:$('single-population').value};
}

async function predict(kind,path,body){
  if(!state.ready){showError(kind,'The model service is not ready. Reconnect first.');return;}
  invalidate(kind);const sequence=state.sequence[kind];const controller=new AbortController();state.controllers[kind]=controller;state.busy[kind]=true;updateButtons();
  try{
    const response=await fetchJSON(path,{method:'POST',body:JSON.stringify(body),signal:controller.signal});
    if(sequence!==state.sequence[kind])return;
    const expected=kind==='single'?1:kind==='batch'?body.items.length:body.intervals.length;
    const items=validatePredictions(kind==='single'?response:response.predictions,expected);
    if(items.some(p=>p.run_id!==state.catalog.run_id))throw new Error('The active model run changed. Reconnect and predict again.');
    if(kind==='profile')profileGeometry(items);
    state.results[kind]=items;
    if(kind==='single')renderSingle(items[0]);else if(kind==='batch')renderBatch(items);else renderProfile(items);
  }catch(error){if(error.name==='AbortError'||sequence!==state.sequence[kind])return;clearResult(kind);showError(kind,error.message);}
  finally{if(sequence===state.sequence[kind]){state.busy[kind]=false;updateButtons();}}
}

function renderNotices(container,items){
  container.replaceChildren();const warnings=[...new Set(items.flatMap(item=>item.warnings))];
  for(const message of warnings){
    if(message.startsWith('Prediction is based on')||message.startsWith('A calibrated uncertainty'))continue;
    container.append(el('p',message));
  }
}

function renderSingle(p){
  $('single-empty').hidden=true;$('single-result').hidden=false;
  setText('result-voltage',number(p.predicted_voltage_V));setText('result-ion',p.working_ion);setText('result-reaction',p.interval.formula_charge+' → '+p.interval.formula_discharge);
  $('result-tags').replaceChildren(...[p.model.family.toUpperCase(),populationLabel(p.model.training_population),p.extrapolates_working_ion?'Held-out ion estimate':'Working ion seen in training'].map(text=>el('span',text,'tag')));
  setText('result-loading',concise(p.interval.ion_per_host_charge)+' → '+concise(p.interval.ion_per_host_discharge));
  setText('result-error-label',p.extrapolates_working_ion?'Reference '+p.working_ion+'-set MAE':'Reference holdout MAE');setText('result-error-value',number(p.evaluation_reference.all?.mae_V)+' V');
  const details=[['Charge atom fraction',concise(p.interval.fracA_charge)],['Discharge atom fraction',concise(p.interval.fracA_discharge)],['Training intervals',p.model.training_rows.toLocaleString('en-US')],['Training ions',p.model.training_ions.join(', ')],['Development CV MAE',number(p.model.cv_mean_mae_V)+' V'],['Representation',p.model.representation==='scaled'?'Scaled features · no PCA':'PCA components']];
  $('result-details').replaceChildren(...details.flatMap(([key,value])=>[el('dt',key),el('dd',value)]));renderNotices($('single-notices'),[p]);
}

function table(headers,rows,numericColumns=[]){
  const node=el('table'),head=el('thead'),tr=el('tr');
  headers.forEach((text,i)=>{const th=el('th',text,numericColumns.includes(i)?'numeric':undefined);th.scope='col';tr.append(th);});head.append(tr);node.append(head);const body=el('tbody');
  for(const row of rows){const line=el('tr');row.forEach((value,i)=>{const td=el('td',undefined,numericColumns.includes(i)?'numeric':undefined);if(value instanceof Node)td.append(value);else td.textContent=value;line.append(td);});body.append(line);}node.append(body);return node;
}
function reactionCell(p){return el('span',p.interval.formula_charge+' → '+p.interval.formula_discharge,'reaction-cell');}
function rowNotices(p){if(p.extrapolates_working_ion)return 'Held-out ion';if(p.predicted_voltage_V<0)return 'Negative voltage';if(p.changed_training_constant_features)return 'Unseen descriptor values';return '—';}
function renderBatch(items){
  $('batch-results').hidden=false;setText('batch-summary',items.length+' predictions · original row order preserved');
  $('batch-table').replaceChildren(table(['Row','Reaction','Ion','Voltage (V)','Model','Population','Notice'],items.map((p,i)=>[i+1,reactionCell(p),p.working_ion,number(p.predicted_voltage_V),p.model.family.toUpperCase(),populationLabel(p.model.training_population),rowNotices(p)]),[0,3]));
}

let stateCounter=0;
function addState(formula='',sg=''){
  if($('profile-states').children.length>=51)return;
  const id=++stateCounter,row=el('div',undefined,'state-row');row.dataset.state=id;row.append(el('span','', 'state-number'));
  const formulaField=el('div'),label=el('label','Chemical formula'),input=el('input');input.id='state-'+id+'-formula';input.className='state-formula';input.value=formula;input.placeholder='Endpoint formula';input.required=true;input.maxLength=256;input.autocomplete='off';input.spellcheck=false;label.htmlFor=input.id;formulaField.append(label,input);
  const symmetryField=el('div'),sgLabel=el('label','Space group'),sgInput=el('input');sgInput.id='state-'+id+'-sg';sgInput.className='state-sg';sgInput.type='number';sgInput.min=1;sgInput.max=230;sgInput.step=1;sgInput.required=true;sgInput.value=sg;sgLabel.htmlFor=sgInput.id;const system=el('p',capitalize(crystalSystem(sg))||'—','system-label');symmetryField.append(sgLabel,sgInput,system);
  const remove=el('button','×','remove-state');remove.type='button';remove.addEventListener('click',()=>{if($('profile-states').children.length>3){invalidate('profile');row.remove();renumberStates();}});row.append(formulaField,symmetryField,remove);$('profile-states').append(row);renumberStates();
}
function renumberStates(){const rows=[...$('profile-states').children];rows.forEach((row,index)=>{row.querySelector('.state-number').textContent=String(index+1).padStart(2,'0');row.querySelector('.state-formula').setAttribute('aria-label','Endpoint '+(index+1)+' formula');row.querySelector('.state-sg').setAttribute('aria-label','Endpoint '+(index+1)+' space-group number');const remove=row.querySelector('.remove-state');remove.disabled=rows.length<=3;remove.setAttribute('aria-label','Remove endpoint '+(index+1));});$('add-state').disabled=rows.length>=51;}
function profileRequest(){
  const states=[...$('profile-states').children].map(row=>({formula:row.querySelector('.state-formula').value.trim(),sg:Number(row.querySelector('.state-sg').value)}));
  if(states.some(s=>!s.formula||!crystalSystem(s.sg)))throw new Error('Complete every endpoint formula and space-group number.');
  const intervals=states.slice(0,-1).map((a,i)=>({working_ion:$('profile-ion').value,formula_charge:a.formula,formula_discharge:states[i+1].formula,spacegroup_number_charge:a.sg,spacegroup_number_discharge:states[i+1].sg,crystal_system_charge:crystalSystem(a.sg),crystal_system_discharge:crystalSystem(states[i+1].sg),model_family:$('profile-family').value,training_population:$('profile-population').value}));
  return {intervals};
}

function svgNode(tag,attributes={},text){const node=document.createElementNS('http://www.w3.org/2000/svg',tag);for(const [name,value]of Object.entries(attributes))node.setAttribute(name,String(value));if(text!==undefined)node.textContent=text;return node;}
function renderChart(items){
  const g=profileGeometry(items),svg=svgNode('svg',{viewBox:`0 0 ${g.W} ${g.H}`,role:'img','aria-labelledby':'profile-title profile-description'});
  svg.append(svgNode('title',{id:'profile-title'},'Predicted average voltage by insertion interval'),svgNode('desc',{id:'profile-description'},'Horizontal segments show predicted interval averages in volts. The table below gives all loading limits and values. Dashed vertical joins are visual guides.'));
  for(const tick of g.yTicks){svg.append(svgNode('line',{x1:g.left,x2:g.W-g.right,y1:tick.position,y2:tick.position,stroke:'#e0e9e3','stroke-width':1}));svg.append(svgNode('text',{x:g.left-12,y:tick.position+4,'text-anchor':'end',fill:'#597269','font-size':11},number(tick.value,2)));}
  for(const tick of g.xTicks){svg.append(svgNode('line',{x1:tick.position,x2:tick.position,y1:g.H-g.bottom,y2:g.H-g.bottom+5,stroke:'#819f90'}));svg.append(svgNode('text',{x:tick.position,y:g.H-g.bottom+22,'text-anchor':'middle',fill:'#597269','font-size':11},concise(tick.value)));}
  svg.append(svgNode('line',{x1:g.left,x2:g.W-g.right,y1:g.H-g.bottom,y2:g.H-g.bottom,stroke:'#90aa9b'}));
  g.segments.forEach((segment,index)=>{
    if(index){const previous=g.segments[index-1];svg.append(svgNode('line',{x1:segment.x1,x2:segment.x1,y1:previous.y,y2:segment.y,stroke:'#72a58c','stroke-width':1.4,'stroke-dasharray':'4 4'}));}
    const line=svgNode('line',{x1:segment.x1,x2:segment.x2,y1:segment.y,y2:segment.y,stroke:'#087768','stroke-width':3.5,'stroke-linecap':'round','data-interval':index+1});line.append(svgNode('title',{},`Interval ${index+1}: ${segment.prediction.interval.formula_charge} → ${segment.prediction.interval.formula_discharge}; ${number(segment.prediction.predicted_voltage_V)} V`));svg.append(line);
  });
  svg.append(svgNode('text',{x:g.W/2,y:g.H-13,'text-anchor':'middle',fill:'#3e5b4e','font-size':12},'Working-ion atoms per reduced host'),svgNode('text',{transform:`translate(17 ${(g.H-g.bottom+g.top)/2}) rotate(-90)`,'text-anchor':'middle',fill:'#3e5b4e','font-size':12},'Average voltage (V)'));
  $('profile-chart').replaceChildren(svg);
}
function renderProfile(items){
  $('profile-results').hidden=false;const first=items[0];setText('profile-summary',items.length+' intervals · '+first.working_ion+' · '+first.model.family.toUpperCase()+' · '+populationLabel(first.model.training_population));renderChart(items);
  $('profile-table').replaceChildren(table(['Interval','Reaction','Loading start','Loading end','Voltage (V)'],items.map((p,i)=>[i+1,reactionCell(p),concise(p.interval.ion_per_host_charge),concise(p.interval.ion_per_host_discharge),number(p.predicted_voltage_V)]),[0,2,3,4]));renderNotices($('profile-notices'),items);
}

function renderModels(){
  const catalog=state.catalog,cards=catalog.models;
  $('model-populations').replaceChildren(...['mixed_ions','li_only'].map(population=>{
    const card=cards.find(c=>c.training_population===population&&c.selected_by_cv),node=el('article',undefined,'card population-card');node.append(el('p','SELECTED BY DEVELOPMENT CV','eyebrow'),el('h2',populationLabel(population)));
    if(card){const family=el('div',card.family.toUpperCase(),'model-name');family.append(el('span',card.training_rows.toLocaleString('en-US')+' training intervals'));node.append(family,el('p','Training ions: '+card.training_ions.join(', ')),el('p',card.representation==='scaled'?'Scaled composition and symmetry features, without PCA.':'PCA-transformed composition and symmetry features.'));}return node;
  }));
  const rows=cards.map(card=>{const name=el('span',card.family.toUpperCase());if(card.selected_by_cv)name.append(el('span','CV choice','chosen'));return [populationLabel(card.training_population),name,number(card.cv_mean_mae_V),number(card.holdout?.all?.mae_V),number(card.held_out_Na?.all?.mae_V),number(card.held_out_K?.all?.mae_V)];});
  $('models-table').replaceChildren(table(['Population','Family','CV MAE','Holdout MAE','Na MAE','K MAE'],rows,[2,3,4,5]));
  const baselines=catalog.median_baseline_by_population||{};const available=Object.entries(baselines).filter(([,b])=>typeof b.holdout?.all?.mae_V==='number');
  $('baseline-summary').replaceChildren(...available.map(([population,baseline])=>el('p',populationLabel(population)+' median baseline · holdout MAE '+number(baseline.holdout.all.mae_V)+' V')));
  setText('active-run','Active training run: '+catalog.run_id);
}

function download(content,name,type){const blob=new Blob([content],{type}),url=URL.createObjectURL(blob),a=el('a');a.href=url;a.download=name;document.body.append(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),1500);}
function downloadResults(kind){const predictions=state.results[kind];if(!predictions)return;download(predictionCSV(predictions),'voltage_'+kind+'_results.csv','text/csv;charset=utf-8');}

$('single-form').addEventListener('input',()=>{invalidate('single');updateSystems();updateRouting();});
$('single-form').addEventListener('change',()=>{invalidate('single');updateSystems();updateRouting();});
$('single-form').addEventListener('submit',event=>{event.preventDefault();try{predict('single','/predict',singleRequest());}catch(error){showError('single',error.message);}});
$('load-example').addEventListener('click',()=>{invalidate('single');$('single-ion').value='Li';$('charge-formula').value=EXAMPLE.formula_charge;$('discharge-formula').value=EXAMPLE.formula_discharge;$('charge-sg').value=62;$('discharge-sg').value=62;$('single-family').value='auto';$('single-population').value='auto';updateSystems();updateRouting();});
$('single-csv').addEventListener('click',()=>downloadResults('single'));$('single-json').addEventListener('click',()=>{if(state.results.single)download(JSON.stringify(state.results.single[0],null,2),'voltage_prediction.json','application/json');});
$('batch-template').addEventListener('click',()=>download(inputTemplate(),'reaction_input_template.csv','text/csv;charset=utf-8'));
$('batch-file').addEventListener('change',async event=>{
  state.requests=null;invalidate('batch');const version=++state.fileVersion,file=event.target.files[0];if(!file){setText('batch-file-note','Choose a CSV matching the input template.');return;}
  try{if(file.size>1024*1024)throw new Error('The file exceeds 1 MB. Split it into smaller batches.');const text=await file.text();if(version!==state.fileVersion)return;state.requests=csvToRequests(text);setText('batch-file-note',file.name+' · '+state.requests.length+' reaction rows ready.');}
  catch(error){if(version===state.fileVersion){setText('batch-file-note',file.name);showError('batch',error.message);}}
  finally{if(version===state.fileVersion)updateButtons();}
});
$('batch-form').addEventListener('submit',event=>{event.preventDefault();if(!state.requests){showError('batch','Choose a valid CSV file first.');return;}predict('batch','/predict/batch',{items:state.requests});});$('batch-csv').addEventListener('click',()=>downloadResults('batch'));
$('profile-form').addEventListener('input',()=>{invalidate('profile');for(const row of $('profile-states').children)row.querySelector('.system-label').textContent=capitalize(crystalSystem(row.querySelector('.state-sg').value))||'—';});
$('profile-form').addEventListener('change',()=>invalidate('profile'));
$('add-state').addEventListener('click',()=>{invalidate('profile');addState();});
$('profile-form').addEventListener('submit',event=>{event.preventDefault();try{predict('profile','/predict/profile',profileRequest());}catch(error){showError('profile',error.message);}});$('profile-csv').addEventListener('click',()=>downloadResults('profile'));
$('reconnect').addEventListener('click',connect);
addState();addState();addState();updateSystems();updateRouting();connect();
