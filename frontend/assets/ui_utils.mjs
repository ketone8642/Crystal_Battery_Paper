export const IONS = ['Li','Na','K','Mg','Ca','Zn','Al','Y'];
export const REQUIRED = ['working_ion','formula_charge','formula_discharge','crystal_system_charge','spacegroup_number_charge','crystal_system_discharge','spacegroup_number_discharge'];
const OPTIONAL = ['fracA_charge','fracA_discharge','model_family','training_population'];
export const EXAMPLE = {working_ion:'Li',formula_charge:'FePO4',formula_discharge:'LiFePO4',crystal_system_charge:'orthorhombic',spacegroup_number_charge:62,crystal_system_discharge:'orthorhombic',spacegroup_number_discharge:62,model_family:'auto',training_population:'auto'};
export const populationLabel = value => value === 'li_only' ? 'Li only' : value === 'mixed_ions' ? 'Mixed ions' : value;
export const number = (value, digits=3) => typeof value === 'number' && Number.isFinite(value) ? value.toLocaleString('en-US',{maximumFractionDigits:digits,minimumFractionDigits:digits}) : '—';
export const concise = value => typeof value === 'number' && Number.isFinite(value) ? String(Number(value.toPrecision(6))) : '—';
export const capitalize = text => text ? text[0].toUpperCase()+text.slice(1) : '';

export function crystalSystem(value) {
  const n=Number(value);
  if(!Number.isInteger(n)||n<1||n>230) return null;
  return [[2,'triclinic'],[15,'monoclinic'],[74,'orthorhombic'],[142,'tetragonal'],[167,'trigonal'],[194,'hexagonal'],[230,'cubic']].find(([max])=>n<=max)[1];
}

export function parseCSV(text) {
  text=text.replace(/^\uFEFF/,'');
  const rows=[];let row=[],field='',quoted=false,closed=false;
  function endField(){row.push(field);field='';closed=false;}
  function endRow(){endField();rows.push(row);row=[];}
  for(let i=0;i<text.length;i++) {
    const c=text[i];
    if(quoted){if(c==='"'){if(text[i+1]==='"'){field+='"';i++;}else{quoted=false;closed=true;}}else field+=c;continue;}
    if(closed&&c!==','&&c!=='\r'&&c!=='\n')throw new Error('CSV has characters after a closing quote. Use a comma or a new line.');
    if(c==='"'){if(field.length)throw new Error('CSV contains a quote inside an unquoted field.');quoted=true;}
    else if(c===',')endField();
    else if(c==='\n'||c==='\r'){endRow();if(c==='\r'&&text[i+1]==='\n')i++;}
    else field+=c;
  }
  if(quoted)throw new Error('CSV contains an unclosed quoted field.');
  if(field!==''||closed||row.length)endRow();
  return rows.filter(values=>values.some(value=>value.trim()!==''));
}

export function csvToRequests(text) {
  const rows=parseCSV(text);
  if(rows.length<2)throw new Error('Add a header and at least one reaction row.');
  if(rows.length>129)throw new Error('A batch can contain at most 128 reaction rows. Split this file into smaller batches.');
  const headers=rows[0].map(h=>h.trim());
  if(new Set(headers).size!==headers.length)throw new Error('CSV headers must be unique.');
  const missing=REQUIRED.filter(h=>!headers.includes(h));
  if(missing.length)throw new Error('Missing required columns: '+missing.join(', '));
  const unknown=headers.filter(h=>![...REQUIRED,...OPTIONAL].includes(h));
  if(unknown.length)throw new Error('Unexpected columns: '+unknown.join(', ')+'. Use the input template.');
  return rows.slice(1).map((values,index)=>{
    const label='Row '+(index+1)+': ';
    if(values.length!==headers.length)throw new Error(label+'column count does not match the header.');
    const item={};
    headers.forEach((header,j)=>{
      const value=values[j].trim();
      if(!value&&!REQUIRED.includes(header))return;
      if(!value)throw new Error(label+header+' is empty.');
      if(header.startsWith('spacegroup_number_')){
        if(!/^\d+$/.test(value)||!crystalSystem(Number(value)))throw new Error(label+header+' must be an integer from 1 to 230.');
        item[header]=Number(value);
      }else if(header.startsWith('fracA_')){
        const fraction=Number(value);
        if(!Number.isFinite(fraction)||fraction<0||fraction>=1)throw new Error(label+header+' must be an atom fraction between 0 and 1.');
        item[header]=fraction;
      }else item[header]=value;
    });
    if(!IONS.includes(item.working_ion))throw new Error(label+'unsupported working ion.');
    for(const side of ['charge','discharge']){
      if(item['formula_'+side].length>256)throw new Error(label+'formula is too long.');
      if(crystalSystem(item['spacegroup_number_'+side])!==item['crystal_system_'+side])throw new Error(label+side+' crystal system does not match its space-group number.');
    }
    if(item.model_family&&!['auto','dnn','svr','krr'].includes(item.model_family))throw new Error(label+'unknown model family.');
    if(item.training_population&&!['auto','li_only','mixed_ions'].includes(item.training_population))throw new Error(label+'unknown training population.');
    return item;
  });
}

export function encodeCSV(rows) {
  return rows.map(row=>row.map(value=>{
    if(value===null||value===undefined)return '';
    let text=String(value);
    // Preserve negative numeric voltages, but prevent string fields becoming spreadsheet formulas.
    if(typeof value==='string'&&/^[=+\-@\t\r]/.test(text))text="'"+text;
    return /[",\r\n]/.test(text)?'"'+text.replaceAll('"','""')+'"':text;
  }).join(',')).join('\r\n')+'\r\n';
}

export function inputTemplate() {
  const headers=[...REQUIRED,'model_family','training_population'];
  return encodeCSV([headers,...[EXAMPLE,{...EXAMPLE,model_family:'svr'}].map(item=>headers.map(h=>item[h]))]);
}

export function predictionCSV(predictions) {
  const headers=['row','run_id','working_ion','formula_charge','formula_discharge','ion_per_host_charge','ion_per_host_discharge','fracA_charge','fracA_discharge','predicted_voltage_V','model_family','training_population','candidate_id','extrapolates_working_ion','changed_training_constant_features','evaluation_scope','reference_dataset_mae_V','warnings'];
  return encodeCSV([headers,...predictions.map((p,i)=>[i+1,p.run_id,p.working_ion,p.interval.formula_charge,p.interval.formula_discharge,p.interval.ion_per_host_charge,p.interval.ion_per_host_discharge,p.interval.fracA_charge,p.interval.fracA_discharge,p.predicted_voltage_V,p.model.family,p.model.training_population,p.model.candidate_id,p.extrapolates_working_ion,p.changed_training_constant_features,p.evaluation_reference.scope,p.evaluation_reference.all?.mae_V,p.warnings.join(' | ')])]);
}

export function friendlyError(detail) {
  if(typeof detail==='string')return detail;
  if(Array.isArray(detail))return detail.slice(0,4).map(error=>{
    const path=(error.loc||[]).filter(p=>p!=='body');
    const numeric=path.find(p=>typeof p==='number');
    const prefix=numeric!==undefined?'Row '+(numeric+1)+': ':'';
    const field=path.at(-1);
    return prefix+(typeof field==='string'?field.replaceAll('_',' ')+': ':'')+String(error.msg||'Invalid input').replace(/^Value error, /,'');
  }).join('\n');
  return 'The request could not be completed.';
}

export function validatePredictions(value,expectedCount) {
  const items=Array.isArray(value)?value:[value];
  if(items.length!==expectedCount)throw new Error('The API returned a different number of predictions.');
  for(const p of items){
    if(!p||!Number.isFinite(p.predicted_voltage_V)||!p.model||!p.interval||!Array.isArray(p.warnings)||typeof p.run_id!=='string')throw new Error('The API returned an incomplete prediction.');
    for(const key of ['ion_per_host_charge','ion_per_host_discharge','fracA_charge','fracA_discharge'])if(!Number.isFinite(p.interval[key]))throw new Error('The API returned invalid interval information.');
    if(p.interval.ion_per_host_discharge<=p.interval.ion_per_host_charge)throw new Error('The API returned an invalid loading interval.');
  }
  return items;
}

export function profileGeometry(predictions) {
  validatePredictions(predictions,predictions.length);
  if(predictions.length<2)throw new Error('A voltage profile needs at least two intervals.');
  for(let i=1;i<predictions.length;i++){
    const a=predictions[i-1].interval.ion_per_host_discharge,b=predictions[i].interval.ion_per_host_charge;
    if(Math.abs(a-b)>1e-8*Math.max(1,Math.abs(a),Math.abs(b)))throw new Error('Profile intervals are not contiguous.');
  }
  const W=720,H=340,left=64,right=24,top=26,bottom=63;
  const xMin=predictions[0].interval.ion_per_host_charge,xMax=predictions.at(-1).interval.ion_per_host_discharge;
  const ys=predictions.map(p=>p.predicted_voltage_V),min=Math.min(...ys),max=Math.max(...ys);
  const padding=Math.max((max-min)*.15,Math.abs(max)*.02,.1),yMin=min-padding,yMax=max+padding;
  const x=value=>left+(value-xMin)/(xMax-xMin)*(W-left-right);
  const y=value=>H-bottom-(value-yMin)/(yMax-yMin)*(H-top-bottom);
  const segments=predictions.map(p=>({x1:x(p.interval.ion_per_host_charge),x2:x(p.interval.ion_per_host_discharge),y:y(p.predicted_voltage_V),prediction:p}));
  return {W,H,left,right,top,bottom,segments,
    xTicks:Array.from({length:6},(_,i)=>{const value=xMin+(xMax-xMin)*i/5;return {value,position:x(value)};}),
    yTicks:Array.from({length:5},(_,i)=>{const value=yMin+(yMax-yMin)*i/4;return {value,position:y(value)};})};
}
