const $ = id => document.getElementById(id);
let latestResult = null, previewURL = null;
function setMode(image) { $('imageForm').hidden = !image; $('textForm').hidden = image; for (const [id,on] of [['imageTab',image],['textTab',!image]]) { $(id).classList.toggle('active',on); $(id).setAttribute('aria-selected',String(on)); } }
$('imageTab').onclick = () => setMode(true); $('textTab').onclick = () => setMode(false);
function preview(file) { if(previewURL) URL.revokeObjectURL(previewURL); previewURL = URL.createObjectURL(file); $('preview').src = previewURL; $('preview').hidden = false; }
$('file').onchange = () => { if($('file').files[0]) preview($('file').files[0]); };
function showResult(result) {
  latestResult=result; $('empty').hidden=true; $('result').hidden=false;
  $('urgency').textContent=result.urgency; $('urgency').dataset.label=result.urgency;
  $('latency').textContent=`${result.latency_ms ?? '—'} ms server processing`;
  $('alert').hidden=!result.critical_alert; $('review').textContent=(result.review_reasons||[]).join(' '); $('review').hidden=!result.review_required;
  $('outFindings').textContent=result.findings || '(empty)'; $('outImpression').textContent=result.impression || '(empty)';
  $('entities').replaceChildren(); for(const ent of result.entities||[]) {const tr=document.createElement('tr'); for(const v of [ent.text,ent.label,ent.assertion]) {const td=document.createElement('td');td.textContent=v;tr.append(td);} $('entities').append(tr);}
  if(!(result.entities||[]).length){const tr=document.createElement('tr'),td=document.createElement('td');td.colSpan=3;td.textContent='No entities found by the prototype tagger.';tr.append(td);$('entities').append(tr);}
  $('scores').replaceChildren(); for(const [label,score] of Object.entries(result.probabilities||{})){const row=document.createElement('div');row.className='score-row';const a=document.createElement('span'),b=document.createElement('span');a.textContent=label;b.textContent=(score*100).toFixed(1)+'%';row.append(a,b);$('scores').append(row);}
  $('raw').textContent=JSON.stringify(result,null,2);
}
async function analyze(path,options) { $('message').textContent='Processing report...'; document.querySelectorAll('button').forEach(b=>b.disabled=true); latestResult=null; $('result').hidden=true; $('empty').hidden=false;
  try {const response=await fetch(path,options);let data=await response.json();if(!response.ok)throw Error(typeof data.detail==='string'?data.detail:JSON.stringify(data.detail));showResult(data);$('message').textContent='Analysis complete. Review the source report and the model limitations.';}
  catch(error){$('message').textContent=error.message;}finally{document.querySelectorAll('button').forEach(b=>b.disabled=false);}
}
$('imageForm').onsubmit=event=>{event.preventDefault();const file=$('file').files[0];if(!file)return;if(file.size>8000000){$('message').textContent='Choose a file smaller than 8 MB.';return;}const body=new FormData();body.append('file',file);analyze('/predict-image',{method:'POST',body});};
$('textForm').onsubmit=event=>{event.preventDefault();if(!$('findings').value.trim()&&!$('impression').value.trim()){$('message').textContent='Enter Findings or Impression.';return;}analyze('/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({findings:$('findings').value,impression:$('impression').value})});};
document.querySelectorAll('[data-sample]').forEach(button=>button.onclick=async()=>{try{const response=await fetch('/samples/'+button.dataset.sample);if(!response.ok)throw Error('Sample not available.');const file=new File([await response.blob()],button.dataset.sample,{type:'image/png'});preview(file);setMode(true);const body=new FormData();body.append('file',file);await analyze('/predict-image',{method:'POST',body});}catch(error){$('message').textContent=error.message;}});
$('download').onclick=()=>{if(!latestResult)return;const link=document.createElement('a');const url=URL.createObjectURL(new Blob([JSON.stringify(latestResult,null,2)],{type:'application/json'}));link.href=url;link.download='radnote-result.json';link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};
fetch('/health').then(r=>r.json()).then(h=>{$('health').textContent=h.status==='ok'?'Service ready':(h.classifier_ready?'Text ready · OCR unavailable':'Service unavailable');}).catch(()=>{$('health').textContent='Cannot reach service';});
