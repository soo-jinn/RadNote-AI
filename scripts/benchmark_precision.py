"""Paired FP32/INT8 engineering benchmark; optional private classifier holdout.

No private report text or row predictions are exported. Synthetic page text is public.
"""
import argparse
import asyncio
import csv
import gc
import hashlib
import json
import os
import platform
import sys
import time
from collections import Counter
from pathlib import Path
import numpy as np
import psutil
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.benchmark_api import normalize, distance, percentile
from api.main import app, lifespan, image_result, result_for, health

ROOT=Path(__file__).resolve().parents[1]
LABELS=['Routine','Urgent','Critical']

def class_metrics(expected,observed):
    matrix=[[sum(a==left and b==right for a,b in zip(expected,observed)) for right in LABELS] for left in LABELS]
    report={}
    f1=[]
    for i,label in enumerate(LABELS):
        tp=matrix[i][i];support=sum(matrix[i]);predicted=sum(row[i] for row in matrix)
        precision=tp/predicted if predicted else 0.
        recall=tp/support if support else None
        score=2*precision*recall/(precision+recall) if recall is not None and precision+recall else 0.
        report[label]={'support':support,'precision':precision,'recall':recall,'f1':score if support else None}
        f1.append(score)
    return {'rows':len(expected),'accuracy':sum(a==b for a,b in zip(expected,observed))/len(expected),
            'macro_f1_three_labels_zero_for_unsupported':sum(f1)/len(f1),'per_class':report,
            'confusion_matrix':matrix,'label_order':LABELS,'validation_scope':'Weak-label agreement only; missing class support is unassessable.'}

def ner_metrics():
    records=[r for r in json.loads((ROOT/'samples/ner/annotations.json').read_text()) if r['partition']=='test']
    tagger=app.state.ner;correct=total=0;tp=Counter();pred=Counter();true=Counter()
    for row in records:
        ids=np.asarray([[tagger.vocabulary.get(token.lower(),1) for token in row['tokens']]],dtype=np.int64)
        labels=tagger.session.run(None,{'tokens':ids})[0][0].argmax(-1)
        output=[tagger.tags[int(index)] if token_id!=1 else 'O' for index,token_id in zip(labels,ids[0])]
        for a,b in zip(row['tags'],output):
            true[a]+=1;pred[b]+=1;total+=1;correct+=a==b
            if a==b:tp[a]+=1
    per_tag={}
    for tag in tagger.tags:
        precision=tp[tag]/pred[tag] if pred[tag] else 0.
        recall=tp[tag]/true[tag] if true[tag] else None
        f1=2*precision*recall/(precision+recall) if recall is not None and precision+recall else 0.
        per_tag[tag]={'support':true[tag],'precision':precision,'recall':recall,'f1':f1 if true[tag] else None}
    return {'test_sentences':len(records),'tokens':total,'token_accuracy':correct/total,'per_tag':per_tag,
            'scope':'Synthetic weak BIO token agreement; shared lexicon across family partitions, not clinical span validation.'}

async def benchmark(args):
    manifest=json.loads((ROOT/'samples/ocr/manifest.json').read_text())
    selected=[r for r in manifest if r['id'] in {'case_001','case_006','case_021','case_026','case_041','case_046'}][:args.limit]
    private=[]
    if args.classifier_test_csv:
        with args.classifier_test_csv.open(encoding='utf-8',newline='') as handle:private=list(csv.DictReader(handle))
        if not private or any(row.get('urgency_label') not in LABELS for row in private):raise ValueError('Classifier holdout must contain the three supported urgency labels and at least one row.')
    results=[];private_outputs={};synthetic_classifier={}
    process=psutil.Process()
    for precision in ['fp32','int8']:
        os.environ['RADNOTE_MODEL_PRECISION']=precision;os.environ['RADNOTE_ENABLE_OCR']='1'
        # An explicit model override would invalidate this paired comparison.
        os.environ.pop('RADNOTE_MODEL_PATH',None)
        start=time.perf_counter()
        async with lifespan(app):
            ready=health()
            if ready['status']!='ok':raise RuntimeError(ready)
            startup=(time.perf_counter()-start)*1000
            rows=[]
            for case in selected:
                content=(ROOT/'samples/ocr'/case['image']).read_bytes()
                for _ in range(args.warmups):image_result(content,'image/png',time.perf_counter())
                for trial in range(args.repeats):
                    begin=time.perf_counter();output=image_result(content,'image/png',begin);elapsed=(time.perf_counter()-begin)*1000
                    reference=normalize(case['reference_text']);observed=normalize(output['extracted_text'])
                    rows.append({'case':case['id'],'trial':trial+1,'milliseconds':elapsed,'rss_mb':process.memory_info().rss/1024**2,
                                 'cer_errors':distance(reference,observed),'cer_length':len(reference),
                                 'wer_errors':distance(reference.split(),observed.split()),'wer_length':len(reference.split()),
                                 'scenario_label':case['scenario_label'],'predicted_label':output['urgency']})
            metrics=ner_metrics()
            labels=[];scores=[]
            for case in selected:
                label,probabilities=app.state.classifier.predict(case['findings']+'\n'+case['impression'])
                labels.append(label);scores.append(probabilities)
            synthetic_classifier[precision]=(labels,scores)
            private_report=None
            if private:
                observed=[];probabilities=[]
                for row in private:
                    label,values=app.state.classifier.predict((row.get('findings','')+'\n'+row.get('impression','')).strip())
                    observed.append(label);probabilities.append(values)
                private_outputs[precision]=(observed,probabilities)
                private_report=class_metrics([r['urgency_label'] for r in private],observed)
            result={'precision':precision,'health':ready,'model_load_ms':startup,'unique_synthetic_pages':len(selected),'trials':len(rows),
                    'repeats':args.repeats,'warmups_per_page':args.warmups,'ocr_cer':sum(r['cer_errors'] for r in rows)/sum(r['cer_length'] for r in rows),
                    'ocr_wer':sum(r['wer_errors'] for r in rows)/sum(r['wer_length'] for r in rows),
                    'mean_pipeline_ms':sum(r['milliseconds'] for r in rows)/len(rows),
                    'p50_pipeline_ms':percentile([r['milliseconds'] for r in rows],.5),'p95_pipeline_ms':percentile([r['milliseconds'] for r in rows],.95),
                    'max_sampled_rss_mb':max(r['rss_mb'] for r in rows),'synthetic_scenario_agreement':sum(r['scenario_label']==r['predicted_label'] for r in rows)/len(rows),
                    'ner_synthetic_token_metrics':metrics,'classifier_private_holdout_metrics':private_report,'rows':rows}
            results.append(result);print(precision+': '+json.dumps({k:result[k] for k in ['unique_synthetic_pages','mean_pipeline_ms','ocr_cer','ocr_wer']}),flush=True)
        gc.collect()
    def fidelity(outputs):
        a,ap=outputs['fp32'];b,bp=outputs['int8']
        return {'rows':len(a),'matching_labels':sum(x==y for x,y in zip(a,b)),'max_probability_drift':max(abs(x.get(label,0)-y.get(label,0)) for x,y in zip(ap,bp) for label in LABELS)}
    files=[{'file':p.name,'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted((ROOT/'models').glob('*.onnx'))]
    output={'created_utc':__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),'platform':platform.platform(),
            'python':platform.python_version(),'cpu_threads':os.environ.get('RADNOTE_CPU_THREADS','2'),'models':files,'results':results,
            'classifier_same_synthetic_text_fidelity':fidelity(synthetic_classifier),'classifier_private_holdout_fidelity':fidelity(private_outputs) if private else None,
            'notes':['Sequential FP32 then INT8, repeated matched synthetic pages, one named host. No speedup or independent clinical accuracy guarantee.',
                     'Supported Conv/MatMul layers are INT8; unsupported operations remain floating point.',
                     'Private holdout metrics are absent unless an authorized CSV was supplied. No raw reports or private row predictions are exported.']}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(output,indent=2),encoding='utf-8')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=Path('results/vm/precision_comparison.json'))
    parser.add_argument('--limit',type=int,default=6);parser.add_argument('--repeats',type=int,default=3);parser.add_argument('--warmups',type=int,default=1)
    parser.add_argument('--classifier-test-csv',type=Path)
    args=parser.parse_args()
    if not 1<=args.limit<=6 or args.repeats<1 or args.warmups<0:parser.error('Use limit 1-6, repeats >=1, warmups >=0.')
    asyncio.run(benchmark(args))
