"""Compare direct pipeline execution with client-server HTTP on identical inputs."""
import argparse
import asyncio
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path
import httpx
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.benchmark_api import percentile

ROOT=Path(__file__).resolve().parents[1]

async def run(args):
    manifest=json.loads((ROOT/'samples/ocr/manifest.json').read_text())
    cases=[r for r in manifest if r['id'] in {'case_001','case_021','case_041'}]
    rows=[];model_health=None
    def save(case,kind,trial,start,result,content):
        rows.append({'case':case['id'],'kind':kind,'trial':trial,'elapsed_ms':(time.perf_counter()-start)*1000,
                     'input_sha256':hashlib.sha256(content).hexdigest(),'urgency':result['urgency'],
                     'extracted_text_sha256':hashlib.sha256((result.get('extracted_text') or '').encode()).hexdigest(),
                     'server_processing_ms':result.get('latency_ms'),'model_sha256':result['model_sha256']})
    if args.mode=='direct':
        from api.main import app,lifespan,health,image_result,result_for
        os.environ['RADNOTE_ENABLE_OCR']='1';os.environ['RADNOTE_MODEL_PRECISION']='int8';os.environ.pop('RADNOTE_MODEL_PATH',None)
        async with lifespan(app):
            model_health=health()
            if model_health['status']!='ok':raise RuntimeError(model_health)
            for case in cases:
                payload=json.dumps({'findings':case['findings'],'impression':case['impression']},sort_keys=True).encode()
                image=(ROOT/'samples/ocr'/case['image']).read_bytes()
                for kind,content in [('text',payload),('image',image)]:
                    for trial in range(args.warmups+args.repeats):
                        begin=time.perf_counter()
                        result=result_for(case['findings'],case['impression'],started=begin) if kind=='text' else image_result(content,'image/png',begin)
                        if trial>=args.warmups:save(case,kind,trial-args.warmups+1,begin,result,content)
    else:
        with httpx.Client(timeout=180) as client:
            model_health=client.get(args.server+'/health').json()
            if model_health['status']!='ok' or model_health.get('precision')!='int8':raise RuntimeError('HTTP comparison requires a healthy INT8 backend.')
            for case in cases:
                value={'findings':case['findings'],'impression':case['impression']};payload=json.dumps(value,sort_keys=True).encode()
                image=(ROOT/'samples/ocr'/case['image']).read_bytes()
                for kind,content in [('text',payload),('image',image)]:
                    for trial in range(args.warmups+args.repeats):
                        begin=time.perf_counter()
                        response=client.post(args.server+'/predict',json=value) if kind=='text' else client.post(args.server+'/predict-image',files={'file':(case['id']+'.png',content,'image/png')})
                        response.raise_for_status();result=response.json()
                        if trial>=args.warmups:save(case,kind,trial-args.warmups+1,begin,result,content)
    summary={kind:{'trials':len(values),'mean_ms':sum(values)/len(values),'p50_ms':percentile(values,.5),'p95_ms':percentile(values,.95)} for kind in ['text','image'] if (values:=[r['elapsed_ms'] for r in rows if r['kind']==kind])}
    result={'mode':args.mode,'server':args.server if args.mode=='http' else None,'platform':platform.platform(),'health':model_health,
            'warmups_per_case_kind':args.warmups,'repeats':args.repeats,'summary':summary,'rows':rows,
            'scope':'Direct in-process pipeline versus HTTP through the client-server route, on matching synthetic inputs. Both may run in the same Ubuntu VM; Docker bridge overhead is not a physical LAN measurement.'}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(summary,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=['direct','http'],required=True);p.add_argument('--server',default='http://client:8765')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--repeats',type=int,default=3);p.add_argument('--warmups',type=int,default=1);args=p.parse_args()
    if args.repeats<1 or args.warmups<0:p.error('Use repeats >=1 and warmups >=0.')
    asyncio.run(run(args))
