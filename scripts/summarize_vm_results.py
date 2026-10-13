"""Create a factual benchmark summary and verify matched comparison inputs."""
import argparse
import json
from pathlib import Path

def summarize(directory):
    data={name:json.loads((directory/name).read_text()) for name in ['precision_comparison.json','local_execution.json','client_server_execution.json','integration.json','benchmark_summary.json','load_sustained.json']}
    direct=data['local_execution.json'];remote=data['client_server_execution.json']
    key=lambda row:(row['case'],row['kind'],row['trial'])
    left={key(r):r for r in direct['rows']};right={key(r):r for r in remote['rows']}
    matched=set(left)==set(right) and all(left[k]['input_sha256']==right[k]['input_sha256'] and left[k]['model_sha256']==right[k]['model_sha256'] for k in left)
    output_agreement=matched and all(left[k]['urgency']==right[k]['urgency'] and left[k]['extracted_text_sha256']==right[k]['extracted_text_sha256'] for k in left)
    integration=data['integration.json']['passed'];ocr=data['benchmark_summary.json'];load=data['load_sustained.json']
    checks={'same_cases_inputs_and_classifier':matched,'same_direct_http_predictions_and_ocr':output_agreement,'live_integration':integration,'sixty_page_uploads':ocr['successes']==60 and ocr['failures']==0,'sustained_load_no_failures':all(r['failures']==0 for r in load['load'].values())}
    stats_path=directory/'docker_stats.jsonl'
    stats=[json.loads(line) for line in stats_path.read_text().splitlines() if line.strip()] if stats_path.is_file() else []
    checks['backend_and_client_cpu_memory_records']=all(any(component in r.get('Name','') and r.get('CPUPerc') and r.get('MemUsage') for r in stats) for component in ['backend','client'])
    readiness={'technical_checks':checks,'all_technical_checks_passed':all(checks.values()),'scope':'These runtime checks do not establish clinical validity or complete human submission evidence.'}
    (directory/'verification_summary.json').write_text(json.dumps(readiness,indent=2),encoding='utf-8')
    lines=['# Ubuntu VM benchmark results','', 'Generated from actual files in this evidence directory.','', '## Runtime checks','']+[f'- {name}: {value}' for name,value in checks.items()]
    lines+=['','## Local versus client-server latency','', '| Input | Direct mean (ms) | HTTP mean (ms) | Direct p95 (ms) | HTTP p95 (ms) |','| --- | ---: | ---: | ---: | ---: |']
    for kind in ['text','image']:
        a=direct['summary'][kind];b=remote['summary'][kind];lines.append(f"| {kind} | {a['mean_ms']:.3f} | {b['mean_ms']:.3f} | {a['p95_ms']:.3f} | {b['p95_ms']:.3f} |")
    lines+=['','## FP32 versus supported-layer INT8','', '| Precision | Mean pipeline (ms) | OCR CER | OCR WER | Sampled RSS (MB) |','| --- | ---: | ---: | ---: | ---: |']
    for row in data['precision_comparison.json']['results']:
        lines.append(f"| {row['precision']} | {row['mean_pipeline_ms']:.3f} | {row['ocr_cer']:.5f} | {row['ocr_wer']:.5f} | {row['max_sampled_rss_mb']:.1f} |")
    lines+=['','INT8 applies to supported layers. OCR and NER labels are synthetic. Classifier rare-class support and weak-label limitations remain.','', 'Both routes run in the declared VM setup. These results do not describe Raspberry Pi hardware or an independent physical network.']
    (directory/'VM_Benchmark_Summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(readiness,indent=2))
    if not readiness['all_technical_checks_passed']:raise SystemExit('Some runtime checks failed. Review evidence before recording the final demonstration.')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',type=Path,default=Path('results/vm'));summarize(p.parse_args().directory)
