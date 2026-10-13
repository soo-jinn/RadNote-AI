"""Check transfer completeness and model hashes before installing dependencies."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def main():
    required=['Dockerfile','docker-compose.yml','requirements-runtime.txt','api/main.py','client/web_client.py','client/report_client.py','web/index.html','models/model_manifest.json','samples/ocr/manifest.json','samples/ner/annotations.json']
    missing=[p for p in required if not (ROOT/p).is_file()]
    if missing:raise SystemExit('Missing project files: '+', '.join(missing))
    manifest=json.loads((ROOT/'models/model_manifest.json').read_text())
    for row in manifest['models']:
        path=ROOT/'models'/row['file']
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']:raise SystemExit('Model missing or corrupted: '+row['file'])
    cases=json.loads((ROOT/'samples/ocr/manifest.json').read_text())
    if len(cases)!=60 or any(not (ROOT/'samples/ocr'/r['image']).is_file() for r in cases):raise SystemExit('Synthetic report-page sample set is incomplete.')
    print(f'Transfer checks passed: {len(manifest["models"])} verified ONNX models, 60 synthetic pages, API/client/configuration present.')

if __name__=='__main__':main()
