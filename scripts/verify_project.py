"""Exercise live server, real OCR, lightweight CLI, and browser proxy; save evidence."""
import argparse
import json
import platform
import sys
from pathlib import Path
import httpx
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from client.pi_client import upload

if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server",default="http://127.0.0.1:8014")
    parser.add_argument("--proxy",default="http://127.0.0.1:8765")
    parser.add_argument("--output",type=Path,default=Path("results/integration.json"))
    args=parser.parse_args();checks={};responses={}
    with httpx.Client(timeout=180) as client:
        health=client.get(args.server+"/health").json()
        checks["all_components_ready"]=health["status"]=="ok" and health["ocr_ready"] and health["classifier_ready"] and health["ner_ready"]
        checks["onnx_cpu_backend"]=health["classifier_backend"]=="ONNX Runtime" and health["providers"]==["CPUExecutionProvider"]
        checks["direct_ui"]=client.get(args.server+"/").status_code==200
        checks["proxy_ui"]=client.get(args.proxy+"/").status_code==200
        checks["proxy_health"]=client.get(args.proxy+"/health").json()==health
        text=client.post(args.proxy+"/predict",json={"findings":"Lungs clear.","impression":"No acute disease."})
        checks["proxy_text"]=text.status_code==200
        checks["review_flag"]=text.json()["review_required"]
        for case in ["case_001","case_021","case_041"]:
            image=Path("samples/ocr/images")/(case+".png")
            result=upload(args.server,image)
            with image.open("rb") as stream:
                proxy=client.post(args.proxy+"/predict-image",files={"file":(image.name,stream,"image/png")})
            checks[case+"_real_ocr"]=bool(result.get("extracted_text"))
            checks[case+"_proxy_same_result"]=proxy.status_code==200 and proxy.json()["extracted_text"]==result["extracted_text"] and proxy.json()["urgency"]==result["urgency"]
            responses[case]=result
    evidence={"platform":platform.platform(),"hardware_scope":"Laptop simulation of Pi client; Raspberry Pi hardware not tested", "health":health,"checks":checks,"responses":responses,"passed":all(checks.values())}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(evidence,indent=2),encoding="utf-8")
    print(json.dumps(checks,indent=2))
    if not evidence["passed"]:
        raise SystemExit("Live integration check failed")
