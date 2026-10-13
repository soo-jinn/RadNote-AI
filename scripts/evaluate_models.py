"""Save model sizes, quantized classifier metrics/parity and OCR precision comparison.

No private report text or row-level labels/predictions are written to outputs.
"""
import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
import time
from pathlib import Path
import joblib
import numpy as np
import onnx
import pandas as pd
import onnxruntime as ort
from sklearn.metrics import classification_report,confusion_matrix
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from radnote.inference import OnnxClassifier
from radnote.ocr import ReportOCR
from scripts.benchmark_api import normalize,distance

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--test-csv",type=Path,required=True);args=parser.parse_args()
    frame=pd.read_csv(args.test_csv).fillna("");texts=[f"{a}\n{b}".strip() for a,b in zip(frame.findings,frame.impression)]
    reference=joblib.load("models/urgency_classifier.joblib");expected=reference.predict(texts);probabilities=reference.predict_proba(texts)
    deployed=OnnxClassifier(Path("models/urgency_classifier_int8.onnx"));observed=[];scores=[]
    for text in texts:
        label,values=deployed.predict(text);observed.append(label);scores.append([values[c] for c in reference.classes_])
    labels=["Routine","Urgent","Critical"];matches=sum(a==b for a,b in zip(expected,observed))
    result={"platform":platform.platform(),"classifier":{"held_out_rows":len(texts),"parity_matches":matches,"parity_fraction":matches/len(texts),"max_probability_drift":float(np.abs(probabilities-np.asarray(scores)).max()),"classification_report":classification_report(frame.urgency_label,observed,labels=labels,output_dict=True,zero_division=0),"confusion_matrix":confusion_matrix(frame.urgency_label,observed,labels=labels).tolist(),"confusion_matrix_label_order":labels,"note":"Weak-label agreement only. Critical support is zero; recall not assessable. No private row-level data exported."},"models":[]}
    for path in sorted(Path("models").glob("*.onnx")):
        model=onnx.load(str(path));result["models"].append({"file":path.name,"bytes":path.stat().st_size,"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"int8_weight_tensors":sum(t.data_type==onnx.TensorProto.INT8 for t in model.graph.initializer),"quantize_nodes":sum(n.op_type=="QuantizeLinear" for n in model.graph.node),"dequantize_nodes":sum(n.op_type=="DequantizeLinear" for n in model.graph.node)})
    manifest=json.loads(Path("samples/ocr/manifest.json").read_text());ids={"case_001","case_006","case_021","case_026","case_041","case_046"};subset=[c for c in manifest if c['id'] in ids];comparison=[]
    for precision in ["fp32","int8"]:
        reader=ReportOCR(Path("models/easyocr"),False,2,precision);rows=[]
        for case in subset:
            start=time.perf_counter();text=reader.read((Path("samples/ocr")/case["image"]).read_bytes(),"image/png");elapsed=(time.perf_counter()-start)*1000
            expected_text=normalize(case["reference_text"]);actual=normalize(text)
            rows.append({"id":case['id'],"milliseconds":elapsed,"cer_errors":distance(expected_text,actual),"cer_length":len(expected_text),"wer_errors":distance(expected_text.split(),actual.split()),"wer_length":len(expected_text.split())})
        comparison.append({"precision":precision,"sample_count":len(rows),"cer":sum(r['cer_errors'] for r in rows)/sum(r['cer_length'] for r in rows),"wer":sum(r['wer_errors'] for r in rows)/sum(r['wer_length'] for r in rows),"mean_processing_ms":sum(r['milliseconds'] for r in rows)/len(rows),"rows":rows});del reader
    result["ocr_precision_comparison"]={"results":comparison,"note":"Six fixed synthetic evaluation pages; sequential FP32 then INT8, one warm host. Small paired engineering comparison, not clinical validation or hardware speed guarantee. Calibration pages are separate."}
    result["versions"]={name:importlib.metadata.version(name) for name in ["fastapi","uvicorn","onnxruntime","onnx","easyocr","torch","torchvision","scikit-learn","numpy","Pillow"]};result["python"]=platform.python_version()
    Path("results/model_evaluation.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps({"parity":f"{matches}/{len(texts)}","probability_drift":result['classifier']['max_probability_drift'],"ocr_comparison":comparison},indent=2))
    if matches!=len(texts):raise SystemExit("Quantization changed classifier labels; review model_evaluation.json")

if __name__=="__main__":main()
