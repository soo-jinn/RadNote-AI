"""Convert the fitted LinearClassifier head to MatMul and statically quantize it.

Only private TRAIN partition calibrates activation ranges; do not use test rows.
The TF-IDF/string operators remain floating point/native, not INT8.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import onnx
from onnx import helper,numpy_helper
from onnxruntime.quantization import CalibrationDataReader,quantize_static,QuantFormat,QuantType

class Reader(CalibrationDataReader):
    def __init__(self,name,texts):self.name=name;self.values=iter(texts)
    def get_next(self):
        value=next(self.values,None)
        return {self.name:np.asarray([[value]],dtype=object)} if value is not None else None

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--calibration-csv",type=Path,required=True);args=parser.parse_args()
    source=Path("models/urgency_classifier.onnx");model=onnx.load(str(source));nodes=[];classes=None
    for node in model.graph.node:
        if node.op_type!="LinearClassifier":nodes.append(node);continue
        attrs={a.name:helper.get_attribute_value(a) for a in node.attribute};classes=[v.decode() for v in attrs["classlabels_strings"]]
        if attrs.get("post_transform")!=b"SOFTMAX":raise ValueError("Expected fitted multinomial softmax classifier")
        weights=np.asarray(attrs["coefficients"],dtype=np.float32).reshape(len(classes),-1).T.copy();bias=np.asarray(attrs["intercepts"],dtype=np.float32)
        model.graph.value_info.extend([helper.make_tensor_value_info(node.input[0],onnx.TensorProto.FLOAT,[None,weights.shape[0]]),helper.make_tensor_value_info("head_linear",onnx.TensorProto.FLOAT,[None,len(classes)]),helper.make_tensor_value_info("head_logits",onnx.TensorProto.FLOAT,[None,len(classes)])])
        model.graph.initializer.extend([numpy_helper.from_array(weights,"classifier_weights"),numpy_helper.from_array(bias,"classifier_bias"),numpy_helper.from_array(np.asarray(classes,dtype=object),"class_labels")])
        nodes.extend([helper.make_node("MatMul",[node.input[0],"classifier_weights"],["head_linear"],name="urgency_matmul"),helper.make_node("Add",["head_linear","classifier_bias"],["head_logits"]),helper.make_node("Softmax",["head_logits"],[node.output[1]],axis=1),helper.make_node("ArgMax",["head_logits"],["head_indices"],axis=1,keepdims=0),helper.make_node("Gather",["class_labels","head_indices"],[node.output[0]],axis=0)])
    if classes is None:raise ValueError("No LinearClassifier head found")
    del model.graph.node[:];model.graph.node.extend(nodes);floating=Path("models/urgency_classifier_matmul_fp32.onnx");onnx.checker.check_model(model);onnx.save(model,str(floating))
    frame=pd.read_csv(args.calibration_csv).fillna("")
    texts=[f"{a}\n{b}".strip() for a,b in zip(frame.findings,frame.impression)]
    target=Path("models/urgency_classifier_int8.onnx")
    quantize_static(str(floating),str(target),Reader(model.graph.input[0].name,texts),quant_format=QuantFormat.QDQ,activation_type=QuantType.QUInt8,weight_type=QuantType.QInt8,op_types_to_quantize=["MatMul"],per_channel=True)
    onnx.checker.check_model(str(target))
    quantized=onnx.load(str(target))
    if not any(t.data_type==onnx.TensorProto.INT8 for t in quantized.graph.initializer):raise RuntimeError("Quantization produced no INT8 weights")
    metadata=json.loads(Path("models/metadata.json").read_text());metadata.update({"class_order":classes,"model_format":"Static INT8 QDQ classifier MatMul; TF-IDF/preprocessing remain native/float32","calibration_rows":len(texts),"calibration_partition":"training only; raw rows not packaged"})
    Path("models/metadata.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    print(f"Quantized classifier head with {len(texts)} training calibration rows.")

if __name__=="__main__":main()
