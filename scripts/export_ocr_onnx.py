"""Export pretrained EasyOCR networks and static INT8 Conv/MatMul variants.

Calibration uses separate original rendered text pages, not the 60 evaluation
pages or Open-I reports. LSTM and postprocessing retain their native types.
"""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
import torch
from torch import nn
import easyocr
import onnx
from onnxruntime.quantization import CalibrationDataReader, quantize_static, QuantFormat, QuantType
from PIL import Image, ImageDraw, ImageFont
from easyocr.imgproc import resize_aspect_ratio, normalizeMeanVariance
from easyocr.recognition import AlignCollate

class Recognition(nn.Module):
    def __init__(self, model):
        super().__init__(); self.model=model
    def forward(self, image):
        visual=self.model.FeatureExtraction(image)
        # Height is 1 after the VGG feature extractor; avoid dynamic adaptive pooling.
        visual=visual.mean(dim=2).permute(0,2,1)
        return self.model.Prediction(self.model.SequenceModeling(visual).contiguous())

class Detector(nn.Module):
    def __init__(self, model):
        super().__init__();self.model=model
    def forward(self, image):
        return self.model(image)[0]

class Reader(CalibrationDataReader):
    def __init__(self, values):self.values=iter(values)
    def get_next(self):return next(self.values,None)

def calibrations():
    fonts=["C:/Windows/Fonts/arial.ttf","/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]
    font_path=next((p for p in fonts if Path(p).exists()),None)
    sentences=[
        "FINDINGS: Bilateral lungs are expanded and heart contours are regular.",
        "IMPRESSION: A focal right basilar density warrants further assessment.",
        "FINDINGS: Left-sided fluid with adjacent compressive change is visible.",
        "IMPRESSION: New marked right pleural air collection needs prompt attention.",
        "FINDINGS: The aortic contour is prominent; no acute skeletal abnormality.",
        "IMPRESSION: Stable study compared with the preceding examination."
    ]
    detect=[];recognize=[]
    for i,sentence in enumerate(sentences):
        font=ImageFont.truetype(font_path,24+i%3) if font_path else ImageFont.load_default(size=25)
        page=Image.new("RGB",(1000,700),"white");draw=ImageDraw.Draw(page)
        draw.text((40,55),sentence[:58],font=font,fill="black")
        draw.text((40,105),sentence[58:],font=font,fill="black")
        page=page.rotate((i%3)-1,fillcolor="white")
        resized,*_=resize_aspect_ratio(np.asarray(page),2560,cv2.INTER_LINEAR,mag_ratio=1.)
        detect.append({"image":np.transpose(normalizeMeanVariance(resized),(2,0,1))[None].astype(np.float32)})
        line=Image.new("L",(1000,50),255);ImageDraw.Draw(line).text((5,5),sentence,font=font,fill=0)
        recognize.append({"image":AlignCollate(imgH=64,imgW=1000)([line]).numpy()})
    return detect,recognize

def export(output,cache):
    torch.set_num_threads(2);output.mkdir(parents=True,exist_ok=True)
    reader=easyocr.Reader(["en"],gpu=False,quantize=False,model_storage_directory=str(cache),download_enabled=False,verbose=False)
    models=[("ocr_detector",Detector(reader.detector).eval(),torch.zeros(1,3,704,1024),{0:"batch",2:"height",3:"width"},{0:"batch",1:"out_height",2:"out_width"}),
            ("ocr_recognizer",Recognition(reader.recognizer).eval(),torch.zeros(1,1,64,512),{0:"batch",3:"width"},{0:"batch",1:"steps"})]
    calibration=calibrations();report=[]
    for index,(name,model,dummy,input_axes,output_axes) in enumerate(models):
        floating=output/(name+"_fp32.onnx");quantized=output/(name+"_int8.onnx")
        print("Exporting "+name,flush=True)
        torch.onnx.export(model,dummy,str(floating),input_names=["image"],output_names=["output"],dynamic_axes={"image":input_axes,"output":output_axes},opset_version=17,dynamo=False)
        onnx.checker.check_model(str(floating))
        quantize_static(str(floating),str(quantized),Reader(calibration[index]),quant_format=QuantFormat.QDQ,activation_type=QuantType.QUInt8,weight_type=QuantType.QInt8,op_types_to_quantize=["Conv","MatMul"],per_channel=True)
        onnx.checker.check_model(str(quantized))
        graph=onnx.load(str(quantized));integer_weights=sum(x.data_type==onnx.TensorProto.INT8 for x in graph.graph.initializer)
        report.append({"name":name,"fp32_bytes":floating.stat().st_size,"int8_bytes":quantized.stat().st_size,"int8_initializers":integer_weights,"calibration_examples":6,"calibration_source":"Original separate rendered sentences; no evaluation pages or Open-I rows","precision":"QDQ static INT8 Conv/MatMul; LSTM and other operators remain floating point"})
        print(name+" exported and quantized",flush=True)
    (output/"ocr_export.json").write_text(json.dumps(report,indent=2),encoding="utf-8")

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--output",type=Path,default=Path("models"));parser.add_argument("--cache",type=Path,default=Path("models/easyocr"))
    args=parser.parse_args();export(args.output,args.cache)
