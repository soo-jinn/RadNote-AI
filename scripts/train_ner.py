"""Train/export a small BIO tagger on disclosed synthetic lexicon annotations.

Train/validation/test use different sentence families, not the OCR test corpus.
This learns lexicon-guided synthetic labels; it is not expert-annotated clinical NER.
"""
import json
import random
import sys
from pathlib import Path
import numpy as np
import torch
from torch import nn
from onnxruntime.quantization import CalibrationDataReader,quantize_static,QuantFormat,QuantType
from sklearn.metrics import classification_report
import onnxruntime as ort
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from radnote.entities import TERMS,extract_entities
from radnote.ner import TOKEN

class Tagger(nn.Module):
    def __init__(self,vocab_size,tag_count):
        super().__init__();self.embedding=nn.Embedding(vocab_size,32,padding_idx=0);self.conv=nn.Conv1d(32,64,3,padding=1);self.output=nn.Linear(64,tag_count)
    def forward(self,tokens):
        context=torch.relu(self.conv(self.embedding(tokens).transpose(1,2))).transpose(1,2)
        return self.output(context)

def make_records():
    families=["The report describes {term}.","There is {term} in this examination.","No {term} is identified.","Possible {term} is described.","The study demonstrates {term}.","History of {term} is recorded.","Assessment: {term} is present.","Impression: {term} is seen.","Current observations include {term}.","Findings: no evidence of {term}.","Review indicates {term}.","There may be {term} on this study."]
    records=[];rng=random.Random(179);all_terms=[term for terms in TERMS.values() for term in terms]
    for family,template in enumerate(families):
        compound=[rng.choice(all_terms)+rng.choice([" and "," with "," adjacent to ","; "," "])+rng.choice(all_terms) for _ in range(100)]
        for terms in list(TERMS.values())+[compound]:
            for term in terms:
                text=template.format(term=term);matches=list(TOKEN.finditer(text));tags=["O"]*len(matches)
                spans=sorted(extract_entities(text),key=lambda x:-(x["end"]-x["start"]))
                for span in spans:
                    indices=[i for i,m in enumerate(matches) if m.start()>=span["start"] and m.end()<=span["end"]]
                    if indices and all(tags[i]=="O" for i in indices):
                        for j,i in enumerate(indices):tags[i]=("B-" if j==0 else "I-")+span["label"]
                records.append({"family":family,"partition":"train" if family<8 else "validation" if family<10 else "test","text":text,"tokens":[m.group().lower() for m in matches],"tags":tags,"annotation_source":"synthetic lexicon weak annotation"})
        for text in ["The examination is otherwise unremarkable.","Comparison is unavailable.","No additional observations are described."]:
            records.append({"family":family,"partition":"train" if family<8 else "validation" if family<10 else "test","text":text,"tokens":[m.group().lower() for m in TOKEN.finditer(text)],"tags":["O"]*len(list(TOKEN.finditer(text))),"annotation_source":"synthetic negative sentence"})
    return records

class Reader(CalibrationDataReader):
    def __init__(self,values):self.values=iter(values)
    def get_next(self):return next(self.values,None)

def main():
    torch.manual_seed(179);random.seed(179);torch.set_num_threads(2)
    records=make_records();vocabulary={"<pad>":0,"<unk>":1}
    # Only training tokens define vocabulary. Held-out context may be unknown.
    for record in records:
        if record["partition"]=="train":
            for token in record["tokens"]:
                if token not in vocabulary:vocabulary[token]=len(vocabulary)
    tags=["O"]+[f"{prefix}-{label}" for label in TERMS for prefix in ["B","I"]]
    max_length=max(len(r["tokens"]) for r in records)
    def arrays(partition):
        subset=[r for r in records if r["partition"]==partition];x=np.zeros((len(subset),max_length),dtype=np.int64);y=np.full_like(x,-100)
        for i,r in enumerate(subset):
            x[i,:len(r["tokens"])]=[vocabulary.get(t,1) for t in r["tokens"]];y[i,:len(r["tags"])]=[tags.index(t) for t in r["tags"]]
        return x,y
    x,y=arrays("train");model=Tagger(len(vocabulary),len(tags));optimizer=torch.optim.Adam(model.parameters(),lr=.008);loss_fn=nn.CrossEntropyLoss(ignore_index=-100)
    best=None;best_loss=float("inf");vx,vy=arrays("validation")
    for epoch in range(80):
        model.train();order=np.random.default_rng(179+epoch).permutation(len(x))
        for start in range(0,len(order),64):
            batch=order[start:start+64];optimizer.zero_grad();loss=loss_fn(model(torch.from_numpy(x[batch])).reshape(-1,len(tags)),torch.from_numpy(y[batch]).reshape(-1));loss.backward();optimizer.step()
        model.eval()
        with torch.no_grad():validation=float(loss_fn(model(torch.from_numpy(vx)).reshape(-1,len(tags)),torch.from_numpy(vy).reshape(-1)))
        if validation<best_loss:best_loss=validation;best={k:v.detach().clone() for k,v in model.state_dict().items()}
    model.load_state_dict(best);model.eval();out=Path("models");out.mkdir(exist_ok=True)
    torch.save(model.state_dict(),out/"ner_synthetic.pt")
    torch.onnx.export(model,torch.from_numpy(x[:1]),str(out/"ner_fp32.onnx"),input_names=["tokens"],output_names=["logits"],dynamic_axes={"tokens":{0:"batch",1:"length"},"logits":{0:"batch",1:"length"}},opset_version=17,dynamo=False)
    quantize_static(str(out/"ner_fp32.onnx"),str(out/"ner_int8.onnx"),Reader([{"tokens":row[None]} for row in x[:40]]),quant_format=QuantFormat.QDQ,activation_type=QuantType.QUInt8,weight_type=QuantType.QInt8,op_types_to_quantize=["Conv","MatMul"],per_channel=True)
    tx,ty=arrays("test");session=ort.InferenceSession(str(out/"ner_int8.onnx"),providers=["CPUExecutionProvider"]);prediction=session.run(None,{"tokens":tx})[0].argmax(-1);prediction[tx==1]=0;mask=ty!=-100
    report=classification_report(ty[mask],prediction[mask],labels=list(range(len(tags))),target_names=tags,output_dict=True,zero_division=0)
    metadata={"architecture":"Embedding32 + Conv1D64 + BIO token linear head","vocabulary":vocabulary,"tags":tags,"validation_status":"synthetic_weak_labels_only","epoch_limit":80,"best_validation_loss":best_loss,"partitions":{p:sum(r['partition']==p for r in records) for p in ['train','validation','test']},"calibration":"40 training sentences; held-out sentences excluded","quantization":"static INT8 QDQ Conv/MatMul; embedding float32","test_token_metrics":report,"limitations":"Template-family holdout of lexicon-generated annotations. Not clinical/entity-span validation; unknown tokens forced O at runtime."}
    (out/"ner_metadata.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    Path("samples/ner").mkdir(parents=True,exist_ok=True);Path("samples/ner/annotations.json").write_text(json.dumps(records,indent=2),encoding="utf-8")
    Path("results/ner_synthetic_metrics.json").write_text(json.dumps({k:v for k,v in metadata.items() if k not in ['vocabulary','tags']},indent=2),encoding="utf-8")
    print(json.dumps({"partitions":metadata['partitions'],"token_macro_f1":report['macro avg']['f1-score'],"validation_loss":best_loss},indent=2))

if __name__=="__main__":main()
