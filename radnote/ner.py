"""Compact synthetic-trained ONNX token tagger; not clinical NER validation."""
import json
import re
from pathlib import Path
import numpy as np
import onnxruntime as ort

TOKEN=re.compile(r"\w+(?:['-]\w+)*|[^\w\s]",re.UNICODE)

class OnnxNER:
    def __init__(self,directory:Path,threads=2):
        self.metadata=json.loads((directory/"ner_metadata.json").read_text())
        options=ort.SessionOptions();options.intra_op_num_threads=threads;options.inter_op_num_threads=1
        self.session=ort.InferenceSession(str(directory/"ner_int8.onnx"),sess_options=options,providers=["CPUExecutionProvider"])
        self.vocabulary=self.metadata["vocabulary"];self.tags=self.metadata["tags"]
    def extract(self,text):
        # Chunks bound runtime cost for long reports. Padding gives identical
        # boundary behavior to training; context flags are added after tagging.
        tokens=list(TOKEN.finditer(text));entities=[]
        for offset in range(0,len(tokens),256):
            section=tokens[offset:offset+256]
            ids=[self.vocabulary.get(m.group().lower(),1) for m in section]
            array=np.asarray([ids],dtype=np.int64)
            logits=self.session.run(None,{"tokens":array})[0][0]
            tags=[self.tags[int(i)] if token_id!=1 else "O" for i,token_id in zip(logits.argmax(axis=-1),ids)]
            current=None
            for match,tag in zip(section,tags):
                prefix=tag[:1];label=tag[2:] if tag!="O" else None
                if tag=="O":current=None;continue
                if prefix=="I" and current is not None and current["label"]==label:
                    current["end"]=match.end();current["text"]=text[current["start"]:current["end"]]
                else:
                    current={"text":match.group(),"label":label,"start":match.start(),"end":match.end()};entities.append(current)
            for entity in entities:
                if "assertion" in entity:continue
                left=max(text.rfind(".",0,entity["start"]),text.rfind(";",0,entity["start"]))+1
                cue=re.split(r"\b(?:but|however|findings|impression)\b:?",text[left:entity["start"]].lower())[-1]
                entity["assertion"]="negated" if re.search(r"\b(no|without|absent|negative for|no evidence of)\b",cue) else "uncertain" if re.search(r"\b(possible|may|cannot exclude|suspected|questionable)\b",cue) else "historical" if re.search(r"\b(history of|previous|resolved|prior)\b",cue) else "affirmed"
        return entities
