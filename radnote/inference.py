"""CPU-only ONNX classifier. Results carry the supplied model's evaluation status."""
import hashlib
import json
from pathlib import Path
import numpy as np
import onnxruntime as ort

class OnnxClassifier:
    def __init__(self, model_path: Path, threads: int = 2):
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(model_path), sess_options=options, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.sha256 = hashlib.sha256(model_path.read_bytes()).hexdigest()
        metadata_path = model_path.parent / "metadata.json"
        self.metadata = json.loads(metadata_path.read_text()) if metadata_path.is_file() else {"validation_status": "unvalidated"}

    def predict(self, text: str) -> tuple[str, dict[str, float]]:
        labels, scores, *_ = self.session.run(None, {self.input_name: np.asarray([[text]], dtype=object)})
        urgency = str(labels[0])
        if isinstance(scores, list) and scores and isinstance(scores[0], dict):
            probabilities = {str(k): float(v) for k, v in scores[0].items()}
        else:
            classes = self.metadata.get("class_order")
            if not classes:
                raise RuntimeError("Dense ONNX scores require class_order in models/metadata.json")
            probabilities = {label: float(score) for label, score in zip(classes, np.asarray(scores)[0])}
        return urgency, probabilities
