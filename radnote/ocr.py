"""CPU OCR with bounded input dimensions and explicit model cache location."""
import io
import threading
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

MAX_BYTES = 8_000_000
MAX_PIXELS = 16_000_000

class ImageInputError(ValueError):
    pass

def decode_image(content: bytes, mime: str) -> np.ndarray:
    if mime not in {"image/png", "image/jpeg"}:
        raise ImageInputError("Upload a PNG or JPEG synthetic report page.")
    if not content or len(content) > MAX_BYTES:
        raise ImageInputError("The image must be nonempty and at most 8 MB.")
    try:
        with Image.open(io.BytesIO(content)) as source:
            if source.format not in {"PNG", "JPEG"}:
                raise ImageInputError("The decoded file must be PNG or JPEG.")
            if source.width * source.height > MAX_PIXELS:
                raise ImageInputError("The image exceeds the 16 megapixel limit.")
            image = ImageOps.exif_transpose(source).convert("RGB")
            image.thumbnail((1800, 1800))
            return np.asarray(image)
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
        raise ImageInputError("The image cannot be decoded.") from error

class ReportOCR:
    def __init__(self, model_dir, allow_download: bool, threads: int, precision: str = "int8"):
        import torch
        import easyocr
        torch.set_num_threads(threads)
        from pathlib import Path
        import onnxruntime as ort
        from easyocr.detection import get_textbox
        from easyocr.utils import CTCLabelConverter
        directory=Path(model_dir).parent
        options=ort.SessionOptions();options.intra_op_num_threads=threads;options.inter_op_num_threads=1
        # EasyOCR supplies image geometry and CTC decoding. Both neural forward
        # passes use ONNX Runtime CPU and the statically calibrated INT8 graphs.
        class Network:
            def __init__(self,path,detector=False):
                self.session=ort.InferenceSession(str(path),sess_options=options,providers=["CPUExecutionProvider"]);self.detector=detector
            def eval(self):return self
            def __call__(self,image,text=None):
                output=self.session.run(None,{"image":image.detach().cpu().numpy()})[0]
                value=torch.from_numpy(output)
                return (value,None) if self.detector else value
        self.reader = easyocr.Reader(["en"], gpu=False, detector=False, recognizer=False, quantize=False, model_storage_directory=str(model_dir), download_enabled=allow_download, verbose=False)
        self.reader.get_textbox=get_textbox
        if precision not in {"int8","fp32"}:raise ValueError("Unsupported OCR precision")
        self.reader.detector=Network(directory/f"ocr_detector_{precision}.onnx",True)
        self.reader.recognizer=Network(directory/f"ocr_recognizer_{precision}.onnx")
        self.reader.converter=CTCLabelConverter(self.reader.character,{}, {})
        self.backend=f"ONNX Runtime CPU / {precision} Conv-MatMul"
        self.lock = threading.Lock()

    def read(self, content: bytes, mime: str) -> str:
        pixels = decode_image(content, mime)
        with self.lock:
            spans = self.reader.readtext(pixels, detail=1, paragraph=False, workers=0)
        # Detection can split a short cue ("No") into a separate box and place
        # it after the main paragraph. Reconstruct single-column page order
        # from geometry so that the cue precedes its finding.
        boxes=[]
        for points,text,confidence in spans:
            xs=[float(p[0]) for p in points];ys=[float(p[1]) for p in points]
            boxes.append((sum(ys)/len(ys),min(xs),max(ys)-min(ys),text))
        boxes.sort(key=lambda box:(box[0],box[1]));lines=[];current=[];center=height=0
        for y,x,h,text in boxes:
            if current and abs(y-center)>max(height*.6,h*.6,6):
                lines.append(" ".join(t for _,t in sorted(current)));current=[]
            if not current:center=y;height=h
            current.append((x,text))
        if current:lines.append(" ".join(t for _,t in sorted(current)))
        return "\n".join(lines).strip()
