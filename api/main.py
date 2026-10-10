"""ONNX-backed report text and synthetic report-page inference API."""
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, model_validator
from starlette.concurrency import run_in_threadpool
from radnote.text import clean_text, combine, parse_report
from radnote.ner import OnnxNER
from radnote.inference import OnnxClassifier
from radnote.ocr import ReportOCR, ImageInputError, MAX_BYTES

ROOT = Path(__file__).resolve().parents[1]
logger = logging.getLogger("radnote")

class ReportTextRequest(BaseModel):
    findings: str = Field(default="", max_length=20000)
    impression: str = Field(default="", max_length=10000)
    @model_validator(mode="after")
    def nonempty(self):
        if not combine(self.findings, self.impression):
            raise ValueError("Supply Findings or Impression text.")
        return self

@asynccontextmanager
async def lifespan(app: FastAPI):
    threads = max(1, min(8, int(os.getenv("RADNOTE_CPU_THREADS", "2"))))
    model_path = Path(os.getenv("RADNOTE_MODEL_PATH", str(ROOT / "models/urgency_classifier_int8.onnx")))
    app.state.classifier = OnnxClassifier(model_path, threads)
    app.state.ner = OnnxNER(ROOT / "models", threads)
    app.state.ocr = None
    app.state.ocr_error = None
    app.state.ocr_requested = os.getenv("RADNOTE_ENABLE_OCR", "1") == "1"
    app.state.threshold = float(os.getenv("RADNOTE_REVIEW_THRESHOLD", "0.70"))
    if not 0 <= app.state.threshold <= 1:
        raise ValueError("RADNOTE_REVIEW_THRESHOLD must be between 0 and 1")
    if app.state.ocr_requested:
        try:
            app.state.ocr = ReportOCR(Path(os.getenv("RADNOTE_OCR_MODEL_DIR", str(ROOT / "models/easyocr"))), os.getenv("RADNOTE_ALLOW_MODEL_DOWNLOAD", "0") == "1", threads)
        except Exception as error:
            logger.warning("OCR initialization failed: %s", type(error).__name__)
            app.state.ocr_error = "OCR models or dependencies unavailable. Install the ocr extra and run scripts/cache_ocr.py."
    yield
    app.state.ocr = None
    app.state.classifier = None

app = FastAPI(title="RadNote-AI", version="1.0.0", description="Academic report-text prototype. Image input is for synthetic report pages only. Radiographs are excluded.", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=ROOT / "web"), name="static")
app.mount("/samples", StaticFiles(directory=ROOT / "samples/ocr/images"), name="samples")

@app.get("/")
def home():
    return FileResponse(ROOT / "web/index.html")

@app.get("/health")
def health():
    classifier = app.state.classifier
    ready = classifier is not None and (not app.state.ocr_requested or app.state.ocr is not None)
    return {"status": "ok" if ready else "degraded", "classifier_ready": classifier is not None, "ocr_ready": app.state.ocr is not None, "ner_ready": app.state.ner is not None, "ner_backend":"ONNX Runtime / synthetic-trained BIO tagger / static INT8 Conv-MatMul", "ocr_backend":getattr(app.state.ocr,"backend",None), "classifier_backend": "ONNX Runtime", "providers": classifier.session.get_providers() if classifier else [], "validation_status": classifier.metadata.get("validation_status") if classifier else None, "ocr_error": app.state.ocr_error, "version": "1.0.0"}

@app.get("/ready")
def readiness():
    result = health()
    return JSONResponse(result, status_code=200 if result["status"] == "ok" else 503)

def result_for(findings, impression, extracted_text=None, warnings=None, started=None):
    classifier = app.state.classifier
    if classifier is None:
        raise HTTPException(503, "Classifier unavailable.")
    text = combine(findings, impression)
    if not text or len(text) > 30000:
        raise HTTPException(422, "Report text must contain 1 to 30000 characters.")
    urgency, probabilities = classifier.predict(text)
    score = probabilities.get(urgency, 0.0)
    reasons = list(warnings or [])
    if classifier.metadata.get("validation_status") != "validated":
        reasons.append("Model validation incomplete: very limited Urgent support and no Critical held-out cases.")
    if score < app.state.threshold:
        reasons.append("Classifier score is below the configured review threshold.")
    return {"request_id": str(uuid.uuid4()), "urgency": urgency, "critical_alert": urgency == "Critical", "probabilities": probabilities, "review_required": bool(reasons), "review_reasons": reasons, "findings": clean_text(findings), "impression": clean_text(impression), "entities": app.state.ner.extract(text), "extracted_text": extracted_text, "latency_ms": round((time.perf_counter() - started) * 1000, 2) if started else None, "model_sha256": classifier.sha256, "note": "Academic prototype. Urgency uses weak labels; NER uses synthetic weak annotations. Scores are not calibrated confidence. Review every result."}

@app.post("/predict")
def predict(request: ReportTextRequest):
    started = time.perf_counter()
    return result_for(request.findings, request.impression, started=started)

def image_result(content, mime, started):
    try:
        extracted = app.state.ocr.read(content, mime)
    except ImageInputError as error:
        raise HTTPException(400, str(error)) from error
    except Exception as error:
        logger.error("OCR failed: %s", type(error).__name__)
        raise HTTPException(500, "OCR failed. Check the server environment; no uploaded data was saved.") from error
    if not extracted:
        raise HTTPException(422, "No report text was extracted.")
    findings, impression, warnings = parse_report(extracted)
    return result_for(findings, impression, extracted, warnings, started)

@app.post("/predict-image")
async def predict_image(file: UploadFile = File(...)):
    started = time.perf_counter()
    if app.state.ocr is None:
        raise HTTPException(503, app.state.ocr_error or "OCR is disabled.")
    if file.content_type not in {"image/png", "image/jpeg"}:
        raise HTTPException(415, "Upload a PNG or JPEG synthetic report page.")
    content = await file.read(MAX_BYTES + 1)
    await file.close()
    if len(content) > MAX_BYTES:
        raise HTTPException(413, "Image exceeds 8 MB.")
    return await run_in_threadpool(image_result, content, file.content_type, started)
