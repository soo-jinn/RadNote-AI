"""Sprint 1 API draft for Carlo's review and adaptation."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path
from typing import Any

import joblib
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field, model_validator


class ReportTextRequest(BaseModel):
    findings: str = Field(default="", max_length=20000)
    impression: str = Field(default="", max_length=10000)

    @model_validator(mode="after")
    def require_report_text(self) -> "ReportTextRequest":
        if not (self.findings.strip() or self.impression.strip()):
            raise ValueError("At least one of findings or impression must contain text")
        return self


class PredictionResponse(BaseModel):
    urgency: str
    critical_alert: bool
    probabilities: dict[str, float]
    entities: list[dict[str, Any]] = Field(default_factory=list)
    extracted_text: str | None = None
    note: str


def _load_model() -> Any | None:
    configured_path = os.getenv("RADNOTE_MODEL_PATH")
    if not configured_path:
        return None
    model_path = Path(configured_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"RADNOTE_MODEL_PATH does not exist: {model_path}")
    return joblib.load(model_path)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.classifier = _load_model()
    app.state.ocr_reader = None
    app.state.ner = None
    if os.getenv("RADNOTE_ENABLE_OCR") == "1":
        try:
            import easyocr
            app.state.ocr_reader = easyocr.Reader(["en"], gpu=False)
        except ImportError as error:
            raise RuntimeError("Install the optional OCR dependencies to enable report-image upload") from error
    try:
        from scripts.extract_entities import build_pipeline
        app.state.ner = build_pipeline()
    except ImportError:
        pass
    yield
    app.state.classifier = None
    app.state.ocr_reader = None
    app.state.ner = None


app = FastAPI(
    title="RadNote-AI Sprint 1 API draft",
    description="Report text classifier. Optional OCR accepts synthetic report-page images only; radiographs are out of scope.",
    version="0.2.0-draft",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "classifier_ready": getattr(app.state, "classifier", None) is not None,
        "ocr_ready": getattr(app.state, "ocr_reader", None) is not None,
        "ner_ready": getattr(app.state, "ner", None) is not None,
    }


def _predict_text(classifier: Any, text: str, extracted_text: str | None = None) -> PredictionResponse:
    urgency = str(classifier.predict([text])[0])
    class_names = [str(label) for label in classifier.named_steps["classifier"].classes_]
    scores = classifier.predict_proba([text])[0]
    probabilities = {label: float(score) for label, score in zip(class_names, scores)}
    ner = getattr(app.state, "ner", None)
    entities = []
    if ner is not None:
        entities = [
            {"text": ent.text, "label": ent.label_, "start": ent.start_char, "end": ent.end_char}
            for ent in ner(text).ents
        ]
    return PredictionResponse(
        urgency=urgency,
        critical_alert=urgency == "Critical",
        probabilities=probabilities,
        entities=entities,
        extracted_text=extracted_text,
        note="Prototype output from weakly labeled report text; not a diagnosis. Scores may be uncalibrated.",
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(request: ReportTextRequest) -> PredictionResponse:
    classifier = getattr(app.state, "classifier", None)
    if classifier is None:
        raise HTTPException(status_code=503, detail="Train a classifier and set RADNOTE_MODEL_PATH")
    text = f"{request.findings.strip()}\n{request.impression.strip()}".strip()
    return _predict_text(classifier, text)


@app.post("/predict-image", response_model=PredictionResponse)
async def predict_synthetic_report_image(file: UploadFile = File(...)) -> PredictionResponse:
    """OCR a synthetic report-page image; radiograph interpretation is out of scope."""
    reader = getattr(app.state, "ocr_reader", None)
    if reader is None:
        raise HTTPException(status_code=503, detail="OCR is disabled; set RADNOTE_ENABLE_OCR=1 and install the OCR extra")
    classifier = getattr(app.state, "classifier", None)
    if classifier is None:
        raise HTTPException(status_code=503, detail="Train a classifier and set RADNOTE_MODEL_PATH")
    if file.content_type not in {"image/png", "image/jpeg"}:
        raise HTTPException(status_code=415, detail="Upload a PNG or JPEG synthetic report page")
    content = await file.read(8_000_001)
    if len(content) > 8_000_000:
        raise HTTPException(status_code=413, detail="Image exceeds the 8 MB limit")
    try:
        import numpy as np
        from PIL import Image
        image = Image.open(BytesIO(content)).convert("RGB")
        extracted = " ".join(reader.readtext(np.asarray(image), detail=0, paragraph=True))
    except Exception as error:
        raise HTTPException(status_code=400, detail=f"Could not decode/OCR the uploaded report page: {error}") from error
    if not extracted.strip():
        raise HTTPException(status_code=422, detail="OCR did not extract report text")
    return _predict_text(classifier, extracted, extracted_text=extracted)
