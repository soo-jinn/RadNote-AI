"""Meaningful regression checks for the actual ONNX API and report handling."""
import io
import os
from pathlib import Path
import pytest
from PIL import Image
from fastapi.testclient import TestClient
from radnote.entities import extract_entities
from radnote.text import parse_report
from radnote.ocr import decode_image, ImageInputError

os.environ["RADNOTE_ENABLE_OCR"]="0"
from api.main import app

@pytest.fixture
def client():
    with TestClient(app) as handle:
        yield handle

def test_actual_onnx_text_api_and_review(client):
    health=client.get("/health").json()
    assert health["classifier_backend"]=="ONNX Runtime"
    assert health["providers"]==["CPUExecutionProvider"]
    result=client.post("/predict",json={"findings":"The lungs are clear.","impression":"No acute disease."})
    assert result.status_code==200
    data=result.json()
    assert data["urgency"] in {"Routine","Urgent","Critical"}
    assert abs(sum(data["probabilities"].values())-1)<1e-5
    assert data["review_required"]
    assert len(data["model_sha256"])==64
    assert data["latency_ms"]>=0

def test_empty_and_oversized_text(client):
    assert client.post("/predict",json={}).status_code==422
    assert client.post("/predict",json={"findings":"x"*20001}).status_code==422

def test_disabled_ocr_returns_actionable_error(client):
    response=client.post("/predict-image",files={"file":("report.png",b"invalid","image/png")})
    assert response.status_code==503

def test_image_upload_processing_and_limits(client):
    # OCR unit is replaced here to isolate upload contract; live benchmark exercises real OCR.
    class StubOCR:
        def read(self,content,mime):
            decode_image(content,mime)
            return "FINDINGS: The lungs are clear.\nIMPRESSION: No acute disease."
    app.state.ocr=StubOCR()
    image=io.BytesIO();Image.new("RGB",(100,100),"white").save(image,format="PNG")
    result=client.post("/predict-image",files={"file":("report.png",image.getvalue(),"image/png")})
    assert result.status_code==200 and result.json()["extracted_text"]
    assert client.post("/predict-image",files={"file":("x.svg",b"<svg/>","image/svg+xml")}).status_code==415
    assert client.post("/predict-image",files={"file":("x.png",b"invalid","image/png")}).status_code==400
    assert client.post("/predict-image",files={"file":("x.png",b"x"*8000001,"image/png")}).status_code==413

def test_readiness_degrades_when_requested_ocr_missing(client):
    app.state.ocr_requested=True
    assert client.get("/ready").status_code==503

def test_negation_does_not_cross_sentence():
    entities=extract_entities("No pneumothorax. New left pleural effusion.")
    findings={e["text"]:e["assertion"] for e in entities if e["label"]=="FINDING"}
    assert findings=={"pneumothorax":"negated","pleural effusion":"affirmed"}

def test_sections_exclude_preamble_and_support_merged_ocr():
    findings,impression,warnings=parse_report("Report page FINDINGS: Lungs clear. IMPRESSION: No acute disease.")
    assert findings=="Lungs clear." and impression=="No acute disease." and not warnings
    assert parse_report("Lungs clear")[2]

def test_negation_survives_ocr_line_wrap_and_resets_at_heading():
    text="FINDINGS: No pleural effusion or\npneumothorax.\nIMPRESSION: New left pleural effusion."
    findings=[e for e in extract_entities(text) if e["label"]=="FINDING"]
    assert [e["assertion"] for e in findings]==["negated","negated","affirmed"]

def test_actual_decode_pixel_limit():
    image=io.BytesIO();Image.new("L",(5000,4000),255).save(image,format="PNG")
    with pytest.raises(ImageInputError,match="megapixel"):
        decode_image(image.getvalue(),"image/png")

def test_frontend_assets_and_synthetic_sample(client):
    assert client.get("/").status_code==200
    assert client.get("/static/app.js").status_code==200
    assert client.get("/samples/case_001.png").status_code==200

def test_real_onnx_ner_negation(client):
    result=client.post("/predict",json={"findings":"No pleural effusion or\npneumothorax.","impression":"No acute disease."}).json()
    findings={e["text"]:e["assertion"] for e in result["entities"] if e["label"]=="FINDING"}
    assert findings["pleural effusion"]=="negated" and findings["pneumothorax"]=="negated"

def test_critical_alert_contract_when_classifier_returns_critical(client):
    original=app.state.classifier
    class StubClassifier:
        metadata=original.metadata;sha256=original.sha256
        def predict(self,text):return "Critical",{"Critical":.9,"Urgent":.08,"Routine":.02}
    app.state.classifier=StubClassifier()
    result=client.post("/predict",json={"impression":"Tension pneumothorax."}).json()
    assert result["critical_alert"] and result["review_required"]
    app.state.classifier=original
