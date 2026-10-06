# FastAPI starter - Carlo review and commit required

This draft provides `/health`, text `/predict`, and optional `/predict-image` for OCR of **synthetic report-page images only**. Radiograph pixels are outside project scope. OCR is disabled unless explicitly enabled; the classifier still operates on extracted text. NER uses a small spaCy EntityRuler lexicon prototype, not a trained clinical NER model. Carlo and Sonnelo must review/adapt and commit their own parts from their GitHub accounts.

## Local run

From the codebase directory, install the API extra and point the app at a trained classifier file:

```powershell
uv sync --extra api --extra ner --extra ocr
$env:RADNOTE_MODEL_PATH = "artifacts\urgency_classifier.joblib"
$env:RADNOTE_ENABLE_OCR = "1"
uv run uvicorn api.main:app --host 127.0.0.1 --port 8000
```

`GET /health` reports classifier/OCR/NER readiness. `POST /predict` accepts JSON with `findings` and `impression`. `POST /predict-image` accepts PNG/JPEG and returns OCR text, rule-based entities, and the text model result. Use only team-created synthetic report-page samples in the Sprint 1 demonstration. The endpoint cannot verify that a user-uploaded image is synthetic, so keep it local during the class demonstration. Logistic Regression scores are not guaranteed to be calibrated confidence estimates.

Do not bind the service to a public interface or expose report text until the team has reviewed VM networking and access controls. No trained model is included in this repository.

## Before Carlo commits

- Review and adapt this code, then commit it from Carlo's own GitHub account.
- Run the FastAPI app on the Ubuntu CPU VM and verify `/health`, text `/predict`, and synthetic report-page `/predict-image`.
- Review the upload restrictions and OCR model download/storage plan before enabling OCR on the VM.
- Coordinate the entity schema with Sonnelo, expand the NER lexicon, and create actual PNG/JPEG renderings from the synthetic SVG source files.
- Confirm the course's Huawei Ascend/Atlas requirement against the proposal's current ONNX Runtime CPU plan.
- Capture Carlo's real VM environment evidence and code-review video; this starter code is not proof of deployment.
