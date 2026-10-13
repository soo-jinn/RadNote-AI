# RadNote-AI

CPE179P Group 4 final implementation for a professor-approved Ubuntu VM demonstration. The application processes report text and synthetic text-page images. Radiographs are outside scope. This is an academic system with explicit review flags and limited weak/synthetic model validation.

## Runtime architecture

An Ubuntu browser contacts the portable client container at localhost:8765. That client sends HTTP requests over the Compose network to the FastAPI backend at backend:8000. OCR, synthetic BIO entity tagging and urgency inference use ONNX Runtime CPU. The default precision is static INT8 for supported layers, with float/native operations retained where required. Both containers run inside the Ubuntu VM; no Raspberry Pi is required.

## Start on Ubuntu

Run from the repository root containing this README, Dockerfile and docker-compose.yml:

```bash
python3 scripts/preflight.py
bash deploy/setup_docker_ubuntu.sh
cp .env.example .env
sudo docker compose build backend client
sudo docker compose up -d --wait --wait-timeout 240 backend client
sudo docker compose ps
curl --fail http://127.0.0.1:8000/health
curl --fail http://127.0.0.1:8765/ready
```

Open **http://127.0.0.1:8765** in the Ubuntu browser. The backend interface and API reference are available at **http://127.0.0.1:8000** and **/docs**. Engine installation/build needs internet access initially. Supplied ONNX artifacts allow runtime inference without downloading pretrained weights. `scripts/cache_ocr.py` is needed only when rebuilding OCR graphs from their original PyTorch networks.

The package targets an amd64 CPU Ubuntu VM. Suggested rehearsal resources are 8 GB RAM, 4 vCPUs and 40 GB available disk, subject to the host's capacity. Record actual allocations and measurements rather than treating these suggestions as validated minimums.

## Complete VM verification and benchmarking

```bash
mkdir -p results/vm
bash deploy/verify_vm.sh 2>&1 | tee results/vm/verification_console.log
```

This checks hashes, builds images, records hardware/versions, measures FP32/INT8 and direct/HTTP execution, verifies API/client integration, uploads all 60 synthetic pages, runs sustained load and collects Docker CPU/memory. It saves actual evidence in `results/vm/`, including `verification_summary.json` and `VM_Benchmark_Summary.md`. Heavy local benchmarks run with this project's serving workload stopped. Results retain their measurement host/protocol.

The full run is preparation work. For the live segment, restart the already-built service and demonstrate uploads, JSON and `sudo docker stats $(sudo docker compose ps -q backend client)`.

## API and CLI

- `GET /health`: model readiness, precision and CPU provider.
- `GET /ready`: HTTP 200 when required components are ready; 503 otherwise.
- `GET /metrics`: sampled process RSS, cumulative CPU time and thread count.
- `POST /predict`: nonempty Findings/Impression JSON.
- `POST /predict-image`: PNG/JPEG multipart synthetic report page, at most 8 MB and 16 megapixels.

```bash
sudo docker compose exec -T client python client/report_client.py \
  --server http://backend:8000 --image samples/ocr/images/case_001.png
```

Responses include urgency, critical_alert, probabilities, review_required/reasons, sections, entities, extracted text, model hash and server processing time. Scores are not calibrated confidence. Legacy `pi_client.py` / `pi_web.py` modules remain for existing imports; they are portable Python code and do not require Pi hardware.

## Models, data and evaluation limits

Eight ONNX files are supplied: FP32 and supported-layer INT8 versions of classifier, OCR detector, OCR recognizer and NER. `models/model_manifest.json` records sizes/hashes. The FP32 OCR detector is about 83 MB and is retained for reproducible comparisons.

The classifier uses word unigram/bigram TF-IDF and balanced logistic regression trained on 658 weak-label Open-I rows. Test support is 137 Routine, 4 Urgent and 0 Critical. Existing weak-label accuracy is 0.98582 and three-label macro-F1 is 0.55314; Critical recall is unassessable. Entities are displayed separately from classifier features. NER learns synthetic BIO labels; unknown tokens are forced to O. The OCR corpus is 60 original synthetic pages, not radiographs or independent clinical cases. Every result remains flagged for review.

`scripts/benchmark_precision.py` measures paired precision and synthetic NER behavior. An optional authorized private classifier CSV enables aggregate holdout metrics without exporting private text/row predictions. Public engineering benchmarks do not need private data.

Older Windows results are in `results/reference_windows_oct06/`. Final-source Windows checks are in `results/verification_windows_oct13/` when provided. These are separate from `results/vm/` and cannot be described as VM/Docker/Pi results.

## Container controls

Docker uses dependency, backend runtime and lightweight client targets. Processes are non-root, root filesystems are read-only, temporary directories are writable, Linux UTF-8 locale is generated for ONNX string normalization, and Compose configures CPU/memory/pid caps, health dependencies, dropped capabilities and loopback publication. Configured caps differ from measured usage. Institutional authentication, TLS, high availability and clinical validation are outside the proven scope.

## Source layout and documentation

Use root paths: `api/`, `client/`, `deploy/`, `models/`, `radnote/`, `results/`, `samples/`, `scripts/`, `tests/`, `web/` and root configuration. Merge assigned paths into the existing repository and preserve unrelated Sprint 1 files pending reviewed retirement.

Reports, architecture diagrams, schedules, signed forms, peer evaluations and recordings go to the separate documentation repository/course folders. Document-building code is outside this system repository. The full handoff includes the detailed Ubuntu walkthrough and a tool that refreshes the report from actual VM results.

## Development without Docker

```bash
uv sync --frozen --extra ocr --extra dev --extra training
uv run --no-sync uvicorn api.main:app --host 127.0.0.1 --port 8000
uv run --no-sync pytest -q
```

ONNX inference needs runtime/OCR dependencies but no raw private reports or original PyTorch checkpoint cache. Re-export uses authorized data, training dependencies and reproduction controls. Linux string normalization requires en_US.UTF-8; Docker generates it.

## AI use and provenance

OpenAI Codex assisted source, synthetic data, documentation and authorized local checks. Pretrained OCR comes from EasyOCR; NER uses synthetic annotations and classification uses Open-I weak labels. Members document their real review/execution/changes and supply genuine contribution evidence. Model limits and measurement origins remain visible.

References: [EasyOCR](https://github.com/JaidedAI/EasyOCR), [ONNX Runtime quantization](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html), [Docker multistage builds](https://docs.docker.com/build/building/multi-stage/), [Docker Ubuntu installation](https://docs.docker.com/engine/install/ubuntu/).
