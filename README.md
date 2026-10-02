# RadNote-AI — Sprint 1 classifier baseline

This folder contains Gilbert's Sprint 1 contribution: a reproducible TF-IDF + Logistic Regression baseline for assigning one of the proposal's urgency labels (`Routine`, `Urgent`, `Critical`) to the **text** in a chest-radiology report.

The proposal's data source is the Indiana University Chest X-ray Collection (Open-I). The working corpus is not included here. Keep source-derived data in the group's restricted working directory and use only a course-approved sample in any public repository. This code expects CSV files prepared by the data/preprocessing owner; it does not infer or clinically validate urgency labels.

## Expected CSV columns

Training and optional test CSV files must each contain:

| Column | Meaning |
| --- | --- |
| `findings` | Report Findings text; empty is allowed |
| `impression` | Report Impression text; empty is allowed |
| `urgency_label` | One of `Routine`, `Urgent`, `Critical` |

The group's MeSH-to-urgency mapping and spot-check are still team decisions. Do not describe weak labels as clinically validated.

## Run the baseline

From this directory, install the project in an isolated environment, then run:

```powershell
uv sync
uv run python scripts/train_classifier.py --train-csv PATH\TO\train.csv --test-csv PATH\TO\test.csv --model-out artifacts\urgency_classifier.joblib --metrics-out artifacts\metrics.json
```

Omit `--test-csv` to train without an evaluation report. The script writes macro and weighted precision/recall/F1, per-class metrics, and a confusion matrix when a test set is supplied. Use a held-out split prepared before fitting; do not use the test set to tune rules or thresholds.

## Export the fitted classifier to ONNX

ONNX conversion is optional and must be checked in the target runtime. Install the export extra and run:

```powershell
uv sync --extra onnx
uv run python scripts/export_classifier_onnx.py --model artifacts\urgency_classifier.joblib --output artifacts\urgency_classifier.onnx
```

This exports the scikit-learn text pipeline only. It does not export OCR or spaCy, and this initial contribution does not claim the ONNX artifact has been validated on Ascend/Atlas or on the Ubuntu deployment VM.

## Sprint status

- Implemented: configurable baseline training/evaluation and an ONNX export path.
- Pending: train on the group's prepared data, review the weak-label policy and class counts, run conversion, inspect the ONNX inputs/outputs, and evaluate it in the deployment environment.
- No real dataset, credentials, model binaries, or patient information belongs in this repository.
