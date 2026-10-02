# RadNote-AI

This folder contains a reproducible TF-IDF + Logistic Regression baseline for assigning one of the proposal's urgency labels (`Routine`, `Urgent`, `Critical`) to the **text** in a chest-radiology report.

The proposal's data source is the Indiana University Chest X-ray Collection (Open-I). The working corpus is not included here. Keep source-derived data in the group's restricted working directory and use only a course-approved sample in any public repository. This code expects CSV files prepared by the data/preprocessing owner; it does not infer or clinically validate urgency labels.

## Expected CSV columns

Training and optional test CSV files must each contain:

| Column | Meaning |
| --- | --- |
| `findings` | Report Findings text; empty is allowed |
| `impression` | Report Impression text; empty is allowed |
| `urgency_label` | One of `Routine`, `Urgent`, `Critical` |

The proposed rule draft and engineering gates are in the shared documentation folder. The script `scripts/label_openi_weak.py` creates provisional labels from explicit Impression language and uses MeSH only as a corroborating concept signal. MeSH descriptors alone do not encode acuity. Do not describe weak labels as clinically validated.

The X-ray pixels are outside project scope. The classifier uses Findings/Impression report text only. OCR, if demonstrated, is limited to team-created synthetic report-page images in the shared sample folder.

## Run the baseline

From this directory, install the project in an isolated environment, then run:

```powershell
uv sync
uv run python scripts/train_classifier.py --train-csv PATH\TO\train.csv --test-csv PATH\TO\test.csv --model-out artifacts\urgency_classifier.joblib --metrics-out artifacts\metrics.json
```

Omit `--test-csv` to train without an evaluation report. The script writes macro and weighted precision/recall/F1, per-class metrics, and a confusion matrix when a test set is supplied. Use a held-out split prepared before fitting; do not use the test set to tune rules or thresholds.

Before trying training, generate labels only in the private data directory:

```powershell
uv run python scripts/label_openi_weak.py --input ..\..\..\private-data\openi\openi_reports_unlabeled.csv --output ..\..\..\private-data\openi\openi_reports_weak_labeled.csv
```

The script is a provisional labeling proposal, not a substitute for the team's label audit. The current pass labels only 940 of 3,955 reports automatically (912 Routine, 27 Urgent, 1 Critical); the remaining 3,015 are `REVIEW`. This is too imbalanced to support a meaningful three-class evaluation. Do not train/report a three-class score until the group reviews the mapping and improves class coverage with audited labels.

## Export the fitted classifier to ONNX

ONNX conversion is optional and must be checked in the target runtime. Install the export extra and run:

```powershell
uv sync --extra onnx
uv run python scripts/export_classifier_onnx.py --model artifacts\urgency_classifier.joblib --output artifacts\urgency_classifier.onnx
```

This exports the scikit-learn text pipeline only. It does not export OCR or spaCy. The selected accessible target is an Ubuntu CPU VM; after export, run the parity check there:

```powershell
uv sync --extra onnx
uv run --with onnxruntime python scripts/verify_onnx_cpu.py --model artifacts\urgency_classifier.joblib --onnx artifacts\urgency_classifier.onnx --csv PATH\TO\held_out.csv
```

The parity script requires exact label agreement and the ONNX Runtime CPU provider. Conversion and target execution are still pending; do not claim them as verified until this command completes successfully on the VM. Ascend/Atlas is not currently an accessible deployment target, so confirm whether the course rubric requires it.

## Sprint status

- Implemented: configurable baseline training/evaluation and an ONNX export path.
- Pending: train on the group's prepared data, review the weak-label policy and class counts, run conversion, inspect the ONNX inputs/outputs, and evaluate it in the deployment environment.
- No real dataset, credentials, model binaries, or patient information belongs in this repository.
