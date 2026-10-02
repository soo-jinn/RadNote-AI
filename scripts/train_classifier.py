"""Train and evaluate the RadNote-AI Sprint 1 urgency text baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline


LABELS = ("Routine", "Urgent", "Critical")
REQUIRED_COLUMNS = ("findings", "impression", "urgency_label")


def _read_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path).fillna("")
    missing = set(REQUIRED_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(sorted(missing))}")
    if frame.empty:
        raise ValueError(f"{path} contains no rows")
    return frame


def _combine_report_text(frame: pd.DataFrame) -> list[str]:
    return [
        f"{findings}\n{impression}".strip()
        for findings, impression in zip(frame["findings"], frame["impression"])
    ]


def _validate_labels(frame: pd.DataFrame, path: Path) -> None:
    labels = set(frame["urgency_label"].astype(str).str.strip())
    unexpected = labels - set(LABELS)
    if unexpected:
        raise ValueError(
            f"{path} has unsupported urgency labels: {', '.join(sorted(unexpected))}. "
            f"Allowed values: {', '.join(LABELS)}"
        )


def train(train_csv: Path) -> Pipeline:
    frame = _read_csv(train_csv)
    _validate_labels(frame, train_csv)
    observed = set(frame["urgency_label"].astype(str).str.strip())
    missing = set(LABELS) - observed
    if missing:
        raise ValueError(
            f"{train_csv} has no training examples for: {', '.join(sorted(missing))}"
        )
    model = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, strip_accents="unicode")),
            ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )
    model.fit(_combine_report_text(frame), frame["urgency_label"].astype(str).str.strip())
    return model


def evaluate(model: Pipeline, test_csv: Path) -> dict[str, Any]:
    frame = _read_csv(test_csv)
    _validate_labels(frame, test_csv)
    predictions = model.predict(_combine_report_text(frame))
    return {
        "support_rows": int(len(frame)),
        "labels": list(LABELS),
        "classification_report": classification_report(
            frame["urgency_label"].astype(str).str.strip(),
            predictions,
            labels=list(LABELS),
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            frame["urgency_label"].astype(str).str.strip(), predictions, labels=list(LABELS)
        ).tolist(),
        "confusion_matrix_label_order": list(LABELS),
        "evaluation_note": "Metrics reflect the supplied held-out labels; weak labels are not clinical validation.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", required=True, type=Path)
    parser.add_argument("--test-csv", type=Path)
    parser.add_argument("--model-out", required=True, type=Path)
    parser.add_argument("--metrics-out", type=Path)
    args = parser.parse_args()

    model = train(args.train_csv)
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.model_out)
    print(f"Saved classifier: {args.model_out}")

    if args.test_csv:
        metrics = evaluate(model, args.test_csv)
        if args.metrics_out:
            args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
            args.metrics_out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
            print(f"Saved evaluation metrics: {args.metrics_out}")
        print(json.dumps(metrics["classification_report"], indent=2))


if __name__ == "__main__":
    main()
