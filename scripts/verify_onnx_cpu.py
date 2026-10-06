from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import onnxruntime as ort


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--onnx", required=True, type=Path)
    parser.add_argument("--csv", required=True, type=Path, help="Held-out CSV with findings and impression columns")
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()

    import pandas as pd

    frame = pd.read_csv(args.csv).fillna("").head(args.limit)
    if frame.empty:
        raise SystemExit("Held-out CSV has no rows")
    texts = [f"{a}\n{b}".strip() for a, b in zip(frame["findings"], frame["impression"])]
    sklearn_model = joblib.load(args.model)
    expected = [str(value) for value in sklearn_model.predict(texts)]

    session = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
    if "CPUExecutionProvider" not in session.get_providers():
        raise SystemExit("ONNX Runtime CPU provider is unavailable")
    input_meta = session.get_inputs()
    if len(input_meta) != 1:
        raise SystemExit(f"Expected one text input; found {len(input_meta)}")
    outputs = session.run(None, {input_meta[0].name: np.asarray([[text] for text in texts], dtype=object)})
    matches = False
    for output in outputs:
        try:
            observed = [str(value) for value in np.asarray(output).reshape(-1)]
        except (TypeError, ValueError):
            continue
        if len(observed) == len(expected) and observed == expected:
            matches = True
            break
    if not matches:
        raise SystemExit("ONNX class output did not exactly match scikit-learn predictions")
    print(f"CPU provider: {session.get_providers()}")
    print(f"Prediction parity: 100% ({len(expected)}/{len(expected)})")


if __name__ == "__main__":
    main()
