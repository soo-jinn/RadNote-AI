from __future__ import annotations

import argparse
from pathlib import Path

import joblib
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import StringTensorType


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    model = joblib.load(args.model)
    onnx_model = convert_sklearn(
        model,
        initial_types=[("report_text", StringTensorType([None, 1]))],
        target_opset=17,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(onnx_model.SerializeToString())
    print(f"Saved ONNX classifier: {args.output}")
    print("Validate input/output names and predictions against the scikit-learn model before deployment.")


if __name__ == "__main__":
    main()
