"""Export yolo11n.pt to a fixed 416x416 ONNX model for task 06.

Run from the repository root:
    .venv/bin/python submission/task06/export_onnx.py

The exported .onnx file is ignored by Git; the config JSON is committed.
"""
import hashlib
import json
import shutil
from pathlib import Path

import onnx
import onnxruntime
import torch
import ultralytics
from ultralytics import YOLO

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
WEIGHTS = ROOT / "yolo11n.pt"
OUTPUT = HERE / "yolo11n-416.onnx"

ARGS = {
    "format": "onnx",
    "imgsz": 416,
    "dynamic": False,
    "simplify": False,
    "opset": 12,
    "device": "cpu",
    "batch": 1,
    "half": False,
    "int8": False,
}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    model = YOLO(str(WEIGHTS))
    exported = Path(model.export(**ARGS))
    shutil.copy2(exported, OUTPUT)

    session = onnxruntime.InferenceSession(str(OUTPUT), providers=["CPUExecutionProvider"])
    input_shape = session.get_inputs()[0].shape
    config = {
        "weights": WEIGHTS.name,
        "weights_sha256": sha256(WEIGHTS),
        "output": str(OUTPUT.relative_to(ROOT)),
        "output_sha256": sha256(OUTPUT),
        "format": ARGS["format"],
        "imgsz": ARGS["imgsz"],
        "dynamic": ARGS["dynamic"],
        "simplify": ARGS["simplify"],
        "opset": ARGS["opset"],
        "device": ARGS["device"],
        "batch": ARGS["batch"],
        "half": ARGS["half"],
        "int8": ARGS["int8"],
        "onnx_input_shape": input_shape,
        "versions": {
            "torch": torch.__version__,
            "ultralytics": ultralytics.__version__,
            "onnx": onnx.__version__,
            "onnxruntime": onnxruntime.__version__,
        },
        "command": (
            ".venv/bin/python submission/task06/export_onnx.py"
        ),
    }
    (HERE / "onnx-export-config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(config, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
