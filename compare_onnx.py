"""Compare the fixed-size PyTorch and ONNX inputs used in task six."""

import argparse
import json
from pathlib import Path

import cv2
import torch

from lab import DATA, WEIGHTS, load_detector, predict_frame


INPUT_SIZE = 416
CONFIDENCE = 0.10
FRAMES = (100, 150, 200)
EXPECTED_SHAPE = [1, 3, INPUT_SIZE, INPUT_SIZE]


def compare(onnx_path):
    torch.set_num_threads(3)
    pytorch, pt_class = load_detector(WEIGHTS, task="detect")
    onnx, onnx_class = load_detector(onnx_path, task="detect")
    samples = []
    for frame in FRAMES:
        path = DATA / "images" / "val" / f"rt-{frame:03d}.jpg"
        image = cv2.imread(str(path))
        if image is None:
            raise FileNotFoundError(f"先运行 lab.py prepare，缺少图片：{path}")
        samples.append((frame, image))

    # Run once to initialize the Ultralytics predictor before attaching the hook.
    predict_frame(pytorch, pt_class, samples[0][1], INPUT_SIZE, CONFIDENCE, rect=False)
    pytorch_shapes = []

    def record_input_shape(_module, inputs):
        pytorch_shapes.append(list(inputs[0].shape))

    hook = pytorch.predictor.model.model.register_forward_pre_hook(record_input_shape)
    frames = []
    try:
        for frame, image in samples:
            pytorch_boxes, _ = predict_frame(
                pytorch, pt_class, image, INPUT_SIZE, CONFIDENCE, rect=False,
            )
            onnx_boxes, _ = predict_frame(
                onnx, onnx_class, image, INPUT_SIZE, CONFIDENCE, rect=False,
            )
            frames.append({
                "frame": frame,
                "pytorch": pytorch_boxes,
                "onnxruntime": onnx_boxes,
            })
    finally:
        hook.remove()

    # A fixed ONNX input is enforced by ONNX Runtime on every successful run.
    onnx_shape = onnx.predictor.model.session.get_inputs()[0].shape
    if onnx_shape != EXPECTED_SHAPE or pytorch_shapes != [EXPECTED_SHAPE] * len(samples):
        raise ValueError(f"实际输入尺寸未对齐：PyTorch {pytorch_shapes}，ONNX {onnx_shape}")
    return {
        "pytorch_weights": WEIGHTS.name,
        "onnx_weights": str(onnx_path),
        "imgsz": INPUT_SIZE,
        "conf": CONFIDENCE,
        "rect": False,
        "pytorch_input_shapes": pytorch_shapes,
        "onnx_input_shape": onnx_shape,
        "frames": frames,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, required=True, help="指定自行导出的 ONNX 模型")
    parser.add_argument("--output", type=Path, required=True, help="保存逐图对照的 JSON 路径")
    args = parser.parse_args()
    if args.onnx.suffix.lower() != ".onnx" or not args.onnx.is_file():
        parser.error("--onnx 必须指向存在的 .onnx 文件")
    result = json.dumps(compare(args.onnx), ensure_ascii=False, indent=2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(result + "\n", encoding="utf-8")
    print(result)


if __name__ == "__main__":
    main()
