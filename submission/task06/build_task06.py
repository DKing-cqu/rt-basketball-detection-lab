"""Build task 06 input-size, per-frame, and backend comparison artifacts."""
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

import lab  # noqa: E402

FONT = cv2.FONT_HERSHEY_SIMPLEX
GREEN = (0, 255, 0)
DARK = (24, 24, 24)


def load_comparison():
    path = HERE / "backend-comparison.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def panel_header(width, backend, result):
    bar = np.full((54, width, 3), DARK, np.uint8)
    cv2.putText(bar, backend, (8, 20), FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(bar, result, (8, 44), FONT, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
    return bar


def draw_box(image, detection):
    x1, y1, x2, y2 = [int(round(value)) for value in detection["bbox_xyxy"]]
    cv2.rectangle(image, (x1, y1), (x2, y2), GREEN, 2)
    cv2.putText(image, f"{detection['class']} {detection['confidence']:.3f}",
                (x1, max(y1 - 6, 12)), FONT, 0.42, GREEN, 1, cv2.LINE_AA)


def build_frame_image(frame, data):
    image = cv2.imread(str(ROOT / "dataset-curated" / "images" / "val" / f"rt-{frame:03d}.jpg"))
    if image is None:
        raise FileNotFoundError(f"rt-{frame:03d}.jpg")
    panels = []
    for backend, key in (("PyTorch 416", "pytorch"), ("ONNX Runtime 416", "onnxruntime")):
        detections = data[key]
        panel = image.copy()
        if detections:
            top = max(detections, key=lambda item: item["confidence"])
            draw_box(panel, top)
            result = f"{top['class']}  conf={top['confidence']:.6f}"
        else:
            result = "empty (no sports ball)"
        panels.append(np.vstack([panel_header(image.shape[1], backend, result), panel]))

    combined = np.hstack(panels)
    out = HERE / f"rt-{frame:03d}-backends.png"
    if not cv2.imwrite(str(out), combined):
        raise OSError(out)
    return out


def main():
    result = load_comparison()
    if result["imgsz"] != 416 or result["conf"] != 0.10 or result["rect"] is not False:
        raise ValueError("比较入口的参数不是 416 / 0.10 / rect=False")

    input_rows = [
        {
            "backend": "PyTorch",
            "input_shape": json.dumps(result["pytorch_input_shapes"][0]),
            "evidence": f"forward pre-hook, {len(result['pytorch_input_shapes'])}/3 frames",
        },
        {
            "backend": "ONNX Runtime",
            "input_shape": json.dumps(result["onnx_input_shape"]),
            "evidence": "InferenceSession.get_inputs()[0].shape",
        },
    ]
    with (HERE / "task06-input-sizes.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(input_rows[0]))
        writer.writeheader()
        writer.writerows(input_rows)

    frame_rows = []
    summary_rows = []
    for item in result["frames"]:
        frame = item["frame"]
        pt = item["pytorch"]
        on = item["onnxruntime"]
        if pt or on:
            if len(pt) != 1 or len(on) != 1:
                raise ValueError(f"帧 {frame} 不是一一配对: PyTorch={len(pt)} ONNX={len(on)}")
            if pt[0]["class"] != on[0]["class"]:
                raise ValueError(f"帧 {frame} 类别不同")
            iou = lab.iou(pt[0]["bbox_xyxy"], on[0]["bbox_xyxy"])
            delta = abs(pt[0]["confidence"] - on[0]["confidence"])
            met = iou >= 0.95 and delta <= 0.01
            for backend, det in (("PyTorch", pt[0]), ("ONNX Runtime", on[0])):
                frame_rows.append(
                    {
                        "frame": frame,
                        "backend": backend,
                        "class": det["class"],
                        "bbox_xyxy": json.dumps([round(v, 2) for v in det["bbox_xyxy"]]),
                        "confidence": f"{det['confidence']:.6f}",
                        "empty": "no",
                        "pair_iou": f"{iou:.6f}",
                        "confidence_delta": f"{delta:.9f}",
                        "criterion_met": "yes" if met else "no",
                    }
                )
            summary_rows.append(
                {
                    "frame": frame,
                    "pytorch_count": 1,
                    "onnx_count": 1,
                    "paired": "yes",
                    "iou": f"{iou:.6f}",
                    "confidence_delta": f"{delta:.9f}",
                    "empty_match": "",
                    "criterion_met": "yes" if met else "no",
                }
            )
        else:
            for backend in ("PyTorch", "ONNX Runtime"):
                frame_rows.append(
                    {
                        "frame": frame,
                        "backend": backend,
                        "class": "",
                        "bbox_xyxy": "",
                        "confidence": "",
                        "empty": "yes",
                        "pair_iou": "",
                        "confidence_delta": "",
                        "criterion_met": "yes",
                    }
                )
            summary_rows.append(
                {
                    "frame": frame,
                    "pytorch_count": 0,
                    "onnx_count": 0,
                    "paired": "yes",
                    "iou": "",
                    "confidence_delta": "",
                    "empty_match": "yes",
                    "criterion_met": "yes",
                }
            )
        build_frame_image(frame, item)

    with (HERE / "task06-frames.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(frame_rows[0]))
        writer.writeheader()
        writer.writerows(frame_rows)
    with (HERE / "task06-summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    decision = all(row["criterion_met"] == "yes" for row in summary_rows)
    summary = {
        "input_size": 416,
        "conf": 0.10,
        "rect": False,
        "criteria": "same class, one-to-one, IoU >= 0.95, confidence delta <= 0.01, empty results consistent",
        "frames": summary_rows,
        "switch_backend": decision,
    }
    (HERE / "task06-decision.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
