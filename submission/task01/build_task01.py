"""Build the task 01 artifacts from the prepared repo assets.

Run from the repository root after `lab.py prepare` and the conf=0.10 replay:
    .venv/bin/python submission/task01/build_task01.py

The script writes only into submission/task01/ and never mutates the source
images or dataset.
"""
import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

import lab  # noqa: E402

IMGSZ = 640
FIRST_CONF = 0.25
REPLAY_CONF = 0.10
WEIGHTS = ROOT / "yolo11n.pt"
VAL = ROOT / "dataset-curated" / "images" / "val"
FONT = cv2.FONT_HERSHEY_SIMPLEX
GREEN = (0, 200, 0)
ORANGE = (0, 140, 255)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def top_detection(detections):
    return max(detections, key=lambda item: item["confidence"]) if detections else None


def center_xy(detection):
    x1, y1, x2, y2 = detection["bbox_xyxy"]
    return [round((x1 + x2) / 2, 2), round((y1 + y2) / 2, 2)]


def rounded_detections(detections):
    return [
        {
            "bbox_xyxy": [round(value, 2) for value in item["bbox_xyxy"]],
            "class": item["class"],
            "confidence": round(item["confidence"], 6),
        }
        for item in detections
    ]


def draw_detection(image, detection, color, center=None):
    x1, y1, x2, y2 = [int(round(value)) for value in detection["bbox_xyxy"]]
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    text = f'{detection["class"]} {detection["confidence"]:.3f}'
    (text_w, _), _ = cv2.getTextSize(text, FONT, 0.5, 1)
    text_x = min(max(x1, 4), image.shape[1] - text_w - 4)
    cv2.putText(image, text, (text_x, max(y1 - 8, 14)), FONT, 0.5, color, 1, cv2.LINE_AA)
    if center is not None:
        cx, cy = [int(round(value)) for value in center]
        cv2.drawMarker(image, (cx, cy), color, cv2.MARKER_CROSS, 14, 2)
        cv2.circle(image, (cx, cy), 4, color, -1)


def label_bar(width, title, detail):
    bar = np.full((58, width, 3), 24, np.uint8)
    cv2.putText(bar, title, (8, 22), FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(bar, detail, (8, 46), FONT, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
    return bar


def panel(image, title, detail):
    return np.vstack([label_bar(image.shape[1], title, detail), image])


def predict(image, confidence):
    detections, elapsed = lab.predict_frame(
        model, target, image, IMGSZ, confidence
    )
    return detections, elapsed


def main():
    torch.set_num_threads(3)
    global model, target
    model, target = lab.load_detector(WEIGHTS)

    weights_sha = sha256(WEIGHTS)
    command = ".venv/bin/python submission/task01/build_task01.py"

    # 1. First inference: rt-100 at conf=0.25.
    rt100_path = VAL / "rt-100.jpg"
    rt100 = cv2.imread(str(rt100_path))
    if rt100 is None:
        raise FileNotFoundError(rt100_path)
    rt100_dets, rt100_elapsed = predict(rt100, FIRST_CONF)
    rt100_top = top_detection(rt100_dets)
    rt100_center = center_xy(rt100_top) if rt100_top else None
    rt100_vis = rt100.copy()
    if rt100_top:
        draw_detection(rt100_vis, rt100_top, GREEN, rt100_center)
    cv2.imwrite(str(HERE / "rt-100-box.png"), rt100_vis)

    rt100_record = {
        "source": "dataset-curated/images/val/rt-100.jpg",
        "weights": "yolo11n.pt",
        "weights_sha256": weights_sha,
        "device": "cpu",
        "imgsz": IMGSZ,
        "conf": FIRST_CONF,
        "classes": ["sports ball"],
        "command": command,
        "elapsed_ms": rt100_elapsed,
        "detections": rounded_detections(rt100_dets),
        "top_center_xy": rt100_center,
    }
    (HERE / "rt100-inference.json").write_text(
        json.dumps(rt100_record, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    # 2. rt-225 original vs horizontal flip at conf=0.10.
    rt225_path = VAL / "rt-225.jpg"
    rt225 = cv2.imread(str(rt225_path))
    if rt225 is None:
        raise FileNotFoundError(rt225_path)
    shutil.copy2(rt225_path, HERE / "rt-225.jpg")

    rt225_dets, rt225_elapsed = predict(rt225, REPLAY_CONF)
    rt225_top = top_detection(rt225_dets)
    rt225_center = center_xy(rt225_top) if rt225_top else None

    flip = cv2.flip(rt225, 1)
    flip_path = HERE / "rt-225-flip.png"
    cv2.imwrite(str(flip_path), flip)
    flip_dets, flip_elapsed = predict(flip, REPLAY_CONF)
    flip_top = top_detection(flip_dets)
    flip_center = center_xy(flip_top) if flip_top else None

    width = rt225.shape[1]
    mirrored_center = (
        [round(width - flip_center[0], 2), flip_center[1]] if flip_center else None
    )
    delta = (
        [round(mirrored_center[0] - rt225_center[0], 2),
         round(mirrored_center[1] - rt225_center[1], 2)]
        if mirrored_center and rt225_center else None
    )

    centers = {
        "source": "dataset-curated/images/val/rt-225.jpg",
        "weights": "yolo11n.pt",
        "weights_sha256": weights_sha,
        "device": "cpu",
        "imgsz": IMGSZ,
        "conf": REPLAY_CONF,
        "classes": ["sports ball"],
        "command": command,
        "original": {
            "image": "rt-225.jpg",
            "elapsed_ms": rt225_elapsed,
            "detections": rounded_detections(rt225_dets),
            "top_center_xy": rt225_center,
        },
        "flipped": {
            "image": "rt-225-flip.png",
            "elapsed_ms": flip_elapsed,
            "detections": rounded_detections(flip_dets),
            "top_center_xy": flip_center,
            "mirrored_to_original_xy": mirrored_center,
        },
        "mirror_rule": "x_original = width - x_flipped, y unchanged",
        "width": width,
        "delta_mirrored_px": delta,
    }
    (HERE / "rt-225-centers.json").write_text(
        json.dumps(centers, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    rt225_vis = rt225.copy()
    if rt225_top:
        draw_detection(rt225_vis, rt225_top, GREEN, rt225_center)
    flip_vis = flip.copy()
    if flip_top:
        draw_detection(flip_vis, flip_top, ORANGE, flip_center)

    original_panel = panel(
        rt225_vis,
        "rt-225.jpg (original)",
        f"c=({rt225_center[0]:.2f}, {rt225_center[1]:.2f})  conf={rt225_top['confidence']:.3f}"
        if rt225_center else "no detection",
    )
    flip_detail = (
        f"c=({flip_center[0]:.2f}, {flip_center[1]:.2f})  "
        f"mir={mirrored_center[0]:.2f}  conf={flip_top['confidence']:.3f}"
        if flip_center else "no detection"
    )
    flipped_panel = panel(flip_vis, "rt-225-flip.png (horizontal flip)", flip_detail)
    cv2.imwrite(
        str(HERE / "rt-225-center-comparison.png"),
        np.hstack([original_panel, flipped_panel]),
    )

    # 3. Replay rows for frames 200 and 225, from the conf=0.10 replay log.
    replay_path = HERE / "replay-conf010.jsonl"
    if not replay_path.exists():
        raise FileNotFoundError(
            f"missing {replay_path}; run the replay command from README.md first"
        )
    replay_rows = [json.loads(line) for line in replay_path.read_text(encoding="utf-8").splitlines() if line]
    wanted = {row["frame"]: row for row in replay_rows if row["frame"] in (200, 225)}
    if set(wanted) != {200, 225}:
        raise ValueError("replay log does not contain frames 200 and 225")
    if wanted[200]["detections"]:
        raise ValueError("frame 200 should have no detection at conf=0.10")
    if not wanted[225]["detections"]:
        raise ValueError("frame 225 should have a detection at conf=0.10")

    (HERE / "replay-meta.json").write_text(
        json.dumps(
            {
                "source": "data-src/rt.mp4",
                "weights": "yolo11n.pt",
                "weights_sha256": weights_sha,
                "device": "cpu",
                "imgsz": IMGSZ,
                "conf": REPLAY_CONF,
                "classes": ["sports ball"],
                "command": (
                    ".venv/bin/python lab.py replay --source data-src/rt.mp4 "
                    "--imgsz 640 --conf 0.10 "
                    "--output submission/task01/replay-conf010.jsonl"
                ),
                "frames": len(replay_rows),
                "frame_range": [replay_rows[0]["frame"], replay_rows[-1]["frame"]],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    with (HERE / "replay-frames-200-225.jsonl").open("w", encoding="utf-8") as stream:
        for frame in (200, 225):
            stream.write(json.dumps(wanted[frame], ensure_ascii=False) + "\n")

    # 4. One compact table for the questionnaire.
    rows = [
        {
            "item": "rt-100 first inference",
            "frame": 100,
            "image": "rt-100.jpg",
            "conf_threshold": FIRST_CONF,
            "confidence": round(rt100_top["confidence"], 6) if rt100_top else "",
            "bbox_xyxy": json.dumps(rounded_detections([rt100_top])[0]["bbox_xyxy"]) if rt100_top else "",
            "center_x": rt100_center[0] if rt100_center else "",
            "center_y": rt100_center[1] if rt100_center else "",
            "mirrored_x": "",
            "note": "weights=yolo11n.pt, device=cpu, imgsz=640, class=sports ball",
        },
        {
            "item": "rt-225 original",
            "frame": 225,
            "image": "rt-225.jpg",
            "conf_threshold": REPLAY_CONF,
            "confidence": round(rt225_top["confidence"], 6) if rt225_top else "",
            "bbox_xyxy": json.dumps(rounded_detections([rt225_top])[0]["bbox_xyxy"]) if rt225_top else "",
            "center_x": rt225_center[0] if rt225_center else "",
            "center_y": rt225_center[1] if rt225_center else "",
            "mirrored_x": "",
            "note": "replay frame 225 at conf=0.10",
        },
        {
            "item": "rt-225 flipped",
            "frame": 225,
            "image": "rt-225-flip.png",
            "conf_threshold": REPLAY_CONF,
            "confidence": round(flip_top["confidence"], 6) if flip_top else "",
            "bbox_xyxy": json.dumps(rounded_detections([flip_top])[0]["bbox_xyxy"]) if flip_top else "",
            "center_x": flip_center[0] if flip_center else "",
            "center_y": flip_center[1] if flip_center else "",
            "mirrored_x": mirrored_center[0] if mirrored_center else "",
            "note": "horizontal flip; mirrored_x = width - flip center x",
        },
        {
            "item": "replay frame 200",
            "frame": 200,
            "image": "rt.mp4",
            "conf_threshold": REPLAY_CONF,
            "confidence": "",
            "bbox_xyxy": "",
            "center_x": "",
            "center_y": "",
            "mirrored_x": "",
            "note": "empty detection record retained",
        },
    ]
    with (HERE / "results.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps({
        "rt100": rt100_record,
        "rt225": centers,
        "replay_frame_200": wanted[200],
        "replay_frame_225": wanted[225],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
