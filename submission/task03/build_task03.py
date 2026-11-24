"""Build the task 03 statistics and A/B/C frame comparisons.

Prerequisites (see run_task03.sh):
    submission/task03/eval-A.jsonl  # 640, conf=0.10
    submission/task03/eval-B.jsonl  # 320, conf=0.10
    submission/task03/eval-C.jsonl  # 320, conf=0.25

Run from the repository root:
    .venv/bin/python submission/task03/build_task03.py
"""
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

CONFIGS = {
    "A": (640, 0.10),
    "B": (320, 0.10),
    "C": (320, 0.25),
}
COMPARE_FRAMES = (25, 275)
FONT = cv2.FONT_HERSHEY_SIMPLEX
PRED_COLOR = (0, 255, 0)
GT_COLOR = (190, 190, 190)
DARK = (24, 24, 24)


def dashed_rect(image, box, color, thickness=1, dash=6):
    x1, y1, x2, y2 = [int(round(value)) for value in box]
    for x in range(x1, x2, dash * 2):
        cv2.line(image, (x, y1), (min(x + dash, x2), y1), color, thickness)
        cv2.line(image, (x, y2), (min(x + dash, x2), y2), color, thickness)
    for y in range(y1, y2, dash * 2):
        cv2.line(image, (x1, y), (x1, min(y + dash, y2)), color, thickness)
        cv2.line(image, (x2, y), (x2, min(y + dash, y2)), color, thickness)


def load_eval(config):
    path = HERE / f"eval-{config}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"missing {path}; run submission/task03/run_task03.sh")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def summarize(rows):
    counts = {
        "regular": {"frames": 0, "hits": 0, "false_alarms": 0},
        "edge": {"frames": 0, "hits": 0, "false_alarms": 0},
        "empty": {"frames": 0, "hits": 0, "false_alarms": 0},
    }
    for row in rows:
        group = row["group"]
        if group not in counts:
            continue
        counts[group]["frames"] += 1
        counts[group]["hits"] += int(row["matched"])
        counts[group]["false_alarms"] += int(group == "empty" and bool(row["detections"]))
    return counts


def header_bar(width, lines, height=52):
    bar = np.full((height, width, 3), DARK, np.uint8)
    for index, line in enumerate(lines):
        cv2.putText(bar, line, (8, 20 + index * 22), FONT, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    return bar


def draw_panel(image, gt, detections, config, imgsz, confidence):
    panel = image.copy()
    if gt:
        dashed_rect(panel, gt, GT_COLOR, 1)
        cv2.putText(panel, "GT", (int(gt[0]) + 3, max(int(gt[1]) - 5, 12)),
                    FONT, 0.4, GT_COLOR, 1, cv2.LINE_AA)

    top = max(detections, key=lambda item: item["confidence"]) if detections else None
    best_iou = max((lab.iou(gt, item["bbox_xyxy"]) for item in detections), default=0.0) if gt else 0.0
    hit = bool(gt) and best_iou >= lab.MATCH_IOU
    if top:
        x1, y1, x2, y2 = [int(round(value)) for value in top["bbox_xyxy"]]
        cv2.rectangle(panel, (x1, y1), (x2, y2), PRED_COLOR, 2)
        cv2.putText(panel, f"pred {top['confidence']:.3f}",
                    (x1, min(y2 + 15, image.shape[0] - 4)), FONT, 0.42, PRED_COLOR, 1, cv2.LINE_AA)

    result = (
        f"hit=yes  IoU={best_iou:.3f}  pred={top['confidence']:.3f}"
        if top else "hit=no  IoU=0  pred=none"
    )
    lines = [f"{config}: imgsz={imgsz}  conf={confidence:.2f}", result]
    return np.vstack([header_bar(image.shape[1], lines), panel])


def build_comparison(frame, rows_by_config):
    row = None
    for rows in rows_by_config.values():
        row = next(item for item in rows if item["frame"] == frame)
        break
    image = cv2.imread(str(ROOT / "dataset-curated" / "images" / "val" / f"rt-{frame:03d}.jpg"))
    if image is None:
        raise FileNotFoundError(f"rt-{frame:03d}.jpg")
    gt = row["annotation_xyxy"]

    panels = []
    for config, (imgsz, confidence) in CONFIGS.items():
        rows = rows_by_config[config]
        current = next(item for item in rows if item["frame"] == frame)
        panels.append(draw_panel(image, gt, current["detections"], config, imgsz, confidence))

    width = image.shape[1] * len(panels)
    title = np.full((34, width, 3), DARK, np.uint8)
    cv2.putText(title, f"rt-{frame:03d}.jpg  {image.shape[1]}x{image.shape[0]}  |  "
                       "GT dashed gray, prediction solid green",
                (8, 23), FONT, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
    combined = np.vstack([title, np.hstack(panels)])
    out = HERE / f"rt-{frame:03d}-ABC-comparison.png"
    if not cv2.imwrite(str(out), combined):
        raise OSError(out)
    return out


def main():
    all_rows = {config: load_eval(config) for config in CONFIGS}
    stats = {config: summarize(rows) for config, rows in all_rows.items()}

    csv_rows = []
    for config, (imgsz, confidence) in CONFIGS.items():
        counts = stats[config]
        regular = counts["regular"]
        edge = counts["edge"]
        empty = counts["empty"]
        ball_hits = regular["hits"] + edge["hits"]
        ball_total = regular["frames"] + edge["frames"]
        csv_rows.append(
            {
                "config": config,
                "imgsz": imgsz,
                "conf": f"{confidence:.2f}",
                "regular_hits": regular["hits"],
                "regular_total": regular["frames"],
                "regular_fraction": f"{regular['hits']}/{regular['frames']}",
                "edge_hits": edge["hits"],
                "edge_total": edge["frames"],
                "edge_fraction": f"{edge['hits']}/{edge['frames']}",
                "empty_false_alarms": empty["false_alarms"],
                "empty_total": empty["frames"],
                "empty_fraction": f"{empty['false_alarms']}/{empty['frames']}",
                "ball_hits": ball_hits,
                "ball_total": ball_total,
                "ball_fraction": f"{ball_hits}/{ball_total}",
            }
        )

    with (HERE / "task03-stats.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)

    for frame in COMPARE_FRAMES:
        out = build_comparison(frame, all_rows)
        print(f"{frame}: {out}")

    print(json.dumps(csv_rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
