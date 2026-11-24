"""Build the task 05 split audit, Jackson per-frame results, and match figure.

Run after the fixed independent test:
    .venv/bin/python lab.py eval --weights yolo11n.pt --split test --imgsz 416 --conf 0.10 \
      --output submission/task05/eval-jackson.jsonl
    .venv/bin/python submission/task05/build_task05.py
"""
import csv
import json
import sys
from pathlib import Path

import cv2

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

import lab  # noqa: E402

SPLIT_ORDER = {"train": 0, "val": 1, "test": 2}
CORRECTIONS = [
    {
        "frame": "rt-125",
        "wrong_split": "train",
        "correct_split": "val",
        "reason": "RT 是同一段录像，所有 RT 帧必须同组。",
    },
    {
        "frame": "jackson-011",
        "wrong_split": "val",
        "correct_split": "test",
        "reason": "Jackson 是留出的独立测试录像，所有 Jackson 帧必须同组。",
    },
]


def load_eval():
    path = HERE / "eval-jackson.jsonl"
    if not path.exists():
        raise FileNotFoundError(path)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def draw_label(image, origin, text, color, scale=0.9, thickness=2):
    (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x, y = origin
    y = max(y, text_h + 8)
    cv2.rectangle(image, (x, y - text_h - 6), (x + text_w + 8, y + 4), (0, 0, 0), -1)
    cv2.putText(image, text, (x + 4, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)
    return y


def draw_box(image, box, color, label, thickness=4):
    x1, y1, x2, y2 = [int(round(value)) for value in box]
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    draw_label(image, (x1, y1 - 8), label, color, scale=0.8, thickness=2)


def build_match_figure(rows):
    row = next(item for item in rows if item["frame"] == 35)
    if not row["detections"]:
        raise ValueError("frame 35 should be the frame with output")
    image = cv2.imread(str(ROOT / "dataset-curated" / "images" / "test" / "jackson-035.jpg"))
    if image is None:
        raise FileNotFoundError("jackson-035.jpg")
    detection = max(row["detections"], key=lambda item: item["confidence"])

    draw_box(image, row["annotation_xyxy"], (255, 128, 0), "GT basketball")
    draw_box(
        image,
        detection["bbox_xyxy"],
        (0, 0, 255),
        f"pred {detection['confidence']:.3f}",
    )

    header = [
        "jackson-035.jpg  1440x1920  imgsz=416  conf=0.10",
        f"GT blue / pred red  |  IoU={row['best_iou']:.3f}  matched={'yes' if row['matched'] else 'no'}",
    ]
    bar_h = 86
    cv2.rectangle(image, (0, 0), (1180, bar_h), (0, 0, 0), -1)
    for index, text in enumerate(header):
        cv2.putText(image, text, (14, 30 + index * 32), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255), 2, cv2.LINE_AA)

    out = HERE / "jackson-035-match.png"
    if not cv2.imwrite(str(out), image):
        raise OSError(out)
    return out


def main():
    rows = load_eval()
    annotations = {row["frame"]: row for row in lab.read_annotations() if row["source"] == "jackson"}
    if len(rows) != 8 or len(annotations) != 8:
        raise ValueError("Jackson test split should contain 8 frames")

    result_rows = []
    for row in rows:
        frame = row["frame"]
        has_ball = bool(row["annotation_xyxy"])
        detections = row["detections"]
        missed = int(has_ball and not row["matched"])
        empty_false_alarm = int(not has_ball and bool(detections))
        extra_boxes = len(detections) - int(row["matched"])
        if not has_ball:
            note = "empty frame; false alarm" if empty_false_alarm else "empty frame; no false alarm"
        elif row["matched"]:
            note = "matched"
        elif detections:
            note = "ball missed; extra box"
        else:
            note = "ball missed"
        result_rows.append(
            {
                "frame": frame,
                "time_s": row["time_s"],
                "group": row["group"],
                "has_ball": "yes" if has_ball else "no",
                "annotation_xyxy": json.dumps(row["annotation_xyxy"], separators=(",", ":")),
                "detections": json.dumps(detections, separators=(",", ":")),
                "best_iou": f"{row['best_iou']:.4f}",
                "matched": "yes" if row["matched"] else "no",
                "missed": missed,
                "empty_false_alarm": empty_false_alarm,
                "extra_boxes": extra_boxes,
                "note": note,
            }
        )

    with (HERE / "task05-jackson-results.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result_rows[0]))
        writer.writeheader()
        writer.writerows(result_rows)

    ball_frames = sum(row["has_ball"] == "yes" for row in result_rows)
    matched = sum(row["matched"] == "yes" for row in result_rows)
    missed = sum(row["missed"] for row in result_rows)
    empty_frames = len(result_rows) - ball_frames
    empty_false_alarms = sum(row["empty_false_alarm"] for row in result_rows)
    extra_boxes = sum(row["extra_boxes"] for row in result_rows)
    totals = [
        {"metric": "matched", "numerator": matched, "denominator": ball_frames},
        {"metric": "missed_ball_frames", "numerator": missed, "denominator": ball_frames},
        {"metric": "empty_false_alarm_frames", "numerator": empty_false_alarms, "denominator": empty_frames},
        {"metric": "extra_boxes", "numerator": extra_boxes, "denominator": ""},
    ]
    with (HERE / "task05-totals.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["metric", "numerator", "denominator"])
        writer.writeheader()
        writer.writerows(totals)

    source_rows = []
    for source in sorted({row["source"] for row in lab.read_annotations()},
                         key=lambda name: (SPLIT_ORDER[min(
                             item["split"] for item in lab.read_annotations() if item["source"] == name
                         )], name)):
        splits = {item["split"] for item in lab.read_annotations() if item["source"] == source}
        if len(splits) != 1:
            raise ValueError(f"{source} 仍跨多个 split: {splits}")
        frames = sorted(item["frame"] for item in lab.read_annotations() if item["source"] == source)
        source_rows.append(
            {
                "split": splits.pop(),
                "source": source,
                "frame_count": len(frames),
                "frames": ",".join(str(frame) for frame in frames),
            }
        )
    with (HERE / "task05-splits.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(source_rows[0]))
        writer.writeheader()
        writer.writerows(source_rows)

    audit = {
        "corrections": CORRECTIONS,
        "corrected_sources": {
            split: [row["source"] for row in source_rows if row["split"] == split]
            for split in ("train", "val", "test")
        },
        "crossings_remaining": 0,
    }
    (HERE / "task05-split-audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    figure = build_match_figure(rows)
    print(f"figure: {figure}")
    print(json.dumps({"totals": totals, "splits": source_rows}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
