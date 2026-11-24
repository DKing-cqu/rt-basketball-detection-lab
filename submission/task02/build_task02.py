"""Build the task 02 CVAT package and box-restoration images.

Run from the repository root after `lab.py prepare`:
    .venv/bin/python submission/task02/build_task02.py

The package follows CVAT's `Ultralytics YOLO Detection` export layout:
    data.yaml
    train.txt
    images/train/*.jpg
    labels/train/*.txt

The restoration images are drawn from the coordinates read back out of the
ZIP, not from annotations.csv, so the check covers the exported labels.
"""
import csv
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

import cv2

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

import lab  # noqa: E402

TARGETS = ("rt-125", "rt-150", "rt-200", "plos02-055")
RESTORE = ("rt-150", "plos02-055")
ZIP_PATH = HERE / "cvat-ultralytics-yolo-detection.zip"


def load_rows():
    rows = {}
    for row in lab.read_annotations():
        if row["stem"] in TARGETS:
            rows[row["stem"]] = row
    missing = [stem for stem in TARGETS if stem not in rows]
    if missing:
        raise ValueError(f"annotations.csv 缺少: {missing}")
    return rows


def normalize(box, width, height):
    x1, y1, x2, y2 = box
    return (
        (x1 + x2) / 2 / width,
        (y1 + y2) / 2 / height,
        (x2 - x1) / width,
        (y2 - y1) / height,
    )


def label_line(row):
    if not row["box"]:
        return ""
    cx, cy, bw, bh = normalize(row["box"], row["width"], row["height"])
    return f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n"


def source_path(row):
    return ROOT / "dataset-curated" / "images" / row["split"] / (row["stem"] + ".jpg")


def draw_restored(stem, row, label_text, output):
    image = cv2.imread(str(source_path(row)))
    if image is None:
        raise FileNotFoundError(source_path(row))
    height, width = image.shape[:2]
    fields = label_text.split()
    if len(fields) != 5:
        raise ValueError(f"{stem} 标签不是 detection 五元组: {label_text!r}")
    class_id = int(fields[0])
    cx, cy, bw, bh = map(float, fields[1:])
    if class_id != 0:
        raise ValueError(f"{stem} 类别编号不是 0: {class_id}")

    x1 = round((cx - bw / 2) * width)
    y1 = round((cy - bh / 2) * height)
    x2 = round((cx + bw / 2) * width)
    y2 = round((cy + bh / 2) * height)
    expected = row["box"]
    error = max(abs(a - b) for a, b in zip((x1, y1, x2, y2), expected))
    if error > 1:
        raise ValueError(f"{stem} 还原框偏差 {error}px，超出 1px")

    thickness = 2 if width < 1000 else 5
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), thickness)

    text_lines = [
        f"{stem}.jpg  {width}x{height}",
        f"basketball  xyxy=({x1},{y1},{x2},{y2})",
    ]
    font_scale = 0.4 if width < 1000 else 1.0
    text_thickness = 1 if width < 1000 else 2
    sizes = [cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness)[0] for line in text_lines]
    text_w = max(size[0] for size in sizes)
    text_h = max(size[1] for size in sizes)
    line_gap = 6 if width < 1000 else 12
    bar_h = text_h * 2 + line_gap + 16
    cv2.rectangle(image, (0, 0), (text_w + 16, bar_h), (0, 0, 0), -1)
    for index, line in enumerate(text_lines):
        y = 8 + text_h + index * (text_h + line_gap)
        cv2.putText(
            image, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX,
            font_scale, (255, 255, 255), text_thickness, cv2.LINE_AA,
        )
    if not cv2.imwrite(str(output), image):
        raise OSError(output)
    return (x1, y1, x2, y2), error


def main():
    rows = load_rows()

    for stem, row in rows.items():
        image = cv2.imread(str(source_path(row)))
        if image is None:
            raise FileNotFoundError(source_path(row))
        height, width = image.shape[:2]
        if (width, height) != (row["width"], row["height"]):
            raise ValueError(f"{stem} 尺寸不符: 图 {width}x{height}, 表 {row['width']}x{row['height']}")

    staging = Path(tempfile.mkdtemp(prefix="task02-cvat-"))
    try:
        (staging / "images" / "train").mkdir(parents=True)
        (staging / "labels" / "train").mkdir(parents=True)
        (staging / "data.yaml").write_text(
            "path: ./\ntrain: train.txt\nnames:\n  0: basketball\n",
            encoding="utf-8",
            newline="\n",
        )
        (staging / "train.txt").write_text(
            "".join(f"images/train/{stem}.jpg\n" for stem in TARGETS),
            encoding="utf-8",
            newline="\n",
        )
        for stem in TARGETS:
            row = rows[stem]
            shutil.copy2(source_path(row), staging / "images" / "train" / f"{stem}.jpg")
            (staging / "labels" / "train" / f"{stem}.txt").write_text(
                label_line(row), encoding="utf-8", newline="\n"
            )

        with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(staging.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(staging).as_posix())
    finally:
        shutil.rmtree(staging)

    check_rows = []
    with zipfile.ZipFile(ZIP_PATH) as archive:
        names = set(archive.namelist())
        for stem in TARGETS:
            row = rows[stem]
            label_name = f"labels/train/{stem}.txt"
            image_name = f"images/train/{stem}.jpg"
            if image_name not in names or label_name not in names:
                raise ValueError(f"ZIP 缺少 {stem} 的图片或标签")
            label_text = archive.read(label_name).decode("utf-8")
            lines = [line for line in label_text.splitlines() if line.strip()]
            check_rows.append(
                {
                    "filename": f"{stem}.jpg",
                    "width": row["width"],
                    "height": row["height"],
                    "has_label": "yes" if lines else "no",
                    "label_count": len(lines),
                    "class_id": 0 if lines else "",
                    "class_name": "basketball" if lines else "",
                    "bbox_xyxy": ",".join(str(value) for value in row["box"]) if row["box"] else "",
                    "normalized_label": lines[0] if lines else "",
                    "label_file": label_name,
                }
            )

    for stem in RESTORE:
        with zipfile.ZipFile(ZIP_PATH) as archive:
            label_text = archive.read(f"labels/train/{stem}.txt").decode("utf-8").strip()
        box, error = draw_restored(stem, rows[stem], label_text, HERE / f"{stem}-restored.png")
        print(f"{stem}: exported label → xyxy={box}, max error={error}px")

    with (HERE / "four-image-check.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(check_rows[0]))
        writer.writeheader()
        writer.writerows(check_rows)

    print(f"package: {ZIP_PATH} ({ZIP_PATH.stat().st_size} bytes)")
    for row in check_rows:
        print(row)


if __name__ == "__main__":
    main()
