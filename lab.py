"""Prepare, train, and compare a small basketball detection exercise."""
import argparse
import csv
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import subprocess
import time

import cv2
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "dataset-curated"
WEIGHTS = ROOT / "yolo11n.pt"
SPLITS = ("train", "val", "test")
MATCH_IOU = 0.50


def read_annotations():
    rows = []
    with (ROOT / "annotations.csv").open(encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source):
            for key in ("frame", "width", "height"):
                row[key] = int(row[key])
            row["time_s"] = float(row["time_s"])
            row["box"] = [int(row[k]) for k in ("x1", "y1", "x2", "y2")] if row["x1"] else []
            row["stem"] = f"{row['source']}-{row['frame']:03}"
            rows.append(row)
    return rows


def get_asset(path, url, digest):
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        pending = path.with_suffix(".part")
        subprocess.run(
            ["curl", "-fLsS", "--retry", "2", "--max-time", "60", "-A",
             "RT-vision-education-demo/1.0", "-o", str(pending), url], check=True,
        )
        if hashlib.sha256(pending.read_bytes()).hexdigest() != digest:
            raise ValueError(f"下载文件校验失败: {path.name}")
        pending.replace(path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise ValueError(f"现有文件校验失败: {path.name}")


def prepare():
    metadata = json.loads((ROOT / "sources.json").read_text(encoding="utf-8"))
    grouped = defaultdict(dict)
    for row in read_annotations():
        grouped[row["source"]][row["frame"]] = row
    for key, frames in grouped.items():
        asset = metadata[key]
        path = ROOT / asset["file"]
        get_asset(path, asset["url"], asset["sha256"])
        reader = cv2.VideoCapture(str(path))
        try:
            for index in range(max(frames) + 1):
                ok, image = reader.read()
                if not ok:
                    raise RuntimeError(f"无法解码: {key} 第 {index} 帧")
                if index in frames:
                    write_sample(image, frames[index])
        finally:
            reader.release()
    for split in SPLITS:
        (DATA / "labels" / (split + ".cache")).unlink(missing_ok=True)
    get_asset(WEIGHTS, metadata["weights"]["url"], metadata["weights"]["sha256"])
    (ROOT / "data.yaml").write_text(
        f"path: {DATA.as_posix()}\ntrain: images/train\nval: images/val\n"
        "test: images/test\nnames:\n  0: basketball\n", encoding="utf-8",
    )
    print("数据已准备，可打开 annotations.csv 核对来源和标注。")


def write_sample(image, row):
    w, h = row["width"], row["height"]
    if image.shape[:2] != (h, w):
        raise ValueError(f"图片尺寸不符: {row['stem']}")
    pictures = DATA / "images" / row["split"]
    labels = DATA / "labels" / row["split"]
    pictures.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(pictures / (row["stem"] + ".jpg")), image, [cv2.IMWRITE_JPEG_QUALITY, 92]):
        raise OSError(f"无法写图: {row['stem']}")
    line = ""
    if row["box"]:
        a, b, c, d = row["box"]
        line = f"0 {(a+c)/2/w:.8f} {(b+d)/2/h:.8f} {(c-a)/w:.8f} {(d-b)/h:.8f}\n"
    (labels / (row["stem"] + ".txt")).write_text(line, encoding="utf-8")


def train():
    model = YOLO(str(WEIGHTS))
    model.train(
        data=str(ROOT / "data.yaml"), device="cpu", imgsz=416, epochs=15, batch=8,
        workers=0, seed=42, deterministic=True, optimizer="AdamW", lr0=.001,
        mosaic=0.0, mixup=0.0, scale=.20, plots=False, project=str(ROOT / "runs"),
        name="lesson-fit", exist_ok=True, verbose=False,
    )


def load_detector(weights):
    detector = YOLO(str(weights))
    names = {name.lower(): index for index, name in detector.names.items()}
    target = names.get("basketball", names.get("sports ball"))
    if target is None:
        raise ValueError("权重缺少 basketball 或 sports ball 类别")
    return detector, target


def predict_frame(model, target, image, size, confidence):
    start = time.perf_counter()
    result = model.predict(
        image, classes=[target], imgsz=size, conf=confidence, device="cpu", verbose=False,
    )[0]
    detections = [
        {"bbox_xyxy": box, "class": model.names[target], "confidence": value}
        for box, value in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist())
    ]
    return detections, round((time.perf_counter() - start) * 1000, 2)


def iou(box_a, box_b):
    a, b, c, d = box_a
    e, f, g, h = box_b
    intersect = max(0, min(c, g) - max(a, e)) * max(0, min(d, h) - max(b, f))
    union = (c-a)*(d-b) + (g-e)*(h-f) - intersect
    return intersect / union if union > 0 else 0.0


def group(row):
    if not row["box"]:
        return "empty"
    x1, y1, x2, y2 = row["box"]
    if x1 == 0 or y1 == 0 or x2 == row["width"] or y2 == row["height"]:
        return "edge"
    scaled_short = min(x2-x1, y2-y1)*640/max(row["width"], row["height"])
    return "small" if scaled_short <= 36 else "regular"


def evaluate(args, model, target):
    source = read_annotations()
    report = defaultdict(lambda: {"frames": 0, "hits": 0, "misses": 0, "false_alarm_frames": 0, "extra_boxes": 0})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for row in source:
            if row["split"] != args.split:
                continue
            image = cv2.imread(str(DATA / "images" / args.split / (row["stem"] + ".jpg")))
            if image is None:
                raise FileNotFoundError(f"先准备数据: {row['stem']}")
            boxes, elapsed = predict_frame(model, target, image, args.imgsz, args.conf)
            label = row["box"]
            best_iou = max((iou(label, b["bbox_xyxy"]) for b in boxes), default=0) if label else 0
            hit = bool(label) and best_iou >= MATCH_IOU
            category = group(row)
            stats = report[category]
            stats["frames"] += 1
            stats["hits"] += hit
            stats["misses"] += bool(label) and not hit
            stats["false_alarm_frames"] += not label and bool(boxes)
            stats["extra_boxes"] += len(boxes) - int(hit)
            data = {key: row[key] for key in ("source", "frame", "time_s", "width", "height")}
            data.update(group=category, annotation_xyxy=label, detections=boxes,
                        best_iou=round(best_iou, 4), matched=hit, elapsed_ms=elapsed)
            stream.write(json.dumps(data, ensure_ascii=False) + "\n")
    summary = dict(weights=str(args.weights), imgsz=args.imgsz, conf=args.conf,
                   split=args.split, match_iou=MATCH_IOU, groups=report)
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def replay(args, model, target):
    reader = cv2.VideoCapture(str(args.source))
    if not reader.isOpened():
        raise FileNotFoundError(args.source)
    fps = reader.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        reader.release()
        raise ValueError("视频帧率无效")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with args.output.open("w", encoding="utf-8") as stream:
            frame = 0
            while args.limit is None or frame < args.limit:
                ok, image = reader.read()
                if not ok:
                    break
                boxes, elapsed = predict_frame(model, target, image, args.imgsz, args.conf)
                height, width = image.shape[:2]
                data = dict(frame=frame, time_s=round(frame/fps, 4), width=width,
                            height=height, detections=boxes, elapsed_ms=elapsed)
                stream.write(json.dumps(data, ensure_ascii=False) + "\n")
                frame += 1
    finally:
        reader.release()
    print(f"{frame} frames → {args.output}")


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    sub = cli.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare", help="下载来源视频并生成指定帧和标签")
    sub.add_parser("train", help="以 CPU 微调 15 轮")
    for name in ("eval", "replay"):
        cmd = sub.add_parser(name, help="评测指定素材" if name == "eval" else "逐帧运行视频")
        cmd.add_argument("--weights", type=Path, default=WEIGHTS)
        cmd.add_argument("--imgsz", type=int, default=640)
        cmd.add_argument("--conf", type=float, default=.25)
        cmd.add_argument("--output", type=Path, required=True)
        if name == "eval":
            cmd.add_argument("--split", choices=SPLITS, default="val")
        else:
            cmd.add_argument("--source", type=Path, required=True)
            cmd.add_argument("--limit", type=int)
    args = cli.parse_args()
    if args.command == "prepare":
        prepare()
        return
    torch.set_num_threads(3)
    if args.command == "train":
        train()
        return
    if args.imgsz < 32 or not 0 < args.conf < 1:
        cli.error("输入尺寸至少为 32，置信度应在 0 到 1 之间")
    model, target = load_detector(args.weights)
    if args.command == "eval":
        evaluate(args, model, target)
    else:
        replay(args, model, target)


if __name__ == "__main__":
    main()
