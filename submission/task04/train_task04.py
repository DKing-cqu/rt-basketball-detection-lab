"""Train the task 04 full-finetune and frozen-first-10-layers schemes.

Run one scheme at a time (see run_task04.sh):
    .venv/bin/python submission/task04/train_task04.py --scheme full
    .venv/bin/python submission/task04/train_task04.py --scheme frozen10

Both schemes start from yolo11n.pt and use the chapter-3 settings, changing
only `freeze` and the run name.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import torch
from ultralytics import YOLO

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
WEIGHTS = ROOT / "yolo11n.pt"
DATA = ROOT / "data.yaml"

SCHEMES = {
    "full": {
        "freeze": 0,
        "run": "task04-full",
        "config": "full-finetune-config.json",
        "results": "full-finetune-results.csv",
        "args": "full-finetune-args.yaml",
        "freeze_check": "full-finetune-freeze-check.json",
    },
    "frozen10": {
        "freeze": 10,
        "run": "task04-frozen10",
        "config": "frozen10-config.json",
        "results": "frozen10-results.csv",
        "args": "frozen10-args.yaml",
        "freeze_check": "frozen10-freeze-check.json",
    },
}

COMMON = {
    "data": str(DATA),
    "device": "cpu",
    "imgsz": 416,
    "epochs": 5,
    "batch": 8,
    "workers": 0,
    "seed": 42,
    "deterministic": True,
    "optimizer": "AdamW",
    "lr0": 0.001,
    "mosaic": 0.0,
    "mixup": 0.0,
    "scale": 0.20,
    "plots": False,
    "project": str(ROOT / "runs"),
    "exist_ok": True,
    "verbose": True,
}


def freeze_names(freeze):
    return [f"model.{index}." for index in range(freeze)] + [".dfl"]


def set_freeze_state(model, freeze):
    names = freeze_names(freeze)
    for key, parameter in model.model.named_parameters():
        parameter.requires_grad = not any(name in key for name in names)


def parameter_counts(model):
    total = sum(parameter.numel() for parameter in model.model.parameters())
    trainable = sum(parameter.numel() for parameter in model.model.parameters() if parameter.requires_grad)
    return {"total": total, "trainable": trainable, "frozen": total - trainable}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scheme", choices=sorted(SCHEMES), required=True)
    args = parser.parse_args()
    scheme = SCHEMES[args.scheme]

    torch.set_num_threads(3)
    model = YOLO(str(WEIGHTS))
    set_freeze_state(model, scheme["freeze"])
    counts_before = parameter_counts(model)
    frozen_before = {
        key: parameter.detach().clone()
        for key, parameter in model.model.named_parameters()
        if not parameter.requires_grad
    }

    train_args = dict(COMMON)
    train_args.update(freeze=scheme["freeze"], name=scheme["run"])
    config = {
        "scheme": args.scheme,
        "weights": "yolo11n.pt",
        **train_args,
    }
    (HERE / scheme["config"]).write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    model.train(**train_args)

    after = dict(model.model.named_parameters())
    missing = [key for key in frozen_before if key not in after]
    max_diff = 0.0
    if not missing:
        for key, before in frozen_before.items():
            max_diff = max(max_diff, (after[key].detach() - before).abs().max().item())
    check = {
        "scheme": args.scheme,
        "freeze": scheme["freeze"],
        "frozen_layers": list(range(scheme["freeze"])),
        "always_frozen_by_ultralytics": [".dfl"],
        "frozen_parameter_tensors": len(frozen_before),
        "missing_after_training": missing,
        "frozen_max_abs_diff_after_training": max_diff,
        "frozen_unchanged": not missing and max_diff == 0.0,
        **counts_before,
    }
    (HERE / scheme["freeze_check"]).write_text(
        json.dumps(check, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    run_dir = ROOT / "runs" / scheme["run"]
    shutil.copy2(run_dir / "results.csv", HERE / scheme["results"])
    shutil.copy2(run_dir / "args.yaml", HERE / scheme["args"])
    print(json.dumps(check, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
