"""Build the task 04 per-epoch, parameter, and fixed-evaluation tables.

Prerequisites (see run_task04.sh):
    full-finetune-results.csv, frozen10-results.csv
    full-finetune-freeze-check.json, frozen10-freeze-check.json
    eval-full.summary.json, eval-frozen10.summary.json

Run from the repository root:
    .venv/bin/python submission/task04/build_task04.py
"""
import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

SCHEMES = {
    "full": {
        "results": "full-finetune-results.csv",
        "freeze": "full-finetune-freeze-check.json",
        "eval": "eval-full.summary.json",
        "config": "full-finetune-config.json",
    },
    "frozen10": {
        "results": "frozen10-results.csv",
        "freeze": "frozen10-freeze-check.json",
        "eval": "eval-frozen10.summary.json",
        "config": "frozen10-config.json",
    },
}


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def main():
    epoch_rows = []
    parameter_rows = []
    eval_rows = []

    configs = {}
    for scheme, files in SCHEMES.items():
        configs[scheme] = json.loads((HERE / files["config"]).read_text())
        for row in read_csv(HERE / files["results"]):
            epoch_rows.append(
                {
                    "scheme": scheme,
                    "epoch": row["epoch"],
                    "time_s": row["time"],
                    "train_box_loss": row["train/box_loss"],
                    "train_cls_loss": row["train/cls_loss"],
                    "train_dfl_loss": row["train/dfl_loss"],
                    "precision": row["metrics/precision(B)"],
                    "recall": row["metrics/recall(B)"],
                    "mAP50": row["metrics/mAP50(B)"],
                    "mAP50_95": row["metrics/mAP50-95(B)"],
                    "val_box_loss": row["val/box_loss"],
                    "val_cls_loss": row["val/cls_loss"],
                    "val_dfl_loss": row["val/dfl_loss"],
                    "lr_pg0": row["lr/pg0"],
                }
            )

        check = json.loads((HERE / files["freeze"]).read_text())
        parameter_rows.append(
            {
                "scheme": scheme,
                "freeze": check["freeze"],
                "frozen_layers": ",".join(str(layer) for layer in check["frozen_layers"]) or "-",
                "total_parameters": check["total"],
                "trainable_parameters": check["trainable"],
                "frozen_parameters": check["frozen"],
                "frozen_parameter_tensors": check["frozen_parameter_tensors"],
                "frozen_unchanged": str(check["frozen_unchanged"]).lower(),
                "frozen_max_abs_diff": check["frozen_max_abs_diff_after_training"],
            }
        )

        summary = json.loads((HERE / files["eval"]).read_text())
        groups = summary["groups"]
        regular = groups["regular"]
        edge = groups["edge"]
        empty = groups["empty"]
        eval_rows.append(
            {
                "scheme": scheme,
                "weights": summary["weights"],
                "imgsz": summary["imgsz"],
                "conf": summary["conf"],
                "match_iou": summary["match_iou"],
                "regular_hits": regular["hits"],
                "regular_total": regular["frames"],
                "regular_fraction": f"{regular['hits']}/{regular['frames']}",
                "edge_hits": edge["hits"],
                "edge_total": edge["frames"],
                "edge_fraction": f"{edge['hits']}/{edge['frames']}",
                "empty_false_alarms": empty["false_alarm_frames"],
                "empty_total": empty["frames"],
                "empty_fraction": f"{empty['false_alarm_frames']}/{empty['frames']}",
                "ball_hits": regular["hits"] + edge["hits"],
                "ball_total": regular["frames"] + edge["frames"],
                "ball_fraction": f"{regular['hits'] + edge['hits']}/{regular['frames'] + edge['frames']}",
            }
        )

    base = dict(configs["full"])
    base.pop("scheme", None)
    base.pop("freeze", None)
    base.pop("name", None)
    current = dict(configs["frozen10"])
    current.pop("scheme", None)
    current.pop("freeze", None)
    current.pop("name", None)
    if base != current:
        raise ValueError("两组配置除 freeze/name 外仍有差异")

    for name, rows in (
        ("task04-epochs.csv", epoch_rows),
        ("task04-parameters.csv", parameter_rows),
        ("task04-eval.csv", eval_rows),
    ):
        with (HERE / name).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    print(json.dumps({"epochs": epoch_rows, "parameters": parameter_rows, "eval": eval_rows},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
