#!/usr/bin/env bash
# Reproduce the task 03 evaluation and comparison figures.
# Usage (from the repository root):
#   bash submission/task03/run_task03.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PY="$ROOT/.venv/bin/python"

"$PY" lab.py eval --weights yolo11n.pt --split val --imgsz 640 --conf 0.10 \
  --output submission/task03/eval-A.jsonl
"$PY" lab.py eval --weights yolo11n.pt --split val --imgsz 320 --conf 0.10 \
  --output submission/task03/eval-B.jsonl
"$PY" lab.py eval --weights yolo11n.pt --split val --imgsz 320 --conf 0.25 \
  --output submission/task03/eval-C.jsonl
"$PY" submission/task03/build_task03.py
