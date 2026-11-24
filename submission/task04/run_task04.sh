#!/usr/bin/env bash
# Reproduce the task 04 training, fixed evaluation, and summary tables.
# Usage (from the repository root):
#   bash submission/task04/run_task04.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PY="$ROOT/.venv/bin/python"

"$PY" submission/task04/train_task04.py --scheme full 2>&1 | tee submission/task04/train-full.log
"$PY" submission/task04/train_task04.py --scheme frozen10 2>&1 | tee submission/task04/train-frozen10.log

"$PY" lab.py eval --weights runs/task04-full/weights/best.pt --split val --imgsz 640 --conf 0.25 \
  --output submission/task04/eval-full.jsonl
"$PY" lab.py eval --weights runs/task04-frozen10/weights/best.pt --split val --imgsz 640 --conf 0.25 \
  --output submission/task04/eval-frozen10.jsonl

"$PY" submission/task04/build_task04.py
