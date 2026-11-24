#!/usr/bin/env bash
# Reproduce the task 05 split audit, Jackson independent test, and match figure.
# Usage (from the repository root):
#   bash submission/task05/run_task05.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PY="$ROOT/.venv/bin/python"

"$PY" lab.py eval --weights yolo11n.pt --split test --imgsz 416 --conf 0.10 \
  --output submission/task05/eval-jackson.jsonl
"$PY" submission/task05/build_task05.py
