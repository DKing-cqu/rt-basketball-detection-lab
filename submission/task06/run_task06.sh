#!/usr/bin/env bash
# Reproduce the task 06 ONNX export, backend comparison, and tables.
# Usage (from the repository root):
#   bash submission/task06/run_task06.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PY="$ROOT/.venv/bin/python"

"$PY" submission/task06/export_onnx.py 2>&1 | tee submission/task06/export.log
"$PY" compare_onnx.py --onnx submission/task06/yolo11n-416.onnx \
  --output submission/task06/backend-comparison.json 2>&1 | tee submission/task06/compare.log
"$PY" submission/task06/build_task06.py
