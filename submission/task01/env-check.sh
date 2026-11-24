#!/usr/bin/env bash
# Reproduce the task 01 environment record.
# Usage (from the repository root):
#   bash submission/task01/env-check.sh | tee submission/task01/env.txt
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PY="$ROOT/.venv/bin/python"

if [ ! -x "$PY" ]; then
  echo "missing interpreter: $PY" >&2
  exit 1
fi

echo "== repository =="
pwd
echo
echo "== project interpreter =="
"$PY" - <<'PY'
import sys
print("sys.executable:", sys.executable)
print("python:", sys.version.replace("\n", " "))
PY
echo
echo "== venv config =="
cat .venv/pyvenv.cfg
echo
echo "== required packages =="
"$PY" - <<'PY'
import platform

import cv2
import numpy as np
import onnx
import onnxruntime
import torch
import torchvision
import ultralytics

print("platform:", platform.platform())
print("PyTorch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("Ultralytics:", ultralytics.__version__)
print("OpenCV:", cv2.__version__)
print("NumPy:", np.__version__)
print("ONNX:", onnx.__version__)
print("ONNX Runtime:", onnxruntime.__version__)
print("ONNX Runtime providers:", onnxruntime.get_available_providers())
PY
echo
echo "== CPU execution check =="
"$PY" - <<'PY'
import torch

print("torch.version.cuda:", torch.version.cuda)
print("torch.cuda.is_available():", torch.cuda.is_available())
print("torch.cuda.device_count():", torch.cuda.device_count())
print("torch.backends.mkldnn.is_available():", torch.backends.mkldnn.is_available())
print("default CPU threads:", torch.get_num_threads())
print("lab.py replay/inference uses device='cpu' and torch.set_num_threads(3)")
PY
echo
echo "== pip pins =="
"$PY" -m pip freeze | grep -E '^(torch|torchvision|ultralytics|opencv-python|numpy|onnx|onnxruntime)==' | sort
