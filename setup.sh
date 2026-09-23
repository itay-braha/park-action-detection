#!/usr/bin/env bash
# One-shot environment setup for Linux (and macOS for CPU-only tinkering).
#   bash setup.sh                  # CUDA 12.8 wheels (RTX 20xx..50xx, A100, H100 with a recent driver)
#   TORCH_CUDA=cu126 bash setup.sh # pick another wheel: cu118, cu126, cu128, cpu ...
#   SKIP_WEIGHTS=1 bash setup.sh   # do not download the 1.1 GB checkpoint
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python3}
TORCH_CUDA=${TORCH_CUDA:-cu128}

"$PY" -c 'import sys; assert sys.version_info >= (3, 9), "need Python >= 3.9"'
[ -d .venv ] || "$PY" -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch torchvision --index-url "https://download.pytorch.org/whl/${TORCH_CUDA}"
python -m pip install -r requirements.txt
if [ "${SKIP_WEIGHTS:-0}" = "1" ]; then
  python tools/setup_tuber.py
else
  python tools/setup_tuber.py --weights csn152-ava22
fi
python -c "import torch; print('torch', torch.__version__, '| CUDA available:', torch.cuda.is_available())"
echo "Done. Activate with:  source .venv/bin/activate   then try:  python tests/smoke_test.py"
