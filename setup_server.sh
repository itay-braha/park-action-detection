#!/usr/bin/env bash
# One-time setup on the Linux GPU server, for BOTH models (TubeR + LART) in one conda env.
#   bash setup_server.sh --smpl /path/to/basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl
# Takes ~30-60 min (compiles detectron2 + downloads ~12 GB of weights). Safe to re-run.
set -eo pipefail     # no -u: conda's compiler (de)activate scripts read unset vars (ADDR2LINE, CONDA_BACKUP_*)
cd "$(dirname "$0")"
SMPL=""
while [[ $# -gt 0 ]]; do case "$1" in --smpl) SMPL="$(readlink -f "$2")"; shift 2;; *) echo "unknown arg $1"; exit 1;; esac; done
[[ -n "$SMPL" && -f "$SMPL" ]] || { echo "usage: bash setup_server.sh --smpl /path/to/basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl"; exit 1; }
export ENV_NAME=${ENV_NAME:-cvproject}

echo "######## 1/3  LART environment (PyTorch 2.0.1 + CUDA 11.7, detectron2, PHALP, HMR 2.0) ########"
TORCH_SPEC=${TORCH_SPEC:-"torch==2.0.1 torchvision==0.15.2"} bash lart_replication/setup_linux.sh --smpl "$SMPL"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
echo "######## 2/3  TubeR (same env) ########"
grep -vE "^(jupyterlab|ipykernel|kiwisolver)" requirements.txt > /tmp/req_tuber.txt
python -m pip install -r /tmp/req_tuber.txt "numpy<2" imageio-ffmpeg
python tools/setup_tuber.py --weights csn152-ava22

echo "######## 3/3  self-check ########"
python - <<'PY'
import torch
print("GPUs:", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
      "| bf16 supported:", torch.cuda.is_bf16_supported(), "(no -> TubeR uses fp16 automatically)")
import parktuber.model, sys; sys.path.insert(0, "lart_replication")
import lartlab, lart, phalp, hmr2, detectron2
print("TubeR and LART imports OK")
PY
echo
echo "Done. Every new ssh session:   conda activate $ENV_NAME"
echo "First test (2 minutes of video, ~minutes of compute):"
echo "  python run.py --model lart --video /path/video.mp4 --out results/test_lart --start 01:00:00 --duration 120 --gpus 0,1 --render"
