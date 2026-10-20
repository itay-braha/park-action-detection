#!/usr/bin/env bash
# Full LART environment on Linux: AVA replication + the video demo (detectron2 ViTDet, HMR 2.0, PHALP, MViT).
#
#   bash setup_linux.sh --smpl /path/to/basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl
#   TORCH_SPEC="torch==2.0.1 torchvision==0.15.2" TORCH_INDEX=https://download.pytorch.org/whl/cu118 bash setup_linux.sh ...
#
# Requirements on the machine: NVIDIA driver, conda (or mamba), gcc/g++ <= 12, and a CUDA toolkit (nvcc) whose
# MAJOR.MINOR matches the PyTorch wheel - detectron2 and neural-renderer are compiled against it.
#   check:  nvidia-smi ; nvcc --version ; gcc --version
# Defaults = the versions the authors used (PyTorch 2.0 + CUDA 11.7), which suit pre-Hopper GPUs (V100/A100/RTX 20-40).
# For H100 or RTX 50xx you need e.g. TORCH_SPEC="torch==2.7.1 torchvision==0.22.1" TORCH_INDEX=.../cu128 and CUDA 12.8.
set -euo pipefail
cd "$(dirname "$0")"

SMPL=""
while [[ $# -gt 0 ]]; do case "$1" in --smpl) SMPL="$2"; shift 2;; *) echo "unknown arg $1"; exit 1;; esac; done

ENV_NAME=${ENV_NAME:-lart}
PY_VER=${PY_VER:-3.10}
TORCH_SPEC=${TORCH_SPEC:-"torch==2.0.0 torchvision==0.15.1"}
TORCH_INDEX=${TORCH_INDEX:-https://download.pytorch.org/whl/cu117}

source "$(conda info --base)/etc/profile.d/conda.sh"
conda env list | grep -q "^${ENV_NAME} " || conda create -y -n "$ENV_NAME" python="$PY_VER"
conda activate "$ENV_NAME"
python -m pip install --upgrade pip wheel setuptools

echo "== PyTorch ($TORCH_SPEC from $TORCH_INDEX)"
python -m pip install $TORCH_SPEC --index-url "$TORCH_INDEX"
python - <<'EOF'
import torch, subprocess, re
print("torch", torch.__version__, "cuda", torch.version.cuda, "| GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
try:
    nvcc = subprocess.run(["nvcc", "--version"], capture_output=True, text=True).stdout
    v = re.search(r"release (\d+\.\d+)", nvcc).group(1)
    if v != torch.version.cuda:
        print(f"WARNING: nvcc {v} != torch CUDA {torch.version.cuda}; detectron2 / neural-renderer builds may fail")
except Exception:
    print("WARNING: nvcc not found - install a CUDA toolkit matching", torch.version.cuda)
EOF

echo "== pure-Python dependencies"
python -m pip install -r requirements-linux.txt

echo "== compiled packages (built against the torch above, hence --no-build-isolation)"
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-"$(python -c 'import torch;c=torch.cuda.get_device_capability();print(f"{c[0]}.{c[1]}")')"}
python -m pip install --no-build-isolation "git+https://github.com/facebookresearch/detectron2.git"
python -m pip install --no-build-isolation "git+https://github.com/shubham-goel/NMR.git"          # neural-renderer-pytorch
python -m pip install --no-deps "git+https://github.com/facebookresearch/pytorchvideo.git"
python -m pip install --no-deps "git+https://github.com/brjathu/SlowFast.git"                  # MViT feature extractor

echo "== pinned + patched LART / PHALP / 4D-Humans, assets, SMPL"
python tools/setup_lart.py --demo ${SMPL:+--smpl "$SMPL"}
# install the three repos without their declared deps (chumpy & pytube are broken/not needed: we convert SMPL ourselves)
python -m pip install --no-deps -e third_party/PHALP -e third_party/4D-Humans -e third_party/LART

echo "== headless OpenGL for mesh rendering (PHALP/pyrender): EGL needs libegl1, OSMesa needs libosmesa6"
python - <<'EOF'
import os
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
try:
    import pyrender
    r = pyrender.OffscreenRenderer(64, 64); r.delete(); print("pyrender EGL offscreen: OK")
except Exception as e:
    print("pyrender EGL failed:", e, "\n  -> try: export PYOPENGL_PLATFORM=osmesa  (apt install libosmesa6-dev)")
EOF

python -c "import detectron2, neural_renderer, hmr2, phalp, lart, slowfast; print('imports OK')"
echo
echo "Done. Next:"
echo "  conda activate $ENV_NAME"
echo "  python tools/download_ava_val.py                        # 27 GB, for the AVA replication"
echo "  python eval_ava.py                                      # -> ~45.1 mAP"
echo "  PYOPENGL_PLATFORM=egl python run_demo.py --video third_party/LART/assets/jump.mp4"
