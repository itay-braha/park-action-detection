#!/usr/bin/env bash
# Full LART environment on Linux: AVA replication + the video demo (detectron2 ViTDet, HMR 2.0, PHALP, MViT).
#
#   bash setup_linux.sh --smpl /path/to/basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl
#   TORCH_SPEC="torch==2.0.1 torchvision==0.15.2" TORCH_INDEX=https://download.pytorch.org/whl/cu118 bash setup_linux.sh ...
#
# Requirements on the machine: NVIDIA driver, conda (or mamba), gcc/g++ <= 12, and a CUDA toolkit (nvcc) whose
# MAJOR.MINOR matches the PyTorch wheel - detectron2 and neural-renderer are compiled against it.
#   check:  nvidia-smi ; nvcc --version ; gcc --version
# Defaults = the versions the authors used (PyTorch 2.0 + CUDA 11.7): right for Titan RTX / RTX 20-40 / V100 / A100.
# For H100 or RTX 50xx you need e.g. TORCH_SPEC="torch==2.7.1 torchvision==0.22.1" TORCH_INDEX=.../cu128 and CUDA 12.8.
set -eo pipefail     # no -u: conda's compiler (de)activate scripts read unset vars (ADDR2LINE, CONDA_BACKUP_*)
cd "$(dirname "$0")"

SMPL=""
while [[ $# -gt 0 ]]; do case "$1" in --smpl) SMPL="$2"; shift 2;; *) echo "unknown arg $1"; exit 1;; esac; done

ENV_NAME=${ENV_NAME:-cvproject}
PY_VER=${PY_VER:-3.10}
TORCH_SPEC=${TORCH_SPEC:-"torch==2.0.0 torchvision==0.15.1"}
TORCH_INDEX=${TORCH_INDEX:-https://download.pytorch.org/whl/cu117}

source "$(conda info --base)/etc/profile.d/conda.sh"
conda env list | grep -q "^${ENV_NAME} " || conda create -y -n "$ENV_NAME" python="$PY_VER"
conda activate "$ENV_NAME"
python -m pip install --upgrade pip wheel setuptools

echo "== PyTorch ($TORCH_SPEC from $TORCH_INDEX)"
python -m pip install $TORCH_SPEC --index-url "$TORCH_INDEX"
TORCH_CUDA=$(python -c "import torch; print(torch.version.cuda)")
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, '| GPUs:', [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])"

# detectron2 + neural-renderer are compiled: they need nvcc == torch's CUDA version, and a gcc that CUDA accepts
# (CUDA 11.x: gcc <= 11). If the machine doesn't have them, install both INSIDE the conda env (system untouched).
NVCC_VER=$(nvcc --version 2>/dev/null | sed -n 's/.*release \([0-9]*\.[0-9]*\).*/\1/p' || true)
if [[ "$NVCC_VER" != "$TORCH_CUDA" ]]; then
    echo "== nvcc '${NVCC_VER:-none}' != torch CUDA $TORCH_CUDA -> installing cuda-toolkit $TORCH_CUDA into the env"
    conda install -y -c "nvidia/label/cuda-${TORCH_CUDA}.0" cuda-toolkit || conda install -y -c nvidia "cuda-toolkit=${TORCH_CUDA}"
    export CUDA_HOME="$CONDA_PREFIX"
fi
GCC_MAJOR=$(gcc -dumpversion 2>/dev/null | cut -d. -f1 || echo 99)
if [[ "${TORCH_CUDA%%.*}" == "11" && "${GCC_MAJOR:-99}" -gt 11 ]]; then
    echo "== gcc $GCC_MAJOR is too new for CUDA $TORCH_CUDA -> installing gcc/g++ 11 into the env"
    conda install -y -c conda-forge "gxx_linux-64=11" "gcc_linux-64=11"
    export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc" CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"
fi
command -v ffmpeg >/dev/null || conda install -y -c conda-forge ffmpeg     # needed by process_videos.py

echo "== pure-Python dependencies"
python -m pip install -r requirements-linux.txt

echo "== compiled packages (built against the torch above, hence --no-build-isolation)"
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-"$(python -c 'import torch;c=torch.cuda.get_device_capability();print(f"{c[0]}.{c[1]}")')"}
python -c "import detectron2" 2>/dev/null ||
    python -m pip install --no-build-isolation "git+https://github.com/facebookresearch/detectron2.git"
# neural-renderer-pytorch: AT_DISPATCH_*(x.type(), ...) no longer compiles on PyTorch >= 2.2 -> x.scalar_type()
if ! python -c "import neural_renderer" 2>/dev/null; then
    rm -rf third_party/NMR && git clone -q https://github.com/shubham-goel/NMR.git third_party/NMR
    sed -i -E 's/(AT_DISPATCH_[A-Z_]+\(\s*[A-Za-z_]+)\.type\(\)/\1.scalar_type()/' third_party/NMR/neural_renderer/cuda/*.cu
    python -m pip install --no-build-isolation ./third_party/NMR
fi
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
echo "  PYOPENGL_PLATFORM=egl python run_demo.py video.source=third_party/LART/assets/jump.mp4   # authors sample"
echo "  python process_videos.py /path/to/park_video.mp4 --out outputs/test --chunk-min 1 --max-chunks 1   # timing test"
