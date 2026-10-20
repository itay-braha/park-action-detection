#!/usr/bin/env bash
# The notebook's Part D experiments as CLI runs (results cached in outputs/nb_*/summary.json, which the notebook loads).
#   bash tools/run_ablations.sh            (from lart_replication/, after the AVA val download)
set -e
cd "$(dirname "$0")/.."
PY=${PY:-python}
COMMON="--videos 4 --batch-size 16 --workers 6"
run() { name=$1; shift; [ -f "outputs/nb_$name/summary.json" ] && { echo "[cached] $name"; return; }
        echo "[run] $name $*"; $PY eval_ava.py $COMMON --name "nb_$name" "$@" 2>&1 | grep -E "frame-mAP|Error|error" || true; }
run full
run ctx0 --ablation context_pm0
run ctx6 --ablation context_pm6
run ctx30 --ablation context_pm30
run seed1 --seed 1
run seed2 --seed 2
run no_others --ablation no_others
run no_appearance --ablation no_appearance
run no_pose --ablation no_pose
run oracle_GT --test-type track.fullframe@GT
echo "all ablations done"
