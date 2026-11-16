#!/usr/bin/env bash
set -euo pipefail
cd /data/wangkaiyan2024/compared_methods
export CUDA_VISIBLE_DEVICES=0 MKL_THREADING_LAYER=GNU OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
PY=/home/wangkaiyan2024/miniconda3/envs/wbm/bin/python
RUN=artifacts/wapirl_pipeline_20261111
mkdir "$RUN"
trap 'code=$?; printf "%s\n" "$code" > "$RUN/exit_code.txt"' EXIT
"$PY" -u methods/wapirl/wapirl.py --config methods/wapirl/config.json --device cuda > "$RUN/train.log" 2>&1
"$PY" -u scripts/evaluate.py --config methods/wapirl/config.json --split valid --output-dir artifacts/wapirl_valid_v1 --device cuda > "$RUN/valid.log" 2>&1
"$PY" -u scripts/evaluate.py --config methods/wapirl/config.json --split test --output-dir artifacts/wapirl_test_v1 --device cuda > "$RUN/test.log" 2>&1
