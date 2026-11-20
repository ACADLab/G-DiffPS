#!/bin/bash
set -euo pipefail
export PDK_ROOT="/Users/deepak/Desktop/work/G-DiffPS/pdk"
export PATH="/opt/homebrew/bin:$PATH"
cd /Users/deepak/Desktop/work/G-DiffPS
PY=/opt/homebrew/anaconda3/envs/gdiffps/bin/python
ROOT=results/long_runs
SUMMARY="$ROOT/master_log.txt"
echo "START $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee "$SUMMARY"

run_one () {
  local steps=$1
  local name=$2
  local out="$ROOT/$name"
  echo "" | tee -a "$SUMMARY"
  echo "=== $name  steps=$steps  $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a "$SUMMARY"
  mkdir -p "$out"
  /usr/bin/time -p $PY train_diffusion.py \
    --total-timesteps "$steps" \
    --actor cfm --sizing log \
    --encoder gin --action-space device \
    --bounds electrical --switch-model ideal \
    --batch-size 32 --buffer-size 5000 --seed 42 \
    --run-dir "$out" \
    2>&1 | tee "$ROOT/${name}.console.log"
  echo "DONE $name exit=$? $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$SUMMARY"
}

run_one 20000 train_gin_device_20k
run_one 30000 train_gin_device_30k
run_one 50000 train_gin_device_50k

echo "ALL_DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$SUMMARY"
