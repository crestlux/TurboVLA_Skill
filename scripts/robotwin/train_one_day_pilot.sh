#!/usr/bin/env bash
# Short diagnostic run; this does not reproduce the paper's 55k training budget.
set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export RUN_ID="${RUN_ID:-turbovla_clean50_pilot5k_2x3090_$(date +%Y%m%d_%H%M%S)}"
export CUDA_VISIBLE_DEVICES=0,1 NUM_PROCESSES=2
export PER_DEVICE_BATCH_SIZE=12 GRADIENT_ACCUMULATION_STEPS=8
export MAX_TRAIN_STEPS=5000 SAVE_INTERVAL=1000 WARMUP_STEPS=1000
export LEARNING_RATE=5e-5 EMA_DECAY=0.999 LOGGING_FREQUENCY=50
export ACCELERATE_CONFIG=experiments/robotwin/configs/deepspeed_zero1.yaml
export ROBOTWIN_TEST_NUM=10
exec bash "${SCRIPT_DIR}/train_then_evaluate.sh"
