#!/usr/bin/env bash
# Run after the GPUs are free. Uses the real recipe for eight bounded updates.
set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/activate_env.sh"
cd "${TURBOVLA_ROOT}"
active_gpu_pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
if [[ -n "${active_gpu_pids}" ]]; then
    echo "GPUs are in use. Run this benchmark after the current training/evaluation finishes." >&2
    exit 1
fi
export RUN_ID="${RUN_ID:-turbovla_preprocessing_bench_$(date +%Y%m%d_%H%M%S)}"
export CUDA_VISIBLE_DEVICES=0,1 NUM_PROCESSES=2
export PER_DEVICE_BATCH_SIZE=12 GRADIENT_ACCUMULATION_STEPS=8
export MAX_TRAIN_STEPS=8 SAVE_INTERVAL=1000 LOGGING_FREQUENCY=1 WARMUP_STEPS=1000
export TRAIN_ENTRYPOINT=scripts/robotwin/benchmark_train.py
export ACCELERATE_CONFIG=experiments/robotwin/configs/deepspeed_zero1.yaml
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 MAX_JOBS=4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR="${CONDA_PREFIX}/var/triton" TRITON_HOME="${CONDA_PREFIX}/var/triton-home"
bash "${SCRIPT_DIR}/train.sh" \
    --trainer.save_format safetensors --trainer.freeze_modules model.text_encoder.bert \
    --datasets.vla_data.num_workers 4 --datasets.vla_data.prefetch_factor 2 \
    --datasets.vla_data.video_backend_kwargs.thread_count 2 \
    --datasets.vla_data.video_backend_kwargs.close_codec true
