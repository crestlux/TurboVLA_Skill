#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/activate_env.sh"
cd "$TURBOVLA_ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export PYTHONPATH="$TURBOVLA_ROOT:$TURBOVLA_ROOT/third_party/starvla_runtime:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=disabled
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export MAX_JOBS=4
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-^lo,docker0,virbr0,veth}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TRITON_CACHE_DIR="$CONDA_PREFIX/var/triton"
export TRITON_HOME="$CONDA_PREFIX/var/triton-home"
mkdir -p "$TRITON_CACHE_DIR" "$TRITON_HOME/autotune"

run_id="${RUN_ID:-turbovla_clean50_smoke_$(date +%Y%m%d_%H%M%S)}"
output_dir="$RUN_ROOT_DIR/$run_id"
if [[ -e "$output_dir" ]]; then
    echo "Output already exists: $output_dir" >&2
    exit 1
fi
mkdir -p "$output_dir/provenance"
cp scripts/robotwin/smoke_train.{sh,py} "$output_dir/provenance/"
cp third_party/starvla_runtime/starVLA/training/train{_starvla,_robotwin_clean_act_pi05_recipe}.py "$output_dir/provenance/"
git rev-parse HEAD > "$output_dir/provenance/repo.commit"
git diff > "$output_dir/provenance/source.patch"
python -m pip freeze > "$output_dir/provenance/pip-freeze.txt"
date --iso-8601=seconds > "$output_dir/started_at.txt"
echo "[SMOKE] output_dir=$output_dir"

set +e
"$STARVLA_PYTHON" -m accelerate.commands.launch \
  --config_file experiments/robotwin/configs/deepspeed_zero2.yaml \
  --num_processes 2 --main_process_port "${MAIN_PROCESS_PORT:-29631}" \
  scripts/robotwin/smoke_train.py \
  --config_yaml experiments/robotwin/configs/clean50.yaml \
  --run_root_dir "$RUN_ROOT_DIR" --run_id "$run_id" \
  --datasets.vla_data.per_device_batch_size "${PER_DEVICE_BATCH_SIZE:-6}" \
  --datasets.vla_data.num_workers 4 --datasets.vla_data.prefetch_factor 2 \
  --trainer.gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-16}" \
  --trainer.max_train_steps 20 --trainer.num_warmup_steps 5 \
  --trainer.eval_interval 10 --trainer.save_interval 20 --trainer.logging_frequency 1 \
  --trainer.save_format safetensors --trainer.freeze_modules model.text_encoder.bert \
  2>&1 | tee "$output_dir/train.log"
smoke_exit=${PIPESTATUS[0]}
printf '%s\n' "$smoke_exit" > "$output_dir/exit_code.txt"
date --iso-8601=seconds > "$output_dir/finished_at.txt"
exit "$smoke_exit"
