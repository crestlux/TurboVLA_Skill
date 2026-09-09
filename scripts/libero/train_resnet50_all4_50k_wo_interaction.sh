#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export EXPERIMENT_NAME="${EXPERIMENT_NAME:-libero_all4_50k_1gpu_bs32_resnet50_os16_tf456_warmup10k_wo_interaction}"
export CHECKPOINT_NAME="${CHECKPOINT_NAME:-turbovla_all4_50k_bs32_resnet50_os16_tf456_wo_interaction}"
export INTERACTION_LAYERS=0

exec bash "${script_dir}/train_resnet50_all4_50k.sh" "$@"
