#!/usr/bin/env bash
# Usage: source scripts/default_sfm/activate_env.sh
sfm_repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if ! declare -F conda >/dev/null; then
  sfm_conda_exe="${SFM_CONDA_EXE:-${CONDA_EXE:-/home/crestlux/miniforge3/bin/conda}}"
  source "$("$sfm_conda_exe" info --base)/etc/profile.d/conda.sh"
fi
conda activate "${SFM_ENV_NAME:-turbovla-sfm}"
export USE_TF=0
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=online
export SFM_DATA_ROOT="${SFM_DATA_ROOT:-$sfm_repo_root/data/default_sfm}"
export DINOV3_ROOT="${DINOV3_ROOT:-$sfm_repo_root/pretrained/backbones/dinov3-vitb16}"
export BERT_ROOT="${BERT_ROOT:-$sfm_repo_root/pretrained/backbones/bert-base-uncased}"
unset sfm_repo_root sfm_conda_exe
