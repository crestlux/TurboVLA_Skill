#!/usr/bin/env bash
# Source this file to activate the prepared environment on jack's workstation.
# This file only sets up the shell; it does not launch training or evaluation.
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo "Use: source /home/jack/TurboVLA/scripts/robotwin/activate_env.sh" >&2
    exit 1
fi

source /home/jack/miniforge3/etc/profile.d/conda.sh || return 1
conda activate turbovla-robotwin || return 1

export TURBOVLA_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
export STARVLA_PYTHON="${CONDA_PREFIX}/bin/python"
export CUDA_HOME="${CONDA_PREFIX}"
export ROBOTWIN_DATA_ROOT=/home/jack/datasets/RoboTwin
export BERT_MODEL_PATH="${TURBOVLA_ROOT}/pretrained/backbones/bert-base-uncased"
export TURBOVLA_INIT_CKPT="${TURBOVLA_ROOT}/pretrained/backbones/groundingdino/groundingdino_swint_ogc.pth"
export DINOV3_MODEL_PATH="${TURBOVLA_ROOT}/pretrained/backbones/dinov3-vitl16"
export ROBOTWIN_STORE=/data/jack/robotwin2_turbovla
export RUN_ROOT_DIR="${ROBOTWIN_STORE}/runs"
export USE_TF=0
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled
