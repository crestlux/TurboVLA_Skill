#!/usr/bin/env bash
# Pinned 2026-09-27 public 150k EMA release with explicit post-LayerNorm features.
set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export ROBOTWIN_TEST_NUM="${ROBOTWIN_TEST_NUM:-10}"
export ROBOTWIN_EVAL_MODE="${ROBOTWIN_EVAL_MODE:-demo_clean}"
export RUN_ID="${RUN_ID:-turbovla_release150000_postnorm_${ROBOTWIN_EVAL_MODE}_eval${ROBOTWIN_TEST_NUM}_$(date +%Y%m%d_%H%M%S)}"
checkpoint="${REPO_ROOT}/pretrained/converted/turbovla_robotwin_9cdb576e_150k_ema_postnorm/steps_150000_ema_pytorch_model.safetensors"
validation="${REPO_ROOT}/experiments/robotwin/setup_records/compatibility_20260928/fixed150k_transformers4576.json"
exec bash "${SCRIPT_DIR}/evaluate_verified_checkpoint.sh" "${checkpoint}" "${validation}" "$@"
