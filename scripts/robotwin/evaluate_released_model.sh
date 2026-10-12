#!/usr/bin/env bash
# The former uncorrected 55k run is preserved in its pipeline provenance.
# New calls use the explicitly compatible bundle and validated evaluator.
set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export ROBOTWIN_TEST_NUM="${ROBOTWIN_TEST_NUM:-10}"
export RUN_ID="${RUN_ID:-turbovla_release55000_postnorm_eval${ROBOTWIN_TEST_NUM}_$(date +%Y%m%d_%H%M%S)}"
converted_dir="${CONVERTED_DIR:-${REPO_ROOT}/pretrained/converted/turbovla_robotwin_cb530054_55k_ema_postnorm}"
validation="${RELEASE_VALIDATION:-${REPO_ROOT}/experiments/robotwin/setup_records/compatibility_20260928/fixed55k_transformers4576.json}"
exec bash "${SCRIPT_DIR}/evaluate_verified_checkpoint.sh" \
    "${converted_dir}/steps_55000_ema_model.safetensors" "${validation}" "$@"
