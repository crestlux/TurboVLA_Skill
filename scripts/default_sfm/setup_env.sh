#!/usr/bin/env bash
# Install dependencies only. This script never preprocesses data or starts training.
set -euo pipefail
sfm_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
sfm_env_name="${SFM_ENV_NAME:-turbovla-sfm}"
sfm_conda="${SFM_CONDA_EXE:-${CONDA_EXE:-conda}}"
if ! "$sfm_conda" run -n "$sfm_env_name" python --version >/dev/null 2>&1; then
  "$sfm_conda" create -n "$sfm_env_name" python=3.10 pip rsync=3.4.4 -y
fi
"$sfm_conda" install -n "$sfm_env_name" --freeze-installed rsync=3.4.4 -y
"$sfm_conda" run --no-capture-output -n "$sfm_env_name" python -m pip install \
  -c "$sfm_root/scripts/default_sfm/constraints-cu124.txt" \
  torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
"$sfm_conda" run --no-capture-output -n "$sfm_env_name" python -m pip install \
  -c "$sfm_root/scripts/default_sfm/constraints-cu124.txt" -e "$sfm_root[real_robot]"
"$sfm_conda" run --no-capture-output -n "$sfm_env_name" python -m pip check
