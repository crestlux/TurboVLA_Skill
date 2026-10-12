#!/usr/bin/env bash
# Run inside tmux. Successful training gates checkpoint validation and rollout evaluation.
set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/activate_env.sh"
cd "${TURBOVLA_ROOT}"

export MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-55000}"
export NUM_PROCESSES="${NUM_PROCESSES:-2}"
export PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-6}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-16}"
export WARMUP_STEPS="${WARMUP_STEPS:-1000}" LEARNING_RATE="${LEARNING_RATE:-5e-5}"
export EMA_DECAY="${EMA_DECAY:-0.999}" SAVE_INTERVAL="${SAVE_INTERVAL:-5000}"
export LOGGING_FREQUENCY="${LOGGING_FREQUENCY:-50}"
export ROBOTWIN_TEST_NUM="${ROBOTWIN_TEST_NUM:-100}"
export ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-experiments/robotwin/configs/deepspeed_zero2.yaml}"
export GLOBAL_BATCH_SIZE=$((NUM_PROCESSES * PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))
if (( MAX_TRAIN_STEPS < 1 || SAVE_INTERVAL < 1 || MAX_TRAIN_STEPS % SAVE_INTERVAL != 0 || ROBOTWIN_TEST_NUM < 1 )); then
    echo "Training steps must be positive and divisible by save interval; evaluation trials must be positive." >&2
    exit 1
fi
export RUN_ID="${RUN_ID:-turbovla_clean50_${MAX_TRAIN_STEPS}steps_2x3090_$(date +%Y%m%d_%H%M%S)}"
export PIPELINE_DIR="${ROBOTWIN_STORE}/pipelines/${RUN_ID}"
export TRAIN_OUTPUT_DIR="${RUN_ROOT_DIR}/${RUN_ID}"
export ROBOTWIN_LOG_ROOT="${ROBOTWIN_STORE}/evaluation/${RUN_ID}_seed0"
export ROBOTWIN_PATH=/home/jack/practice/RoboTwin-TurboVLA
export ROBOTWIN_PYTHON=/home/jack/miniforge3/envs/robotwin-turbovla/bin/python
export EVALUATION_READY_FILE="${TURBOVLA_ROOT}/experiments/robotwin/setup_records/evaluation_ready.json"

if [[ -e "${PIPELINE_DIR}" || -e "${TRAIN_OUTPUT_DIR}" || -e "${ROBOTWIN_LOG_ROOT}" ]]; then
    echo "Refusing to overwrite an existing run: ${RUN_ID}" >&2
    exit 1
fi
mkdir -p "${PIPELINE_DIR}/provenance"
exec > >(tee -a "${PIPELINE_DIR}/pipeline.log") 2>&1
echo "Pipeline started at $(date -Is), PID=$$, run=${RUN_ID}"
echo "$$" > "${PIPELINE_DIR}/pid"

stage=starting
write_status() {
    "${STARVLA_PYTHON}" - "$1" "${2:-0}" <<'PY'
import datetime, json, os
from pathlib import Path
import sys
root = Path(os.environ['PIPELINE_DIR'])
status = dict(stage=sys.argv[1], exit_code=int(sys.argv[2]),
              updated_at=datetime.datetime.now().astimezone().isoformat(),
              run_id=os.environ['RUN_ID'], training_dir=os.environ['TRAIN_OUTPUT_DIR'],
              evaluation_dir=os.environ['ROBOTWIN_LOG_ROOT'],
              target_optimizer_steps=int(os.environ['MAX_TRAIN_STEPS']),
              global_batch_size=int(os.environ['GLOBAL_BATCH_SIZE']),
              per_device_batch_size=int(os.environ['PER_DEVICE_BATCH_SIZE']),
              gradient_accumulation_steps=int(os.environ['GRADIENT_ACCUMULATION_STEPS']),
              accelerate_config=os.environ['ACCELERATE_CONFIG'],
              eval_tasks=50, eval_trials_per_task=int(os.environ['ROBOTWIN_TEST_NUM']),
              train_seed=42, eval_seed=0)
temp = root / 'status.json.tmp'
temp.write_text(json.dumps(status, indent=2) + '\n')
temp.replace(root / 'status.json')
with (root / 'events.jsonl').open('a') as handle:
    handle.write(json.dumps(status) + '\n')
PY
}
on_exit() {
    local rc=$?
    trap - EXIT
    if (( rc != 0 )); then
        write_status "${stage}_failed" "${rc}" || true
        echo "Pipeline failed during ${stage} at $(date -Is), exit=${rc}" >&2
    fi
    printf '%s\n' "${rc}" > "${PIPELINE_DIR}/exit_code.txt"
    exit "${rc}"
}
trap on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 MAX_JOBS=4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_ENDPOINT=https://huggingface.co
export PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR="${CONDA_PREFIX}/var/triton"
export TRITON_HOME="${CONDA_PREFIX}/var/triton-home"
mkdir -p "${TRITON_CACHE_DIR}" "${TRITON_HOME}/.triton/autotune"

git rev-parse HEAD > "${PIPELINE_DIR}/provenance/turbovla.commit"
git diff > "${PIPELINE_DIR}/provenance/turbovla.patch"
"${STARVLA_PYTHON}" -m pip freeze > "${PIPELINE_DIR}/provenance/training-packages.txt"
cp "$0" "${SCRIPT_DIR}/train.sh" "${SCRIPT_DIR}/activate_env.sh" \
    "${SCRIPT_DIR}/summarize_evaluation.py" "${SCRIPT_DIR}/evaluate.sh" \
    "${SCRIPT_DIR}/start_eval.sh" "${SCRIPT_DIR}/eval_task.sh" \
    "${SCRIPT_DIR}/run_policy_server.sh" "${SCRIPT_DIR}/train_one_day_pilot.sh" \
    "${SCRIPT_DIR}/pipeline_status.py" "${PIPELINE_DIR}/provenance/"
cp experiments/robotwin/configs/clean50.yaml "${ACCELERATE_CONFIG}" \
    "${PIPELINE_DIR}/provenance/"
"${STARVLA_PYTHON}" - <<'PY'
import os, shutil, yaml
from pathlib import Path
cfg = yaml.safe_load(Path(os.environ['ACCELERATE_CONFIG']).read_text())
shutil.copy2(cfg['deepspeed_config']['deepspeed_config_file'], Path(os.environ['PIPELINE_DIR']) / 'provenance')
PY
cp third_party/starvla_runtime/starVLA/training/{train_starvla.py,train_robotwin_clean_act_pi05_recipe.py} \
    "${PIPELINE_DIR}/provenance/"
cp "${ROBOTWIN_DATA_ROOT}/READY.json" "${PIPELINE_DIR}/provenance/dataset-ready.json"
nvidia-smi > "${PIPELINE_DIR}/provenance/gpu-before.txt"

stage=training
write_status "${stage}"
bash "${SCRIPT_DIR}/train.sh" \
    --trainer.save_format safetensors \
    --trainer.freeze_modules model.text_encoder.bert \
    --datasets.vla_data.num_workers 4 \
    --datasets.vla_data.prefetch_factor 2 \
    2>&1 | tee "${PIPELINE_DIR}/train.log"
echo 0 > "${PIPELINE_DIR}/training_exit_code.txt"

stage=checkpoint_validation
write_status "${stage}"
export TRAIN_CKPT="${TRAIN_OUTPUT_DIR}/checkpoints/steps_${MAX_TRAIN_STEPS}_ema_model.safetensors"
"${STARVLA_PYTHON}" - <<'PY'
import hashlib, json, os
from pathlib import Path
import torch
from safetensors import safe_open
root = Path(os.environ['TRAIN_OUTPUT_DIR'])
ckpt = Path(os.environ['TRAIN_CKPT'])
steps = [json.loads(line)['steps'] for line in (root / 'summary.jsonl').read_text().splitlines() if line.strip()]
assert steps[-1] == int(os.environ['MAX_TRAIN_STEPS']), steps
torch.set_num_threads(4)
with safe_open(ckpt, framework='pt', device='cpu') as handle:
    keys = list(handle.keys())
    assert len(keys) == 878, len(keys)
    for key in keys:
        assert torch.isfinite(handle.get_tensor(key)).all(), key
digest = hashlib.sha256()
with ckpt.open('rb') as handle:
    for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
        digest.update(block)
result = dict(path=str(ckpt), bytes=ckpt.stat().st_size, tensors=len(keys),
              finite=True, sha256=digest.hexdigest(), optimizer_steps=steps[-1])
(Path(os.environ['PIPELINE_DIR']) / 'checkpoint_validation.json').write_text(json.dumps(result, indent=2) + '\n')
print(result)
PY

stage=evaluation_preflight
write_status "${stage}"
"${STARVLA_PYTHON}" - <<'PY'
import json, os, subprocess
from pathlib import Path
ready = json.loads(Path(os.environ['EVALUATION_READY_FILE']).read_text())
assert ready['status'] == 'ready', ready
assert ready['robotwin_path'] == os.environ['ROBOTWIN_PATH']
assert ready['robotwin_python'] == os.environ['ROBOTWIN_PYTHON']
assert subprocess.check_output(['git', '-C', os.environ['ROBOTWIN_PATH'], 'rev-parse', 'HEAD'], text=True).strip() == ready['robotwin_commit']
print('Evaluation environment validated:', ready)
PY
cp "${EVALUATION_READY_FILE}" "${PIPELINE_DIR}/provenance/"
git -C "${ROBOTWIN_PATH}" diff > "${PIPELINE_DIR}/provenance/robotwin.patch"
"${ROBOTWIN_PYTHON}" -m pip freeze > "${PIPELINE_DIR}/provenance/simulator-packages.txt"

stage=evaluating
write_status "${stage}"
export ROBOTWIN_SEED=0 ROBOTWIN_JOBS_PER_GPU=1
export POLICY_NAME="${RUN_ID}" ROBOTWIN_USE_BF16=1
bash "${SCRIPT_DIR}/evaluate.sh" "${TRAIN_CKPT}" 2>&1 | tee "${PIPELINE_DIR}/evaluation.log"
"${STARVLA_PYTHON}" "${SCRIPT_DIR}/summarize_evaluation.py" "${ROBOTWIN_LOG_ROOT}" --trials "${ROBOTWIN_TEST_NUM}"
write_status completed
echo "Training and all $((50 * ROBOTWIN_TEST_NUM)) evaluation trials completed at $(date -Is)."
