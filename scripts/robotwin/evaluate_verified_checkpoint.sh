#!/usr/bin/env bash
# Evaluate an immutable, numerically validated release bundle in the pinned simulator.
set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/activate_env.sh"
cd "${TURBOVLA_ROOT}"
if (( $# < 2 )); then
    echo "Usage: bash scripts/robotwin/evaluate_verified_checkpoint.sh <checkpoint> <validation.json> [task ...]" >&2
    exit 2
fi
export POLICY_CHECKPOINT="$(realpath "$1")" VALIDATION_RECORD="$(realpath "$2")"
shift 2
tasks=("$@")
if (( ${#tasks[@]} == 0 )); then tasks=(all); fi
export ROBOTWIN_EVAL_MODE="${ROBOTWIN_EVAL_MODE:-demo_clean}"
case "${ROBOTWIN_EVAL_MODE}" in
    demo_clean|demo_randomized) ;;
    *) echo "Invalid ROBOTWIN_EVAL_MODE: ${ROBOTWIN_EVAL_MODE}" >&2; exit 2 ;;
esac
export ROBOTWIN_TEST_NUM="${ROBOTWIN_TEST_NUM:-10}"
# Match the official 100-episode instruction pool even in shorter evaluations.
export ROBOTWIN_INSTRUCTION_SAMPLE_NUM="${ROBOTWIN_INSTRUCTION_SAMPLE_NUM:-100}"
[[ "${ROBOTWIN_TEST_NUM}" =~ ^[1-9][0-9]*$ ]] || { echo 'Invalid trial count' >&2; exit 2; }
[[ "${ROBOTWIN_INSTRUCTION_SAMPLE_NUM}" =~ ^[1-9][0-9]*$ ]] || { echo 'Invalid instruction sample count' >&2; exit 2; }
export RUN_ID="${RUN_ID:-turbovla_verified_${ROBOTWIN_EVAL_MODE}_$(date +%Y%m%d_%H%M%S)}"
[[ "${RUN_ID}" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo 'Invalid run ID' >&2; exit 2; }
export PIPELINE_DIR="${ROBOTWIN_STORE}/pipelines/${RUN_ID}"
export ROBOTWIN_LOG_ROOT="${ROBOTWIN_STORE}/evaluation/${RUN_ID}_seed0"
export ROBOTWIN_PATH=/home/jack/practice/RoboTwin-TurboVLA-bf44be5
export ROBOTWIN_PYTHON=/home/jack/miniforge3/envs/robotwin-turbovla/bin/python
export SIMULATOR_VALIDATION="${TURBOVLA_ROOT}/experiments/robotwin/setup_records/compatibility_20260928/simulator_environment.json"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export ROBOTWIN_SEED=0 ROBOTWIN_USE_BF16=1
export ROBOTWIN_JOBS_PER_GPU="${ROBOTWIN_JOBS_PER_GPU:-2}"
[[ "${ROBOTWIN_JOBS_PER_GPU}" =~ ^[1-9][0-9]*$ ]] || { echo 'Invalid jobs per GPU' >&2; exit 2; }
export ROBOTWIN_BASE_PORT="${ROBOTWIN_BASE_PORT:-7500}" POLICY_NAME="${RUN_ID}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 MAX_JOBS=4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="${TURBOVLA_ROOT}:${TURBOVLA_ROOT}/third_party/starvla_runtime"
export TRITON_CACHE_DIR="${CONDA_PREFIX}/var/triton" TRITON_HOME="${CONDA_PREFIX}/var/triton-home"
if [[ -e "${PIPELINE_DIR}" || -e "${ROBOTWIN_LOG_ROOT}" ]]; then
    echo "Refusing to overwrite ${RUN_ID}" >&2; exit 1
fi
IFS=',' read -r -a selected_gpus <<< "${CUDA_VISIBLE_DEVICES}"
for selected_gpu in "${selected_gpus[@]}"; do
    active_gpu_pids="$(nvidia-smi --id="${selected_gpu}" --query-compute-apps=pid --format=csv,noheader)"
    if [[ -n "${active_gpu_pids}" ]]; then
        echo "Selected GPU ${selected_gpu} is occupied; evaluation not started." >&2
        exit 1
    fi
done
mkdir -p "${PIPELINE_DIR}/provenance"
exec > >(tee -a "${PIPELINE_DIR}/pipeline.log") 2>&1
echo "$$" > "${PIPELINE_DIR}/pid"
"${STARVLA_PYTHON}" - "${tasks[@]}" <<'PY'
import datetime, hashlib, json, os, re, shutil, subprocess, sys
from pathlib import Path
import torch, transformers
repo=Path(os.environ['TURBOVLA_ROOT']); root=Path(os.environ['PIPELINE_DIR'])
checkpoint=Path(os.environ['POLICY_CHECKPOINT']); validation_path=Path(os.environ['VALIDATION_RECORD'])
validation=json.loads(validation_path.read_text())
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''): h.update(block)
    return h.hexdigest()
assert validation['status']=='passed'
assert Path(validation['checkpoint']).resolve()==checkpoint
assert validation['feature_source']=='last_hidden_state'
assert validation['transformers_version']==transformers.__version__
assert validation['torch_version']==torch.__version__
assert validation['comparison']['bitwise_equal']
for file,key in [(checkpoint,'checkpoint_sha256'),(checkpoint.parent/'config.yaml','config_sha256'),(checkpoint.parent/'dataset_statistics.json','statistics_sha256')]:
    assert sha(file)==validation[key],str(file)
simulator=json.loads(Path(os.environ['SIMULATOR_VALIDATION']).read_text())
assert simulator['status']=='ready' and simulator['robotwin_path']==os.environ['ROBOTWIN_PATH']
assert simulator['robotwin_commit']=='bf44be51cf5717a5595ce59447f2cf5263d2aa95'
assert subprocess.check_output(['git','-C',os.environ['ROBOTWIN_PATH'],'rev-parse','HEAD'],text=True).strip()==simulator['robotwin_commit']
tasks=sys.argv[1:]
assert tasks==['all'] or ('all' not in tasks and len(tasks)==len(set(tasks)))
meta=dict(run_id=os.environ['RUN_ID'],pipeline_script='evaluate_verified_checkpoint.sh',training_performed=False,
          tmux_session=os.environ.get('EVALUATION_TMUX_SESSION'),
          weights='EMA',checkpoint=str(checkpoint),checkpoint_optimizer_steps=int(re.search(r'steps_(\d+)',checkpoint.name)[1]),
          evaluation_dir=os.environ['ROBOTWIN_LOG_ROOT'],eval_tasks=50 if tasks==['all'] else len(tasks),
          eval_mode=os.environ['ROBOTWIN_EVAL_MODE'],eval_task_names=tasks,eval_trials_per_task=int(os.environ['ROBOTWIN_TEST_NUM']),eval_seed=0,
          instruction_sample_num=int(os.environ['ROBOTWIN_INSTRUCTION_SAMPLE_NUM']),feature_fix_applied=True,
          cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],jobs_per_gpu=int(os.environ['ROBOTWIN_JOBS_PER_GPU']),
          evaluation_slots=len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))*int(os.environ['ROBOTWIN_JOBS_PER_GPU']),
          python_random_seed=os.environ.get('ROBOTWIN_PYTHON_RANDOM_SEED'),
          feature_source=validation['feature_source'],compatibility_profile=validation['compatibility_profile'],
          simulator_commit=simulator['robotwin_commit'],started_at=datetime.datetime.now().astimezone().isoformat())
(root/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
destination=root/'provenance'
for p in [validation_path,Path(os.environ['SIMULATOR_VALIDATION']),checkpoint.parent/'config.yaml',checkpoint.parent/'config.release.yaml',checkpoint.parent/'dataset_statistics.json',checkpoint.parent/'conversion.json']:
    shutil.copy2(p,destination/p.name)
relative=[p.relative_to(repo) for folder in ['turbovla','experiments/robotwin/evaluation','third_party/starvla_runtime/starVLA/model/framework','third_party/starvla_runtime/deployment/model_server'] for p in (repo/folder).rglob('*.py')]
relative += [Path('scripts/robotwin')/name for name in ['evaluate_verified_checkpoint.sh','evaluate_released_model.sh','evaluate_latest_released_model.sh','start_eval.sh','eval_task.sh','run_policy_server.sh','summarize_evaluation.py','validate_feature_compatibility.py']]
relative += [Path('experiments/robotwin/evaluation/deploy_policy.yml')]
for p in relative:
    target=destination/'source'/p; target.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(repo/p,target)
shutil.copy2(Path(os.environ['ROBOTWIN_PATH'])/'script/eval_policy.py',destination/'simulator_eval_policy.py')
(destination/'SHA256.json').write_text(json.dumps({str(p.relative_to(destination)):sha(p) for p in destination.rglob('*') if p.is_file()},indent=2)+'\n')
print(json.dumps(meta,indent=2),flush=True)
PY
stage=prepared
write_status() {
    "${STARVLA_PYTHON}" - "$1" "${2:-0}" <<'PY'
import datetime,json,os,sys
from pathlib import Path
root=Path(os.environ['PIPELINE_DIR']);d=json.loads((root/'metadata.json').read_text())
d.update(stage=sys.argv[1],exit_code=int(sys.argv[2]),updated_at=datetime.datetime.now().astimezone().isoformat())
tmp=root/'status.json.tmp';tmp.write_text(json.dumps(d,indent=2)+'\n');tmp.replace(root/'status.json')
with (root/'events.jsonl').open('a') as f:f.write(json.dumps(d)+'\n')
active=dict(d,status=d['stage'],pipeline_dir=str(root),last_observed_at=d['updated_at'])
(Path(os.environ['TURBOVLA_ROOT'])/'experiments/robotwin/setup_records/active-evaluation.json').write_text(json.dumps(active,indent=2)+'\n')
PY
}
on_exit() {
    local rc=$?
    trap - EXIT
    if (( rc != 0 )); then write_status "${stage}_failed" "${rc}" || true; fi
    printf '%s\n' "${rc}" > "${PIPELINE_DIR}/exit_code.txt"
    exit "${rc}"
}
trap on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
write_status "${stage}"
"${STARVLA_PYTHON}" -m pip freeze > "${PIPELINE_DIR}/provenance/policy-packages.txt"
"${ROBOTWIN_PYTHON}" -m pip freeze > "${PIPELINE_DIR}/provenance/simulator-packages.txt"
git diff > "${PIPELINE_DIR}/provenance/turbovla.patch"
git -C "${ROBOTWIN_PATH}" diff > "${PIPELINE_DIR}/provenance/robotwin.patch"
stage=evaluating
write_status "${stage}"
bash "${SCRIPT_DIR}/start_eval.sh" --mode "${ROBOTWIN_EVAL_MODE}" --name "${RUN_ID}" --ckpt "${POLICY_CHECKPOINT}" \
    --jobs-per-gpu "${ROBOTWIN_JOBS_PER_GPU}" --base-port "${ROBOTWIN_BASE_PORT}" "${tasks[@]}" 2>&1 | tee "${PIPELINE_DIR}/evaluation.log"
stage=summarizing
write_status "${stage}"
summary_args=(--trials "${ROBOTWIN_TEST_NUM}" --mode "${ROBOTWIN_EVAL_MODE}")
if [[ "${tasks[*]}" != all ]]; then summary_args+=(--tasks "${tasks[@]}"); fi
"${STARVLA_PYTHON}" "${SCRIPT_DIR}/summarize_evaluation.py" "${ROBOTWIN_LOG_ROOT}" "${summary_args[@]}"
stage=completed
write_status "${stage}"
echo "Verified-checkpoint evaluation completed at $(date -Is)."
