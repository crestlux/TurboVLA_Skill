#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
cd "${repo_root}"

data_root="${LIBERO_DATA_ROOT:-/data/jack/libero_datasets}"
bert_path="${BERT_MODEL_PATH:-${repo_root}/pretrained/backbones/bert-base-uncased}"
resnet_weights="${RESNET50_WEIGHTS:-${repo_root}/pretrained/backbones/resnet50/resnet50-11ad3fa6.pth}"
init_ckpt="${TURBOVLA_INIT_CKPT:-${repo_root}/pretrained/backbones/groundingdino/groundingdino_swint_ogc.pth}"

experiment_name="${EXPERIMENT_NAME:-libero_all4_50k_1gpu_bs32_resnet50_os16_tf456_warmup10k}"
checkpoint_name="${CHECKPOINT_NAME:-turbovla_all4_50k_bs32_resnet50_os16_tf456}"
interaction_layers="${INTERACTION_LAYERS:-6}"
run_root="${RUN_ROOT:-/data/jack/turbovla_runs}"
log_root="${LOG_ROOT:-/data/jack/log/turbovla}"
requested_run_dir="${RUN_DIR:-}"
requested_run_tag="${RUN_TAG:-}"
resume_mode="${RESUME_MODE:-none}"
gpu_id="${GPU_ID:-0}"

dataset_10="${data_root}/libero_10_no_noops/1.0.0"
dataset_goal="${data_root}/libero_goal_no_noops/1.0.0"
dataset_object="${data_root}/libero_object_no_noops/1.0.0"
dataset_spatial="${data_root}/libero_spatial_no_noops/1.0.0"
dataset_dirs="${dataset_10},${dataset_goal},${dataset_object},${dataset_spatial}"

for required_path in \
    "${dataset_10}" \
    "${dataset_goal}" \
    "${dataset_object}" \
    "${dataset_spatial}" \
    "${bert_path}/config.json" \
    "${resnet_weights}" \
    "${init_ckpt}"; do
    if [[ ! -e "${required_path}" ]]; then
        echo "[ERROR] Required path not found: ${required_path}" >&2
        exit 1
    fi
done

if [[ ! "${interaction_layers}" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] INTERACTION_LAYERS must be a non-negative integer; got ${interaction_layers}" >&2
    exit 1
fi

if [[ "${interaction_layers}" == "0" ]]; then
    interaction_preload_args=(--no_require_feature_enhancer_preload)
else
    interaction_preload_args=(--require_feature_enhancer_preload)
fi

if [[ "${resume_mode}" != "none" ]]; then
    if [[ -z "${requested_run_dir}" ]]; then
        echo "[ERROR] RESUME_MODE=${resume_mode} requires RUN_DIR to identify the existing run." >&2
        exit 1
    fi
    if [[ ! -d "${requested_run_dir}" ]]; then
        echo "[ERROR] Resume directory not found: ${requested_run_dir}" >&2
        exit 1
    fi
    run_dir="${requested_run_dir}"
else
    if [[ -n "${requested_run_dir}" ]]; then
        run_dir="${requested_run_dir}"
        if compgen -G "${run_dir}/*.pth" > /dev/null; then
            echo "[ERROR] Checkpoints already exist in ${run_dir}." >&2
            echo "        Choose a new RUN_DIR or set RESUME_MODE=all." >&2
            exit 1
        fi
        mkdir -p "${run_dir}"
    else
        mkdir -p "${run_root}"
        if [[ -n "${requested_run_tag}" ]]; then
            if [[ ! "${requested_run_tag}" =~ ^run[0-9]+$ ]]; then
                echo "[ERROR] RUN_TAG must look like run01, run02, ...; got ${requested_run_tag}" >&2
                exit 1
            fi
            run_dir="${run_root}/${experiment_name}_${requested_run_tag}"
            if ! mkdir "${run_dir}" 2>/dev/null; then
                echo "[ERROR] Requested run already exists: ${run_dir}" >&2
                exit 1
            fi
        else
            run_dir=""
            for ((run_index = 1; run_index <= 999; run_index++)); do
                printf -v candidate_tag "run%02d" "${run_index}"
                candidate_dir="${run_root}/${experiment_name}_${candidate_tag}"
                if mkdir "${candidate_dir}" 2>/dev/null; then
                    run_dir="${candidate_dir}"
                    break
                fi
            done
            if [[ -z "${run_dir}" ]]; then
                echo "[ERROR] Could not allocate a run directory under ${run_root}." >&2
                exit 1
            fi
        fi
    fi
fi

run_name="$(basename "${run_dir}")"
if [[ "${run_name}" =~ _(run[0-9]+)$ ]]; then
    run_tag="${BASH_REMATCH[1]}"
else
    run_tag="${requested_run_tag:-custom}"
fi
checkpoint_prefix="${CHECKPOINT_PREFIX:-${checkpoint_name}_${run_tag}_step}"
log_dir="${LOG_DIR:-${log_root}/${run_name}}"

mkdir -p "${log_dir}"
log_path="${log_dir}/train_$(date +%Y%m%d_%H%M%S).log"

echo "[INFO] TurboVLA ResNet-50-OS16 | LIBERO All4"
echo "[INFO] run_tag=${run_tag} physical_GPU=${gpu_id} interaction_layers=${interaction_layers}"
echo "[INFO] global_batch=32 steps=50000 warmup=10000 workers=6"
echo "[INFO] run_dir=${run_dir}"
echo "[INFO] log_path=${log_path}"

CUDA_VISIBLE_DEVICES="${gpu_id}" \
USE_TF=0 \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
PYTHONUNBUFFERED=1 \
python experiments/libero/train.py \
  --dataset_dir "${dataset_10}" \
  --dataset_dirs "${dataset_dirs}" \
  --dataset_split train \
  --stats_path experiments/libero/configs/libero_all4_stats.json \
  --stats_key libero_all4_no_noops \
  --normalize_binary_gripper auto \
  --vision_backbone resnet50 \
  --resnet_weights "${resnet_weights}" \
  --resnet_output_stride 16 \
  --vision_precision bf16_autocast \
  --bert_path "${bert_path}" \
  --pretrained_init_ckpt "${init_ckpt}" \
  --checkpoint_dir "${run_dir}" \
  --checkpoint_prefix "${checkpoint_prefix}" \
  --resume_mode "${resume_mode}" \
  --batch_size 32 \
  --grad_accum_steps 1 \
  --max_steps 50000 \
  --lr_schedule_steps 50000 \
  --warmup_steps 10000 \
  --min_lr_ratio 1.0 \
  --lr 5e-5 \
  --head_lr 5e-5 \
  --vision_lr 5e-5 \
  --weight_decay 1e-10 \
  --head_weight_decay 1e-10 \
  --vision_weight_decay 1e-10 \
  --precision fp32 \
  --num_workers 6 \
  --shuffle_buffer 512 \
  --step_mix_buffer_size 64 \
  --expected_image_size 256 \
  --vla_feature_enhancer_layers "${interaction_layers}" \
  "${interaction_preload_args[@]}" \
  --chunk_size 12 \
  --save_steps 10000 \
  --log_freq 20 \
  --seed 42 \
  --no_allow_hf_download \
  "$@" \
  2>&1 | tee "${log_path}"
