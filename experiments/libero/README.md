# LIBERO

This directory contains the released LIBERO recipe and public entry points for the root `turbovla` package. The configuration uses two DINOv3 views, an online BERT encoder, 7-D actions, and a chunk size of 12. The task and episode rollout protocol is the VLA-Adapter-derived implementation under `third_party/vla_adapter/`.

## Layout

```text
../../turbovla/             model, RLDS data, training, and policy code
configs/                    released mixed-suite normalization statistics
train.py                    public mixed-suite training entry point
evaluate.py                 public single-checkpoint evaluation entry point
../../scripts/libero/       data, statistics, and evaluation utilities
../../third_party/          VLA-Adapter rollout protocol and license
```

## Environment

The reference environment used PyTorch 2.3.1, torchvision 0.18.1, transformers 4.56, TensorFlow 2.20, and tensorflow-datasets 4.9.3.

```bash
pip install -e ".[libero]"
```

Install LIBERO separately. DINOv3 and GroundingDINO weights are not included.

## Data

```text
data/libero/
|-- libero_10_no_noops/1.0.0/
|-- libero_goal_no_noops/1.0.0/
|-- libero_object_no_noops/1.0.0/
`-- libero_spatial_no_noops/1.0.0/
```

The repository does not redistribute demonstrations. Use `scripts/libero/regenerate_libero_no_noops.py` with the official LIBERO data, then convert the derived trajectories to TFDS/RLDS. Released normalization statistics are in `experiments/libero/configs/libero_all4_stats.json`.

## Training

The released recipe uses full DINOv3 ViT-B unfreezing, global batch 256 on four GPUs, 80k optimizer steps, 10k warmup steps, learning rate `5e-5`, seed 42, FP32 policy parameters, and BF16 autocast for DINOv3.

```bash
torchrun --nproc_per_node=4 experiments/libero/train.py \
  --dataset_dir data/libero/libero_10_no_noops/1.0.0 \
  --dataset_dirs "data/libero/libero_10_no_noops/1.0.0,data/libero/libero_goal_no_noops/1.0.0,data/libero/libero_object_no_noops/1.0.0,data/libero/libero_spatial_no_noops/1.0.0" \
  --stats_path experiments/libero/configs/libero_all4_stats.json \
  --stats_key libero_all4_no_noops \
  --dinov3_path facebook/dinov3-vitb16-pretrain-lvd1689m \
  --bert_path google-bert/bert-base-uncased \
  --allow_hf_download \
  --pretrained_init_ckpt /path/to/groundingdino_swint_ogc.pth \
  --checkpoint_dir outputs/checkpoints
```

### ResNet-50 vision-backbone experiment

Use `resnet50` with output stride 16 to replace only the vision encoder while retaining the two 256px views, 256 visual tokens per view, six bidirectional interaction layers, action decoder, optimizer settings, and EMA. ImageNet-1K V2 and DINOv3 use the same resize-free ImageNet mean/std preprocessing in this setup. This comparison still changes both architecture and pretraining source: supervised ImageNet-1K V2 replaces self-supervised DINOv3 LVD-1689M.

Download the official torchvision weights once, then add these options to the same All4 training recipe:

```bash
mkdir -p pretrained/backbones/resnet50
wget -c https://download.pytorch.org/models/resnet50-11ad3fa6.pth \
  -O pretrained/backbones/resnet50/resnet50-11ad3fa6.pth

python experiments/libero/train.py \
  --dataset_dir /data/jack/libero_datasets/libero_10_no_noops/1.0.0 \
  --dataset_dirs "/data/jack/libero_datasets/libero_10_no_noops/1.0.0,/data/jack/libero_datasets/libero_goal_no_noops/1.0.0,/data/jack/libero_datasets/libero_object_no_noops/1.0.0,/data/jack/libero_datasets/libero_spatial_no_noops/1.0.0" \
  --stats_path experiments/libero/configs/libero_all4_stats.json \
  --stats_key libero_all4_no_noops \
  --vision_backbone resnet50 \
  --resnet_weights pretrained/backbones/resnet50/resnet50-11ad3fa6.pth \
  --resnet_output_stride 16 \
  --vision_precision bf16_autocast \
  --bert_path pretrained/backbones/bert-base-uncased \
  --pretrained_init_ckpt pretrained/backbones/groundingdino/groundingdino_swint_ogc.pth \
  --checkpoint_dir outputs/checkpoints/resnet50_os16
```

The checkpoint records the active backbone, output stride, and initialization source in `model_config`. Evaluation detects a ResNet checkpoint automatically and does not require `--dinov3_path`.

For the local one-GPU comparison used in this research repository, `scripts/libero/train_resnet50_all4_50k.sh` fixes batch size 32, 50k steps, 10k warmup steps, six workers, seed 42, EMA 0.999, and the same All4 statistics as the DINOv3 baseline. GPU 0 is used by default. Repeated invocations reserve matching `_run01`, `_run02`, ... checkpoint and log directories automatically; paths, GPU, and an explicit `RUN_TAG` can be overridden with environment variables.

The matched ResNet-50 w/o interaction experiment uses the same script internally and changes only the interaction depth to zero (while skipping the now-inapplicable interaction preload):

```bash
bash scripts/libero/train_resnet50_all4_50k_wo_interaction.sh
```

## Evaluation

One invocation evaluates one checkpoint on one suite. Checkpoint selection and multi-GPU scheduling are intentionally outside the evaluator.

```bash
python experiments/libero/evaluate.py \
  --ckpt_path pretrained/TurboVLA/checkpoints/libero/libero_object.pth \
  --dinov3_path /path/to/dinov3-vitb \
  --bert_path /path/to/bert-base-uncased \
  --stats_path experiments/libero/configs/libero_all4_stats.json \
  --stats_key libero_all4_no_noops \
  --task_suite_name libero_object \
  --num_trials_per_task 50 \
  --chunk_size 12 \
  --num_open_loop_steps 12 \
  --seed 7 \
  --precision bf16 \
  --result_json_path outputs/evaluation/object.json
```


Use `--dry_run_model_load true` to validate checkpoint compatibility without a simulator rollout. Complete release checkpoints contain BERT, DINOv3, the vision-language interaction module, and the action head in one strict-loadable state dict. `configs/online_text_layout.json` stores tokenizer padding metadata for the 40 standard instructions; it contains no embeddings or cached model outputs.

The rollout protocol is derived from [VLA-Adapter](https://github.com/OpenHelix-Team/VLA-Adapter/commit/23fa0c9c159e2aa04341cdd3e924f44061311060). See `third_party/licenses/VLA-Adapter.txt` for the retained license and attribution.
