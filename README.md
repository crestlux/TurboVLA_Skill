# TurboVLA Research

[![Upstream](https://img.shields.io/badge/Upstream-TurboVLA-181717?logo=github)](https://github.com/H-EmbodVis/TurboVLA)
[![License](https://img.shields.io/badge/License-Apache%202.0-2563eb)](LICENSE)

> [!IMPORTANT]
> This is an **unofficial, independently maintained research repository** built on
> [TurboVLA](https://github.com/H-EmbodVis/TurboVLA). It is not affiliated with or maintained by the original authors.

This repository is a research workspace for developing and evaluating new vision-language-action (VLA) methods on top
of TurboVLA. The current baseline supports TurboVLA evaluation on LIBERO and adds support for the LIBERO-PRO
robustness benchmark.

No project-specific performance claims are reported yet. Results from the original TurboVLA project should be cited and
interpreted as upstream results, not results produced by this repository.

## Research scope

- New VLA architectures, objectives, and training strategies
- LIBERO baseline reproduction
- LIBERO-PRO robustness and generalization evaluation
- Controlled ablations and reproducible experiment tracking

## Current status

- [x] TurboVLA checkpoint loading with DINOv3 and BERT
- [x] Standard LIBERO rollout evaluation
- [x] LIBERO-PRO suite registration and rollout evaluation
- [x] Separate, per-command selection of LIBERO and LIBERO-PRO
- [ ] Project-specific method implementation
- [ ] Training and ablation results
- [ ] Reproduced benchmark table

## Current benchmark extension

The current project extends the upstream TurboVLA evaluation workflow with:

- LIBERO-PRO variants for the LIBERO-10 checkpoint:
  - `libero_10_object` — object perturbation
  - `libero_10_swap` — position perturbation
  - `libero_10_lan` — semantic/language perturbation
  - `libero_10_task` — task perturbation
  - `libero_10_env` — environment perturbation
- Per-command benchmark source selection through `--libero_root`
- Independent LIBERO configuration selection through `LIBERO_CONFIG_PATH`

Research changes will be documented here as the project develops.

## Installation

Clone this repository and create the LIBERO evaluation environment:

```bash
git clone https://github.com/jaemin720/TurboVLA-Research.git
cd TurboVLA-Research

conda create -n turbovla-libero python=3.10 -y
conda activate turbovla-libero

python -m pip install torch==2.3.1 torchvision==0.18.1 \
  --index-url https://download.pytorch.org/whl/cu121
python -m pip install -e ".[libero]"
```

The simulator combination validated for the current LIBERO evaluation setup is:

```bash
python -m pip install matplotlib==3.5.3 mujoco==2.3.7
```

LIBERO and LIBERO-PRO both expose the Python package name `libero`. Keep their source trees separate and do not install
LIBERO-PRO's pinned requirements into the TurboVLA environment.

## Model assets

Download the released TurboVLA checkpoints:

```bash
python -m pip install -U huggingface_hub
hf download H-EmbodVis/TurboVLA --local-dir pretrained/TurboVLA
```

The LIBERO policy additionally requires:

- DINOv3 ViT-B, stored in a local directory containing `config.json`, `model.safetensors`, and
  `preprocessor_config.json`
- BERT base uncased, either in the Hugging Face cache or a local directory

Large model files, datasets, outputs, and machine-specific configuration files are intentionally excluded from Git.

## Benchmark setup

### Standard LIBERO

Clone LIBERO outside this repository:

```bash
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git /path/to/LIBERO
```

If a persistent editable install is needed, install it without replacing the TurboVLA dependency set:

```bash
python -m pip install -e /path/to/LIBERO \
  --no-deps \
  --no-build-isolation \
  --config-settings editable_mode=compat
```

Use a dedicated LIBERO config directory, for example `/path/to/config/libero/config.yaml`:

```yaml
benchmark_root: /path/to/LIBERO/libero/libero
bddl_files: /path/to/LIBERO/libero/libero/bddl_files
init_states: /path/to/LIBERO/libero/libero/init_files
datasets: /path/to/libero_datasets
assets: /path/to/LIBERO/libero/libero/assets
```

### LIBERO-PRO

Clone LIBERO-PRO separately. Do **not** run its `pip install -r requirements.txt` or `pip install -e .` commands in the
TurboVLA environment.

```bash
git clone https://github.com/Zxy-MLlab/LIBERO-PRO.git /path/to/LIBERO-PRO

hf download zhouxueyang/LIBERO-Pro \
  --repo-type dataset \
  --include "bddl_files/**" "init_files/**" \
  --local-dir /path/to/LIBERO-PRO/libero_data

cp -a /path/to/LIBERO-PRO/libero_data/bddl_files/. \
  /path/to/LIBERO-PRO/libero/libero/bddl_files/
cp -a /path/to/LIBERO-PRO/libero_data/init_files/. \
  /path/to/LIBERO-PRO/libero/libero/init_files/
```

Create a separate config such as `/path/to/config/libero-pro/config.yaml`:

```yaml
benchmark_root: /path/to/LIBERO-PRO/libero/libero
bddl_files: /path/to/LIBERO-PRO/libero/libero/bddl_files
init_states: /path/to/LIBERO-PRO/libero/libero/init_files
datasets: /path/to/libero_datasets
assets: /path/to/LIBERO-PRO/libero/libero/assets
```

Pre-generated BDDL and initialization files are sufficient for the five standard LIBERO-PRO perturbation suites. The
dynamic perturbation generator is only needed for custom perturbation combinations.

## Evaluation

The `long.pth` checkpoint is used for standard LIBERO-10 and its LIBERO-PRO variants. Start with one task and one trial:

```bash
LIBERO_CONFIG_PATH=/path/to/config/libero-pro \
CUDA_VISIBLE_DEVICES=0 \
MUJOCO_EGL_DEVICE_ID=0 \
USE_TF=0 \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
python experiments/libero/evaluate.py \
  --ckpt_path pretrained/TurboVLA/checkpoints/libero/long.pth \
  --libero_root /path/to/LIBERO-PRO \
  --dinov3_path /path/to/dinov3-vitb16 \
  --bert_path /path/to/bert-base-uncased \
  --allow_hf_download false \
  --stats_path experiments/libero/configs/libero_all4_stats.json \
  --stats_key libero_all4_no_noops \
  --task_suite_name libero_10_object \
  --task_ids 0 \
  --num_trials_per_task 1 \
  --chunk_size 12 \
  --num_open_loop_steps 12 \
  --seed 7 \
  --precision bf16 \
  --mujoco_gl egl \
  --pyopengl_platform egl \
  --result_json_path outputs/evaluation/libero_pro_10_object_smoke.json
```

For the standard LIBERO-10 smoke test, change these three values together:

```text
LIBERO_CONFIG_PATH=/path/to/config/libero
--libero_root /path/to/LIBERO
--task_suite_name libero_10
```

For a full evaluation, remove `--task_ids 0` and set `--num_trials_per_task 50`. A failed episode is a valid benchmark
outcome; model loading, environment rollout, and result JSON creation should complete without an exception.

See [experiments/libero/README.md](experiments/libero/README.md) for the upstream training and evaluation entry points.

## Development workflow

Keep `main` runnable and develop unverified methods on a research branch:

```bash
git switch main
git pull --ff-only origin main
git switch -c research/method-name
```

Commit source code and lightweight experiment configuration. Do not commit pretrained models, datasets, videos, logs,
tokens, or machine-specific absolute paths.

The original TurboVLA repository is retained as the `upstream` remote:

```bash
git fetch upstream
git merge upstream/main
```

Review upstream changes before merging because they may overlap with research modifications.

## Upstream projects and attribution

This project builds on the following work:

- [TurboVLA](https://github.com/H-EmbodVis/TurboVLA) — base model, training code, checkpoints, and evaluation runtime
- [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) — manipulation benchmark
- [LIBERO-PRO](https://github.com/Zxy-MLlab/LIBERO-PRO) — robustness and generalization benchmark
- [DINOv3](https://github.com/facebookresearch/dinov3) — visual encoder
- [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) — vision-language interaction components
- [VLA-Adapter](https://github.com/OpenHelix-Team/VLA-Adapter) — LIBERO rollout protocol

Please use the official repositories for authoritative installation instructions, pretrained artifacts, and reported
results.

## Citation

If this code is useful in your research, cite the original TurboVLA and any benchmarks used in your experiments:

```bibtex
@article{xie2026turbovla,
  title   = {TurboVLA: Real-Time Vision-Language-Action Model at
             32 Hz on an RTX 4090 with <1 GB VRAM},
  author  = {Xie, Hengyi and Yao, Chenfei and Wu, Xianjin and
             Xi, Xuanyang and Tang, Yiping and Xu, Di and
             Zhu, Yingying and Liang, Dingkang and Bai, Xiang and Ding, Han},
  journal = {arXiv preprint arXiv:2607.27205},
  year    = {2026}
}

@article{zhou2025liberopro,
  title   = {LIBERO-PRO: Towards Robust and Fair Evaluation of
             Vision-Language-Action Models Beyond Memorization},
  author  = {Zhou, Xueyang and Xu, Yangming and Tie, Guiyao and
             Chen, Yongchao and Zhang, Guowen and Chu, Duanfeng and
             Zhou, Pan and Sun, Lichao},
  journal = {arXiv preprint arXiv:2510.03827},
  year    = {2025}
}
```

## License

This derivative repository retains the upstream [Apache License 2.0](LICENSE). Third-party components remain subject to
their respective licenses and attribution requirements.
