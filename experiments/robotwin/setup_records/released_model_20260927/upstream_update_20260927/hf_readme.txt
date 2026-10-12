---
language:
- en
license: other
license_name: dinov3-license
license_link: https://huggingface.co/H-EmbodVis/TurboVLA/blob/main/DINOv3_LICENSE.md
pipeline_tag: robotics
tags:
- robotics
- embodied-ai
- vision-language-action
- robot-manipulation
- pytorch
- dinov3
- libero
- robotwin
arxiv: "2607.27205"
---

<h1 align="center">TurboVLA: Real-Time Vision-Language-Action Model at 32 Hz on an RTX 4090 with &lt;1 GB VRAM</h1>

<div align="center">
  <a href="https://arxiv.org/abs/2607.27205"><img src="https://img.shields.io/badge/arXiv-Paper-b31b1b?logo=arxiv" alt="Paper"></a>
  <a href="https://h-embodvis.github.io/TurboVLA/"><img src="https://img.shields.io/badge/Homepage-Project-orange.svg?logo=googlehome" alt="Project page"></a>
  <a href="https://github.com/H-EmbodVis/TurboVLA"><img src="https://img.shields.io/badge/GitHub-Repository-green?logo=github" alt="GitHub repository"></a>
  <a href="https://huggingface.co/H-EmbodVis/TurboVLA/blob/main/DINOv3_LICENSE.md"><img src="https://img.shields.io/badge/Weights-DINOv3%20License-blue" alt="DINOv3 License"></a>

  <h5><em>Hengyi Xie<sup>1*</sup>, Chenfei Yao<sup>1*</sup>, Xianjin Wu<sup>1</sup>, Xuanyang Xi<sup>2</sup>, Yiping Tang<sup>2</sup>, Di Xu<sup>2</sup>, Yingying Zhu<sup>1</sup>, Dingkang Liang<sup>1&dagger;</sup>, Xiang Bai<sup>1</sup>, Han Ding<sup>1</sup></em></h5>

  <sup>1</sup> Huazhong University of Science and Technology, China<br>
  <sup>2</sup> Huawei Technologies Co. Ltd, China<br>
  <sup>*</sup> Equal contribution, listed alphabetically by surname. <sup>&dagger;</sup> Project lead.
</div>

## 🔍 Overview

**TurboVLA** is a real-time vision-language-action model for robotic manipulation. It replaces the conventional LLM-centric vision-to-language-to-action pathway with a direct vision-plus-language-to-action design: visual observations and language instructions are encoded independently, exchanged through lightweight bidirectional interaction, and decoded into continuous action chunks.

This repository contains the official TurboVLA checkpoints for the **LIBERO** and **RoboTwin 2.0** evaluations.

<details>
  <summary><b>Abstract</b></summary>

Vision-language-action (VLA) models commonly adopt an LLM-centric V → L → A pathway, where visual observations are projected into the representation space of a large language model before being decoded into robot actions. Although effective, this design incurs substantial computation and memory overhead at every policy invocation. In this work, we introduce TurboVLA, a new VLA paradigm that reformulates the conventional V → L → A pathway as a direct V + L → A mapping. Instead of using a large language model as the central interface between perception and action, TurboVLA independently encodes visual observations and language instructions, directly exchanges information between them through lightweight bidirectional vision-language interaction, and predicts continuous action chunks with a compact decoder. This simple design constructs task-conditioned representations directly from visual and linguistic features, significantly reducing the computational and memory costs of VLA inference. On LIBERO, TurboVLA achieves 97.7% average success with only 0.2B parameters, 31.2 ms inference latency, and 0.9 GB inference VRAM on a consumer-grade RTX 4090, matching or outperforming substantially larger VLA policies. These results establish TurboVLA as a simple and effective alternative to the prevailing LLM-centric VLA paradigm, offering a new perspective on how vision, language, and action can be connected for efficient robotic manipulation.
</details>

### ✨ Key Ideas

- Direct vision-plus-language-to-action modeling without an LLM-centric action pathway.
- Independent DINOv3 visual and BERT language encoders with lightweight bidirectional interaction.
- Compact action-chunk decoders for low-latency and memory-efficient inference.
- Official checkpoints for both LIBERO and RoboTwin 2.0.

## 🎯 Task, Inputs, and Outputs

TurboVLA predicts continuous robot action chunks from synchronized multi-view RGB observations, a natural-language instruction, and the robot proprioceptive state.

| Benchmark | Inputs | Output | Action horizon | Visual encoder |
| --- | --- | --- | ---: | --- |
| LIBERO | Two RGB views, language instruction, robot state | 7-D continuous actions | 12 | DINOv3 ViT-B |
| RoboTwin 2.0 | Three 224 x 224 RGB views, language instruction, 14-D robot state | 14-D absolute joint-position actions | 50 | DINOv3 ViT-L |

## 📦 Released Checkpoints

### LIBERO

The LIBERO model was jointly trained on all four suites in a single mixed-suite training run. A single unified checkpoint is released for evaluation across all four suites.

| Evaluation suites | Checkpoint |
| --- | --- |
| LIBERO-Object, LIBERO-Goal, LIBERO-Spatial, and LIBERO-Long | `checkpoints/libero/turbovla_libero.pth` |

The unified LIBERO file is a PyTorch checkpoint containing `model_state_dict` and the corresponding embedded `model_config`. Optimizer and scheduler states are not included. The checkpoint is compatible with the official TurboVLA loader.

### RoboTwin 2.0

The RoboTwin model was jointly trained on all 50 tasks using both Clean and Randomized data in a single task-balanced run. A single 150,000-step EMA checkpoint is released for evaluation.

| Training data | Sampling | Weights | Checkpoint |
| --- | --- | --- | --- |
| RoboTwin 2.0 Clean + Randomized, all 50 tasks | Task-balanced; Randomized:Clean = 10:1 | EMA at 150,000 optimizer updates | `checkpoints/robotwin/steps_150000_ema_pytorch_model.pt` |

The RoboTwin file is a PyTorch model `state_dict`; its model configuration and normalization statistics are provided in `config.yaml` and `dataset_statistics.json`. Optimizer and scheduler states are not included.

## 🛠️ Usage

You can download the complete model repository using either the Hugging Face CLI or the Python SDK.

**Option 1: Hugging Face CLI (Recommended)**

```bash
pip install -U huggingface_hub
hf download H-EmbodVis/TurboVLA --local-dir TurboVLA-model
```

**Option 2: Python SDK**

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="H-EmbodVis/TurboVLA",
    repo_type="model",
    local_dir="TurboVLA-model",
)
```

TurboVLA is a custom PyTorch VLA implementation and is not directly loadable with `transformers.AutoModel.from_pretrained()`. For installation and evaluation instructions, use the [official GitHub repository](https://github.com/H-EmbodVis/TurboVLA).

## 📂 Repository Contents

| Path | Description |
| --- | --- |
| `checkpoints/libero/turbovla_libero.pth` | Unified checkpoint for all four LIBERO evaluation suites |
| `checkpoints/robotwin/steps_150000_ema_pytorch_model.pt` | RoboTwin 2.0 mixed-data, task-balanced EMA checkpoint |
| `libero_all4_stats.json` | Mixed-suite LIBERO normalization statistics |
| `config.yaml` | RoboTwin model and inference configuration |
| `dataset_statistics.json` | RoboTwin mixed-data state and action normalization statistics |
| `config.json` | Hugging Face model metadata and checkpoint index |
| `DINOv3_LICENSE.md` | DINOv3 terms applicable to the released checkpoint parameters |
| `LICENSE` | Apache-2.0 license for the TurboVLA project code |

## ⚖️ License

The TurboVLA source code is released under Apache-2.0. The released checkpoints contain parameters derived from DINOv3 and are distributed subject to the included [DINOv3 License](DINOv3_LICENSE.md). Users must comply with all applicable third-party license terms.

## 📖 Citation

If you find this work useful, please consider citing:

```bibtex
@article{xie2026turbovla,
  title  = {TurboVLA: Real-Time Vision-Language-Action Model at
            32 Hz on an RTX 4090 with <1 GB VRAM},
  author = {Xie, Hengyi and Yao, Chenfei and Wu, Xianjin and
            Xi, Xuanyang and Tang, Yiping and Xu, Di and
            Zhu, Yingying and Liang, Dingkang and Bai, Xiang and
            Ding, Han},
  journal = {arXiv preprint arXiv:2607.27205},
  year   = {2026}
}
```
