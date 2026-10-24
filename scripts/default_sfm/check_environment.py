#!/usr/bin/env python3
"""Validate dependencies, offline model loading and CUDA. Never train a model."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch  # noqa: E402


def write_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--distributed-check", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "executable": sys.executable,
        "training_started": False,
        "optimizer_steps": 0,
    }
    if args.distributed_check:
        import torch.distributed as dist

        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl")
        world_size = dist.get_world_size()
        value = torch.tensor([dist.get_rank() + 1.0], device=f"cuda:{local_rank}")
        dist.all_reduce(value)
        assert value.item() == world_size * (world_size + 1) / 2
        report.update(status="passed", backend=dist.get_backend(), world_size=world_size, all_reduce=value.item())
        if dist.get_rank() == 0:
            write_report(args.output, report)
        dist.destroy_process_group()
        return

    names = {
        "torch": "torch",
        "torchvision": "torchvision",
        "transformers": "transformers",
        "timm": "timm",
        "numpy": "numpy",
        "h5py": "h5py",
        "opencv-python-headless": "cv2",
        "scipy": "scipy",
        "imageio": "imageio",
        "imageio-ffmpeg": "imageio_ffmpeg",
        "PyYAML": "yaml",
        "wandb": "wandb",
    }
    report["packages"] = {}
    for distribution, module in names.items():
        importlib.import_module(module)
        report["packages"][distribution] = importlib.metadata.version(distribution)
    excluded = ("libero", "robosuite", "mujoco", "sapien", "tensorflow", "tensorflow-datasets", "deepspeed")
    installed = {d.metadata["Name"].lower() for d in importlib.metadata.distributions()}
    assert not set(excluded) & installed, set(excluded) & installed
    report["excluded_packages_absent"] = list(excluded)
    assert torch.cuda.is_available(), "CUDA is unavailable; run this check on the GPU host"
    report["cuda_runtime"] = torch.version.cuda
    report["gpus"] = []
    with torch.inference_mode():
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                x = torch.ones(64, 64, device=f"cuda:{i}", dtype=torch.bfloat16)
                assert (x @ x == 64).all().item()
                # Exercise the cuDNN GRU runtime without learning weights.
                gru = torch.nn.GRU(6, 16, batch_first=True).to(f"cuda:{i}").eval()
                sequence, hidden = gru(torch.zeros(1, 50, 6, device=f"cuda:{i}"))
                assert torch.isfinite(sequence).all().item() and hidden.shape == (1, 1, 16)
                report["gpus"].append(
                    {
                        "index": i,
                        "name": torch.cuda.get_device_name(i),
                        "bf16_matmul": "passed",
                        "gru_forward": "passed",
                    }
                )
                del gru, sequence, hidden, x

    import turbovla
    from turbovla.models import TurboVLA, TurboVLAConfig

    assert Path(turbovla.__file__).resolve().is_relative_to(root), turbovla.__file__
    config = TurboVLAConfig()
    config.text.model_name_or_path = str(root / "pretrained/backbones/bert-base-uncased")
    config.vision.model_name_or_path = str(root / "pretrained/backbones/dinov3-vitb16")
    # Synthetic compatibility probe only; these are not experiment choices.
    config.action.state_dim = 19
    config.action.horizon = 2
    torch.manual_seed(0)
    model = TurboVLA(config).cuda(0).eval()
    with torch.inference_mode():
        output = model(
            ["Insert the audio jack."],
            torch.zeros(1, 2, 3, 256, 256, device="cuda:0"),
            torch.zeros(1, 19, device="cuda:0"),
        )
    assert output.shape == (1, 2, 7) and torch.isfinite(output).all().item()
    report.update(
        status="passed",
        turbovla_module=turbovla.__file__,
        model_mode="eval + inference_mode",
        synthetic_forward_shape=list(output.shape),
        synthetic_dimensions_are_not_training_settings=True,
        pretrained_action_checkpoint_loaded=False,
        wandb="import verified; no run created; future training must use online mode",
    )
    report["backbones"] = {
        "vision": str(Path(config.vision.model_name_or_path).resolve()),
        "text": str(Path(config.text.model_name_or_path).resolve()),
    }
    write_report(args.output, report)


if __name__ == "__main__":
    main()
