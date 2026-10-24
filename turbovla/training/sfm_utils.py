"""Training primitives for the confirmed SFM contract.

Masked L1, image jitter, optimizer groups, sample-based LR and EMA follow the
archived TurboVLA USB runner. The task-balanced sampler and accumulation are
explicit additions for the nine SFM experiments.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import Sampler


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class TaskBalancedSampler(Sampler):
    """Uniform task then uniform frame, with replacement and exact resumption.

    Each logical batch uses a generator derived solely from seed and committed
    sample cursor. DataLoader prefetch and microbatch size never advance it.
    """
    def __init__(self, shards, batch_size, seed, budget):
        if min(batch_size, budget) < 1:
            raise ValueError("Positive batch size and sample budget are required")
        self.batch_size, self.seed, self.budget, self.seen = batch_size, seed, budget, 0
        indices = defaultdict(list)
        offset = 0
        for shard in shards:
            indices[shard["task"]].extend(range(offset, offset + shard["frames"]))
            offset += shard["frames"]
        self.size = offset
        self.tasks = sorted(indices)
        self.indices = [torch.tensor(indices[name], dtype=torch.int64) for name in self.tasks]
        if not self.tasks or any(len(x) == 0 for x in self.indices):
            raise ValueError("Empty task")
        self.task_sizes = [len(x) for x in self.indices]

    def batch_at(self, cursor):
        count = min(self.batch_size, self.budget - cursor)
        if count < 1:
            raise ValueError("Sample cursor is outside the budget")
        generator = torch.Generator().manual_seed(self.seed + cursor)
        tasks = torch.randint(len(self.tasks), (count,), generator=generator)
        batch = torch.empty(count, dtype=torch.int64)
        for i, indices in enumerate(self.indices):
            mask = tasks == i
            draws = torch.randint(len(indices), (int(mask.sum()),), generator=generator)
            batch[mask] = indices[draws]
        return batch.tolist()

    def __iter__(self):
        for cursor in range(self.seen, self.budget, self.batch_size):
            yield self.batch_at(cursor)

    def __len__(self):
        return math.ceil((self.budget - self.seen) / self.batch_size)

    def consume(self, count):
        if count != min(self.batch_size, self.budget - self.seen) or count < 1:
            raise ValueError("Consumed logical batch differs from sampler")
        self.seen += count

    def state_dict(self):
        return {key: getattr(self, key) for key in
                ("size", "tasks", "task_sizes", "batch_size", "seed", "budget", "seen")}

    def load_state_dict(self, state):
        for key, value in self.state_dict().items():
            if key != "seen" and state[key] != value:
                raise ValueError(f"Sampler changed on resume: {key}")
        if not 0 <= state["seen"] <= self.budget:
            raise ValueError("Invalid committed sample cursor")
        self.seen = state["seen"]


def masked_l1_sum(predicted, target, mask):
    if predicted.shape != target.shape or mask.shape != predicted.shape[:2] or mask.dtype != torch.bool:
        raise ValueError("Expected [B,H,A] actions and boolean [B,H] mask")
    return (predicted.float()[mask] - target.float()[mask]).abs().sum()


def normalize_images(images, settings, augment):
    """Original USB photometric transform; shared between the two current views."""
    if images.dtype != torch.uint8 or images.ndim != 5 or images.shape[2] != 3:
        raise ValueError("Expected uint8 RGB [B,V,3,H,W]")
    values = images.float() / 255
    jitter_settings = settings.get("color_jitter") if augment else None
    if jitter_settings is not None:
        from torchvision.transforms import ColorJitter
        jitter_settings = dict(jitter_settings)
        top, bottom, left, right = jitter_settings.pop("padding_tblr")
        height, width = values.shape[-2:]
        if min(top, bottom, left, right) < 0 or top + bottom >= height or left + right >= width:
            raise ValueError("Invalid letterbox padding")
        jitter = ColorJitter(**jitter_settings)
        content = values[..., top:height-bottom, left:width-right]
        values[..., top:height-bottom, left:width-right] = torch.stack([jitter(sample) for sample in content])
    mean = values.new_tensor(settings["mean"]).view(1, 1, 3, 1, 1)
    std = values.new_tensor(settings["std"]).view(1, 1, 3, 1, 1)
    return (values - mean) / std


def build_optimizer(model, train):
    vision, policy = [], []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            (vision if name.startswith("vision_encoder.backbone.") else policy).append(parameter)
    return torch.optim.AdamW([
        {"params": policy, "lr": train["learning_rate"], "initial_lr": train["learning_rate"], "name": "policy"},
        {"params": vision, "lr": train["vision_learning_rate"], "initial_lr": train["vision_learning_rate"], "name": "vision"},
    ], betas=tuple(train["betas"]), eps=train["epsilon"], weight_decay=train["weight_decay"])


def set_learning_rate(optimizer, train, seen, upcoming):
    position = min(train["max_samples"], seen + upcoming)
    warmup, target = train["warmup_samples"], train["max_samples"]
    if position < warmup:
        multiplier = position / warmup
    else:
        progress = (position - warmup) / (target - warmup)
        multiplier = train["min_lr_ratio"] + (1 - train["min_lr_ratio"]) * .5 * (1 + math.cos(math.pi * progress))
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * multiplier


class EMA:
    def __init__(self, model, decay, reference_batch):
        if not 0 <= decay < 1 or reference_batch < 1:
            raise ValueError("Invalid EMA configuration")
        self.decay, self.reference_batch = decay, reference_batch
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model, count):
        decay = self.decay ** (count / self.reference_batch)
        for name, value in model.state_dict().items():
            if value.is_floating_point():
                self.shadow[name].lerp_(value.detach(), 1 - decay)
            else:
                self.shadow[name].copy_(value)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def capture_rng(device):
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None}


def restore_rng(state, device):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state(state["cuda"], device)


def move_batch(batch, device, start=0, stop=None):
    return {key: (value[start:stop].to(device, non_blocking=True) if isinstance(value, torch.Tensor)
                  else value[start:stop]) for key, value in batch.items()}


def forward_batch(model, batch, pixels):
    extra = {"ft_history": batch["ft_history"]} if "ft_history" in batch else {}
    return model(batch["instruction"], pixels, batch["state"], **extra)


def parameter_hashes(model):
    digests = {key: hashlib.sha256() for key in ("common", "state_projection", "ft_gru", "adaln")}
    for name, value in sorted(model.state_dict().items()):
        key = ("state_projection" if "state_projection" in name else "ft_gru" if "ft_history_encoder" in name
               else "adaln" if "adaln_modulations" in name else "common")
        digests[key].update(name.encode())
        digests[key].update(str((tuple(value.shape), value.dtype)).encode())
        digests[key].update(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return {key: value.hexdigest() for key, value in digests.items()}
