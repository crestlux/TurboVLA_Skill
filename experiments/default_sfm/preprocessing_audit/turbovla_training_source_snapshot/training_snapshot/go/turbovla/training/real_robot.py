"""Single-GPU real-robot TurboVLA training with resumable sample accounting.

Checkpoint/sampler design follows the local RepresentationFM training runner.
Supports image/state and instruction-conditioned models with the ACT/L1 head.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler

from turbovla.data.real_robot import RealRobotDataset, fit_ft_history_statistics, normalize_images
from turbovla.data.ft_history import resampling_contract
from turbovla.models import TurboVLAConfig, build_turbovla
from turbovla.models.configuration import FTHistoryConfig
from turbovla.training.progress import format_progress, progress_fields

SCHEMA = "turbovla_real_robot_v1"


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


class SampleBudgetSampler(Sampler):
    """Prefetch never advances the saved cursor; only completed updates do."""
    def __init__(self, size, batch_size, seed, budget):
        if min(size, batch_size, budget) < 1:
            raise ValueError("Positive size, batch size and sample budget are required")
        self.size, self.batch_size, self.seed, self.budget = size, batch_size, seed, budget
        self.seen = 0

    def __len__(self):
        return math.ceil(min(self.size - self.seen % self.size, self.budget - self.seen) / self.batch_size)

    def __iter__(self):
        epoch, cursor = divmod(self.seen, self.size)
        indices = torch.randperm(self.size, generator=torch.Generator().manual_seed(self.seed + epoch)).tolist()
        end = cursor + min(self.size - cursor, self.budget - self.seen)
        for start in range(cursor, end, self.batch_size):
            yield indices[start:min(end, start + self.batch_size)]

    def consume(self, count):
        expected = min(self.batch_size, self.size - self.seen % self.size, self.budget - self.seen)
        if count != expected or count < 1:
            raise ValueError("Consumed batch does not match the sampler")
        self.seen += count

    def state_dict(self):
        return {key: getattr(self, key) for key in ("size", "batch_size", "seed", "budget", "seen")}

    def load_state_dict(self, state):
        for key in ("size", "batch_size", "seed", "budget"):
            if state[key] != getattr(self, key):
                raise ValueError(f"Sampler {key} changed on resume")
        if not 0 <= state["seen"] <= self.budget:
            raise ValueError("Invalid sampler cursor")
        self.seen = state["seen"]

    @property
    def target_steps(self):
        epochs, tail = divmod(self.budget, self.size)
        return epochs * math.ceil(self.size / self.batch_size) + math.ceil(tail / self.batch_size)


def masked_l1_loss(predicted, target, mask):
    if predicted.shape != target.shape or mask.shape != predicted.shape[:2] or mask.dtype != torch.bool:
        raise ValueError("Expected matching [B,H,A] actions and boolean [B,H] mask")
    if not mask.any():
        raise ValueError("No valid action targets")
    # Select first: arbitrary padded values, including NaN, have no effect.
    return (predicted.float()[mask] - target.float()[mask]).abs().mean()


class EMA:
    def __init__(self, model, decay, reference_batch):
        if not 0 <= decay < 1 or reference_batch < 1:
            raise ValueError("Invalid EMA configuration")
        self.decay, self.reference_batch = decay, reference_batch
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model, samples):
        decay = self.decay ** (samples / self.reference_batch)
        for name, value in model.state_dict().items():
            if value.is_floating_point():
                self.shadow[name].lerp_(value.detach(), 1 - decay)
            else:
                self.shadow[name].copy_(value)


def configure(config):
    config = json.loads(json.dumps(config))
    ft = FTHistoryConfig(**config["model"].get("ft_history", {}))
    if ft.enabled:
        resampling = resampling_contract(config["data"].get("ft_history_resampling"))
        if resampling is not None:
            config["data"]["ft_history_resampling"] = resampling
            factor = round(resampling["source_hz"] / resampling["target_hz"])
            if resampling["source_length"] // factor != ft.history_length:
                raise ValueError("Data and model FT-history resampled lengths differ")
        if ft.mean is None:
            statistics = fit_ft_history_statistics(config["data"], ft.history_length, ft.input_dim)
            config["model"]["ft_history"] = {**config["model"]["ft_history"], **statistics}
        for key, expected in {"use_ft_history": True, "ft_history_length": ft.history_length,
                              "ft_history_dim": ft.input_dim}.items():
            if key in config["data"] and config["data"][key] != expected:
                raise ValueError(f"Data and model FT-history settings differ: {key}")
            config["data"][key] = expected
    elif config["data"].get("use_ft_history", False):
        raise ValueError("Data and model FT-history settings differ")
    config["model"] = TurboVLAConfig.from_mapping(config["model"]).to_dict()
    if config["model"]["use_language"]:
        config["data"]["use_language"] = True
        if not config["model"]["text"]["local_files_only"]:
            raise ValueError("Use local pretrained language weights")
    elif config["data"].get("use_language", False):
        raise ValueError("Data and model language settings differ")
    action, vision, data = config["model"]["action"], config["model"]["vision"], config["data"]
    if (action["state_dim"], action["action_dim"], action["horizon"]) != (19, 7, data["horizon"]):
        raise ValueError("Expected original 19D state, 7D action and stored horizon")
    if vision["num_views"] != len(data["camera_keys"]) or vision["image_size"] != data["image_size"]:
        raise ValueError("Model camera count/resolution differs from data")
    if vision["frozen"] or not vision["local_files_only"]:
        raise ValueError("Use local pretrained vision weights and fine-tune the backbone")
    train = config["train"]
    for key in ("batch_size", "max_samples", "log_interval", "save_interval", "keep_checkpoints"):
        if int(train[key]) < 1:
            raise ValueError(f"{key} must be positive")
    if train["num_workers"] < 0 or not 0 <= train["warmup_samples"] < train["max_samples"]:
        raise ValueError("Invalid workers or warmup")
    if config["images"].get("color_jitter") is not None:
        normalize_images(torch.zeros(1, len(data["camera_keys"]), 3, data["image_size"], data["image_size"],
                                     dtype=torch.uint8), config["images"]["mean"], config["images"]["std"],
                         color_jitter=config["images"]["color_jitter"])
    return config


def batch_to_device(batch, device):
    return {key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()}


def forward_batch(model, batch, pixels):
    """One input contract for training, replay and checkpoint verification."""
    extra = {"ft_history": batch["ft_history"]} if "ft_history" in batch else {}
    return model(batch.get("instruction"), pixels, batch["state"], **extra)


def build_optimizer(model, train):
    vision, policy = [], []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            (vision if name.startswith("vision_encoder.backbone.") else policy).append(parameter)
    return torch.optim.AdamW([
        {"params": policy, "lr": train["learning_rate"], "initial_lr": train["learning_rate"], "name": "policy"},
        {"params": vision, "lr": train["vision_learning_rate"], "initial_lr": train["vision_learning_rate"], "name": "vision"},
    ], betas=tuple(train["betas"]), weight_decay=train["weight_decay"])


def set_learning_rate(optimizer, train, completed_samples, upcoming_samples):
    position = min(train["max_samples"], completed_samples + upcoming_samples)
    warmup, target = train["warmup_samples"], train["max_samples"]
    if position < warmup:
        multiplier = position / warmup
    else:
        progress = (position - warmup) / (target - warmup)
        multiplier = train["min_lr_ratio"] + (1 - train["min_lr_ratio"]) * .5 * (1 + math.cos(math.pi * progress))
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * multiplier


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_rng():
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def read_checkpoint(path):
    # Only checkpoints produced by this trusted local training runner.
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != SCHEMA:
        raise ValueError("Not a TurboVLA real-robot checkpoint")
    return payload


def load_weights(model, payload, weights="raw"):
    if payload["schema"] != SCHEMA or TurboVLAConfig.from_mapping(payload["config"]["model"]).to_dict() != model.config.to_dict():
        raise ValueError("Checkpoint model configuration differs")
    if weights not in ("raw", "ema"):
        raise ValueError("Expected raw or ema weights")
    state = payload[weights]
    expected = model.state_dict()
    if state.keys() != expected.keys():
        raise ValueError("Checkpoint parameter names differ")
    for name, value in state.items():
        if value.shape != expected[name].shape or not torch.isfinite(value).all():
            raise ValueError(f"Invalid checkpoint tensor {name}")
    if model.config.ft_history.enabled:
        # These are fixed preprocessing coefficients, not trainable/EMA weights.
        for name in ("mean", "scale"):
            key = f"action_head.ft_history_encoder.{name}"
            if not torch.equal(state[key].cpu(), expected[key].cpu()):
                raise ValueError("Checkpoint FT-history normalization differs from model configuration")
    model.load_state_dict(state, strict=True)


def train_model(config, output, device="cuda", resume=None, stop_after_step=None):
    config = configure(config)
    train, data, model_config = config["train"], config["data"], config["model"]
    device, output = torch.device(device), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if not resume and (output / "config.json").exists():
        raise FileExistsError("Use a new output directory or explicit --resume")
    seed_everything(train["seed"])
    torch.set_num_threads(4)
    model = build_turbovla(model_config).to(device)
    if not model_config["use_language"] and (model.text_encoder is not None or model.vision_language_interaction is not None):
        raise AssertionError("Language modules must not exist")
    dataset = RealRobotDataset(data)
    sampler = SampleBudgetSampler(len(dataset), train["batch_size"], train["seed"], train["max_samples"])
    loader_generator = torch.Generator().manual_seed(train["seed"])
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=train["num_workers"],
                        pin_memory=device.type == "cuda", persistent_workers=train["num_workers"] > 0,
                        generator=loader_generator)
    optimizer = build_optimizer(model, train)
    ema = EMA(model, train["ema_decay"], train["ema_reference_batch"])
    amp = bool(train["amp"]) and device.type == "cuda"
    if amp and not torch.cuda.is_bf16_supported():
        raise ValueError("BF16 requested on an unsupported device")
    step, prior_seconds = 0, 0.0
    metrics_path = output / "metrics.jsonl"
    if resume:
        payload = read_checkpoint(resume)
        if payload["config"] != config or payload["data_signature"] != dataset.signature():
            raise ValueError("Training configuration or dataset changed on resume")
        load_weights(model, payload, "ema")
        ema.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}
        load_weights(model, payload, "raw")
        optimizer.load_state_dict(payload["optimizer"])
        sampler.load_state_dict(payload["sampler"])
        loader_generator.set_state(payload["loader_generator"])
        restore_rng(payload["rng"])
        step, prior_seconds = payload["step"], payload["elapsed_seconds"]
        if metrics_path.exists():
            lines = metrics_path.read_text().splitlines()
            retained = [line for line in lines if json.loads(line)["step"] <= step]
            if retained != lines:
                (output / f"metrics_before_resume_{time.time_ns()}.jsonl").write_text("\n".join(lines) + "\n")
                metrics_path.write_text("\n".join(retained) + "\n")
        del payload
    write_json(output / "config.json", config)
    write_json(output / "data_contract.json", dataset.metadata())
    write_json(output / "data_signature.json", dataset.signature())
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model.train()
    parameter_count = sum(p.numel() for p in model.parameters())
    print(f"TurboVLA | language={model_config['use_language']} | ft_history_gru={data.get('use_ft_history', False)} | "
          f"{data['action_mode']} | parameters={parameter_count:,} | "
          f"frames={len(dataset):,} | batch={train['batch_size']} | target samples={sampler.budget:,} | "
          f"target steps={sampler.target_steps:,} | starting step={step}", flush=True)

    def status(state, **extra):
        value = {"status": state, "step": step, "samples_seen": sampler.seen, "target_samples": sampler.budget,
                 "target_steps": sampler.target_steps, "elapsed_seconds": prior_seconds + time.perf_counter() - started,
                 "parameters": parameter_count, **extra}
        value.update(progress_fields(step, sampler.target_steps, sampler.seen, sampler.budget, value["elapsed_seconds"]))
        write_json(output / "training_status.json", value)
        return value

    def save():
        payload = {"schema": SCHEMA, "config": config, "step": step, "samples_seen": sampler.seen,
                   "raw": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                   "ema": {k: v.detach().cpu().clone() for k, v in ema.shadow.items()},
                   "optimizer": optimizer.state_dict(), "sampler": sampler.state_dict(), "rng": capture_rng(),
                   "loader_generator": loader_generator.get_state(), "data_signature": dataset.signature(),
                   "dataset_metadata": dataset.metadata(), "elapsed_seconds": prior_seconds + time.perf_counter() - started}
        path = output / f"checkpoint_{step:06d}.pt"
        temporary = path.with_suffix(".pt.tmp")
        torch.save(payload, temporary)
        temporary.replace(path)
        latest_temp = output / "latest.pt.tmp"
        latest_temp.unlink(missing_ok=True)
        os.link(path, latest_temp)
        latest_temp.replace(output / "latest.pt")
        # Only prune this run's own checkpoints; latest and the newest N survive.
        for old in sorted(output.glob("checkpoint_*.pt"))[:-train["keep_checkpoints"]]:
            old.unlink()
        print(f"Checkpoint saved | step={step} | samples={sampler.seen}", flush=True)

    last_saved = -1
    try:
        status("running")
        iterator = iter(loader)
        with metrics_path.open("a", buffering=1) as metrics:
            while sampler.seen < sampler.budget and (stop_after_step is None or step < stop_after_step):
                try:
                    batch = next(iterator)
                except StopIteration:
                    iterator = iter(loader)
                    batch = next(iterator)
                batch = batch_to_device(batch, device)
                count = len(batch["state"])
                set_learning_rate(optimizer, train, sampler.seen, count)
                optimizer.zero_grad(set_to_none=True)
                pixels = normalize_images(batch["images"], config["images"]["mean"], config["images"]["std"],
                                          color_jitter=config["images"].get("color_jitter"))
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
                    predicted = forward_batch(model, batch, pixels)
                    loss = masked_l1_loss(predicted, batch["actions"], batch["action_mask"])
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite loss")
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), train["max_grad_norm"])
                if not torch.isfinite(grad_norm):
                    raise FloatingPointError("Nonfinite gradients; optimizer was not updated")
                if step == 0:
                    vision_grad = sum(float(p.grad.detach().float().square().sum())
                                      for p in model.vision_encoder.parameters() if p.grad is not None) ** .5
                    if not vision_grad > 0:
                        raise ValueError("Vision backbone is not receiving gradients")
                    print(f"DINO backbone gradient norm={vision_grad:.6g}; language={model_config['use_language']}", flush=True)
                    if model_config["use_language"]:
                        projection_grad = model.text_encoder.text_projection.weight.grad
                        if projection_grad is None or not projection_grad.norm() > 0:
                            raise ValueError("Language projection is not receiving gradients")
                        print(f"Task instruction={batch['instruction'][0]!r}; language projection gradient norm="
                              f"{float(projection_grad.norm()):.6g}", flush=True)
                    if data.get("use_ft_history", False):
                        ft_grad = sum(float(p.grad.detach().float().square().sum())
                                      for p in model.action_head.ft_history_encoder.gru.parameters() if p.grad is not None) ** .5
                        if not ft_grad > 0:
                            raise ValueError("FT-history GRU is not receiving gradients")
                        print(f"FT-history GRU gradient norm={ft_grad:.6g}; hidden state reset per window", flush=True)
                optimizer.step()
                ema.update(model, count)
                sampler.consume(count)
                step += 1
                if step == 1 or step % train["log_interval"] == 0 or sampler.seen == sampler.budget or step == stop_after_step:
                    record = {"step": step, "samples_seen": sampler.seen, "loss": float(loss.detach()),
                              "gradient_norm": float(grad_norm), "learning_rates": {g["name"]: g["lr"] for g in optimizer.param_groups},
                              "elapsed_seconds": prior_seconds + time.perf_counter() - started,
                              "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else 0,
                              "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30 if device.type == "cuda" else 0}
                    record.update(progress_fields(step, sampler.target_steps, sampler.seen, sampler.budget, record["elapsed_seconds"]))
                    metrics.write(json.dumps(record) + "\n")
                    print(json.dumps(record), flush=True)
                    print(format_progress(data["action_mode"], record, sampler.target_steps), flush=True)
                    status("running", last_metric=record)
                if step % train["save_interval"] == 0 or sampler.seen == sampler.budget or step == stop_after_step:
                    save()
                    last_saved = step
        if last_saved != step:
            save()
        return status("completed" if sampler.seen == sampler.budget else "stopped_for_test")
    except BaseException as error:
        status("failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        dataset.close()
