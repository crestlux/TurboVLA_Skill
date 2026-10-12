"""Run the real RoboTwin recipe with bounded training and per-rank audit records."""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import torch

from starVLA.training import train_robotwin_clean_act_pi05_recipe as recipe


class SmokeTrainer(recipe.EMAVLATrainer):
    def prepare_training(self):
        if not 1 <= int(self.config.trainer.max_train_steps) <= 20:
            raise ValueError("The smoke entrypoint is limited to 1–20 optimizer updates.")
        super().prepare_training()
        self.smoke_micro_batches = 0
        self.smoke_ema_updates = 0
        self.smoke_eval_calls = 0
        self.smoke_save_calls = 0
        self.smoke_updates = []
        self.smoke_pending_losses = []
        self.smoke_data_seconds = 0.0
        self.smoke_start = time.perf_counter()
        self.smoke_update_start = self.smoke_start
        module = self.accelerator.unwrap_model(self.model)
        candidates = [(name, p) for name, p in module.named_parameters()
                      if p.requires_grad and "action_head" in name]
        self.smoke_sample_name, self.smoke_sample_param = candidates[-1]
        self.smoke_sample_initial = self.smoke_sample_param.detach().float().cpu().clone()
        torch.cuda.reset_peak_memory_stats(self.accelerator.device)

    def _get_next_batch(self):
        start = time.perf_counter()
        batch = super()._get_next_batch()
        self.smoke_data_seconds += time.perf_counter() - start
        return batch

    def _update_ema(self):
        super()._update_ema()
        self.smoke_ema_updates += 1

    def _train_step(self, batch_vla, batch_vlm=None):
        before = (int(self.model.global_steps), int(self.lr_scheduler.last_epoch),
                  self.smoke_ema_updates)
        metrics = super()._train_step(batch_vla, batch_vlm=batch_vlm)
        loss = float(metrics["action_dit_loss"])
        assert math.isfinite(loss), f"Non-finite loss: {loss}"
        self.smoke_micro_batches += 1
        self.smoke_pending_losses.append(loss)
        after = (int(self.model.global_steps), int(self.lr_scheduler.last_epoch),
                 self.smoke_ema_updates)
        expected = int(self.accelerator.sync_gradients)
        assert tuple(b - a for a, b in zip(before, after)) == (expected,) * 3, (before, after)
        assert not self.optimizer.step_was_skipped, "Optimizer skipped an update"
        if expected:
            now = time.perf_counter()
            accumulation = int(self.accelerator.gradient_accumulation_steps)
            assert len(self.smoke_pending_losses) == accumulation
            update = {
                "optimizer_updates": after[0], "scheduler_updates": after[1],
                "ema_updates": after[2], "micro_batches": self.smoke_micro_batches,
                "mean_micro_batch_loss": sum(self.smoke_pending_losses) / accumulation,
                "learning_rate": float(self.lr_scheduler.get_last_lr()[0]),
                "wall_seconds_since_previous_update": now - self.smoke_update_start,
                "data_wait_seconds_total": self.smoke_data_seconds,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(self.accelerator.device),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(self.accelerator.device),
            }
            self.smoke_updates.append(update)
            self.smoke_pending_losses.clear()
            self.smoke_update_start = now
            print(f"[SMOKE rank={self.accelerator.process_index}] {json.dumps(update)}", flush=True)
        return metrics

    def eval_action_model(self, step_metrics=None):
        self.smoke_eval_calls += 1
        return super().eval_action_model(step_metrics)

    def _save_checkpoint(self):
        self.smoke_save_calls += 1
        return super()._save_checkpoint()

    def train(self):
        status = "failed"
        try:
            super().train()
            steps = int(self.config.trainer.max_train_steps)
            accumulation = int(self.accelerator.gradient_accumulation_steps)
            assert self.completed_steps == int(self.model.global_steps) == steps
            assert int(self.lr_scheduler.last_epoch) == self.smoke_ema_updates == steps
            assert self.smoke_micro_batches == steps * accumulation
            assert self.smoke_eval_calls == steps // int(self.config.trainer.eval_interval)
            assert self.smoke_save_calls == steps // int(self.config.trainer.save_interval)
            assert not self.smoke_pending_losses
            delta = self.smoke_sample_param.detach().float().cpu() - self.smoke_sample_initial
            assert torch.isfinite(delta).all() and delta.abs().max().item() > 0
            status = "passed"
        finally:
            report = {
                "status": status, "rank": self.accelerator.process_index,
                "world_size": self.accelerator.num_processes,
                "global_batch": self.total_batch_size,
                "micro_batches": self.smoke_micro_batches,
                "optimizer_updates": int(self.model.global_steps),
                "scheduler_updates": int(self.lr_scheduler.last_epoch),
                "ema_updates": self.smoke_ema_updates,
                "batch_eval_calls": self.smoke_eval_calls,
                "checkpoint_save_calls": self.smoke_save_calls,
                "training_and_save_wall_seconds": time.perf_counter() - self.smoke_start,
                "data_wait_seconds": self.smoke_data_seconds,
                "sample_parameter": self.smoke_sample_name,
                "sample_parameter_max_change": float((self.smoke_sample_param.detach().float().cpu()
                                                       - self.smoke_sample_initial).abs().max()),
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(self.accelerator.device),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(self.accelerator.device),
                "updates": self.smoke_updates,
            }
            path = Path(self.config.output_dir) / f"smoke_audit_rank{self.accelerator.process_index}.json"
            path.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    recipe.EMAVLATrainer = SmokeTrainer
    recipe.main()
