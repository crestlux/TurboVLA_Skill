"""Bounded real optimizer benchmark, using smoke audits without checkpoint I/O."""
from smoke_train import SmokeTrainer, recipe


class BenchmarkTrainer(SmokeTrainer):
    def _finalize_training(self):
        # Benchmark output consists of per-rank JSON audits, not model weights.
        self.accelerator.wait_for_everyone()


if __name__ == "__main__":
    recipe.EMAVLATrainer = BenchmarkTrainer
    recipe.main()
