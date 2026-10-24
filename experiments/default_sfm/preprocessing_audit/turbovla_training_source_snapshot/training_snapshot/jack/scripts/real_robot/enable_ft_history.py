#!/usr/bin/env python3
"""Create an FT-history GRU recipe from an existing baseline without changing it."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from turbovla.data.real_robot import RealRobotDataset
from turbovla.data.ft_history import resampling_contract
from turbovla.training.real_robot import configure, write_json


def enable_ft_history(config, *, history_length=50, hidden_dim=128, num_layers=1, std_floor=0.01,
                      source_hz=None, target_hz=None):
    config = json.loads(json.dumps(config))
    config.pop("comparison", None)
    config["experiment"] = config.get("experiment", "turbovla") + "_ft_history_gru"
    if (source_hz is None) != (target_hz is None):
        raise ValueError("Supply both source_hz and target_hz")
    if source_hz is not None:
        settings = resampling_contract(dict(source_hz=source_hz, target_hz=target_hz, source_length=history_length))
        config["data"]["ft_history_resampling"] = settings
        history_length //= round(source_hz / target_hz)
        config["experiment"] += f"_{target_hz:g}hz_nominal"
    config["model"]["ft_history"] = dict(enabled=True, history_length=history_length, input_dim=6,
                                          hidden_dim=hidden_dim, num_layers=num_layers, std_floor=std_floor)
    for key in ("use_ft_history", "ft_history_length", "ft_history_dim"):
        config["data"].pop(key, None)
    config = configure(config)
    dataset = RealRobotDataset({**config["data"], "preload_images": False})
    dataset.close()
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Existing baseline training JSON")
    parser.add_argument("--output", type=Path, required=True, help="New JSON; must not already exist")
    parser.add_argument("--history-length", type=int, default=50)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--std-floor", type=float, default=0.01)
    parser.add_argument("--source-hz", type=float)
    parser.add_argument("--target-hz", type=float)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; choose a new recipe path")
    config = enable_ft_history(json.loads(args.config.read_text()), history_length=args.history_length,
                               hidden_dim=args.hidden_dim, num_layers=args.num_layers, std_floor=args.std_floor,
                               source_hz=args.source_hz, target_hz=args.target_hz)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, config)
    print(json.dumps({"output": str(args.output.resolve()), "ft_history": config["model"]["ft_history"]}, indent=2))


if __name__ == "__main__":
    main()
