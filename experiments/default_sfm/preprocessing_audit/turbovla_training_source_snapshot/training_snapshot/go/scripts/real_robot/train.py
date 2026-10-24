#!/usr/bin/env python3
"""Train TurboVLA on preprocessed real-robot HDF5 chunks."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from turbovla.training.real_robot import train_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--stop-after-step", type=int, help="Save and stop at a completed step for smoke/resume checks")
    args = parser.parse_args()
    result = train_model(json.loads(args.config.read_text()), args.output, args.device, args.resume, args.stop_after_step)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
