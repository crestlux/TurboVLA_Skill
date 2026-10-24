#!/usr/bin/env python3
"""Describe one trusted raw robot demo without transforming or saving its data."""

from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.file.open("rb") as stream:
        frames = pickle.load(stream)
    assert isinstance(frames, list) and frames
    observations = frames[0]["observations"]
    dt = np.diff(np.asarray([frame["time"] for frame in frames], dtype=np.float64))
    finite = {}
    shapes = {}
    for key in ("state", "right_force_history", "right_state_history"):
        finite[key] = all(np.isfinite(frame["observations"][key]).all() for frame in frames)
        shapes[key] = [list(shape) for shape in sorted({tuple(frame["observations"][key].shape) for frame in frames})]
    finite["actions"] = all(np.isfinite(frame["actions"]).all() for frame in frames)
    report = {
        "file": str(args.file.resolve()),
        "scope": "One raw demo; not a full-cohort schema or timing audit",
        "preprocessing_applied": False,
        "frame_count": len(frames),
        "observation_schema": {
            key: {"shape": list(value.shape), "dtype": str(value.dtype)} for key, value in observations.items()
        },
        "shapes_across_frames": shapes,
        "numeric_values_finite": finite,
        "action_shape": list(frames[0]["actions"].shape),
        "action_format_counts": dict(Counter(frame["infos"].get("action_format") for frame in frames)),
        "observed_timestamp_delta_seconds": {
            "min": float(dt.min()),
            "median": float(np.median(dt)),
            "max": float(dt.max()),
        },
        "ft_sampling_hz": {"value": 500, "source": "user", "per_sensor_sample_timestamps_verified": False},
        "frames_with_gripper_target_receipts": sum(
            "gripper_target_percent" in frame["infos"].get("command_receipts", {}).get("right", {}) for frame in frames
        ),
        "state_field_layout_from_collection_code": [
            {"name": "right/gripper_pose", "slice": [0, 1]},
            {"name": "right/tcp_force", "slice": [1, 4]},
            {"name": "right/tcp_pose_xyz_rpy", "slice": [4, 10]},
            {"name": "right/tcp_torque", "slice": [10, 13]},
            {"name": "right/tcp_vel", "slice": [13, 19]},
        ],
        "layout_source": "csilab:Real_Robo/legacy/replay_demo_to_video.py build_state_index + collection proprio_keys",
        "note": "Raw state already contains 6 F/T components. GRU variants must decide whether to retain them.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps({k: report[k] for k in ("frame_count", "numeric_values_finite", "action_format_counts")}, indent=2)
    )


if __name__ == "__main__":
    main()
