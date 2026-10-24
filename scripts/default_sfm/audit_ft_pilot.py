#!/usr/bin/env python3
"""Read nine task-balanced demos to discuss F/T scaling; never fit production stats."""

from __future__ import annotations

import argparse
import gc
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from audit_rpy import NumericAuditReader


def statistics(values: np.ndarray) -> dict:
    x = values.reshape(-1, 6).astype(np.float64)
    if not np.isfinite(x).all():
        raise ValueError("Nonfinite F/T sample; do not silently replace or normalize it")
    std = x.std(axis=0)
    return {
        "sample_count_per_channel": len(x), "mean": x.mean(axis=0).tolist(),
        "std_population": std.tolist(), "min": x.min(axis=0).tolist(), "max": x.max(axis=0).tolist(),
        "quantiles": {str(q): np.quantile(x, q, axis=0).tolist() for q in (.01, .5, .99)},
        "std_below_0_01": (std < .01).tolist(), "std_below_1e_6": (std < 1e-6).tolist(),
        "scale_ratio_floor_0_01_over_1e_6": (np.maximum(std, .01) / np.maximum(std, 1e-6)).tolist(),
    }


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    audit = root / "experiments/default_sfm/preprocessing_audit"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=root / "data/default_sfm")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.resolve().is_relative_to(args.raw_root.resolve()):
        raise ValueError("Use a new output file outside the raw dataset")
    inventory = json.loads((root / "experiments/default_sfm/setup_records/copy_manifest.json").read_text())
    starts = {r["path"]: r for r in json.loads((audit / "initial_stationary_audit.json").read_text())["demos"]}
    source_records = {r["path"]: r for r in inventory["files"]}
    tasks = sorted({r["task"] for r in inventory["folders"]})
    selected = [min(r["path"] for r in inventory["files"]
                    if r["path"].startswith(task + "_p1/") and r["path"].endswith(".pkl")) for task in tasks]
    entries, combined = [], []
    for task, relative in zip(tasks, selected):
        path = args.raw_root / relative
        before = path.stat()
        source = source_records[relative]
        if (before.st_size, before.st_mtime_ns) != (source["bytes"], source["mtime_ns"]):
            raise ValueError(f"Raw metadata changed: {relative}")
        with path.open("rb") as handle:
            reader = NumericAuditReader(handle)
            frames = pickle.load(reader)
        start = starts[relative]["start"]
        if starts[relative]["skipped"]:
            raise ValueError(f"No initial motion detected: {relative}")
        histories = []
        for frame in frames[start:]:
            history = np.asarray(frame["observations"]["right_force_history"])
            if history.shape != (1, 50, 6):
                raise ValueError(f"Unexpected current-observation F/T shape {history.shape}: {relative}")
            histories.append(history[0])
        history = np.stack(histories).astype(np.float64)
        state = np.stack([np.asarray(f["observations"]["state"])[0] for f in frames[start:]])
        latest_state_ft = state[:, [1, 2, 3, 10, 11, 12]].astype(np.float64)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError(f"Raw source changed while reading: {relative}")
        entry = {
            "task": task, "path": relative, "source_sha256_from_copy_manifest": source["sha256"],
            "raw_frames": len(frames), "start": start, "retained_frames": len(history),
            "all_zero_windows": int(np.all(history == 0, axis=(1, 2)).sum()),
            "all_zero_six_axis_samples": int(np.all(history == 0, axis=2).sum()),
            "repeated_identical_samples_entire_window": int(np.all(history == history[:, :1], axis=(1, 2)).sum()),
            "latest_history_vs_instantaneous_state_absolute_difference_max": np.max(np.abs(history[:, -1] - latest_state_ft), axis=0).tolist(),
            "statistics": statistics(history),
        }
        entries.append(entry); combined.append(history)
        print(json.dumps({"task": task, "retained_frames": len(history), "all_zero_windows": entry["all_zero_windows"]}), flush=True)
        del frames, histories, state
        gc.collect()
    full = np.concatenate(combined)
    report = {
        "status": "passed", "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Pilot only: first lexicographic demo at p1 for each of nine tasks; 9 of 450 demos.",
        "selection": selected, "channel_order": ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"],
        "summary": {"demos": len(entries), "retained_frames": len(full),
                    "all_zero_windows": sum(r["all_zero_windows"] for r in entries),
                    "statistics": statistics(full)},
        "demos": entries, "sampling_rate_hz": {"value": 500, "source": "user", "per_sample_timestamps_verified": False},
        "zero_window_semantics": {
            "status": "intentional_valid_input", "source": "user_confirmation",
            "preserve": True, "include_in_statistics": True, "remove_or_impute": False,
        },
        "latest_sample_comparison_limit": "Equality or differences with instantaneous state do not establish sensor timestamp alignment.",
        "production_statistics_fitted": False, "normalization_floor_selected": False,
        "raw_data_modified": False, "training_preprocessing_applied": False, "training_started": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2); handle.write("\n")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
