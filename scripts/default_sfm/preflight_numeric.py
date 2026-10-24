#!/usr/bin/env python3
"""Read-only full-cohort numeric preflight; skipped RGB buffers are never saved."""
import argparse
import gc
import json
from pathlib import Path
import pickle
import time

import numpy as np
import yaml

from audit_rpy import NumericAuditReader
from preprocess import AUDIT, REPO, Decoder, check_collection, references, utc, validate_plan, write_json
from turbovla.data.sfm_preprocessing import (
    Moments, chunk_actions, delta_actions, held_targets, initial_start, require, sample, state_physical,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=REPO / "data/default_sfm")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists() and not args.output.resolve().is_relative_to(args.raw_root.resolve()), "Use new output outside raw data")
    plan = yaml.safe_load((REPO / "experiments/default_sfm/experiment_plan.yaml").read_text())
    validate_plan(plan)
    manifest = json.loads((REPO / "experiments/default_sfm/setup_records/copy_manifest.json").read_text())
    collection, _ = check_collection(args.raw_root, manifest["files"])
    expected = {r["path"]: r for r in json.loads((AUDIT / "initial_stationary_audit.json").read_text())["demos"]}
    records = sorted((r for r in manifest["files"] if r["path"].endswith(".pkl")), key=lambda r: r["path"])
    ref = references()
    state_m, ft_m, action_m = Moments(), Moments(), Moments()
    results = []
    started = time.monotonic()
    for record in records:
        path = args.raw_root / record["path"]
        before = path.stat()
        require((before.st_size, before.st_mtime_ns) == (record["bytes"], record["mtime_ns"]), "Input metadata differs")
        with path.open("rb") as f:
            rows = pickle.load(NumericAuditReader(f))
        state = np.stack([sample(r["observations"]["state"], (19,), "state") for r in rows])
        target = np.stack([sample(r["actions"], (7,), "action") for r in rows])
        times = np.asarray([r["time"] for r in rows])
        require(np.isfinite(times).all() and np.all(np.diff(times) > 0), f"Nonmonotonic times: {path}")
        require({r.get("infos", {}).get("action_format") for r in rows} == {"absolute_tcp_xyz_rpy_gripper"}, "Unexpected action format")
        terminals = [bool(r.get("dones", False) or r.get("terminated", False) or r.get("truncated", False)) for r in rows]
        require(terminals[-1] and sum(terminals) == 1, f"Not one complete episode: {path}")
        cfg = plan["dataset"]["frame_selection"]["detection_config"]
        start, _ = initial_start(state, cfg)
        require(start == expected[record["path"]]["start"] == ref["crop_start"](rows, Decoder(), cfg)[0], "Start mismatch")
        held, _, present, fallback = held_targets(rows, state, collection["env"]["right"]["gripper_control"])
        legacy = ref["GripperLabels"](rows, collection, Decoder(), dict(source_mode="binary", mode="absolute", command_range=[-1., 1.], observation_range=[0., 1.], grasp={"enabled": False}), ["right"]).labels[:, 0]
        require(np.array_equal(held, legacy), "Held/reference mismatch")
        ft = np.stack([sample(r["observations"]["right_force_history"], (50, 6), "FT") for r in rows[start:]])
        physical = state_physical(state[start:])
        delta = delta_actions(state, target, held)[start:]
        actions, mask = chunk_actions(delta, 5)
        state_m.add(physical); ft_m.add(ft); action_m.add(actions[mask][:, :6])
        after = path.stat()
        require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), "Raw input changed")
        results.append(dict(path=record["path"], original_frames=len(rows), start=start, frames=len(ft),
            all_zero_ft_windows=int(np.all(ft == 0, axis=(1, 2)).sum()), receipt_frames=int(present.sum()),
            receipt_fallback_frames=fallback, valid_action_entries=int(mask.sum()),
            post_wrap_jumps_gt_pi=(np.abs(np.diff(physical[:, 4:7], axis=0)) > np.pi).sum(axis=0).tolist(),
            reference_held_labels_match=True, reference_start_matches=True))
        del rows
        gc.collect()
        if len(results) % 25 == 0:
            print(json.dumps({"checked": len(results), "total": 450, "elapsed_seconds": time.monotonic() - started}), flush=True)
    require(len(results) == 450 and sum(r["frames"] for r in results) == 138985, "Wrong cohort totals")
    summary = {key: sum(r[key] for r in results) for key in ["original_frames", "frames", "all_zero_ft_windows", "receipt_frames", "receipt_fallback_frames", "valid_action_entries"]}
    summary["removed_initial_frames"] = summary["original_frames"] - summary["frames"]
    summary["demos"] = len(results)
    report = dict(status="passed", checked_at_utc=utc(), scope="full 450-demo numeric preflight; RGB skipped, not training data",
        raw_sha256_rechecked=False, raw_metadata_matches=True, raw_data_modified=False, training_started=False,
        zero_history_semantics="intentional_valid_input", summary=summary,
        physical_statistics={"state": state_m.result("standard", 1e-6), "ft": ft_m.result("standard", .01),
                             "action_pose": action_m.result("symlog_minmax", 1e-6, 20.)},
        post_wrap_jumps_gt_pi=np.sum([r["post_wrap_jumps_gt_pi"] for r in results], axis=0).tolist(),
        demos=results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
