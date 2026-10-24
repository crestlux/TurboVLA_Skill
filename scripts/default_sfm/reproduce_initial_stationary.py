#!/usr/bin/env python3
"""Reproduce recorded start-index diagnostics without converting training data."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def main() -> None:
    repo = Path(__file__).resolve().parents[2]
    audit = repo / "experiments/default_sfm/preprocessing_audit"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=repo / "data/default_sfm")
    parser.add_argument("--numeric-root", type=Path, default=audit / "rpy")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw_root = args.raw_root.resolve()
    output = args.output.resolve()
    require(not output.is_relative_to(raw_root), "Diagnostic output must be outside raw data")
    require(not output.exists(), f"Refusing to overwrite existing output: {output}")

    plan_path = repo / "experiments/default_sfm/experiment_plan.yaml"
    plan = yaml.safe_load(plan_path.read_text())
    selection = plan["dataset"]["frame_selection"]
    require(selection["detection_parameters_confirmed"], "Start detection is not confirmed")
    cfg = selection["detection_config"]
    reference_path = audit / "initial_stationary_audit.json"
    reference = json.loads(reference_path.read_text())
    require(cfg == reference["config"], "Current settings differ from the recorded diagnostic")
    expected = {row["path"]: row for row in reference["demos"]}

    source = repo / reference["algorithm_source"]
    require(digest(source) == reference["algorithm_source_sha256"], "Reference code hash changed")
    # Execute only these two verified source functions; do not import the converter
    # or any command that writes HDF5 datasets or fits normalization statistics.
    nodes = [node for node in ast.parse(source.read_text()).body
             if isinstance(node, ast.FunctionDef) and node.name in {"numeric", "crop_start"}]
    require({node.name for node in nodes} == {"numeric", "crop_start"}, "Missing reference functions")
    scope = {"np": np, "Rotation": Rotation, "PreprocessError": ValueError}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), scope)

    class CachedDecoder:
        def channel(self, observation, key):
            indices = {"right/tcp_pose": slice(4, 10), "right/gripper_pose": slice(0, 1)}
            return scope["numeric"](observation["state"][indices[key]], key)

    numeric_report_path = args.numeric_root / "report.json"
    numeric = json.loads(numeric_report_path.read_text())
    require(numeric["status"] == "verified", "The numeric audit is incomplete")
    require(len(numeric["demos"]) == len(expected) == 450, "Expected the complete 450-demo cohort")
    require({row["path"] for row in numeric["demos"]} == set(expected), "Demo path sets differ")
    rows = []
    for record in numeric["demos"]:
        raw = raw_root / record["path"]
        st = raw.stat()
        require((st.st_size, st.st_mtime_ns) == (record["source_bytes"], record["source_mtime_ns"]),
                f"Raw metadata differs from the numeric audit: {record['path']}")
        cache = args.numeric_root / record["numeric_artifact"]
        with np.load(cache, allow_pickle=False) as data:
            state, times = data["state"], data["time"]
        require(state.shape == (record["frame_count"], 19) and np.isfinite(state).all(),
                f"Invalid cached state: {record['path']}")
        start, info = scope["crop_start"](
            [{"observations": {"state": value}} for value in state], CachedDecoder(), cfg)
        xyz, rpy, grip = state[:, 4:7].astype(float), state[:, 7:10].astype(float), state[:, 0].astype(float)
        rotations = Rotation.from_euler("xyz", rpy)
        moving = ((np.linalg.norm(xyz - xyz[0], axis=1) > cfg["translation_threshold_m"])
                  | ((rotations[0].inv() * rotations).magnitude() > cfg["rotation_threshold_rad"])
                  | (np.abs(grip - grip[0]) > cfg["gripper_threshold"]))
        n = cfg["consecutive_frames"]
        hits = np.flatnonzero(np.convolve(moving.astype(int), np.ones(n, dtype=int), mode="valid") == n)
        independent = max(0, int(hits[0]) - cfg["pre_roll_frames"]) if len(hits) else None
        require(start == independent, f"Independent start-index check failed: {record['path']}")
        kept = len(state) - start if start is not None else 0
        result = {"path": record["path"], **info, "retained_frames": kept,
                  "removed_initial_frames": len(state) - kept, "skipped": start is None,
                  "trimmed_prefix_duration_s": float(times[start] - times[0]) if start is not None else None}
        for key in ("start", "detected_motion_frame", "original_frames", "retained_frames", "skipped"):
            require(result[key] == expected[record["path"]][key],
                    f"Stored diagnostic differs for {record['path']}: {key}")
        rows.append(result)

    summary = {"demos": len(rows), "no_motion_demos": sum(row["skipped"] for row in rows),
               "original_frames": sum(row["original_frames"] for row in rows),
               "retained_frames": sum(row["retained_frames"] for row in rows),
               "removed_initial_frames": sum(row["removed_initial_frames"] for row in rows)}
    require(all(value == reference["summary"][key] for key, value in summary.items()),
            "Aggregate counts differ from the recorded diagnostic")
    report = {"status": "passed", "checked_at_utc": datetime.now(timezone.utc).isoformat(),
              "mode": "read_only_numeric_diagnostic", "config": cfg,
              "plan_sha256": digest(plan_path), "reference_audit_sha256": digest(reference_path),
              "reference_source_sha256": digest(source), "numeric_report_sha256": digest(numeric_report_path),
              "raw_metadata_checks": "all source sizes and mtime_ns match; not a new raw-file content hash pass",
              "per_demo_start_indices_match_reference": True, "summary": summary, "demos": rows,
              "raw_data_modified": False, "training_preprocessing_applied": False,
              "normalization_fitted": False, "training_started": False}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(json.dumps({"status": report["status"], "summary": summary, "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
