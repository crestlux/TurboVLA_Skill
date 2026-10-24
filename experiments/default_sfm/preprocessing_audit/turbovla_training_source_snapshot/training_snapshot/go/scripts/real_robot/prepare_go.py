#!/usr/bin/env python3
"""Audit GO HDF5 bundles and prepare instruction-labelled TurboVLA recipes."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys

import h5py
import numpy as np
from scipy.spatial.transform import Rotation
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from turbovla.data.real_robot import (
    GripperRelabelConfig, RealRobotDataset, inverse_normalized_actions,
    latch_gripper_commands, read_task_instructions,
)
from turbovla.evaluation.real_trajectory_metrics import decode_targets, target_errors
from turbovla.training.real_robot import configure, write_json

DEFAULT_INSTRUCTION = "Insert the USB connector into the USB port."


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            result.update(block)
    return result.hexdigest()


def audit_episode(path, mode, statistics, previous, default, gripper_labels="latched"):
    require(gripper_labels in ("latched", "original"), "Unknown gripper target semantics")
    digest = sha256(path)
    require(digest == previous["output_sha256"], f"Previous validation hash differs: {path}")
    shared, image_stats = {}, {}
    with h5py.File(path, "r") as f:
        n, horizon, dimensions = f["actions"].shape
        require((horizon, dimensions) == (50, 7), "Expected stored [T,50,7] actions")
        require(f.attrs["action_transform"] == mode and f.attrs["image_color"] == "RGB", "Schema mismatch")
        require(json.loads(f["metadata/normalization"][()]) == statistics, "Normalization mismatch")
        contract = json.loads(f["metadata/action_contract"][()])
        require(contract["mode"] == mode and contract["relative_gripper_action_mode"] == "absolute", "Action contract mismatch")
        require(statistics["actions"]["method"] == "symlog_minmax", "Expected current GO normalization")
        mask, actions = f["action_mask"][:], f["actions"][:]
        expected_mask = np.arange(n)[:, None] + np.arange(horizon)[None, :] < n
        require(mask.dtype == np.bool_ and np.array_equal(mask, expected_mask), "Invalid tail mask")
        require(np.isfinite(actions).all() and np.abs(actions[mask]).max() <= 1 + 1e-10, "Invalid normalized actions")
        require(not actions[~mask].any(), "Nonzero normalized padding")
        physical = inverse_normalized_actions(torch.from_numpy(actions), statistics["actions"]).numpy()
        anchor, offset, xyz, quat, grip = decode_targets(
            physical, f["action_reference_poses"][:, 0], mask, mode, contract["relative_action_frame_yaw_deg"])
        targets = f["source_actions"][:]
        source_commands = targets[:, 6].copy()
        if gripper_labels == "latched":
            require("gripper_targets" in f and contract.get("gripper_relabel") is not None,
                    "Expected stored latched gripper targets")
            labels = f["gripper_targets"][:]
            targets[:, 6] = labels
        else:
            require("gripper_targets" not in f and contract.get("gripper_relabel") is None
                    and "metadata/gripper_relabel" not in f,
                    "Original-command dataset must not contain active gripper relabeling")
            require(contract.get("gripper") == "original command, no relabeling",
                    "Expected the original signed gripper command contract")
            require(statistics["actions"]["offset"][6] == 0 and statistics["actions"]["scale"][6] == 1,
                    "Original gripper commands must retain reference normalization coefficients")
            require(np.isfinite(source_commands).all() and np.abs(source_commands).max() <= 1,
                    "Invalid original signed gripper commands")
            labels = source_commands
        position, rotation, gripper = target_errors(xyz, quat, grip, targets[anchor + offset])
        require(position.max() < 1e-7 and rotation.max() < 1e-7 and gripper.max() < 1e-10,
                "Actions do not reconstruct the original physical TCP/gripper targets")
        state_stats = statistics["observations"]["state"]
        state = f["observations/state"][:] * np.asarray(state_stats["scale"]) + np.asarray(state_stats["offset"])
        columns = statistics["schema"]["state_columns"]["state"]
        gripper_column = columns.index("right/gripper_pose/pos")
        if gripper_labels == "latched":
            relabel = contract["gripper_relabel"]
            settings = GripperRelabelConfig(**{key: relabel[key] for key in (
                "method", "close_threshold", "open_threshold", "initial_pose_threshold")})
            expected_labels = latch_gripper_commands(source_commands, state[0, gripper_column], settings)
            require(np.array_equal(labels, expected_labels), "Stored gripper latch differs from original commands")
        pose = state[:, [columns.index(f"right/tcp_pose/{key}") for key in ("x", "y", "z", "roll", "pitch", "yaw")]]
        reference = f["action_reference_poses"][:, 0]
        require(np.abs(pose[:, :3] - reference[:, :3]).max() < 1e-7, "State/reference positions differ")
        state_rotation_error = (Rotation.from_euler("xyz", pose[:, 3:]).inv()
                                * Rotation.from_quat(reference[:, 3:], scalar_first=True)).magnitude().max()
        require(state_rotation_error < 1e-7, "State/reference orientations differ")

        # Read every numerical entry and every stored image, including unused
        # next observations, and compare their digests across the two modes.
        def inspect(name, dataset):
            if not isinstance(dataset, h5py.Dataset) or not dataset.ndim:
                return
            require(len(dataset) == n, f"Episode length differs: {name}")
            values = dataset[:]
            if name != "timestamps":
                require(np.isfinite(values).all(), f"Nonfinite values: {name}")
            if "/image_" in name:
                require(values.shape == (n, 224, 224, 3) and values.dtype == np.uint8, f"Invalid RGB camera: {name}")
                require(not values[:, :28].any() and not values[:, 196:].any(), f"Invalid letterbox padding: {name}")
                content = values[:, 28:196]
                deviations = content.reshape(n, -1).std(axis=1)
                require(np.all(deviations > 1), f"Blank/corrupt camera frames: {name}")
                image_stats[name] = {"frames": n, "min_frame_std": float(deviations.min()),
                                     "mean_intensity": float(content.mean())}
            if name not in ("actions",):
                shared[name] = {"shape": list(values.shape), "dtype": str(values.dtype),
                                "sha256": hashlib.sha256(values.tobytes()).hexdigest()}
        f.visititems(inspect)
        timestamps = f["timestamps"][:]
        finite = np.isfinite(timestamps)
        require(finite.all() or not finite.any(), "Partially missing timestamps")
        require(not finite.any() or np.all(np.diff(timestamps) > 0), "Non-increasing timestamps")
        indices = f["source_indices"][:]
        require(np.array_equal(indices, np.arange(indices[0], indices[0] + n)), "Source frame indices are discontinuous")
        instructions, sources = read_task_instructions(f, n, default)
        entry = {"instructions": instructions, "sources": sources}
        report = {"file": path.name, "frames": n, "valid_chunk_entries": int(mask.sum()), "sha256": digest,
                  "previous_validation_hash_matches": True, "source_file_exists": Path(f.attrs["source_file"]).is_file(),
                  "max_reconstruction_position_error_mm": float(position.max()),
                  "max_reconstruction_rotation_error_deg": float(rotation.max()),
                  "max_reconstruction_gripper_error": float(gripper.max()),
                  "timestamps_missing": int((~finite).sum()),
                  "timestamp_step_median_s": float(np.median(np.diff(timestamps))) if finite.all() else None,
                  "gripper_target_kind": gripper_labels,
                  "gripper_closed_frames": int((labels <= -.5).sum()), "gripper_open_frames": int((labels >= .5).sum()),
                  "gripper_hold_frames": int(((labels > -.5) & (labels < .5)).sum()),
                  "gripper_exact_zero_frames": int((labels == 0).sum()),
                  "source_command_counts": {"close": int((source_commands <= -.5).sum()),
                                            "hold": int(((source_commands > -.5) & (source_commands < .5)).sum()),
                                            "open": int((source_commands >= .5).sum())},
                  "gripper_valid_chunk_counts": {"close": int((grip <= -.5).sum()),
                                                 "hold": int(((grip > -.5) & (grip < .5)).sum()),
                                                 "open": int((grip >= .5).sum())},
                  "instruction_sources": dict(Counter(sources)), "images": image_stats,
                  "physical_action_min": physical[mask].min(axis=0).tolist(),
                  "physical_action_max": physical[mask].max(axis=0).tolist(),
                  "numeric_and_image_digests": shared}
    return report, entry


def prepare(source, output, default=DEFAULT_INSTRUCTION, epochs=80, batch_size=64,
            gripper_labels="latched", reference_run=None):
    source, output = Path(source).resolve(), Path(output).resolve()
    require(epochs > 0 and batch_size > 0 and default.strip(), "Invalid training recipe")
    require(gripper_labels in ("latched", "original"), "Unknown gripper target semantics")
    reference_run = Path(reference_run).resolve() if reference_run is not None else None
    reference_audit = (json.loads((reference_run / "integrity_report.json").read_text())
                       if reference_run is not None else None)
    require(not (source / "INCOMPLETE").exists(), "Source preprocessing is incomplete")
    previous = json.loads((source / "validation_report.json").read_text())
    require(previous.get("validated") is True, "Source does not have a successful preprocessing validation")
    output.mkdir(parents=True, exist_ok=False)
    incomplete = output / "INCOMPLETE"
    incomplete.write_text("Full GO dataset audit and instruction preparation in progress.\n")
    result = {"passed": False, "created_utc": datetime.now(timezone.utc).isoformat(), "source": str(source),
              "output": str(output), "default_instruction": default, "epochs": epochs, "modes": {},
              "source_data_mutated": False, "image_preprocessing": "reuse validated crop/letterbox without applying it again",
              "gripper_target_kind": gripper_labels,
              "comparison_reference_run": str(reference_run) if reference_run is not None else None,
              "training_augmentation": {"brightness": .2, "contrast": .2, "saturation": .2, "hue": .02,
                                        "padding_tblr": [28, 28, 0, 0]},
              "split": "all 50 demonstrations for training; no held-out validation or test set"}
    for mode in ("delta", "relative"):
        original, prepared = source / mode, output / "prepared" / mode
        prepared.mkdir(parents=True)
        statistics = json.loads((original / "normalization.json").read_text())
        manifest = json.loads((original / "manifest.json").read_text())
        retained = {Path(item["output"]).name: item for item in manifest if not item.get("skipped", False)}
        paths = sorted(original.glob("*.hdf5"))
        require([path.name for path in paths] == sorted(retained), "Manifest episode names differ")
        old_checks = {item["file"]: item for item in previous["modes"][mode]["episode_checks"]}
        checks, instructions = [], {}
        for index, path in enumerate(paths):
            check, entry = audit_episode(path, mode, statistics, old_checks[path.name], default, gripper_labels)
            require(check["frames"] == retained[path.name]["frames"], "Manifest frame count differs")
            if reference_audit is not None:
                baseline = reference_audit["modes"][mode]["episode_checks"][index]
                require(baseline["file"] == path.name, "Comparison episode pairing differs")
                baseline_digests = {name: value for name, value in baseline["numeric_and_image_digests"].items()
                                    if name != "gripper_targets"}
                current_digests = {name: value for name, value in check["numeric_and_image_digests"].items()
                                   if name != "gripper_targets"}
                require(current_digests == baseline_digests, "Compared observations/source commands differ from reference run")
                require(old_checks[path.name].get("source_sha256") == baseline["sha256"],
                        "Ablation source is not the audited reference episode")
                baseline_root = Path(reference_audit["source"]) / mode
                with h5py.File(path, "r") as current, h5py.File(baseline_root / path.name, "r") as previous_file:
                    require(np.array_equal(current["actions"][:, :, :6], previous_file["actions"][:, :, :6]),
                            "Normalized arm action channels differ from reference")
                    baseline_stats = json.loads(previous_file["metadata/normalization"][()])
                    for target in ("actions",):
                        for coefficient in ("method", "temperature", "offset", "scale"):
                            require(statistics[target][coefficient] == baseline_stats[target][coefficient],
                                    "Action normalization coefficients differ from reference")
                    require(statistics["observations"] == baseline_stats["observations"],
                            "State normalization differs from reference")
                check["reference_shared_observations_and_source_commands_equal"] = True
                check["reference_arm_actions_bitwise_equal"] = True
                check["reference_normalization_coefficients_equal"] = True
            if mode == "relative":
                other = result["modes"]["delta"]["episode_checks"][index]
                require(path.name == other["file"], "Delta/relative episode pairing differs")
                require(check["numeric_and_image_digests"] == other["numeric_and_image_digests"],
                        "Delta/relative observations or source targets differ")
                require(instructions_match(output / "prepared/delta/instructions.json", path.name, entry),
                        "Delta/relative instructions differ")
            checks.append(check)
            instructions[path.name] = entry
            (prepared / path.name).symlink_to(path)
            print(f"{mode}: audited {index + 1}/{len(paths)} {path.name}", flush=True)
        frames = sum(item["frames"] for item in checks)
        valid = sum(item["valid_chunk_entries"] for item in checks)
        require((len(paths), frames) == (50, 10090), "Unexpected GO episode/frame count")
        require(valid == statistics["actions"]["count"], "Action statistics count differs")
        fitted_dimensions = 7 if gripper_labels == "latched" else 6
        require(np.allclose(np.min([item["physical_action_min"] for item in checks], axis=0)[:fitted_dimensions],
                            statistics["actions"]["min"][:fitted_dimensions], atol=1e-12),
                "Action minima differ from normalization")
        require(np.allclose(np.max([item["physical_action_max"] for item in checks], axis=0)[:fitted_dimensions],
                            statistics["actions"]["max"][:fitted_dimensions], atol=1e-12),
                "Action maxima differ from normalization")
        for name in ("normalization.json", "config.yaml", "inference_config.json"):
            shutil.copy2(original / name, prepared / name)
        for name in ("gripper_relabel.json", "gripper_ablation.json", "episode_summary.csv"):
            if (original / name).is_file():
                shutil.copy2(original / name, prepared / name)
        prepared_manifest = [dict(item, output=str(prepared / Path(item["output"]).name)) for item in manifest]
        write_json(prepared / "manifest.json", prepared_manifest)
        write_json(prepared / "instructions.json", {"version": 1, "default_instruction": default, "episodes": instructions})
        config_path = (reference_run / mode / "config.json" if reference_run is not None
                       else ROOT / f"experiments/real_robot/configs/ff_usb_v2_{mode}.json")
        config = json.loads(config_path.read_text())
        baseline_config = json.loads(config_path.read_text()) if reference_run is not None else None
        config.pop("comparison", None)
        config["experiment"] = ("go_turbovla_original_gripper_instruction_color_jitter" if gripper_labels == "original"
                                else "go_turbovla_instruction_color_jitter")
        config["data"].update(dataset_dir=str(prepared), expected_episodes=len(paths), expected_frames=frames,
                              use_language=True, default_instruction=default, gripper_relabel=None)
        config["model"]["use_language"] = True
        config["model"]["text"] = {"model_name_or_path": str(ROOT / "pretrained/backbones/bert-base-uncased"),
                                   "max_length": 32, "padding_length": 16, "frozen": True, "local_files_only": True}
        config["model"]["interaction"]["num_layers"] = 6
        config["images"]["color_jitter"] = result["training_augmentation"]
        config["train"].update(batch_size=batch_size, max_samples=frames * epochs, num_workers=2)
        config = configure(config)
        if baseline_config is not None:
            require(config["model"] == baseline_config["model"] and config["train"] == baseline_config["train"]
                    and config["images"] == baseline_config["images"], "Model/training/augmentation differs from reference run")
            baseline_data = dict(baseline_config["data"], dataset_dir=str(prepared))
            require(config["data"] == baseline_data, "Non-label data inputs differ from reference run")
            baseline_instruction_path = Path(baseline_config["data"]["dataset_dir"]) / "instructions.json"
            require(json.loads(baseline_instruction_path.read_text())["episodes"] == instructions,
                    "Task instructions differ from reference run")
            config["comparison"] = {"reference_run": str(reference_run), "changed": "stored gripper target semantics",
                                    "same_model_training_images_and_instructions": True,
                                    "initialization": "fresh pretrained DINOv3/BERT; not resumed from reference checkpoint"}
        write_json(output / f"config_{mode}.json", config)
        result["modes"][mode] = {"episodes": len(paths), "frames": frames, "valid_chunk_entries": valid,
                                 "instruction_sources": dict(sum((Counter(item["instruction_sources"]) for item in checks), Counter())),
                                 "gripper_target_kind": gripper_labels,
                                 "gripper_target_frame_counts": {key: sum(item[f"gripper_{key}_frames"] for item in checks)
                                                                 for key in ("closed", "hold", "open")},
                                 "source_command_counts": dict(sum((Counter(item["source_command_counts"]) for item in checks), Counter())),
                                 "source_metadata_sha256": {name: sha256(original / name) for name in (
                                     "normalization.json", "manifest.json", "config.yaml", "inference_config.json")},
                                 "episode_checks": checks}
    incomplete.unlink()
    try:
        for mode in ("delta", "relative"):
            config = json.loads((output / f"config_{mode}.json").read_text())
            dataset = RealRobotDataset({**config["data"], "preload_images": False})
            result["modes"][mode]["loader_metadata"] = dataset.metadata()
            result["modes"][mode]["training_data_signature"] = dataset.signature()
            dataset.close()
    except BaseException:
        incomplete.write_text("Prepared dataset failed training-loader validation.\n")
        raise
    result.update(passed=True, delta_relative_shared_data_bitwise_equal=True,
                  all_source_hashes_match_previous_full_preprocessing_validation=True)
    if reference_run is not None:
        result["reference_model_training_augmentation_and_instructions_identical"] = True
    write_json(output / "integrity_report.json", result)
    print(json.dumps({key: value for key, value in result.items() if key != "modes"}, indent=2), flush=True)
    return result


def instructions_match(path, name, entry):
    return json.loads(path.read_text())["episodes"][name] == entry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("/data/jack/go_preprocessed"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--gripper-labels", choices=["latched", "original"], default="latched")
    parser.add_argument("--reference-run", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    prepare(args.source, args.output, args.instruction, args.epochs, args.batch_size, args.gripper_labels, args.reference_run)


if __name__ == "__main__":
    main()
