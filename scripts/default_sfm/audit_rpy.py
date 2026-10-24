"""Read-only RPY boundary audit of all default_sfm demos (no training conversion)."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import pickle
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


IMAGE_BYTES = 480 * 640 * 3
CAMERAS = ("head_cam_d455", "wrist_cam_d405_a")
TWO_PI = 2 * np.pi


class NumericAuditReader:
    """Skip only 921600-byte raw camera payloads; never use its images for display.

    The numeric arrays in this cohort are much smaller (state: 76 bytes;
    state history: 3800 bytes). Pickle metadata and every numeric byte are read
    unchanged. A full conventional load is compared before this optimization
    is enabled across the cohort.
    """

    def __init__(self, stream):
        self.stream = stream
        self.zeros = bytes(IMAGE_BYTES)
        self.skipped_image_buffers = 0

    def read(self, n=-1):
        if n == IMAGE_BYTES:
            self.stream.seek(n, 1)
            self.skipped_image_buffers += 1
            return self.zeros
        return self.stream.read(n)

    def readinto(self, buffer):
        if len(buffer) == IMAGE_BYTES:
            self.stream.seek(len(buffer), 1)
            buffer[:] = self.zeros
            self.skipped_image_buffers += 1
            return len(buffer)
        return self.stream.readinto(buffer)

    def readline(self, *args):
        return self.stream.readline(*args)


def fingerprint(frames):
    digest = hashlib.sha256()
    for frame in frames:
        for group in ("observations", "next_observations"):
            for key in ("state", "right_state_history", "right_force_history"):
                array = np.asarray(frame[group][key])
                assert array.nbytes < IMAGE_BYTES
                digest.update(str((array.shape, array.dtype)).encode())
                digest.update(array.tobytes())
        digest.update(np.asarray(frame["actions"]).tobytes())
        digest.update(str(frame["time"]).encode())
        digest.update(str(frame["infos"].get("action_format")).encode())
    return digest.hexdigest()


def validate_reader(path):
    with path.open("rb") as stream:
        frames = pickle.load(stream)
    conventional = fingerprint(frames)
    n = len(frames)
    del frames
    gc.collect()
    with path.open("rb") as stream:
        reader = NumericAuditReader(stream)
        frames = pickle.load(reader)
    numeric = fingerprint(frames)
    assert conventional == numeric and len(frames) == n
    assert reader.skipped_image_buffers > 0
    return {"status": "passed", "file": str(path), "frame_count": n,
            "full_load_numeric_sha256": conventional, "skipping_reader_numeric_sha256": numeric,
            "skipped_image_buffers": reader.skipped_image_buffers,
            "compared_fields": "all current/next state, state_history, force_history, actions, timestamps and action_format"}


def angle_statistics(value):
    value = np.asarray(value, dtype=np.float64)
    assert value.shape[-1] == 3 and np.isfinite(value).all()
    mapped = np.remainder(value, TWO_PI)
    selective = value.copy()
    selective[..., [0, 2]] = mapped[..., [0, 2]]
    output = {"range_min_rad": value.reshape(-1, 3).min(axis=0).tolist(),
              "range_max_rad": value.reshape(-1, 3).max(axis=0).tolist()}
    for name, angles in (("raw", value), ("all_axes_0_2pi", mapped), ("roll_yaw_0_2pi_pitch_signed", selective)):
        # History windows use axis -2: never compare the end of one window
        # against the beginning of another overlapping window.
        difference = np.diff(angles, axis=-2)
        flat = difference.reshape(-1, 3)
        mask = np.abs(flat) > np.pi
        output[name] = {"jumps_gt_pi": mask.sum(axis=0).tolist(),
                        "max_abs_step_rad": np.abs(flat).max(axis=0).tolist() if flat.size else [0., 0., 0.],
                        "examples": []}
        if value.ndim == 2:
            for axis in range(3):
                indices = np.flatnonzero(mask[:, axis])[:2]
                for i in indices:
                    output[name]["examples"].append({"axis": ["roll", "pitch", "yaw"][axis],
                        "frame_before": int(i), "frame_after": int(i + 1),
                        "before_rad": float(angles[i, axis]), "after_rad": float(angles[i + 1, axis])})
    return output


def audit_one(args):
    root, relative, output = args
    path = Path(root) / relative
    before = path.stat()
    started = time.monotonic()
    with path.open("rb") as stream:
        reader = NumericAuditReader(stream)
        frames = pickle.load(reader)
    for frame in frames:
        for group in ("observations", "next_observations"):
            observation = frame[group]
            assert np.asarray(observation["state"]).shape == (1, 19)
            assert np.asarray(observation["right_state_history"]).shape == (1, 50, 19)
            for camera in CAMERAS:
                image = np.asarray(observation[camera])
                assert image.shape == (1, 480, 640, 3) and image.dtype == np.uint8
    state = np.stack([np.asarray(f["observations"]["state"])[0] for f in frames])
    next_state = np.stack([np.asarray(f["next_observations"]["state"])[0] for f in frames])
    action = np.stack([np.asarray(f["actions"]) for f in frames])
    history = np.stack([np.asarray(f["observations"]["right_state_history"])[0, :, 7:10] for f in frames])
    times = np.asarray([f["time"] for f in frames])
    formats = dict(Counter(f["infos"].get("action_format", "missing") for f in frames))
    assert set(formats) == {"absolute_tcp_xyz_rpy_gripper"}, formats
    after = path.stat()
    assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
    key = hashlib.sha256(relative.encode()).hexdigest()[:16]
    np.savez_compressed(Path(output) / "numeric" / f"{key}.npz", state=state, action=action, time=times)
    return {"path": relative, "frame_count": len(frames), "source_bytes": before.st_size,
        "source_mtime_ns": before.st_mtime_ns, "skipped_image_buffers": reader.skipped_image_buffers,
        "numeric_artifact": f"numeric/{key}.npz", "action_formats": formats,
        "state_rpy": angle_statistics(state[:, 7:10]),
        "next_state_rpy": angle_statistics(next_state[:, 7:10]),
        "action_rpy": angle_statistics(action[:, 3:6]),
        "state_history_rpy_within_windows": angle_statistics(history),
        "seconds": round(time.monotonic() - started, 3)}


def aggregate(rows):
    result = {}
    for field in ("state_rpy", "next_state_rpy", "action_rpy", "state_history_rpy_within_windows"):
        summary = {"range_min_rad": np.min([r[field]["range_min_rad"] for r in rows], axis=0).tolist(),
                   "range_max_rad": np.max([r[field]["range_max_rad"] for r in rows], axis=0).tolist()}
        for mode in ("raw", "all_axes_0_2pi", "roll_yaw_0_2pi_pitch_signed"):
            counts = np.asarray([r[field][mode]["jumps_gt_pi"] for r in rows])
            summary[mode] = {"jumps_gt_pi": counts.sum(axis=0).tolist(),
                "demos_with_jumps": (counts > 0).sum(axis=0).tolist(),
                "max_abs_step_rad": np.max([r[field][mode]["max_abs_step_rad"] for r in rows], axis=0).tolist()}
        result[field] = summary
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/data/crestlux/default_sfm"))
    parser.add_argument("--inventory", type=Path, default=Path("experiments/default_sfm/setup_records/source_inventory.json"))
    parser.add_argument("--output", type=Path, default=Path("experiments/default_sfm/preprocessing_audit/rpy"))
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    (args.output / "numeric").mkdir(parents=True, exist_ok=True)
    inventory = json.loads(args.inventory.read_text())
    paths = [row["path"] for row in inventory["files"] if row["path"].endswith(".pkl")]
    validation = validate_reader(args.root / paths[0])
    print("Numeric-only reader matches conventional full load", flush=True)
    report = {"status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "expected_demos": len(paths), "axis_order": ["roll", "pitch", "yaw"],
        "jump_definition": "absolute change greater than pi between adjacent recorded samples, within each demo",
        "history_count_note": "Within-window transitions; overlapping windows are counted separately",
        "reader_validation": validation, "raw_data_modified": False, "training_preprocessing_applied": False,
        "demos": []}

    def save():
        report["completed_demos"] = len(report["demos"])
        report["total_frames"] = sum(r["frame_count"] for r in report["demos"])
        report["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
        temporary = args.output / "report.json.tmp"
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(args.output / "report.json")

    save()
    jobs = [(str(args.root), p, str(args.output)) for p in paths]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for row in pool.map(audit_one, jobs):
            report["demos"].append(row)
            if len(report["demos"]) % 10 == 0:
                save()
                print(f"RPY audit {len(report['demos'])}/{len(paths)}", flush=True)
    report["summary"] = aggregate(report["demos"])
    report["status"] = "verified"
    report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    save()
    print(json.dumps({"demos": len(paths), "frames": report["total_frames"], "state_rpy": report["summary"]["state_rpy"]}), flush=True)


if __name__ == "__main__":
    main()
