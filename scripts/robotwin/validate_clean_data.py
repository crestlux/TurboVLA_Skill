#!/usr/bin/env python3
"""Validate RoboTwin schemas, episode alignment, and every decoded video frame.

Can validate complete tasks while the verified downloader fills other tasks.
Uses the system FFmpeg software AV1 decoder; does not allocate a GPU or unpack
videos onto disk. The original dataset files are never rewritten.
"""

from __future__ import annotations

import argparse
import ast
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import pyarrow.parquet as pq

from download_clean_verified import atomic_json, now, verify, REVISION

ROOT = Path(__file__).resolve().parents[2]
CAMERAS = ("cam_high", "cam_left_wrist", "cam_right_wrist")


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def registry_tasks():
    tree = ast.parse((ROOT / "experiments/robotwin/data_registry/data_config.py").read_text())
    return next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "_CLEAN50_TASKS" for t in n.targets))


def decode_video(item):
    path, expected_frames, feature, fps = item
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                            "-show_entries", "stream=codec_name,width,height,nb_frames,avg_frame_rate",
                            "-of", "json", str(path)], capture_output=True, text=True,
                           timeout=60, check=True)
    stream = json.loads(probe.stdout)["streams"][0]
    expected_shape = feature["shape"]
    if (stream["height"], stream["width"]) != tuple(expected_shape[1:]):
        raise ValueError(f"Video shape mismatch: {path}")
    if int(stream["nb_frames"]) != expected_frames:
        raise ValueError(f"Video frame count mismatch: {path}")
    numerator, denominator = map(int, stream["avg_frame_rate"].split("/"))
    if abs(numerator / denominator - fps) > 1e-6:
        raise ValueError(f"Video FPS mismatch: {path}")
    command = ["ffmpeg", "-hide_banner", "-v", "error", "-xerror", "-err_detect", "explode",
               "-threads", "1"]
    if stream["codec_name"] == "av1":
        command += ["-c:v", "libaom-av1"]
    command += ["-i", str(path), "-map", "0:v:0", "-threads", "1", "-filter_threads", "1",
                "-vsync", "0", "-progress", "pipe:1", "-nostats", "-f", "null", "-"]
    decoded = subprocess.run(command, capture_output=True, text=True, timeout=240)
    if decoded.returncode or decoded.stderr.strip():
        raise ValueError(f"Video decode failed: {path}: {decoded.stderr[-1200:]}")
    counts = [int(line.split("=", 1)[1]) for line in decoded.stdout.splitlines() if line.startswith("frame=")]
    if not counts or counts[-1] != expected_frames:
        raise ValueError(f"Decoded frame count mismatch: {path}: {counts[-1:]}")
    return dict(frames=counts[-1], codec=stream["codec_name"],
                width=stream["width"], height=stream["height"])


def check_task(root, task, entries, workers):
    for entry in entries:
        reason = verify(root, entry)
        if reason != "ok":
            raise ValueError(f"{entry['rfilename']}: {reason}")
    directory = root / task
    info = json.loads((directory / "meta/info.json").read_text())
    if info["codebase_version"] != "v2.1" or info["total_episodes"] != 50:
        raise ValueError(f"Unexpected dataset version or episode count: {task}")
    if info["total_videos"] != 150 or info["fps"] != 15:
        raise ValueError(f"Unexpected video metadata: {task}")
    expected_modality = json.loads((ROOT / "experiments/robotwin/configs/modality.json").read_text())
    if json.loads((directory / "meta/modality.json").read_text()) != expected_modality:
        raise ValueError(f"Modality differs from TurboVLA config: {task}")
    episodes = read_jsonl(directory / "meta/episodes.jsonl")
    descriptions = read_jsonl(directory / "meta/tasks.jsonl")
    if sorted(e["episode_index"] for e in episodes) != list(range(50)):
        raise ValueError(f"Incomplete episode metadata: {task}")
    task_text = {t["task_index"]: t["task"] for t in descriptions}
    if len(task_text) != info["total_tasks"] or any(not text.strip() for text in task_text.values()):
        raise ValueError(f"Invalid instruction metadata: {task}")
    for key in ("observation.state", "action"):
        if info["features"][key]["shape"] != [14]:
            raise ValueError(f"Wrong feature dimensions: {task}/{key}")
    videos = []
    frames = 0
    for episode in sorted(episodes, key=lambda e: e["episode_index"]):
        index = episode["episode_index"]
        fields = dict(episode_index=index, episode_chunk=index // info["chunks_size"])
        path = directory / info["data_path"].format(**fields)
        table = pq.read_table(path)
        length = episode["length"]
        if table.num_rows != length or length < 1:
            raise ValueError(f"Episode length mismatch: {path}")
        for key in ("observation.state", "action"):
            array = np.asarray(table[key].to_pylist(), dtype=np.float32)
            if array.shape != (length, 14) or not np.isfinite(array).all():
                raise ValueError(f"Invalid numeric features: {path}/{key}")
        if not np.all(table["episode_index"].to_numpy() == index):
            raise ValueError(f"Episode indices inconsistent: {path}")
        if not np.array_equal(table["frame_index"].to_numpy(), np.arange(length)):
            raise ValueError(f"Frame indices inconsistent: {path}")
        timestamps = table["timestamp"].to_numpy()
        if not np.allclose(timestamps, np.arange(length) / info["fps"], atol=1e-4, rtol=1e-5):
            raise ValueError(f"Timestamps inconsistent: {path}")
        labels = set(table["task_index"].to_pylist())
        if not labels.issubset(task_text):
            raise ValueError(f"Unknown instruction index: {path}")
        if not {task_text[label] for label in labels}.issubset(set(episode["tasks"])):
            raise ValueError(f"Instruction/episode mismatch: {path}")
        frames += length
        for camera in CAMERAS:
            key = f"observation.images.{camera}"
            videos.append((directory / info["video_path"].format(**fields, video_key=key),
                           length, info["features"][key], info["fps"]))
    if frames != info["total_frames"]:
        raise ValueError(f"Total frames inconsistent: {task}")
    results = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(decode_video, item) for item in videos]):
            results.append(future.result())
    return dict(task=task, episodes=len(episodes), transitions=frames,
                verified_files=len(entries), decoded_videos=len(results),
                decoded_frames=sum(r["frames"] for r in results),
                video_formats=dict(Counter(f"{r['codec']} {r['width']}x{r['height']}" for r in results)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--local-dir", type=Path, default=Path("/home/jack/datasets/RoboTwin/Clean"))
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--wait-for-download-status", type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if manifest["sha"] != REVISION:
        raise ValueError("Unexpected manifest revision")
    tasks = registry_tasks()
    grouped = {task: [e for e in manifest["siblings"] if e["rfilename"].startswith(task + "/")]
               for task in tasks}
    manifest_tasks = {e["rfilename"].split("/", 1)[0] for e in manifest["siblings"] if "/" in e["rfilename"]}
    if len(tasks) != 50 or set(tasks) != manifest_tasks:
        raise ValueError("Task manifest differs from TurboVLA registry")
    state = dict(started_at=now(), updated_at=now(), dataset_root=str(args.local_dir),
                 revision=REVISION, state="validating", completed=False,
                 tasks=[], errors=[], decoder="system FFmpeg libaom-av1 (CPU, full frames)",
                 validation_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    if args.resume and args.report.exists():
        previous = json.loads(args.report.read_text())
        if previous["revision"] != REVISION or previous["dataset_root"] != str(args.local_dir):
            raise ValueError("Previous validation report belongs to different data")
        for result in previous["tasks"]:
            if all(verify(args.local_dir, entry) == "ok" for entry in grouped[result["task"]]):
                state["tasks"].append(result)
        state["resumed_tasks"] = len(state["tasks"])
        state["previous_validation_script_sha256"] = previous["validation_script_sha256"]
    atomic_json(args.report, state)
    try:
        finished = {task["task"] for task in state["tasks"]}
        unfinished = [task for task in tasks if task not in finished]
        while unfinished:
            ready = [task for task in unfinished if all(
                (args.local_dir / e["rfilename"]).is_file()
                and (args.local_dir / e["rfilename"]).stat().st_size == e["size"]
                for e in grouped[task])]
            for task in ready:
                print(f"{now()} Checking {task}", flush=True)
                result = check_task(args.local_dir, task, grouped[task], args.workers)
                state["tasks"].append(result)
                unfinished.remove(task)
                state["updated_at"] = now()
                atomic_json(args.report, state)
                print(f"{now()} PASS {task}: {result['episodes']} episodes, {result['decoded_videos']} videos, {result['decoded_frames']} decoded frames; tasks={len(state['tasks'])}/50", flush=True)
            if unfinished:
                if not args.wait_for_download_status:
                    raise ValueError(f"Incomplete tasks: {unfinished}")
                download = json.loads(args.wait_for_download_status.read_text())
                if download["state"] == "incomplete" and download.get("errors"):
                    raise RuntimeError("Downloader stopped; inspect its status report")
                if download.get("completed") and not ready:
                    raise RuntimeError(f"Download reports complete but tasks are missing: {unfinished}")
                time.sleep(15)
        state["summary"] = {key: sum(task[key] for task in state["tasks"]) for key in
                            ("episodes", "transitions", "verified_files", "decoded_videos", "decoded_frames")}
        state.update(state="complete", completed=True)
    except Exception as error:
        state["errors"].append(dict(type=type(error).__name__, message=str(error)))
        state["state"] = "failed"
    finally:
        state["updated_at"] = now()
        atomic_json(args.report, state)
    print(json.dumps({k: v for k, v in state.items() if k != "tasks"}, indent=2), flush=True)
    return 0 if state["completed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
