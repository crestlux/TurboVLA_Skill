#!/usr/bin/env python3
"""Measure the existing LIBERO loader on CPU without training or clearing caches.

Reports startup, saturated loader throughput, and consumer wait with a fixed
compute delay. The paced measurement is a simulation, not a GPU training trace.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")


def cache_residency(paths):
    """Query Linux mincore without faulting file contents into memory."""
    libc = ctypes.CDLL(None, use_errno=True)
    libc.mmap.restype = ctypes.c_void_p
    libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int,
                         ctypes.c_int, ctypes.c_int, ctypes.c_long]
    libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
    libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    page_size = os.sysconf("SC_PAGE_SIZE")
    result = []
    for path in paths:
        size = total = resident = 0
        files = sorted(Path(path).glob("*tfrecord*"))
        for file in files:
            length = file.stat().st_size
            if not length:
                continue
            fd = os.open(file, os.O_RDONLY)
            address = libc.mmap(None, length, 1, 1, fd, 0)
            if address == ctypes.c_void_p(-1).value:
                os.close(fd)
                raise OSError(ctypes.get_errno(), "mmap failed")
            try:
                pages = (ctypes.c_ubyte * ((length + page_size - 1) // page_size))()
                if libc.mincore(address, length, pages):
                    raise OSError(ctypes.get_errno(), "mincore failed")
                total += len(pages)
                resident += sum(p & 1 for p in pages)
                size += length
            finally:
                libc.munmap(address, length)
                os.close(fd)
        result.append(dict(path=str(path), files=len(files), bytes=size,
                           pages=total, resident_pages=resident,
                           resident_fraction=resident / total if total else None))
    return result


def io_counters(pids):
    result = {}
    for pid in pids:
        try:
            rows = Path(f"/proc/{pid}/io").read_text().splitlines()
            result[str(pid)] = {key: int(value) for key, value in
                                (row.split(":", 1) for row in rows)}
        except (OSError, ValueError):
            pass
    return result


def disk_counters():
    return {row.split()[2]: list(map(int, row.split()[3:]))
            for row in Path("/proc/diskstats").read_text().splitlines()
            if row.split()[2] in {"sda", "sdb", "nvme0n1"}}


def describe(values):
    import numpy as np
    x = np.asarray(values)
    return dict(count=len(values), mean_s=float(x.mean()),
                p50_s=float(np.percentile(x, 50)),
                p95_s=float(np.percentile(x, 95)),
                p99_s=float(np.percentile(x, 99)), max_s=float(x.max()),
                over_100ms=int((x > 0.1).sum()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("/data/jack/libero_datasets"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batches", type=int, default=128)
    parser.add_argument("--warmup-batches", type=int, default=16)
    parser.add_argument("--step-mix-buffer", type=int, default=64)
    parser.add_argument("--compute-seconds", type=float, default=0.8)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; choose a new path")
    if min(args.batch_size, args.workers, args.batches, args.warmup_batches) <= 0:
        parser.error("Batch size, workers, batches, and warmup must be positive")
    if args.compute_seconds < 0:
        parser.error("Compute delay must be nonnegative")

    import torch
    from turbovla.data.mixed_suite import LiberoMixedRLDSDataset, vla_collate_fn

    paths = [str(args.dataset_root / name / "1.0.0") for name in
             ("libero_10_no_noops", "libero_goal_no_noops",
              "libero_object_no_noops", "libero_spatial_no_noops")]
    dataset_kwargs = dict(dataset_dir=paths[0], dataset_dirs=paths,
                          LOCAL_DINOV3_PATH=str(ROOT / "pretrained/backbones/dinov3-vitb16"),
                          stats_path=str(ROOT / "experiments/libero/configs/libero_all4_stats.json"),
                          stats_key="libero_all4_no_noops", chunk_size=12,
                          shuffle_buffer=512, shuffle_steps_within_episode=True,
                          step_mix_buffer_size=args.step_mix_buffer, seed=42,
                          expected_image_size=256, local_files_only=True)
    result = dict(measured_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                  configuration={k: str(v) if isinstance(v, Path) else v
                                 for k, v in vars(args).items()},
                  dataset_kwargs=dataset_kwargs, pin_memory=False,
                  prefetch_factor=2, gpu_training=False,
                  package_versions={p: importlib.metadata.version(p) for p in
                                    ("torch", "transformers", "tensorflow", "tensorflow_datasets")},
                  source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in (ROOT / "turbovla/data/libero_rlds.py",
                                           ROOT / "turbovla/data/mixed_suite.py")},
                  thread_environment={k: os.environ.get(k) for k in
                                      ("OMP_NUM_THREADS", "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS")},
                  cache_before=cache_residency(paths), disk_before=disk_counters(),
                  loadavg_before=list(os.getloadavg()))
    started = time.perf_counter()
    dataset = LiberoMixedRLDSDataset(**dataset_kwargs)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                        num_workers=args.workers, collate_fn=vla_collate_fn,
                                        pin_memory=False, persistent_workers=True,
                                        prefetch_factor=2, timeout=180)
    iterator = iter(loader)
    pids = [os.getpid()] + [w.pid for w in iterator._workers]
    result["process_io_before_first_batch"] = io_counters(pids)
    before = time.perf_counter()
    batch = next(iterator)
    result["first_next_s"] = time.perf_counter() - before
    result["dataset_and_iterator_to_first_batch_s"] = time.perf_counter() - started
    result["batch_shape"] = list(batch[0]["dinov3"].shape)
    del batch
    print(json.dumps({k: result[k] for k in
                      ("first_next_s", "dataset_and_iterator_to_first_batch_s", "batch_shape")}), flush=True)
    try:
        for phase, delay in (("unpaced", 0.0), ("paced", args.compute_seconds)):
            for _ in range(args.warmup_batches):
                batch = next(iterator)
                del batch
                if delay:
                    time.sleep(delay)
            durations = []
            phase_started = time.perf_counter()
            for index in range(args.batches):
                before = time.perf_counter()
                batch = next(iterator)
                durations.append(time.perf_counter() - before)
                del batch
                if delay:
                    time.sleep(delay)
                if (index + 1) % 32 == 0:
                    print(f"{phase}: {index + 1}/{args.batches} batches", flush=True)
            elapsed = time.perf_counter() - phase_started
            result[phase] = dict(wait=describe(durations), waits_s=durations,
                                 elapsed_s=elapsed,
                                 samples_per_s=args.batches * args.batch_size / elapsed,
                                 wait_fraction=sum(durations) / elapsed,
                                 process_io=io_counters(pids), disk=disk_counters())
            print(json.dumps({phase: {k: result[phase][k] for k in
                                      ("wait", "elapsed_s", "samples_per_s", "wait_fraction")}}), flush=True)
        result["cache_after"] = cache_residency(paths)
        result["loadavg_after"] = list(os.getloadavg())
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(f"Saved {args.output}", flush=True)
    finally:
        iterator._shutdown_workers()


if __name__ == "__main__":
    main()
