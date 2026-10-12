#!/usr/bin/env python3
"""Resume a pinned RoboTwin Clean download with retries and content verification.

Requires the Hub's dataset-info JSON obtained with files_metadata/blobs=true.
Progress counts only files whose size AND content hash match that manifest.
The installed training environment and its libraries are not modified.
"""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import threading
import time
from urllib.parse import urlparse

os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "45")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "30")

REPO = "StarVLA/RoboTwin-Clean"
REVISION = "070d3b86d7db06f924702a82725dba0f4d89a433"


def now():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def verify(root, entry):
    path = root / entry["rfilename"]
    if not path.is_file():
        return "missing"
    size = path.stat().st_size
    if size != entry["size"]:
        return "size_mismatch"
    expected = entry.get("lfs", {}).get("sha256")
    if expected:
        digest = hashlib.sha256()
    else:
        expected = entry["blobId"]
        digest = hashlib.sha1(f"blob {size}\0".encode())
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return "ok" if digest.hexdigest() == expected else "hash_mismatch"


def retry_delay(headers):
    delays = []
    value = headers.get("Retry-After", "")
    if value:
        try:
            delays.append(float(value))
        except ValueError:
            try:
                delays.append((parsedate_to_datetime(value) - datetime.now(timezone.utc)).total_seconds())
            except (TypeError, ValueError):
                pass
    delays.extend(float(x) for x in re.findall(r"(?:^|;)\s*t=(\d+)", headers.get("RateLimit", "")))
    return max([3.0] + [x + 3 for x in delays]) if delays else 305.0


class RequestLimiter:
    def __init__(self, requests_per_second):
        self.interval = 1 / requests_per_second
        self.next_request = 0.0
        self.blocked_until = 0.0
        self.lock = threading.Lock()

    def acquire(self):
        while True:
            with self.lock:
                current = time.monotonic()
                delay = max(self.next_request, self.blocked_until) - current
                if delay <= 0:
                    self.next_request = current + self.interval
                    return
            time.sleep(min(delay, 1.0))

    def pause(self, seconds):
        with self.lock:
            self.blocked_until = max(self.blocked_until, time.monotonic() + seconds)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--local-dir", type=Path, default=Path("/home/jack/datasets/RoboTwin/Clean"))
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--requests-per-second", type=float, default=8.0)
    parser.add_argument("--attempts", type=int, default=12)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.workers < 1 or args.requests_per_second <= 0 or args.attempts < 1:
        parser.error("Workers, request rate, and attempts must be positive")
    manifest = json.loads(args.manifest.read_text())
    if manifest.get("id") != REPO or manifest.get("sha") != REVISION:
        parser.error("Manifest does not match the pinned dataset revision")
    entries = manifest["siblings"]
    if len(entries) != 10404 or sum(e["size"] for e in entries) != 4021934537:
        parser.error("Unexpected manifest file count or total size")
    for entry in entries:
        path = PurePosixPath(entry["rfilename"])
        if path.is_absolute() or ".." in path.parts:
            parser.error("Unsafe filename in manifest")

    root = args.local_dir.resolve()
    state = dict(started_at=now(), updated_at=now(), repo_id=REPO, revision=REVISION,
                 dataset_root=str(root), manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
                 expected_files=len(entries), expected_bytes=sum(e["size"] for e in entries),
                 verified_files=0, verified_bytes=0, state="verifying_existing",
                 workers=args.workers, requests_per_second=args.requests_per_second,
                 completed=False, retries=0, errors=[])
    atomic_json(args.status, state)
    pending = []
    for entry in entries:
        result = verify(root, entry)
        if result == "ok":
            state["verified_files"] += 1
            state["verified_bytes"] += entry["size"]
        else:
            pending.append((entry, result))
    state["existing_verified_files"] = state["verified_files"]
    state["remaining_files"] = len(pending)
    print(f"{now()} Verified existing: {state['verified_files']}/{len(entries)}; remaining: {len(pending)}", flush=True)

    if args.verify_only:
        state.update(state="complete" if not pending else "incomplete", completed=not pending,
                     updated_at=now(), errors=[dict(path=e["rfilename"], reason=reason) for e, reason in pending])
        atomic_json(args.status, state)
        return 0 if not pending else 1

    import requests
    from huggingface_hub import configure_http_backend, hf_hub_download
    from huggingface_hub.utils import logging as hf_logging
    hf_logging.set_verbosity_error()
    limiter = RequestLimiter(args.requests_per_second)
    stop = threading.Event()
    state_lock = threading.Lock()

    class LimitedSession(requests.Session):
        def send(self, request, **kwargs):
            if urlparse(request.url).hostname == "huggingface.co":
                limiter.acquire()
            response = super().send(request, **kwargs)
            if response.status_code == 429:
                limiter.pause(retry_delay(response.headers))
            return response

    configure_http_backend(backend_factory=LimitedSession)

    def fetch(item):
        entry, previous = item
        force = previous in {"size_mismatch", "hash_mismatch"}
        for attempt in range(1, args.attempts + 1):
            if stop.is_set():
                raise RuntimeError("Stopped after another file failed")
            try:
                hf_hub_download(repo_id=REPO, repo_type="dataset", revision=REVISION,
                                filename=entry["rfilename"], local_dir=root, force_download=force,
                                endpoint="https://huggingface.co")
                result = verify(root, entry)
                if result != "ok":
                    force = True
                    raise OSError(f"Verification failed: {result}")
                return entry
            except (requests.RequestException, OSError) as error:
                response = getattr(error, "response", None)
                code = response.status_code if response is not None else None
                retryable = code is None or code in {408, 425, 429, 500, 502, 503, 504}
                if not retryable or attempt == args.attempts:
                    raise RuntimeError(f"{entry['rfilename']}: {type(error).__name__}, status={code}, attempt={attempt}") from None
                delay = retry_delay(response.headers) if code == 429 else min(60, 2 ** attempt)
                if code == 429:
                    limiter.pause(delay)
                with state_lock:
                    state["retries"] += 1
                print(f"{now()} Retry {entry['rfilename']}: {type(error).__name__}, status={code}, wait={delay:.0f}s, attempt={attempt}", flush=True)
                if stop.wait(delay):
                    raise RuntimeError("Stopped after another file failed")

    state["state"] = "downloading"
    atomic_json(args.status, state)
    last_report = time.monotonic()
    try:
        remaining = iter(pending)
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            active = {}
            def refill():
                while len(active) < args.workers * 2 and not stop.is_set():
                    try:
                        item = next(remaining)
                    except StopIteration:
                        break
                    active[executor.submit(fetch, item)] = item[0]["rfilename"]
            refill()
            while active:
                done, _ = wait(active, timeout=1, return_when=FIRST_COMPLETED)
                for future in done:
                    name = active.pop(future)
                    try:
                        entry = future.result()
                    except Exception as error:
                        state["errors"].append(dict(path=name, reason=str(error)))
                        stop.set()
                        continue
                    state["verified_files"] += 1
                    state["verified_bytes"] += entry["size"]
                refill()
                if time.monotonic() - last_report >= 15:
                    state.update(updated_at=now(), remaining_files=len(entries) - state["verified_files"])
                    atomic_json(args.status, state)
                    print(f"{now()} Verified {state['verified_files']}/{len(entries)} files; {state['verified_bytes'] / 1e9:.3f}/4.022 GB; retries={state['retries']}", flush=True)
                    last_report = time.monotonic()
    except KeyboardInterrupt:
        stop.set()
        state["errors"].append(dict(reason="Interrupted"))
    finally:
        complete = state["verified_files"] == len(entries) and not state["errors"]
        state.update(updated_at=now(), remaining_files=len(entries) - state["verified_files"],
                     state="complete" if complete else "incomplete", completed=complete)
        atomic_json(args.status, state)
    print(json.dumps(state, indent=2), flush=True)
    return 0 if state["completed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
