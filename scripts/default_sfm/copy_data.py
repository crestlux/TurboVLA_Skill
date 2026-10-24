#!/usr/bin/env python3
"""Resume source-preserving copies of the inventoried 450-demo cohort."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from inventory import compare, sha256
from inventory import inventory as inspect_inventory
from stream_receive import receive


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--destination", type=Path, default=Path("/data/crestlux/default_sfm"))
    parser.add_argument("--host", default="csilab")
    parser.add_argument("--zstd", default="zstd")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--compress-level", type=int, default=6)
    parser.add_argument("--records", type=Path, default=Path("experiments/default_sfm/setup_records"))
    args = parser.parse_args()
    inventory = json.loads(args.inventory.read_text())
    assert inventory["demo_count"] == 450 and inventory["folder_count"] == 45
    args.destination.mkdir(parents=True, exist_ok=True)
    args.records.mkdir(parents=True, exist_ok=True)
    logs = args.records / "transfer_logs"
    logs.mkdir(exist_ok=True)
    verification_dir = args.records / "file_verification"
    verification_dir.mkdir(exist_ok=True)
    lock = threading.Lock()
    state = {
        "status": "copying",
        "source_host": args.host,
        "source_root": inventory["root"],
        "destination": str(args.destination.resolve()),
        "total_bytes": inventory["total_bytes"],
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "folders": {},
    }

    def update(name: str, status: str, **kwargs) -> None:
        with lock:
            state["folders"][name] = {"status": status, **kwargs}
            state["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
            path = args.records / "transfer_status.json"
            temporary = path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(state, indent=2) + "\n")
            temporary.replace(path)
            if status in {"verified", "failed"}:
                print(f"{name}: {status}", flush=True)

    def copy(folder: dict) -> int:
        name = folder["folder"]
        if Path(name).name != name or name.startswith("."):
            raise ValueError(f"Invalid task folder: {name}")
        original_files = [x for x in inventory["files"] if x["path"].split("/")[0] == name]
        source_evidence = verification_dir / f"{name}.source.json"
        local_evidence = verification_dir / f"{name}.local.json"
        if source_evidence.exists() and local_evidence.exists():
            previous_source = json.loads(source_evidence.read_text())
            previous_local = json.loads(local_evidence.read_text())
            current = inspect_inventory(args.destination.resolve(), False, 1, name)

            def fields(rows):
                return {row["path"]: (row["bytes"], row["mtime_ns"]) for row in rows}

            if (
                previous_local.get("verification", {}).get("status") == "verified"
                and fields(previous_source["files"]) == fields(original_files)
                and fields(previous_local["files"]) == fields(current["files"])
            ):
                update(
                    name,
                    "verified",
                    total_bytes=folder["total_bytes"],
                    verification=previous_local["verification"],
                    reused_verified_folder=True,
                )
                return 0
        reference_files, local_files = [], []
        verified_bytes = 0
        for original in original_files:
            relative = original["path"]
            destination = args.destination / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            row = None
            local_digest = None
            if destination.is_file() and destination.stat().st_size == original["bytes"]:
                update(
                    name,
                    "verifying_existing",
                    current_file=relative,
                    verified_files=len(local_files),
                    verified_bytes=verified_bytes,
                    total_bytes=folder["total_bytes"],
                )
                remote_command = shlex.join(["python3", "-", "--root", inventory["root"], "--hash-file", relative])
                with (logs / f"{name}.hash.log").open("a") as stream:
                    source = subprocess.run(
                        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=30", args.host, remote_command],
                        input=Path(__file__).with_name("inventory.py").read_bytes(),
                        stdout=subprocess.PIPE,
                        stderr=stream,
                        check=True,
                    )
                row = json.loads(source.stdout)
                if (row["bytes"], row["mtime_ns"]) != (original["bytes"], original["mtime_ns"]):
                    raise RuntimeError(f"Source changed since inventory: {relative}")
                local_digest = sha256(destination)
            if row is None or local_digest != row["sha256"]:
                update(
                    name,
                    "copying",
                    current_file=relative,
                    verified_files=len(local_files),
                    verified_bytes=verified_bytes,
                    total_bytes=folder["total_bytes"],
                )
                row, local_digest = receive(
                    args.host, inventory["root"], original, destination, args.zstd, args.compress_level
                )
            if destination.stat().st_size != row["bytes"] or local_digest != row["sha256"]:
                raise RuntimeError(f"SHA-256 mismatch: {relative}")
            reference_files.append(row)
            local_files.append(
                {
                    "path": relative,
                    "bytes": destination.stat().st_size,
                    "mtime_ns": destination.stat().st_mtime_ns,
                    "sha256": local_digest,
                }
            )
            verified_bytes += row["bytes"]
            # Clean up only this verified file's partial copy from our earlier
            # rsync attempt. Original data and completed files are never deleted.
            partial = destination.parent / ".rsync-partial" / destination.name
            if partial.is_file() and not partial.is_symlink():
                partial.unlink()
            if partial.parent.is_dir() and not any(partial.parent.iterdir()):
                partial.parent.rmdir()
            for abandoned in destination.parent.glob(f".{destination.name}.*.sfm-partial"):
                if abandoned.is_file() and not abandoned.is_symlink():
                    abandoned.unlink()
        reference = {
            "root": inventory["root"],
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "hash_algorithm": "sha256",
            "files": reference_files,
        }
        (verification_dir / f"{name}.source.json").write_text(json.dumps(reference, indent=2) + "\n")
        actual = inspect_inventory(args.destination.resolve(), False, 1, name)
        if {x["path"] for x in actual["files"]} != {x["path"] for x in local_files}:
            raise RuntimeError(f"Unexpected local files in {name}")
        actual["files"] = local_files
        actual["hash_algorithm"] = "sha256"
        actual["verification"] = compare(actual, reference)
        (verification_dir / f"{name}.local.json").write_text(json.dumps(actual, indent=2) + "\n")
        ok = actual["verification"]["status"] == "verified"
        update(
            name, "verified" if ok else "failed", total_bytes=folder["total_bytes"], verification=actual["verification"]
        )
        return 0 if ok else 1

    def copy_or_report(folder: dict) -> int:
        try:
            return copy(folder)
        except Exception as error:
            update(folder["folder"], "failed", reason=str(error))
            return 1

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        codes = list(pool.map(copy_or_report, inventory["folders"]))
    state["status"] = "verified" if all(code == 0 for code in codes) else "failed"
    if state["status"] == "verified":
        final = inspect_inventory(args.destination.resolve(), False, 1)
        hashes = {}
        for folder in inventory["folders"]:
            evidence = json.loads((verification_dir / f"{folder['folder']}.source.json").read_text())
            hashes.update({row["path"]: row["sha256"] for row in evidence["files"]})
        for row in final["files"]:
            row["sha256"] = hashes[row["path"]]
        final["hash_algorithm"] = "sha256"
        final["verification"] = {
            "status": "verified",
            "method": "per-folder source/local SHA-256 comparison",
            "source_host": args.host,
            "source_root": inventory["root"],
        }
        (args.records / "copy_manifest.json").write_text(json.dumps(final, indent=2) + "\n")
    state["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    (args.records / "transfer_status.json").write_text(json.dumps(state, indent=2) + "\n")
    print(state["status"], flush=True)
    raise SystemExit(int(any(codes)))


if __name__ == "__main__":
    main()
