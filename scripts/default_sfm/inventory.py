#!/usr/bin/env python3
"""Read-only inventory of the 9 x 5 x 10 default_sfm cohort (stdlib only)."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

TASKS = ("audio_jack", "barlatch", "bnc", "hdmi", "knob", "lan", "outlet", "rotary_switch", "usb")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inventory(root: Path, hashes: bool, workers: int, folder_name: str | None = None) -> dict:
    expected = {f"{task}_p{position}" for task in TASKS for position in range(1, 6)}
    if folder_name is not None:
        if folder_name not in expected:
            raise ValueError(f"Invalid cohort folder: {folder_name}")
        expected = {folder_name}
        found = {folder_name} if (root / folder_name).is_dir() else set()
    else:
        found = {p.name for p in root.iterdir() if p.is_dir() and re.fullmatch(r".+_p[1-5]", p.name)}
    if found != expected:
        raise ValueError(f"Unexpected cohort: missing={sorted(expected - found)}, extra={sorted(found - expected)}")
    folders, paths = [], []
    for name in sorted(expected):
        folder = root / name
        if folder.is_symlink():
            raise ValueError(f"Source task folder must be a real directory: {folder}")
        entries = sorted(folder.rglob("*"))
        if any(p.is_symlink() for p in entries):
            raise ValueError(f"Unexpected symlink inside {folder}")
        files = [p for p in entries if p.is_file()]
        demos = [p for p in files if p.suffix == ".pkl"]
        if len(demos) != 10:
            raise ValueError(f"{name}: expected 10 demos, got {len(demos)}")
        task, position = name.rsplit("_p", 1)
        folders.append(
            {
                "folder": name,
                "task": task,
                "position": int(position),
                "demo_count": len(demos),
                "file_count": len(files),
                "total_bytes": sum(p.stat().st_size for p in files),
            }
        )
        paths.extend(files)

    def record(path: Path) -> dict:
        before = path.stat()
        row = {"path": path.relative_to(root).as_posix(), "bytes": before.st_size, "mtime_ns": before.st_mtime_ns}
        if hashes:
            row["sha256"] = sha256(path)
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise RuntimeError(f"Source changed while hashing: {path}")
        return row

    records = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, row in enumerate(pool.map(record, paths), 1):
            records.append(row)
            if hashes and (i % 10 == 0 or i == len(paths)):
                print(f"Hashed {i}/{len(paths)} files", file=sys.stderr, flush=True)
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "selection": "Nine named tasks, p1 through p5; test and usb_insert excluded",
        "task_count": len({row["task"] for row in folders}),
        "folder_count": len(folders),
        "demo_count": sum(row["demo_count"] for row in folders),
        "file_count": len(records),
        "total_bytes": sum(row["bytes"] for row in records),
        "suffix_counts": dict(Counter(Path(row["path"]).suffix for row in records)),
        "excluded_top_level_entries": sorted(p.name for p in root.iterdir() if p.name not in expected),
        "hash_algorithm": "sha256" if hashes else None,
        "folders": folders,
        "files": records,
    }


def compare(actual: dict, reference: dict) -> dict:
    a = {x["path"]: x for x in actual["files"]}
    b = {x["path"]: x for x in reference["files"]}
    errors = [f"missing:{p}" for p in sorted(b.keys() - a.keys())]
    errors += [f"unexpected:{p}" for p in sorted(a.keys() - b.keys())]
    hashes = reference["hash_algorithm"] == "sha256"
    for path in sorted(a.keys() & b.keys()):
        for field in ["bytes", "sha256"] if hashes else ["bytes"]:
            if a[path].get(field) != b[path].get(field):
                errors.append(f"{field}:{path}")
    return {
        "status": "verified" if not errors else "failed",
        "errors": errors,
        "comparison": "exact relative file set, sizes, SHA-256" if hashes else "exact relative file set and sizes",
        "reference_root": reference["root"],
        "reference_created_at_utc": reference["created_at_utc"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--sha256", action="store_true")
    parser.add_argument("--folder", help="Inspect one of the 45 cohort folders only")
    parser.add_argument("--hash-file", help="Hash a single inventoried relative path without reading other demos")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--compare", type=Path)
    parser.add_argument("--output", default="-")
    args = parser.parse_args()
    if args.hash_file:
        root = args.root.resolve()
        path = root / args.hash_file
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError(f"Invalid source path: {path}")
        before = path.stat()
        digest = sha256(path)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError(f"Source changed while hashing: {path}")
        print(
            json.dumps(
                {"path": args.hash_file, "bytes": before.st_size, "mtime_ns": before.st_mtime_ns, "sha256": digest}
            )
        )
        return
    report = inventory(args.root.resolve(), args.sha256, args.workers, args.folder)
    if args.compare:
        reference = json.loads(args.compare.read_text())
        report["verification"] = compare(report, reference)
    payload = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output == "-":
        print(payload, end="")
    else:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(payload)
        temporary.replace(output)
        print(
            json.dumps(
                {k: report[k] for k in ("task_count", "folder_count", "demo_count", "file_count", "total_bytes")}
            )
        )
    if args.compare and report["verification"]["status"] != "verified":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
