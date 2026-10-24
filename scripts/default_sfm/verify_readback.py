#!/usr/bin/env python3
"""Read completed destination files back and compare every source SHA-256."""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from inventory import sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/data/crestlux/default_sfm"))
    parser.add_argument("--records", type=Path, default=Path("experiments/default_sfm/setup_records"))
    args = parser.parse_args()
    expected = json.loads((args.records / "source_inventory.json").read_text())
    expected_files = {row["path"]: row for row in expected["files"]}
    result = {
        "status": "waiting_for_copy",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(args.root.resolve()),
        "method": "SHA-256 from reopening each destination file after copying",
        "verified_folders": [],
        "file_count": 0,
        "total_bytes": 0,
        "errors": [],
        "files": [],
    }
    output = args.records / "readback_verification.json"

    def save() -> None:
        result["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
        temporary = output.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(result, indent=2) + "\n")
        temporary.replace(output)

    def check(reference: dict) -> dict:
        path = args.root / reference["path"]
        before = path.stat()
        digest = sha256(path)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
            raise RuntimeError(f"Destination changed during readback: {path}")
        if before.st_size != reference["bytes"] or digest != reference["sha256"]:
            raise RuntimeError(f"Destination SHA-256 mismatch: {path}")
        return {"path": reference["path"], "bytes": before.st_size, "sha256": digest}

    save()
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            while len(result["verified_folders"]) < expected["folder_count"]:
                progressed = False
                for folder in expected["folders"]:
                    name = folder["folder"]
                    if name in result["verified_folders"]:
                        continue
                    source_path = args.records / "file_verification" / f"{name}.source.json"
                    local_path = args.records / "file_verification" / f"{name}.local.json"
                    try:
                        reference = json.loads(source_path.read_text())
                        local = json.loads(local_path.read_text())
                    except (FileNotFoundError, json.JSONDecodeError):
                        continue
                    if local.get("verification", {}).get("status") != "verified":
                        continue
                    wanted = {p for p in expected_files if p.split("/")[0] == name}
                    if {row["path"] for row in reference["files"]} != wanted:
                        raise RuntimeError(f"Source evidence file set differs: {name}")
                    for row in reference["files"]:
                        original = expected_files[row["path"]]
                        if (row["bytes"], row["mtime_ns"]) != (original["bytes"], original["mtime_ns"]):
                            raise RuntimeError(f"Source evidence changed: {row['path']}")
                    rows = list(pool.map(check, reference["files"]))
                    result["files"].extend(rows)
                    result["verified_folders"].append(name)
                    result["file_count"] += len(rows)
                    result["total_bytes"] += sum(row["bytes"] for row in rows)
                    save()
                    print(f"Readback {len(result['verified_folders'])}/45: {name}", flush=True)
                    progressed = True
                transfer = json.loads((args.records / "transfer_status.json").read_text())
                if transfer["status"] == "failed":
                    raise RuntimeError("Copy process failed; readback verification is incomplete")
                if not progressed:
                    time.sleep(10)
        assert result["file_count"] == expected["file_count"]
        assert result["total_bytes"] == expected["total_bytes"]
        result["status"] = "verified"
    except Exception as error:
        result["status"] = "failed"
        result["errors"].append(str(error))
        save()
        raise
    result["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    save()
    print("All 517 destination files passed readback SHA-256 verification", flush=True)


if __name__ == "__main__":
    main()
