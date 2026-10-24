#!/usr/bin/env python3
"""Print environment and raw-data preparation status without starting training."""

from __future__ import annotations

import json
from pathlib import Path

root = Path(__file__).resolve().parents[2]
records = root / "experiments/default_sfm/setup_records"


def load(name: str) -> dict:
    path = records / name
    return json.loads(path.read_text()) if path.exists() else {}


transfer = load("transfer_status.json")
folders = transfer.get("folders", {})
verified_bytes = sum(
    row.get("total_bytes", 0) if row["status"] == "verified" else row.get("verified_bytes", 0)
    for row in folders.values()
)
verified_folder_count = sum(row["status"] == "verified" for row in folders.values())
dataset = root / "data/default_sfm"
demos = list(dataset.glob("*_p[1-5]/*/*.pkl"))
total = transfer.get("total_bytes", 0)
print(
    json.dumps(
        {
            "environment": load("environment_validation.json").get("status", "pending"),
            "three_gpu_nccl": load("nccl_validation.json").get("status", "pending"),
            "wandb_access": load("wandb_access.json").get("status", "pending"),
            "transfer": transfer.get("status", "pending"),
            "disk_readback": load("readback_verification.json").get("status", "pending"),
            "readback_file_count": load("readback_verification.json").get("file_count", 0),
            "complete_demo_files": len(demos),
            "expected_demo_files": 450,
            "complete_demo_bytes": sum(path.stat().st_size for path in demos),
            "verified_folders": verified_folder_count,
            "expected_folders": 45,
            "verified_bytes": verified_bytes,
            "total_bytes": total,
            "verified_percent": round(100 * verified_bytes / total, 2) if total else 0,
            "failed_folders": {name: row for name, row in folders.items() if row["status"] == "failed"},
            "training_started": False,
            "preprocessing_applied": False,
        },
        indent=2,
    )
)
