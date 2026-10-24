#!/usr/bin/env python3
"""Verify released SFM shards, common statistics, and all nine dataset views."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import h5py
import numpy as np
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from turbovla.data.default_sfm import DefaultSFMDataset
from turbovla.data.sfm_preprocessing import normalize, select_variant, state_physical
from preprocess import sha256, utc, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--numeric-preflight", type=Path)
    parser.add_argument("--allow-smoke", action="store_true")
    parser.add_argument("--rehash", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Use a new verification output")
    root = args.dataset
    manifest = json.loads((root / "manifest.json").read_text())
    contract = json.loads((root / "data_contract.json").read_text())
    stats = json.loads((root / "normalization.json").read_text())
    if args.numeric_preflight:
        preflight = json.loads(args.numeric_preflight.read_text())
        for kind in ("state", "ft", "action_pose"):
            if preflight["physical_statistics"][kind] != stats[kind]:
                raise ValueError(f"Production statistics differ from independent numeric pass: {kind}")
    dataset_views = {name: DefaultSFMDataset(root, name, allow_smoke=args.allow_smoke, max_open_files=1)
                     for name in contract["variants"]}
    episode_offset = 0
    sampled, pixels = 0, 0
    all_instructions = set()
    for row in manifest["shards"]:
        path = root / row["file"]
        if path.stat().st_size != row["bytes"] or (args.rehash and sha256(path) != row["sha256"]):
            raise ValueError(f"Shard size or SHA-256 differs: {path}")
        with h5py.File(path, "r") as f:
            count = len(f["actions"])
            np.testing.assert_array_equal(f["source_indices"][:], np.arange(row["start"], row["original_frames"]))
            indices = np.arange(count)[:, None] + np.arange(5)
            np.testing.assert_array_equal(f["action_mask"][:], indices < count)
            # Offline/inference numeric transforms for every retained observation.
            state = normalize(state_physical(f["source_state"][:]), stats["state"]).astype(np.float32)
            ft = normalize(f["ft_history_raw"][:], stats["ft"]).astype(np.float32)
            np.testing.assert_array_equal(f["observations/state"][:], state)
            np.testing.assert_array_equal(f["observations/ft_history"][:], ft)
            for key, digest in row["processed_image_pixel_sha256"].items():
                actual = hashlib.sha256()
                for start in range(0, count, 16):
                    actual.update(f["observations/" + key][start:start + 16].tobytes())
                if actual.hexdigest() != digest:
                    raise ValueError(f"Image pixels changed: {path}/{key}")
                pixels += count
            for i in sorted({0, count // 2, count - 1}):
                for name, dataset in dataset_views.items():
                    sample = dataset[episode_offset + i]
                    spec = contract["variants"][name]
                    np.testing.assert_array_equal(sample["state"], select_variant(state[i], ft[i], spec))
                    np.testing.assert_array_equal(sample["actions"], f["actions"][i])
                    np.testing.assert_array_equal(sample["action_mask"], f["action_mask"][i])
                    if sample["state"].shape != (spec["model_state_dim"],) or sample["images"].shape != (2, 3, 224, 224):
                        raise ValueError("Wrong variant model input shape")
                    if not np.isfinite(sample["state"]).all() or sample["source_index"] != row["start"] + i:
                        raise ValueError("Invalid model input/source mapping")
                    if spec["ft_method"] != "state_embedding":
                        np.testing.assert_array_equal(sample["ft_history"], ft[i])
                    for view, key in enumerate(("image_head", "image_wrist_right")):
                        np.testing.assert_array_equal(sample["images"][view], f["observations/" + key][i].transpose(2, 0, 1))
                    if sample["instruction"] != contract["instruction_mapping"][row["task"]]:
                        raise ValueError("Wrong task instruction")
                    all_instructions.add(sample["instruction"])
                    sampled += 1
        episode_offset += row["frames"]
    batches = {}
    for name, dataset in dataset_views.items():
        # Exercise two worker processes; loading data performs no model step.
        probe_indices = [0, 1, 2, len(dataset) // 2, len(dataset) - 2, len(dataset) - 1]
        # Drain a finite probe so workers finish normally instead of aborting
        # while a discarded iterator still has prefetched tensor transfers.
        for batch in DataLoader(Subset(dataset, probe_indices), batch_size=3, num_workers=2,
                                multiprocessing_context="spawn", timeout=30):
            if batch["actions"].shape != (3, 5, 7):
                raise ValueError("Wrong batched action shape")
        batches[name] = {key: list(value.shape) for key, value in batch.items() if hasattr(value, "shape")}
        dataset.close()
    result = dict(status="passed", checked_at_utc=utc(), dataset=str(root.resolve()),
                  smoke_test=manifest["smoke_test"], episodes=manifest["episodes"], frames=episode_offset,
                  variants=len(dataset_views), sampled_variant_items=sampled, image_frames_readback=pixels,
                  task_instructions=len(all_instructions), all_shard_hashes_rechecked=args.rehash,
                  independent_numeric_statistics_match=bool(args.numeric_preflight),
                  normalization_sha256=manifest["normalization_sha256"], loader_batches=batches,
                  training_started=False, optimizer_steps=0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
