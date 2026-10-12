"""Validate a renamed release and compare inference with pinned official sources."""
from __future__ import annotations

import argparse
import collections
import hashlib
import importlib
import json
import sys
import types
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--converted-dir", type=Path, required=True)
    parser.add_argument("--official-source", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    import av
    import numpy as np
    import pyarrow.parquet as pq
    import torch
    from PIL import Image
    from safetensors import safe_open
    from starVLA.model.framework.base_framework import baseframework
    from convert_released_checkpoint import renamed_key

    torch.set_num_threads(4)
    filename = "steps_55000_ema_model.safetensors"
    source_path = args.release_dir / "checkpoints/robotwin" / filename
    converted_path = args.converted_dir / filename
    download = json.loads((args.release_dir / "download-verification.json").read_text())
    expected = next(row["sha256"] for row in download["files"] if row["path"].endswith(filename))
    assert sha256(source_path) == expected
    assert (args.release_dir / "dataset_statistics.json").read_bytes() == (
        args.converted_dir / "dataset_statistics.json"
    ).read_bytes()
    dtypes = collections.Counter()
    with safe_open(source_path, framework="pt", device="cpu") as source, safe_open(
        converted_path, framework="pt", device="cpu"
    ) as converted:
        assert len(source.keys()) == len(converted.keys()) == 878
        assert {renamed_key(key) for key in source.keys()} == set(converted.keys())
        for key in source.keys():
            a, b = source.get_tensor(key), converted.get_tensor(renamed_key(key))
            assert a.dtype == b.dtype and torch.equal(a, b), key
            assert torch.isfinite(a).all(), key
            dtypes[str(a.dtype)] += 1

    model = baseframework.from_pretrained(str(converted_path)).eval()
    # Import the downloaded official model under a separate package name, so
    # both implementations can be compared without changing installed sources.
    package = types.ModuleType("official_turbovla")
    package.__path__ = [str(args.official_source / "turbovla")]
    sys.modules[package.__name__] = package
    official_module = importlib.import_module("official_turbovla.models.turbovla")
    official_config = importlib.import_module("official_turbovla.models.configuration")
    config = model.model.config.to_dict()
    for key in ("backbone_type", "resnet_weights", "resnet_output_stride", "load_pretrained"):
        config["vision"].pop(key, None)
    reference = official_module.TurboVLA(official_config.TurboVLAConfig.from_mapping(config)).eval()
    reference.load_state_dict(model.model.state_dict(), strict=True)
    model = model.to(device="cuda", dtype=torch.bfloat16)
    reference = reference.to(device="cuda", dtype=torch.bfloat16)

    stats = json.loads((args.release_dir / "dataset_statistics.json").read_text())["new_embodiment"]["state"]
    low, high = np.asarray(stats["min"]), np.asarray(stats["max"])
    examples = []
    tasks = ["adjust_bottle", "grab_roller", "stack_blocks_three"]
    for task in tasks:
        task_dir = args.data_root / task
        episode = json.loads((task_dir / "meta/episodes.jsonl").read_text().splitlines()[0])
        frame_index = episode["length"] // 2
        images = []
        for camera in ("cam_high", "cam_left_wrist", "cam_right_wrist"):
            path = task_dir / f"videos/chunk-000/observation.images.{camera}/episode_000000.mp4"
            with av.open(str(path)) as container:
                stream = container.streams.video[0]
                stream.thread_count = 2
                try:
                    for index, frame in enumerate(container.decode(stream)):
                        if index == frame_index:
                            images.append(Image.fromarray(frame.to_ndarray(format="rgb24")))
                            break
                finally:
                    stream.codec_context.close(strict=False)
        assert len(images) == 3
        state = np.asarray(pq.read_table(
            task_dir / "data/chunk-000/episode_000000.parquet", columns=["observation.state"]
        )["observation.state"][frame_index].as_py(), dtype=np.float32)
        state = np.clip(2 * (state - low) / np.maximum(high - low, 1e-8) - 1, -1, 1).astype(np.float32)
        examples.append(dict(image=images, lang=episode["tasks"][0], state=state[None]))

    comparisons = []
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for batch in ([examples[0]], examples):
            instructions, samples, states = model._model_inputs(batch)
            actual = model.model(instructions, samples, states)
            expected_actions = reference(instructions, samples, states)
            assert actual.shape == (len(batch), 50, 14)
            assert torch.isfinite(actual).all() and torch.isfinite(expected_actions).all()
            torch.testing.assert_close(actual, expected_actions, rtol=0, atol=0)
            public_api = model.predict_action(batch)["normalized_actions"]
            assert np.array_equal(public_api, actual.float().cpu().numpy())
            comparisons.append(dict(batch_size=len(batch), output_shape=list(actual.shape),
                                    max_abs_difference=float((actual - expected_actions).abs().max()),
                                    bitwise_equal=True))
    source_comparison = json.loads((args.official_source.parent / "source-comparison.json").read_text())
    report = dict(status="passed", checkpoint=str(converted_path),
                  release_revision=download["revision"], source_sha256=sha256(source_path),
                  converted_sha256=sha256(converted_path), tensors=878,
                  tensor_values_and_dtypes_preserved=True, tensor_dtypes=dict(dtypes),
                  finite=True, strict_load=True, normalization_statistics_preserved=True,
                  official_code_commit=source_comparison["official_commit"],
                  reference_scope="Pinned currently published core model; identical wrapper inspected. This does not establish equivalence with an unavailable historical training runtime.",
                  inference_precision="bfloat16", sampled_training_tasks=tasks,
                  forward_comparisons=comparisons,
                  peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
