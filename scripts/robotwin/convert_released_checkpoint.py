"""Convert released RoboTwin tensor names and declare their inference compatibility.

Writes a new inference bundle, preserving the downloaded release. Requires the
RoboTwin policy environment and local BERT / DINOv3 ViT-L assets. This conversion
checks every tensor and performs a strict load; rollout success must be measured
separately. See experiments/robotwin/REPRODUCE_KO.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


PREFIX_MAP = (
    ("action_model.action_policy.action_head.", "model.action_head.decoder.action_projection."),
    ("action_model.action_policy.", "model.action_head.decoder."),
    ("action_model.state_proj.out_norm.", "model.action_head.state_projection.output_norm."),
    ("action_model.state_proj.net.", "model.action_head.state_projection.net."),
    ("dinov3.model.", "model.vision_encoder.backbone."),
    ("feature_enhancer.", "model.vision_language_interaction."),
    ("text_encoder.text_proj.", "model.text_encoder.text_projection."),
    ("text_encoder.bert.", "model.text_encoder.bert."),
    ("vision_proj.norm_in.", "model.vision_projection.input_norm."),
    ("vision_proj.norm_out.", "model.vision_projection.output_norm."),
    ("vision_proj.mlp.", "model.vision_projection.mlp."),
    ("vision_proj.skip.", "model.vision_projection.skip."),
)
EXACT_MAP = {
    "action_model.state_proj.pos": "model.action_head.state_projection.position",
    "view_embed": "model.view_embedding",
    "vision_pos_embed": "model.patch_position_embedding",
    "vision_pos_scale": "model.patch_position_scale",
}


def renamed_key(key: str) -> str:
    if key in EXACT_MAP:
        return EXACT_MAP[key]
    for old, new in PREFIX_MAP:
        if key.startswith(old):
            return new + key[len(old) :]
    raise ValueError(f"Unknown release tensor; conversion stopped: {key}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-name", default="steps_55000_ema_model.safetensors")
    parser.add_argument("--feature-source", choices=("hidden_states", "last_hidden_state"), default="last_hidden_state")
    parser.add_argument("--compatibility-profile", choices=("libero", "robotwin"), default="robotwin")
    parser.add_argument("--bert-path", type=Path, required=True)
    parser.add_argument("--dinov3-path", type=Path, required=True)
    args = parser.parse_args()

    # Keep --help and the key mapping usable without training dependencies.
    from omegaconf import OmegaConf
    from safetensors.torch import load_file, save_file
    import torch

    from starVLA.model.framework.base_framework import build_framework

    source_root = args.release_dir.resolve()
    output_root = args.output_dir.resolve()
    source_checkpoint = source_root / "checkpoints" / "robotwin" / args.checkpoint_name
    checkpoint_name = source_checkpoint.stem + ".safetensors"
    source_config = source_root / "config.yaml"
    source_stats = source_root / "dataset_statistics.json"
    if output_root.exists():
        raise FileExistsError(f"Choose a new output directory: {output_root}")
    for path in (source_checkpoint, source_config, source_stats):
        if not path.is_file():
            raise FileNotFoundError(path)

    old = OmegaConf.load(source_config)
    if old.framework.name != "GroundingDINODiT":
        raise ValueError("This converter only supports released GroundingDINODiT bundles")
    download = json.loads((source_root / "download-verification.json").read_text())
    expected = next(row for row in download["files"] if row["path"] == f"checkpoints/robotwin/{args.checkpoint_name}")
    if sha256(source_checkpoint) != expected["sha256"]:
        raise ValueError("Release checkpoint differs from its verified download record")
    vision_config = json.loads((args.dinov3_path / "config.json").read_text())
    if (vision_config.get("hidden_size"), vision_config.get("num_hidden_layers")) != (1024, 24):
        raise ValueError("RoboTwin clean50 requires DINOv3 ViT-L (hidden_size=1024, 24 layers)")

    repo_root = Path(__file__).resolve().parents[2]
    cfg = OmegaConf.load(repo_root / "experiments/robotwin/configs/clean50.yaml")
    fw = cfg.framework
    fw.compatibility_profile = args.compatibility_profile
    fw.text = OmegaConf.merge(fw.text, old.framework.text)
    fw.text.bert_path = str(args.bert_path.resolve())
    fw.text.local_files_only = True
    fw.vision.model_path = str(args.dinov3_path.resolve())
    fw.vision.local_files_only = True
    fw.vision.feature_source = args.feature_source
    for key in ("attn_implementation", "freeze_vision_encoder", "image_size", "num_views"):
        fw.vision[key] = old.framework.dinov3[key]
    for old_key, new_key in (
        ("vision_dropout", "dropout"),
        ("vision_pos_init_std", "position_init_std"),
        ("vision_pos_scale_init", "position_scale_init"),
    ):
        fw.vision[new_key] = old.framework.dinov3[old_key]
    fw.interaction = OmegaConf.merge(fw.interaction, old.framework.fusion)
    action = old.framework.action_model
    fw.interaction.dim_feedforward = action.act_dim_feedforward
    for old_key, new_key in (
        ("action_dim", "action_dim"),
        ("state_dim", "state_dim"),
        ("action_horizon", "horizon"),
        ("act_num_layers", "num_layers"),
        ("act_dropout", "dropout"),
        ("act_state_tokens", "num_state_tokens"),
        ("act_state_hidden_dim", "state_hidden_dim"),
        ("act_mlp_hidden_dim", "mlp_hidden_dim"),
        ("act_loss_type", "loss_type"),
    ):
        fw.action[new_key] = action[old_key]
    if (action.act_hidden_dim, action.act_nheads) != (fw.interaction.hidden_dim, fw.interaction.nheads):
        raise ValueError("The released action and interaction dimensions must agree")
    fw.initialization.pretrained_ckpt = ""
    fw.initialization.load_pretrained = False
    cfg.datasets.vla_data.data_root_dir = "unused_for_inference"
    cfg.datasets.vla_data.statistics_cache_dir = str(output_root / ".unused_statistics_cache")
    cfg.output_dir = str(output_root)
    cfg.run_root_dir = str(output_root.parent)
    cfg.run_id = output_root.name
    cfg.trainer.pretrained_checkpoint = None
    cfg.trainer.save_format = "safetensors"
    cfg.trainer.freeze_modules = "model.text_encoder.bert"
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))

    if source_checkpoint.suffix == ".safetensors":
        source = load_file(str(source_checkpoint), device="cpu")
    elif source_checkpoint.suffix in {".pt", ".pth"}:
        source = torch.load(source_checkpoint, map_location="cpu", weights_only=True)
        for key in ("model_state_dict", "state_dict", "model"):
            if isinstance(source, dict) and isinstance(source.get(key), dict):
                source = source[key]
                break
    else:
        raise ValueError(f"Unsupported checkpoint format: {source_checkpoint.suffix}")
    if not isinstance(source, dict) or not all(isinstance(value, torch.Tensor) for value in source.values()):
        raise TypeError("The released checkpoint must contain a tensor state_dict")
    converted = {}
    for key, value in source.items():
        target = renamed_key(key)
        if target in converted:
            raise ValueError(f"Duplicate converted key: {target}")
        converted[target] = value

    model = build_framework(cfg)
    model.load_state_dict(converted, strict=True)
    del model

    # Original tensor values and dtypes are saved directly, without recasting.
    output_root.mkdir(parents=True, exist_ok=False)
    save_file(converted, str(output_root / checkpoint_name))
    roundtrip = load_file(str(output_root / checkpoint_name), device="cpu")
    for key, value in converted.items():
        if value.dtype != roundtrip[key].dtype or not torch.equal(value, roundtrip[key]):
            raise AssertionError(f"Saved tensor changed: {key}")
        if not torch.isfinite(value).all():
            raise ValueError(f"Nonfinite tensor: {key}")
    OmegaConf.save(cfg, output_root / "config.yaml", resolve=True)
    shutil.copy2(source_stats, output_root / "dataset_statistics.json")
    shutil.copy2(source_config, output_root / "config.release.yaml")
    report = {
        "source_checkpoint": str(source_checkpoint),
        "source_sha256": sha256(source_checkpoint),
        "source_revision": download["revision"],
        "converted_checkpoint": str(output_root / checkpoint_name),
        "converted_sha256": sha256(output_root / checkpoint_name),
        "converted_tensors": len(converted),
        "tensor_values_and_dtypes_preserved": True,
        "finite": True,
        "feature_source": args.feature_source,
        "compatibility_profile": args.compatibility_profile,
        "strict_load": True,
        "note": "Tensor values preserved; feature source/profile are explicit. Forward parity and rollout success require separate validation.",
    }
    (output_root / "conversion.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Strict load passed: {len(converted)} tensors. Output: {output_root}")


if __name__ == "__main__":
    main()
