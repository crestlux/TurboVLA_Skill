from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from urllib.parse import urlparse

import torch
from torch import nn
from transformers import AutoModel

from .configuration import VisionEncoderConfig


def _load_pretrained_model(config: VisionEncoderConfig):
    kwargs = {
        "local_files_only": config.local_files_only,
        "trust_remote_code": False,
    }
    if config.attention_implementation:
        try:
            return AutoModel.from_pretrained(
                config.model_name_or_path,
                attn_implementation=config.attention_implementation,
                **kwargs,
            )
        except Exception as error:
            if config.attention_implementation == "flash_attention_2":
                try:
                    return AutoModel.from_pretrained(
                        config.model_name_or_path,
                        attn_implementation="sdpa",
                        **kwargs,
                    )
                except Exception:
                    pass
            print(
                f"[TurboVLA] vision attention backend {config.attention_implementation!r} unavailable; "
                f"using model default ({type(error).__name__}).",
                flush=True,
            )
    return AutoModel.from_pretrained(config.model_name_or_path, **kwargs)


class DINOv3VisionEncoder(nn.Module):
    def __init__(self, config: VisionEncoderConfig) -> None:
        super().__init__()
        self.config = config
        self.backbone = _load_pretrained_model(config)
        self.hidden_size = self._hidden_size(self.backbone.config)
        self.patch_size = self._patch_size(self.backbone.config)
        self.prefix_tokens = self._prefix_tokens(self.backbone.config, default=5)
        self.num_patches = (config.image_size // self.patch_size) ** 2

        embeddings = getattr(self.backbone, "embeddings", None)
        mask_token = getattr(embeddings, "mask_token", None)
        if mask_token is not None:
            mask_token.requires_grad_(False)
        if config.frozen:
            self.backbone.requires_grad_(False)

    @staticmethod
    def _hidden_size(config) -> int:
        if hasattr(config, "hidden_size"):
            return int(config.hidden_size)
        if hasattr(config, "vision_config") and hasattr(config.vision_config, "hidden_size"):
            return int(config.vision_config.hidden_size)
        raise AttributeError("cannot infer DINOv3 hidden size")

    @staticmethod
    def _patch_size(config) -> int:
        if hasattr(config, "patch_size"):
            return int(config.patch_size)
        if hasattr(config, "vision_config") and hasattr(config.vision_config, "patch_size"):
            return int(config.vision_config.patch_size)
        raise AttributeError("cannot infer DINOv3 patch size")

    @staticmethod
    def _prefix_tokens(config, default: int = 0) -> int:
        if hasattr(config, "num_register_tokens"):
            return int(config.num_register_tokens) + 1
        if hasattr(config, "vision_config") and hasattr(config.vision_config, "num_register_tokens"):
            return int(config.vision_config.num_register_tokens) + 1
        return int(default)

    def set_compute_precision(self, precision: str) -> None:
        if precision not in {"fp32", "bf16", "bf16_autocast"}:
            raise ValueError(f"unsupported DINOv3 precision: {precision}")
        self.config.compute_precision = precision
        if precision == "bf16":
            self.backbone.to(dtype=torch.bfloat16)
        else:
            self.backbone.float()

    def _encode_images(self, pixel_values: torch.Tensor) -> torch.Tensor:
        height, width = pixel_values.shape[-2:]
        if height % self.patch_size or width % self.patch_size:
            raise ValueError(
                f"DINOv3 input size {(height, width)} must be divisible by patch size {self.patch_size}"
            )
        expected_patches = (height // self.patch_size) * (width // self.patch_size)
        precision = self.config.compute_precision
        if precision == "bf16":
            pixel_values = pixel_values.to(dtype=torch.bfloat16)
        autocast_context = nullcontext()
        if precision == "bf16_autocast" and pixel_values.device.type == "cuda":
            autocast_context = torch.autocast(device_type="cuda", dtype=torch.bfloat16)

        grad_context = torch.no_grad() if self.config.frozen else nullcontext()
        with grad_context:
            with autocast_context:
                outputs = self.backbone(
                    pixel_values=pixel_values,
                    output_hidden_states=self.config.feature_source == "hidden_states",
                )
        if self.config.feature_source == "last_hidden_state":
            # Stable post-LayerNorm feature across Transformers 4.56 and 4.57.
            tokens = outputs.last_hidden_state
        else:
            tokens = outputs.hidden_states[-1] if outputs.hidden_states is not None else outputs.last_hidden_state
        if tokens.shape[1] == expected_patches + self.prefix_tokens:
            return tokens[:, self.prefix_tokens :, :]
        if tokens.shape[1] == expected_patches:
            return tokens
        raise RuntimeError(
            f"DINOv3 produced {tokens.shape[1]} tokens; expected {expected_patches} patches "
            f"or {expected_patches + self.prefix_tokens} tokens including prefixes"
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if pixel_values.ndim != 5:
            raise ValueError(f"pixel_values must be [B,V,3,H,W], got {tuple(pixel_values.shape)}")
        batch_size, num_views = pixel_values.shape[:2]
        if num_views != self.config.num_views:
            raise ValueError(f"expected {self.config.num_views} views, got {num_views}")
        if self.config.encode_views_separately:
            return torch.stack(
                [self._encode_images(pixel_values[:, view_idx]) for view_idx in range(num_views)],
                dim=1,
            )
        flat = pixel_values.flatten(0, 1)
        tokens = self._encode_images(flat)
        return tokens.view(batch_size, num_views, tokens.shape[1], tokens.shape[2])


def _checkpoint_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                checkpoint = value
                break
    if not isinstance(checkpoint, dict):
        raise TypeError(f"unsupported ResNet checkpoint type: {type(checkpoint)}")

    prefixes = ("module.", "vision_encoder.backbone.", "backbone.")
    cleaned = {}
    for key, value in checkpoint.items():
        for prefix in prefixes:
            if key.startswith(prefix):
                key = key[len(prefix) :]
                break
        cleaned[key] = value
    return cleaned


def _resolve_resnet_weights(config: VisionEncoderConfig):
    from torchvision.models import ResNet50_Weights

    value = str(config.resnet_weights).strip()
    if not config.load_pretrained or value.lower() in {"", "none", "random"}:
        return None, None

    local_path = Path(value).expanduser()
    if local_path.is_file():
        return None, local_path
    if local_path.suffix in {".pth", ".pt", ".bin"} or "/" in value:
        raise FileNotFoundError(f"local ResNet-50 checkpoint not found: {local_path}")

    try:
        weights = ResNet50_Weights[value]
    except KeyError as error:
        choices = ", ".join(item.name for item in ResNet50_Weights)
        raise ValueError(
            f"unknown resnet_weights={value!r}; use a local checkpoint, 'none', or one of: {choices}"
        ) from error

    if not config.local_files_only:
        return weights, None

    filename = Path(urlparse(weights.url).path).name
    cached_path = Path(torch.hub.get_dir()) / "checkpoints" / filename
    if not cached_path.is_file():
        raise FileNotFoundError(
            f"ResNet-50 weights are not cached at {cached_path}. Download {weights.url} to that path, "
            "pass --resnet_weights /path/to/resnet50-11ad3fa6.pth, or allow downloads."
        )
    return None, cached_path


class ResNet50VisionEncoder(nn.Module):
    """ResNet-50 feature map encoder with one token per spatial C5 cell."""

    def __init__(self, config: VisionEncoderConfig) -> None:
        super().__init__()
        from torchvision.models import resnet50

        self.config = config
        replace_stride_with_dilation = [False, False, config.resnet_output_stride == 16]
        enum_weights, local_weights = _resolve_resnet_weights(config)
        self.backbone = resnet50(
            weights=enum_weights,
            replace_stride_with_dilation=replace_stride_with_dilation,
        )
        if local_weights is not None:
            checkpoint = torch.load(local_weights, map_location="cpu")
            self.backbone.load_state_dict(_checkpoint_state_dict(checkpoint), strict=True)

        # These classification-only modules are not part of the visual token encoder.
        self.backbone.avgpool = nn.Identity()
        self.backbone.fc = nn.Identity()
        self.hidden_size = 2048
        self.patch_size = int(config.resnet_output_stride)
        self.num_patches = (config.image_size // self.patch_size) ** 2

        if config.frozen:
            self.backbone.requires_grad_(False)

    def set_compute_precision(self, precision: str) -> None:
        if precision not in {"fp32", "bf16", "bf16_autocast"}:
            raise ValueError(f"unsupported ResNet-50 precision: {precision}")
        self.config.compute_precision = precision
        if precision == "bf16":
            self.backbone.to(dtype=torch.bfloat16)
        else:
            self.backbone.float()

    def _encode_images(self, pixel_values: torch.Tensor) -> torch.Tensor:
        height, width = pixel_values.shape[-2:]
        if height % self.patch_size or width % self.patch_size:
            raise ValueError(
                f"ResNet-50 input size {(height, width)} must be divisible by output stride {self.patch_size}"
            )

        precision = self.config.compute_precision
        if precision == "bf16":
            pixel_values = pixel_values.to(dtype=torch.bfloat16)
        autocast_context = nullcontext()
        if precision == "bf16_autocast" and pixel_values.device.type == "cuda":
            autocast_context = torch.autocast(device_type="cuda", dtype=torch.bfloat16)

        grad_context = torch.no_grad() if self.config.frozen else nullcontext()
        with grad_context:
            with autocast_context:
                x = self.backbone.conv1(pixel_values)
                x = self.backbone.bn1(x)
                x = self.backbone.relu(x)
                x = self.backbone.maxpool(x)
                x = self.backbone.layer1(x)
                x = self.backbone.layer2(x)
                x = self.backbone.layer3(x)
                x = self.backbone.layer4(x)

        expected_height = height // self.patch_size
        expected_width = width // self.patch_size
        if tuple(x.shape[-2:]) != (expected_height, expected_width):
            raise RuntimeError(
                f"ResNet-50 produced feature map {tuple(x.shape[-2:])}; "
                f"expected {(expected_height, expected_width)}"
            )
        return x.flatten(2).transpose(1, 2)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if pixel_values.ndim != 5:
            raise ValueError(f"pixel_values must be [B,V,3,H,W], got {tuple(pixel_values.shape)}")
        batch_size, num_views = pixel_values.shape[:2]
        if num_views != self.config.num_views:
            raise ValueError(f"expected {self.config.num_views} views, got {num_views}")
        if self.config.encode_views_separately:
            return torch.stack(
                [self._encode_images(pixel_values[:, view_idx]) for view_idx in range(num_views)],
                dim=1,
            )
        flat = pixel_values.flatten(0, 1)
        tokens = self._encode_images(flat)
        return tokens.view(batch_size, num_views, tokens.shape[1], tokens.shape[2])


def build_vision_encoder(config: VisionEncoderConfig) -> nn.Module:
    if config.backbone_type == "dinov3":
        return DINOv3VisionEncoder(config)
    if config.backbone_type == "resnet50":
        return ResNet50VisionEncoder(config)
    raise ValueError(f"unsupported vision backbone: {config.backbone_type}")
