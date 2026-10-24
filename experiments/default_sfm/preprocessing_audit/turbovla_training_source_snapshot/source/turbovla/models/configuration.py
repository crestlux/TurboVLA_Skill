from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any, Mapping


@dataclass
class TextEncoderConfig:
    model_name_or_path: str = "bert-base-uncased"
    max_length: int = 256
    padding_length: int | None = None
    padding_length_by_instruction: dict[str, int] = field(default_factory=dict)
    sub_sentence_present: bool = True
    frozen: bool = True
    force_eval_when_frozen: bool = True
    zero_padded_tokens: bool = False
    local_files_only: bool = True
    attention_implementation: str | None = None


@dataclass
class VisionEncoderConfig:
    backbone_type: str = "dinov3"
    model_name_or_path: str = "facebook/dinov3-vitb16-pretrain-lvd1689m"
    resnet_weights: str = "IMAGENET1K_V2"
    resnet_output_stride: int = 16
    load_pretrained: bool = True
    image_size: int = 256
    num_views: int = 2
    position_embedding: str = "view"
    encode_views_separately: bool = True
    frozen: bool = False
    local_files_only: bool = True
    attention_implementation: str | None = None
    compute_precision: str = "bf16_autocast"
    # Existing checkpoints retain their historical dependency-specific behavior.
    # Official releases evaluated with Transformers 4.56 use last_hidden_state.
    feature_source: str = "hidden_states"
    position_init_std: float = 0.01
    position_scale_init: float = 0.01
    dropout: float = 0.1


@dataclass
class InteractionConfig:
    hidden_dim: int = 256
    nheads: int = 8
    num_layers: int = 6
    dim_feedforward: int = 2048
    enhancer_inner_dim: int = 1024
    text_dropout: float = 0.0
    fusion_dropout: float = 0.0
    fusion_droppath: float = 0.1
    padding_strategy: str = "key_padding_mask"
    residual_style: str = "normalized"
    attention_backend: str = "manual"
    compute_precision: str = "fp32"


@dataclass
class ActionHeadConfig:
    action_dim: int = 7
    state_dim: int = 8
    horizon: int = 12
    num_state_tokens: int = 2
    num_layers: int = 3
    mlp_hidden_dim: int = 512
    state_hidden_dim: int = 256
    dropout: float = 0.1


@dataclass
class FTHistoryConfig:
    enabled: bool = False
    history_length: int = 50
    input_dim: int = 6
    hidden_dim: int = 128
    num_layers: int = 1
    mean: list[float] | None = None
    std: list[float] | None = None
    std_floor: float = 0.01

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("ft_history.enabled must be a boolean")
        for name in ("history_length", "input_dim", "hidden_dim", "num_layers"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"ft_history.{name} must be a positive integer")
        if self.input_dim != 6:
            raise ValueError("FT history must have six channels: Fx, Fy, Fz, Tx, Ty, Tz")
        if not math.isfinite(self.std_floor) or self.std_floor <= 0:
            raise ValueError("ft_history.std_floor must be finite and positive")
        if (self.mean is None) != (self.std is None):
            raise ValueError("Supply both FT-history mean and std, or fit both from training data")
        for name in ("mean", "std"):
            values = getattr(self, name)
            if values is not None and (len(values) != self.input_dim or not all(math.isfinite(v) for v in values)):
                raise ValueError(f"ft_history.{name} must contain six finite values")
        if self.std is not None and any(value < 0 for value in self.std):
            raise ValueError("ft_history.std must be nonnegative")


@dataclass
class TurboVLAConfig:
    name: str = "TurboVLA"
    compatibility_profile: str = "libero"
    use_language: bool = True
    text: TextEncoderConfig = field(default_factory=TextEncoderConfig)
    vision: VisionEncoderConfig = field(default_factory=VisionEncoderConfig)
    interaction: InteractionConfig = field(default_factory=InteractionConfig)
    action: ActionHeadConfig = field(default_factory=ActionHeadConfig)
    ft_history: FTHistoryConfig = field(default_factory=FTHistoryConfig)

    def __post_init__(self) -> None:
        if not isinstance(self.use_language, bool):
            raise ValueError("use_language must be a boolean")
        if self.name != "TurboVLA":
            raise ValueError(f"model name must be 'TurboVLA', got {self.name!r}")
        if self.compatibility_profile not in {"libero", "robotwin"}:
            raise ValueError("compatibility_profile must be libero or robotwin")
        if self.vision.num_views < 1:
            raise ValueError("vision.num_views must be positive")
        if self.vision.backbone_type not in {"dinov3", "resnet50"}:
            raise ValueError("vision.backbone_type must be 'dinov3' or 'resnet50'")
        if self.vision.backbone_type == "resnet50":
            if self.vision.resnet_output_stride not in {16, 32}:
                raise ValueError("vision.resnet_output_stride must be 16 or 32")
            if self.vision.image_size % self.vision.resnet_output_stride != 0:
                raise ValueError("vision.image_size must be divisible by vision.resnet_output_stride")
        if self.vision.position_embedding not in {"view", "learned_patch"}:
            raise ValueError("vision.position_embedding must be 'view' or 'learned_patch'")
        if self.vision.compute_precision not in {"fp32", "bf16", "bf16_autocast"}:
            raise ValueError("vision.compute_precision must be fp32, bf16, or bf16_autocast")
        if self.vision.feature_source not in {"hidden_states", "last_hidden_state"}:
            raise ValueError("vision.feature_source must be hidden_states or last_hidden_state")
        if self.text.padding_length is not None:
            if self.text.padding_length < 1:
                raise ValueError("text.padding_length must be positive")
            if self.text.padding_length > self.text.max_length:
                raise ValueError("text.padding_length cannot exceed text.max_length")
        for instruction, length in self.text.padding_length_by_instruction.items():
            if not instruction:
                raise ValueError("text.padding_length_by_instruction cannot contain an empty instruction")
            if length < 1 or length > self.text.max_length:
                raise ValueError(f"invalid text padding length {length} for instruction {instruction!r}")
        if self.interaction.hidden_dim % self.interaction.nheads != 0:
            raise ValueError("interaction.hidden_dim must be divisible by interaction.nheads")
        if self.interaction.padding_strategy not in {"key_padding_mask", "zero_fill"}:
            raise ValueError("interaction.padding_strategy must be key_padding_mask or zero_fill")
        if self.interaction.residual_style not in {"normalized", "pre_norm"}:
            raise ValueError("interaction.residual_style must be normalized or pre_norm")
        if self.interaction.attention_backend not in {"manual", "sdpa"}:
            raise ValueError("interaction.attention_backend must be manual or sdpa")
        if self.interaction.compute_precision not in {"fp32", "bf16_autocast"}:
            raise ValueError("interaction.compute_precision must be fp32 or bf16_autocast")
        if self.action.action_dim < 1 or self.action.state_dim < 1 or self.action.horizon < 1:
            raise ValueError("action dimensions and horizon must be positive")
        if self.ft_history.enabled and self.ft_history.mean is None:
            raise ValueError("FT-history normalization is missing; configure the real-robot training recipe to fit it")

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        # Keep legacy serialized configurations and strict checkpoint loading
        # unchanged when the optional branch is disabled.
        if not self.ft_history.enabled:
            result.pop("ft_history")
        return result

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "TurboVLAConfig":
        data = dict(payload)
        return cls(
            name=str(data.get("name", "TurboVLA")),
            compatibility_profile=str(data.get("compatibility_profile", "libero")),
            use_language=data.get("use_language", True),
            text=TextEncoderConfig(**dict(data.get("text", {}))),
            vision=VisionEncoderConfig(**dict(data.get("vision", {}))),
            interaction=InteractionConfig(**dict(data.get("interaction", {}))),
            action=ActionHeadConfig(**dict(data.get("action", {}))),
            ft_history=FTHistoryConfig(**dict(data.get("ft_history", {}))),
        )
