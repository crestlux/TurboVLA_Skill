from __future__ import annotations

import torch
from torch import nn

from .components.utils import MLP
from .configuration import ActionHeadConfig, FTHistoryConfig


class StateProjection(nn.Module):
    def __init__(self, config: ActionHeadConfig, hidden_dim: int) -> None:
        super().__init__()
        self.num_tokens = int(config.num_state_tokens)
        self.hidden_dim = int(hidden_dim)
        self.net = nn.Sequential(
            nn.LayerNorm(config.state_dim),
            nn.Linear(config.state_dim, config.state_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.state_hidden_dim, self.num_tokens * self.hidden_dim),
        )
        self.position = nn.Parameter(torch.randn(1, self.num_tokens, self.hidden_dim) * 0.02)
        self.output_norm = nn.LayerNorm(self.hidden_dim)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        if state.ndim == 3:
            state = state[:, -1]
        if state.ndim != 2:
            raise ValueError(f"state must be [B,D] or [B,T,D], got {tuple(state.shape)}")
        tokens = self.net(state).view(state.shape[0], self.num_tokens, self.hidden_dim)
        return self.output_norm(tokens + self.position.to(device=tokens.device, dtype=tokens.dtype))


class ACTDecoder(nn.Module):
    def __init__(self, config: ActionHeadConfig, hidden_dim: int, nheads: int, dim_feedforward: int) -> None:
        super().__init__()
        self.horizon = int(config.horizon)
        self.action_queries = nn.Embedding(self.horizon, hidden_dim)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=nheads,
            dim_feedforward=dim_feedforward,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=config.num_layers)
        self.action_projection = MLP(hidden_dim, config.mlp_hidden_dim, config.action_dim, 3)

    def forward(self, memory: torch.Tensor) -> torch.Tensor:
        queries = self.action_queries.weight.unsqueeze(0).expand(memory.shape[0], -1, -1)
        hidden = self.decoder(tgt=queries, memory=memory)
        return torch.tanh(self.action_projection(hidden))


class FTHistoryEncoder(nn.Module):
    """Encode a complete oldest-to-newest window; no hidden state crosses calls."""

    def __init__(self, config: FTHistoryConfig, hidden_dim: int) -> None:
        super().__init__()
        self.history_length, self.input_dim = config.history_length, config.input_dim
        self.register_buffer("mean", torch.tensor(config.mean, dtype=torch.float32))
        self.register_buffer("scale", torch.tensor(config.std, dtype=torch.float32).clamp_min(config.std_floor))
        self.gru = nn.GRU(config.input_dim, config.hidden_dim, num_layers=config.num_layers, batch_first=True)
        self.projection = nn.Sequential(nn.Linear(config.hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim))

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        if history.ndim != 3 or tuple(history.shape[1:]) != (self.history_length, self.input_dim):
            raise ValueError(f"ft_history must be [B,{self.history_length},{self.input_dim}], got {tuple(history.shape)}")
        # cuDNN GRU stays in parameter precision (FP32 during BF16 AMP training).
        # Zero-valued measurements remain valid samples, never padding masks.
        with torch.autocast(device_type=self.mean.device.type, enabled=False):
            history = history.to(device=self.mean.device, dtype=self.gru.weight_ih_l0.dtype)
            normalized = (history - self.mean.to(dtype=history.dtype)) / self.scale.to(dtype=history.dtype)
            _, hidden = self.gru(normalized)  # implicit fresh zero h0 for each window
            return self.projection(hidden[-1]).unsqueeze(1)


class TurboVLAActionHead(nn.Module):
    def __init__(self, config: ActionHeadConfig, hidden_dim: int, nheads: int, dim_feedforward: int,
                 ft_history: FTHistoryConfig | None = None) -> None:
        super().__init__()
        self.state_projection = StateProjection(config, hidden_dim)
        self.decoder = ACTDecoder(config, hidden_dim, nheads, dim_feedforward)
        # Construct last so shared legacy parameters retain their seeded values.
        self.ft_history_encoder = FTHistoryEncoder(ft_history, hidden_dim) if ft_history and ft_history.enabled else None

    def forward(self, vision_language_tokens: torch.Tensor, state: torch.Tensor,
                ft_history: torch.Tensor | None = None) -> torch.Tensor:
        state = state.to(device=vision_language_tokens.device, dtype=vision_language_tokens.dtype)
        state_tokens = self.state_projection(state)
        memory = [vision_language_tokens, state_tokens]
        if self.ft_history_encoder is not None:
            if ft_history is None:
                raise ValueError("GRU-conditioned policy requires ft_history")
            if ft_history.shape[0] != state.shape[0]:
                raise ValueError("FT-history batch size differs from state")
            memory.append(self.ft_history_encoder(ft_history).to(device=state.device, dtype=state_tokens.dtype))
        return self.decoder(torch.cat(memory, dim=1))
