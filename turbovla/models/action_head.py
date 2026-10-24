from __future__ import annotations

from contextlib import contextmanager

import torch
from torch import nn

from .components.utils import MLP
from .configuration import ActionHeadConfig, FTHistoryConfig


@contextmanager
def initialization_stream(seed: int | None, offset: int):
    """Keep common decoder/GRU weights identical despite different state widths."""
    if seed is None:
        yield
    else:
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed + offset)
            yield


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
    def __init__(self, config: ActionHeadConfig, hidden_dim: int, nheads: int, dim_feedforward: int,
                 adaln: bool = False) -> None:
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
        self.adaln_modulations = None
        if adaln:
            # One scale/shift pair for each of the three LayerNorms in each layer.
            # Construct after the ordinary decoder to preserve its initialization.
            self.adaln_modulations = nn.ModuleList([
                nn.ModuleList([nn.Linear(hidden_dim, 2 * hidden_dim) for _ in range(3)])
                for _ in range(config.num_layers)
            ])
            for layer in self.adaln_modulations:
                for modulation in layer:
                    nn.init.zeros_(modulation.weight)
                    nn.init.zeros_(modulation.bias)

    def forward(self, memory: torch.Tensor, ft_condition: torch.Tensor | None = None) -> torch.Tensor:
        queries = self.action_queries.weight.unsqueeze(0).expand(memory.shape[0], -1, -1)
        if self.adaln_modulations is None:
            if ft_condition is not None:
                raise ValueError("FT condition supplied to a decoder without AdaLN")
            hidden = self.decoder(tgt=queries, memory=memory)
        else:
            if ft_condition is None or ft_condition.shape != (memory.shape[0], memory.shape[2]):
                raise ValueError("AdaLN requires the same [B,D] GRU output as the memory token")
            hidden = queries
            for layer, modulations in zip(self.decoder.layers, self.adaln_modulations):
                if not layer.norm_first:
                    raise ValueError("SFM AdaLN requires the original pre-norm ACT decoder")

                def conditioned_norm(value, norm, modulation):
                    scale, shift = modulation(ft_condition).chunk(2, dim=-1)
                    return norm(value) * (1 + scale[:, None]) + shift[:, None]

                q = conditioned_norm(hidden, layer.norm1, modulations[0])
                hidden = hidden + layer.dropout1(layer.self_attn(q, q, q, need_weights=False)[0])
                q = conditioned_norm(hidden, layer.norm2, modulations[1])
                hidden = hidden + layer.dropout2(layer.multihead_attn(q, memory, memory, need_weights=False)[0])
                q = conditioned_norm(hidden, layer.norm3, modulations[2])
                hidden = hidden + layer.dropout3(layer.linear2(layer.dropout(layer.activation(layer.linear1(q)))))
            if self.decoder.norm is not None:
                hidden = self.decoder.norm(hidden)
        return torch.tanh(self.action_projection(hidden))


class FTHistoryEncoder(nn.Module):
    """Original USB GRU/last-hidden/Linear/LayerNorm, with pre-normalized input."""

    def __init__(self, config: FTHistoryConfig, hidden_dim: int) -> None:
        super().__init__()
        if config.input_normalization != "pre_normalized":
            raise ValueError("Normalize FT exactly once in the shared preprocessing")
        self.history_length, self.input_dim = config.history_length, config.input_dim
        self.gru = nn.GRU(config.input_dim, config.hidden_dim, num_layers=config.num_layers, batch_first=True)
        self.projection = nn.Sequential(nn.Linear(config.hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim))

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        if history.ndim != 3 or tuple(history.shape[1:]) != (self.history_length, self.input_dim):
            raise ValueError(f"ft_history must be [B,{self.history_length},{self.input_dim}]")
        weight = self.gru.weight_ih_l0
        # Match USB: FP32 cuDNN GRU, implicit fresh zero h0 on every call.
        with torch.autocast(device_type=weight.device.type, enabled=False):
            history = history.to(device=weight.device, dtype=weight.dtype)
            _, hidden = self.gru(history)
            return self.projection(hidden[-1]).unsqueeze(1)


class TurboVLAActionHead(nn.Module):
    def __init__(self, config: ActionHeadConfig, hidden_dim: int, nheads: int, dim_feedforward: int,
                 ft_history: FTHistoryConfig | None = None) -> None:
        super().__init__()
        ft_history = ft_history or FTHistoryConfig()
        with initialization_stream(config.initialization_seed, 101):
            self.state_projection = StateProjection(config, hidden_dim)
        with initialization_stream(config.initialization_seed, 202):
            self.decoder = ACTDecoder(config, hidden_dim, nheads, dim_feedforward, adaln=ft_history.adaln)
        with initialization_stream(config.initialization_seed, 303):
            self.ft_history_encoder = FTHistoryEncoder(ft_history, hidden_dim) if ft_history.enabled else None

    def forward(self, vision_language_tokens: torch.Tensor, state: torch.Tensor,
                ft_history: torch.Tensor | None = None) -> torch.Tensor:
        state = state.to(device=vision_language_tokens.device, dtype=vision_language_tokens.dtype)
        state_tokens = self.state_projection(state)
        memory = [vision_language_tokens, state_tokens]
        condition = None
        if self.ft_history_encoder is not None:
            if ft_history is None or ft_history.shape[0] != state.shape[0]:
                raise ValueError("GRU policy requires a matching FT-history batch")
            token = self.ft_history_encoder(ft_history).to(device=state.device, dtype=state_tokens.dtype)
            memory.append(token)
            if self.decoder.adaln_modulations is not None:
                condition = token[:, 0]
        elif ft_history is not None:
            raise ValueError("Flat-history policy takes its 300 FT values through state")
        return self.decoder(torch.cat(memory, dim=1), ft_condition=condition)
