"""JEPA predictor implementations."""

from typing import Optional

import torch
import torch.nn as nn


class MLPPredictor(nn.Module):
    """Token-wise MLP predictor from context latent space to target latent space."""

    def __init__(
        self,
        d_model: int = 256,
        hidden_dim: Optional[int] = None,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        hidden_dim = hidden_dim or d_model * 4

        layers = []
        in_dim = d_model
        for _ in range(max(n_layers, 1)):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, d_model))

        self.net = nn.Sequential(*layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.net(x))


class TransformerPredictor(nn.Module):
    """Light transformer predictor for token-level latent prediction."""

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ffn: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        d_ffn = d_ffn or d_model * 4
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ffn,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.proj(self.encoder(x)))


class MoEPredictor(nn.Module):
    """Dense top-k mixture-of-experts predictor."""

    def __init__(
        self,
        d_model: int = 256,
        hidden_dim: Optional[int] = None,
        n_experts: int = 8,
        top_k: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        hidden_dim = hidden_dim or d_model * 4
        self.n_experts = n_experts
        self.top_k = min(top_k, n_experts)

        self.router = nn.Linear(d_model, n_experts)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, d_model),
            )
            for _ in range(n_experts)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        flat = x.reshape(B * N, D)

        logits = self.router(flat)
        top_values, top_indices = torch.topk(logits, k=self.top_k, dim=-1)
        top_weights = torch.softmax(top_values, dim=-1)

        expert_outputs = torch.stack([expert(flat) for expert in self.experts], dim=1)
        gather_idx = top_indices.unsqueeze(-1).expand(-1, -1, D)
        selected = torch.gather(expert_outputs, dim=1, index=gather_idx)
        mixed = (selected * top_weights.unsqueeze(-1)).sum(dim=1)

        return self.norm(mixed.reshape(B, N, D))


def create_predictor(
    predictor_type: str = 'mlp',
    d_model: int = 256,
    hidden_dim: Optional[int] = None,
    n_heads: int = 4,
    n_layers: int = 2,
    n_experts: int = 8,
    top_k: int = 2,
    dropout: float = 0.1,
) -> nn.Module:
    predictor_type = predictor_type.lower()
    if predictor_type == 'mlp':
        return MLPPredictor(d_model=d_model, hidden_dim=hidden_dim, n_layers=n_layers, dropout=dropout)
    if predictor_type == 'transformer':
        return TransformerPredictor(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ffn=hidden_dim,
            dropout=dropout,
        )
    if predictor_type == 'moe':
        return MoEPredictor(
            d_model=d_model,
            hidden_dim=hidden_dim,
            n_experts=n_experts,
            top_k=top_k,
            dropout=dropout,
        )
    raise ValueError(f"Unknown JEPA predictor_type: {predictor_type}")
