"""DINO training wrapper for the multi-modal brain foundation model."""

import copy
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dino_loss import DINOLoss


class DINOHead(nn.Module):
    """Projection head used by DINO student and teacher networks."""

    def __init__(
        self,
        in_dim: int = 256,
        out_dim: int = 4096,
        hidden_dim: int = 2048,
        bottleneck_dim: int = 256,
        n_layers: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()

        if n_layers <= 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim), nn.GELU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            for _ in range(n_layers - 2):
                layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.GELU()])
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)

        self.last_layer = nn.Linear(bottleneck_dim, out_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        x = F.normalize(x, dim=-1)
        return self.last_layer(x)


def augment_brain_batch(
    obj: Any,
    noise_std: float = 0.01,
    dropout_prob: float = 0.05,
) -> Any:
    """Apply light tensor augmentations while preserving the nested batch shape."""
    if isinstance(obj, torch.Tensor):
        if not obj.is_floating_point():
            return obj
        out = obj
        if noise_std > 0:
            out = out + torch.randn_like(out) * noise_std
        if dropout_prob > 0:
            out = F.dropout(out, p=dropout_prob, training=True)
        return out
    if isinstance(obj, dict):
        return {key: augment_brain_batch(value, noise_std, dropout_prob) for key, value in obj.items()}
    if isinstance(obj, list):
        return [augment_brain_batch(value, noise_std, dropout_prob) for value in obj]
    if isinstance(obj, tuple):
        return tuple(augment_brain_batch(value, noise_std, dropout_prob) for value in obj)
    return obj


class MultiModalDINO(nn.Module):
    """
    DINO wrapper around BrainFoundationModel.

    The student receives augmented views and the teacher is an EMA copy that
    predicts centered probability targets from the unaugmented batch.
    """

    def __init__(
        self,
        foundation_model: nn.Module,
        d_model: int = 256,
        out_dim: int = 4096,
        hidden_dim: int = 2048,
        bottleneck_dim: int = 256,
        head_layers: int = 3,
        head_dropout: float = 0.0,
        teacher_momentum: float = 0.996,
        student_temp: float = 0.1,
        teacher_temp: float = 0.04,
        center_momentum: float = 0.9,
        n_student_views: int = 2,
        noise_std: float = 0.01,
        dropout_prob: float = 0.05,
    ):
        super().__init__()
        self.foundation = foundation_model
        self.teacher_foundation = copy.deepcopy(foundation_model)
        self.teacher_foundation.eval()
        for param in self.teacher_foundation.parameters():
            param.requires_grad = False

        self.student_head = DINOHead(
            in_dim=d_model,
            out_dim=out_dim,
            hidden_dim=hidden_dim,
            bottleneck_dim=bottleneck_dim,
            n_layers=head_layers,
            dropout=head_dropout,
        )
        self.teacher_head = copy.deepcopy(self.student_head)
        self.teacher_head.eval()
        for param in self.teacher_head.parameters():
            param.requires_grad = False

        self.teacher_momentum = teacher_momentum
        self.n_student_views = n_student_views
        self.noise_std = noise_std
        self.dropout_prob = dropout_prob
        self.loss_fn = DINOLoss(
            out_dim=out_dim,
            student_temp=student_temp,
            teacher_temp=teacher_temp,
            center_momentum=center_momentum,
        )

    def train(self, mode: bool = True):
        super().train(mode)
        self.teacher_foundation.eval()
        self.teacher_head.eval()
        return self

    @staticmethod
    def _pool_latents(H: torch.Tensor) -> torch.Tensor:
        return H.mean(dim=1)

    @torch.no_grad()
    def update_teacher(self, momentum: Optional[float] = None):
        momentum = self.teacher_momentum if momentum is None else momentum
        self._ema_update(self.foundation, self.teacher_foundation, momentum)
        self._ema_update(self.student_head, self.teacher_head, momentum)

    @staticmethod
    @torch.no_grad()
    def _ema_update(source: nn.Module, target: nn.Module, momentum: float):
        source_params = dict(source.named_parameters())
        for name, target_param in target.named_parameters():
            target_param.data.mul_(momentum).add_(source_params[name].data, alpha=1.0 - momentum)

        source_buffers = dict(source.named_buffers())
        for name, target_buffer in target.named_buffers():
            if name not in source_buffers:
                continue
            source_buffer = source_buffers[name]
            if target_buffer.is_floating_point():
                target_buffer.data.mul_(momentum).add_(source_buffer.data, alpha=1.0 - momentum)
            else:
                target_buffer.data.copy_(source_buffer.data)

    def _student_view(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        if not self.training:
            return batch
        return augment_brain_batch(batch, noise_std=self.noise_std, dropout_prob=self.dropout_prob)

    def forward(
        self,
        batch: Dict[str, Any],
        return_loss: bool = True,
    ) -> Dict[str, Any]:
        with torch.no_grad():
            teacher_H = self.teacher_foundation(batch)
            teacher_logits = self.teacher_head(self._pool_latents(teacher_H))

        student_logits = []
        n_views = self.n_student_views if self.training else 1
        for _ in range(n_views):
            view = self._student_view(batch)
            student_H = self.foundation(view)
            student_logits.append(self.student_head(self._pool_latents(student_H)))

        loss = None
        metrics = {}
        if return_loss:
            loss, metrics = self.loss_fn(student_logits, teacher_logits)

        return {
            "student_logits": student_logits,
            "teacher_logits": teacher_logits,
            "loss": loss,
            "metrics": metrics,
        }

    def encode(self, batch: Dict[str, Any]) -> torch.Tensor:
        return self.foundation(batch)

    def get_parameters(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        encoder = sum(p.numel() for p in self.foundation.parameters())
        head = sum(p.numel() for p in self.student_head.parameters())
        teacher = sum(p.numel() for p in self.teacher_foundation.parameters())
        teacher += sum(p.numel() for p in self.teacher_head.parameters())
        return {
            "total": total,
            "encoder": encoder,
            "decoder": head,
            "teacher": teacher,
        }
