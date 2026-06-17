"""DINO loss for multi-modal brain foundation model training."""

from typing import Dict, Iterable, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


class DINOLoss(nn.Module):
    """Cross-entropy between centered teacher probabilities and student logits."""

    def __init__(
        self,
        out_dim: int,
        student_temp: float = 0.1,
        teacher_temp: float = 0.04,
        center_momentum: float = 0.9,
    ):
        super().__init__()
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))

    def forward(
        self,
        student_logits: torch.Tensor | Iterable[torch.Tensor],
        teacher_logits: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        if isinstance(student_logits, torch.Tensor):
            student_logits = [student_logits]
        else:
            student_logits = list(student_logits)

        teacher_probs = F.softmax((teacher_logits.detach() - self.center) / self.teacher_temp, dim=-1)
        total = teacher_logits.new_tensor(0.0)

        for logits in student_logits:
            student_log_probs = F.log_softmax(logits / self.student_temp, dim=-1)
            total = total + -(teacher_probs * student_log_probs).sum(dim=-1).mean()

        loss = total / max(len(student_logits), 1)
        with torch.no_grad():
            self._update_center(teacher_logits)

        teacher_entropy = -(teacher_probs * teacher_probs.clamp_min(1e-6).log()).sum(dim=-1).mean()
        metrics = {
            "L_dino": loss.item(),
            "teacher_entropy": teacher_entropy.item(),
            "center_norm": self.center.norm().item(),
            "loss_total": loss.item(),
        }
        return loss, metrics

    @torch.no_grad()
    def _update_center(self, teacher_logits: torch.Tensor):
        batch_center = teacher_logits.detach().mean(dim=0, keepdim=True)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(batch_center)
            batch_center.div_(dist.get_world_size())
        self.center.mul_(self.center_momentum).add_(batch_center, alpha=1.0 - self.center_momentum)
