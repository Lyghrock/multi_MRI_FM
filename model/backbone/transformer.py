"""
Transformer Encoder Implementation with PyTorch Native Acceleration

Uses torch.nn.functional.scaled_dot_product_attention (SDPA) for:
1. Flash Attention (when available)
2. Memory-efficient Attention
3. PyTorch 2.0+ optimized kernels

Supports fp16/bf16 training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math


def _get_attentionImplementation() -> str:
    """
    Get the best available attention implementation.
    Returns: 'flash', 'math', or 'efficient'
    """
    if hasattr(F, 'scaled_dot_product_attention'):
        # PyTorch 2.0+ with flash attention support
        return 'flash'
    return 'math'


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Self-Attention using PyTorch native SDPA.

    Automatically uses Flash Attention when available for:
    - 2-4x speedup on A100/H100
    - Reduced memory usage (O(N) vs O(N^2))
    - Numerical stability improvement
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        bias: bool = True,
        use_flash: bool = True
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.use_flash = use_flash

        # Combined QKV projection for efficiency (single matmul)
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

        self.dropout = nn.Dropout(dropout)
        self.dropout_p = dropout

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False
    ) -> torch.Tensor:
        """
        Args:
            x: (B, N, d_model)
            attention_mask: (B, N) or (B, N, N), True/1 for valid, False/0 for masked
            is_causal: Whether to use causal masking (for decoder)

        Returns:
            (B, N, d_model)
        """
        B, N, _ = x.shape

        # Single QKV projection: O(3d^2) instead of O(3d^2) but fewer kernel launches
        qkv = self.qkv_proj(x)  # (B, N, 3*d_model)
        qkv = qkv.reshape(B, N, 3, self.n_heads, self.d_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, N, d)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each: (B, H, N, d)

        # Use PyTorch native SDPA with Flash Attention
        if hasattr(F, 'scaled_dot_product_attention'):
            # is_causal=True creates causal mask automatically
            # attention_mask should be additive mask (0 for valid, -inf for masked)
            attn_mask = None
            if attention_mask is not None:
                if attention_mask.dtype == torch.bool:
                    attn_mask = torch.where(attention_mask, 0.0, -float('inf'))
                    if attention_mask.dim() == 2:
                        attn_mask = attn_mask.unsqueeze(1)  # (B, 1, N) for broadcasting

            if self.use_flash and q.is_cuda:
                with torch.backends.cuda.sdp_kernel(
                    enable_flash=True,
                    enable_math=True,
                    enable_mem_efficient=True
                ):
                    out = F.scaled_dot_product_attention(
                        q, k, v,
                        attn_mask=attn_mask,
                        dropout_p=self.dropout_p if self.training else 0.0,
                        is_causal=is_causal
                    )
            else:
                out = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=attn_mask,
                    dropout_p=self.dropout_p if self.training else 0.0,
                    is_causal=is_causal
                )
        else:
            # Fallback to manual attention
            scale = self.d_head ** -0.5
            attn = (q @ k.transpose(-2, -1)) * scale

            if attention_mask is not None:
                if attention_mask.dim() == 2:
                    attn_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                else:
                    attn_mask = attention_mask
                attn = attn.masked_fill(attn_mask == 0, float('-inf'))

            attn = F.softmax(attn, dim=-1)
            attn = self.dropout(attn)
            out = attn @ v

        # Reshape back: (B, H, N, d) -> (B, N, d_model)
        out = out.transpose(1, 2).contiguous()
        out = out.view(B, N, self.d_model)
        out = self.out_proj(out)

        return out


class CrossAttention(nn.Module):
    """
    PyTorch native Cross-Attention (Multi-Head Cross-Attention).

    Query comes from one modality, Key/Value from another.
    Uses SDPA with Flash Attention for efficiency.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        bias: bool = True,
        use_flash: bool = True
    ):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.use_flash = use_flash

        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.kv_proj = nn.Linear(d_model, 2 * d_model, bias=bias)  # Shared K,V projection
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

        self.dropout = nn.Dropout(dropout)
        self.dropout_p = dropout

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            query: (B, N_q, d_model) - query from one modality
            key_value: (B, N_kv, d_model) - key and value from another modality
            attention_mask: (B, N_kv) mask for key/value

        Returns:
            (B, N_q, d_model)
        """
        B, N_q, _ = query.shape
        N_kv = key_value.shape[1]

        # Project Q, K, V
        q = self.q_proj(query)  # (B, N_q, d)
        kv = self.kv_proj(key_value)  # (B, N_kv, 2d)
        kv = kv.reshape(B, N_kv, 2, self.n_heads, self.d_head)
        kv = kv.permute(2, 0, 3, 1, 4)  # (2, B, H, N_kv, d)
        k, v = kv[0], kv[1]  # Each: (B, H, N_kv, d)

        # Reshape Q
        q = q.reshape(B, N_q, self.n_heads, self.d_head)
        q = q.transpose(1, 2)  # (B, H, N_q, d)

        # SDPA with Flash Attention
        if hasattr(F, 'scaled_dot_product_attention'):
            attn_mask = None
            if attention_mask is not None:
                if attention_mask.dim() == 1:
                    attn_mask = attention_mask.unsqueeze(0).unsqueeze(1)  # (1, 1, N_kv)
                else:
                    attn_mask = attention_mask
                attn_mask = torch.where(attn_mask == 0, -float('inf'), 0.0)

            if self.use_flash and q.is_cuda:
                with torch.backends.cuda.sdp_kernel(
                    enable_flash=True,
                    enable_math=True,
                    enable_mem_efficient=True
                ):
                    out = F.scaled_dot_product_attention(
                        q, k, v,
                        attn_mask=attn_mask,
                        dropout_p=self.dropout_p if self.training else 0.0,
                        is_causal=False
                    )
            else:
                out = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=attn_mask,
                    dropout_p=self.dropout_p if self.training else 0.0,
                    is_causal=False
                )
        else:
            scale = self.d_head ** -0.5
            attn = (q @ k.transpose(-2, -1)) * scale

            if attention_mask is not None:
                if attention_mask.dim() == 1:
                    attn_mask = attention_mask.unsqueeze(0).unsqueeze(2)
                else:
                    attn_mask = attention_mask
                attn = attn.masked_fill(attn_mask == 0, float('-inf'))

            attn = F.softmax(attn, dim=-1)
            attn = self.dropout(attn)
            out = attn @ v

        # Reshape back
        out = out.transpose(1, 2).contiguous()
        out = out.view(B, N_q, self.d_model)
        out = self.out_proj(out)

        return out


class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network with optional SwiGLU.
    """

    def __init__(
        self,
        d_model: int,
        d_ffn: int,
        dropout: float = 0.1,
        activation: str = 'gelu',
        use_swiglu: bool = False
    ):
        super().__init__()

        self.use_swiglu = use_swiglu

        if use_swiglu:
            # SwiGLU: x * sigmoid(W1*x) + (1-x) * W3*x
            self.w1 = nn.Linear(d_model, d_ffn, bias=False)
            self.w2 = nn.Linear(d_ffn, d_model, bias=False)
            self.w3 = nn.Linear(d_model, d_ffn, bias=False)
        else:
            self.w1 = nn.Linear(d_model, d_ffn, bias=True)
            self.w2 = nn.Linear(d_ffn, d_model, bias=True)

        self.dropout = nn.Dropout(dropout)

        if activation == 'gelu':
            self.act = nn.GELU()
        elif activation == 'silu':
            self.act = nn.SiLU()
        else:
            self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_swiglu:
            out = self.act(self.w1(x)) * x + self.w3(x)
            out = self.w2(out)
        else:
            out = self.act(self.w1(x))
            out = self.dropout(out)
            out = self.w2(out)
        out = self.dropout(out)
        return out


class TransformerBlock(nn.Module):
    """
    Transformer Encoder Block with Pre-LayerNorm.

    Features:
    - PyTorch native SDPA (Flash Attention)
    - fp16/bf16 compatibility
    - Optional SwiGLU FFN
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ffn: Optional[int] = None,
        dropout: float = 0.1,
        attention_dropout: float = 0.0,
        activation: str = 'gelu',
        norm_first: bool = True,
        use_flash: bool = True,
        use_swiglu: bool = False
    ):
        super().__init__()

        if d_ffn is None:
            d_ffn = d_model * 4

        self.norm_first = norm_first
        self.use_flash = use_flash

        # Pre-norm
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Self-attention
        self.self_attn = MultiHeadAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=attention_dropout,
            use_flash=use_flash
        )

        # Feed-forward
        self.ffn = FeedForward(
            d_model=d_model,
            d_ffn=d_ffn,
            dropout=dropout,
            activation=activation,
            use_swiglu=use_swiglu
        )

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: (B, N, d_model)
            attention_mask: (B, N) or (B, N, N)

        Returns:
            (B, N, d_model)
        """
        if self.norm_first:
            # Pre-LayerNorm (more stable for fp16)
            x = x + self.dropout1(self.self_attn(self.norm1(x), attention_mask))
            x = x + self.dropout2(self.ffn(self.norm2(x)))
        else:
            # Post-LayerNorm
            x = self.norm1(x + self.dropout1(self.self_attn(x, attention_mask)))
            x = self.norm2(x + self.dropout2(self.ffn(x)))

        return x


class TransformerEncoder(nn.Module):
    """
    Transformer Encoder with PyTorch native acceleration.

    Features:
    - Flash Attention via SDPA
    - Automatic mixed precision (fp16/bf16)
    - Intermediate outputs (for MAE)
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        d_ffn: Optional[int] = None,
        dropout: float = 0.1,
        attention_dropout: float = 0.0,
        activation: str = 'gelu',
        norm_first: bool = True,
        return_intermediate: bool = False,
        use_flash: bool = True,
        use_swiglu: bool = False
    ):
        super().__init__()

        if d_ffn is None:
            d_ffn = d_model * 4

        self.d_model = d_model
        self.n_layers = n_layers
        self.return_intermediate = return_intermediate

        # Create transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_ffn=d_ffn,
                dropout=dropout,
                attention_dropout=attention_dropout,
                activation=activation,
                norm_first=norm_first,
                use_flash=use_flash,
                use_swiglu=use_swiglu
            )
            for _ in range(n_layers)
        ])

        # Final layer norm
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: (B, N, d_model)
            attention_mask: (B, N)

        Returns:
            (B, N, d_model) or (intermediate_outputs, final_output)
        """
        intermediate = [] if self.return_intermediate else None

        for block in self.blocks:
            x = block(x, attention_mask)
            if self.return_intermediate:
                intermediate.append(x)

        x = self.norm(x)

        if self.return_intermediate:
            return intermediate, x
        return x


class CrossModalAttention(nn.Module):
    """
    Cross-Attention for modality fusion.
    Uses PyTorch native CrossAttention for efficiency.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        dropout: float = 0.1,
        use_flash: bool = True
    ):
        super().__init__()

        self.cross_attn = CrossAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            use_flash=use_flash
        )

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            query: (B, N_q, d)
            key_value: (B, N_kv, d)
            attention_mask: (B, N_kv)

        Returns:
            (B, N_q, d)
        """
        x = self.cross_attn(query, key_value, attention_mask)
        x = self.dropout(x)
        x = self.norm(query + x)
        return x


def check_flash_attention_available() -> bool:
    """Check if Flash Attention is available in PyTorch."""
    if not hasattr(F, 'scaled_dot_product_attention'):
        return False
    try:
        with torch.backends.cuda.sdp_kernel(enable_flash=True):
            return True
    except:
        return False
