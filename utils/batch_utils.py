"""
Batch utilities for multi-modal MRI training.

The preprocessed dataset stores modality data in nested dictionaries with
patches and ROI-level tensors. Models train on a smaller canonical contract:

    sMRI: (B, N, 3)
    fMRI: (B, T, N)
    dMRI: (B, N, N)

This module converts between those forms without requiring real data-specific
assumptions beyond the available tensors.
"""

from typing import Any, Dict, Optional

import torch


def move_to_device(obj: Any, device: torch.device | str) -> Any:
    """Recursively move tensors in a nested batch to device."""
    if isinstance(obj, torch.Tensor):
        return obj.to(device, non_blocking=True)
    if isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, list):
        return [move_to_device(v, device) for v in obj]
    if isinstance(obj, tuple):
        return tuple(move_to_device(v, device) for v in obj)
    return obj


def first_tensor(obj: Any) -> Optional[torch.Tensor]:
    """Return the first tensor found inside a nested object."""
    if isinstance(obj, torch.Tensor):
        return obj
    if isinstance(obj, dict):
        for value in obj.values():
            tensor = first_tensor(value)
            if tensor is not None:
                return tensor
    if isinstance(obj, (list, tuple)):
        for value in obj:
            tensor = first_tensor(value)
            if tensor is not None:
                return tensor
    return None


def infer_batch_size(batch: Dict[str, Any]) -> int:
    tensor = first_tensor(batch)
    if tensor is None:
        raise ValueError("Cannot infer batch size from a batch with no tensors")
    if tensor.dim() == 0:
        return 1
    return tensor.shape[0]


def _cast(tensor: torch.Tensor, device=None, dtype=None) -> torch.Tensor:
    if device is not None:
        tensor = tensor.to(device, non_blocking=True)
    if dtype is not None and tensor.is_floating_point():
        tensor = tensor.to(dtype=dtype)
    return tensor.float() if tensor.dtype == torch.float16 else tensor


def _ensure_batch(tensor: torch.Tensor, min_dim: int) -> torch.Tensor:
    while tensor.dim() < min_dim:
        tensor = tensor.unsqueeze(0)
    return tensor


def _first_present(mapping: Dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def _ensure_patch_batch(tensor: torch.Tensor, patch_dims: int) -> Optional[torch.Tensor]:
    """Normalize patch tensors to (B, K, *patch_shape)."""
    if tensor.dim() == patch_dims + 1:
        return tensor.unsqueeze(0)
    if tensor.dim() == patch_dims + 2:
        return tensor
    return None


def _extract_patch_tensor(
    value: Any,
    keys: list[str],
    patch_dims: int,
    device=None,
    dtype=None,
) -> Optional[torch.Tensor]:
    if value is None:
        return None
    if isinstance(value, dict):
        tensor = _first_present(value, keys)
    else:
        tensor = value

    if not isinstance(tensor, torch.Tensor):
        return None

    tensor = _ensure_patch_batch(tensor, patch_dims)
    if tensor is None:
        return None
    return _cast(tensor, device, dtype)


def _extract_patch_from_batch(
    batch: Dict[str, Any],
    top_key: str,
    modality_key: str,
    nested_keys: list[str],
    patch_dims: int,
    device=None,
    dtype=None,
) -> Optional[torch.Tensor]:
    tensor = _extract_patch_tensor(batch.get(top_key), nested_keys, patch_dims, device, dtype)
    if tensor is not None:
        return tensor

    nested = batch.get(modality_key)
    if isinstance(nested, dict):
        return _extract_patch_tensor(nested, nested_keys, patch_dims, device, dtype)
    return None


def _extract_smri(value: Any, n_rois: int, device=None, dtype=None) -> Optional[torch.Tensor]:
    if value is None:
        return None
    if isinstance(value, dict):
        tensor = _first_present(value, ["roi", "features", "sMRI_roi"])
        if tensor is None:
            return None
    else:
        tensor = value

    if not isinstance(tensor, torch.Tensor):
        return None
    tensor = _ensure_batch(tensor, 3)
    if tensor.dim() != 3:
        return None
    if tensor.shape[-1] != 3 and tensor.shape[1] == 3:
        tensor = tensor.transpose(1, 2)
    if tensor.shape[1] != n_rois:
        tensor = tensor[:, :n_rois]
    return _cast(tensor, device, dtype)


def _extract_fmri(value: Any, n_rois: int, device=None, dtype=None) -> Optional[torch.Tensor]:
    if value is None:
        return None
    nested = isinstance(value, dict)
    if isinstance(value, dict):
        tensor = _first_present(value, ["timeseries", "time_series", "time_patches", "fMRI"])
        if tensor is None:
            return None
    else:
        tensor = value

    if not isinstance(tensor, torch.Tensor):
        return None

    # Nested (N_time_patches, T_patch, N) -> (1, N_time_patches*T_patch, N).
    # Direct 3D tensors are treated as already batched (B, T, N).
    if nested and tensor.dim() == 3 and tensor.shape[-1] == n_rois:
        tensor = tensor.unsqueeze(0)
    # (B, N_time_patches, T_patch, N) -> (B, T, N)
    if tensor.dim() == 4 and tensor.shape[-1] == n_rois:
        B, K, T, N = tensor.shape
        tensor = tensor.reshape(B, K * T, N)
    elif tensor.dim() == 2 and tensor.shape[-1] == n_rois:
        tensor = tensor.unsqueeze(0)
    elif tensor.dim() != 3:
        return None

    if tensor.shape[-1] != n_rois:
        tensor = tensor[..., :n_rois]
    return _cast(tensor, device, dtype)


def _extract_dmri(value: Any, n_rois: int, device=None, dtype=None) -> Optional[torch.Tensor]:
    if value is None:
        return None
    if isinstance(value, dict):
        tensor = _first_present(value, ["SC", "SC_matrix", "sc", "matrix", "dMRI_SC_placeholder"])
        if tensor is None:
            return None
    else:
        tensor = value

    if not isinstance(tensor, torch.Tensor):
        return None
    tensor = _ensure_batch(tensor, 3)
    if tensor.dim() != 3:
        return None
    if tensor.shape[-2] != n_rois or tensor.shape[-1] != n_rois:
        tensor = tensor[:, :n_rois, :n_rois]
    return _cast(tensor, device, dtype)


def canonicalize_brain_batch(
    batch: Dict[str, Any],
    n_rois: int = 200,
    device=None,
    dtype=None,
) -> Dict[str, torch.Tensor]:
    """
    Convert nested dataset batches to canonical tensors consumed by encoders.
    Missing modalities are left absent; the foundation model handles them.
    """
    if not isinstance(batch, dict):
        raise TypeError(f"Expected batch dict, got {type(batch)!r}")

    canonical: Dict[str, torch.Tensor] = {}

    smri = _extract_smri(batch.get("sMRI"), n_rois, device, dtype)
    if smri is not None:
        canonical["sMRI"] = smri
    smri_patches = _extract_patch_from_batch(
        batch,
        "sMRI_patches",
        "sMRI",
        ["patches", "voxel_patches", "sMRI_patches"],
        patch_dims=3,
        device=device,
        dtype=dtype,
    )
    if smri_patches is not None:
        canonical["sMRI_patches"] = smri_patches

    fmri = _extract_fmri(batch.get("fMRI"), n_rois, device, dtype)
    if fmri is not None:
        canonical["fMRI"] = fmri
    fmri_fc_patches = _extract_patch_from_batch(
        batch,
        "fMRI_fc_patches",
        "fMRI",
        ["fc_patches", "FC_patches", "matrix_patches", "fMRI_fc_patches"],
        patch_dims=2,
        device=device,
        dtype=dtype,
    )
    if fmri_fc_patches is not None:
        canonical["fMRI_fc_patches"] = fmri_fc_patches

    dmri = _extract_dmri(batch.get("dMRI"), n_rois, device, dtype)
    if dmri is not None:
        canonical["dMRI"] = dmri
    dmri_fa_patches = _extract_patch_from_batch(
        batch,
        "dMRI_FA_patches",
        "dMRI",
        ["FA_patches", "fa_patches", "dMRI_FA_patches"],
        patch_dims=3,
        device=device,
        dtype=dtype,
    )
    if dmri_fa_patches is not None:
        canonical["dMRI_FA_patches"] = dmri_fa_patches
    dmri_md_patches = _extract_patch_from_batch(
        batch,
        "dMRI_MD_patches",
        "dMRI",
        ["MD_patches", "md_patches", "dMRI_MD_patches"],
        patch_dims=3,
        device=device,
        dtype=dtype,
    )
    if dmri_md_patches is not None:
        canonical["dMRI_MD_patches"] = dmri_md_patches
    dmri_sc_patches = _extract_patch_from_batch(
        batch,
        "dMRI_SC_patches",
        "dMRI",
        ["SC_patches", "sc_patches", "matrix_patches", "dMRI_SC_patches"],
        patch_dims=2,
        device=device,
        dtype=dtype,
    )
    if dmri_sc_patches is not None:
        canonical["dMRI_SC_patches"] = dmri_sc_patches

    return canonical


def upper_triangular_from_matrix(matrix: torch.Tensor, n_rois: Optional[int] = None) -> torch.Tensor:
    """Extract flattened upper triangular values from a batch of square matrices."""
    matrix = _ensure_batch(matrix, 3)
    if n_rois is None:
        n_rois = matrix.shape[-1]
    matrix = matrix[:, :n_rois, :n_rois]
    i, j = torch.triu_indices(n_rois, n_rois, offset=1, device=matrix.device)
    return matrix[:, i, j]


def functional_connectivity_upper(timeseries: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Compute flattened ROI functional connectivity from (B, T, N) time-series.
    Returns zeros when T < 2.
    """
    timeseries = _ensure_batch(timeseries, 3).float()
    B, T, N = timeseries.shape
    if T < 2:
        return torch.zeros(B, N * (N - 1) // 2, device=timeseries.device, dtype=timeseries.dtype)

    centered = timeseries - timeseries.mean(dim=1, keepdim=True)
    std = centered.std(dim=1, keepdim=True).clamp_min(eps)
    normalized = centered / std
    corr = torch.bmm(normalized.transpose(1, 2), normalized) / max(T - 1, 1)
    corr = corr.clamp(min=-1.0 + eps, max=1.0 - eps)
    corr = corr - torch.diag_embed(torch.diagonal(corr, dim1=-2, dim2=-1))
    return upper_triangular_from_matrix(corr, N)


def make_mae_targets(
    batch: Dict[str, Any],
    n_rois: int = 200,
    device=None,
    dtype=None,
) -> Dict[str, torch.Tensor]:
    """Create modality targets matching MAE decoder output shapes."""
    canonical = canonicalize_brain_batch(batch, n_rois=n_rois, device=device, dtype=dtype)
    targets: Dict[str, torch.Tensor] = {}

    if "sMRI" in canonical:
        targets["sMRI"] = canonical["sMRI"]

    if "fMRI" in canonical:
        targets["fMRI"] = functional_connectivity_upper(canonical["fMRI"])

    if "dMRI" in canonical:
        targets["dMRI"] = upper_triangular_from_matrix(canonical["dMRI"], n_rois)

    return targets
