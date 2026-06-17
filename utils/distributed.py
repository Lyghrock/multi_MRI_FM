"""
Distributed Training Utilities

Provides utilities for torchrun-based distributed training with DDP.
"""

import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from typing import Optional


def is_main_process() -> bool:
    """Check if current process is the main process (rank 0)."""
    return get_rank() == 0


def get_rank() -> int:
    """Get current process rank."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def get_world_size() -> int:
    """Get total number of processes."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1


def get_local_rank() -> int:
    """Get local rank (GPU index within node)."""
    if 'LOCAL_RANK' in os.environ:
        return int(os.environ['LOCAL_RANK'])
    return 0


def get_backend() -> str:
    """Get the backend being used."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_backend()
    return 'gloo'  # Default for CPU


def barrier():
    """Synchronize all processes."""
    if get_world_size() > 1:
        dist.barrier()


def reduce_tensor(tensor: torch.Tensor, op=dist.ReduceOp.SUM) -> torch.Tensor:
    """
    Reduce tensor across all processes.

    Args:
        tensor: Tensor to reduce
        op: Reduction operation (SUM, MEAN, etc.)

    Returns:
        Reduced tensor (only valid on main process)
    """
    if get_world_size() == 1:
        return tensor

    rt = tensor.clone()
    dist.all_reduce(rt, op=op)

    if op == dist.ReduceOp.SUM:
        rt.div_(get_world_size())

    return rt


def broadcast_tensor(tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
    """
    Broadcast tensor from source process to all processes.

    Args:
        tensor: Tensor to broadcast
        src: Source rank

    Returns:
        Broadcasted tensor
    """
    if get_world_size() == 1:
        return tensor

    dist.broadcast(tensor, src=src)
    return tensor


def gather_tensors(tensor: torch.Tensor) -> Optional[list]:
    """
    Gather tensors from all processes to main process.

    Args:
        tensor: Tensor to gather

    Returns:
        List of tensors (only on main process, None on others)
    """
    if get_world_size() == 1:
        return [tensor]

    # Get tensor size
    tensor_size = torch.tensor([tensor.numel()], device=tensor.device, dtype=torch.long)
    size_list = [torch.zeros_like(tensor_size) for _ in range(get_world_size())]
    dist.all_gather(size_list, tensor_size)

    # Gather tensors
    max_size = max(s.item() for s in size_list)
    padded_tensor = torch.zeros(max_size, device=tensor.device, dtype=tensor.dtype)
    padded_tensor[:tensor.numel()] = tensor.flatten()

    tensor_list = [torch.zeros(max_size, device=tensor.device, dtype=tensor.dtype)
                   for _ in range(get_world_size())]
    dist.all_gather(tensor_list, padded_tensor)

    # Trim to original sizes
    result = [t[:size_list[i].item()].reshape(-1, *tensor.shape[1:])
              for i, t in enumerate(tensor_list)]

    return result


def sync_model(model: torch.nn.Module):
    """
    Synchronize model parameters from main process to all processes.
    Useful for ensuring all workers start with the same weights.
    """
    if get_world_size() == 1:
        return

    for param in model.parameters():
        dist.broadcast(param.data, src=0)


def init_distributed(args) -> dict:
    """
    Initialize distributed training.

    Reads environment variables set by torchrun to set up DDP.

    Args:
        args: Argument namespace with training arguments

    Returns:
        Dictionary with distributed training info
    """
    # Get distributed info from environment (set by torchrun)
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ['RANK'])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.local_rank = int(os.environ['LOCAL_RANK'])

        # Set CUDA device
        if torch.cuda.is_available():
            torch.cuda.set_device(args.local_rank)

        # Initialize process group
        if not dist.is_initialized():
            dist.init_process_group(
                backend='nccl' if torch.cuda.is_available() else 'gloo',
                init_method='env://',
                world_size=args.world_size,
                rank=args.rank,
            )

            # Synchronize before continuing
            barrier()
    else:
        # Non-distributed mode (single process)
        args.rank = 0
        args.world_size = 1
        args.local_rank = 0

    return {
        'rank': args.rank,
        'world_size': args.world_size,
        'local_rank': args.local_rank,
    }


def cleanup_distributed():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


class DistributedSampler:
    """Custom distributed sampler with proper seeding."""

    def __init__(self, dataset, shuffle=True, seed=42):
        self.dataset = dataset
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        # Add extra samples to make it evenly divisible
        total_size = (len(indices) // get_world_size() + 1) * get_world_size()
        indices += indices[:(total_size - len(indices))]

        # Subsample
        indices = indices[get_rank()::get_world_size()]
        return iter(indices)

    def set_epoch(self, epoch):
        """Set epoch for proper shuffling with DDP."""
        self.epoch = epoch


def spawn_workers(fn, nprocs, args, nGPUs_per_node=1):
    """
    Spawn multiple processes for distributed training.

    Alternative to torchrun, useful for interactive environments.

    Args:
        fn: Function to run in each worker
        nprocs: Number of processes to spawn
        args: Arguments to pass to fn
        nGPUs_per_node: Number of GPUs per node
    """
    if nGPUs_per_node > 1 and torch.cuda.is_available():
        mp.spawn(
            fn,
            args=(nprocs, nGPUs_per_node, args),
            nprocs=nprocs,
            join=True,
        )
    else:
        fn(0, nprocs, nGPUs_per_node, args)
