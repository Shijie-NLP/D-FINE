"""
reference
- https://github.com/pytorch/vision/blob/main/references/detection/utils.py
- https://github.com/facebookresearch/detr/blob/master/util/misc.py#L406

Copyright(c) 2023 lyuwenyu. All Rights Reserved.

Refactored for robust multi-node distributed training, strict reproducibility,
and type safety.
"""

import atexit
import builtins

# Setup logger for the module instead of relying entirely on print overrides
import logging
import os
import random
import time
import warnings
from typing import Any, Optional

import numpy as np
import torch
import torch.backends.cudnn
import torch.distributed
import torch.nn as nn
from torch.nn.parallel import DataParallel as DP
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler

# Assuming custom DataLoader is required by the original repository structure
from ..data.dataloader import DataLoader


logger = logging.getLogger(__name__)


def is_dist_available_and_initialized() -> bool:
    """Check if the distributed environment is available and initialized."""
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def get_rank() -> int:
    """Get the global rank of the current process."""
    if not is_dist_available_and_initialized():
        return 0
    return torch.distributed.get_rank()


def get_world_size() -> int:
    """Get the total number of processes in the distributed group."""
    if not is_dist_available_and_initialized():
        return 1
    return torch.distributed.get_world_size()


def is_main_process() -> bool:
    """Check if the current process is the master process (rank 0)."""
    return get_rank() == 0


def safe_barrier() -> None:
    """Synchronize distributed processes safely."""
    if is_dist_available_and_initialized():
        torch.distributed.barrier()


def safe_get_rank() -> int:
    """Alias for get_rank to maintain API compatibility."""
    return get_rank()


def setup_print(is_main: bool, method: str = "builtin") -> None:
    """
    Disables printing when not in the master process.
    [SAC Note]: Overriding builtins is an anti-pattern. Kept for strict I/O compatibility.
    """
    if method == "builtin":
        builtin_print = builtins.print
    elif method == "rich":
        try:
            import rich

            builtin_print = rich.print
        except ImportError:
            builtin_print = builtins.print
            warnings.warn("rich module not found, falling back to builtin print.")
    else:
        raise ValueError(f"Unsupported print method: {method}")

    def custom_print(*args: Any, **kwargs: Any) -> None:
        force = kwargs.pop("force", False)
        if is_main or force:
            builtin_print(*args, **kwargs)

    builtins.print = custom_print


def setup_distributed(
    print_rank: int = 0,
    print_method: str = "builtin",
    seed: Optional[int] = None,
) -> bool:
    """
    Initialize the distributed environment.
    """
    try:
        # Elastic launch variables
        local_rank = int(os.environ.get("LOCAL_RANK", -1))

        torch.distributed.init_process_group(init_method="env://")
        torch.distributed.barrier()

        rank = torch.distributed.get_rank()

        # [SAC Fix]: Must use LOCAL_RANK or modulo device count for multi-node setups
        device_count = torch.cuda.device_count()
        device_id = local_rank if local_rank != -1 else (rank % device_count)

        torch.cuda.set_device(device_id)
        torch.cuda.empty_cache()

        enabled_dist = True
        if get_rank() == print_rank:
            print(f"Initialized distributed mode with backend: {torch.distributed.get_backend()}")

    except (RuntimeError, ValueError) as e:
        enabled_dist = False
        print(f"Distributed mode initialization failed: {e}. Running in single-GPU mode.")

    setup_print(get_rank() == print_rank, method=print_method)

    if seed is not None:
        setup_seed(seed)

    return enabled_dist


@atexit.register
def cleanup() -> None:
    """Cleanup distributed environment upon exit."""
    if is_dist_available_and_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


def save_on_master(*args: Any, **kwargs: Any) -> None:
    """Save PyTorch objects only on the master process."""
    if is_main_process():
        torch.save(*args, **kwargs)


def warp_model(
    model: nn.Module,
    sync_bn: bool = False,
    dist_mode: str = "ddp",
    find_unused_parameters: bool = False,
    compile: bool = False,
    compile_mode: str = "reduce-overhead",
    **kwargs: Any,
) -> nn.Module:
    """
    Wraps the model for distributed training and/or torch.compile.
    [SAC Note]: Kept typo 'warp_model' to ensure strictly identical API imports.
    """
    if is_dist_available_and_initialized():
        rank = get_rank()
        local_rank = int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count()))

        if sync_bn:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

        if dist_mode == "dp":
            model = DP(model, device_ids=[local_rank], output_device=local_rank)
        elif dist_mode == "ddp":
            model = DDP(
                model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=find_unused_parameters,
            )
        else:
            raise ValueError(f"Unsupported dist_mode: {dist_mode}")

    if compile:
        model = torch.compile(model, mode=compile_mode)

    return model


def is_parallel(model: nn.Module) -> bool:
    """Returns True if model is of type DP or DDP."""
    return isinstance(model, (DP, DDP))


def de_parallel(model: nn.Module) -> nn.Module:
    """De-parallelize a model: returns the core module."""
    return model.module if is_parallel(model) else model


def check_compile() -> bool:
    """Check if the current GPU supports efficient torch.compile."""
    gpu_ok = False
    if torch.cuda.is_available():
        device_cap = torch.cuda.get_device_capability()
        # V100 (7.0), A100 (8.0), H100 (9.0) and Ampere (8.6)
        if device_cap[0] >= 7:
            gpu_ok = True

    if not gpu_ok:
        warnings.warn("GPU architecture may not fully support torch.compile optimizations. Speedup might be limited.")
    return gpu_ok


def is_compile(model: nn.Module) -> bool:
    """Check if the model has been optimized by torch.compile."""
    try:
        import torch._dynamo

        return isinstance(model, torch._dynamo.OptimizedModule)
    except ImportError:
        return False


def de_complie(model: nn.Module) -> nn.Module:
    """
    Unwraps a compiled model.
    [SAC Note]: Kept typo 'de_complie' to ensure strictly identical API imports.
    """
    return model._orig_mod if is_compile(model) else model


def de_model(model: nn.Module) -> nn.Module:
    """Recursively unwrap a model from compilation and distribution."""
    return de_parallel(de_complie(model))


def warp_loader(loader: DataLoader, shuffle: bool = False) -> DataLoader:
    """
    Wraps a DataLoader with a DistributedSampler if DDP is initialized.
    [SAC Note]: Kept typo 'warp_loader' to ensure strictly identical API imports.
    """
    if is_dist_available_and_initialized():
        sampler = DistributedSampler(loader.dataset, shuffle=shuffle)
        # Re-instantiate the dataloader with the distributed sampler
        loader = type(loader)(
            loader.dataset,
            batch_size=loader.batch_size,
            sampler=sampler,
            drop_last=loader.drop_last,
            collate_fn=loader.collate_fn,
            pin_memory=loader.pin_memory,
            num_workers=loader.num_workers,
        )
    return loader


def reduce_dict(data: dict[str, torch.Tensor], avg: bool = True) -> dict[str, torch.Tensor]:
    """
    Reduce a dictionary of tensors across all distributed processes.
    """
    world_size = get_world_size()
    if world_size < 2:
        return data

    with torch.no_grad():
        keys, values = [], []
        for k in sorted(data.keys()):
            keys.append(k)
            # Ensure tensor is strictly on the correct device and detached
            val = data[k].detach().clone()
            if not val.is_cuda:
                val = val.cuda()
            values.append(val)

        values = torch.stack(values, dim=0)
        torch.distributed.all_reduce(values, op=torch.distributed.ReduceOp.SUM)

        if avg:
            values /= world_size

        return {k: v for k, v in zip(keys, values)}  # noqa


def all_gather(data: Any) -> list[Any]:
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors).
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    data_list = [None for _ in range(world_size)]
    torch.distributed.all_gather_object(data_list, data)
    return data_list


def sync_time() -> float:
    """Synchronize CUDA device and return current time."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


def setup_seed(seed: int, deterministic: bool = False) -> None:
    """
    Setup random seed for optimal reproducibility.
    """
    # Offset seed by rank to prevent identical augmentations across GPUs
    local_seed = seed + get_rank()

    random.seed(local_seed)
    np.random.seed(local_seed)
    torch.manual_seed(local_seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(local_seed)

    if torch.backends.cudnn.is_available() and deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False  # Critical for strict reproducibility
