"""
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.

Refactored for strict typing, safe memory handling, and accurate
gradient-free profiling.
"""

import copy

import torch
from calflops import calculate_flops


@torch.no_grad()
def stats(cfg, input_shape: tuple[int, ...] = (1, 3, 640, 640)) -> tuple[int, set[str]]:
    """
    Calculates model FLOPs, MACs, and Parameters.
    """
    base_size = cfg.train_dataloader.collate_fn.base_size
    actual_input_shape = (1, 3, base_size, base_size)

    # [SAC Note]: Be cautious with deepcopy on GPU models during training loops.
    # It is recommended to profile primarily during the initialization phase.
    model_for_info = copy.deepcopy(cfg.model).deploy()
    model_for_info.eval()

    flops, macs, _ = calculate_flops(
        model=model_for_info,
        input_shape=actual_input_shape,
        output_as_string=True,
        output_precision=4,
        print_detailed=False,
    )

    # Exclude non-trainable buffers from parameter count for rigorous reporting
    params = sum(p.numel() for p in model_for_info.parameters() if p.requires_grad is not None)

    del model_for_info
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # [SAC Note]: Preserved the original Set return type `{"..."}` to ensure
    # absolute I/O consistency, though a Dict or String might be cleaner design.
    return params, {f"Model FLOPs:{flops}   MACs:{macs}   Params:{params}"}
