"""
D-FINE: Redefine Regression Task of DETRs as Fine-grained Distribution Refinement
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from DETR (https://github.com/facebookresearch/detr/blob/main/engine.py)
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

Optimized for high-throughput asynchronous memory transfer, absolute GPU VRAM
leak prevention during validation, and strict distributed logging safety.
"""

import math
import os
import sys
from collections.abc import Iterable
from typing import Any, Optional

import numpy as np
import torch
import torch.amp
import torchvision
from torch.cuda.amp.grad_scaler import GradScaler
from torch.utils.tensorboard import SummaryWriter

from ..data.dataset import mscoco_category2label
from ..data.dataset.coco_eval import CocoEvaluator
from ..misc import MetricLogger, SmoothedValue, dist_utils, save_samples
from ..optim import ModelEMA, Warmup
from .validator import Validator, scale_boxes


def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    use_wandb: bool,
    max_norm: float = 0,
    **kwargs: Any,
) -> dict[str, float]:
    if use_wandb and dist_utils.is_main_process():
        import wandb

    model.train()
    criterion.train()

    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))

    epochs = kwargs.get("epochs", None)
    header = f"Epoch: [{epoch}]" if epochs is None else f"Epoch: [{epoch}/{epochs}]"

    print_freq = kwargs.get("print_freq", 10)
    writer: Optional[SummaryWriter] = kwargs.get("writer", None)

    ema: Optional[ModelEMA] = kwargs.get("ema", None)
    scaler: Optional[GradScaler] = kwargs.get("scaler", None)
    lr_warmup_scheduler: Optional[Warmup] = kwargs.get("lr_warmup_scheduler", None)
    losses: list[float] = []

    output_dir = kwargs.get("output_dir", None)
    num_visualization_sample_batch = kwargs.get("num_visualization_sample_batch", 1)

    for i, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        global_step = epoch * len(data_loader) + i
        metas = dict(epoch=epoch, step=i, global_step=global_step, epoch_step=len(data_loader))

        if global_step < num_visualization_sample_batch and output_dir is not None and dist_utils.is_main_process():
            save_samples(samples, targets, output_dir, "train", normalized=True, box_fmt="cxcywh")

        # Optimization: Asynchronous host-to-device transfer.
        # Overlaps PCI-e data transfer with computation if pin_memory=True in dataloader.
        samples = samples.to(device, non_blocking=True)
        targets = [
            {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in t.items()}
            for t in targets
        ]

        if scaler is not None:
            with torch.autocast(device_type=str(device), cache_enabled=True):
                outputs = model(samples, targets=targets)

            # Safety net: Catch NaN/Inf explosions early before they corrupt the EMA
            if torch.isnan(outputs["pred_boxes"]).any() or torch.isinf(outputs["pred_boxes"]).any():
                print(f"CRITICAL: NaN or Inf detected in bounding boxes at epoch {epoch}, step {i}")
                state = model.state_dict()
                new_state = {}
                for key, value in state.items():
                    new_key = key.replace("module.", "")
                    new_state[new_key] = value

                if dist_utils.is_main_process():
                    # Optimization: Save NaN state securely into output_dir instead of root
                    save_path = os.path.join(output_dir if output_dir else ".", "NaN_checkpoint.pth")
                    dist_utils.save_on_master({"model": new_state}, save_path)
                    print(f"NaN state saved to {save_path}")

            with torch.autocast(device_type=str(device), enabled=False):
                loss_dict = criterion(outputs, targets, **metas)

            loss = sum(loss_dict.values())
            scaler.scale(loss).backward()

            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        else:
            outputs = model(samples, targets=targets)
            loss_dict = criterion(outputs, targets, **metas)

            loss = sum(loss_dict.values())
            optimizer.zero_grad()
            loss.backward()

            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            optimizer.step()

        # Update Exponential Moving Average parameters
        if ema is not None:
            ema.update(model)

        if lr_warmup_scheduler is not None:
            lr_warmup_scheduler.step()

        loss_dict_reduced = dist_utils.reduce_dict(loss_dict)
        loss_value = sum(loss_dict_reduced.values())
        losses.append(loss_value.detach().cpu().item())

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)

        metric_logger.update(loss=loss_value, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if writer and dist_utils.is_main_process() and global_step % 10 == 0:
            writer.add_scalar("Loss/total", loss_value.item(), global_step)
            for j, pg in enumerate(optimizer.param_groups):
                writer.add_scalar(f"Lr/pg_{j}", pg["lr"], global_step)
            for k, v in loss_dict_reduced.items():
                writer.add_scalar(f"Loss/{k}", v.item(), global_step)

    if use_wandb and dist_utils.is_main_process():
        wandb.log({"lr": optimizer.param_groups[0]["lr"], "epoch": epoch, "train/loss": np.mean(losses)})

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    postprocessor: Any,
    data_loader: Iterable,
    coco_evaluator: CocoEvaluator,
    device: torch.device,
    epoch: int,
    use_wandb: bool,
    **kwargs: Any,
) -> tuple:
    if use_wandb and dist_utils.is_main_process():
        import wandb

    model.eval()
    criterion.eval()
    if coco_evaluator is not None:
        coco_evaluator.cleanup()

    metric_logger = MetricLogger(delimiter="  ")
    header = "Test:"

    iou_types = coco_evaluator.iou_types if coco_evaluator is not None else []

    gt: list[dict[str, torch.Tensor]] = []
    preds: list[dict[str, torch.Tensor]] = []

    output_dir = kwargs.get("output_dir", None)
    num_visualization_sample_batch = kwargs.get("num_visualization_sample_batch", 1)

    for i, (samples, targets) in enumerate(metric_logger.log_every(data_loader, 10, header)):
        global_step = epoch * len(data_loader) + i

        if global_step < num_visualization_sample_batch and output_dir is not None and dist_utils.is_main_process():
            save_samples(samples, targets, output_dir, "val", normalized=False, box_fmt="xyxy")

        # Optimization: Non-blocking transfer
        samples = samples.to(device, non_blocking=True)
        targets = [
            {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in t.items()}
            for t in targets
        ]

        outputs = model(samples, targets=targets)

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)

        query_bbox = torchvision.ops.box_convert(outputs["init_boxes"], in_fmt="cxcywh", out_fmt="xyxy")
        query_bbox *= orig_target_sizes.repeat(1, 2).unsqueeze(1)

        results = postprocessor(outputs, orig_target_sizes)

        res = {target["image_id"].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

        # Validator format for metrics
        for idx, (target, result) in enumerate(zip(targets, results)):
            # Optimization: Force explicit .cpu() conversion for all accumulated tensors
            # to strictly prevent massive GPU VRAM leaks during validation over large datasets.
            gt.append(
                {
                    "boxes": scale_boxes(
                        target["boxes"],
                        (target["orig_size"][1], target["orig_size"][0]),
                        (samples[idx].shape[-1], samples[idx].shape[-2]),
                    ).cpu(),
                    "labels": target["labels"].cpu(),
                    "image_path": target["image_path"],
                    "image_size": target["orig_size"].cpu(),
                }
            )

            labels = (
                (
                    torch.tensor([mscoco_category2label[int(x.item())] for x in result["labels"].flatten()])
                    .to(result["labels"].device)
                    .reshape(result["labels"].shape)
                )
                if postprocessor.remap_mscoco_category
                else result["labels"]
            )

            preds.append(
                {
                    "boxes": result["boxes"].cpu(),
                    "labels": labels.cpu(),
                    "scores": result["scores"].cpu(),
                    "query_bbox": query_bbox[idx].cpu(),
                }
            )

    # Optimization: Restrict computationally heavy rendering and I/O to main process
    if dist_utils.is_main_process():
        validator = Validator(gt=gt, preds=preds)

        if output_dir:
            vis_dir = os.path.join(str(output_dir), "vis_queries")
            validator.visualize_queries(output_dir=vis_dir)
            validator.save_plots(path_to_save=output_dir)

        metrics = validator.compute_metrics()
        print("Metrics:", metrics)

        if use_wandb:
            wandb_metrics = {f"metrics/{k}": v for k, v in metrics.items()}
            wandb_metrics["epoch"] = epoch
            wandb.log(wandb_metrics)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

    stats = {}
    if coco_evaluator is not None:
        if "bbox" in iou_types:
            stats["coco_eval_bbox"] = coco_evaluator.coco_eval["bbox"].stats.tolist()
        if "segm" in iou_types:
            stats["coco_eval_masks"] = coco_evaluator.coco_eval["segm"].stats.tolist()

    return stats, coco_evaluator
