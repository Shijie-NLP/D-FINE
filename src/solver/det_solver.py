"""
D-FINE: Redefine Regression Task of DETRs as Fine-grained Distribution Refinement
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright (c) 2023 lyuwenyu. All Rights Reserved.

Optimized for strict distributed I/O safety, clear two-stage curriculum
learning control flow, and robust fallback mechanisms.
"""

import datetime
import json
import time
from typing import Any

import torch

from ..misc import dist_utils
from ..misc.profiler_utils import stats
from ._solver import BaseSolver
from .det_engine import evaluate, train_one_epoch


class DetSolver(BaseSolver):
    """
    Detection Task Solver orchestrating the complete training and evaluation lifecycle.
    Implements advanced Two-Stage training with dynamic EMA resetting, critical for
    stabilizing dense prediction architectures during late-stage convergence.
    """

    def fit(self) -> None:
        self.train()
        args = self.cfg
        metric_names = ["AP50:95", "AP50", "AP75", "APsmall", "APmedium", "APlarge"]

        # ----------------------------------------------------------------------
        # W&B Initialization
        # ----------------------------------------------------------------------
        if self.use_wandb and dist_utils.is_main_process():
            import wandb

            wandb.init(
                project=args.yaml_cfg.get("project_name", "D-FINE"),
                name=args.yaml_cfg.get("exp_name", "experiment"),
                config=args.yaml_cfg,
            )
            wandb.watch(self.model)

        n_parameters, model_stats = stats(self.cfg)
        if dist_utils.is_main_process():
            print(model_stats)
            print("-" * 42 + " Start training " + "-" * 42)

        top1: float = 0.0
        best_stat: dict[str, Any] = {"epoch": -1}

        # Safely extract the curriculum learning threshold
        stop_epoch = getattr(self.train_dataloader.collate_fn, "stop_epoch", args.epochs)

        # ----------------------------------------------------------------------
        # Resume Checkpoint Evaluation
        # ----------------------------------------------------------------------
        if self.last_epoch > 0:
            module = self.ema.module if self.ema else self.model
            test_stats, coco_evaluator = evaluate(
                module,
                self.criterion,
                self.postprocessor,
                self.val_dataloader,
                self.evaluator,
                self.device,
                self.last_epoch,
                self.use_wandb,
            )
            for k in test_stats:
                best_stat["epoch"] = self.last_epoch
                best_stat[k] = test_stats[k][0]
                top1 = test_stats[k][0]
                if dist_utils.is_main_process():
                    print(f"Resumed best_stat: {best_stat}")

        best_stat_print = best_stat.copy()
        start_time = time.time()
        start_epoch = self.last_epoch + 1

        # ----------------------------------------------------------------------
        # Core Training Loop
        # ----------------------------------------------------------------------
        for epoch in range(start_epoch, args.epochs):
            self.train_dataloader.set_epoch(epoch)
            if dist_utils.is_dist_available_and_initialized():
                self.train_dataloader.sampler.set_epoch(epoch)

            # Stage-2 Transition: Reload Stage-1 optimal weights and reset EMA decay
            if epoch == stop_epoch:
                self.load_resume_state(str(self.output_dir / "best_stg1.pth"))
                if self.ema:
                    ema_restart_decay = getattr(self.train_dataloader.collate_fn, "ema_restart_decay", 0.9999)
                    self.ema.decay = ema_restart_decay
                    if dist_utils.is_main_process():
                        print(f"Stage 2 Triggered: Refresh EMA at epoch {epoch} with decay {self.ema.decay}")

            train_stats = train_one_epoch(
                self.model,
                self.criterion,
                self.train_dataloader,
                self.optimizer,
                self.device,
                epoch,
                epochs=args.epochs,
                max_norm=args.clip_max_norm,
                print_freq=args.print_freq,
                ema=self.ema,
                scaler=self.scaler,
                lr_warmup_scheduler=self.lr_warmup_scheduler,
                writer=self.writer,
                use_wandb=self.use_wandb,
                output_dir=self.output_dir,
            )

            if self.lr_warmup_scheduler is None or self.lr_warmup_scheduler.finished():
                self.lr_scheduler.step()

            self.last_epoch += 1

            # Routine Stage-1 Checkpointing
            if self.output_dir and epoch < stop_epoch:
                checkpoint_paths = [self.output_dir / "last.pth"]
                if (epoch + 1) % args.checkpoint_freq == 0:
                    checkpoint_paths.append(self.output_dir / f"checkpoint{epoch:04}.pth")
                for checkpoint_path in checkpoint_paths:
                    dist_utils.save_on_master(self.state_dict(), checkpoint_path)

            # Evaluation
            module = self.ema.module if self.ema else self.model
            test_stats, coco_evaluator = evaluate(
                module,
                self.criterion,
                self.postprocessor,
                self.val_dataloader,
                self.evaluator,
                self.device,
                epoch,
                self.use_wandb,
                output_dir=self.output_dir,
            )

            # ------------------------------------------------------------------
            # Metrics Tracking & Two-Stage Policy Execution
            # ------------------------------------------------------------------
            for k in test_stats:
                current_score = test_stats[k][0]

                if self.writer and dist_utils.is_main_process():
                    for i, v in enumerate(test_stats[k]):
                        self.writer.add_scalar(f"Test/{k}_{i}", v, epoch)

                # Update local best stat for the metric
                if k in best_stat:
                    best_stat["epoch"] = epoch if current_score > best_stat[k] else best_stat["epoch"]
                    best_stat[k] = max(best_stat[k], current_score)
                else:
                    best_stat["epoch"] = epoch
                    best_stat[k] = current_score

                # Global Best Breakthrough
                if best_stat[k] > top1:
                    best_stat_print["epoch"] = epoch
                    top1 = best_stat[k]

                    if self.output_dir:
                        save_name = "best_stg2.pth" if epoch >= stop_epoch else "best_stg1.pth"
                        dist_utils.save_on_master(self.state_dict(), self.output_dir / save_name)

                best_stat_print[k] = max(best_stat[k], top1)

                if dist_utils.is_main_process():
                    print(f"Epoch [{epoch}] global best_stat: {best_stat_print}")

                # Stage-2 Curriculum Enforcement: Fallback & Accelerated EMA Decay
                if best_stat["epoch"] == epoch and self.output_dir:
                    if epoch >= stop_epoch:
                        if current_score > top1:
                            top1 = current_score
                            dist_utils.save_on_master(self.state_dict(), self.output_dir / "best_stg2.pth")
                    else:
                        top1 = max(current_score, top1)
                        dist_utils.save_on_master(self.state_dict(), self.output_dir / "best_stg1.pth")

                elif epoch >= stop_epoch:
                    # Model failed to surpass best metric in Stage-2.
                    # SAC Insight: Rollback to Stage-1 optimal weights and accelerate EMA decay
                    # to aggressively search local optima space.
                    best_stat = {"epoch": -1}
                    if self.ema:
                        self.ema.decay -= 0.0001
                        self.load_resume_state(str(self.output_dir / "best_stg1.pth"))
                        if dist_utils.is_main_process():
                            print(f"Fallback Triggered: Refresh EMA at epoch {epoch} with decay {self.ema.decay}")

            # ------------------------------------------------------------------
            # Logging & Export
            # ------------------------------------------------------------------
            log_stats = {
                **{f"train_{k}": v for k, v in train_stats.items()},
                **{f"test_{k}": v for k, v in test_stats.items()},
                "epoch": epoch,
                "n_parameters": n_parameters,
            }

            if self.use_wandb and dist_utils.is_main_process():
                import wandb

                wandb_logs = {
                    f"metrics/{name}": test_stats["coco_eval_bbox"][idx] for idx, name in enumerate(metric_names)
                }
                wandb_logs["epoch"] = epoch
                wandb.log(wandb_logs)

            if self.output_dir and dist_utils.is_main_process():
                with (self.output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                if coco_evaluator is not None and "bbox" in coco_evaluator.coco_eval:
                    eval_dir = self.output_dir / "eval"
                    eval_dir.mkdir(exist_ok=True)

                    filenames = ["latest.pth"]
                    if epoch % 50 == 0:
                        filenames.append(f"{epoch:03}.pth")

                    for name in filenames:
                        torch.save(
                            coco_evaluator.coco_eval["bbox"].eval,
                            eval_dir / name,
                        )

        if dist_utils.is_main_process():
            total_time = time.time() - start_time
            total_time_str = str(datetime.timedelta(seconds=int(total_time)))
            print(f"Training completed. Total time: {total_time_str}")

    def val(self) -> None:
        """Executes purely evaluation workflow."""
        self.eval()

        module = self.ema.module if self.ema else self.model
        test_stats, coco_evaluator = evaluate(
            module,
            self.criterion,
            self.postprocessor,
            self.val_dataloader,
            self.evaluator,
            self.device,
            epoch=-1,
            use_wandb=False,
            output_dir=self.output_dir,
        )

        if self.output_dir:
            dist_utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, self.output_dir / "eval.pth")
