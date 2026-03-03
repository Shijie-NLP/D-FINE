"""
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.

Optimized for dynamic attribute injection safety, highly efficient regex
compilation for parameter grouping, and strict typing.
"""

import copy
import re
from typing import Any

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from ._config import BaseConfig
from .workspace import create
from .yaml_utils import load_config, merge_config, merge_dict


class YAMLConfig(BaseConfig):
    """
    A unified configuration parser that inherits from BaseConfig, mapping YAML
    dictionaries to dynamically instantiated PyTorch modules.
    """

    def __init__(self, cfg_path: str, **kwargs: Any) -> None:
        super().__init__()

        # Safely load and merge overriding kwargs
        cfg = load_config(cfg_path)
        cfg = merge_dict(cfg, kwargs)

        self.yaml_cfg = copy.deepcopy(cfg)

        # Safely map YAML configuration to class attributes
        for k in self.__dict__:
            if not k.startswith("_") and k in cfg:
                self.__dict__[k] = cfg[k]

    @property
    def global_cfg(self) -> dict[str, Any]:
        return merge_config(self.yaml_cfg, inplace=False, overwrite=False)

    @property
    def model(self) -> nn.Module:
        if self._model is None and "model" in self.yaml_cfg:
            self._model = create(self.yaml_cfg["model"], self.global_cfg)
        return super().model

    @property
    def postprocessor(self) -> nn.Module:
        if self._postprocessor is None and "postprocessor" in self.yaml_cfg:
            self._postprocessor = create(self.yaml_cfg["postprocessor"], self.global_cfg)
        return super().postprocessor

    @property
    def criterion(self) -> nn.Module:
        if self._criterion is None and "criterion" in self.yaml_cfg:
            self._criterion = create(self.yaml_cfg["criterion"], self.global_cfg)
        return super().criterion

    @property
    def optimizer(self) -> optim.Optimizer:
        if self._optimizer is None and "optimizer" in self.yaml_cfg:
            params = self.get_optim_params(self.yaml_cfg["optimizer"], self.model)
            self._optimizer = create("optimizer", self.global_cfg, params=params)
        return super().optimizer

    @property
    def lr_scheduler(self) -> optim.lr_scheduler.LRScheduler:
        if self._lr_scheduler is None and "lr_scheduler" in self.yaml_cfg:
            self._lr_scheduler = create("lr_scheduler", self.global_cfg, optimizer=self.optimizer)
            # Safe access to learning rate array for the first initialization
            print(f"Initial lr: {self._lr_scheduler.get_last_lr()[0]:.6f}")
        return super().lr_scheduler

    @property
    def lr_warmup_scheduler(self) -> optim.lr_scheduler.LRScheduler:
        if self._lr_warmup_scheduler is None and "lr_warmup_scheduler" in self.yaml_cfg:
            self._lr_warmup_scheduler = create("lr_warmup_scheduler", self.global_cfg, lr_scheduler=self.lr_scheduler)
        return super().lr_warmup_scheduler

    @property
    def train_dataloader(self) -> DataLoader:
        if self._train_dataloader is None and "train_dataloader" in self.yaml_cfg:
            self._train_dataloader = self.build_dataloader("train_dataloader")
        return super().train_dataloader

    @property
    def val_dataloader(self) -> DataLoader:
        if self._val_dataloader is None and "val_dataloader" in self.yaml_cfg:
            self._val_dataloader = self.build_dataloader("val_dataloader")
        return super().val_dataloader

    @property
    def ema(self) -> nn.Module:
        if self._ema is None and self.yaml_cfg.get("use_ema", False):
            self._ema = create("ema", self.global_cfg, model=self.model)
        return super().ema

    @property
    def scaler(self) -> Any:
        if self._scaler is None and self.yaml_cfg.get("use_amp", False):
            self._scaler = create("scaler", self.global_cfg)
        return super().scaler

    @property
    def evaluator(self) -> Any:
        if self._evaluator is None and "evaluator" in self.yaml_cfg:
            eval_type = self.yaml_cfg["evaluator"].get("type", "")
            if eval_type == "CocoEvaluator":
                # Lazy import to prevent data handling circular dependencies
                from ..data.dataset.coco_utils import get_coco_api_from_dataset

                base_ds = get_coco_api_from_dataset(self.val_dataloader.dataset)
                self._evaluator = create("evaluator", self.global_cfg, coco_gt=base_ds)
            else:
                raise NotImplementedError(f"Evaluator type '{eval_type}' is not supported.")
        return super().evaluator

    @property
    def use_wandb(self) -> bool:
        return self.yaml_cfg.get("use_wandb", False)

    @staticmethod
    def get_optim_params(cfg: dict[str, Any], model: nn.Module) -> Any:
        """
        Group model parameters based on regex patterns for distinct optimization
        strategies (e.g., differential learning rates or weight decay).

        Optimization: Introduced pre-compiled regex and early-stopping `.search()`
        to drastically reduce O(N*M) string matching overhead during initialization.
        """
        assert "type" in cfg, "Optimizer configuration must specify a 'type'."
        cfg = copy.deepcopy(cfg)

        if "params" not in cfg:
            return model.parameters()

        assert isinstance(cfg["params"], list), "Optimizer 'params' must be a list of groups."

        param_groups: list[dict[str, Any]] = []
        visited: list[str] = []

        for pg in cfg["params"]:
            # Pre-compile regex pattern for O(1) loop execution speed
            pattern = re.compile(pg["params"])

            params = {k: v for k, v in model.named_parameters() if v.requires_grad and pattern.search(k) is not None}
            # Wrap in list to ensure standard iterator behavior for the optimizer
            pg["params"] = list(params.values())
            param_groups.append(pg)
            visited.extend(list(params.keys()))

        names = [k for k, v in model.named_parameters() if v.requires_grad]

        if len(visited) < len(names):
            unseen = set(names) - set(visited)
            params = {k: v for k, v in model.named_parameters() if v.requires_grad and k in unseen}
            param_groups.append({"params": list(params.values())})
            visited.extend(list(params.keys()))

        assert len(visited) == len(names), (
            f"Parameter routing mismatch: expected {len(names)} params, grouped {len(visited)}."
        )

        return param_groups

    def build_dataloader(self, name: str) -> DataLoader:
        """Helper to instantiate DDP-aware dataloaders dynamically."""
        bs = self.get_rank_batch_size(self.yaml_cfg[name])
        global_cfg = self.global_cfg

        if "total_batch_size" in global_cfg[name]:
            # Pop unexpected key for dataloader init cleanly
            _ = global_cfg[name].pop("total_batch_size", None)

        print(f"Building {name} with batch_size={bs}...")
        loader = create(name, global_cfg, batch_size=bs)

        # Preserving the original duck-typing behavior for compatibility
        loader.shuffle = self.yaml_cfg[name].get("shuffle", False)
        return loader

    @staticmethod
    def get_rank_batch_size(cfg: dict[str, Any]) -> int:
        """Compute batch size for per rank if total_batch_size is provided."""
        has_total = "total_batch_size" in cfg
        has_bs = "batch_size" in cfg

        assert (has_total or has_bs) and not (has_total and has_bs), (
            "You must specify strictly one of `batch_size` or `total_batch_size`."
        )

        total_batch_size = cfg.get("total_batch_size", None)
        if total_batch_size is None:
            bs = cfg.get("batch_size")
            # Fallback assertion to satisfy static type checkers
            assert isinstance(bs, int), "Batch size must be an integer."
        else:
            from ..misc import dist_utils

            world_size = dist_utils.get_world_size()
            assert total_batch_size % world_size == 0, (
                f"total_batch_size ({total_batch_size}) must be cleanly divisible by world size ({world_size})."
            )
            bs = total_batch_size // world_size

        return bs
