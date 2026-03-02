"""
Optimized for robust distributed training, strict typing, and safe state_dict manipulation
during transfer learning (e.g., Objects365 to COCO).
"""

import atexit
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from ..core import BaseConfig
from ..misc import dist_utils


def to(m: Optional[nn.Module], device: torch.device) -> Optional[nn.Module]:
    """Safely transfers a module to the target device if the module exists."""
    if m is None:
        return None
    return m.to(device)


def remove_module_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """
    Strips the 'module.' prefix appended by DistributedDataParallel (DDP).
    Optimized using C-level dictionary comprehension for fast execution on massive models.
    """
    return {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}


class BaseSolver:
    """
    Abstract base class for task solvers. Handles distributed environment setup,
    model wrapping, EMA synchronization, and state_dict management.
    """

    def __init__(self, cfg: BaseConfig) -> None:
        self.cfg = cfg

        # Mapping indices from Objects365 to COCO for transfer learning / fine-tuning
        # fmt: off
        self.obj365_ids: list[int] = [
            0, 46, 5, 58, 114, 55, 116, 65, 21, 40, 176, 127, 249, 24, 56, 139, 92, 78,
            99, 96, 144, 295, 178, 180, 38, 39, 13, 43, 120, 219, 148, 173, 165, 154,
            137, 113, 145, 146, 204, 8, 35, 10, 88, 84, 93, 26, 112, 82, 265, 104, 141,
            152, 234, 143, 150, 97, 2, 50, 25, 75, 98, 153, 37, 73, 115, 132, 106, 61,
            163, 134, 277, 81, 133, 18, 94, 30, 169, 70, 328, 226,
        ]
        # fmt: on

        # Explicit type declarations for dynamic attributes
        self.model: Optional[nn.Module] = None
        self.ema: Optional[nn.Module] = None
        self.criterion: Optional[nn.Module] = None
        self.postprocessor: Optional[nn.Module] = None
        self.optimizer: Optional[Optimizer] = None
        self.lr_scheduler: Optional[LRScheduler] = None
        self.lr_warmup_scheduler: Optional[LRScheduler] = None
        self.writer: Any = None
        self.device: Optional[torch.device] = None

    def _setup(self) -> None:
        """Avoid instantiating unnecessary classes before distributed environment is ready."""
        cfg = self.cfg

        if cfg.device:
            device = torch.device(cfg.device)
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = cfg.model

        # NOTE: Must load tuning state before EMA instance building to ensure
        # EMA initializes with the correctly transferred weights.
        if self.cfg.tuning:
            print(f"Tuning checkpoint from {self.cfg.tuning}")
            self.load_tuning_state(self.cfg.tuning)

        # SAC Note: Retaining the original typo 'warp_model' for compatibility
        # with your internal dist_utils library.
        self.model = dist_utils.warp_model(
            self.model.to(device),
            sync_bn=cfg.sync_bn,
            find_unused_parameters=cfg.find_unused_parameters,
        )

        self.criterion = to(cfg.criterion, device)
        self.postprocessor = to(cfg.postprocessor, device)

        self.ema = to(cfg.ema, device)
        self.scaler = cfg.scaler

        self.device = device
        self.last_epoch = self.cfg.last_epoch

        self.output_dir = Path(cfg.output_dir) if cfg.output_dir else Path(".")
        if cfg.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)

        self.writer = cfg.writer

        if self.writer:
            atexit.register(self.writer.close)
            if dist_utils.is_main_process():
                self.writer.add_text("config", f"{cfg.__repr__():s}", 0)

        self.use_wandb = self.cfg.use_wandb
        if self.use_wandb:
            try:
                import wandb  # noqa

                self.use_wandb = True
            except ImportError:
                print("Warning: wandb is not installed. Disabling wandb logging.")
                self.use_wandb = False

    def cleanup(self) -> None:
        """Ensures all I/O writers are gracefully closed upon termination."""
        if self.writer:
            atexit.register(self.writer.close)

    def train(self) -> None:
        self._setup()
        self.optimizer = self.cfg.optimizer
        self.lr_scheduler = self.cfg.lr_scheduler
        self.lr_warmup_scheduler = self.cfg.lr_warmup_scheduler

        # Retaining 'warp_loader' typo for compatibility
        self.train_dataloader = dist_utils.warp_loader(
            self.cfg.train_dataloader, shuffle=self.cfg.train_dataloader.shuffle
        )
        self.val_dataloader = dist_utils.warp_loader(self.cfg.val_dataloader, shuffle=self.cfg.val_dataloader.shuffle)

        self.evaluator = self.cfg.evaluator

        # NOTE: Instantiating order matters for resuming specific training epochs
        if self.cfg.resume:
            print(f"Resume checkpoint from {self.cfg.resume}")
            self.load_resume_state(self.cfg.resume)

    def eval(self) -> None:
        self._setup()

        self.val_dataloader = dist_utils.warp_loader(self.cfg.val_dataloader, shuffle=self.cfg.val_dataloader.shuffle)
        self.evaluator = self.cfg.evaluator

        if self.cfg.resume:
            print(f"Resume checkpoint from {self.cfg.resume}")
            self.load_resume_state(self.cfg.resume)

    def state_dict(self) -> dict[str, Any]:
        """Collects state dicts from all trackable components for checkpointing."""
        state: dict[str, Any] = {}
        state["date"] = datetime.now().isoformat()
        state["last_epoch"] = self.last_epoch

        for k, v in self.__dict__.items():
            if hasattr(v, "state_dict"):
                v_unwrapped = dist_utils.de_parallel(v)
                state[k] = v_unwrapped.state_dict()

        return state

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Safely restores state dicts into respective components."""
        if "last_epoch" in state:
            self.last_epoch = state["last_epoch"]
            print(f"Load last_epoch: {self.last_epoch}")

        for k, v in self.__dict__.items():
            if hasattr(v, "load_state_dict") and k in state:
                v_unwrapped = dist_utils.de_parallel(v)
                v_unwrapped.load_state_dict(state[k])
                print(f"Load {k}.state_dict")

            elif hasattr(v, "load_state_dict") and k not in state:
                if k == "ema":
                    model = getattr(self, "model", None)
                    if model is not None:
                        ema = dist_utils.de_parallel(v)
                        model_state_dict = remove_module_prefix(model.state_dict())
                        ema.load_state_dict({"module": model_state_dict})
                        print(f"Load {k}.state_dict initialized from model.state_dict")
                else:
                    print(f"Warning: Did not load {k}.state_dict (Not found in checkpoint)")

    def load_resume_state(self, path: str) -> None:
        """Resumes strictly matching states (weights, optimizer, epoch)."""
        if path.startswith("http"):
            state = torch.hub.load_state_dict_from_url(path, map_location="cpu")
        else:
            state = torch.load(path, map_location="cpu")

        self.load_state_dict(state)

    def load_tuning_state(self, path: str) -> None:
        """Loads model weights for fine-tuning, adjusting mismatched classifier heads."""
        if path.startswith("http"):
            state = torch.hub.load_state_dict_from_url(path, map_location="cpu")
        else:
            state = torch.load(path, map_location="cpu")

        module = dist_utils.de_parallel(self.model)

        # Prioritize EMA weights if available for better generalization
        if "ema" in state:
            pretrain_state_dict = state["ema"]["module"]
        else:
            pretrain_state_dict = state["model"]

        # Adjust head parameters between datasets (e.g., Object365 to COCO)
        try:
            adjusted_state_dict = self._adjust_head_parameters(module.state_dict(), pretrain_state_dict)
            stat, infos = self._matched_state(module.state_dict(), adjusted_state_dict)
        except Exception as e:
            print(f"Head adjustment failed ({e}). Falling back to strict matching.")
            stat, infos = self._matched_state(module.state_dict(), pretrain_state_dict)

        module.load_state_dict(stat, strict=False)
        print(f"Tuning state loaded. Infos: {infos}")

    @staticmethod
    def _matched_state(
        state: dict[str, torch.Tensor], params: dict[str, torch.Tensor]
    ) -> tuple[dict[str, torch.Tensor], dict[str, list[str]]]:
        """Filters out parameter mismatches gracefully."""
        missed_list = []
        unmatched_list = []
        matched_state = {}

        for k, v in state.items():
            if k in params:
                if v.shape == params[k].shape:
                    matched_state[k] = params[k]
                else:
                    unmatched_list.append(k)
            else:
                missed_list.append(k)

        return matched_state, {"missed": missed_list, "unmatched": unmatched_list}

    def _adjust_head_parameters(
        self, cur_state_dict: dict[str, torch.Tensor], pretrain_state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Adjusts classifier and regressor heads for dataset transfer."""

        # Safely remove mismatched denoising class embeddings
        if (
            "decoder.denoising_class_embed.weight" in pretrain_state_dict
            and "decoder.denoising_class_embed.weight" in cur_state_dict
        ):
            if (
                pretrain_state_dict["decoder.denoising_class_embed.weight"].size()
                != cur_state_dict["decoder.denoising_class_embed.weight"].size()
            ):
                del pretrain_state_dict["decoder.denoising_class_embed.weight"]

        head_param_names = [
            "decoder.enc_score_head.weight",
            "decoder.enc_score_head.bias",
        ]
        for i in range(8):
            head_param_names.extend([f"decoder.dec_score_head.{i}.weight", f"decoder.dec_score_head.{i}.bias"])

        for param_name in head_param_names:
            if param_name in cur_state_dict and param_name in pretrain_state_dict:
                cur_tensor = cur_state_dict[param_name]
                pretrain_tensor = pretrain_state_dict[param_name]

                adjusted_tensor = self.map_class_weights(cur_tensor, pretrain_tensor)
                if adjusted_tensor is not None:
                    pretrain_state_dict[param_name] = adjusted_tensor
                else:
                    print(f"Cannot adjust parameter '{param_name}' due to size mismatch.")

        return pretrain_state_dict

    def map_class_weights(self, cur_tensor: torch.Tensor, pretrain_tensor: torch.Tensor) -> torch.Tensor:
        """Maps class weights from pretrain model to current model based on Obj365->COCO IDs."""
        if pretrain_tensor.size() == cur_tensor.size():
            return pretrain_tensor

        adjusted_tensor = cur_tensor.clone().detach()
        adjusted_tensor.requires_grad = False

        if pretrain_tensor.size(0) > cur_tensor.size(0):
            # Target is smaller (e.g. 80 COCO classes from 365 Obj365 classes)
            for coco_id, obj_id in enumerate(self.obj365_ids):
                adjusted_tensor[coco_id] = pretrain_tensor[obj_id + 1]
        else:
            # Target is larger
            for coco_id, obj_id in enumerate(self.obj365_ids):
                adjusted_tensor[obj_id + 1] = pretrain_tensor[coco_id]

        return adjusted_tensor

    def fit(self) -> None:
        raise NotImplementedError("The `fit` method must be implemented by subclasses.")

    def val(self) -> None:
        raise NotImplementedError("The `val` method must be implemented by subclasses.")
