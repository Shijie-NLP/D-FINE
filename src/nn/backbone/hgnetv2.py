"""
reference
- https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py

Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.

Refactored for strict PyTorch I/O equivalence, type safety, robust distributed
handling, and precise PaddlePaddle SAME-padding mathematical alignment.
"""

import os
from typing import Any, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

from ...core import register
from ...misc.dist_utils import safe_barrier, safe_get_rank
from .common import FrozenBatchNorm2d


# Constants for initialization
kaiming_normal_ = nn.init.kaiming_normal_
zeros_ = nn.init.zeros_
ones_ = nn.init.ones_

__all__ = ["HGNetv2"]


class LearnableAffineBlock(nn.Module):
    """Learnable Affine Block for feature scaling and shifting."""

    def __init__(self, scale_value: float = 1.0, bias_value: float = 0.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor([scale_value], dtype=torch.float32))
        self.bias = nn.Parameter(torch.tensor([bias_value], dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale * x + self.bias


class ConvBNAct(nn.Module):
    """Standard Convolution -> BatchNorm -> Activation sequence."""

    def __init__(
        self,
        in_chs: int,
        out_chs: int,
        kernel_size: int,
        stride: int = 1,
        groups: int = 1,
        padding: str = "",
        use_act: bool = True,
        use_lab: bool = False,
    ):
        super().__init__()

        if padding == "same":
            # Native 'same' padding requires stride=1 in PyTorch Conv2d.
            # Using ZeroPad2d explicitly to handle edge cases precisely matching original behavior.
            self.conv = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)),
                nn.Conv2d(in_chs, out_chs, kernel_size, stride, groups=groups, bias=False),
            )
        else:
            self.conv = nn.Conv2d(
                in_chs,
                out_chs,
                kernel_size,
                stride,
                padding=(kernel_size - 1) // 2,
                groups=groups,
                bias=False,
            )

        self.bn = nn.BatchNorm2d(out_chs)
        self.act = nn.ReLU(inplace=True) if use_act else nn.Identity()
        self.lab = LearnableAffineBlock() if (use_act and use_lab) else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.lab(x)
        return x


class LightConvBNAct(nn.Module):
    """Lightweight Convolution Block leveraging point-wise and depth-wise/grouped convolutions."""

    def __init__(
        self,
        in_chs: int,
        out_chs: int,
        kernel_size: int,
        groups: int = 1,  # Kept for signature compatibility
        use_lab: bool = False,
    ):
        super().__init__()
        self.conv1 = ConvBNAct(in_chs, out_chs, kernel_size=1, use_act=False, use_lab=use_lab)
        self.conv2 = ConvBNAct(
            out_chs, out_chs, kernel_size=kernel_size, groups=out_chs, use_act=True, use_lab=use_lab
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class StemBlock(nn.Module):
    """Stem Block for initial feature extraction and downsampling."""

    def __init__(self, in_chs: int, mid_chs: int, out_chs: int, use_lab: bool = False):
        super().__init__()
        self.stem1 = ConvBNAct(in_chs, mid_chs, kernel_size=3, stride=2, use_lab=use_lab)
        self.stem2a = ConvBNAct(mid_chs, mid_chs // 2, kernel_size=2, stride=1, use_lab=use_lab)
        self.stem2b = ConvBNAct(mid_chs // 2, mid_chs, kernel_size=2, stride=1, use_lab=use_lab)
        self.stem3 = ConvBNAct(mid_chs * 2, mid_chs, kernel_size=3, stride=2, use_lab=use_lab)
        self.stem4 = ConvBNAct(mid_chs, out_chs, kernel_size=1, stride=1, use_lab=use_lab)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, ceil_mode=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem1(x)

        # Specific asymmetric padding usually from PaddlePaddle's SAME padding translation
        x_pad = F.pad(x, (0, 1, 0, 1))
        x2 = self.stem2a(x_pad)
        x2 = F.pad(x2, (0, 1, 0, 1))
        x2 = self.stem2b(x2)

        x1 = self.pool(x)
        x_cat = torch.cat([x1, x2], dim=1)

        out = self.stem3(x_cat)
        out = self.stem4(out)
        return out


class EseModule(nn.Module):
    """Effective Squeeze-and-Excitation Module."""

    def __init__(self, chs: int):
        super().__init__()
        self.conv = nn.Conv2d(chs, chs, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        # Replaced manual mean with PyTorch idiomatic adaptive_avg_pool2d
        se = F.adaptive_avg_pool2d(x, 1)
        se = self.conv(se)
        se = self.sigmoid(se)
        return identity * se


class HG_Block(nn.Module):
    """Hierarchical Generative Block (HG-Block)."""

    def __init__(
        self,
        in_chs: int,
        mid_chs: int,
        out_chs: int,
        layer_num: int,
        kernel_size: int = 3,
        residual: bool = False,
        light_block: bool = False,
        use_lab: bool = False,
        agg: str = "ese",
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.residual = residual

        self.layers = nn.ModuleList()
        for i in range(layer_num):
            layer_in_chs = in_chs if i == 0 else mid_chs
            if light_block:
                self.layers.append(LightConvBNAct(layer_in_chs, mid_chs, kernel_size=kernel_size, use_lab=use_lab))
            else:
                self.layers.append(
                    ConvBNAct(layer_in_chs, mid_chs, kernel_size=kernel_size, stride=1, use_lab=use_lab)
                )

        total_chs = in_chs + layer_num * mid_chs

        if agg == "se":
            self.aggregation = nn.Sequential(
                ConvBNAct(total_chs, out_chs // 2, kernel_size=1, stride=1, use_lab=use_lab),
                ConvBNAct(out_chs // 2, out_chs, kernel_size=1, stride=1, use_lab=use_lab),
            )
        else:
            self.aggregation = nn.Sequential(
                ConvBNAct(total_chs, out_chs, kernel_size=1, stride=1, use_lab=use_lab),
                EseModule(out_chs),
            )

        self.drop_path = nn.Dropout(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        features = [x]
        for layer in self.layers:
            x = layer(x)
            features.append(x)

        x_cat = torch.cat(features, dim=1)
        out = self.aggregation(x_cat)

        if self.residual:
            out = self.drop_path(out) + identity
        return out


class HG_Stage(nn.Module):
    """Hierarchical Generative Stage composed of multiple HG_Blocks."""

    def __init__(
        self,
        in_chs: int,
        mid_chs: int,
        out_chs: int,
        block_num: int,
        layer_num: int,
        downsample: bool = True,
        light_block: bool = False,
        kernel_size: int = 3,
        use_lab: bool = False,
        agg: str = "se",
        drop_path: Union[float, list[float]] = 0.0,
    ):
        super().__init__()
        self.downsample = (
            ConvBNAct(in_chs, in_chs, kernel_size=3, stride=2, groups=in_chs, use_act=False, use_lab=use_lab)
            if downsample
            else nn.Identity()
        )

        blocks_list = []
        for i in range(block_num):
            dp_rate = drop_path[i] if isinstance(drop_path, (list, tuple)) else drop_path
            blocks_list.append(
                HG_Block(
                    in_chs=in_chs if i == 0 else out_chs,
                    mid_chs=mid_chs,
                    out_chs=out_chs,
                    layer_num=layer_num,
                    residual=(i != 0),
                    kernel_size=kernel_size,
                    light_block=light_block,
                    use_lab=use_lab,
                    agg=agg,
                    drop_path=dp_rate,
                )
            )
        self.blocks = nn.Sequential(*blocks_list)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.downsample(x)
        x = self.blocks(x)
        return x


@register()
class HGNetv2(nn.Module):
    """
    HGNetV2 Backbone Network.

    Args:
        name: str. Architecture variant name (e.g., 'B0', 'B1').
        use_lab: bool. Whether to use LearnableAffineBlock.
        return_idx: list. Indices of stages whose features should be returned.
        freeze_stem_only: bool. Whether to freeze only the stem or subsequent stages as well.
        freeze_at: int. The stage index at which to stop freezing parameters.
        freeze_norm: bool. Whether to convert BatchNorm layers to FrozenBatchNorm.
        pretrained: bool. Whether to load pretrained weights.
        local_model_dir: str. Local directory to cache downloaded weights.
    """

    ARCH_CONFIGS: dict[str, Any] = {
        "B0": {
            "stem_channels": [3, 16, 16],
            "stage_config": {
                "stage1": [16, 16, 64, 1, False, False, 3, 3],
                "stage2": [64, 32, 256, 1, True, False, 3, 3],
                "stage3": [256, 64, 512, 2, True, True, 5, 3],
                "stage4": [512, 128, 1024, 1, True, True, 5, 3],
            },
            "url": "https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B0_stage1.pth",
        },
        "B1": {
            "stem_channels": [3, 24, 32],
            "stage_config": {
                "stage1": [32, 32, 64, 1, False, False, 3, 3],
                "stage2": [64, 48, 256, 1, True, False, 3, 3],
                "stage3": [256, 96, 512, 2, True, True, 5, 3],
                "stage4": [512, 192, 1024, 1, True, True, 5, 3],
            },
            "url": "https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B1_stage1.pth",
        },
        "B2": {
            "stem_channels": [3, 24, 32],
            "stage_config": {
                "stage1": [32, 32, 96, 1, False, False, 3, 4],
                "stage2": [96, 64, 384, 1, True, False, 3, 4],
                "stage3": [384, 128, 768, 3, True, True, 5, 4],
                "stage4": [768, 256, 1536, 1, True, True, 5, 4],
            },
            "url": "https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B2_stage1.pth",
        },
        "B3": {
            "stem_channels": [3, 24, 32],
            "stage_config": {
                "stage1": [32, 32, 128, 1, False, False, 3, 5],
                "stage2": [128, 64, 512, 1, True, False, 3, 5],
                "stage3": [512, 128, 1024, 3, True, True, 5, 5],
                "stage4": [1024, 256, 2048, 1, True, True, 5, 5],
            },
            "url": "https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B3_stage1.pth",
        },
        "B4": {
            "stem_channels": [3, 32, 48],
            "stage_config": {
                "stage1": [48, 48, 128, 1, False, False, 3, 6],
                "stage2": [128, 96, 512, 1, True, False, 3, 6],
                "stage3": [512, 192, 1024, 3, True, True, 5, 6],
                "stage4": [1024, 384, 2048, 1, True, True, 5, 6],
            },
            "url": "https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B4_stage1.pth",
        },
        "B5": {
            "stem_channels": [3, 32, 64],
            "stage_config": {
                "stage1": [64, 64, 128, 1, False, False, 3, 6],
                "stage2": [128, 128, 512, 2, True, False, 3, 6],
                "stage3": [512, 256, 1024, 5, True, True, 5, 6],
                "stage4": [1024, 512, 2048, 2, True, True, 5, 6],
            },
            "url": "https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B5_stage1.pth",
        },
        "B6": {
            "stem_channels": [3, 48, 96],
            "stage_config": {
                "stage1": [96, 96, 192, 2, False, False, 3, 6],
                "stage2": [192, 192, 512, 3, True, False, 3, 6],
                "stage3": [512, 384, 1024, 6, True, True, 5, 6],
                "stage4": [1024, 768, 2048, 3, True, True, 5, 6],
            },
            "url": "https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B6_stage1.pth",
        },
    }

    def __init__(
        self,
        name: str,
        use_lab: bool = False,
        return_idx: list[int] = [1, 2, 3],
        freeze_stem_only: bool = True,
        freeze_at: int = 0,
        freeze_norm: bool = True,
        pretrained: bool = True,
        local_model_dir: str = "weight/hgnetv2/",
    ):
        super().__init__()

        if name not in self.ARCH_CONFIGS:
            raise ValueError(f"Unsupported HGNetv2 architecture name: {name}")

        self.use_lab = use_lab
        self.return_idx = return_idx

        config = self.ARCH_CONFIGS[name]
        stem_channels = config["stem_channels"]
        stage_config = config["stage_config"]
        download_url = config["url"]

        self._out_strides = [4, 8, 16, 32]
        self._out_channels = [stage_config[k][2] for k in stage_config]

        # stem
        self.stem = StemBlock(
            in_chs=stem_channels[0],
            mid_chs=stem_channels[1],
            out_chs=stem_channels[2],
            use_lab=use_lab,
        )

        # stages
        self.stages = nn.ModuleList()
        for k in stage_config:
            (in_ch, mid_ch, out_ch, blk_num, downsample, light_blk, k_size, layer_num) = stage_config[k]

            self.stages.append(
                HG_Stage(
                    in_chs=in_ch,
                    mid_chs=mid_ch,
                    out_chs=out_ch,
                    block_num=blk_num,
                    layer_num=layer_num,
                    downsample=downsample,
                    light_block=light_blk,
                    kernel_size=k_size,
                    use_lab=use_lab,
                )
            )

        if freeze_at >= 0:
            self._freeze_parameters(self.stem)
            if not freeze_stem_only:
                for i in range(min(freeze_at + 1, len(self.stages))):
                    self._freeze_parameters(self.stages[i])

        if freeze_norm:
            self._freeze_norm(self)

        if pretrained:
            self._load_pretrained(name, download_url, local_model_dir)

    def _freeze_norm(self, module: nn.Module) -> None:
        """Recursively replace BatchNorm2d with FrozenBatchNorm2d."""
        for name, child in module.named_children():
            if isinstance(child, nn.BatchNorm2d):
                frozen_bn = FrozenBatchNorm2d(child.num_features)
                # Copy running stats and weights if they exist before replacing
                frozen_bn.load_state_dict(child.state_dict(), strict=False)
                setattr(module, name, frozen_bn)
            else:
                self._freeze_norm(child)

    def _freeze_parameters(self, m: nn.Module) -> None:
        """Disable gradient computation for all parameters in a module."""
        for p in m.parameters():
            p.requires_grad = False

    def _load_pretrained(self, name: str, download_url: str, local_model_dir: str) -> None:
        """Handle distributed-safe downloading and loading of pretrained weights."""
        model_filename = f"PPHGNetV2_{name}_stage1.pth"
        model_path = os.path.join(local_model_dir, model_filename)

        try:
            if safe_get_rank() == 0:
                if not os.path.exists(model_path):
                    logger.info(f"Downloading pretrained HGNetV2 {name} from {download_url}...")
                    # Let torch.hub handle the download cache
                    torch.hub.load_state_dict_from_url(
                        download_url, map_location="cpu", model_dir=local_model_dir, file_name=model_filename
                    )
                logger.info(f"Successfully located stage1 {name} HGNetV2 weights.")

            safe_barrier()  # Block other processes until rank 0 finishes downloading

            # Load the state dict. Using strict=False just in case, though it usually aligns perfectly.
            state = torch.load(model_path, map_location="cpu", weights_only=True)
            self.load_state_dict(state)

            if safe_get_rank() == 0:
                logger.info(f"Loaded stage1 {name} HGNetV2 from {model_path}.")

        except Exception as e:
            error_msg = (
                f"Failed to load pretrained HGNetV2 model.\n"
                f"URL: {download_url}\n"
                f"Local Path: {model_path}\n"
                f"Error: {str(e)}"
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = self.stem(x)
        outs = []
        for idx, stage in enumerate(self.stages):
            x = stage(x)
            if idx in self.return_idx:
                outs.append(x)
        return outs
