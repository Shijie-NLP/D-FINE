"""
D-FINE: Redefine Regression Task of DETRs as Fine-grained Distribution Refinement
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright (c) 2023 lyuwenyu. All Rights Reserved.

Optimized for strict typing, structural re-parameterization safety, and
zero-copy position embedding generation to avoid CUDA synchronization bottlenecks.
"""

import copy
from typing import Callable, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...core import register
from .utils import get_activation


__all__ = ["HybridEncoder"]


class ConvNormLayer_fuse(nn.Module):
    """
    Fused Convolution and Normalization Layer.
    Supports structural re-parameterization to convert Conv+BN into a single Conv
    for deployment, drastically reducing memory access cost (MAC) during inference.
    """

    def __init__(
        self,
        ch_in: int,
        ch_out: int,
        kernel_size: int,
        stride: int,
        g: int = 1,
        padding: Optional[int] = None,
        bias: bool = False,
        act: Optional[str] = None,
    ):
        super().__init__()
        padding = (kernel_size - 1) // 2 if padding is None else padding
        self.conv = nn.Conv2d(ch_in, ch_out, kernel_size, stride, groups=g, padding=padding, bias=bias)
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = nn.Identity() if act is None else get_activation(act)

        self.ch_in = ch_in
        self.ch_out = ch_out
        self.kernel_size = kernel_size
        self.stride = stride
        self.g = g
        self.padding = padding
        self.bias = bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self, "conv_bn_fused"):
            y = self.conv_bn_fused(x)
        else:
            y = self.norm(self.conv(x))
        return self.act(y)

    def convert_to_deploy(self) -> None:
        """Fuses the batch normalization parameters into the convolutional weights."""
        if not hasattr(self, "conv_bn_fused"):
            self.conv_bn_fused = nn.Conv2d(
                self.ch_in,
                self.ch_out,
                self.kernel_size,
                self.stride,
                groups=self.g,
                padding=self.padding,
                bias=True,
            )

        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv_bn_fused.weight.data = kernel
        self.conv_bn_fused.bias.data = bias
        self.__delattr__("conv")
        self.__delattr__("norm")

    def get_equivalent_kernel_bias(self) -> tuple[torch.Tensor, torch.Tensor]:
        kernel3x3, bias3x3 = self._fuse_bn_tensor()
        return kernel3x3, bias3x3

    def _fuse_bn_tensor(self) -> tuple[torch.Tensor, torch.Tensor]:
        kernel = self.conv.weight
        running_mean = self.norm.running_mean
        running_var = self.norm.running_var
        gamma = self.norm.weight
        beta = self.norm.bias
        eps = self.norm.eps

        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std


class ConvNormLayer(nn.Module):
    """Standard Convolution and Normalization Layer."""

    def __init__(
        self,
        ch_in: int,
        ch_out: int,
        kernel_size: int,
        stride: int,
        g: int = 1,
        padding: Optional[int] = None,
        bias: bool = False,
        act: Optional[str] = None,
    ):
        super().__init__()
        padding = (kernel_size - 1) // 2 if padding is None else padding
        self.conv = nn.Conv2d(ch_in, ch_out, kernel_size, stride, groups=g, padding=padding, bias=bias)
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = nn.Identity() if act is None else get_activation(act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class SCDown(nn.Module):
    """Spatial Channel Downsampling module."""

    def __init__(self, c1: int, c2: int, k: int, s: int):
        super().__init__()
        self.cv1 = ConvNormLayer_fuse(c1, c2, 1, 1)
        self.cv2 = ConvNormLayer_fuse(c2, c2, k, s, c2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cv2(self.cv1(x))


class VGGBlock(nn.Module):
    """VGG-style block utilizing structural re-parameterization."""

    def __init__(self, ch_in: int, ch_out: int, act: str = "relu"):
        super().__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.conv1 = ConvNormLayer(ch_in, ch_out, 3, 1, padding=1, act=None)
        self.conv2 = ConvNormLayer(ch_in, ch_out, 1, 1, padding=0, act=None)

        # Check against pure string to prevent instantiation errors
        # when act is already an nn.Identity or similar module.
        self.act = nn.Identity() if act is None else (get_activation(act) if isinstance(act, str) else act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self, "conv"):
            y = self.conv(x)
        else:
            y = self.conv1(x) + self.conv2(x)
        return self.act(y)

    def convert_to_deploy(self) -> None:
        if not hasattr(self, "conv"):
            self.conv = nn.Conv2d(self.ch_in, self.ch_out, 3, 1, padding=1)

        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv.weight.data = kernel
        self.conv.bias.data = bias
        self.__delattr__("conv1")
        self.__delattr__("conv2")

    def get_equivalent_kernel_bias(self) -> tuple[torch.Tensor, torch.Tensor]:
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1), bias3x3 + bias1x1

    def _pad_1x1_to_3x3_tensor(self, kernel1x1: Union[torch.Tensor, int]) -> Union[torch.Tensor, int]:
        if isinstance(kernel1x1, int) and kernel1x1 == 0:
            return 0
        return F.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch: ConvNormLayer) -> tuple[Union[torch.Tensor, int], Union[torch.Tensor, int]]:
        if branch is None:
            return 0, 0
        kernel = branch.conv.weight
        running_mean = branch.norm.running_mean
        running_var = branch.norm.running_var
        gamma = branch.norm.weight
        beta = branch.norm.bias
        eps = branch.norm.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std


class ELAN(nn.Module):
    """Efficient Layer Aggregation Network block (CSP-ELAN)."""

    def __init__(
        self,
        c1: int,
        c2: int,
        c3: int,
        c4: int,
        n: int = 2,
        bias: bool = False,
        act: str = "silu",
        bottletype: Callable = VGGBlock,
    ):
        super().__init__()
        self.c = c3
        self.cv1 = ConvNormLayer_fuse(c1, c3, 1, 1, bias=bias, act=act)
        self.cv2 = nn.Sequential(
            bottletype(c3 // 2, c4, act=get_activation(act)),
            ConvNormLayer_fuse(c4, c4, 3, 1, bias=bias, act=act),
        )
        self.cv3 = nn.Sequential(
            bottletype(c4, c4, act=get_activation(act)),
            ConvNormLayer_fuse(c4, c4, 3, 1, bias=bias, act=act),
        )
        self.cv4 = ConvNormLayer_fuse(c3 + (2 * c4), c2, 1, 1, bias=bias, act=act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, dim=1))
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, dim=1))


class RepNCSPELAN4(nn.Module):
    """Reparameterized Cross Stage Partial ELAN-4 Block."""

    def __init__(
        self,
        c1: int,
        c2: int,
        c3: int,
        c4: int,
        n: int = 3,
        bias: bool = False,
        act: str = "silu",
    ):
        super().__init__()
        self.c = c3 // 2
        self.cv1 = ConvNormLayer_fuse(c1, c3, 1, 1, bias=bias, act=act)
        self.cv2 = nn.Sequential(
            CSPLayer(c3 // 2, c4, n, 1, bias=bias, act=act, bottletype=VGGBlock),
            ConvNormLayer_fuse(c4, c4, 3, 1, bias=bias, act=act),
        )
        self.cv3 = nn.Sequential(
            CSPLayer(c4, c4, n, 1, bias=bias, act=act, bottletype=VGGBlock),
            ConvNormLayer_fuse(c4, c4, 3, 1, bias=bias, act=act),
        )
        self.cv4 = ConvNormLayer_fuse(c3 + (2 * c4), c2, 1, 1, bias=bias, act=act)

    def forward_chunk(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, dim=1))
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, dim=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).split((self.c, self.c), dim=1))
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, dim=1))


class CSPLayer(nn.Module):
    """Cross Stage Partial Layer."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int = 3,
        expansion: float = 1.0,
        bias: bool = False,
        act: str = "silu",
        bottletype: Callable = VGGBlock,
    ):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = ConvNormLayer_fuse(in_channels, hidden_channels, 1, 1, bias=bias, act=act)
        self.conv2 = ConvNormLayer_fuse(in_channels, hidden_channels, 1, 1, bias=bias, act=act)
        self.bottlenecks = nn.Sequential(
            *[bottletype(hidden_channels, hidden_channels, act=get_activation(act)) for _ in range(num_blocks)]
        )
        if hidden_channels != out_channels:
            self.conv3 = ConvNormLayer_fuse(hidden_channels, out_channels, 1, 1, bias=bias, act=act)
        else:
            self.conv3 = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_1 = self.conv1(x)
        x_1 = self.bottlenecks(x_1)
        x_2 = self.conv2(x)
        return self.conv3(x_1 + x_2)


class TransformerEncoderLayer(nn.Module):
    """Single layer of the Transformer Encoder for feature refinement."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
        normalize_before: bool = False,
    ):
        super().__init__()
        self.normalize_before = normalize_before

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout, batch_first=True)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = get_activation(activation)

    @staticmethod
    def with_pos_embed(tensor: torch.Tensor, pos_embed: Optional[torch.Tensor]) -> torch.Tensor:
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        pos_embed: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = src
        if self.normalize_before:
            src = self.norm1(src)

        q = k = self.with_pos_embed(src, pos_embed)
        src, _ = self.self_attn(q, k, value=src, attn_mask=src_mask)

        src = residual + self.dropout1(src)
        if not self.normalize_before:
            src = self.norm1(src)

        residual = src
        if self.normalize_before:
            src = self.norm2(src)

        src = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = residual + self.dropout2(src)

        if not self.normalize_before:
            src = self.norm2(src)

        return src


class TransformerEncoder(nn.Module):
    """Full Transformer Encoder stack."""

    def __init__(
        self,
        encoder_layer: TransformerEncoderLayer,
        num_layers: int,
        norm: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        pos_embed: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=src_mask, pos_embed=pos_embed)

        if self.norm is not None:
            output = self.norm(output)

        return output


@register()
class HybridEncoder(nn.Module):
    """
    Hybrid CNN-Transformer Encoder.
    Fuses multi-scale features via a robust FPN + PAN architecture and refines
    the highest-level semantics using an internal Transformer Encoder.
    """

    __share__ = ["eval_spatial_size"]

    def __init__(
        self,
        in_channels: list[int] = [512, 1024, 2048],
        feat_strides: list[int] = [8, 16, 32],
        hidden_dim: int = 256,
        nhead: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.0,
        enc_act: str = "gelu",
        use_encoder_idx: list[int] = [2],
        num_encoder_layers: int = 1,
        pe_temperature: float = 10000.0,
        expansion: float = 1.0,
        depth_mult: float = 1.0,
        act: str = "silu",
        eval_spatial_size: Optional[tuple[int, int]] = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.feat_strides = feat_strides
        self.hidden_dim = hidden_dim
        self.use_encoder_idx = use_encoder_idx
        self.num_encoder_layers = num_encoder_layers
        self.pe_temperature = pe_temperature
        self.eval_spatial_size = eval_spatial_size
        self.out_channels = [hidden_dim for _ in range(len(in_channels))]
        self.out_strides = feat_strides

        # Channel projection module
        self.input_proj = nn.ModuleList()
        for in_channel in in_channels:
            proj = nn.Sequential()
            proj.add_module("conv", nn.Conv2d(in_channel, hidden_dim, kernel_size=1, bias=False))
            proj.add_module("norm", nn.BatchNorm2d(hidden_dim))
            self.input_proj.append(proj)

        # Transformer encoder module for the deepest features
        encoder_layer = TransformerEncoderLayer(
            hidden_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=enc_act,
        )

        self.encoder = nn.ModuleList(
            [TransformerEncoder(copy.deepcopy(encoder_layer), num_encoder_layers) for _ in range(len(use_encoder_idx))]
        )

        # Top-down FPN (Feature Pyramid Network)
        self.lateral_convs = nn.ModuleList()
        self.fpn_blocks = nn.ModuleList()
        for _ in range(len(in_channels) - 1, 0, -1):
            self.lateral_convs.append(ConvNormLayer_fuse(hidden_dim, hidden_dim, 1, 1))
            self.fpn_blocks.append(
                RepNCSPELAN4(
                    hidden_dim * 2,
                    hidden_dim,
                    hidden_dim * 2,
                    round(expansion * hidden_dim // 2),
                    round(3 * depth_mult),
                )
            )

        # Bottom-up PAN (Path Aggregation Network)
        self.downsample_convs = nn.ModuleList()
        self.pan_blocks = nn.ModuleList()
        for _ in range(len(in_channels) - 1):
            self.downsample_convs.append(
                nn.Sequential(
                    SCDown(hidden_dim, hidden_dim, 3, 2),
                )
            )
            self.pan_blocks.append(
                RepNCSPELAN4(
                    hidden_dim * 2,
                    hidden_dim,
                    hidden_dim * 2,
                    round(expansion * hidden_dim // 2),
                    round(3 * depth_mult),
                )
            )

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        if self.eval_spatial_size:
            for idx in self.use_encoder_idx:
                stride = self.feat_strides[idx]
                pos_embed = self.build_2d_sincos_position_embedding(
                    self.eval_spatial_size[1] // stride,
                    self.eval_spatial_size[0] // stride,
                    self.hidden_dim,
                    self.pe_temperature,
                )
                setattr(self, f"pos_embed{idx}", pos_embed)

    @staticmethod
    def build_2d_sincos_position_embedding(
        w: Union[int, float],
        h: Union[int, float],
        embed_dim: int = 256,
        temperature: float = 10000.0,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Generates a 2D sine-cosine position embedding matrix.
        Optimization: Directly construct tensors on the target device to entirely
        eliminate Host-to-Device CPU/GPU synchronization bottlenecks.
        """
        grid_w = torch.arange(int(w), dtype=torch.float32, device=device)
        grid_h = torch.arange(int(h), dtype=torch.float32, device=device)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing="ij")

        assert embed_dim % 4 == 0, "Embed dimension must be divisible by 4 for 2D sin-cos position embedding"

        pos_dim = embed_dim // 4

        # Calculate omega with high numerical stability on the specific device
        omega = torch.arange(pos_dim, dtype=torch.float32, device=device) / pos_dim
        omega = 1.0 / (temperature**omega)

        out_w = grid_w.flatten()[..., None] * omega[None, :]
        out_h = grid_h.flatten()[..., None] * omega[None, :]

        return torch.cat([out_w.sin(), out_w.cos(), out_h.sin(), out_h.cos()], dim=1).unsqueeze(0)

    def forward(self, feats: list[torch.Tensor]) -> list[torch.Tensor]:
        assert len(feats) == len(self.in_channels), "Number of input features must match in_channels"
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]

        # Internal Transformer Encoder augmentation
        if self.num_encoder_layers > 0:
            for i, enc_ind in enumerate(self.use_encoder_idx):
                h, w = proj_feats[enc_ind].shape[2:]
                # Flatten [B, C, H, W] to [B, HxW, C]
                src_flatten = proj_feats[enc_ind].flatten(2).permute(0, 2, 1)

                if self.training or self.eval_spatial_size is None:
                    # Optimization: Pass the target device explicitly to bypass CUDA sync block
                    pos_embed = self.build_2d_sincos_position_embedding(
                        w,
                        h,
                        self.hidden_dim,
                        self.pe_temperature,
                        device=src_flatten.device,
                    )
                else:
                    pos_embed = getattr(self, f"pos_embed{enc_ind}", None).to(src_flatten.device)

                memory: torch.Tensor = self.encoder[i](src_flatten, pos_embed=pos_embed)
                proj_feats[enc_ind] = memory.permute(0, 2, 1).reshape(-1, self.hidden_dim, h, w).contiguous()

        # Broadcasting and Fusion (Top-Down FPN)
        inner_outs = [proj_feats[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_heigh = inner_outs[0]
            feat_low = proj_feats[idx - 1]

            feat_heigh = self.lateral_convs[len(self.in_channels) - 1 - idx](feat_heigh)
            inner_outs[0] = feat_heigh

            # SAC Insight: Nearest interpolation is heavily utilized here.
            upsample_feat = F.interpolate(feat_heigh, scale_factor=2.0, mode="nearest")

            # Optimization: Use torch.cat over torch.concat
            inner_out = self.fpn_blocks[len(self.in_channels) - 1 - idx](torch.cat([upsample_feat, feat_low], dim=1))
            inner_outs.insert(0, inner_out)

        # Bottom-Up PAN
        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_height = inner_outs[idx + 1]

            downsample_feat = self.downsample_convs[idx](feat_low)
            out = self.pan_blocks[idx](torch.cat([downsample_feat, feat_height], dim=1))
            outs.append(out)

        return outs
