"""
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.

Optimized for strict static typing, structural re-parameterization safety,
and clear architectural macro-topology.
"""

from typing import Any, Optional

import torch
import torch.nn as nn

from ...core import register


__all__ = ["DFINE"]


@register()
class DFINE(nn.Module):
    """
    Top-level D-FINE architecture assembling the backbone, encoder, and decoder.
    Serves as the central dataflow orchestrator for the object detection pipeline.
    """

    __inject__ = ["backbone", "encoder", "decoder"]

    def __init__(
        self,
        backbone: nn.Module,
        encoder: nn.Module,
        decoder: nn.Module,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x: torch.Tensor, targets: Optional[list[dict[str, Any]]] = None) -> dict[str, torch.Tensor]:
        """
        Executes the macro-forward pass chaining feature extraction, multi-scale
        encoding, and fine-grained distribution refinement decoding.

        Args:
            x (torch.Tensor): Batched input image tensors.
            targets (Optional[List[Dict]]): Ground truth annotations required
                                            during training for denoising / assignment.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing prediction logits,
                                     bounding boxes, and auxiliary outputs.
        """
        # Feature extraction via deep CNN/Transformer backbone
        x = self.backbone(x)

        # Multi-scale feature fusion and semantic enhancement
        x = self.encoder(x)

        # Bounding box distribution regression and refinement
        # SAC Note: If injecting Continuous Density-Manifold (CD-MQA) priors,
        # intercept and pass the density embeddings here.
        x = self.decoder(x, targets)

        return x

    def deploy(self) -> "DFINE":
        """
        Switches the entire topology to evaluation mode and recursively applies
        structural re-parameterization (e.g., fusing Batch Normalization layers
        into Convolutions) to maximize inference speed and minimize MAC overhead.
        """
        self.eval()

        for m in self.modules():
            if hasattr(m, "convert_to_deploy"):
                # Dynamically call the module-specific deploy conversion routine
                m.convert_to_deploy()

        return self
