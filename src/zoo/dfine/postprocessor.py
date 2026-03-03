"""
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.

Refactored for zero-allocation tensor operations, vectorized mapping,
and ONNX/TensorRT modern compliance.
"""

import torch
import torch.nn as nn
import torchvision

from ...core import register


__all__ = ["DFINEPostProcessor"]


@register()
class DFINEPostProcessor(nn.Module):
    __share__ = [
        "num_classes",
        "use_focal_loss",
        "num_top_queries",
        "remap_mscoco_category",
    ]

    def __init__(
        self,
        num_classes: int = 80,
        use_focal_loss: bool = True,
        num_top_queries: int = 300,
        remap_mscoco_category: bool = False,
    ) -> None:
        super().__init__()
        self.use_focal_loss = use_focal_loss
        self.num_top_queries = num_top_queries
        self.num_classes = int(num_classes)
        self.remap_mscoco_category = remap_mscoco_category
        self.deploy_mode = False

        # [SAC Note]: Pre-allocate the MSCOCO mapping tensor during initialization
        # to avoid Host-Device syncs and python-loop bottlenecks during the forward pass.
        if self.remap_mscoco_category:
            from ...data.dataset import mscoco_label2category

            # Determine maximum label index to size the buffer appropriately
            if isinstance(mscoco_label2category, dict):
                max_label = max(mscoco_label2category.keys())
            else:
                max_label = len(mscoco_label2category) - 1

            mapping = torch.zeros(max_label + 1, dtype=torch.long)

            iterator = (
                mscoco_label2category.items()
                if isinstance(mscoco_label2category, dict)
                else enumerate(mscoco_label2category)
            )
            for k, v in iterator:
                mapping[k] = v

            self.register_buffer("mscoco_mapping", mapping)

    def extra_repr(self) -> str:
        return f"use_focal_loss={self.use_focal_loss}, num_classes={self.num_classes}, num_top_queries={self.num_top_queries}"

    def forward(self, outputs: dict[str, torch.Tensor], orig_target_sizes: torch.Tensor):
        logits, boxes = outputs["pred_logits"], outputs["pred_boxes"]

        # 1. Base scaling tensor creation
        scale_tensor = orig_target_sizes.repeat(1, 2).unsqueeze(1)

        # Helper function to decode and scale boxes efficiently in-place
        def _decode_and_scale(b: torch.Tensor) -> torch.Tensor:
            b_xyxy = torchvision.ops.box_convert(b, in_fmt="cxcywh", out_fmt="xyxy")
            b_xyxy.mul_(scale_tensor)
            return b_xyxy

        # 2. Process final predictions
        bbox_pred = _decode_and_scale(boxes)

        if self.use_focal_loss:
            scores = logits.sigmoid()
            scores, index = torch.topk(scores.flatten(1), self.num_top_queries, dim=-1)

            labels = index % self.num_classes
            index = torch.div(index, self.num_classes, rounding_mode="floor")

            final_boxes = bbox_pred.gather(dim=1, index=index.unsqueeze(-1).expand(-1, -1, bbox_pred.shape[-1]))

        else:
            scores = logits.softmax(dim=-1)[..., :-1]
            scores, labels = scores.max(dim=-1)

            if scores.shape[1] > self.num_top_queries:
                scores, index = torch.topk(scores, self.num_top_queries, dim=-1)
                labels = torch.gather(labels, dim=1, index=index)

                final_boxes = torch.gather(
                    bbox_pred, dim=1, index=index.unsqueeze(-1).expand(-1, -1, bbox_pred.shape[-1])
                )
            else:
                final_boxes = bbox_pred

        if self.deploy_mode:
            return labels, final_boxes, scores

        if self.remap_mscoco_category:
            labels = self.mscoco_mapping[labels]

        # 3. Process intermediate trajectory boxes for visualization
        enc_aux_boxes = None
        if "enc_aux_outputs" in outputs:
            # Resulting shape: [num_layers, batch_size, num_queries, 4]
            enc_aux_boxes = torch.stack(
                [_decode_and_scale(aux["pred_boxes"]) for aux in outputs["enc_aux_outputs"]], dim=0
            )
            enc_aux_scores = torch.stack([aux["pred_logits"].sigmoid() for aux in outputs["enc_aux_outputs"]], dim=0)

        pre_boxes = None
        if "pre_outputs" in outputs and "pred_boxes" in outputs["pre_outputs"]:
            # Resulting shape: [batch_size, num_queries, 4]
            pre_boxes = _decode_and_scale(outputs["pre_outputs"]["pred_boxes"])
            pre_scores = outputs["pre_outputs"]["pred_logits"].sigmoid()

        aux_boxes = None
        if "aux_outputs" in outputs:
            aux_boxes = torch.stack([_decode_and_scale(aux["pred_boxes"]) for aux in outputs["aux_outputs"]], dim=0)
            aux_scores = torch.stack([aux["pred_logits"].sigmoid() for aux in outputs["aux_outputs"]], dim=0)

        # 4. Pack everything into the batch-wise results dictionary
        results = []
        for i in range(len(labels)):
            res_dict = {"labels": labels[i], "boxes": final_boxes[i], "scores": scores[i]}

            # Inject auxiliary trajectory data aligned by batch index
            if enc_aux_boxes is not None:
                res_dict["enc_aux_boxes"] = enc_aux_boxes[:, i, :, :]  # [num_layers, num_queries, 4]
                res_dict["enc_aux_scores"] = enc_aux_scores[:, i, :]
            if pre_boxes is not None:
                res_dict["pre_boxes"] = pre_boxes[i]  # [num_queries, 4]
                res_dict["pre_scores"] = pre_scores[i]
            if aux_boxes is not None:
                res_dict["aux_boxes"] = aux_boxes[:, i, :, :]
                res_dict["aux_scores"] = aux_scores[:, i, :]

            results.append(res_dict)

        return results

    def deploy(self):
        self.eval()
        self.deploy_mode = True
        return self
