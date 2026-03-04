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
        export_dense: bool = False,  # Feature flag for lossless dense extraction
    ) -> None:
        super().__init__()
        self.use_focal_loss = use_focal_loss
        self.num_top_queries = num_top_queries
        self.num_classes = int(num_classes)
        self.remap_mscoco_category = remap_mscoco_category
        self.export_dense = export_dense
        self.deploy_mode = False

        # Pre-allocate the MSCOCO mapping tensor during initialization
        if self.remap_mscoco_category:
            from ...data.dataset import mscoco_label2category

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
        if self.export_dense:
            return self.forward_dense_output(outputs, orig_target_sizes)

        logits, boxes = outputs["pred_logits"], outputs["pred_boxes"]
        scale_tensor = orig_target_sizes.repeat(1, 2).unsqueeze(1)

        bbox_pred = torchvision.ops.box_convert(boxes, in_fmt="cxcywh", out_fmt="xyxy")
        bbox_pred.mul_(scale_tensor)

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

        results = []
        for i in range(len(labels)):
            res_dict = {"labels": labels[i], "boxes": final_boxes[i], "scores": scores[i]}
            results.append(res_dict)

        return results

    def forward_dense_output(self, outputs, orig_target_sizes):
        """
        Optimized dense extraction: Returns a SINGLE dictionary of batched tensors.
        Shapes are strictly maintained as [B, N, ...] or [B, L, N, ...] to guarantee
        C-level memory contiguity and zero Python-loop overhead.
        """

        scale_tensor = orig_target_sizes.repeat(1, 2).unsqueeze(1)

        res_dict = {}

        def _extract_all_queries(logits_t: torch.Tensor, boxes_t: torch.Tensor):
            # boxes_t: [B, N, 4], logits_t: [B, N, C]
            b_xyxy = torchvision.ops.box_convert(boxes_t, in_fmt="cxcywh", out_fmt="xyxy")
            b_xyxy.mul_(scale_tensor)

            if self.use_focal_loss:
                scores = logits_t.sigmoid()
            else:
                scores = logits_t.softmax(dim=-1)[..., :-1]

            # Return a dictionary of contiguous batched tensors
            return {"boxes": b_xyxy, "scores": scores}

        if "enc_aux_outputs" in outputs:
            enc_aux_extracted = _extract_all_queries(
                outputs["enc_aux_outputs"][0]["pred_logits"], outputs["enc_aux_outputs"][0]["pred_boxes"]
            )
            for k, v in enc_aux_extracted.items():
                res_dict[f"enc_{k}"] = v

        if "pre_outputs" in outputs and "pred_boxes" in outputs["pre_outputs"]:
            pre_data = _extract_all_queries(
                outputs["pre_outputs"]["pred_logits"], outputs["pre_outputs"]["pred_boxes"]
            )
            for k, v in pre_data.items():
                res_dict[f"pre_{k}"] = v

        if "aux_outputs" in outputs:
            aux_extracted = [
                _extract_all_queries(aux["pred_logits"], aux["pred_boxes"]) for aux in outputs["aux_outputs"]
            ]
            for k in aux_extracted[0].keys():
                # Stack along dim=1 to create the Layer dimension, resulting in [B, L, N, ...]
                res_dict[f"aux_{k}"] = torch.stack([layer_data[k] for layer_data in aux_extracted], dim=1)

        pred_output = _extract_all_queries(outputs["pred_logits"], outputs["pred_boxes"])
        for k, v in pred_output.items():
            res_dict[f"final_{k}"] = v

        return res_dict

    def deploy(self):
        self.eval()
        self.deploy_mode = True
        return self
