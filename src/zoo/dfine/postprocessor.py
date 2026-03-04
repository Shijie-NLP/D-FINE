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
        logits, boxes = outputs["pred_logits"], outputs["pred_boxes"]

        # Base scaling tensor creation
        scale_tensor = orig_target_sizes.repeat(1, 2).unsqueeze(1)

        # ---------------------------------------------------------------------
        # BRANCH 1: DENSE EXTRACTION (For Visualization & Manifold Analysis)
        # ---------------------------------------------------------------------
        if self.export_dense:

            def _extract_all_queries(logits_t: torch.Tensor, boxes_t: torch.Tensor):
                b_xyxy = torchvision.ops.box_convert(boxes_t, in_fmt="cxcywh", out_fmt="xyxy")
                b_xyxy.mul_(scale_tensor)

                if self.use_focal_loss:
                    full_scores = logits_t.sigmoid()
                else:
                    full_scores = logits_t.softmax(dim=-1)[..., :-1]

                max_scores, labels_idx = full_scores.max(dim=-1)

                if self.remap_mscoco_category:
                    labels_idx = self.mscoco_mapping[labels_idx]

                return b_xyxy, labels_idx, max_scores, full_scores

            # Extract for all layers
            final_boxes, final_labels, final_scores, final_full = _extract_all_queries(logits, boxes)

            enc_aux_data = pre_data = aux_data = None

            if "enc_aux_outputs" in outputs:
                enc_aux_data = [
                    _extract_all_queries(aux["pred_logits"], aux["pred_boxes"]) for aux in outputs["enc_aux_outputs"]
                ]

            if "pre_outputs" in outputs and "pred_boxes" in outputs["pre_outputs"]:
                pre_data = _extract_all_queries(
                    outputs["pre_outputs"]["pred_logits"], outputs["pre_outputs"]["pred_boxes"]
                )

            if "aux_outputs" in outputs:
                aux_data = [
                    _extract_all_queries(aux["pred_logits"], aux["pred_boxes"]) for aux in outputs["aux_outputs"]
                ]

            results = []
            for i in range(len(final_labels)):
                res_dict = {
                    "labels": final_labels[i],
                    "boxes": final_boxes[i],
                    "scores": final_scores[i],
                    "full_scores": final_full[i],
                }

                if enc_aux_data:
                    res_dict["enc_aux_boxes"] = torch.stack([d[0][i] for d in enc_aux_data], dim=0)
                    res_dict["enc_aux_labels"] = torch.stack([d[1][i] for d in enc_aux_data], dim=0)
                    res_dict["enc_aux_scores"] = torch.stack([d[2][i] for d in enc_aux_data], dim=0)
                    res_dict["enc_aux_full_scores"] = torch.stack([d[3][i] for d in enc_aux_data], dim=0)

                if pre_data:
                    res_dict["pre_boxes"] = pre_data[0][i]
                    res_dict["pre_labels"] = pre_data[1][i]
                    res_dict["pre_scores"] = pre_data[2][i]
                    res_dict["pre_full_scores"] = pre_data[3][i]

                if aux_data:
                    res_dict["aux_boxes"] = torch.stack([d[0][i] for d in aux_data], dim=0)
                    res_dict["aux_labels"] = torch.stack([d[1][i] for d in aux_data], dim=0)
                    res_dict["aux_scores"] = torch.stack([d[2][i] for d in aux_data], dim=0)
                    res_dict["aux_full_scores"] = torch.stack([d[3][i] for d in aux_data], dim=0)

                results.append(res_dict)
            return results

        # ---------------------------------------------------------------------
        # BRANCH 2: STANDARD TOP-K (For Evaluation, mAP, and Deployment)
        # ---------------------------------------------------------------------
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

    def deploy(self):
        self.eval()
        self.deploy_mode = True
        return self
