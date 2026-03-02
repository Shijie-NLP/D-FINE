"""
Optimized for zero-copy threshold sweeping, deterministic state management,
and highly efficient batched tensor operations. All dead code eliminated.
"""

import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional, Union

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import seaborn as sns
import torch
from loguru import logger
from matplotlib.lines import Line2D
from torchvision.ops import box_iou


PALETTE = {
    "blue": "#4C72B0",
    "orange": "#DD8452",
    "green": "#55A868",
    "red": "#C44E52",
    "light_gray": "#F5F5F5",
    "grid": "#E0E0E0",
}

SIZE_COLORS = {
    "small": "#FF4444",  # Red
    "medium": "#FFA500",  # Orange
    "large": "#4488FF",  # Blue
}

SMALL_THR = 32**2  # area < 1024
LARGE_THR = 96**2  # area >= 9216


def _apply_base_style(
    ax: plt.Axes, title: Optional[str] = None, xlabel: Optional[str] = None, ylabel: Optional[str] = None
) -> None:
    """Unified axis styling for high-quality plots."""
    ax.set_facecolor(PALETTE["light_gray"])
    ax.grid(color=PALETTE["grid"], linewidth=0.8, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#CCCCCC")
    if title:
        ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=11)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=11)


class Validator:
    """
    Evaluation Engine computing precision, recall, F1, and class-wise IoUs.
    Optimized to bypass deepcopies during dynamic threshold sweeping.
    """

    def __init__(
        self,
        gt: list[dict[str, torch.Tensor]],
        preds: list[dict[str, torch.Tensor]],
        conf_thresh: float = 0.5,
        iou_thresh: float = 0.5,
    ) -> None:
        """
        Format example:
        gt = [{'labels': tensor([0]), 'boxes': tensor([[561.0, 297.0, 661.0, 359.0]])}, ...]
        bboxes are in absolute [x1, y1, x2, y2] format.
        """
        self.gt = gt
        self.preds = preds
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.thresholds = np.arange(0.2, 1.0, 0.05)

        self.conf_matrix: Optional[np.ndarray] = None
        self.class_to_idx: dict[int, int] = {}
        self.metrics_per_class: dict[int, Any] = {}

    def compute_metrics(self, extended: bool = False) -> dict[str, Any]:
        """Calculates default metrics using the base confidence threshold."""
        metrics, conf_matrix, class_to_idx = self._compute_main_metrics(self.conf_thresh)

        # Safely preserve states for plotting
        self.conf_matrix = conf_matrix
        self.class_to_idx = class_to_idx

        if not extended:
            metrics.pop("extended_metrics", None)

        return metrics

    def _compute_main_metrics(self, current_conf_thresh: float) -> tuple[dict[str, Any], np.ndarray, dict[int, int]]:
        """
        Pure function computing core evaluation metrics without mutating class states.
        Avoids deepcopy completely by applying boolean masks dynamically.
        """
        metrics_per_class, conf_matrix, class_to_idx = self._compute_metrics_and_confusion_matrix(current_conf_thresh)

        tps, fps, fns = 0, 0, 0
        ious: list[float] = []
        extended_metrics: dict[str, float] = {}

        for key, value in metrics_per_class.items():
            tps += value["TPs"]
            fps += value["FPs"]
            fns += value["FNs"]
            ious.extend(value["IoUs"])

            extended_metrics[f"precision_{key}"] = (
                value["TPs"] / (value["TPs"] + value["FPs"]) if value["TPs"] + value["FPs"] > 0 else 0.0
            )
            extended_metrics[f"recall_{key}"] = (
                value["TPs"] / (value["TPs"] + value["FNs"]) if value["TPs"] + value["FNs"] > 0 else 0.0
            )
            extended_metrics[f"iou_{key}"] = np.mean(value["IoUs"]) if value["IoUs"] else 0.0

        precision = tps / (tps + fps) if (tps + fps) > 0 else 0.0
        recall = tps / (tps + fns) if (tps + fns) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        iou = float(np.mean(ious)) if ious else 0.0

        results = {
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "iou": iou,
            "TPs": tps,
            "FPs": fps,
            "FNs": fns,
            "extended_metrics": extended_metrics,
        }

        return results, conf_matrix, class_to_idx

    def _compute_metrics_and_confusion_matrix(
        self, current_conf_thresh: float
    ) -> tuple[dict[int, Any], np.ndarray, dict[int, int]]:
        metrics_per_class = defaultdict(lambda: {"TPs": 0, "FPs": 0, "FNs": 0, "IoUs": []})

        all_classes = set()
        for pred in self.preds:
            all_classes.update(pred["labels"].tolist())
        for gt in self.gt:
            all_classes.update(gt["labels"].tolist())

        all_classes = sorted(all_classes)
        class_to_idx = {cls_id: idx for idx, cls_id in enumerate(all_classes)}
        n_classes = len(all_classes)
        conf_matrix = np.zeros((n_classes + 1, n_classes + 1), dtype=int)  # +1 for background class

        for pred, gt in zip(self.preds, self.gt):
            # Optimization: Dynamic threshold masking replacing expensive deepcopies
            keep_idxs = pred["scores"] >= current_conf_thresh
            pred_boxes = pred["boxes"][keep_idxs]
            pred_labels = pred["labels"][keep_idxs]

            gt_boxes = gt["boxes"]
            gt_labels = gt["labels"]

            n_preds = len(pred_boxes)
            n_gts = len(gt_boxes)

            if n_preds == 0 and n_gts == 0:
                continue

            ious = box_iou(pred_boxes, gt_boxes) if n_preds > 0 and n_gts > 0 else torch.tensor([])

            matched_pred_indices = set()
            matched_gt_indices = set()

            if ious.numel() > 0:
                ious_mask = ious >= self.iou_thresh
                pred_indices, gt_indices = torch.nonzero(ious_mask, as_tuple=True)
                iou_values = ious[pred_indices, gt_indices]

                sorted_indices = torch.argsort(-iou_values)
                pred_indices = pred_indices[sorted_indices]
                gt_indices = gt_indices[sorted_indices]
                iou_values = iou_values[sorted_indices]

                for pred_idx, gt_idx, iou in zip(pred_indices, gt_indices, iou_values):
                    if pred_idx.item() in matched_pred_indices or gt_idx.item() in matched_gt_indices:
                        continue
                    matched_pred_indices.add(pred_idx.item())
                    matched_gt_indices.add(gt_idx.item())

                    pred_label = pred_labels[pred_idx].item()
                    gt_label = gt_labels[gt_idx].item()

                    pred_cls_idx = class_to_idx[pred_label]
                    gt_cls_idx = class_to_idx[gt_label]

                    conf_matrix[gt_cls_idx, pred_cls_idx] += 1

                    if pred_label == gt_label:
                        metrics_per_class[gt_label]["TPs"] += 1
                        metrics_per_class[gt_label]["IoUs"].append(iou.item())
                    else:
                        metrics_per_class[gt_label]["FNs"] += 1
                        metrics_per_class[pred_label]["FPs"] += 1
                        metrics_per_class[gt_label]["IoUs"].append(0)
                        metrics_per_class[pred_label]["IoUs"].append(0)

            # Unmatched predictions (False Positives)
            unmatched_pred_indices = set(range(n_preds)) - matched_pred_indices
            for pred_idx in unmatched_pred_indices:
                pred_label = pred_labels[pred_idx].item()
                pred_cls_idx = class_to_idx[pred_label]
                conf_matrix[n_classes, pred_cls_idx] += 1
                metrics_per_class[pred_label]["FPs"] += 1
                metrics_per_class[pred_label]["IoUs"].append(0)

            # Unmatched ground truths (False Negatives)
            unmatched_gt_indices = set(range(n_gts)) - matched_gt_indices
            for gt_idx in unmatched_gt_indices:
                gt_label = gt_labels[gt_idx].item()
                gt_cls_idx = class_to_idx[gt_label]
                conf_matrix[gt_cls_idx, n_classes] += 1
                metrics_per_class[gt_label]["FNs"] += 1
                metrics_per_class[gt_label]["IoUs"].append(0)

        return dict(metrics_per_class), conf_matrix, class_to_idx

    def save_plots(self, path_to_save: str) -> None:
        path_to_save = Path(path_to_save)
        path_to_save.mkdir(parents=True, exist_ok=True)

        # ── 1. Confusion Matrix ─────────────────────────────────────────────────
        if self.conf_matrix is not None:
            class_labels = [str(cls_id) for cls_id in self.class_to_idx.keys()] + ["background"]
            n = len(class_labels)

            cm = self.conf_matrix.astype(float)
            cm_norm = cm / (cm.sum(axis=1, keepdims=True) + 1e-9)

            fig, axes = plt.subplots(
                1,
                2,
                figsize=(max(10, n * 1.2) * 2, max(8, n * 1.0)),
                constrained_layout=True,
            )
            fig.suptitle("Confusion Matrix", fontsize=15, fontweight="bold", y=1.02)

            for ax, data, fmt, subtitle in zip(
                axes,
                [cm.astype(int), cm_norm],
                ["d", ".2f"],
                ["Counts", "Row-normalized (Recall per class)"],
            ):
                sns.heatmap(
                    data,
                    ax=ax,
                    annot=True,
                    fmt=fmt,
                    cmap="Blues",
                    linewidths=0.4,
                    linecolor="#DDDDDD",
                    xticklabels=class_labels,
                    yticklabels=class_labels,
                    cbar_kws={"shrink": 0.8, "label": "Count" if fmt == "d" else "Ratio"},
                    annot_kws={"size": max(7, 11 - n // 4)},
                    vmin=0,
                    vmax=(None if fmt == "d" else 1.0),
                )
                ax.set_title(subtitle, fontsize=11, pad=8)
                ax.set_xlabel("Predicted label", fontsize=10)
                ax.set_ylabel("True label", fontsize=10)
                ax.tick_params(axis="x", rotation=45, labelsize=9)
                ax.tick_params(axis="y", rotation=0, labelsize=9)

            fig.savefig(path_to_save / "confusion_matrix.png", dpi=150, bbox_inches="tight")
            plt.close(fig)

        # ── 2. Precision / Recall / F1 Curves ───────────────────────────────────
        thresholds = self.thresholds
        precisions, recalls, f1_scores = [], [], []

        for threshold in thresholds:
            # Optimization: Pure function call, zero-copy evaluation
            metrics, _, _ = self._compute_main_metrics(threshold)
            precisions.append(metrics["precision"])
            recalls.append(metrics["recall"])
            f1_scores.append(metrics["f1"])

        best_idx = len(f1_scores) - np.argmax(f1_scores[::-1]) - 1
        best_threshold = thresholds[best_idx]
        best_f1 = f1_scores[best_idx]
        best_precision = precisions[best_idx]
        best_recall = recalls[best_idx]

        fig, (ax_prf, ax_pr) = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
        fig.suptitle("Detection Metrics vs Confidence Threshold", fontsize=14, fontweight="bold")

        _apply_base_style(
            ax_prf, title="Precision · Recall · F1 vs Threshold", xlabel="Confidence Threshold", ylabel="Score"
        )

        ax_prf.plot(
            thresholds,
            precisions,
            color=PALETTE["blue"],
            linewidth=2.2,
            marker="o",
            markersize=4,
            label="Precision",
            zorder=3,
        )
        ax_prf.plot(
            thresholds,
            recalls,
            color=PALETTE["orange"],
            linewidth=2.2,
            marker="s",
            markersize=4,
            label="Recall",
            zorder=3,
        )
        ax_prf.plot(
            thresholds,
            f1_scores,
            color=PALETTE["green"],
            linewidth=2.5,
            marker="^",
            markersize=5,
            label="F1 Score",
            zorder=3,
        )

        ax_prf.fill_between(thresholds, precisions, recalls, alpha=0.08, color=PALETTE["blue"])

        ax_prf.axvline(
            best_threshold,
            color=PALETTE["red"],
            linewidth=1.5,
            linestyle="--",
            zorder=2,
            label=f"Best threshold = {best_threshold:.2f}",
        )
        ax_prf.scatter([best_threshold], [best_f1], color=PALETTE["red"], zorder=5, s=80, marker="*")
        ax_prf.annotate(
            f"F1={best_f1:.3f}\nP={best_precision:.3f}\nR={best_recall:.3f}",
            xy=(best_threshold, best_f1),
            xytext=(best_threshold + (thresholds[-1] - thresholds[0]) * 0.05, best_f1 - 0.12),
            fontsize=8.5,
            color=PALETTE["red"],
            arrowprops=dict(arrowstyle="->", color=PALETTE["red"], lw=1.2),
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=PALETTE["red"], alpha=0.85),
        )

        ax_prf.set_ylim(-0.02, 1.05)
        ax_prf.xaxis.set_major_locator(mticker.MultipleLocator(0.1))
        ax_prf.yaxis.set_major_locator(mticker.MultipleLocator(0.1))
        ax_prf.legend(fontsize=9, framealpha=0.9)

        _apply_base_style(ax_pr, title="Precision–Recall Curve", xlabel="Recall", ylabel="Precision")

        from matplotlib.collections import LineCollection

        points = np.array([recalls, precisions]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(
            segments, cmap="RdYlGn", norm=plt.Normalize(thresholds.min(), thresholds.max()), linewidth=2.5, zorder=3
        )
        lc.set_array(thresholds[:-1])
        ax_pr.add_collection(lc)
        cbar = fig.colorbar(lc, ax=ax_pr, shrink=0.85)
        cbar.set_label("Threshold", fontsize=9)

        ax_pr.scatter(
            [best_recall],
            [best_precision],
            color=PALETTE["red"],
            zorder=5,
            s=100,
            marker="*",
            label=f"Best thr={best_threshold:.2f}",
        )
        ax_pr.set_xlim(-0.02, 1.05)
        ax_pr.set_ylim(-0.02, 1.05)
        ax_pr.xaxis.set_major_locator(mticker.MultipleLocator(0.1))
        ax_pr.yaxis.set_major_locator(mticker.MultipleLocator(0.1))
        ax_pr.legend(fontsize=9, framealpha=0.9)

        fig.savefig(path_to_save / "metrics_vs_threshold.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        logger.info(
            f"Best Threshold: {round(best_threshold, 2)} | "
            f"F1: {round(best_f1, 3)} | Precision: {round(best_precision, 3)} | Recall: {round(best_recall, 3)}"
        )

    def visualize_queries(self, output_dir: str = "vis_preds", max_images: Optional[int] = None) -> None:
        """
        Renders visualizations per image:
          - GT bounding box: Dashed rectangles, color-coded by size.
          - Query centers: Solid circular markers, color-coded by size.
        """
        os.makedirs(output_dir, exist_ok=True)
        n = len(self.gt) if max_images is None else min(max_images, len(self.gt))

        for i in range(n):
            gt_item = self.gt[i]
            pred_item = self.preds[i]

            image_path = gt_item.get("image_path", None)
            image_size = gt_item.get("image_size", None)

            img = self._load_image(image_path, image_size)

            gt_boxes = gt_item["boxes"]
            query_boxes = pred_item.get("init_boxes", None)

            fig, ax = self._draw_single(img, gt_boxes, query_boxes, image_path)

            if image_path:
                stem = os.path.splitext(os.path.basename(image_path))[0]
            else:
                stem = f"image_{i:04d}"

            save_path = os.path.join(output_dir, f"{stem}.png")
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"[{i + 1}/{n}] saved → {save_path}")

    def _draw_single(
        self,
        img: np.ndarray,
        gt_boxes: Optional[torch.Tensor],
        query_boxes: Optional[torch.Tensor],
        image_path: Optional[str] = None,
    ) -> tuple[plt.Figure, plt.Axes]:
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        ax.imshow(img)
        ax.axis("off")

        title = os.path.basename(image_path) if image_path else "Query Visualization"
        ax.set_title(title, fontsize=10, fontweight="bold", pad=6)

        if gt_boxes is not None and len(gt_boxes) > 0:
            boxes_np = self._to_numpy(gt_boxes)
            for x1, y1, x2, y2 in boxes_np:
                cat = _get_size_category(x1, y1, x2, y2)
                color = SIZE_COLORS[cat]
                rect = patches.Rectangle(
                    (x1, y1),
                    x2 - x1,
                    y2 - y1,
                    linewidth=2,
                    edgecolor=color,
                    facecolor="none",
                    linestyle="--",
                    alpha=0.95,
                )
                ax.add_patch(rect)

        if query_boxes is not None and len(query_boxes) > 0:
            qboxes_np = self._to_numpy(query_boxes)
            for x1, y1, x2, y2 in qboxes_np:
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                cat = _get_size_category(x1, y1, x2, y2)
                color = SIZE_COLORS[cat]
                ax.plot(
                    cx,
                    cy,
                    "o",
                    color=color,
                    markersize=5,
                    markeredgecolor="white",
                    markeredgewidth=0.6,
                    alpha=0.85,
                )

        legend_handles = [
            patches.Patch(
                facecolor="none",
                edgecolor=SIZE_COLORS["large"],
                linewidth=2,
                linestyle="--",
                label="GT Box  – Large  (≥96²)",
            ),
            patches.Patch(
                facecolor="none",
                edgecolor=SIZE_COLORS["medium"],
                linewidth=2,
                linestyle="--",
                label="GT Box  – Medium (32²–96²)",
            ),
            patches.Patch(
                facecolor="none",
                edgecolor=SIZE_COLORS["small"],
                linewidth=2,
                linestyle="--",
                label="GT Box  – Small  (<32²)",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=SIZE_COLORS["large"],
                markersize=8,
                markeredgecolor="white",
                label="Query – Large",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=SIZE_COLORS["medium"],
                markersize=8,
                markeredgecolor="white",
                label="Query – Medium",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=SIZE_COLORS["small"],
                markersize=8,
                markeredgecolor="white",
                label="Query – Small",
            ),
        ]
        ax.legend(handles=legend_handles, loc="upper right", fontsize=8, framealpha=0.75, edgecolor="gray")

        fig.tight_layout()
        return fig, ax

    @staticmethod
    def _to_numpy(t: Union[torch.Tensor, np.ndarray, list]) -> np.ndarray:
        """Unifies Tensor/ndarray/list conversion to a standard (N, 4) numpy array."""
        if isinstance(t, torch.Tensor):
            return t.detach().cpu().float().numpy()
        return np.array(t, dtype=np.float32)

    @staticmethod
    def _load_image(image_path: Optional[str], image_size: Any) -> np.ndarray:
        """Preferentially load the original image; fallback to a solid gray background."""
        if image_path and os.path.exists(image_path):
            bgr = cv2.imread(image_path)
            if bgr is not None:
                return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        if image_size is not None:
            if isinstance(image_size, torch.Tensor):
                H, W = int(image_size[0].item()), int(image_size[1].item())
            else:
                H, W = int(image_size[0]), int(image_size[1])
        else:
            H, W = 512, 512

        return np.full((H, W, 3), 60, dtype=np.uint8)


def scale_boxes(boxes: torch.Tensor, orig_shape: tuple[int, int], resized_shape: tuple[int, int]) -> torch.Tensor:
    """
    Scales bounding boxes securely avoiding in-place modification warnings.
    Boxes expected in [x1, y1, x2, y2] format.
    """
    boxes = boxes.clone()
    scale_x = orig_shape[1] / resized_shape[1]
    scale_y = orig_shape[0] / resized_shape[0]

    # Vectorized stride-based assignment
    boxes[:, 0::2] *= scale_x
    boxes[:, 1::2] *= scale_y

    return boxes


def _get_size_category(x1: float, y1: float, x2: float, y2: float) -> str:
    """Categorizes bounding box sizes strictly according to COCO standards."""
    area = (x2 - x1) * (y2 - y1)
    if area < SMALL_THR:
        return "small"
    elif area < LARGE_THR:
        return "medium"
    else:
        return "large"
