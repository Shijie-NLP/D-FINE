"""
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
Optimized for Layer-wise Ablation and Robust FiftyOne Visualization.
"""

import argparse
import os
import sys


os.environ["FIFTYONE_DATASET_ZOO_DIR"] = os.path.expanduser("~/Data/datasets/fiftyone")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import subprocess
import time

import fiftyone as fo
import fiftyone.core.labels as fol
import fiftyone.core.models as fom
import fiftyone.zoo as foz
import numpy as np
import torch
import torchvision.transforms as transforms
from fiftyone.types import FiftyOneDataset
from PIL import Image
from tqdm import tqdm

from src.core import YAMLConfig


def kill_existing_mongod():
    """
    Forcefully terminate hanging MongoDB processes from previous FiftyOne sessions.
    Crucial for avoiding port conflicts during iterative debugging.
    """
    try:
        result = subprocess.run(["ps", "aux"], stdout=subprocess.PIPE, text=True)
        processes = result.stdout.splitlines()

        for process in processes:
            if "mongod" in process and "--dbpath" in process:
                # Extract PID safely
                pid = int(process.split()[1])
                print(f"[System] Terminating zombie mongod process (PID: {pid})...")
                os.kill(pid, 9)
    except Exception as e:
        print(f"[Warning] Failed to terminate mongod gracefully: {e}")


kill_existing_mongod()

# COCO 91-category mapping
# fmt: off
LABEL_MAP = {
    1: "person", 2: "bicycle", 3: "car", 4: "motorbike", 5: "aeroplane", 6: "bus",
    7: "train", 8: "truck", 9: "boat", 10: "trafficlight", 11: "firehydrant",
    12: "streetsign", 13: "stopsign", 14: "parkingmeter", 15: "bench", 16: "bird",
    17: "cat", 18: "dog", 19: "horse", 20: "sheep", 21: "cow", 22: "elephant",
    23: "bear", 24: "zebra", 25: "giraffe", 26: "hat", 27: "backpack", 28: "umbrella",
    29: "shoe", 30: "eyeglasses", 31: "handbag", 32: "tie", 33: "suitcase",
    34: "frisbee", 35: "skis", 36: "snowboard", 37: "sportsball", 38: "kite",
    39: "baseballbat", 40: "baseballglove", 41: "skateboard", 42: "surfboard",
    43: "tennisracket", 44: "bottle", 45: "plate", 46: "wineglass", 47: "cup",
    48: "fork", 49: "knife", 50: "spoon", 51: "bowl", 52: "banana", 53: "apple",
    54: "sandwich", 55: "orange", 56: "broccoli", 57: "carrot", 58: "hotdog",
    59: "pizza", 60: "donut", 61: "cake", 62: "chair", 63: "sofa", 64: "pottedplant",
    65: "bed", 66: "mirror", 67: "diningtable", 68: "window", 69: "desk",
    70: "toilet", 71: "door", 72: "tv", 73: "laptop", 74: "mouse", 75: "remote",
    76: "keyboard", 77: "cellphone", 78: "microwave", 79: "oven", 80: "toaster",
    81: "sink", 82: "refrigerator", 83: "blender", 84: "book", 85: "clock",
    86: "vase", 87: "scissors", 88: "teddybear", 89: "hairdrier", 90: "toothbrush",
    91: "hairbrush",
}
# fmt: on


class DFineEvaluator(fom.Model):
    """
    FiftyOne Model Wrapper for D-FINE.
    Translates PyTorch outputs into FiftyOne's normalized coordinate space.
    """

    def __init__(self, cfg):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = cfg.model.eval().to(self.device)
        self.postprocessor = cfg.postprocessor.eval().to(self.device)

        # SAC NOTE: Direct resize alters aspect ratios.
        # For rigorous benchmark evaluation, consider implementing LetterBox padding here.
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((640, 640))])

    @property
    def media_type(self):
        return "image"

    @property
    def ragged_batches(self):
        return False

    @staticmethod
    def _convert_dense_predictions(bboxes, scores, img_w, img_h, labels=None):
        num_queries, num_classes = scores.shape
        k = min(3, num_classes)

        bboxes_w = bboxes[:, 2] - bboxes[:, 0]
        bboxes_h = bboxes[:, 3] - bboxes[:, 1]

        fo_bboxes = torch.stack(
            [bboxes[:, 0] / img_w, bboxes[:, 1] / img_h, bboxes_w / img_w, bboxes_h / img_h], dim=-1
        )  # shape: [num_queries, 4]

        if num_classes == 1:
            assert labels is not None
            topk_scores = scores  # [num_queries, 1]
            if torch.is_tensor(labels):
                # 确保维度是对齐的 [num_queries, 1]
                topk_labels = labels.unsqueeze(1) if labels.dim() == 1 else labels
            else:
                topk_labels = torch.tensor(labels, device=scores.device).unsqueeze(1)
        else:
            # 利用底层 C++/CUDA 完成 Top-K 提取
            topk_scores, topk_indices = torch.topk(scores, k, dim=-1)
            topk_labels = topk_indices + 1  # 加上 COCO 偏移量

        topk_scores_np = topk_scores.cpu().numpy()
        topk_labels_np = topk_labels.cpu().numpy()
        fo_bboxes_np = fo_bboxes.cpu().numpy()

        mask = topk_scores_np >= 1e-4
        valid_q_indices, valid_k_indices = np.where(mask)

        detections = []

        for q_idx, k_idx in zip(valid_q_indices, valid_k_indices):
            score = topk_scores_np[q_idx, k_idx]
            label_id = int(topk_labels_np[q_idx, k_idx])

            # fo_bboxes_np[q_idx] 是一个 shape 为 (4,) 的 numpy 数组，直接转 list
            bbox = fo_bboxes_np[q_idx].tolist()

            detection = fol.Detection(
                label=LABEL_MAP.get(label_id, "unknown"),
                bounding_box=bbox,
                confidence=float(score),
                query_id=str(q_idx),
            )
            detections.append(detection)

        return fol.Detections(detections=detections)

    def predict_all(self, images):
        """Batch inference for high-throughput visualization."""
        batch_size = len(images)
        image_tensors = []
        orig_sizes = []

        for img in images:
            # 动态提取每张图的原始尺寸，严格以 [W, H] 记录供 Postprocessor 使用
            h, w = img.shape[:2]
            orig_sizes.append([w, h])

            pil_img = Image.fromarray(img).convert("RGB")
            image_tensors.append(self.transform(pil_img))

        batch_tensor = torch.stack(image_tensors).to(self.device)
        orig_target_sizes = torch.tensor(orig_sizes, device=self.device)

        with torch.no_grad():
            outputs = self.model(batch_tensor)
            # predictions 是一个 List[Dict]，长度为 batch_size
            predictions = self.postprocessor(outputs, orig_target_sizes)

        # 遍历 Batch 中的每一张图片，重构多层字典
        batch_results = []
        for b_idx in tqdm(range(batch_size)):
            layer_predictions = {}
            pred_b = predictions[b_idx]
            orig_w, orig_h = orig_sizes[b_idx]

            # 1. 提取该图片的最终层
            layer_predictions["predictions_final"] = self._convert_dense_predictions(
                pred_b["boxes"],
                pred_b["scores"].unsqueeze(-1),
                orig_w,
                orig_h,
                labels=pred_b["labels"],
            )

            if "enc_aux_boxes" in pred_b and "enc_aux_scores" in pred_b:
                layer_predictions["predictions_enc"] = self._convert_dense_predictions(
                    pred_b["enc_aux_boxes"][0], pred_b["enc_aux_scores"][0], orig_w, orig_h
                )

            if "pre_boxes" in pred_b and "pre_scores" in pred_b:
                layer_predictions["predictions_pre"] = self._convert_dense_predictions(
                    pred_b["pre_boxes"], pred_b["pre_scores"], orig_w, orig_h
                )

            if "aux_boxes" in pred_b and "aux_scores" in pred_b:
                aux_boxes = pred_b["aux_boxes"]  # [num_layers, num_queries, 4]
                aux_scores = pred_b["aux_scores"]  # [num_layers, num_queries, num_classes]
                num_layers = aux_boxes.shape[0]

                for layer_idx in range(num_layers):
                    l_boxes = aux_boxes[layer_idx]
                    l_scores = aux_scores[layer_idx]

                    layer_predictions[f"predictions_layer_{layer_idx}"] = self._convert_dense_predictions(
                        l_boxes, l_scores, orig_w, orig_h
                    )

            batch_results.append(layer_predictions)

        return batch_results


def main(args):
    try:
        # Check if pre-computed views exist to save redundant inference time
        if os.path.exists("saved_predictions_view"):
            print("[Status] Loading cached predictions view from disk...")
            dataset = foz.load_zoo_dataset("coco-2017", split="validation", dataset_name="D-FINE")  # noqa
            dataset.persistent = True
            predictions_view = fo.Dataset.from_dir(
                dataset_dir="saved_predictions_view",
                dataset_type=FiftyOneDataset,  # noqa
            ).view()
            session = fo.launch_app(view=predictions_view, port=5151)

            session.wait()

        else:
            print("[Status] Initializing D-FINE Model and FiftyOne Dataset...")
            dataset = foz.load_zoo_dataset(
                "coco-2017",
                split="validation",
                dataset_name="D-FINE",  # noqa
                dataset_dir="saved_predictions_view",
            )
            dataset.persistent = True
            predictions_view = dataset.take(20, seed=51)

            # Load Model Configuration & Weights
            cfg = YAMLConfig(args.config, resume=args.resume)
            if "HGNetv2" in cfg.yaml_cfg:
                cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

            if args.resume:
                checkpoint = torch.load(args.resume, map_location="cpu")
                state = checkpoint["ema"]["module"] if "ema" in checkpoint else checkpoint["model"]
            else:
                raise ValueError("Checkpoint required: Currently only supporting resume to load model.state_dict.")

            cfg.model.load_state_dict(state)

            # Subset for rapid debugging (adjust limit as necessary for full validation)

            model = DFineEvaluator(cfg)

            print("[Inference] Executing single-pass multi-layer inference...")
            predictions_view.apply_model(model, batch_size=32, progress=True)

            # Persist the view for future fast-loading
            predictions_view.export(export_dir="saved_predictions_view", dataset_type=fo.types.FiftyOneDataset)

            session = fo.launch_app(view=predictions_view, port=5151)

            session.wait()

        print("[Status] Session is live. Press Ctrl+C to exit.")
        while True:
            time.sleep(1)

    except Exception as e:
        print(f"[Error] Fatal exception during execution: {e}")
    finally:
        print("[Status] Safely tearing down FiftyOne session...")
        if "session" in locals() and session is not None:
            session.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FiftyOne Visualization for D-FINE Layer-wise Output")
    parser.add_argument("--config", "-c", type=str, required=True, help="Path to YAML configuration file")
    parser.add_argument("--resume", "-r", type=str, required=True, help="Path to model checkpoint")
    args = parser.parse_args()

    main(args)
