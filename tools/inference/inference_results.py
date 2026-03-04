"""
Refactored for High-Efficiency Layer-wise Spatial Snapshots.
- Fixed catastrophic list indexing bugs.
- Vectorized label mapping (Zero Python-loop overhead).
- Bypassed default postprocessor to extract independent layer-wise Top-K.
- Robust PyTorch Native Serialization (.pt).
"""

import argparse
import os
import pickle
import sys

import numpy as np
import torch
from tqdm import tqdm


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.core import YAMLConfig
from src.solver.validator import scale_boxes


def strip_to_numpy_dense(data, parent_key=""):
    """
    1. Retains ALL dense queries (No Top-K pruning).
    2. Strips all PyTorch metadata by converting to raw, contiguous NumPy arrays.
    3. Enforces strict Precision Asymmetry (FP32 for boxes, FP16 for scores/logits).
    """
    if isinstance(data, dict):
        cleaned_dict = {}
        for k, v in data.items():
            if isinstance(v, torch.Tensor):
                # Detach and convert directly to NumPy, destroying the PyTorch graph/storage links
                np_array = v.detach().cpu().numpy()

                # Apply Precision Asymmetry to minimize the dense footprint
                if np_array.dtype == np.float32 or np_array.dtype == np.float64:
                    if "box" in k.lower() or "box" in parent_key.lower():
                        cleaned_dict[k] = np_array.astype(np.float32)
                    else:
                        cleaned_dict[k] = np_array.astype(np.float16)
                else:
                    cleaned_dict[k] = np_array
            elif isinstance(v, (dict, list)):
                cleaned_dict[k] = strip_to_numpy_dense(v, k)
            else:
                cleaned_dict[k] = v
        return cleaned_dict

    elif isinstance(data, list):
        return [strip_to_numpy_dense(item, parent_key) for item in data]
    else:
        return data


def main(args):
    cfg = YAMLConfig(
        args.config,
        resume=args.resume,
        val_dataloader={"total_batch_size": 32, "dataset": {"num_samples": args.num_samples}},
    )

    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        state = checkpoint["ema"]["module"] if "ema" in checkpoint else checkpoint["model"]
    else:
        raise ValueError("Checkpoint required: Currently only supporting resume to load model.state_dict.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg.model.load_state_dict(state)
    model = cfg.model.eval().to(device)

    postprocessor = cfg.postprocessor.eval().to(device)
    postprocessor.export_dense = True

    all_results = []

    print(f"Starting inference on {len(cfg.val_dataloader)} batches...")
    for i, (samples, targets) in enumerate(tqdm(cfg.val_dataloader)):
        samples = samples.to(device, non_blocking=True)
        targets = [
            {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in t.items()}
            for t in targets
        ]

        with torch.no_grad():
            outputs = model(samples, targets=targets)

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)

        # batched_results is now a SINGLE dict of shape [B, L, N, ...]
        batched_results = postprocessor(outputs, orig_target_sizes)

        # SAC-Level Optimization: Bulk PCIe transfer.
        # Move the entire batched output to CPU instantly, eliminating loop overhead.
        batched_results = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in batched_results.items()}

        batch_size = samples.shape[0]
        results = []

        for idx in range(batch_size):
            # Deterministically slice the batched tensors along the Batch dimension (dim=0)
            sample_dict = {k: v[idx] for k, v in batched_results.items()}

            orig_w, orig_h = targets[idx]["orig_size"].tolist()
            gt_dict = {
                "boxes": scale_boxes(
                    targets[idx]["boxes"],
                    (orig_h, orig_w),
                    (samples.shape[-2], samples.shape[-1]),  # (H, W) of the input tensor
                ),
                "labels": targets[idx]["labels"],
                "image_path": targets[idx].get("image_path", ""),
                "image_size": targets[idx]["orig_size"],
            }

            for k, v in gt_dict.items():
                if k in ["boxes", "labels"]:
                    sample_dict[f"gt_{k}"] = gt_dict[k]
                else:
                    sample_dict[k] = v

            # CRITICAL: Immediately apply the NumPy Top-K Pruning.
            # This destroys the PyTorch memory views before they are appended to the global list,
            # preventing the out-of-memory serialization crash.
            sample_dict = strip_to_numpy_dense(sample_dict)

            results.append(sample_dict)

        all_results.extend(results)

    # Use standard Python pickle (bypassing PyTorch entirely)
    print(f"\nSaving {len(all_results)} highly-compressed pure NumPy snapshots to {args.output_file} ...")
    with open(args.output_file, "wb") as f:
        pickle.dump(all_results, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("Done! The dataset is now completely decoupled from PyTorch internals.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract and Save Dense Layer-wise Spatial Snapshots (No Top-K)")
    parser.add_argument("--config", "-c", type=str, required=True, help="Path to YAML configuration file")
    parser.add_argument("--resume", "-r", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--num-samples", "-n", type=int, default=100, help="Number of samples to visualize")
    parser.add_argument(
        "--output-file", "-o", type=str, default="inference_dense_results.pkl", help="Output file path (.pt)"
    )
    args = parser.parse_args()

    main(args)
