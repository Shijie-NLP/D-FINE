"""
Refactored for High-Efficiency Layer-wise Spatial Snapshots.
- Fixed catastrophic list indexing bugs.
- Vectorized label mapping (Zero Python-loop overhead).
- Bypassed default postprocessor to extract independent layer-wise Top-K.
- Robust PyTorch Native Serialization (.pt).
"""

import argparse
import os
import sys

import torch
from tqdm import tqdm


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.core import YAMLConfig
from src.solver.validator import scale_boxes


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

        # results 是当前 batch 长度的 list[dict]
        results = postprocessor(outputs, orig_target_sizes)

        batch_size = samples.shape[0]
        for idx in range(batch_size):
            orig_h, orig_w = targets[idx]["orig_size"].tolist()
            gt_dict = {
                "boxes": scale_boxes(
                    targets[idx]["boxes"],
                    (orig_h, orig_w),
                    (samples.shape[-2], samples.shape[-1]),  # (H, W) of the input tensor
                ).cpu(),
                "labels": targets[idx]["labels"].cpu(),
                "image_id": targets[idx].get("image_id", torch.tensor(-1)).cpu(),
                "image_path": targets[idx].get("image_path", ""),
                "image_size": targets[idx]["orig_size"].cpu(),
            }

            results[idx]["gt"] = gt_dict

            for k, v in results[idx].items():
                if isinstance(v, torch.Tensor):
                    results[idx][k] = v.cpu()

        all_results.extend(results)

    save_path = args.output_file
    print(f"\nSaving {len(all_results)} highly-cohesive spatial snapshots to {save_path} ...")
    torch.save(all_results, save_path)
    print("Done! Data is ready for FiftyOne visualization and downstream analysis.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract and Save Dense Layer-wise Spatial Snapshots (No Top-K)")
    parser.add_argument("--config", "-c", type=str, required=True, help="Path to YAML configuration file")
    parser.add_argument("--resume", "-r", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--num-samples", "-n", type=int, default=100, help="Number of samples to visualize")
    parser.add_argument(
        "--output-file", "-o", type=str, default="inference_dense_results.pt", help="Output file path (.pt)"
    )
    args = parser.parse_args()

    main(args)
