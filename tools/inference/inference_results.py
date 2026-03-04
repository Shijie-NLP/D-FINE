import os
import sys


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
import argparse

import torch

from src.core import YAMLConfig
from src.data.dataset import mscoco_category2label
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
        checkpoint = torch.load(args.resume, map_location="cpu")
        state = checkpoint["ema"]["module"] if "ema" in checkpoint else checkpoint["model"]
    else:
        raise ValueError("Checkpoint required: Currently only supporting resume to load model.state_dict.")

    cfg.model.load_state_dict(state)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = cfg.model.eval().to(device)
    postprocessor = cfg.postprocessor.eval().to(device)

    gt: list[dict[str, torch.Tensor]] = []
    preds: list[dict[str, torch.Tensor]] = []

    # 为了快速测试 UI，建议你可以暂时在 log_every 这里限制一下 batch 数量
    for i, (samples, targets) in enumerate(cfg.val_dataloader):
        samples = samples.to(device, non_blocking=True)
        targets = [
            {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in t.items()}
            for t in targets
        ]

        # 这里的 outputs 包含了所有的中间层，但下面的 postprocessor 只提取了最终层
        with torch.no_grad():
            outputs = model(samples, targets=targets)

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessor(outputs, orig_target_sizes)

        for idx, (target, result) in enumerate(zip(targets, results)):
            gt.append(
                {
                    "boxes": scale_boxes(
                        target["boxes"],
                        (target["orig_size"][1].item(), target["orig_size"][0].item()),
                        (samples[idx].shape[-1], samples[idx].shape[-2]),
                    ).cpu(),
                    "labels": target["labels"].cpu(),
                    "image_path": target["image_path"],
                    "image_size": target["orig_size"].cpu(),
                }
            )

            labels = (
                (
                    torch.tensor([mscoco_category2label[int(x.item())] for x in result["labels"].flatten()])
                    .to(result["labels"].device)
                    .reshape(result["labels"].shape)
                )
                if postprocessor.remap_mscoco_category
                else result["labels"]
            )

            preds.append(
                {
                    "boxes": result["boxes"].cpu(),
                    "labels": labels.cpu(),
                    "scores": result["scores"].cpu(),
                }
            )

    return gt, preds


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FiftyOne Visualization for D-FINE Layer-wise Output")
    parser.add_argument("--config", "-c", type=str, required=True, help="Path to YAML configuration file")
    parser.add_argument("--resume", "-r", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--num-samples", "-n", type=int, default=100, help="Number of samples to visualize")
    args = parser.parse_args()

    main(args)
