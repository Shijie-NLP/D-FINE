"""
D-FINE: Redefine Regression Task of DETRs as Fine-grained Distribution Refinement
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright (c) 2023 lyuwenyu. All Rights Reserved.

Optimized for robust distributed execution, secure environment pathing,
and safe configuration overrides.
"""

import argparse
import sys
from pathlib import Path
from pprint import pprint
from typing import Any

import torch


# Optimization: Modernized absolute path resolution to prevent import failures
# across diverse distributed file systems (e.g., Ceph, NFS, or symlinked dirs).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.core import YAMLConfig, yaml_utils
from src.misc import dist_utils
from src.solver import TASKS


DEBUG_MODE = False

if DEBUG_MODE:
    # SAC Warning: Globally mutating torch.Tensor.__repr__ is a dangerous
    # monkey-patching anti-pattern. It can severely break third-party logging
    # hooks. Ensure this remains strictly False during production multi-node runs.
    def custom_repr(self: torch.Tensor) -> str:
        return f"{{Tensor:{tuple(self.shape)}}} {_original_repr(self)}"

    _original_repr = torch.Tensor.__repr__
    torch.Tensor.__repr__ = custom_repr


def safe_get_rank() -> int:
    """Safely retrieves the distributed rank ID, avoiding initialization crashes."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0


def main(args: argparse.Namespace) -> None:
    """Main execution pipeline for training or evaluation."""

    # Initialize distributed process group
    dist_utils.setup_distributed(args.print_rank, args.print_method, seed=args.seed)

    assert not (args.tuning and args.resume), (
        "Conflict Error: Only one of 'from_scratch', 'resume', or 'tuning' can be specified at a time."
    )

    # Safely parse CLI string updates (e.g., -u model.lr=0.001)
    update_dict: dict[str, Any] = yaml_utils.parse_cli(args.update)

    # Safely merge argparse arguments into the update dictionary,
    # strictly ignoring None values to prevent overwriting YAML defaults.
    cli_args_dict = {k: v for k, v in args.__dict__.items() if k != "update" and v is not None}
    update_dict.update(cli_args_dict)

    # Instantiate the unified configuration parser
    cfg = YAMLConfig(args.config, **update_dict)

    # SAC Note: This hardcoded architectural override introduces strong coupling.
    # If using other backbones in the future, consider moving this flag to the YAML schema.
    if args.resume or args.tuning:
        if "HGNetv2" in cfg.yaml_cfg:
            cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

    if safe_get_rank() == 0:
        print("\n" + "=" * 50)
        print("Experiment Configuration:")
        print("=" * 50)
        pprint(cfg.__dict__)
        print("=" * 50 + "\n")

    # Dynamically instantiate the solver based on the task definition
    solver = TASKS[cfg.yaml_cfg["task"]](cfg)

    # Route execution logic
    if args.test_only:
        solver.val()
    else:
        solver.fit()

    # Gracefully destroy the distributed process group
    dist_utils.cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="D-FINE Training and Evaluation Script")

    # Priority 0 arguments (Core functionality)
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="Path to the YAML configuration file",
    )
    parser.add_argument("-r", "--resume", type=str, help="Resume training from a specific checkpoint")
    parser.add_argument("-t", "--tuning", type=str, help="Fine-tune starting from a specific checkpoint")
    parser.add_argument("-d", "--device", type=str, help="Target device (e.g., 'cuda:0', 'cpu')")
    parser.add_argument("--seed", type=int, help="Random seed for experimental reproducibility")
    parser.add_argument(
        "--use-amp",
        action="store_true",
        help="Enable Automatic Mixed Precision (AMP) training",
    )
    parser.add_argument("--output-dir", type=str, help="Directory to save weights and logs")
    parser.add_argument("--summary-dir", type=str, help="Directory to save TensorBoard summaries")
    parser.add_argument(
        "--test-only",
        action="store_true",
        default=False,
        help="Run exclusively in evaluation mode",
    )

    # Priority 1 arguments (Dynamic overrides)
    parser.add_argument(
        "-u",
        "--update",
        nargs="+",
        help="Update specific YAML config values via CLI (e.g., a=1 b=2)",
    )

    # Environment and distributed arguments
    parser.add_argument(
        "--print-method",
        type=str,
        default="builtin",
        help="Standard output routing method",
    )
    parser.add_argument(
        "--print-rank",
        type=int,
        default=0,
        help="Rank ID authorized to print to console",
    )
    parser.add_argument("--local-rank", type=int, help="Local rank ID (automatically set by torchrun)")

    parsed_args = parser.parse_args()

    main(parsed_args)
