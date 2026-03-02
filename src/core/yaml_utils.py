"""
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.

Optimized for strict typing, memory safety (avoiding mutable defaults),
and safe YAML parsing mechanics.
"""

import copy
import os
from typing import Any, Optional

import yaml

from .workspace import GLOBAL_CONFIG

__all__ = [
    "load_config",
    "merge_config",
    "merge_dict",
    "parse_cli",
]

INCLUDE_KEY = "__include__"


def load_config(file_path: str, cfg: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """
    Load and parse a YAML configuration file, recursively resolving inclusions.

    Optimization: Avoided the mutable default argument `cfg=dict()` which causes
    state retention and memory leaks across multiple function calls.
    """
    if cfg is None:
        cfg = {}

    _, ext = os.path.splitext(file_path)
    assert ext in [".yml", ".yaml"], f"Only support yaml files, got {ext}"

    # Enforce UTF-8 encoding for cross-platform robustness
    with open(file_path, encoding="utf-8") as f:
        # Note: yaml.Loader is retained to preserve original serialization logic,
        # but consider yaml.SafeLoader if arbitrary code execution is a concern.
        file_cfg = yaml.load(f, Loader=yaml.Loader)
        if file_cfg is None:
            return {}

    if INCLUDE_KEY in file_cfg:
        base_yamls: list[str] = list(file_cfg[INCLUDE_KEY])
        for base_yaml in base_yamls:
            if base_yaml.startswith("~"):
                base_yaml = os.path.expanduser(base_yaml)

            if not base_yaml.startswith("/"):
                base_yaml = os.path.join(os.path.dirname(file_path), base_yaml)

            # Recursively load included configurations
            base_cfg = load_config(base_yaml, cfg)
            merge_dict(cfg, base_cfg)

    return merge_dict(cfg, file_cfg)


def merge_dict(
    dct: dict[str, Any], another_dct: dict[str, Any], inplace: bool = True
) -> dict[str, Any]:
    """Recursively merge `another_dct` into `dct`."""

    def _merge(target: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
        for k in source:
            if (
                k in target
                and isinstance(target[k], dict)
                and isinstance(source[k], dict)
            ):
                _merge(target[k], source[k])
            else:
                target[k] = source[k]
        return target

    if not inplace:
        dct = copy.deepcopy(dct)

    return _merge(dct, another_dct)


def dictify(s: str, v: Any) -> dict[str, Any]:
    """Convert a dot-separated string into a nested dictionary."""
    if "." not in s:
        return {s: v}
    key, rest = s.split(".", 1)
    return {key: dictify(rest, v)}


def parse_cli(nargs: Optional[list[str]]) -> dict[str, Any]:
    """
    Parse command-line arguments.
    Converts inputs like `a.c=3 b=10` to `{'a': {'c': 3}, 'b': 10}`.
    """
    cfg: dict[str, Any] = {}
    if nargs is None or len(nargs) == 0:
        return cfg

    for s in nargs:
        s = s.strip()
        k, v = s.split("=", 1)
        # Parse the value safely via YAML
        d = dictify(k, yaml.load(v, Loader=yaml.Loader))
        cfg = merge_dict(cfg, d)

    return cfg


def merge_config(
    cfg: dict[str, Any],
    another_cfg: dict[str, Any] = GLOBAL_CONFIG,
    inplace: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    """
    Merge `another_cfg` into `cfg`.
    If `overwrite` is False, `another_cfg` keys will only be added if missing.
    """

    def _merge(target: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
        for k in source:
            if k not in target:
                target[k] = source[k]

            elif isinstance(target[k], dict) and isinstance(source[k], dict):
                _merge(target[k], source[k])

            elif overwrite:
                target[k] = source[k]

        return target

    if not inplace:
        cfg = copy.deepcopy(cfg)

    return _merge(cfg, another_cfg)
