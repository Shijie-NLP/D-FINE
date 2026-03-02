"""
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.

Optimized for strict static typing.
"""

from ._solver import BaseSolver
from .clas_solver import ClasSolver
from .det_solver import DetSolver


__all__ = ["TASKS"]

# Strict type mapping ensures downstream tasks adhere to the BaseSolver contract
TASKS: dict[str, type[BaseSolver]] = {
    "classification": ClasSolver,
    "detection": DetSolver,
}
