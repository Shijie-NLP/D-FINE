"""
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

from .amp import GradScaler
from .ema import ExponentialMovingAverage, ModelEMA
from .optim import (
    SGD,
    Adam,
    AdamW,
    CosineAnnealingLR,
    LambdaLR,
    MultiStepLR,
    OneCycleLR,
)
from .warmup import LinearWarmup, Warmup
