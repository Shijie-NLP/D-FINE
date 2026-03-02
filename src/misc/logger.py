"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
https://github.com/facebookresearch/detr/blob/main/util/misc.py

Refactored to remove legacy PyTorch < 1.8 serialization fossils, ensuring
memory safety and leveraging native C++ backed distributed operators.
"""

import datetime
import time
from collections import defaultdict, deque
from collections.abc import Generator, Iterable
from typing import Any, Optional

import torch
import torch.distributed as tdist

from .dist_utils import get_world_size, is_dist_available_and_initialized


class SmoothedValue:
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size: int = 20, fmt: Optional[str] = None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value: float, n: int = 1) -> None:
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self) -> None:
        """
        Synchronize the global total and count across all processes.
        Warning: does not synchronize the deque!
        """
        if not is_dist_available_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device="cuda")
        tdist.barrier()
        tdist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self) -> float:
        # Return 0.0 if empty to avoid RuntimeError
        if not self.deque:
            return 0.0
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self) -> float:
        if not self.deque:
            return 0.0
        # More efficient python-native sum for small lists rather than tensor dispatch
        return sum(self.deque) / len(self.deque)

    @property
    def global_avg(self) -> float:
        return self.total / self.count if self.count > 0 else 0.0

    @property
    def max(self) -> float:
        return max(self.deque) if self.deque else 0.0

    @property
    def value(self) -> float:
        return self.deque[-1] if self.deque else 0.0

    def __str__(self) -> str:
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value,
        )


def all_gather(data: Any) -> list[Any]:
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors).
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    # [SAC Note]: Replaced the manual byte serialization and padding
    # with modern PyTorch's native all_gather_object.
    data_list = [None for _ in range(world_size)]
    tdist.all_gather_object(data_list, data)

    return data_list


def reduce_dict(input_dict: dict[str, torch.Tensor], average: bool = True) -> dict[str, torch.Tensor]:
    """
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.

    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict

    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])

        values = torch.stack(values, dim=0)
        # Ensure tensor is on cuda before all_reduce
        if not values.is_cuda:
            values = values.cuda()

        tdist.all_reduce(values)
        if average:
            values /= world_size

        reduced_dict = {k: v for k, v in zip(names, values)}  # noqa

    return reduced_dict


class MetricLogger:
    def __init__(self, delimiter: str = "\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr: str) -> Any:
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")

    def __str__(self) -> str:
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(f"{name}: {str(meter)}")
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self) -> None:
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name: str, meter: SmoothedValue) -> None:
        self.meters[name] = meter

    def log_every(self, iterable: Iterable, print_freq: int, header: Optional[str] = None) -> Generator:
        i = 0
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")

        try:
            total_len = len(iterable)
            space_fmt = ":" + str(len(str(total_len))) + "d"
        except TypeError:
            total_len = 0
            space_fmt = ":d"

        if torch.cuda.is_available():
            log_msg = self.delimiter.join(
                [
                    header,
                    "[{0" + space_fmt + "}/{1}]",
                    "eta: {eta}",
                    "{meters}",
                    "time: {time}",
                    "data: {data}",
                    "max mem: {memory:.0f}",
                ]
            )
        else:
            log_msg = self.delimiter.join(
                [
                    header,
                    "[{0" + space_fmt + "}/{1}]",
                    "eta: {eta}",
                    "{meters}",
                    "time: {time}",
                    "data: {data}",
                ]
            )

        MB = 1024.0 * 1024.0

        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)

            if i % print_freq == 0 or (total_len and i == total_len - 1):
                eta_seconds = iter_time.global_avg * (total_len - i) if total_len else 0
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

                if torch.cuda.is_available():
                    print(
                        log_msg.format(
                            i,
                            total_len,
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                            memory=torch.cuda.max_memory_allocated() / MB,
                        )
                    )
                else:
                    print(
                        log_msg.format(
                            i,
                            total_len,
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                        )
                    )
            i += 1
            end = time.time()

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(f"{header} Total time: {total_time_str} ({total_time / (i or 1):.4f} s / it)")
