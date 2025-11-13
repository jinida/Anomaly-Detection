from functools import singledispatch
from collections.abc import Mapping, Sequence
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

@singledispatch
def to_device(obj: any, device: torch.device, non_blocking: bool = False):
    return obj

@to_device.register(torch.Tensor)
def _(obj: torch.Tensor, device: torch.device, non_blocking: bool = False):
    return obj.to(device, non_blocking=non_blocking)

@to_device.register(nn.Module)
def _(obj: nn.Module, device: torch.device, non_blocking: bool = False):
    return obj.to(device, non_blocking=non_blocking)

@to_device.register(list)
def _(obj: list, device: torch.device, non_blocking: bool = False):
    return [to_device(v, device, non_blocking) for v in obj]

@to_device.register(tuple)
def _(obj: tuple, device: torch.device, non_blocking: bool = False):
    return tuple(to_device(v, device, non_blocking) for v in obj)

@to_device.register(dict)
def _(obj: dict, device: torch.device, non_blocking: bool = False):
    return {k: to_device(v, device, non_blocking) for k, v in obj.items()}
