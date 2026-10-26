"""Defines return patterns for various data structures used in the project and type utils."""

from dataclasses import dataclass, fields
from typing import TypeVar

import numpy as np
import torch
from torch import Tensor

AnyNotNone = TypeVar("AnyNotNone")


def assert_non_NaN(x: AnyNotNone | None) -> AnyNotNone:
    """Assert that the input x is not None.

    Parameters:
        x (AnyNotNone): The input to check.

    Returns:
        AnyNotNone: Returns the input.

    Raises:
        ValueError: If the input is None.
    """

    if x is None:
        raise ValueError("Input is None.")

    return x


@dataclass
class SampleCoords:
    """Defines RunoffData.coords return pattern."""

    hru: str
    start_date: str
    end_date: str
    spinup_size: int


@dataclass
class SamplePattern:
    """Defines RunoffData return pattern."""

    dfeatures: np.ndarray
    dtargets: np.ndarray
    coords: SampleCoords
    sfeatures: np.ndarray | None = None


@dataclass
class BatchCoords:
    """Defines RunoffDataModule dataloader return pattern."""

    hru: list[str]
    start_date: list[str]
    end_date: list[str]
    spinup_size: list[int]


@dataclass
class BatchPattern:
    """Defines RunoffDataModule dataloader.coords return pattern."""

    dfeatures: Tensor
    dtargets: Tensor
    coords: BatchCoords
    sfeatures: Tensor | None = None


@dataclass
class ReturnPattern:
    """Defines model return pattern."""

    dtargets: Tensor
    coords: BatchCoords


@dataclass
class HRUSample:
    """Defines a HRU sample return pattern."""

    x: Tensor
    x_routing: Tensor | None
    d: Tensor
    y: Tensor
    prec: Tensor | None
    tair: Tensor | None
    hru_idx: Tensor
    hru_area: Tensor
    is_lake: Tensor | None
    water_freq: Tensor | None
    time_slice: list[int]

    def to(self, device: torch.device | str) -> "HRUSample":
        """Move all Tensor fields of the HRUSample instance to the specified device.

        Args:
            device (torch.device | str): The target device to move the tensors to.

        Returns:
            HRUSample: A new HRUSample instance with all Tensor fields moved to the specified device.
        """

        new_kwargs = {}

        for field in fields(self):
            el = getattr(self, field.name)
            if isinstance(el, Tensor):
                el = el.to(device)
            new_kwargs[field.name] = el

        return HRUSample(**new_kwargs)
