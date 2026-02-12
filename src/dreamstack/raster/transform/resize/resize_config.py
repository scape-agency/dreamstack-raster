"""Resize configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Interpolation = Literal["nearest", "linear", "cubic", "lanczos", "area"]


@dataclass
class ResizeConfig:
    """Configuration for resize operations.

    Attributes:
        interpolation: Interpolation method.
        preserve_aspect: Maintain aspect ratio.
        divisible_by: Ensure dimensions are divisible by this value.
    """

    interpolation: Interpolation = "lanczos"
    preserve_aspect: bool = True
    divisible_by: int | None = None
