# -*- coding: utf-8 -*-

"""Apply curve function."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from dreamstack.raster.core.image import Image

from dreamstack.raster.adjustments.curves.curve import Curve
from dreamstack.raster.adjustments.curves.curves import curves


def apply_curve(
    image: Image, curve: Curve, channel: Optional[str] = None
) -> Image:
    """
    Apply a single curve to image.

    Args:
        image: Input image
        curve: Curve to apply
        channel: Channel to apply to ('red', 'green', 'blue', or None for all)

    Returns:
        Adjusted image
    """
    if channel is None:
        return curves(image, rgb_curve=curve)
    elif channel.lower() == "red":
        return curves(image, red_curve=curve)
    elif channel.lower() == "green":
        return curves(image, green_curve=curve)
    elif channel.lower() == "blue":
        return curves(image, blue_curve=curve)
    else:
        return image.copy()
