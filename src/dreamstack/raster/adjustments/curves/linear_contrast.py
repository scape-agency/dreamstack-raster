# -*- coding: utf-8 -*-

"""Linear contrast curve function."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dreamstack.raster.core.image import Image

from dreamstack.raster.adjustments.curves.create_curve import create_curve
from dreamstack.raster.adjustments.curves.curves import curves


def linear_contrast(
    image: Image, black_point: float = 0, white_point: float = 255
) -> Image:
    """
    Apply linear contrast curve.

    Args:
        image: Input image
        black_point: New black point (0-255)
        white_point: New white point (0-255)

    Returns:
        Adjusted image
    """
    curve = create_curve([(black_point, 0), (white_point, 255)])

    return curves(image, rgb_curve=curve)
