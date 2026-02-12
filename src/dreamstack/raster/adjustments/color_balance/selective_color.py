"""Selective color adjustment function."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from dreamstack.raster.core.image import Image

from dreamstack.raster.adjustments.color_balance._color_utils import (
    _get_color_mask,
)


def selective_color(
    image: Image,
    color: str,
    cyan: float = 0,
    magenta: float = 0,
    yellow: float = 0,
    black: float = 0,
    method: str = "relative",
) -> Image:
    """
    Adjust colors selectively by color range.

    Args:
        image: Input image
        color: Color to adjust ('reds', 'yellows', 'greens', 'cyans', 'blues',
                                'magentas', 'whites', 'neutrals', 'blacks')
        cyan: Cyan adjustment (-100 to 100)
        magenta: Magenta adjustment (-100 to 100)
        yellow: Yellow adjustment (-100 to 100)
        black: Black adjustment (-100 to 100)
        method: 'relative' or 'absolute'

    Returns:
        Adjusted image
    """
    from dreamstack.raster.core.pixel import PixelData

    if image.channels < 3:
        return image.copy()

    data = image.data.astype(np.float32)
    max_val = 255 if image.bit_depth.name == "UINT8" else 65535

    normalized = data[:, :, :3] / max_val

    # Get color mask
    mask = _get_color_mask(normalized, color)

    if mask is None:
        return image.copy()

    # Apply CMYK adjustments
    c_adj = cyan / 100
    m_adj = magenta / 100
    y_adj = yellow / 100
    k_adj = black / 100

    result = normalized.copy()

    if method == "relative":
        # Relative: percentage of current color
        result[:, :, 0] -= c_adj * mask * (1 - normalized[:, :, 0])  # Cyan reduces Red
        result[:, :, 1] -= (
            m_adj * mask * (1 - normalized[:, :, 1])
        )  # Magenta reduces Green
        result[:, :, 2] -= (
            y_adj * mask * (1 - normalized[:, :, 2])
        )  # Yellow reduces Blue

        # Black adjustment
        if k_adj > 0:
            result = result * (1 - k_adj * mask[:, :, np.newaxis])
        elif k_adj < 0:
            result = result + (1 - result) * (-k_adj) * mask[:, :, np.newaxis]
    else:
        # Absolute: fixed amount
        result[:, :, 0] -= c_adj * mask
        result[:, :, 1] -= m_adj * mask
        result[:, :, 2] -= y_adj * mask

        result = result * (1 - k_adj * mask[:, :, np.newaxis])

    result = np.clip(result, 0, 1)

    final = data.copy()
    final[:, :, :3] = result * max_val

    result_image = image.copy()
    result_image._pixel_data = PixelData(
        data=final.astype(image.data.dtype),
        pixel_format=image.pixel_format,
        bit_depth=image.bit_depth,
    )

    return result_image
