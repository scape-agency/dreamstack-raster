# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Over Operation
==============

Composite foreground over background using alpha.

"""


# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    # pylint: disable=import-outside-toplevel
    from numpy.typing import NDArray


def over(
    foreground: NDArray[np.uint8],
    background: NDArray[np.uint8],
    *,
    premultiplied: bool = False,
) -> NDArray[np.uint8]:
    """Composite foreground over background using alpha.

    Standard "over" compositing operation using alpha channel.

    Args:
        foreground: Foreground image (must have alpha channel).
        background: Background image.
        premultiplied: If True, foreground is already premultiplied.

    Returns:
        Composited image.

    Example:
        >>> result = over(logo_with_alpha, photo)
    """
    if len(foreground.shape) != 3 or foreground.shape[2] != 4:
        raise ValueError("Foreground must have alpha channel (RGBA)")

    # Ensure background has alpha
    if len(background.shape) == 2:
        background = np.stack([background] * 3, axis=-1)
    if background.shape[2] == 3:
        bg_alpha = np.full((*background.shape[:2], 1), 255, dtype=np.uint8)
        background = np.concatenate([background, bg_alpha], axis=-1)

    fg = foreground.astype(np.float32) / 255.0
    bg = background.astype(np.float32) / 255.0

    fg_alpha = fg[:, :, 3:4]
    bg_alpha = bg[:, :, 3:4]

    if not premultiplied:
        fg_rgb = fg[:, :, :3] * fg_alpha
    else:
        fg_rgb = fg[:, :, :3]

    bg_rgb = bg[:, :, :3] * bg_alpha

    # Over operation
    out_alpha = fg_alpha + bg_alpha * (1 - fg_alpha)
    out_alpha = np.clip(out_alpha, 1e-6, 1.0)  # Prevent division by zero

    out_rgb = (fg_rgb + bg_rgb * (1 - fg_alpha)) / out_alpha

    result = np.concatenate([out_rgb, out_alpha], axis=-1)
    result = np.clip(result * 255.0, 0, 255).astype(np.uint8)

    return result
