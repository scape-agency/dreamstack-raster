"""Pad to aspect ratio operation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def pad_to_aspect(
    image: NDArray[np.uint8],
    aspect_ratio: tuple[int, int] = (16, 9),
    *,
    pad_color: tuple[int, int, int] = (0, 0, 0),
    position: Literal["center", "top", "bottom", "left", "right"] = "center",
) -> NDArray[np.uint8]:
    """Pad image to achieve target aspect ratio.

    Adds black bars (letterbox/pillarbox) to reach the target aspect.

    Args:
        image: Input image.
        aspect_ratio: Target aspect ratio as (width, height).
        pad_color: Color for padding (BGR).
        position: Position of original image in padded result.

    Returns:
        Padded image with target aspect ratio.

    Example:
        >>> # Add letterbox to make 16:9
        >>> letterboxed = pad_to_aspect(image, (16, 9))
    """
    h, w = image.shape[:2]
    ar_w, ar_h = aspect_ratio

    current_ar = w / h
    target_ar = ar_w / ar_h

    if abs(current_ar - target_ar) < 0.001:
        return image.copy()

    if current_ar > target_ar:
        # Too wide, add vertical padding
        new_height = int(w / target_ar)
        pad_total = new_height - h

        if position == "top":
            pad_top, _pad_bottom = 0, pad_total
        elif position == "bottom":
            pad_top, _pad_bottom = pad_total, 0
        else:  # center
            pad_top = pad_total // 2
            _pad_bottom = pad_total - pad_top

        pad_left = 0
        new_width = w
    else:
        # Too tall, add horizontal padding
        new_width = int(h * target_ar)
        pad_total = new_width - w

        if position == "left":
            pad_left, _pad_right = 0, pad_total
        elif position == "right":
            pad_left, _pad_right = pad_total, 0
        else:  # center
            pad_left = pad_total // 2
            _pad_right = pad_total - pad_left

        pad_top = 0
        new_height = h

    # Create padded image
    channels = image.shape[2] if image.ndim == 3 else 1
    if channels > 1:
        result = np.full(
            (new_height, new_width, channels),
            pad_color[:channels],
            dtype=np.uint8,
        )
    else:
        result = np.full((new_height, new_width), pad_color[0], dtype=np.uint8)

    # Place original image
    result[pad_top : pad_top + h, pad_left : pad_left + w] = image

    return result
