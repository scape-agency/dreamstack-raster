"""
Color Range Selection
=====================

Select pixels within a specified color range.

"""

# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

# =============================================================================
# Type Checking Imports
# =============================================================================

if TYPE_CHECKING:
    # pylint: disable=import-outside-toplevel
    from numpy.typing import NDArray

from dreamstack.raster.selection.shapes.selection import Selection


def color_range(
    image: NDArray[np.uint8],
    target_color: tuple[int, int, int],
    *,
    fuzziness: int = 40,
    color_space: str = "hsv",
    invert: bool = False,
) -> Selection:
    """Select pixels within a color range.

    Creates a selection based on color similarity to target,
    with adjustable fuzziness.

    Args:
        image: Input image (BGR).
        target_color: Target color as BGR tuple.
        fuzziness: Color tolerance (0-128, higher = more selected).
        color_space: Color space for comparison ("hsv", "lab", "rgb").
        invert: If True, invert the selection.

    Returns:
        Selection based on color range.

    Example:
        >>> # Select red tones
        >>> sel = color_range(image, (0, 0, 255), fuzziness=60)
        >>> # Select sky blue
        >>> sel = color_range(image, (255, 200, 100), fuzziness=50)
    """
    # Convert to appropriate color space
    if color_space.lower() == "hsv":
        converted = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2HSV)
        target = cv2.cvtColor(
            np.array([[target_color]], dtype=np.uint8), cv2.COLOR_BGR2HSV
        )[0, 0]

        # HSV requires special handling for hue wraparound
        h, s, v = target

        # Scaled fuzziness for each channel
        h_range = fuzziness // 2  # Hue is 0-179
        s_range = fuzziness
        v_range = fuzziness

        lower = np.array(
            [
                max(0, h - h_range),
                max(0, s - s_range),
                max(0, v - v_range),
            ]
        )
        upper = np.array(
            [
                min(179, h + h_range),
                min(255, s + s_range),
                min(255, v + v_range),
            ]
        )

        mask = cv2.inRange(converted, lower, upper)

        # Handle hue wraparound for red tones
        if h - h_range < 0:
            lower2 = np.array([180 + (h - h_range), lower[1], lower[2]])
            upper2 = np.array([179, upper[1], upper[2]])
            mask2 = cv2.inRange(converted, lower2, upper2)
            mask = cv2.bitwise_or(mask, mask2)
        elif h + h_range > 179:
            lower2 = np.array([0, lower[1], lower[2]])
            upper2 = np.array([h + h_range - 180, upper[1], upper[2]])
            mask2 = cv2.inRange(converted, lower2, upper2)
            mask = cv2.bitwise_or(mask, mask2)

    elif color_space.lower() == "lab":
        converted = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2LAB)
        target = cv2.cvtColor(
            np.array([[target_color]], dtype=np.uint8), cv2.COLOR_BGR2LAB
        )[0, 0]

        lower = np.maximum(0, target.astype(np.int16) - fuzziness).astype(
            np.uint8
        )
        upper = np.minimum(255, target.astype(np.int16) + fuzziness).astype(
            np.uint8
        )

        mask = cv2.inRange(converted, lower, upper)

    else:  # RGB/BGR
        target = np.array(target_color, dtype=np.int16)
        lower = np.maximum(0, target - fuzziness).astype(np.uint8)
        upper = np.minimum(255, target + fuzziness).astype(np.uint8)

        mask = cv2.inRange(image[:, :, :3], lower, upper)

    # Smooth edges slightly
    mask = cv2.GaussianBlur(mask, (3, 3), 0.7)

    if invert:
        mask = 255 - mask

    return Selection(mask=mask)
