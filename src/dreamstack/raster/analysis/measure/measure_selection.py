"""
Measure Selection Function
==========================

Measure properties of a selection.

"""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray


def measure_selection(
    mask: NDArray[np.uint8],
    image: NDArray[np.uint8] | None = None,
) -> dict[str, int | float | tuple[int, int] | tuple[int, int, int, int]]:
    """Measure properties of a selection.

    Parameters
    ----------
    mask : NDArray[np.uint8]
        Binary mask of selection.
    image : NDArray[np.uint8], optional
        Source image for color statistics.

    Returns
    -------
    dict
        Selection measurements.
    """
    # Find non-zero pixels
    coords = np.argwhere(mask > 0)

    if len(coords) == 0:
        return {
            "area": 0,
            "perimeter": 0,
            "centroid": (0, 0),
            "bounds": (0, 0, 0, 0),
        }

    # Bounding box
    y1, x1 = coords.min(axis=0)
    y2, x2 = coords.max(axis=0)

    # Centroid
    cy, cx = coords.mean(axis=0)

    # Find contours for perimeter
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    perimeter = sum(cv2.arcLength(c, True) for c in contours)

    result = {
        "area": len(coords),
        "perimeter": float(perimeter),
        "centroid": (int(cx), int(cy)),
        "bounds": (int(x1), int(y1), int(x2 - x1 + 1), int(y2 - y1 + 1)),
    }

    if image is not None:
        pixels = image[mask > 0]
        result["mean_color"] = tuple(int(v) for v in np.mean(pixels, axis=0))
        result["std_color"] = tuple(float(v) for v in np.std(pixels, axis=0))

    return result
