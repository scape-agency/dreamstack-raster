# -*- coding: utf-8 -*-

"""
Image Measurement Module
========================

Functions for measuring and sampling image data.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from numpy.typing import NDArray


@dataclass
class PixelInfo:
    """Information about a pixel.

    Attributes
    ----------
    x : int
        X coordinate.
    y : int
        Y coordinate.
    rgb : tuple
        RGB values.
    hsv : tuple
        HSV values.
    lab : tuple
        LAB values.
    """

    x: int
    y: int
    rgb: Tuple[int, int, int]
    hsv: Tuple[int, int, int]
    lab: Tuple[int, int, int]


def sample_color(
    image: NDArray[np.uint8],
    x: int,
    y: int,
) -> Tuple[int, int, int]:
    """Sample color at a specific pixel.

    Parameters
    ----------
    image : NDArray[np.uint8]
        Input image (BGR).
    x : int
        X coordinate.
    y : int
        Y coordinate.

    Returns
    -------
    tuple
        BGR color values.
    """
    if x < 0 or x >= image.shape[1] or y < 0 or y >= image.shape[0]:
        raise ValueError(f"Coordinates ({x}, {y}) out of bounds")

    if image.ndim == 2:
        val = int(image[y, x])
        return (val, val, val)

    return tuple(int(v) for v in image[y, x])


def pixel_info(
    image: NDArray[np.uint8],
    x: int,
    y: int,
) -> PixelInfo:
    """Get detailed information about a pixel.

    Parameters
    ----------
    image : NDArray[np.uint8]
        Input image (BGR).
    x : int
        X coordinate.
    y : int
        Y coordinate.

    Returns
    -------
    PixelInfo
        Detailed pixel information.
    """
    bgr = sample_color(image, x, y)
    rgb = (bgr[2], bgr[1], bgr[0])

    # Convert to HSV
    pixel = np.array([[bgr]], dtype=np.uint8)
    hsv_pixel = cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV)
    hsv = tuple(int(v) for v in hsv_pixel[0, 0])

    # Convert to LAB
    lab_pixel = cv2.cvtColor(pixel, cv2.COLOR_BGR2LAB)
    lab = tuple(int(v) for v in lab_pixel[0, 0])

    return PixelInfo(x=x, y=y, rgb=rgb, hsv=hsv, lab=lab)


def color_sampler(
    image: NDArray[np.uint8],
    x: int,
    y: int,
    radius: int = 0,
) -> Tuple[int, int, int]:
    """Sample average color in a region.

    Parameters
    ----------
    image : NDArray[np.uint8]
        Input image.
    x : int
        Center X coordinate.
    y : int
        Center Y coordinate.
    radius : int, optional
        Sample radius. Default is 0 (single pixel).

    Returns
    -------
    tuple
        Average BGR color.
    """
    if radius == 0:
        return sample_color(image, x, y)

    h, w = image.shape[:2]
    x1 = max(0, x - radius)
    y1 = max(0, y - radius)
    x2 = min(w, x + radius + 1)
    y2 = min(h, y + radius + 1)

    region = image[y1:y2, x1:x2]
    avg = np.mean(region, axis=(0, 1))

    if image.ndim == 2:
        val = int(avg)
        return (val, val, val)

    return tuple(int(v) for v in avg)


def measure_selection(
    mask: NDArray[np.uint8],
    image: Optional[NDArray[np.uint8]] = None,
) -> Dict[str, Union[int, float, Tuple[int, int]]]:
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
