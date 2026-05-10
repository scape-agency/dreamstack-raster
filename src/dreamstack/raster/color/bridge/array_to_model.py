# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""Convert between numpy arrays and local color models."""

# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

import numpy as np

from dreamstack.raster.color.models import RGBColorModel


def array_to_rgb(
    array: np.ndarray,
    normalized: bool = True,
) -> RGBColorModel:
    """
    Convert a numpy array to an RGBColorModel.

    Args:
        array: RGB(A) array of shape (3,) or (4,)
        normalized: Whether input values are in 0-1 range (default: True)

    Returns:
        Local RGBColorModel instance
    """
    array = np.asarray(array)

    if array.ndim != 1 or array.shape[0] not in (3, 4):
        raise ValueError(f"Expected shape (3,) or (4,), got {array.shape}")

    if normalized:
        r = int(array[0] * 255)
        g = int(array[1] * 255)
        b = int(array[2] * 255)
    else:
        r = int(array[0])
        g = int(array[1])
        b = int(array[2])

    a = float(array[3]) if array.shape[0] == 4 else 1.0
    if normalized and array.shape[0] == 4:
        a = float(array[3])  # Alpha is already 0-1

    return RGBColorModel(r, g, b, a)


def rgb_to_array(
    rgb: RGBColorModel,
    normalized: bool = True,
    include_alpha: bool = False,
) -> np.ndarray:
    """
    Convert an RGBColorModel to a numpy array.

    Args:
        rgb: Local RGBColorModel instance
        normalized: Whether to output values in 0-1 range (default: True)
        include_alpha: Whether to include alpha channel (default: False)

    Returns:
        Numpy array of shape (3,) or (4,)
    """
    if normalized:
        r, g, b = rgb.to_normalized()
        if include_alpha:
            return np.array([r, g, b, rgb.a], dtype=np.float64)
        return np.array([r, g, b], dtype=np.float64)
    else:
        if include_alpha:
            return np.array(
                [rgb.r, rgb.g, rgb.b, int(rgb.a * 255)], dtype=np.uint8
            )
        return np.array([rgb.r, rgb.g, rgb.b], dtype=np.uint8)


def arrays_to_rgb_list(
    array: np.ndarray,
    normalized: bool = True,
) -> list[RGBColorModel]:
    """
    Convert a 2D/3D array of colors to a list of RGBColorModels.

    Useful for operations that need single-color model objects.

    Args:
        array: Array of shape (N, 3), (N, 4), (H, W, 3), or (H, W, 4)
        normalized: Whether input values are in 0-1 range

    Returns:
        List of RGBColorModel objects
    """
    array = np.asarray(array)

    # Flatten to 2D if 3D image array
    if array.ndim == 3:
        _h, _w, c = array.shape  # h, w unused; only need channel count
        array = array.reshape(-1, c)
    elif array.ndim != 2:
        raise ValueError(f"Expected 2D or 3D array, got {array.ndim}D")

    colors = []
    for i in range(array.shape[0]):
        colors.append(array_to_rgb(array[i], normalized=normalized))

    return colors


def rgb_list_to_arrays(
    colors: list[RGBColorModel],
    normalized: bool = True,
    include_alpha: bool = False,
) -> np.ndarray:
    """
    Convert a list of RGBColorModels to a 2D numpy array.

    Args:
        colors: List of RGBColorModel objects
        normalized: Whether to output values in 0-1 range
        include_alpha: Whether to include alpha channel

    Returns:
        Array of shape (N, 3) or (N, 4)
    """
    arrays = [
        rgb_to_array(c, normalized=normalized, include_alpha=include_alpha)
        for c in colors
    ]
    return np.stack(arrays, axis=0)


__all__: list[str] = [
    "array_to_rgb",
    "rgb_to_array",
    "arrays_to_rgb_list",
    "rgb_list_to_arrays",
]
