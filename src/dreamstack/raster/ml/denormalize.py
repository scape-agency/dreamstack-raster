"""Denormalize operation."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .constants import CAFFE_MEAN, IMAGENET_MEAN, IMAGENET_STD
from .models.normalization_type import NormalizationType


def denormalize(
    image: NDArray[np.float32],
    method: NormalizationType | str = NormalizationType.MINMAX,
) -> NDArray[np.uint8]:
    """Reverse normalization for visualization.

    Parameters
    ----------
    image : NDArray[np.float32]
        Normalized image.
    method : NormalizationType or str
        Method used for normalization.

    Returns
    -------
    NDArray[np.uint8]
        Image in uint8 format (0-255).
    """
    if isinstance(method, str):
        method = NormalizationType(method.lower())

    if method == NormalizationType.MINMAX:
        result = image

    elif method == NormalizationType.IMAGENET or method == NormalizationType.TORCH:
        result = image * IMAGENET_STD + IMAGENET_MEAN

    elif method == NormalizationType.CAFFE:
        result = (image + CAFFE_MEAN) / 255.0

    elif method == NormalizationType.TF:
        result = (image / 2.0) + 0.5

    else:
        result = image

    result = np.clip(result * 255.0, 0, 255)
    return result.astype(np.uint8)
