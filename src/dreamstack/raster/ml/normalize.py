# -*- coding: utf-8 -*-
# pyright: reportArgumentType=false, reportReturnType=false


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Normalize operation."""

# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .constants import CAFFE_MEAN, IMAGENET_MEAN, IMAGENET_STD
from .models.normalization_type import NormalizationType


def normalize(
    image: NDArray,
    method: NormalizationType | str = NormalizationType.MINMAX,
) -> NDArray[np.float32]:
    """Normalize image for ML model input.

    Parameters
    ----------
    image : NDArray
        Input image (uint8 0-255 or float 0-1).
    method : NormalizationType or str
        Normalization method:
        - "minmax": Scale to [0, 1]
        - "standardize": Zero mean, unit variance
        - "imagenet": ImageNet normalization (RGB)
        - "caffe": Caffe BGR mean subtraction
        - "torch": PyTorch normalization
        - "tensorflow": Scale to [-1, 1]

    Returns
    -------
    NDArray[np.float32]
        Normalized image.

    Examples
    --------
    >>> # For ResNet, VGG, etc.
    >>> normalized = normalize(img, "imagenet")

    >>> # For TensorFlow models
    >>> normalized = normalize(img, "tensorflow")
    """
    if isinstance(method, str):
        method = NormalizationType(method.lower())

    # Convert to float
    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0
    else:
        image = image.astype(np.float32)

    if method == NormalizationType.MINMAX:
        return image

    elif method == NormalizationType.STANDARDIZE:
        mean = np.mean(image)
        std = np.std(image) + 1e-7
        return (image - mean) / std

    elif method == NormalizationType.IMAGENET:
        # Assumes RGB input
        return (image - IMAGENET_MEAN) / IMAGENET_STD

    elif method == NormalizationType.TORCH:
        # Same as ImageNet but explicit
        return (image - IMAGENET_MEAN) / IMAGENET_STD

    elif method == NormalizationType.CAFFE:
        # BGR input, mean subtraction
        return (image * 255.0) - CAFFE_MEAN

    elif method == NormalizationType.TF:
        # TensorFlow: scale to [-1, 1]
        return (image - 0.5) * 2.0

    return image
