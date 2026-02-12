"""Preprocess operation."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .bgr_to_rgb import bgr_to_rgb
from .models.normalization_type import NormalizationType
from .normalize import normalize
from .resize_for_model import resize_for_model
from .to_grayscale import to_grayscale


def preprocess(
    image: NDArray[np.uint8],
    *,
    target_size: tuple[int, int] | None = None,
    normalization: NormalizationType | str = NormalizationType.MINMAX,
    color_mode: str = "rgb",
    preserve_aspect: bool = True,
    pad_color: tuple[int, int, int] = (0, 0, 0),
) -> NDArray[np.float32]:
    """Complete preprocessing pipeline for ML models.

    Applies resizing, color conversion, and normalization in one step.

    Parameters
    ----------
    image : NDArray[np.uint8]
        Input image (BGR from OpenCV).
    target_size : tuple[int, int], optional
        Target (width, height). None = no resize.
    normalization : NormalizationType or str, optional
        Normalization method. Default is minmax.
    color_mode : str, optional
        Output color mode: "rgb", "bgr", or "gray".
    preserve_aspect : bool, optional
        Maintain aspect ratio when resizing.
    pad_color : tuple, optional
        Padding color.

    Returns
    -------
    NDArray[np.float32]
        Preprocessed image ready for model input.

    Examples
    --------
    >>> # Preprocess for ResNet
    >>> preprocessed = preprocess(
    ...     img,
    ...     target_size=(224, 224),
    ...     normalization="imagenet",
    ...     color_mode="rgb"
    ... )
    """
    result = image.copy()

    # Color conversion
    if color_mode == "rgb" and result.ndim == 3:
        result = bgr_to_rgb(result)
    elif color_mode == "gray":
        result = to_grayscale(result)

    # Resize
    if target_size is not None:
        result = resize_for_model(
            result,
            target_size,
            preserve_aspect=preserve_aspect,
            pad_color=pad_color,
        )

    # Normalize
    result = normalize(result, normalization)

    return result
