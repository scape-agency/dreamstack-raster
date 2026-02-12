# -*- coding: utf-8 -*-

"""Apply affine matrix operation."""

from __future__ import annotations

from typing import Optional, Tuple, Union

import cv2
import numpy as np
from numpy.typing import NDArray


def apply_affine_matrix(
    image: NDArray[np.uint8],
    matrix: NDArray[np.float32],
    *,
    output_size: Optional[Tuple[int, int]] = None,
    border_mode: str = "constant",
    border_value: Union[int, Tuple[int, int, int]] = 0,
) -> NDArray[np.uint8]:
    """Apply a 2x3 affine transformation matrix.

    Low-level function for applying any affine transformation.

    Parameters
    ----------
    image : NDArray[np.uint8]
        Input image.
    matrix : NDArray[np.float32]
        2x3 affine transformation matrix.
    output_size : tuple[int, int], optional
        Output (width, height). Default is same as input.
    border_mode : str, optional
        Border handling mode. Default is "constant".
    border_value : int or tuple, optional
        Fill value. Default is 0.

    Returns
    -------
    NDArray[np.uint8]
        Transformed image.
    """
    h, w = image.shape[:2]
    if output_size is None:
        output_size = (w, h)

    border_modes = {
        "constant": cv2.BORDER_CONSTANT,
        "reflect": cv2.BORDER_REFLECT,
        "replicate": cv2.BORDER_REPLICATE,
        "wrap": cv2.BORDER_WRAP,
    }
    border = border_modes.get(border_mode, cv2.BORDER_CONSTANT)

    return cv2.warpAffine(
        image, matrix, output_size, borderMode=border, borderValue=border_value
    )
