# -*- coding: utf-8 -*-

"""
Image Translation Operations
============================

Image translation (shifting) operations using affine transformations.
Essential for data augmentation in machine learning pipelines.

"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import cv2
import numpy as np
from numpy.typing import NDArray


def translate(
    image: NDArray[np.uint8],
    tx: int = 0,
    ty: int = 0,
    *,
    border_mode: str = "constant",
    border_value: Union[int, Tuple[int, int, int]] = 0,
) -> NDArray[np.uint8]:
    """Translate (shift) an image by pixel offset.

    Moves the image by the specified number of pixels in
    the x and y directions using an affine transformation.

    Parameters
    ----------
    image : NDArray[np.uint8]
        Input image (H, W) or (H, W, C).
    tx : int, optional
        Translation in x direction (positive = right). Default is 0.
    ty : int, optional
        Translation in y direction (positive = down). Default is 0.
    border_mode : str, optional
        How to fill empty areas:
        - "constant": Fill with border_value (default)
        - "reflect": Reflect the image at the border
        - "replicate": Replicate edge pixels
        - "wrap": Wrap around to opposite edge
    border_value : int or tuple, optional
        Fill value for constant border mode. Default is 0 (black).

    Returns
    -------
    NDArray[np.uint8]
        Translated image, same shape as input.

    Examples
    --------
    >>> # Shift image 100 pixels right and 50 pixels down
    >>> translated = translate(img, tx=100, ty=50)
    
    >>> # Shift with reflection at borders
    >>> translated = translate(img, tx=50, ty=25, border_mode="reflect")
    """
    h, w = image.shape[:2]

    # Create translation matrix
    translation_matrix = np.float32([
        [1, 0, tx],
        [0, 1, ty]
    ])

    # Get border mode
    border_modes = {
        "constant": cv2.BORDER_CONSTANT,
        "reflect": cv2.BORDER_REFLECT,
        "replicate": cv2.BORDER_REPLICATE,
        "wrap": cv2.BORDER_WRAP,
    }
    border = border_modes.get(border_mode, cv2.BORDER_CONSTANT)

    return cv2.warpAffine(
        image,
        translation_matrix,
        (w, h),
        borderMode=border,
        borderValue=border_value
    )


def translate_percentage(
    image: NDArray[np.uint8],
    tx_percent: float = 0.0,
    ty_percent: float = 0.0,
    *,
    border_mode: str = "constant",
    border_value: Union[int, Tuple[int, int, int]] = 0,
) -> NDArray[np.uint8]:
    """Translate image by percentage of dimensions.

    Shifts image by a fraction of its width/height.
    Useful for consistent transformations across different image sizes.

    Parameters
    ----------
    image : NDArray[np.uint8]
        Input image.
    tx_percent : float, optional
        Translation as fraction of width (-1.0 to 1.0). Default is 0.0.
    ty_percent : float, optional
        Translation as fraction of height (-1.0 to 1.0). Default is 0.0.
    border_mode : str, optional
        Border handling mode. Default is "constant".
    border_value : int or tuple, optional
        Fill value for constant border. Default is 0.

    Returns
    -------
    NDArray[np.uint8]
        Translated image.

    Examples
    --------
    >>> # Shift image by 25% of width and 12.5% of height
    >>> translated = translate_percentage(img, tx_percent=0.25, ty_percent=0.125)
    """
    h, w = image.shape[:2]
    tx = int(w * tx_percent)
    ty = int(h * ty_percent)
    return translate(image, tx, ty, border_mode=border_mode, border_value=border_value)


def random_translate(
    image: NDArray[np.uint8],
    max_tx: int = 50,
    max_ty: int = 50,
    *,
    border_mode: str = "constant",
    border_value: Union[int, Tuple[int, int, int]] = 0,
    seed: Optional[int] = None,
) -> NDArray[np.uint8]:
    """Apply random translation for data augmentation.

    Randomly shifts the image within specified bounds.
    Essential for training robust ML models.

    Parameters
    ----------
    image : NDArray[np.uint8]
        Input image.
    max_tx : int, optional
        Maximum translation in x direction. Default is 50.
    max_ty : int, optional
        Maximum translation in y direction. Default is 50.
    border_mode : str, optional
        Border handling mode. Default is "constant".
    border_value : int or tuple, optional
        Fill value for constant border. Default is 0.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    NDArray[np.uint8]
        Randomly translated image.

    Examples
    --------
    >>> # Random shift up to 50 pixels in any direction
    >>> augmented = random_translate(img, max_tx=50, max_ty=50)
    """
    if seed is not None:
        np.random.seed(seed)

    tx = np.random.randint(-max_tx, max_tx + 1)
    ty = np.random.randint(-max_ty, max_ty + 1)

    return translate(image, tx, ty, border_mode=border_mode, border_value=border_value)


def center_to_origin(
    image: NDArray[np.uint8],
    *,
    border_mode: str = "constant",
    border_value: Union[int, Tuple[int, int, int]] = 0,
) -> NDArray[np.uint8]:
    """Move image center to origin (top-left).

    Shifts the image so that the center point is at (0, 0).
    Useful for certain geometric transformations.

    Parameters
    ----------
    image : NDArray[np.uint8]
        Input image.
    border_mode : str, optional
        Border handling mode. Default is "constant".
    border_value : int or tuple, optional
        Fill value for constant border. Default is 0.

    Returns
    -------
    NDArray[np.uint8]
        Translated image with center at origin.
    """
    h, w = image.shape[:2]
    tx = -w // 2
    ty = -h // 2
    return translate(image, tx, ty, border_mode=border_mode, border_value=border_value)


def get_translation_matrix(tx: int, ty: int) -> NDArray[np.float32]:
    """Get affine translation matrix.

    Creates a 2x3 affine transformation matrix for translation.
    Can be combined with other transformations.

    Parameters
    ----------
    tx : int
        Translation in x direction.
    ty : int
        Translation in y direction.

    Returns
    -------
    NDArray[np.float32]
        2x3 translation matrix.

    Examples
    --------
    >>> matrix = get_translation_matrix(100, 50)
    >>> translated = cv2.warpAffine(img, matrix, (w, h))
    """
    return np.float32([
        [1, 0, tx],
        [0, 1, ty]
    ])


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
        image,
        matrix,
        output_size,
        borderMode=border,
        borderValue=border_value
    )
