# -*- coding: utf-8 -*-

"""
Image Rotation Operations
=========================

Comprehensive rotation and flipping operations for images.
Supports arbitrary angles, fixed rotations, and flip operations.

"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import cv2
import numpy as np
from numpy.typing import NDArray


def rotate(
    image: NDArray[np.uint8],
    angle: float,
    *,
    center: Optional[Tuple[float, float]] = None,
    scale: float = 1.0,
    border_mode: str = "constant",
    border_value: Union[int, Tuple[int, int, int]] = 0,
    expand: bool = False,
) -> NDArray[np.uint8]:
    """Rotate image by arbitrary angle.

    Rotates the image around a center point by the specified angle.

    Parameters
    ----------
    image : NDArray[np.uint8]
        Input image (H, W) or (H, W, C).
    angle : float
        Rotation angle in degrees. Positive = counter-clockwise.
    center : tuple[float, float], optional
        Rotation center (x, y). Default is image center.
    scale : float, optional
        Scaling factor to apply during rotation. Default is 1.0.
    border_mode : str, optional
        How to fill empty areas:
        - "constant": Fill with border_value (default)
        - "reflect": Reflect at border
        - "replicate": Replicate edge pixels
    border_value : int or tuple, optional
        Fill value for constant border. Default is 0.
    expand : bool, optional
        If True, expand output to contain full rotated image.
        Default is False (maintain original size).

    Returns
    -------
    NDArray[np.uint8]
        Rotated image.

    Examples
    --------
    >>> # Rotate 45 degrees counter-clockwise
    >>> rotated = rotate(img, 45)

    >>> # Rotate 90 degrees clockwise around top-left
    >>> rotated = rotate(img, -90, center=(0, 0))

    >>> # Rotate and scale down by 50%
    >>> rotated = rotate(img, 180, scale=0.5)
    """
    h, w = image.shape[:2]

    if center is None:
        center = (w / 2, h / 2)

    # Get rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)

    # Determine output size
    if expand:
        # Calculate new bounding box
        cos_val = abs(rotation_matrix[0, 0])
        sin_val = abs(rotation_matrix[0, 1])
        new_w = int((h * sin_val) + (w * cos_val))
        new_h = int((h * cos_val) + (w * sin_val))
        # Adjust center
        rotation_matrix[0, 2] += (new_w - w) / 2
        rotation_matrix[1, 2] += (new_h - h) / 2
        output_size = (new_w, new_h)
    else:
        output_size = (w, h)

    # Get border mode
    border_modes = {
        "constant": cv2.BORDER_CONSTANT,
        "reflect": cv2.BORDER_REFLECT,
        "replicate": cv2.BORDER_REPLICATE,
    }
    border = border_modes.get(border_mode, cv2.BORDER_CONSTANT)

    return cv2.warpAffine(
        image,
        rotation_matrix,
        output_size,
        borderMode=border,
        borderValue=border_value,
    )


def rotate_90(image: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Rotate image 90 degrees counter-clockwise.

    Parameters
    ----------
    image : NDArray[np.uint8]
        Input image.

    Returns
    -------
    NDArray[np.uint8]
        Rotated image with swapped dimensions.
    """
    return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)


def rotate_180(image: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Rotate image 180 degrees.

    Parameters
    ----------
    image : NDArray[np.uint8]
        Input image.

    Returns
    -------
    NDArray[np.uint8]
        Rotated image.
    """
    return cv2.rotate(image, cv2.ROTATE_180)


def rotate_270(image: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Rotate image 270 degrees counter-clockwise (90 clockwise).

    Parameters
    ----------
    image : NDArray[np.uint8]
        Input image.

    Returns
    -------
    NDArray[np.uint8]
        Rotated image with swapped dimensions.
    """
    return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)


def flip_horizontal(image: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Flip image horizontally (mirror).

    Parameters
    ----------
    image : NDArray[np.uint8]
        Input image.

    Returns
    -------
    NDArray[np.uint8]
        Horizontally flipped image.

    Examples
    --------
    >>> mirrored = flip_horizontal(img)
    """
    return cv2.flip(image, 1)


def flip_vertical(image: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Flip image vertically.

    Parameters
    ----------
    image : NDArray[np.uint8]
        Input image.

    Returns
    -------
    NDArray[np.uint8]
        Vertically flipped image.
    """
    return cv2.flip(image, 0)


def flip_both(image: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Flip image both horizontally and vertically.

    Equivalent to 180 degree rotation.

    Parameters
    ----------
    image : NDArray[np.uint8]
        Input image.

    Returns
    -------
    NDArray[np.uint8]
        Flipped image.
    """
    return cv2.flip(image, -1)


def arbitrary_rotate(
    image: NDArray[np.uint8],
    angle: float,
    *,
    keep_aspect: bool = True,
    border_value: Union[int, Tuple[int, int, int]] = 0,
) -> NDArray[np.uint8]:
    """Rotate image by arbitrary angle, preserving full content.

    Always expands canvas to contain the entire rotated image.

    Parameters
    ----------
    image : NDArray[np.uint8]
        Input image.
    angle : float
        Rotation angle in degrees.
    keep_aspect : bool, optional
        If True, maintain aspect ratio. Default is True.
    border_value : int or tuple, optional
        Background fill color. Default is 0.

    Returns
    -------
    NDArray[np.uint8]
        Rotated image with expanded canvas.
    """
    return rotate(image, angle, expand=True, border_value=border_value)


def random_rotate(
    image: NDArray[np.uint8],
    max_angle: float = 30.0,
    *,
    seed: Optional[int] = None,
    border_value: Union[int, Tuple[int, int, int]] = 0,
) -> NDArray[np.uint8]:
    """Apply random rotation for data augmentation.

    Parameters
    ----------
    image : NDArray[np.uint8]
        Input image.
    max_angle : float, optional
        Maximum rotation angle in degrees. Default is 30.
    seed : int, optional
        Random seed for reproducibility.
    border_value : int or tuple, optional
        Background fill. Default is 0.

    Returns
    -------
    NDArray[np.uint8]
        Randomly rotated image.

    Examples
    --------
    >>> # Random rotation between -30 and +30 degrees
    >>> augmented = random_rotate(img, max_angle=30)
    """
    if seed is not None:
        np.random.seed(seed)

    angle = np.random.uniform(-max_angle, max_angle)
    return rotate(image, angle, border_value=border_value)


def get_rotation_matrix(
    center: Tuple[float, float],
    angle: float,
    scale: float = 1.0,
) -> NDArray[np.float32]:
    """Get 2x3 rotation matrix.

    Parameters
    ----------
    center : tuple[float, float]
        Rotation center (x, y).
    angle : float
        Rotation angle in degrees.
    scale : float, optional
        Scaling factor. Default is 1.0.

    Returns
    -------
    NDArray[np.float32]
        2x3 rotation matrix.

    Examples
    --------
    >>> h, w = img.shape[:2]
    >>> matrix = get_rotation_matrix((w/2, h/2), 45, scale=0.8)
    >>> rotated = cv2.warpAffine(img, matrix, (w, h))
    """
    return cv2.getRotationMatrix2D(center, angle, scale)


def rotate_point(
    point: Tuple[float, float],
    angle: float,
    center: Tuple[float, float] = (0, 0),
) -> Tuple[float, float]:
    """Rotate a point around a center.

    Useful for rotating coordinates (e.g., landmarks) along with images.

    Parameters
    ----------
    point : tuple[float, float]
        Point to rotate (x, y).
    angle : float
        Rotation angle in degrees.
    center : tuple[float, float], optional
        Center of rotation. Default is origin.

    Returns
    -------
    tuple[float, float]
        Rotated point coordinates.

    Examples
    --------
    >>> # Rotate landmark along with image
    >>> rotated_img = rotate(img, 45)
    >>> rotated_point = rotate_point((100, 50), 45, center=(w/2, h/2))
    """
    rad = np.deg2rad(angle)
    cos_val = np.cos(rad)
    sin_val = np.sin(rad)

    # Translate to origin
    x = point[0] - center[0]
    y = point[1] - center[1]

    # Rotate
    new_x = x * cos_val - y * sin_val
    new_y = x * sin_val + y * cos_val

    # Translate back
    return (new_x + center[0], new_y + center[1])


def rotate_points(
    points: NDArray[np.float32],
    angle: float,
    center: Tuple[float, float] = (0, 0),
) -> NDArray[np.float32]:
    """Rotate multiple points around a center.

    Parameters
    ----------
    points : NDArray[np.float32]
        Points array of shape (N, 2).
    angle : float
        Rotation angle in degrees.
    center : tuple[float, float], optional
        Center of rotation.

    Returns
    -------
    NDArray[np.float32]
        Rotated points array.
    """
    rad = np.deg2rad(angle)
    cos_val = np.cos(rad)
    sin_val = np.sin(rad)

    # Translate to origin
    translated = points - np.array(center)

    # Rotation matrix
    rotation = np.array([[cos_val, -sin_val], [sin_val, cos_val]])

    # Rotate and translate back
    rotated = translated @ rotation.T
    return rotated + np.array(center)
