# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Bevel Emboss
============

Create bevel and emboss effects.

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

from dreamstack.raster.effects.bevel.bevel_style import (  # pylint: disable=wrong-import-position
    BevelStyle,
    BevelTechnique,
    BevelType,
)


def bevel_emboss(
    image: NDArray[np.uint8],
    style: BevelStyle | None = None,
) -> NDArray[np.uint8]:
    """Apply bevel and emboss effect to an image.

    Creates a 3D raised or sunken appearance using light/shadow.

    Args:
        image: Input image with alpha channel.
        style: Bevel style configuration.

    Returns:
        Image with bevel/emboss applied.

    Example:
        >>> style = BevelStyle(size=10, depth=75)
        >>> result = bevel_emboss(image, style)
    """
    if style is None:
        style = BevelStyle()

    # Ensure BGRA
    if image.ndim == 2:
        img = cv2.cvtColor(
            image, cv2.COLOR_GRAY2BGRA
        )  # pylint: disable=no-member
    elif image.shape[2] == 3:
        img = cv2.cvtColor(
            image, cv2.COLOR_BGR2BGRA
        )  # pylint: disable=no-member
    else:
        img = image.copy()

    _h, _w = img.shape[:2]  # Reserved for future use
    result = img.astype(np.float32)

    # Get alpha mask
    alpha = img[:, :, 3].astype(np.float32) / 255.0

    # Calculate light direction
    angle_rad = np.radians(style.angle)
    altitude_rad = np.radians(style.altitude)

    light_x = np.cos(angle_rad) * np.cos(altitude_rad)
    light_y = np.sin(angle_rad) * np.cos(altitude_rad)

    # Create height map based on bevel type
    if style.bevel_type == BevelType.INNER_BEVEL:
        height_map = _create_inner_bevel(alpha, style.size, style.technique)
    elif style.bevel_type == BevelType.OUTER_BEVEL:
        height_map = _create_outer_bevel(alpha, style.size, style.technique)
    elif style.bevel_type == BevelType.EMBOSS:
        height_map = _create_emboss(alpha, style.size, style.technique)
    else:  # PILLOW_EMBOSS
        height_map = _create_pillow_emboss(alpha, style.size, style.technique)

    # Apply soften
    if style.soften > 0:
        height_map = cv2.GaussianBlur(  # pylint: disable=no-member
            height_map, (0, 0), style.soften
        )

    # Calculate normals from height map
    # pylint: disable=no-member
    sobel_x = cv2.Sobel(height_map, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(height_map, cv2.CV_32F, 0, 1, ksize=3)
    # pylint: enable=no-member

    # Calculate lighting
    depth_factor = style.depth / 100.0
    lighting = sobel_x * light_x + sobel_y * light_y
    lighting = lighting * depth_factor

    # Separate highlights and shadows
    highlights = np.clip(lighting, 0, 1)
    shadows = np.clip(-lighting, 0, 1)

    # Mask to content area
    highlights = highlights * alpha * style.highlight_opacity
    shadows = shadows * alpha * style.shadow_opacity

    # Apply highlight color (screen blend)
    hl_color = np.array(
        [
            style.highlight_color[2],
            style.highlight_color[1],
            style.highlight_color[0],
        ],
        dtype=np.float32,
    )

    sh_color = np.array(
        [
            style.shadow_color[2],
            style.shadow_color[1],
            style.shadow_color[0],
        ],
        dtype=np.float32,
    )

    for c in range(3):
        base = result[:, :, c] / 255.0

        # Screen blend for highlights
        hl = 1 - (1 - base) * (1 - (hl_color[c] / 255.0) * highlights)

        # Multiply blend for shadows
        sh = hl * (1 - shadows) + hl * (sh_color[c] / 255.0) * shadows

        result[:, :, c] = sh * 255

    return np.clip(result, 0, 255).astype(np.uint8)


def _create_inner_bevel(
    alpha: NDArray[np.float32],
    size: int,
    technique: BevelTechnique,
) -> NDArray[np.float32]:
    """Create height map for inner bevel."""
    height = np.zeros_like(alpha)
    kernel = np.ones((3, 3), dtype=np.uint8)

    if technique == BevelTechnique.SMOOTH:
        # Distance from edge
        for i in range(size):
            eroded = cv2.erode(
                alpha, kernel, iterations=i + 1
            )  # pylint: disable=no-member
            height = np.maximum(height, eroded * (i + 1) / size)
    else:
        # Chisel - hard edge
        eroded = cv2.erode(
            alpha, kernel, iterations=size
        )  # pylint: disable=no-member
        height = eroded

    return height * alpha


def _create_outer_bevel(
    alpha: NDArray[np.float32],
    size: int,
    technique: BevelTechnique,
) -> NDArray[np.float32]:
    """Create height map for outer bevel."""
    height = np.zeros_like(alpha)
    kernel = np.ones((3, 3), dtype=np.uint8)

    if technique == BevelTechnique.SMOOTH:
        for i in range(size):
            dilated = cv2.dilate(
                alpha, kernel, iterations=i + 1
            )  # pylint: disable=no-member
            outside = dilated - alpha
            height = np.maximum(height, outside * (size - i) / size)
    else:
        dilated = cv2.dilate(
            alpha, kernel, iterations=size
        )  # pylint: disable=no-member
        height = dilated - alpha

    return height + alpha


def _create_emboss(
    alpha: NDArray[np.float32],
    size: int,
    _technique: BevelTechnique,
) -> NDArray[np.float32]:
    """Create height map for emboss."""
    return (alpha * (size / 10.0)).astype(np.float32)


def _create_pillow_emboss(
    alpha: NDArray[np.float32],
    size: int,
    technique: BevelTechnique,
) -> NDArray[np.float32]:
    """Create height map for pillow emboss."""
    inner = _create_inner_bevel(alpha, size // 2, technique)
    outer = _create_outer_bevel(alpha, size // 2, technique)
    return (inner - (outer - alpha) * 0.5).astype(np.float32)
