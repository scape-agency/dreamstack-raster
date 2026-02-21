# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Pixelation Effects
==================

Pixelation, mosaic, and color quantization effects for stylized rendering.

"""


# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class PixelateConfig:
    """Configuration for pixelation effect.

    Attributes:
        pixel_size: Size of each pixel block.
        preserve_aspect: Maintain aspect ratio when resizing.
        interpolation: Interpolation method (nearest, linear, cubic).
    """

    pixel_size: int = 10
    preserve_aspect: bool = True
    interpolation: str = "nearest"


def pixelate(
    image: NDArray[np.uint8],
    pixel_size: int = 10,
    *,
    _preserve_colors: bool = False,
) -> NDArray[np.uint8]:
    """Apply pixelation effect to an image.

    Creates a retro pixel-art style effect by reducing resolution
    and scaling back up.

    Args:
        image: Input image (BGR, 3 channels).
        pixel_size: Size of each pixel block (larger = more pixelated).
        preserve_colors: Keep original color values in each block.

    Returns:
        Pixelated image.

    Example:
        >>> from dreamstack.raster.filters.stylize import pixelate
        >>> pixelated = pixelate(image, pixel_size=8)
    """
    import cv2

    if pixel_size < 2:
        return image.copy()

    h, w = image.shape[:2]

    # Calculate new size
    new_w = max(1, w // pixel_size)
    new_h = max(1, h // pixel_size)

    # Downscale
    small = cv2.resize(  # pylint: disable=no-member
        image,
        (new_w, new_h),
        interpolation=cv2.INTER_LINEAR,  # pylint: disable=no-member
    )

    # Upscale with nearest neighbor
    result = cv2.resize(  # pylint: disable=no-member
        small,
        (w, h),
        interpolation=cv2.INTER_NEAREST,  # pylint: disable=no-member
    )

    return np.asarray(result, dtype=np.uint8)


def mosaic(
    image: NDArray[np.uint8],
    tile_size: int = 16,
    *,
    show_grid: bool = False,
    grid_color: tuple[int, int, int] = (128, 128, 128),
    grid_thickness: int = 1,
) -> NDArray[np.uint8]:
    """Apply mosaic/tile effect with optional grid lines.

    Similar to pixelate but with optional visible grid separating tiles.

    Args:
        image: Input image (BGR, 3 channels).
        tile_size: Size of each tile.
        show_grid: Draw grid lines between tiles.
        grid_color: Color of grid lines (BGR).
        grid_thickness: Thickness of grid lines.

    Returns:
        Mosaic image.

    Example:
        >>> result = mosaic(image, tile_size=20, show_grid=True)
    """
    import cv2

    result = pixelate(image, tile_size)

    if show_grid:
        h, w = result.shape[:2]

        # Draw vertical lines
        for x in range(0, w, tile_size):
            cv2.line(
                result, (x, 0), (x, h), grid_color, grid_thickness
            )  # pylint: disable=no-member

        # Draw horizontal lines
        for y in range(0, h, tile_size):
            cv2.line(
                result, (0, y), (w, y), grid_color, grid_thickness
            )  # pylint: disable=no-member

    return result


def quantize_colors(
    image: NDArray[np.uint8],
    n_colors: int = 16,
    *,
    _method: str = "kmeans",
    max_iterations: int = 10,
) -> NDArray[np.uint8]:
    """Reduce image to a limited color palette.

    Uses k-means clustering to find optimal color palette.

    Args:
        image: Input image (BGR, 3 channels).
        n_colors: Number of colors in the output palette.
        method: Quantization method (kmeans only for now).
        max_iterations: Maximum k-means iterations.

    Returns:
        Color-quantized image.

    Example:
        >>> posterized = quantize_colors(image, n_colors=8)
    """
    import cv2

    # Reshape for k-means
    h, w = image.shape[:2]
    pixels = image.reshape(-1, 3).astype(np.float32)

    # K-means clustering
    criteria = (
        getattr(cv2, "TERM_CRITERIA_EPS")
        + getattr(cv2, "TERM_CRITERIA_MAX_ITER"),
        max_iterations,
        1.0,
    )

    _, labels, palette = cv2.kmeans(  # pylint: disable=no-member
        pixels,
        n_colors,
        None,  # type: ignore[arg-type]
        criteria,
        10,
        getattr(cv2, "KMEANS_PP_CENTERS"),
    )

    # Map pixels to palette colors
    result = palette[labels.flatten()].reshape(h, w, 3)

    return result.astype(np.uint8)


def match_to_palette(
    image: NDArray[np.uint8],
    palette: list[tuple[int, int, int]],
    *,
    color_space: str = "rgb",
) -> NDArray[np.uint8]:
    """Map image colors to a fixed palette.

    Each pixel is replaced with the closest color from the palette.

    Args:
        image: Input image (BGR, 3 channels).
        palette: List of RGB color tuples.
        color_space: Color space for distance calculation (rgb, lab).

    Returns:
        Image with colors mapped to palette.

    Example:
        >>> # Map to grayscale palette
        >>> grays = [(i, i, i) for i in range(0, 256, 32)]
        >>> result = match_to_palette(image, grays)
        >>>
        >>> # Map to custom palette
        >>> retro = [(0, 0, 0), (255, 255, 255), (255, 0, 0), (0, 255, 0)]
        >>> result = match_to_palette(image, retro)
    """
    import cv2

    if not palette:
        return image.copy()

    # Convert palette to array (BGR for OpenCV)
    palette_bgr = np.array([[b, g, r] for r, g, b in palette], dtype=np.uint8)

    h, w = image.shape[:2]

    if color_space == "lab":
        # Convert to LAB for perceptual color matching
        img_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(
            np.float32
        )  # pylint: disable=no-member
        pixels = img_lab.reshape(-1, 3)

        # Convert palette to LAB
        palette_reshaped = palette_bgr.reshape((1, -1, 3))
        palette_lab = cv2.cvtColor(
            palette_reshaped, cv2.COLOR_BGR2LAB
        ).astype(  # pylint: disable=no-member
            np.float32
        )
        palette_lab = palette_lab.reshape(-1, 3)
    else:
        pixels = image.reshape(-1, 3).astype(np.float32)
        palette_lab = palette_bgr.astype(np.float32)

    # Find closest palette color for each pixel
    # Using broadcasting for efficiency
    distances = np.zeros((len(pixels), len(palette_lab)), dtype=np.float32)

    for i, color in enumerate(palette_lab):
        diff = pixels - color
        distances[:, i] = np.sum(diff**2, axis=1)

    indices = np.argmin(distances, axis=1)

    # Map to palette colors
    result = palette_bgr[indices].reshape(h, w, 3)

    return result


def posterize(
    image: NDArray[np.uint8],
    levels: int = 4,
) -> NDArray[np.uint8]:
    """Reduce color depth by posterizing.

    Reduces the number of color levels per channel.

    Args:
        image: Input image (BGR, 3 channels).
        levels: Number of levels per channel (2-256).

    Returns:
        Posterized image.

    Example:
        >>> result = posterize(image, levels=4)  # 64 total colors (4^3)
    """
    if levels < 2:
        levels = 2
    if levels > 256:
        levels = 256

    # Calculate step size
    step = 256 // levels

    # Quantize
    result = (image // step) * step

    # Adjust maximum to 255
    if step > 1:
        result = np.clip(result + step // 2, 0, 255)

    return result.astype(np.uint8)


def pixelate_and_quantize(
    image: NDArray[np.uint8],
    pixel_size: int = 8,
    n_colors: int = 16,
) -> NDArray[np.uint8]:
    """Combined pixelation and color quantization.

    Creates a retro pixel-art style with limited colors.
    Process: quantize colors first, then pixelate.

    Args:
        image: Input image (BGR, 3 channels).
        pixel_size: Size of pixel blocks.
        n_colors: Number of colors in palette.

    Returns:
        Pixelated and quantized image.

    Example:
        >>> retro = pixelate_and_quantize(image, pixel_size=4, n_colors=16)
    """
    quantized = quantize_colors(image, n_colors)
    return pixelate(quantized, pixel_size)
