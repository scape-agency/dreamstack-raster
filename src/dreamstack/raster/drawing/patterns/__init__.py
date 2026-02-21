# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Pattern Generators
======================================

Generate procedural patterns: checkers, solid colors, noise, gradients.

"""


# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    # pylint: disable=import-outside-toplevel
    from numpy.typing import NDArray


@dataclass
class PatternConfig:
    """Configuration for pattern generation.

    Attributes:
        width: Pattern width in pixels.
        height: Pattern height in pixels.
        channels: Number of color channels (3 for RGB, 4 for RGBA).
        dtype: Data type (uint8 or float32).
    """

    width: int = 512
    height: int = 512
    channels: int = 4
    dtype: Literal["uint8", "float32"] = "uint8"


def constant(
    width: int,
    height: int,
    color: tuple[int, int, int] | tuple[int, int, int, int] = (
        255,
        255,
        255,
        255,
    ),
    *,
    channels: int = 4,
) -> NDArray[np.uint8]:
    """Create a solid color image.

    Generates a uniform color image of specified dimensions.

    Args:
        width: Image width in pixels.
        height: Image height in pixels.
        color: RGBA color tuple (0-255 values).
        channels: Number of channels (3 or 4).

    Returns:
        Solid color image.

    Example:
        >>> # Create red 1920x1080 image
        >>> red = constant(1920, 1080, (255, 0, 0, 255))
        >>>
        >>> # Create semi-transparent blue
        >>> blue = constant(500, 500, (0, 0, 255, 128))
    """
    image = np.zeros((height, width, channels), dtype=np.uint8)

    for i, val in enumerate(color[:channels]):
        image[:, :, i] = val

    return image


def solid(
    width: int,
    height: int,
    color: tuple[int, int, int] | tuple[int, int, int, int] = (
        255,
        255,
        255,
        255,
    ),
) -> NDArray[np.uint8]:
    """Alias for constant(). Create a solid color image.

    Args:
        width: Image width in pixels.
        height: Image height in pixels.
        color: RGBA color tuple.

    Returns:
        Solid color image.
    """
    channels = len(color)
    return constant(width, height, color, channels=channels)


def checker(
    width: int,
    height: int,
    color_a: tuple[int, int, int] = (255, 255, 255),
    color_b: tuple[int, int, int] = (0, 0, 0),
    *,
    cell_size: int = 32,
    cells: int | None = None,
) -> NDArray[np.uint8]:
    """Create a checkerboard pattern.

    Generates alternating colored squares in a grid pattern.

    Args:
        width: Image width in pixels.
        height: Image height in pixels.
        color_a: First color (RGB, 0-255).
        color_b: Second color (RGB, 0-255).
        cell_size: Size of each cell in pixels.
        cells: Number of cells per row (overrides cell_size if provided).

    Returns:
        Checkerboard pattern image.

    Example:
        >>> # Default black/white checker
        >>> check = checker(512, 512)
        >>>
        >>> # Red/blue checker with large cells
        >>> colored = checker(1000, 1000, (255, 0, 0), (0, 0, 255), cell_size=100)
        >>>
        >>> # 8x8 chess board style
        >>> chess = checker(800, 800, cells=8)
    """
    # Calculate cell size from cell count if provided
    if cells is not None:
        cell_size = max(1, width // cells)

    image = np.zeros((height, width, 3), dtype=np.uint8)

    for y in range(height):
        for x in range(width):
            # Determine which cell we're in
            cell_x = x // cell_size
            cell_y = y // cell_size

            # Alternate colors based on cell position
            if (cell_x + cell_y) % 2 == 0:
                image[y, x] = color_a
            else:
                image[y, x] = color_b

    return image


def checker_fast(
    width: int,
    height: int,
    color_a: tuple[int, int, int] = (255, 255, 255),
    color_b: tuple[int, int, int] = (0, 0, 0),
    *,
    cell_size: int = 32,
) -> NDArray[np.uint8]:
    """Create a checkerboard pattern (optimized version).

    Uses numpy broadcasting for faster generation on large images.

    Args:
        width: Image width in pixels.
        height: Image height in pixels.
        color_a: First color (RGB).
        color_b: Second color (RGB).
        cell_size: Size of each cell in pixels.

    Returns:
        Checkerboard pattern image.
    """
    # Create coordinate grids
    y_coords = np.arange(height) // cell_size
    x_coords = np.arange(width) // cell_size

    # Create checker mask
    y_grid, x_grid = np.meshgrid(y_coords, x_coords, indexing="ij")
    mask = (x_grid + y_grid) % 2

    # Create output image
    image = np.zeros((height, width, 3), dtype=np.uint8)

    # Apply colors
    image[mask == 0] = color_a
    image[mask == 1] = color_b

    return image


def transparency_checker(
    width: int,
    height: int,
    *,
    cell_size: int = 8,
    light: int = 255,
    dark: int = 204,
) -> NDArray[np.uint8]:
    """Create a transparency indicator pattern.

    Creates the classic gray/white checker pattern used to
    indicate transparent areas in image editors.

    Args:
        width: Image width.
        height: Image height.
        cell_size: Checker cell size (default 8px like Photoshop).
        light: Light gray value (default 255).
        dark: Dark gray value (default 204).

    Returns:
        Transparency checker pattern.

    Example:
        >>> bg = transparency_checker(800, 600)
    """
    return checker_fast(
        width,
        height,
        (light, light, light),
        (dark, dark, dark),
        cell_size=cell_size,
    )


def noise(
    width: int,
    height: int,
    *,
    channels: int = 3,
    low: int = 0,
    high: int = 255,
    seed: int | None = None,
) -> NDArray[np.uint8]:
    """Generate random noise pattern.

    Creates uniform random noise.

    Args:
        width: Image width.
        height: Image height.
        channels: Number of color channels.
        low: Minimum value.
        high: Maximum value.
        seed: Random seed for reproducibility.

    Returns:
        Random noise image.

    Example:
        >>> # Random RGB noise
        >>> noisy = noise(512, 512)
        >>>
        >>> # Grayscale noise
        >>> gray_noise = noise(512, 512, channels=1)
    """
    if seed is not None:
        np.random.seed(seed)

    return np.random.randint(
        low, high + 1, (height, width, channels), dtype=np.uint8
    )


def gaussian_noise(
    width: int,
    height: int,
    *,
    mean: float = 128.0,
    std: float = 50.0,
    channels: int = 3,
    seed: int | None = None,
) -> NDArray[np.uint8]:
    """Generate Gaussian (normal distribution) noise.

    Args:
        width: Image width.
        height: Image height.
        mean: Mean value (center of distribution).
        std: Standard deviation (spread).
        channels: Number of color channels.
        seed: Random seed for reproducibility.

    Returns:
        Gaussian noise image.

    Example:
        >>> # Mid-gray centered noise
        >>> noisy = gaussian_noise(512, 512)
    """
    if seed is not None:
        np.random.seed(seed)

    values = np.random.normal(mean, std, (height, width, channels))
    return np.clip(values, 0, 255).astype(np.uint8)


def perlin_noise(
    width: int,
    height: int,
    *,
    scale: float = 50.0,
    octaves: int = 4,
    persistence: float = 0.5,
    seed: int | None = None,
) -> NDArray[np.uint8]:
    """Generate Perlin-like noise pattern.

    Creates smooth, organic-looking noise using
    combined octaves of interpolated noise.

    Args:
        width: Image width.
        height: Image height.
        scale: Noise scale (larger = more zoomed out).
        octaves: Number of noise layers.
        persistence: Amplitude decay per octave.
        seed: Random seed for reproducibility.

    Returns:
        Grayscale Perlin-like noise image.

    Example:
        >>> cloud = perlin_noise(512, 512, scale=100)
    """
    if seed is not None:
        np.random.seed(seed)

    def interpolate(a: float, b: float, t: float) -> float:
        """Smooth interpolation."""
        return a + t * t * (3 - 2 * t) * (b - a)

    def generate_octave(w: int, h: int, freq: float) -> NDArray[np.float32]:
        """Generate single octave of noise."""
        # Grid dimensions
        gw = int(np.ceil(w / freq)) + 2
        gh = int(np.ceil(h / freq)) + 2

        # Random gradients at grid points
        noise_grid = np.random.rand(gh, gw).astype(np.float32)

        # Interpolate
        result = np.zeros((h, w), dtype=np.float32)
        for y in range(h):
            for x in range(w):
                # Grid cell
                gx = x / freq
                gy = y / freq

                x0, y0 = int(gx), int(gy)
                x1, y1 = x0 + 1, y0 + 1

                # Fractional position
                fx, fy = gx - x0, gy - y0

                # Bilinear interpolation
                top = interpolate(noise_grid[y0, x0], noise_grid[y0, x1], fx)
                bot = interpolate(noise_grid[y1, x0], noise_grid[y1, x1], fx)
                result[y, x] = interpolate(top, bot, fy)

        return result

    # Combine octaves
    result = np.zeros((height, width), dtype=np.float32)
    amplitude = 1.0
    total_amp = 0.0
    freq = scale

    for _ in range(octaves):
        result += amplitude * generate_octave(width, height, freq)
        total_amp += amplitude
        amplitude *= persistence
        freq /= 2

    # Normalize to 0-255
    result = result / total_amp
    result = (result * 255).astype(np.uint8)

    # Return as 3-channel grayscale
    return np.stack([result, result, result], axis=-1)


def stripes(
    width: int,
    height: int,
    color_a: tuple[int, int, int] = (255, 255, 255),
    color_b: tuple[int, int, int] = (0, 0, 0),
    *,
    stripe_width: int = 10,
    horizontal: bool = False,
) -> NDArray[np.uint8]:
    """Generate a stripe pattern.

    Creates alternating stripes of two colors.

    Args:
        width: Image width.
        height: Image height.
        color_a: First stripe color.
        color_b: Second stripe color.
        stripe_width: Width of each stripe.
        horizontal: If True, horizontal stripes; else vertical.

    Returns:
        Striped pattern image.

    Example:
        >>> # Vertical stripes
        >>> v_stripes = stripes(512, 512)
        >>>
        >>> # Horizontal red/white stripes
        >>> h_stripes = stripes(512, 512, (255, 0, 0), (255, 255, 255),
        ...                     horizontal=True)
    """
    image = np.zeros((height, width, 3), dtype=np.uint8)

    if horizontal:
        coords = np.arange(height) // stripe_width
        mask = coords % 2
        for y in range(height):
            if mask[y] == 0:
                image[y, :] = color_a
            else:
                image[y, :] = color_b
    else:
        coords = np.arange(width) // stripe_width
        mask = coords % 2
        for x in range(width):
            if mask[x] == 0:
                image[:, x] = color_a
            else:
                image[:, x] = color_b

    return image


def grid(
    width: int,
    height: int,
    *,
    background: tuple[int, int, int] = (255, 255, 255),
    line_color: tuple[int, int, int] = (200, 200, 200),
    cell_size: int = 50,
    line_width: int = 1,
) -> NDArray[np.uint8]:
    """Generate a grid pattern.

    Creates a regular grid with lines at specified intervals.

    Args:
        width: Image width.
        height: Image height.
        background: Background color.
        line_color: Grid line color.
        cell_size: Distance between grid lines.
        line_width: Width of grid lines.

    Returns:
        Grid pattern image.

    Example:
        >>> # Paper-like grid
        >>> paper = grid(800, 600, cell_size=20, line_width=1)
    """
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, :] = background

    half_width = line_width // 2

    # Draw vertical lines
    for x in range(0, width, cell_size):
        x0 = max(0, x - half_width)
        x1 = min(width, x + half_width + 1)
        image[:, x0:x1] = line_color

    # Draw horizontal lines
    for y in range(0, height, cell_size):
        y0 = max(0, y - half_width)
        y1 = min(height, y + half_width + 1)
        image[y0:y1, :] = line_color

    return image
