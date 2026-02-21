# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Draw Text
=========

Text rendering on images.

"""


# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

from dreamstack.raster.drawing.text.text_style import FontWeight, TextStyle


def draw_text(
    image: NDArray[np.uint8],
    text: str,
    position: tuple[int, int],
    style: TextStyle | None = None,
    *,
    color: tuple[int, int, int, int] | None = None,
    font_scale: float | None = None,
) -> NDArray[np.uint8]:
    """Draw text on an image.

    Uses OpenCV's built-in fonts. For custom fonts, use PIL/Pillow.

    Args:
        image: Image to draw on.
        text: Text string to render.
        position: Bottom-left position (x, y).
        style: Text style configuration.
        color: Optional color override (RGBA).
        font_scale: Optional font scale override.

    Returns:
        Image with text drawn.

    Example:
        >>> style = TextStyle(font_scale=2.0, color=(255, 0, 0, 255))
        >>> result = draw_text(image, "Hello!", (10, 50), style)
    """
    result = image.copy()

    if style is None:
        style = TextStyle()

    if color is not None:
        style = style.with_color(color)

    if font_scale is not None:
        style = style.with_size(font_scale)

    # Map font weight to OpenCV font
    font = _get_cv2_font(style.font_weight)

    # Convert color to BGR
    bgr_color = (style.color[2], style.color[1], style.color[0])

    # Handle multi-line text
    lines = text.split("\n")
    x, y = position

    line_height = int(style.font_scale * 30 * style.line_spacing)

    # Draw background if specified
    if style.background_color is not None:
        bounds = text_bounds(text, style)
        pad = style.padding

        bg_x = x - pad
        bg_y = y - bounds[1] - pad
        bg_w = bounds[0] + 2 * pad
        bg_h = bounds[1] * len(lines) + 2 * pad

        bg_bgr = (
            style.background_color[2],
            style.background_color[1],
            style.background_color[0],
        )
        cv2.rectangle(
            result,
            (bg_x, bg_y),
            (bg_x + bg_w, bg_y + bg_h),
            bg_bgr,
            -1,
        )

    # Draw each line
    for i, line in enumerate(lines):
        line_y = y + i * line_height
        cv2.putText(
            result,
            line,
            (x, line_y),
            font,
            style.font_scale,
            bgr_color,
            style.thickness,
            cv2.LINE_AA,
        )

    return result


def text_bounds(
    text: str,
    style: TextStyle | None = None,
) -> tuple[int, int]:
    """Calculate text bounding box size.

    Args:
        text: Text string to measure.
        style: Text style configuration.

    Returns:
        Tuple of (width, height) in pixels.

    Example:
        >>> width, height = text_bounds("Hello World", style)
    """
    if style is None:
        style = TextStyle()

    font = _get_cv2_font(style.font_weight)

    # Handle multi-line
    lines = text.split("\n")
    max_width = 0
    total_height = 0

    for line in lines:
        (w, h), baseline = cv2.getTextSize(
            line,
            font,
            style.font_scale,
            style.thickness,
        )
        max_width = max(max_width, w)
        total_height = max(total_height, h + baseline)

    # Account for line spacing
    if len(lines) > 1:
        line_height = int(style.font_scale * 30 * style.line_spacing)
        total_height = line_height * len(lines)

    return (max_width, total_height)


def _get_cv2_font(weight: FontWeight) -> int:
    """Map FontWeight to OpenCV font constant."""
    fonts = {
        FontWeight.NORMAL: cv2.FONT_HERSHEY_SIMPLEX,
        FontWeight.BOLD: cv2.FONT_HERSHEY_DUPLEX,
        FontWeight.ITALIC: cv2.FONT_HERSHEY_SIMPLEX | cv2.FONT_ITALIC,
        FontWeight.BOLD_ITALIC: cv2.FONT_HERSHEY_DUPLEX | cv2.FONT_ITALIC,
    }
    return fonts.get(weight, cv2.FONT_HERSHEY_SIMPLEX)


def render_text(
    text: str,
    font_family: str = "sans-serif",  # pylint: disable=unused-argument
    font_size: float = 24.0,
    font_color: tuple[int, int, int, int] = (255, 255, 255, 255),
    font_weight: str = "normal",
    font_style: str = "normal",
    line_height: float = 1.5,
    alignment: str = "left",
) -> tuple[NDArray[np.uint8], tuple[int, int, int, int]]:
    """Render text to an RGBA pixel array.

    Creates a standalone text image with transparency.

    Args:
        text: Text string to render.
        font_family: Font family name (ignored, uses OpenCV fonts).
        font_size: Font size in points.
        font_color: RGBA color tuple.
        font_weight: Weight (normal, bold).
        font_style: Style (normal, italic).
        line_height: Line height multiplier.
        alignment: Text alignment (left, center, right).

    Returns:
        Tuple of (RGBA image array, bounding box (x, y, width, height)).

    Example:
        >>> pixels, bounds = render_text("Hello!", font_size=32)
    """
    # Map string weights to enum
    weight_map = {
        "normal": FontWeight.NORMAL,
        "bold": FontWeight.BOLD,
    }
    weight = weight_map.get(font_weight.lower(), FontWeight.NORMAL)

    if font_style.lower() == "italic":
        if weight == FontWeight.BOLD:
            weight = FontWeight.BOLD_ITALIC
        else:
            weight = FontWeight.ITALIC

    # Create style
    font_scale = font_size / 24.0  # OpenCV scale relative to base size
    style = TextStyle(
        font_scale=font_scale,
        color=font_color,
        font_weight=weight,
        line_spacing=line_height,
    )

    # Calculate bounds
    width, height = text_bounds(text, style)

    # Add padding
    padding = 10
    canvas_width = width + 2 * padding
    canvas_height = height + 2 * padding

    # Create transparent canvas
    canvas = np.zeros((canvas_height, canvas_width, 4), dtype=np.uint8)

    # Create temp RGB canvas for text
    temp = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

    # Draw text (white on black for mask)
    font = _get_cv2_font(weight)
    lines = text.split("\n")
    line_h = int(style.font_scale * 30 * style.line_spacing)

    y_offset = int(style.font_scale * 30)  # Baseline offset

    for i, line in enumerate(lines):
        x_pos = padding
        y_pos = padding + y_offset + i * line_h

        # Handle alignment
        if alignment == "center":
            line_w, _ = text_bounds(line, style)
            x_pos = (canvas_width - line_w) // 2
        elif alignment == "right":
            line_w, _ = text_bounds(line, style)
            x_pos = canvas_width - line_w - padding

        cv2.putText(  # pylint: disable=no-member
            temp,
            line,
            (x_pos, y_pos),
            font,
            style.font_scale,
            (255, 255, 255),
            style.thickness,
            cv2.LINE_AA,
        )

    # Create alpha from grayscale
    gray = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)  # pylint: disable=no-member

    # Set colors and alpha
    canvas[:, :, 0] = font_color[0]  # R
    canvas[:, :, 1] = font_color[1]  # G
    canvas[:, :, 2] = font_color[2]  # B
    canvas[:, :, 3] = np.asarray(
        gray * (font_color[3] / 255.0), dtype=np.uint8
    )  # A

    bounds = (0, 0, canvas_width, canvas_height)
    return canvas, bounds
