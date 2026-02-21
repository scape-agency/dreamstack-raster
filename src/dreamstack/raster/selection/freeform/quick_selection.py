"""
Quick Selection
===============

Brush-based automatic selection tool.

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
    # pylint: disable=import-outside-toplevel
    from numpy.typing import NDArray

from dreamstack.raster.selection.shapes.selection import Selection


def quick_selection(
    image: NDArray[np.uint8],
    brush_points: list[tuple[int, int]],
    *,
    brush_size: int = 20,
    edge_aware: bool = True,
    auto_enhance: bool = True,
) -> Selection:
    """Create a selection by painting with a smart brush.

    The brush automatically expands to include similar colors
    and respects edges.

    Args:
        image: Input image (BGR or BGRA).
        brush_points: Points where the brush was applied.
        brush_size: Size of the brush stroke.
        edge_aware: Respect edges when expanding selection.
        auto_enhance: Automatically refine edges.

    Returns:
        Selection expanded from brush strokes.

    Example:
        >>> # Paint over the subject
        >>> strokes = [(400, 300), (420, 320), (450, 350)]
        >>> sel = quick_selection(image, strokes, brush_size=30)
    """
    h, w = image.shape[:2]

    if not brush_points:
        return Selection(mask=np.zeros((h, w), dtype=np.uint8))

    # Work with BGR
    if image.ndim == 2:
        src = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        src = image[:, :, :3].copy()
    else:
        src = image.copy()

    # Create initial brush mask
    brush_mask = np.zeros((h, w), dtype=np.uint8)
    for point in brush_points:
        cv2.circle(brush_mask, point, brush_size // 2, 255, -1)

    # Sample colors from brush areas
    sampled_colors = src[brush_mask > 0]

    if len(sampled_colors) == 0:
        return Selection(mask=brush_mask)

    # Calculate mean and std of sampled colors
    mean_color = np.mean(sampled_colors, axis=0)
    std_color = np.std(sampled_colors, axis=0)

    # Color tolerance based on variance
    tolerance = np.maximum(std_color * 2, 15).astype(np.uint8)

    # Create color-based mask
    lower = np.maximum(0, mean_color - tolerance).astype(np.uint8)
    upper = np.minimum(255, mean_color + tolerance).astype(np.uint8)

    color_mask = cv2.inRange(src, lower, upper)

    if edge_aware:
        # Use GrabCut for edge-aware refinement
        # Create initial mask for GrabCut
        gc_mask = np.zeros((h, w), dtype=np.uint8)
        gc_mask[brush_mask > 0] = cv2.GC_FGD  # Definite foreground
        gc_mask[(brush_mask == 0) & (color_mask > 0)] = (
            cv2.GC_PR_FGD
        )  # Probable foreground
        gc_mask[(brush_mask == 0) & (color_mask == 0)] = (
            cv2.GC_PR_BGD
        )  # Probable background

        # Find a bounding rect for GrabCut
        points = np.column_stack(np.where(brush_mask > 0))
        if len(points) > 0:
            y_min, x_min = points.min(axis=0)
            y_max, x_max = points.max(axis=0)

            # Expand rect
            margin = brush_size * 3
            rect = (
                max(0, x_min - margin),
                max(0, y_min - margin),
                min(w, x_max + margin) - max(0, x_min - margin),
                min(h, y_max + margin) - max(0, y_min - margin),
            )

            if rect[2] > 0 and rect[3] > 0:
                bgd_model = np.zeros((1, 65), dtype=np.float64)
                fgd_model = np.zeros((1, 65), dtype=np.float64)

                try:
                    cv2.grabCut(
                        src,
                        gc_mask,
                        rect,
                        bgd_model,
                        fgd_model,
                        5,
                        cv2.GC_INIT_WITH_MASK,
                    )
                    result_mask = np.where(
                        (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD),
                        255,
                        0,
                    ).astype(np.uint8)
                except Exception:  # cv2.error can occur with invalid input
                    # Fall back to simple color mask
                    result_mask = color_mask
            else:
                result_mask = color_mask
        else:
            result_mask = color_mask
    else:
        # Connect brush strokes with color similarity
        result_mask = cv2.bitwise_or(brush_mask, color_mask)

    # Clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    result_mask = cv2.morphologyEx(result_mask, cv2.MORPH_CLOSE, kernel)
    result_mask = cv2.morphologyEx(result_mask, cv2.MORPH_OPEN, kernel)

    if auto_enhance:
        # Smooth edges
        result_mask = cv2.GaussianBlur(result_mask, (5, 5), 1)

    return Selection(mask=result_mask)
